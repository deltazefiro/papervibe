"""LLM interface for abstract rewriting and sentence graying."""

import asyncio
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI

from .prompts import get_renderer


class LLMSettings(BaseSettings):
    """Settings for LLM API access."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    openai_api_key: str
    openai_base_url: Optional[str] = None
    strong_model: str = "gemini-3-pro-preview"
    light_model: str = "gemini-3-flash-preview"
    request_timeout_seconds: float = 30.0


class RewrittenAbstract(BaseModel):
    """Structured output for abstract rewriting."""
    
    abstract: str = Field(description="The rewritten abstract text")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation of changes made")


class GrayedChunk(BaseModel):
    """Structured output for grayed chunk."""
    
    content: str = Field(description="The chunk with \\pvgray{} wrappers applied to less important sentences")


class LLMClient:
    """Async client for LLM operations with concurrency control."""
    
    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        concurrency: int = 8,
        dry_run: bool = False,
    ):
        """
        Initialize LLM client.
        
        Args:
            settings: LLM settings (loads from .env if None)
            concurrency: Maximum concurrent requests
            dry_run: If True, skip actual API calls and return passthroughs
        """
        self.settings = settings or LLMSettings()
        self.dry_run = dry_run
        self.semaphore = asyncio.Semaphore(concurrency)
        self.prompt_renderer = get_renderer()
        self.stats = {
            "abstract_timeouts": 0,
            "abstract_errors": 0,
            "gray_timeouts": 0,
            "gray_errors": 0,
        }
        
        if not dry_run:
            self.client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        else:
            self.client = None
    
    async def rewrite_abstract(self, original_abstract: str) -> str:
        """
        Rewrite an abstract to be more clear and engaging.
        
        Args:
            original_abstract: Original abstract text
            
        Returns:
            Rewritten abstract text
        """
        if self.dry_run:
            return original_abstract
        
        async with self.semaphore:
            system_prompt = self.prompt_renderer.render_rewrite_abstract_system()
            user_prompt = self.prompt_renderer.render_rewrite_abstract_user(original_abstract)

            try:
                response = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.settings.strong_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=RewrittenAbstract,
                        temperature=0.7,
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )
                
                result = response.choices[0].message.parsed
                return result.abstract
            except asyncio.TimeoutError:
                self.stats["abstract_timeouts"] += 1
                print(f"   Warning: Abstract rewrite timed out after {self.settings.request_timeout_seconds}s, using original")
                return original_abstract
            except Exception:
                self.stats["abstract_errors"] += 1
                return original_abstract
    
    async def gray_out_chunk(
        self,
        chunk: str,
        gray_ratio: float = 0.4,
    ) -> str:
        """
        Gray out less important sentences in a text chunk.
        
        Args:
            chunk: Text chunk to process
            gray_ratio: Target ratio of text to gray out (0.0 to 1.0)
            
        Returns:
            Chunk with \\pvgray{} wrappers around less important sentences
            
        Raises:
            Exception: If LLM call fails (including token limit errors)
        """
        if self.dry_run:
            return chunk
        
        async with self.semaphore:
            system_prompt = self.prompt_renderer.render_gray_out_system()
            user_prompt = self.prompt_renderer.render_gray_out_user(chunk, gray_ratio)

            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.settings.light_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_completion_tokens=16000,  # Use almost the full 16K limit
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )
                
                result = response.choices[0].message.content
                if result is None:
                    return chunk
                return result
            except asyncio.TimeoutError:
                self.stats["gray_timeouts"] += 1
                # On timeout, return original chunk
                return chunk
            except Exception as e:
                error_str = str(e).lower()
                # Check for token limit errors
                if any(keyword in error_str for keyword in ['token', 'length', 'context_length', 'too long']):
                    raise Exception(f"Token limit exceeded: chunk is too large ({len(chunk)} chars)")
                # Re-raise other errors
                raise Exception(f"LLM API error: {str(e)[:200]}")
    
    async def gray_out_chunks_parallel(
        self,
        chunks: List[str],
        gray_ratio: float = 0.4,
    ) -> List[str]:
        """
        Gray out multiple chunks in parallel with concurrency control.
        
        Args:
            chunks: List of text chunks to process
            gray_ratio: Target ratio of text to gray out
            
        Returns:
            List of processed chunks with graying applied (or original on error)
        """
        async def process_chunk_safe(chunk: str) -> str:
            """Process a single chunk with error handling."""
            try:
                return await self.gray_out_chunk(chunk, gray_ratio)
            except Exception:
                self.stats["gray_errors"] += 1
                # Return original chunk on error (will be caught by validation later)
                return chunk
        
        tasks = [process_chunk_safe(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
