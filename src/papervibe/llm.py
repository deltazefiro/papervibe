"""LLM interface for abstract rewriting and sentence graying."""

import asyncio
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI


class LLMSettings(BaseSettings):
    """Settings for LLM API access."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    openai_api_key: str
    openai_base_url: Optional[str] = None
    strong_model: str = "gpt-4o"
    light_model: str = "gpt-4o-mini"
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
            system_prompt = """You are an expert academic editor. Your task is to rewrite paper abstracts to be clearer, more engaging, and better structured while preserving all technical content and claims.

Focus on:
- Improving clarity and readability
- Making the narrative flow more logical
- Highlighting key contributions
- Maintaining technical accuracy
- Keeping similar length to the original

Do NOT:
- Add claims not in the original
- Remove important technical details
- Change the fundamental message"""

            user_prompt = f"""Rewrite the following abstract to be clearer and more engaging:

{original_abstract}

Provide the rewritten abstract."""

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
                print(f"   Warning: Abstract rewrite timed out after {self.settings.request_timeout_seconds}s, using original")
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
            system_prompt = """You are an expert at identifying the most important information in academic papers. Your task is to mark less important sentences by wrapping them with \\pvgray{...}.

Rules:
1. Wrap approximately the specified ratio of sentences with \\pvgray{...}
2. Prioritize graying out:
   - Transitional phrases
   - Obvious or well-known statements
   - Less critical details or examples
   - Redundant information
3. Keep important:
   - Key claims and contributions
   - Novel results and findings
   - Technical definitions
   - Critical methodology
4. NEVER gray out section headings, captions, or labels
5. Return ONLY the modified chunk with wrappers added
6. Do NOT change any text content except adding \\pvgray{} wrappers
7. Preserve all LaTeX formatting, commands, and structure EXACTLY
8. Preserve all whitespace, newlines, and indentation EXACTLY as in the original
9. Do NOT add vspace, hspace, or any other spacing/formatting commands
10. Do NOT use "..." as a placeholder - always include the complete original text
11. The ONLY change should be wrapping sentences with \\pvgray{...} - nothing else"""

            user_prompt = f"""Gray out approximately {gray_ratio*100:.0f}% of the less important sentences in this chunk by wrapping them with \\pvgray{{...}}:

{chunk}

CRITICAL: Return the text EXACTLY as provided, with ONLY \\pvgray{{...}} wrappers added. Preserve all whitespace, newlines, and formatting. Do not change anything else."""

            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.settings.light_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_completion_tokens=16000,  # Use almost the full 16K limit for gpt-4o-mini
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )
                
                result = response.choices[0].message.content
                if result is None:
                    return chunk
                return result
            except asyncio.TimeoutError:
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
            except Exception as e:
                # Return original chunk on error (will be caught by validation later)
                return chunk
        
        tasks = [process_chunk_safe(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
