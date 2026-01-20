"""LLM interface for abstract rewriting and content highlighting."""

import asyncio
import json
import logging
from typing import Optional, List, Callable, TypeVar, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .prompts import get_renderer

logger = logging.getLogger(__name__)

T = TypeVar('T')

_DEBUG_DELIM = "-" * 60


def _log_llm_debug(
    title: str,
    metadata: dict[str, str],
    input_text: str,
    output_text: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    lines = [_DEBUG_DELIM, title]
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("input:")
    lines.append(input_text)
    lines.append("output:")
    lines.append(output_text)
    lines.append(_DEBUG_DELIM)
    logger.debug("\n".join(lines))


def is_retryable_error(error: Exception) -> bool:
    """
    Check if error should be retried (rate limit or network only).

    Args:
        error: Exception to check

    Returns:
        True if error is retryable (rate limit or network), False otherwise
    """
    # Rate limit errors (429)
    if isinstance(error, RateLimitError):
        return True

    # Network connection errors (connection refused, DNS, etc.)
    if isinstance(error, APIConnectionError):
        return True

    # API timeout errors (different from asyncio.TimeoutError)
    if isinstance(error, APITimeoutError):
        return True

    # Check error message for rate limit indicators
    error_str = str(error).lower()
    if 'rate_limit' in error_str or 'rate limit' in error_str or '429' in error_str:
        return True

    return False


def print_error(error: Exception, context: str = "") -> None:
    """
    Print full error message to stderr with context.

    Args:
        error: Exception to print
        context: Optional context about what operation failed
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Extract status code if available
    status_code = ""
    if hasattr(error, 'status_code'):
        status_code = f" (status: {error.status_code})"

    if context:
        logger.error("Error: %s", context)
    logger.error("%s%s: %s", error_type, status_code, error_msg)


async def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    context: str = ""
) -> Any:
    """
    Retry function with exponential backoff for retryable errors only.

    Args:
        func: Async function to retry
        max_retries: Maximum number of attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 2.0)
        context: Context string for error messages

    Returns:
        Result of successful function call

    Raises:
        Exception: Re-raises the exception if non-retryable or max retries exceeded
    """
    def _should_retry(error: Exception) -> bool:
        if isinstance(error, asyncio.TimeoutError):
            return False
        return is_retryable_error(error)

    def _log_retry(retry_state: Any) -> None:
        context_label = f" during {context}" if context else ""
        sleep_seconds = 0.0
        if retry_state.next_action is not None:
            sleep_seconds = retry_state.next_action.sleep
        logger.warning(
            "Retryable error%s (attempt %s/%s), retrying in %ss...",
            context_label,
            retry_state.attempt_number,
            max_retries,
            sleep_seconds,
        )

    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=initial_delay),
        reraise=True,
        before_sleep=_log_retry,
    ):
        with attempt:
            return await func()


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


class HighlightedChunk(BaseModel):
    """Structured output for highlighted chunk."""

    content: str = Field(description="The chunk with \\pvhighlight{} wrappers applied to important keywords and sentences")


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
            "highlight_timeouts": 0,
            "highlight_errors": 0,
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

        Raises:
            Exception: On non-retryable errors (auth, invalid request, etc.)
        """
        if self.dry_run:
            _log_llm_debug(
                "LLM rewrite_abstract (dry run)",
                {
                    "model": self.settings.strong_model,
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                },
                original_abstract,
                original_abstract,
            )
            return original_abstract

        async with self.semaphore:
            system_prompt = self.prompt_renderer.render_rewrite_abstract_system()
            user_prompt = self.prompt_renderer.render_rewrite_abstract_user(original_abstract)

            async def _make_request():
                """Inner function for retry logic."""
                return await asyncio.wait_for(
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

            try:
                response = await retry_with_backoff(
                    _make_request,
                    max_retries=3,
                    context="abstract rewrite"
                )
                result = response.choices[0].message.parsed
                _log_llm_debug(
                    "LLM rewrite_abstract",
                    {
                        "model": self.settings.strong_model,
                        "timeout": f"{self.settings.request_timeout_seconds}s",
                    },
                    original_abstract,
                    result.abstract,
                )
                return result.abstract
            except asyncio.TimeoutError:
                # Timeout is graceful - use original abstract
                self.stats["abstract_timeouts"] += 1
                logger.warning(
                    "Abstract rewrite timed out after %ss, using original",
                    self.settings.request_timeout_seconds,
                )
                _log_llm_debug(
                    "LLM rewrite_abstract (timeout fallback)",
                    {
                        "model": self.settings.strong_model,
                        "timeout": f"{self.settings.request_timeout_seconds}s",
                    },
                    original_abstract,
                    original_abstract,
                )
                return original_abstract
            except Exception as e:
                # Print full error and fail fast
                self.stats["abstract_errors"] += 1
                print_error(e, context="Failed to rewrite abstract")
                # Re-raise to fail fast
                raise
    
    async def highlight_chunk(
        self,
        chunk: str,
        highlight_ratio: float = 0.4,
    ) -> str:
        """
        Highlight important keywords and sentences in a text chunk.

        Args:
            chunk: Text chunk to process
            highlight_ratio: Target ratio of content to highlight (0.0 to 1.0)

        Returns:
            Chunk with \\pvhighlight{} wrappers around important content

        Raises:
            Exception: If LLM call fails (including token limit errors, auth errors, etc.)
        """
        if self.dry_run:
            _log_llm_debug(
                "LLM highlight_chunk (dry run)",
                {
                    "model": self.settings.light_model,
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                    "highlight_ratio": str(highlight_ratio),
                },
                chunk,
                chunk,
            )
            return chunk

        async with self.semaphore:
            system_prompt = self.prompt_renderer.render_highlight_system()
            user_prompt = self.prompt_renderer.render_highlight_user(chunk, highlight_ratio)

            async def _make_request():
                """Inner function for retry logic."""
                return await asyncio.wait_for(
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

            try:
                response = await retry_with_backoff(
                    _make_request,
                    max_retries=3,
                    context="highlight chunk"
                )

                result = response.choices[0].message.content
                output_text = result if result is not None else chunk
                _log_llm_debug(
                    "LLM highlight_chunk",
                    {
                        "model": self.settings.light_model,
                        "timeout": f"{self.settings.request_timeout_seconds}s",
                        "highlight_ratio": str(highlight_ratio),
                    },
                    chunk,
                    output_text,
                )
                return output_text
            except asyncio.TimeoutError:
                # Timeout is graceful - return original chunk
                self.stats["highlight_timeouts"] += 1
                _log_llm_debug(
                    "LLM highlight_chunk (timeout fallback)",
                    {
                        "model": self.settings.light_model,
                        "timeout": f"{self.settings.request_timeout_seconds}s",
                        "highlight_ratio": str(highlight_ratio),
                    },
                    chunk,
                    chunk,
                )
                return chunk
            except Exception as e:
                # Check for token limit errors (these are non-retryable)
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['token', 'length', 'context_length', 'too long']):
                    raise Exception(f"Token limit exceeded: chunk is too large ({len(chunk)} chars)")
                # Print full error and re-raise
                self.stats["highlight_errors"] += 1
                print_error(e, context=f"Failed to highlight chunk ({len(chunk)} chars)")
                raise
    
    async def highlight_chunks_parallel(
        self,
        chunks: List[str],
        highlight_ratio: float = 0.4,
    ) -> List[str]:
        """
        Highlight multiple chunks in parallel with concurrency control.

        Args:
            chunks: List of text chunks to process
            highlight_ratio: Target ratio of content to highlight

        Returns:
            List of processed chunks with highlighting applied (or original on error)
        """
        async def process_chunk_safe(chunk: str, index: int) -> str:
            """Process a single chunk with error handling."""
            try:
                return await self.highlight_chunk(chunk, highlight_ratio)
            except Exception as e:
                # Print full error details
                self.stats["highlight_errors"] += 1
                print_error(e, context=f"Chunk {index + 1}/{len(chunks)} failed")
                # Return original chunk on error (will be caught by validation later)
                return chunk

        tasks = [process_chunk_safe(chunk, i) for i, chunk in enumerate(chunks)]
        return await asyncio.gather(*tasks)
