"""Low-level LLM completion interface."""

import asyncio
import json
import logging
from typing import Optional, List, Callable, TypeVar, Any, Type
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _log_llm_debug(
    title: str,
    metadata: dict[str, str],
    input_text: str,
    output_text: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    lines = [title]
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("input:")
    lines.append(input_text)
    lines.append("output:")
    lines.append(output_text)
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
    if "rate_limit" in error_str or "rate limit" in error_str or "429" in error_str:
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
    if hasattr(error, "status_code"):
        status_code = f" (status: {error.status_code})"

    if context:
        logger.error("Error: %s", context)
    logger.error("%s%s: %s", error_type, status_code, error_msg)


async def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    context: str = "",
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

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str
    openai_base_url: Optional[str] = None
    strong_model: str = "gemini-3-pro-preview"
    light_model: str = "gemini-3-flash-preview"
    request_timeout_seconds: float = 30.0


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

    async def complete(
        self,
        model_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Simple completion interface.

        Args:
            model_type: "strong" or "light"
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate

        Returns:
            Completion text

        Raises:
            asyncio.TimeoutError: If request times out
            Exception: On other errors
        """
        if self.dry_run:
            _log_llm_debug(
                f"LLM complete ({model_type}, dry run)",
                {
                    "model": self._get_model(model_type),
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                },
                user_prompt,
                user_prompt,
            )
            return user_prompt

        model = self._get_model(model_type)

        async with self.semaphore:

            async def _make_request():
                """Inner function for retry logic."""
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_completion_tokens"] = max_tokens

                return await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=self.settings.request_timeout_seconds,
                )

            response = await retry_with_backoff(
                _make_request, max_retries=3, context=f"{model_type} completion"
            )

            result = response.choices[0].message.content
            output_text = result if result is not None else ""

            _log_llm_debug(
                f"LLM complete ({model_type})",
                {
                    "model": model,
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                },
                user_prompt,
                output_text,
            )

            return output_text

    async def complete_structured(
        self,
        model_type: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.7,
    ) -> T:
        """
        Structured completion interface with Pydantic model parsing.

        Args:
            model_type: "strong" or "light"
            system_prompt: System prompt
            user_prompt: User prompt
            response_model: Pydantic model for response parsing
            temperature: Temperature for sampling

        Returns:
            Parsed response as response_model instance

        Raises:
            asyncio.TimeoutError: If request times out
            Exception: On other errors
        """
        if self.dry_run:
            # In dry run, return a dummy instance with user_prompt as the first string field
            dummy_data = {}
            for field_name, field_info in response_model.model_fields.items():
                if field_info.annotation == str:
                    dummy_data[field_name] = user_prompt
                    break
            result = response_model(**dummy_data)
            _log_llm_debug(
                f"LLM complete_structured ({model_type}, dry run)",
                {
                    "model": self._get_model(model_type),
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                },
                user_prompt,
                str(result),
            )
            return result

        model = self._get_model(model_type)

        async with self.semaphore:

            async def _make_request():
                """Inner function for retry logic."""
                return await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=response_model,
                        temperature=temperature,
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )

            response = await retry_with_backoff(
                _make_request,
                max_retries=3,
                context=f"{model_type} structured completion",
            )

            result = response.choices[0].message.parsed

            _log_llm_debug(
                f"LLM complete_structured ({model_type})",
                {
                    "model": model,
                    "timeout": f"{self.settings.request_timeout_seconds}s",
                },
                user_prompt,
                str(result),
            )

            return result

    def _get_model(self, model_type: str) -> str:
        """
        Get model name from model type.

        Args:
            model_type: "strong" or "light"

        Returns:
            Model name
        """
        if model_type == "strong":
            return self.settings.strong_model
        elif model_type == "light":
            return self.settings.light_model
        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Must be 'strong' or 'light'."
            )

    # Legacy methods for backward compatibility with tests
    async def rewrite_abstract(self, original_abstract: str) -> str:
        """
        Legacy method for backward compatibility.
        Delegates to process.rewrite_abstract.
        """
        from .process import rewrite_abstract

        return await rewrite_abstract(self, original_abstract)

    async def highlight_chunk(
        self,
        chunk: str,
        highlight_ratio: float = 0.4,
    ) -> str:
        """
        Legacy method for backward compatibility.
        Delegates to process.highlight_chunk.
        """
        from .process import highlight_chunk

        return await highlight_chunk(self, chunk, highlight_ratio)

    async def highlight_chunks_parallel(
        self,
        chunks: List[str],
        highlight_ratio: float = 0.4,
    ) -> List[str]:
        """
        Legacy method for backward compatibility.
        Delegates to process.highlight_chunks_parallel.
        """
        from .process import highlight_chunks_parallel

        return await highlight_chunks_parallel(self, chunks, highlight_ratio)
