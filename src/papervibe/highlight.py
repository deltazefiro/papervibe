"""Highlighting pipeline for LaTeX content."""

import re
import sys
from typing import List, Tuple, Optional, Callable
from papervibe.latex import strip_pvhighlight_wrappers
from papervibe.llm import LLMClient, is_retryable_error, print_error


class HighlightPipelineError(Exception):
    """Exception raised for highlight pipeline errors."""
    pass


def chunk_content(content: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Split content into chunks for parallel processing.

    Chunks are split at blank lines and section boundaries to maintain context.
    If a paragraph is too large, it's further split into smaller pieces.

    Args:
        content: LaTeX content to chunk
        max_chunk_size: Approximate maximum characters per chunk

    Returns:
        List of content chunks
    """
    # Hard safety limit - no chunk should ever exceed this (3x max_chunk_size)
    HARD_LIMIT = max_chunk_size * 3

    # Split by blank lines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', content)

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If single paragraph exceeds max size, split it further
        if para_size > max_chunk_size:
            # Add current chunk if not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split large paragraph by sentences or at fixed intervals
            # Try splitting by sentences first (look for '. ' or '.\n')
            sentences = re.split(r'(\.\s+|\.\n)', para)

            # Rejoin sentence content with delimiters
            rejoined_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    rejoined_sentences.append(sentences[i] + sentences[i + 1])
                else:
                    rejoined_sentences.append(sentences[i])
            if len(sentences) % 2 == 1:
                rejoined_sentences.append(sentences[-1])

            # Now chunk the sentences
            sub_chunk = []
            sub_size = 0
            for sent in rejoined_sentences:
                sent_size = len(sent)

                # If a single sentence is still too large, split at fixed intervals
                # Use HARD_LIMIT to ensure we never exceed safe size
                if sent_size > HARD_LIMIT:
                    if sub_chunk:
                        chunks.append(''.join(sub_chunk))
                        sub_chunk = []
                        sub_size = 0

                    # Split at HARD_LIMIT intervals as last resort
                    for i in range(0, len(sent), HARD_LIMIT):
                        chunks.append(sent[i:i + HARD_LIMIT])
                    continue

                if sub_size + sent_size > max_chunk_size and sub_chunk:
                    chunks.append(''.join(sub_chunk))
                    sub_chunk = [sent]
                    sub_size = sent_size
                else:
                    sub_chunk.append(sent)
                    sub_size += sent_size

            if sub_chunk:
                chunks.append(''.join(sub_chunk))
            continue

        # If adding this paragraph would exceed max size, start new chunk
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \n\n

    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Safety check: ensure no chunk exceeds HARD_LIMIT
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > HARD_LIMIT:
            # Force split at HARD_LIMIT
            for i in range(0, len(chunk), HARD_LIMIT):
                final_chunks.append(chunk[i:i + HARD_LIMIT])
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c.strip()]


def validate_highlighted_chunk(original: str, highlighted: str) -> bool:
    """
    Validate that a highlighted chunk matches the original after stripping wrappers.

    Args:
        original: Original chunk text
        highlighted: Highlighted chunk with \\pvhighlight{} wrappers

    Returns:
        True if validation passes, False otherwise
    """
    stripped = strip_pvhighlight_wrappers(highlighted)

    # Normalize line endings (CRLF -> LF) and trailing whitespace
    # Trailing whitespace doesn't affect LaTeX output and LLMs often strip it
    original_normalized = original.replace('\r\n', '\n').rstrip()
    stripped_normalized = stripped.replace('\r\n', '\n').rstrip()

    return original_normalized == stripped_normalized


async def highlight_content(
    content: str,
    llm_client: LLMClient,
    highlight_ratio: float = 0.4,
    max_retries: int = 2,
) -> str:
    """
    Highlight important keywords and sentences in LaTeX content.

    Args:
        content: LaTeX content to process (pre-references only)
        llm_client: LLM client for processing
        highlight_ratio: Target ratio of content to highlight
        max_retries: Number of retries per chunk if validation fails

    Returns:
        Content with \\pvhighlight{} wrappers applied

    Raises:
        HighlightPipelineError: If processing fails
    """
    # Split into chunks
    chunks = chunk_content(content)

    if not chunks:
        return content

    # Process chunks
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        success = False
        highlighted_chunk = chunk

        for attempt in range(max_retries + 1):
            try:
                # Try to highlight the chunk
                if attempt == 0:
                    highlighted_chunk = await llm_client.highlight_chunk(chunk, highlight_ratio)
                else:
                    # On retry, add a note to be more careful
                    highlighted_chunk = await llm_client.highlight_chunk(
                        chunk,
                        highlight_ratio,
                    )

                # Validate
                if validate_highlighted_chunk(chunk, highlighted_chunk):
                    success = True
                    break
                else:
                    # Validation failed
                    if attempt < max_retries:
                        continue
                    else:
                        # Final retry failed, use original
                        highlighted_chunk = chunk
                        success = True
                        break

            except Exception as e:
                # Check if retryable
                if is_retryable_error(e) and attempt < max_retries:
                    print(f"   Retrying chunk {i+1}/{len(chunks)} (attempt {attempt + 2}/{max_retries + 1})...", file=sys.stderr)
                    continue

                # Non-retryable error or max retries exceeded
                print_error(e, context=f"Chunk {i+1}/{len(chunks)} failed after {attempt + 1} attempt(s)")
                highlighted_chunk = chunk
                success = True
                break

        processed_chunks.append(highlighted_chunk)

    # Recombine chunks
    return '\n\n'.join(processed_chunks)


def count_chunks(content: str, max_chunk_chars: int = 1500) -> int:
    """
    Count the number of chunks that will be created from content.

    Args:
        content: LaTeX content to count chunks for
        max_chunk_chars: Maximum characters per chunk

    Returns:
        Number of chunks that will be created
    """
    chunks = chunk_content(content, max_chunk_size=max_chunk_chars)
    return len(chunks)


async def highlight_content_parallel(
    content: str,
    llm_client: LLMClient,
    highlight_ratio: float = 0.4,
    max_retries: int = 2,
    max_chunk_chars: int = 1500,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> str:
    """
    Highlight content with parallel chunk processing.

    This version processes all chunks in parallel (up to concurrency limit)
    and validates each one individually.

    Args:
        content: LaTeX content to process
        llm_client: LLM client for processing
        highlight_ratio: Target ratio of content to highlight
        max_retries: Number of retries per chunk if validation fails
        max_chunk_chars: Maximum characters per chunk
        progress_callback: Optional callback to call after each chunk is processed

    Returns:
        Content with \\pvhighlight{} wrappers applied
    """
    # Split into chunks
    chunks = chunk_content(content, max_chunk_size=max_chunk_chars)

    if not chunks:
        return content

    # Process all chunks in parallel (first attempt)
    try:
        highlighted_chunks = await llm_client.highlight_chunks_parallel(chunks, highlight_ratio)
    except Exception as e:
        # If parallel processing fails entirely, print error and fall back
        print_error(e, context="Parallel highlight processing failed entirely")
        return content

    # Validate and retry if needed
    final_chunks = []
    for i, (original, highlighted) in enumerate(zip(chunks, highlighted_chunks)):
        try:
            if validate_highlighted_chunk(original, highlighted):
                final_chunks.append(highlighted)
            else:
                # Retry once
                try:
                    retry_result = await llm_client.highlight_chunk(original, highlight_ratio)
                    if validate_highlighted_chunk(original, retry_result):
                        final_chunks.append(retry_result)
                    else:
                        # Use original if validation still fails
                        print(f"   Warning: Chunk {i+1}/{len(chunks)} validation failed after retry, using original", file=sys.stderr)
                        final_chunks.append(original)
                except Exception as e:
                    # Print full error on retry failure
                    print_error(e, context=f"Chunk {i+1}/{len(chunks)} retry failed, using original")
                    final_chunks.append(original)
        except Exception as e:
            # Validation or other error, print full error
            print_error(e, context=f"Chunk {i+1}/{len(chunks)} processing failed, using original")
            final_chunks.append(original)

        # Call progress callback after each chunk is processed
        if progress_callback:
            progress_callback(1)

    return '\n\n'.join(final_chunks)
