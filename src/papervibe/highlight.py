"""Chunking and validation utilities for LaTeX highlighting."""

import logging
import re
from typing import List
from papervibe.latex import strip_pvhighlight_wrappers

logger = logging.getLogger(__name__)


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
