"""Chunking and validation utilities for LaTeX highlighting."""

import logging
import re
from typing import List, Tuple
from papervibe.latex import strip_pvhighlight_wrappers

logger = logging.getLogger(__name__)


# Type alias for chunks with separators: (chunk_text, separator_after)
ChunkWithSep = Tuple[str, str]


def parse_snippets(llm_output: str) -> List[str]:
    """
    Parse LLM output into a list of snippets to highlight.

    Args:
        llm_output: Raw LLM output with one snippet per line

    Returns:
        List of non-empty snippet strings
    """
    snippets = []
    for line in llm_output.strip().split('\n'):
        line = line.strip()
        # Skip empty lines and lines that look like commentary
        if line and not line.startswith('#') and not line.startswith('//'):
            snippets.append(line)
    return snippets


def apply_highlights(original: str, snippets: List[str]) -> Tuple[str, int, int]:
    """
    Apply highlights to original text by wrapping snippets with \\pvhighlight{}.

    This function searches for each snippet in the original text and wraps
    the first match with \\pvhighlight{}. Longer snippets are processed first
    to avoid issues with overlapping matches.

    Args:
        original: Original LaTeX content
        snippets: List of text snippets to highlight

    Returns:
        Tuple of (highlighted_text, matched_count, unmatched_count)
    """
    result = original
    matched = 0
    unmatched = 0

    # Sort snippets by length (longest first) to avoid nested highlights
    sorted_snippets = sorted(snippets, key=len, reverse=True)

    for snippet in sorted_snippets:
        if not snippet:
            continue

        # Search for exact match in the current result
        idx = result.find(snippet)
        if idx != -1:
            # Check if already wrapped (avoid double-wrapping)
            # Look for \pvhighlight{ before this position
            before = result[:idx]
            if before.endswith('\\pvhighlight{'):
                logger.debug("Snippet already wrapped, skipping: %s", snippet[:50])
                continue

            # Wrap the first occurrence
            wrapped = f'\\pvhighlight{{{snippet}}}'
            result = result[:idx] + wrapped + result[idx + len(snippet):]
            matched += 1
            logger.debug("Matched snippet: %s", snippet[:50])
        else:
            unmatched += 1
            logger.debug("Unmatched snippet: %s", snippet[:80])

    return result, matched, unmatched


class HighlightPipelineError(Exception):
    """Exception raised for highlight pipeline errors."""
    pass


def chunk_content_with_seps(content: str, max_chunk_size: int = 1500) -> List[ChunkWithSep]:
    """
    Split content into chunks for parallel processing, preserving separators.

    Chunks are split at blank lines and section boundaries to maintain context.
    If a paragraph is too large, it's further split into smaller pieces.

    This function returns chunks along with the separator that should follow
    each chunk, allowing exact reconstruction of the original content.

    Args:
        content: LaTeX content to chunk
        max_chunk_size: Approximate maximum characters per chunk

    Returns:
        List of (chunk, separator_after) tuples. Rejoin with:
        ''.join(chunk + sep for chunk, sep in result)
    """
    # Hard safety limit - no chunk should ever exceed this (3x max_chunk_size)
    HARD_LIMIT = max_chunk_size * 3

    # Split by blank lines, capturing the separators
    # This gives us [para1, sep1, para2, sep2, para3] (odd length)
    parts = re.split(r'(\n[ \t]*\n)', content)

    # Pair paragraphs with their trailing separators
    paragraphs_with_seps: List[ChunkWithSep] = []
    for i in range(0, len(parts), 2):
        para = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ''
        paragraphs_with_seps.append((para, sep))

    chunks_with_seps: List[ChunkWithSep] = []
    current_paras: List[ChunkWithSep] = []  # List of (para, sep)
    current_size = 0

    def flush_current() -> None:
        """Flush current_paras into a single chunk."""
        nonlocal current_paras, current_size
        if not current_paras:
            return
        # Combine all paras with their separators, except last separator
        combined = ''.join(p + s for p, s in current_paras[:-1])
        combined += current_paras[-1][0]  # Last para without its sep
        last_sep = current_paras[-1][1]   # Last para's sep becomes chunk's sep
        if combined.strip():
            chunks_with_seps.append((combined, last_sep))
        current_paras = []
        current_size = 0

    def split_large_para(para: str, trailing_sep: str) -> List[ChunkWithSep]:
        """Split a large paragraph into smaller chunks."""
        result: List[ChunkWithSep] = []

        # Try splitting by sentences first (look for '. ' or '.\n')
        sentence_parts = re.split(r'(\.\s+|\.\n)', para)

        # Rejoin sentence content with delimiters
        rejoined_sentences = []
        for i in range(0, len(sentence_parts) - 1, 2):
            if i + 1 < len(sentence_parts):
                rejoined_sentences.append(sentence_parts[i] + sentence_parts[i + 1])
            else:
                rejoined_sentences.append(sentence_parts[i])
        if len(sentence_parts) % 2 == 1:
            rejoined_sentences.append(sentence_parts[-1])

        # Now chunk the sentences
        sub_chunk_parts: List[str] = []
        sub_size = 0

        for idx, sent in enumerate(rejoined_sentences):
            sent_size = len(sent)
            is_last = (idx == len(rejoined_sentences) - 1)

            # If a single sentence is still too large, split at fixed intervals
            if sent_size > HARD_LIMIT:
                if sub_chunk_parts:
                    result.append((''.join(sub_chunk_parts), ''))
                    sub_chunk_parts = []
                    sub_size = 0

                # Split at HARD_LIMIT intervals - use empty sep for all but last
                for j in range(0, len(sent), HARD_LIMIT):
                    chunk_text = sent[j:j + HARD_LIMIT]
                    is_last_piece = (j + HARD_LIMIT >= len(sent))
                    chunk_sep = trailing_sep if (is_last and is_last_piece) else ''
                    if chunk_text.strip():
                        result.append((chunk_text, chunk_sep))
                continue

            if sub_size + sent_size > max_chunk_size and sub_chunk_parts:
                result.append((''.join(sub_chunk_parts), ''))
                sub_chunk_parts = [sent]
                sub_size = sent_size
            else:
                sub_chunk_parts.append(sent)
                sub_size += sent_size

        if sub_chunk_parts:
            combined = ''.join(sub_chunk_parts)
            if combined.strip():
                result.append((combined, trailing_sep))

        return result

    for para, sep in paragraphs_with_seps:
        para_size = len(para)

        # If single paragraph exceeds max size, split it further
        if para_size > max_chunk_size:
            # Flush current chunks first
            flush_current()

            # Split the large paragraph
            split_chunks = split_large_para(para, sep)
            chunks_with_seps.extend(split_chunks)
            continue

        # If adding this paragraph would exceed max size, flush and start new
        if current_size + para_size > max_chunk_size and current_paras:
            flush_current()

        current_paras.append((para, sep))
        current_size += para_size + len(sep)

    # Flush remaining
    flush_current()

    # Safety check: ensure no chunk exceeds HARD_LIMIT
    final_chunks: List[ChunkWithSep] = []
    for chunk, sep in chunks_with_seps:
        if len(chunk) > HARD_LIMIT:
            # Force split at HARD_LIMIT - use empty sep for all but last
            for i in range(0, len(chunk), HARD_LIMIT):
                chunk_text = chunk[i:i + HARD_LIMIT]
                is_last = (i + HARD_LIMIT >= len(chunk))
                chunk_sep = sep if is_last else ''
                if chunk_text.strip():
                    final_chunks.append((chunk_text, chunk_sep))
        elif chunk.strip():
            final_chunks.append((chunk, sep))

    return final_chunks


def rejoin_chunks(chunks_with_seps: List[ChunkWithSep]) -> str:
    """
    Rejoin chunks with their original separators.

    Args:
        chunks_with_seps: List of (chunk, separator_after) tuples

    Returns:
        Rejoined content preserving original whitespace
    """
    return ''.join(chunk + sep for chunk, sep in chunks_with_seps)


def chunk_content(content: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Split content into chunks for parallel processing.

    This is a convenience wrapper around chunk_content_with_seps that returns
    just the chunk texts. Use chunk_content_with_seps when you need to preserve
    the original separators for rejoining.

    Args:
        content: LaTeX content to chunk
        max_chunk_size: Approximate maximum characters per chunk

    Returns:
        List of content chunks (without separator information)
    """
    chunks_with_seps = chunk_content_with_seps(content, max_chunk_size)
    return [chunk for chunk, _ in chunks_with_seps]


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
