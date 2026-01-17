"""Tests for gray-out pipeline."""

import pytest
from papervibe.gray import (
    chunk_content,
    validate_grayed_chunk,
    gray_out_content_parallel,
)
from papervibe.llm import LLMClient


def test_chunk_content():
    """Test chunking content by blank lines."""
    content = """First paragraph here.

Second paragraph here.

Third paragraph here.

Fourth paragraph here."""
    
    chunks = chunk_content(content, max_chunk_size=50)
    
    assert len(chunks) > 0
    # Should split at blank lines
    assert all('paragraph' in chunk.lower() for chunk in chunks)


def test_chunk_content_large_paragraph():
    """Test chunking with a paragraph larger than max size."""
    large_para = "A" * 5000
    small_para = "B" * 100
    content = f"{large_para}\n\n{small_para}"
    
    chunks = chunk_content(content, max_chunk_size=1000)
    
    # Large paragraph should be its own chunk
    assert any(len(chunk) > 1000 for chunk in chunks)
    assert len(chunks) >= 2


def test_chunk_content_empty():
    """Test chunking empty content."""
    chunks = chunk_content("")
    assert chunks == []
    
    chunks = chunk_content("   \n\n  ")
    assert chunks == []


def test_validate_grayed_chunk_valid():
    """Test validation of correctly grayed chunk."""
    original = "This is a sentence. This is another sentence."
    grayed = "This is a sentence. \\pvgray{This is another sentence.}"
    
    assert validate_grayed_chunk(original, grayed) is True


def test_validate_grayed_chunk_invalid():
    """Test validation of incorrectly grayed chunk."""
    original = "This is a sentence. This is another sentence."
    grayed = "This is a sentence. \\pvgray{This is a different sentence.}"
    
    assert validate_grayed_chunk(original, grayed) is False


def test_validate_grayed_chunk_no_changes():
    """Test validation when no graying is applied."""
    original = "This is a sentence. This is another sentence."
    grayed = original
    
    assert validate_grayed_chunk(original, grayed) is True


def test_validate_grayed_chunk_whitespace_normalization():
    """Test that validation handles whitespace differences."""
    original = "This is a sentence.\n\nThis is another sentence."
    grayed = "This is a sentence. \\pvgray{This is another sentence.}"
    
    # Should be valid due to whitespace normalization
    assert validate_grayed_chunk(original, grayed) is True


def test_validate_grayed_chunk_nested_braces():
    """Test validation with nested braces in content."""
    original = "Text with {nested} braces here."
    grayed = "\\pvgray{Text with {nested} braces here.}"
    
    assert validate_grayed_chunk(original, grayed) is True


@pytest.mark.asyncio
async def test_gray_out_content_parallel_dry_run():
    """Test parallel gray-out in dry-run mode."""
    content = """First paragraph here.

Second paragraph here.

Third paragraph here."""
    
    client = LLMClient(dry_run=True, concurrency=2)
    result = await gray_out_content_parallel(content, client, gray_ratio=0.4)
    
    # In dry-run mode, should return original content
    assert result == content


@pytest.mark.asyncio
async def test_gray_out_content_parallel_empty():
    """Test parallel gray-out with empty content."""
    client = LLMClient(dry_run=True, concurrency=2)
    result = await gray_out_content_parallel("", client, gray_ratio=0.4)
    
    assert result == ""
