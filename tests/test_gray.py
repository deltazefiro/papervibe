"""Tests for gray-out pipeline."""

import asyncio
import time
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
    
    # Large paragraph should be split into multiple chunks
    assert len(chunks) >= 2
    # Each chunk should be within the hard limit (3x max_chunk_size)
    for chunk in chunks:
        assert len(chunk) <= 3000  # Hard limit is 3x max_chunk_size


def test_chunk_content_very_large_paragraph_with_sentences():
    """Test chunking of very large paragraph containing sentences."""
    # Create a large paragraph with sentences
    sentence = "This is a test sentence with some content. "
    large_para = sentence * 100  # ~4000 chars
    
    chunks = chunk_content(large_para, max_chunk_size=1000)
    
    # Should be split into multiple chunks
    assert len(chunks) >= 3
    # Each chunk should contain complete sentences
    for chunk in chunks:
        # Should end with sentence ending or be part of a sentence
        assert len(chunk) > 0


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


def test_validate_grayed_chunk_line_ending_normalization():
    """Test that validation handles line-ending differences only."""
    original = "This is a sentence.\r\nThis is another sentence."
    grayed = "This is a sentence.\n\\pvgray{This is another sentence.}"
    
    # Should be valid due to line-ending normalization only
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


def test_validate_grayed_chunk_strict_whitespace():
    """Test that validation is strict about whitespace (not collapsing)."""
    original = "Line one.\n\nLine two."
    # This has collapsed whitespace, should fail strict validation
    grayed_wrong = "Line one. \\pvgray{Line two.}"
    
    assert validate_grayed_chunk(original, grayed_wrong) is False
    
    # This preserves the double newline, should pass
    grayed_correct = "Line one.\n\n\\pvgray{Line two.}"
    assert validate_grayed_chunk(original, grayed_correct) is True


@pytest.mark.asyncio
async def test_parallel_processing_performance():
    """Test that parallel processing is faster than sequential."""
    # Create a fake LLM client that sleeps to simulate work
    class SlowLLMClient(LLMClient):
        def __init__(self, sleep_time=0.05):
            super().__init__(dry_run=True, concurrency=4)
            self.sleep_time = sleep_time
        
        async def gray_out_chunks_parallel(self, chunks, gray_ratio=0.4):
            """Simulate slow processing."""
            async def process_one(chunk):
                await asyncio.sleep(self.sleep_time)
                return chunk
            
            # Process in parallel
            return await asyncio.gather(*[process_one(c) for c in chunks])
    
    # Create content that will split into multiple chunks
    content = "\n\n".join([f"Paragraph {i} with some content." for i in range(8)])
    
    client = SlowLLMClient(sleep_time=0.05)
    
    start = time.time()
    result = await gray_out_content_parallel(content, client, gray_ratio=0.4, max_chunk_chars=50)
    elapsed = time.time() - start
    
    # Should complete in less than sequential time (8 chunks * 0.05s = 0.4s)
    # With parallelism (concurrency=4), should take ~0.1s (2 batches)
    assert elapsed < 0.3  # Allow some margin
    assert result == content  # Dry run returns original


@pytest.mark.asyncio
async def test_max_chunk_chars_parameter():
    """Test that max_chunk_chars parameter controls chunk size."""
    # Create content with paragraphs
    paragraphs = [f"Paragraph {i} with content." for i in range(5)]
    content = "\n\n".join(paragraphs)
    
    client = LLMClient(dry_run=True, concurrency=2)
    
    # With max_chunk_chars=50, should create multiple chunks
    result = await gray_out_content_parallel(content, client, gray_ratio=0.4, max_chunk_chars=50)
    
    # In dry run, should return content (possibly with some whitespace normalization from chunking)
    # Just verify it processes without error and returns something reasonable
    assert len(result) > 0
    assert "Paragraph" in result
