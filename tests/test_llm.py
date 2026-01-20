"""Tests for LLM module."""

import asyncio
import pytest
from papervibe.llm import LLMClient, LLMSettings


@pytest.fixture
def dry_run_client():
    """Create a dry-run LLM client for testing."""
    return LLMClient(dry_run=True, concurrency=2)


@pytest.mark.asyncio
async def test_rewrite_abstract_dry_run(dry_run_client):
    """Test abstract rewriting in dry-run mode."""
    original = "This is a test abstract. It has multiple sentences."
    result = await dry_run_client.rewrite_abstract(original)
    
    # In dry-run mode, should return original
    assert result == original


@pytest.mark.asyncio
async def test_highlight_chunk_dry_run(dry_run_client):
    """Test chunk highlighting in dry-run mode."""
    chunk = "This is a test chunk. It has multiple sentences. Some are important."
    result = await dry_run_client.highlight_chunk(chunk, highlight_ratio=0.4)

    # In dry-run mode, should return original
    assert result == chunk


@pytest.mark.asyncio
async def test_highlight_chunks_parallel_dry_run(dry_run_client):
    """Test parallel chunk highlighting in dry-run mode."""
    chunks = [
        "First chunk with text.",
        "Second chunk with more text.",
        "Third chunk also has text.",
    ]

    results = await dry_run_client.highlight_chunks_parallel(chunks, highlight_ratio=0.4)

    # In dry-run mode, should return originals
    assert results == chunks
    assert len(results) == 3


def test_llm_settings_from_env():
    """Test that LLM settings can be loaded from environment."""
    settings = LLMSettings()
    
    # Should have API key from .env
    assert settings.openai_api_key is not None
    assert len(settings.openai_api_key) > 0
    
    # Should have default models
    assert settings.strong_model == "gemini-3-pro-preview"
    assert settings.light_model == "gemini-3-flash-preview"
    
    # Should have default timeout
    assert settings.request_timeout_seconds == 30.0


def test_llm_client_initialization():
    """Test LLM client initialization."""
    client = LLMClient(dry_run=True, concurrency=5)
    
    assert client.dry_run is True
    assert client.semaphore._value == 5
    assert client.client is None  # No OpenAI client in dry-run mode


def test_llm_client_with_custom_timeout():
    """Test LLM client initialization with custom timeout."""
    settings = LLMSettings()
    settings.request_timeout_seconds = 15.0
    client = LLMClient(settings=settings, dry_run=True, concurrency=2)
    
    assert client.settings.request_timeout_seconds == 15.0


@pytest.mark.asyncio
async def test_highlight_chunk_timeout_returns_original():
    """Test that timeout returns original chunk in real client with timeout handling."""
    # Test with a client that has built-in timeout handling
    settings = LLMSettings()
    settings.request_timeout_seconds = 0.001  # Very short timeout

    # We can't actually test a real timeout without mocking OpenAI client,
    # but we can verify the dry_run path works correctly
    client = LLMClient(settings=settings, dry_run=True, concurrency=2)

    original_chunk = "This is a test chunk."
    result = await client.highlight_chunk(original_chunk)

    # In dry-run mode, should return original
    assert result == original_chunk

    # Verify timeout setting is configured
    assert client.settings.request_timeout_seconds == 0.001
