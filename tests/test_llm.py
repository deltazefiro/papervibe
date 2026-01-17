"""Tests for LLM module."""

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
async def test_gray_out_chunk_dry_run(dry_run_client):
    """Test chunk graying in dry-run mode."""
    chunk = "This is a test chunk. It has multiple sentences. Some are important."
    result = await dry_run_client.gray_out_chunk(chunk, gray_ratio=0.4)
    
    # In dry-run mode, should return original
    assert result == chunk


@pytest.mark.asyncio
async def test_gray_out_chunks_parallel_dry_run(dry_run_client):
    """Test parallel chunk graying in dry-run mode."""
    chunks = [
        "First chunk with text.",
        "Second chunk with more text.",
        "Third chunk also has text.",
    ]
    
    results = await dry_run_client.gray_out_chunks_parallel(chunks, gray_ratio=0.4)
    
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
    assert settings.strong_model == "gpt-4o"
    assert settings.light_model == "gpt-4o-mini"


def test_llm_client_initialization():
    """Test LLM client initialization."""
    client = LLMClient(dry_run=True, concurrency=5)
    
    assert client.dry_run is True
    assert client.semaphore._value == 5
    assert client.client is None  # No OpenAI client in dry-run mode
