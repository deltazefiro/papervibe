"""Shared fixtures for integration tests."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from papervibe.arxiv import download_arxiv_source
from papervibe.process import extract_footnotes


# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
PAPERS_JSON = FIXTURES_DIR / "papers.json"
RESPONSES_DIR = FIXTURES_DIR / "responses"


def load_paper_ids() -> List[str]:
    """Load paper IDs from papers.json."""
    with open(PAPERS_JSON) as f:
        data = json.load(f)
    return [
        p["id"] for p in data["papers"]
        if p.get("skip_reason") is None
    ]


def load_stub_abstract(paper_id: str) -> str:
    """Load pre-computed stub abstract for a paper."""
    path = RESPONSES_DIR / paper_id / "abstract.txt"
    if not path.exists():
        raise FileNotFoundError(f"No stub abstract for paper {paper_id}: {path}")
    return path.read_text().strip()


def load_stub_highlights(paper_id: str) -> dict:
    """
    Load pre-computed highlight snippets for a paper.

    Returns:
        Dictionary mapping chunk index to list of snippets
    """
    highlights_dir = RESPONSES_DIR / paper_id / "highlights"
    if not highlights_dir.exists():
        return {}

    result = {}
    for chunk_file in sorted(highlights_dir.glob("chunk_*.txt")):
        # Extract chunk index from filename (e.g., "chunk_000.txt" -> 0)
        chunk_idx = int(chunk_file.stem.split("_")[1])
        snippets = [
            line.strip()
            for line in chunk_file.read_text().splitlines()
            if line.strip()
        ]
        result[chunk_idx] = snippets

    return result


# Use system temp directory for paper cache
_PAPER_CACHE_DIR = None


def get_paper_cache_dir() -> Path:
    """Get or create the paper cache directory in system temp."""
    global _PAPER_CACHE_DIR
    if _PAPER_CACHE_DIR is None:
        _PAPER_CACHE_DIR = Path(tempfile.gettempdir()) / "papervibe_test_cache"
        _PAPER_CACHE_DIR.mkdir(exist_ok=True)
    return _PAPER_CACHE_DIR


def get_paper_source(paper_id: str) -> Path:
    """
    Get paper source directory, downloading if not cached.

    Args:
        paper_id: arXiv paper ID

    Returns:
        Path to the cached source directory
    """
    cache_dir = get_paper_cache_dir()
    safe_id = paper_id.replace("/", "_")
    cached_dir = cache_dir / safe_id

    if not cached_dir.exists():
        cached_dir.mkdir(parents=True)
        download_arxiv_source(paper_id, None, cached_dir)

    return cached_dir


def create_stub_rewriter(stub_text: str):
    """
    Create a stub rewrite_abstract function that returns fixed text.

    Preserves footnotes from original abstract to match real behavior.

    Args:
        stub_text: The stub abstract text to return

    Returns:
        Async function compatible with rewrite_abstract signature
    """
    async def stub_rewrite_abstract(llm_client, original_abstract: str) -> str:
        _, footnotes = extract_footnotes(original_abstract)
        result = stub_text
        if footnotes:
            result = result.rstrip() + " " + " ".join(footnotes)
        return result

    return stub_rewrite_abstract


@pytest.fixture(scope="session")
def paper_cache():
    """Session-scoped fixture providing the paper cache directory."""
    return get_paper_cache_dir()


@pytest.fixture
def work_dir(tmp_path):
    """Provide a clean working directory for each test."""
    return tmp_path
