"""Tests for arXiv module."""

import pytest
from papervibe.arxiv import parse_arxiv_id, ArxivError


def test_parse_new_style_id():
    """Test parsing new-style arXiv IDs."""
    arxiv_id, version = parse_arxiv_id("2107.03374")
    assert arxiv_id == "2107.03374"
    assert version is None


def test_parse_new_style_id_with_version():
    """Test parsing new-style arXiv IDs with version."""
    arxiv_id, version = parse_arxiv_id("2107.03374v2")
    assert arxiv_id == "2107.03374"
    assert version == "v2"


def test_parse_old_style_id():
    """Test parsing old-style arXiv IDs."""
    arxiv_id, version = parse_arxiv_id("hep-th/9901001")
    assert arxiv_id == "hep-th/9901001"
    assert version is None


def test_parse_old_style_id_with_version():
    """Test parsing old-style arXiv IDs with version."""
    arxiv_id, version = parse_arxiv_id("hep-th/9901001v3")
    assert arxiv_id == "hep-th/9901001"
    assert version == "v3"


def test_parse_abs_url():
    """Test parsing arXiv abstract URL."""
    arxiv_id, version = parse_arxiv_id("https://arxiv.org/abs/2107.03374")
    assert arxiv_id == "2107.03374"
    assert version is None


def test_parse_abs_url_with_version():
    """Test parsing arXiv abstract URL with version."""
    arxiv_id, version = parse_arxiv_id("https://arxiv.org/abs/2107.03374v1")
    assert arxiv_id == "2107.03374"
    assert version == "v1"


def test_parse_pdf_url():
    """Test parsing arXiv PDF URL."""
    arxiv_id, version = parse_arxiv_id("https://arxiv.org/pdf/2107.03374.pdf")
    assert arxiv_id == "2107.03374"
    assert version is None


def test_parse_old_style_url():
    """Test parsing old-style arXiv URL."""
    arxiv_id, version = parse_arxiv_id("https://arxiv.org/abs/hep-th/9901001")
    assert arxiv_id == "hep-th/9901001"
    assert version is None


def test_parse_invalid_url():
    """Test that invalid URLs raise errors."""
    with pytest.raises(ArxivError, match="Not an arXiv URL"):
        parse_arxiv_id("https://example.com/paper")


def test_parse_invalid_id():
    """Test that invalid IDs raise errors."""
    with pytest.raises(ArxivError, match="Invalid arXiv ID format"):
        parse_arxiv_id("invalid-id")
    
    with pytest.raises(ArxivError, match="Invalid arXiv ID format"):
        parse_arxiv_id("123")


def test_parse_arxiv_dot_prefix():
    """Test parsing IDs with arXiv. prefix."""
    arxiv_id, version = parse_arxiv_id("arXiv.2501.03218")
    assert arxiv_id == "2501.03218"
    assert version is None
    
    arxiv_id, version = parse_arxiv_id("arXiv.2409.14485v1")
    assert arxiv_id == "2409.14485"
    assert version == "v1"


def test_parse_arxiv_colon_prefix():
    """Test parsing IDs with arXiv: prefix."""
    arxiv_id, version = parse_arxiv_id("arXiv:2510.09608")
    assert arxiv_id == "2510.09608"
    assert version is None
    
    arxiv_id, version = parse_arxiv_id("arXiv:2107.03374v2")
    assert arxiv_id == "2107.03374"
    assert version == "v2"
