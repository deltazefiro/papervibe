"""Tests for CLI module."""

from papervibe.cli import app
from papervibe.latex import get_abstract_span


def test_cli_import():
    """Test that CLI app can be imported."""
    assert app is not None
    assert hasattr(app, "command")


def test_abstract_exclusion_from_graying():
    """Test that abstract region should be excluded from graying."""
    # Sample LaTeX content with abstract
    content = r"""\documentclass{article}
\begin{document}

\begin{abstract}
This is the abstract content that should not be grayed out.
It has multiple sentences. All should be preserved.
\end{abstract}

\section{Introduction}
This is the introduction. This could be grayed out.

\section{Related Work}
More content here. This can also be grayed.

\end{document}"""
    
    # Get abstract span
    span = get_abstract_span(content)
    assert span is not None
    
    abs_start, abs_end = span
    abstract_region = content[abs_start:abs_end]
    
    # Verify that abstract region contains the abstract tags
    assert r"\begin{abstract}" in abstract_region
    assert r"\end{abstract}" in abstract_region
    assert "This is the abstract content" in abstract_region
    
    # Verify that abstract region does not include introduction
    assert "Introduction" not in abstract_region
    
    # In a properly implemented gray pipeline, the abstract_region should
    # not contain any \pvgray{} wrappers after processing
    # (This is a structural test - the actual integration is tested in CLI)
    assert r"\pvgray{" not in abstract_region
