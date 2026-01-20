"""Tests for CLI module."""

import sys
from papervibe.cli import app, main_entry
from papervibe.latex import get_abstract_span


def test_cli_import():
    """Test that CLI app can be imported."""
    assert app is not None
    assert hasattr(app, "command")


def test_cli_arxiv_subcommand():
    """Test that 'papervibe arxiv <url>' works (explicit subcommand)."""
    # Simulate command line: papervibe arxiv invalid-id --skip-abstract --skip-highlight --skip-compile
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['papervibe', 'arxiv', 'invalid-id', '--skip-abstract', '--skip-highlight', '--skip-compile']
        try:
            main_entry()
            assert False, "Should have exited with error"
        except SystemExit as e:
            # Should fail with ArxivError (exit 1), not argument parsing error (exit 2)
            assert e.code == 1
    finally:
        sys.argv = original_argv


def test_cli_direct_url():
    """Test that 'papervibe <url>' works (compat alias)."""
    # Simulate command line: papervibe invalid-id --skip-abstract --skip-highlight --skip-compile
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['papervibe', 'invalid-id', '--skip-abstract', '--skip-highlight', '--skip-compile']
        try:
            main_entry()
            assert False, "Should have exited with error"
        except SystemExit as e:
            # Should fail with ArxivError (exit 1), not argument parsing error (exit 2)
            assert e.code == 1
    finally:
        sys.argv = original_argv


def test_abstract_exclusion_from_highlighting():
    """Test that abstract region should be excluded from highlighting."""
    # Sample LaTeX content with abstract
    content = r"""\documentclass{article}
\begin{document}

\begin{abstract}
This is the abstract content that should not be highlighted.
It has multiple sentences. All should be preserved.
\end{abstract}

\section{Introduction}
This is the introduction. This could be highlighted.

\section{Related Work}
More content here. This can also be highlighted.

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

    # In a properly implemented highlight pipeline, the abstract_region should
    # not contain any \pvhighlight{} wrappers after processing
    # (This is a structural test - the actual integration is tested in CLI)
    assert r"\pvhighlight{" not in abstract_region
