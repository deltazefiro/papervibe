"""Tests for multi-file LaTeX processing."""

import pytest
from pathlib import Path
from papervibe.latex import find_input_files


def test_find_input_files(tmp_path):
    """Test finding files referenced by \\input commands."""
    # Create test directory structure
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    sec_dir = base_dir / "sec"
    sec_dir.mkdir()
    
    # Create main.tex with input commands
    main_content = r"""
\documentclass{article}
\begin{document}
\input{sec/intro}
\input{sec/methods.tex}
\input{conclusion}
\end{document}
"""
    (base_dir / "main.tex").write_text(main_content)
    
    # Create referenced files
    (sec_dir / "intro.tex").write_text("Introduction content")
    (sec_dir / "methods.tex").write_text("Methods content")
    (base_dir / "conclusion.tex").write_text("Conclusion content")
    
    # Test finding input files
    input_files = find_input_files(main_content, base_dir)
    
    # Should find all 3 files
    assert len(input_files) == 3
    
    # Check that paths are resolved correctly
    filenames = {f.name for f in input_files}
    assert filenames == {"intro.tex", "methods.tex", "conclusion.tex"}


def test_find_input_files_nonexistent(tmp_path):
    """Test that nonexistent files are ignored."""
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    
    main_content = r"""
\documentclass{article}
\begin{document}
\input{missing_file}
\end{document}
"""
    
    input_files = find_input_files(main_content, base_dir)
    assert len(input_files) == 0


def test_find_input_files_empty():
    """Test with content that has no input commands."""
    content = r"""
\documentclass{article}
\begin{document}
Hello world!
\end{document}
"""
    
    input_files = find_input_files(content, Path("/tmp"))
    assert len(input_files) == 0
