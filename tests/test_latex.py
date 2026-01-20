"""Tests for LaTeX module."""

import pytest
from pathlib import Path
from papervibe.latex import (
    find_main_tex_file,
    find_references_cutoff,
    extract_abstract,
    replace_abstract,
    has_xcolor_and_pvhighlight,
    inject_preamble,
    strip_pvhighlight_wrappers,
    LatexError,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_find_main_tex_file():
    """Test finding the main .tex file in a directory."""
    main_file = find_main_tex_file(FIXTURE_DIR)
    assert main_file.name == "main.tex"


def test_find_main_tex_file_no_tex(tmp_path):
    """Test error when no .tex files are found."""
    # Use a guaranteed empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(LatexError, match="No .tex files found"):
        find_main_tex_file(empty_dir)


def test_find_references_cutoff():
    """Test finding where references begin."""
    content = (FIXTURE_DIR / "main.tex").read_text()
    cutoff = find_references_cutoff(content)
    
    assert cutoff is not None
    assert "\\begin{thebibliography}" in content[cutoff:cutoff+50]


def test_find_references_cutoff_none():
    """Test when no references section is found."""
    content = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    cutoff = find_references_cutoff(content)
    assert cutoff is None


def test_extract_abstract():
    """Test extracting abstract from LaTeX."""
    content = (FIXTURE_DIR / "main.tex").read_text()
    result = extract_abstract(content)
    
    assert result is not None
    abstract_text, start, end = result
    
    assert "original abstract" in abstract_text
    assert "novel research" in abstract_text
    assert start > 0
    assert end > start


def test_extract_abstract_none():
    """Test when no abstract is found."""
    content = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    result = extract_abstract(content)
    assert result is None


def test_replace_abstract():
    """Test replacing abstract in LaTeX."""
    content = (FIXTURE_DIR / "main.tex").read_text()
    new_abstract = "This is a completely new abstract with different content."
    
    modified = replace_abstract(content, new_abstract)
    
    # Check that new abstract is present
    assert new_abstract in modified
    
    # Check that old abstract is not present
    assert "original abstract" not in modified
    
    # Check that document structure is preserved
    assert "\\begin{abstract}" in modified
    assert "\\end{abstract}" in modified
    assert "\\section{Introduction}" in modified


def test_replace_abstract_no_abstract():
    """Test error when trying to replace non-existent abstract."""
    content = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    
    with pytest.raises(LatexError, match="No abstract found"):
        replace_abstract(content, "New abstract")


def test_has_xcolor_and_pvhighlight():
    """Test detection of xcolor package, default gray color, and pvhighlight macro."""
    # None present
    content1 = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    assert not has_xcolor_and_pvhighlight(content1)

    # Only xcolor
    content2 = "\\usepackage{xcolor}\\begin{document}Hello\\end{document}"
    assert not has_xcolor_and_pvhighlight(content2)

    # xcolor and pvhighlight but no default gray
    content3 = "\\usepackage{xcolor}\\newcommand{\\pvhighlight}[1]{\\textcolor{black}{#1}}\\begin{document}Hello\\end{document}"
    assert not has_xcolor_and_pvhighlight(content3)

    # All present
    content4 = "\\usepackage{xcolor}\\AtBeginDocument{\\color{gray}}\\newcommand{\\pvhighlight}[1]{\\textcolor{black}{#1}}\\begin{document}Hello\\end{document}"
    assert has_xcolor_and_pvhighlight(content4)


def test_inject_preamble():
    """Test injecting xcolor, default gray color, and pvhighlight macro."""
    content = "\\documentclass{article}\\begin{document}Hello\\end{document}"

    modified = inject_preamble(content)

    assert "\\usepackage{xcolor}" in modified
    assert "\\AtBeginDocument{\\color{gray}}" in modified
    assert "\\newcommand{\\pvhighlight}" in modified
    assert "\\begin{document}" in modified

    # Check that components are injected before \begin{document}
    xcolor_pos = modified.find("\\usepackage{xcolor}")
    default_gray_pos = modified.find("\\AtBeginDocument{\\color{gray}}")
    pvhighlight_pos = modified.find("\\newcommand{\\pvhighlight}")
    doc_pos = modified.find("\\begin{document}")
    assert xcolor_pos < doc_pos
    assert default_gray_pos < doc_pos
    assert pvhighlight_pos < doc_pos


def test_inject_preamble_idempotent():
    """Test that preamble injection is idempotent."""
    content = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    
    modified1 = inject_preamble(content)
    modified2 = inject_preamble(modified1)
    
    # Should be the same after second injection
    assert modified1 == modified2


def test_inject_preamble_no_document():
    """Test error when \\begin{document} is not found."""
    content = "\\documentclass{article}Hello"
    
    with pytest.raises(LatexError, match="Could not find"):
        inject_preamble(content)


def test_strip_pvhighlight_wrappers():
    """Test stripping \\pvhighlight wrappers from text."""
    # Simple case
    text1 = "Hello \\pvhighlight{world} test"
    assert strip_pvhighlight_wrappers(text1) == "Hello world test"

    # Multiple wrappers
    text2 = "\\pvhighlight{First} and \\pvhighlight{second} text"
    assert strip_pvhighlight_wrappers(text2) == "First and second text"

    # Nested braces
    text3 = "\\pvhighlight{Text with {nested} braces}"
    assert strip_pvhighlight_wrappers(text3) == "Text with {nested} braces"

    # No wrappers
    text4 = "Plain text without wrappers"
    assert strip_pvhighlight_wrappers(text4) == "Plain text without wrappers"

    # Empty wrapper
    text5 = "Before \\pvhighlight{} after"
    assert strip_pvhighlight_wrappers(text5) == "Before  after"


def test_strip_pvhighlight_validation():
    """Test that stripping wrappers from highlighted text yields original."""
    original = "This is some text. And another sentence."
    highlighted = "This is some text. \\pvhighlight{And another sentence.}"

    stripped = strip_pvhighlight_wrappers(highlighted)
    assert stripped == original
