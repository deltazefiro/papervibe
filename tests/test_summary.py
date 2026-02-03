"""Tests for summary.py - summary PDF generation and merging."""

import pytest
from pathlib import Path
from papervibe.summary import (
    markdown_to_latex,
    _escape_latex,
    _convert_inline,
    create_summary_latex,
    merge_pdfs,
)


class TestMarkdownToLatex:
    """Tests for markdown to LaTeX conversion."""

    def test_heading_level_3(self):
        """Test ### heading conversion."""
        result = markdown_to_latex("### My Heading")
        assert r"\subsection*{My Heading}" in result

    def test_heading_level_2(self):
        """Test ## heading conversion."""
        result = markdown_to_latex("## My Heading")
        assert r"\section*{My Heading}" in result

    def test_heading_level_4(self):
        """Test #### heading conversion."""
        result = markdown_to_latex("#### My Heading")
        assert r"\subsubsection*{My Heading}" in result

    def test_bold_text(self):
        """Test **bold** conversion."""
        result = markdown_to_latex("This is **bold** text")
        assert r"\textbf{bold}" in result

    def test_italic_text(self):
        """Test *italic* conversion."""
        result = markdown_to_latex("This is *italic* text")
        assert r"\textit{italic}" in result

    def test_bullet_list(self):
        """Test bullet list conversion."""
        markdown = """- Item 1
- Item 2
- Item 3"""
        result = markdown_to_latex(markdown)
        assert r"\begin{itemize}" in result
        assert r"\item Item 1" in result
        assert r"\item Item 2" in result
        assert r"\item Item 3" in result
        assert r"\end{itemize}" in result

    def test_mixed_content(self):
        """Test mixed heading, text, and list."""
        markdown = """### TL;DR

This paper introduces a **new method**.

### Key Points

- First point
- Second point"""
        result = markdown_to_latex(markdown)
        assert r"\subsection*{TL;DR}" in result
        assert r"\textbf{new method}" in result
        assert r"\subsection*{Key Points}" in result
        assert r"\begin{itemize}" in result

    def test_inline_math_preserved(self):
        """Test that inline math $...$ is preserved."""
        result = markdown_to_latex("The equation $x^2 + y^2 = z^2$ is famous.")
        assert "$x^2 + y^2 = z^2$" in result
        # Ensure the $ is not escaped
        assert r"\$" not in result

    def test_display_math_preserved(self):
        """Test that display math $$...$$ is preserved."""
        result = markdown_to_latex("The formula is:\n$$E = mc^2$$")
        assert "$$E = mc^2$$" in result

    def test_math_with_special_chars(self):
        """Test math containing special chars like _ and ^."""
        result = markdown_to_latex("Given $x_1, x_2$ and $a^{n+1}$.")
        assert "$x_1, x_2$" in result
        assert "$a^{n+1}$" in result
        # Underscore outside math should be escaped
        assert r"\_" not in result.replace("$x_1, x_2$", "").replace("$a^{n+1}$", "")

    def test_math_and_formatting_mixed(self):
        """Test math combined with bold/italic."""
        result = markdown_to_latex("The **key** equation is $f(x) = x^2$.")
        assert r"\textbf{key}" in result
        assert "$f(x) = x^2$" in result


class TestEscapeLatex:
    """Tests for LaTeX special character escaping."""

    def test_escape_ampersand(self):
        """Test & escaping."""
        result = _escape_latex("A & B")
        assert r"A \& B" == result

    def test_escape_percent(self):
        """Test % escaping."""
        result = _escape_latex("50%")
        assert r"50\%" == result

    def test_escape_dollar(self):
        """Test $ escaping."""
        result = _escape_latex("$100")
        assert r"\$100" == result

    def test_escape_underscore(self):
        """Test _ escaping."""
        result = _escape_latex("var_name")
        assert r"var\_name" == result

    def test_escape_hash(self):
        """Test # escaping."""
        result = _escape_latex("#1")
        assert r"\#1" == result


class TestConvertInline:
    """Tests for inline markdown conversion."""

    def test_bold(self):
        """Test bold conversion."""
        result = _convert_inline("**bold**")
        assert result == r"\textbf{bold}"

    def test_italic(self):
        """Test italic conversion."""
        result = _convert_inline("*italic*")
        assert result == r"\textit{italic}"

    def test_mixed(self):
        """Test mixed bold and italic."""
        result = _convert_inline("**bold** and *italic*")
        assert r"\textbf{bold}" in result
        assert r"\textit{italic}" in result


class TestCreateSummaryLatex:
    """Tests for complete LaTeX document generation."""

    def test_creates_document(self):
        """Test that a complete document is created."""
        result = create_summary_latex("### Test\n\nContent here.", "My Title")
        assert r"\documentclass" in result
        assert r"\begin{document}" in result
        assert r"\end{document}" in result
        assert r"\textbf{My Title}" in result

    def test_includes_content(self):
        """Test that content is included."""
        result = create_summary_latex("### Summary\n\nThis is **important**.")
        assert r"\subsection*{Summary}" in result
        assert r"\textbf{important}" in result


class TestMergePdfs:
    """Tests for PDF merging."""

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="No PDFs to merge"):
            merge_pdfs([], Path("/tmp/output.pdf"))

    def test_merge_nonexistent_files_raises(self, tmp_path):
        """Test that merging only nonexistent files raises ValueError."""
        with pytest.raises(ValueError, match="No pages to merge"):
            merge_pdfs(
                [tmp_path / "nonexistent1.pdf", tmp_path / "nonexistent2.pdf"],
                tmp_path / "output.pdf",
            )
