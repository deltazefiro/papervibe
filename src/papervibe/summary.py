"""Summary page generation and PDF merging."""

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from pypdf import PdfReader, PdfWriter

from .compile import compile_latex, check_latexmk_available, CompileError

logger = logging.getLogger(__name__)


def markdown_to_latex(markdown: str) -> str:
    """
    Convert markdown to LaTeX.

    Handles:
    - ### Headings -> \\section*{...}
    - **bold** -> \\textbf{...}
    - *italic* -> \\textit{...}
    - Bullet lists (- item) -> itemize environment
    - Escapes LaTeX special characters

    Args:
        markdown: Markdown text

    Returns:
        LaTeX formatted text
    """
    lines = markdown.split("\n")
    result_lines = []
    in_list = False

    for line in lines:
        # Check if this is a list item
        list_match = re.match(r"^(\s*)[-*]\s+(.+)$", line)
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            # Close any open list
            if in_list:
                result_lines.append("\\end{itemize}")
                in_list = False

            level = len(heading_match.group(1))
            title = _process_line(heading_match.group(2))
            if level <= 2:
                result_lines.append(f"\\section*{{{title}}}")
            elif level == 3:
                result_lines.append(f"\\subsection*{{{title}}}")
            else:
                result_lines.append(f"\\subsubsection*{{{title}}}")

        elif list_match:
            # Start list if not in one
            if not in_list:
                result_lines.append("\\begin{itemize}")
                in_list = True

            item_text = _process_line(list_match.group(2))
            result_lines.append(f"  \\item {item_text}")

        else:
            # Close any open list if we hit a non-list line
            if in_list and line.strip() == "":
                result_lines.append("\\end{itemize}")
                in_list = False

            # Regular paragraph text
            processed = _process_line(line)
            result_lines.append(processed)

    # Close any remaining open list
    if in_list:
        result_lines.append("\\end{itemize}")

    return "\n".join(result_lines)


def _process_line(text: str) -> str:
    """
    Process a line: preserve math, escape special chars, then convert inline formatting.

    Args:
        text: Text to process

    Returns:
        Processed text with escaping and formatting
    """
    # Extract math expressions to protect them from escaping
    math_placeholders = {}
    counter = 0

    def protect_math(match):
        nonlocal counter
        # Use a placeholder without special characters that won't be escaped
        placeholder = f"MATHPLACEHOLDER{counter}ENDMATH"
        math_placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    # Protect display math ($$...$$) first, then inline math ($...$)
    text = re.sub(r"\$\$(.+?)\$\$", protect_math, text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", protect_math, text)

    # Escape special characters (outside of math)
    escaped = _escape_latex(text)

    # Convert inline formatting (bold, italic)
    formatted = _convert_inline(escaped)

    # Restore math expressions
    for placeholder, math in math_placeholders.items():
        formatted = formatted.replace(placeholder, math)

    return formatted


def _escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters.

    Args:
        text: Text to escape

    Returns:
        Escaped text
    """
    # Order matters: backslash first
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def _convert_inline(text: str) -> str:
    """
    Convert inline markdown formatting (bold, italic).

    Args:
        text: Text with markdown inline formatting

    Returns:
        Text with LaTeX inline formatting
    """
    # Bold: **text** -> \textbf{text}
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)

    # Italic: *text* -> \textit{text}
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)

    return text


def create_summary_latex(summary_markdown: str, paper_title: str = "Paper Summary") -> str:
    """
    Create a complete standalone LaTeX document for the summary.

    Args:
        summary_markdown: Summary in markdown format
        paper_title: Title for the summary page

    Returns:
        Complete LaTeX document as string
    """
    body = markdown_to_latex(summary_markdown)

    document = f"""\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{geometry}}
\\usepackage{{parskip}}
\\usepackage{{amsmath,amssymb}}

\\geometry{{margin=1in}}

\\pagestyle{{empty}}

\\begin{{document}}

\\begin{{center}}
\\Large\\textbf{{{paper_title}}}
\\end{{center}}

\\vspace{{0.5cm}}

{body}

\\end{{document}}
"""
    return document


def generate_summary_pdf(
    summary_markdown: str,
    output_dir: Path,
    paper_title: str = "PaperVibe Summary",
    timeout: int = 60,
) -> Optional[Path]:
    """
    Generate a PDF from a markdown summary.

    Args:
        summary_markdown: Summary in markdown format
        output_dir: Directory to write files to
        paper_title: Title for the summary page
        timeout: Compilation timeout in seconds

    Returns:
        Path to generated PDF, or None if generation failed
    """
    if not check_latexmk_available():
        logger.warning("latexmk not available, cannot generate summary PDF")
        return None

    if not summary_markdown.strip():
        logger.debug("Empty summary, skipping summary PDF generation")
        return None

    # Create the LaTeX document
    latex_content = create_summary_latex(summary_markdown, paper_title)

    # Write to file
    summary_tex = output_dir / "summary.tex"
    summary_tex.write_text(latex_content, encoding="utf-8")

    try:
        pdf_path, _ = compile_latex(summary_tex, output_dir=output_dir, timeout=timeout)
        logger.info("Summary PDF generated: %s", pdf_path)
        return pdf_path
    except CompileError as e:
        logger.warning("Failed to compile summary PDF: %s", e)
        return None


def merge_pdfs(pdfs: list[Path], output_path: Path) -> Path:
    """
    Merge multiple PDFs into one.

    Args:
        pdfs: List of PDF paths to merge (in order)
        output_path: Path for the merged output PDF

    Returns:
        Path to merged PDF

    Raises:
        ValueError: If no valid PDFs provided
        Exception: On merge failure
    """
    if not pdfs:
        raise ValueError("No PDFs to merge")

    writer = PdfWriter()

    for pdf_path in pdfs:
        if not pdf_path.exists():
            logger.warning("PDF not found, skipping: %s", pdf_path)
            continue

        reader = PdfReader(pdf_path)
        for page in reader.pages:
            writer.add_page(page)

    if len(writer.pages) == 0:
        raise ValueError("No pages to merge")

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info("Merged %d PDFs into: %s", len(pdfs), output_path)
    return output_path


def prepend_summary_to_pdf(
    paper_pdf: Path,
    summary_markdown: str,
    output_path: Path,
    paper_title: str = "PaperVibe Summary",
) -> Path:
    """
    Generate a summary PDF and prepend it to the paper PDF.

    Args:
        paper_pdf: Path to the compiled paper PDF
        summary_markdown: Summary in markdown format
        output_path: Path for the final merged PDF
        paper_title: Title for the summary page

    Returns:
        Path to final PDF (merged if summary exists, original otherwise)
    """
    if not summary_markdown.strip():
        logger.debug("No summary to prepend, using original PDF")
        if paper_pdf != output_path:
            shutil.copy2(paper_pdf, output_path)
        return output_path

    # Generate summary PDF in the same directory
    summary_pdf = generate_summary_pdf(
        summary_markdown,
        output_dir=paper_pdf.parent,
        paper_title=paper_title,
    )

    if summary_pdf is None:
        logger.warning("Could not generate summary PDF, using original paper PDF")
        if paper_pdf != output_path:
            shutil.copy2(paper_pdf, output_path)
        return output_path

    # Merge summary + paper
    try:
        merge_pdfs([summary_pdf, paper_pdf], output_path)
        return output_path
    except Exception as e:
        logger.warning("Failed to merge PDFs: %s, using original paper PDF", e)
        if paper_pdf != output_path:
            shutil.copy2(paper_pdf, output_path)
        return output_path
