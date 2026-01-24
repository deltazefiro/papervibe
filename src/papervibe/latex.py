"""LaTeX file processing and manipulation."""

import re
from pathlib import Path
from typing import Optional, Tuple, List

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode


class LatexError(Exception):
    """Base exception for LaTeX-related errors."""

    pass


def strip_latex_comments(text: str) -> str:
    """
    Remove LaTeX comments (unescaped % to end of line) from text.

    Args:
        text: LaTeX text potentially containing comments

    Returns:
        Text with comments removed
    """
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        # Find unescaped % and remove from there to end of line
        new_line = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                # Found unescaped %, skip rest of line
                break
            new_line.append(line[i])
            i += 1
        result_lines.append("".join(new_line))
    return "\n".join(result_lines)


def find_main_tex_file(directory: Path) -> Path:
    """
    Find the main .tex file in a directory using heuristic scoring.

    Heuristics:
    - Prefer files with \\documentclass
    - Prefer files with \\begin{document}
    - Prefer shorter filenames (e.g., main.tex, paper.tex)
    - Penalize files in subdirectories

    Args:
        directory: Directory containing .tex files

    Returns:
        Path to the main .tex file

    Raises:
        LatexError: If no suitable main file is found
    """
    tex_files = list(directory.rglob("*.tex"))

    if not tex_files:
        raise LatexError(f"No .tex files found in {directory}")

    def score_file(path: Path) -> int:
        """Score a .tex file for being the main file (higher is better)."""
        score = 0

        # Penalize files in subdirectories
        if path.parent != directory:
            score -= 100

        # Read file content
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return -1000

        # Strong indicators
        if r"\documentclass" in content:
            score += 100
        if r"\begin{document}" in content:
            score += 50

        # Prefer common main file names
        name_lower = path.stem.lower()
        if name_lower in ["main", "paper", "article", "manuscript"]:
            score += 30
        elif name_lower.startswith("main") or name_lower.startswith("paper"):
            score += 20

        # Prefer shorter names
        score -= len(path.stem) // 2

        return score

    # Score all files and pick the best
    scored_files = [(score_file(f), f) for f in tex_files]
    scored_files.sort(reverse=True, key=lambda x: x[0])

    if scored_files[0][0] < 0:
        raise LatexError(f"Could not identify main .tex file in {directory}")

    return scored_files[0][1]


def find_references_cutoff(content: str) -> Optional[int]:
    """
    Find the character position where references/bibliography begins.

    Args:
        content: LaTeX file content

    Returns:
        Character offset where references start, or None if not found
    """
    # Look for common reference section markers
    patterns = [
        r"\\begin{thebibliography}",
        r"\\bibliography{",
        r"\\printbibliography",
        r"\\section\*?{references}",
        r"\\section\*?{bibliography}",
    ]

    earliest_pos = None
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            pos = match.start()
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos

    return earliest_pos


def extract_abstract(content: str) -> Optional[Tuple[str, int, int]]:
    """
    Extract the abstract from LaTeX content.

    Args:
        content: LaTeX file content

    Returns:
        Tuple of (abstract_text, start_offset, end_offset) or None if not found
        The offsets point to the \\begin{abstract} and \\end{abstract} commands.
        The abstract_text has LaTeX comments stripped.
    """
    # Find \begin{abstract}...\end{abstract}
    pattern = r"(\\begin\{abstract\})(.*?)(\\end\{abstract\})"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if not match:
        return None

    # Strip LaTeX comments from the abstract text
    abstract_text = strip_latex_comments(match.group(2)).strip()
    start = match.start()
    end = match.end()

    return abstract_text, start, end


def get_abstract_span(content: str) -> Optional[Tuple[int, int]]:
    """
    Get the character span of the abstract region (including begin/end tags).

    Args:
        content: LaTeX file content

    Returns:
        Tuple of (start_offset, end_offset) or None if not found
    """
    result = extract_abstract(content)
    if result is None:
        return None
    _, start, end = result
    return start, end


def replace_abstract(content: str, new_abstract: str) -> str:
    """
    Replace the abstract in LaTeX content with a new one, preserving original layout.

    Uses \\pvreplaceblock to overlay the new abstract text on the original abstract's
    invisible footprint, ensuring the rest of the paper's layout is not affected by
    abstract length changes.

    Args:
        content: Original LaTeX content
        new_abstract: New abstract text

    Returns:
        Modified LaTeX content with replaced abstract

    Raises:
        LatexError: If abstract not found or replacement fails
    """
    result = extract_abstract(content)
    if result is None:
        raise LatexError("No abstract found in LaTeX content")

    # extract_abstract returns stripped abstract, but we need raw content for pvreplaceblock
    _, start, end = result

    # Escape % in new abstract (preserves literal % characters)
    new_abstract_escaped = re.sub(r"(?<!\\)%", r"\\%", new_abstract)

    # Find exact positions of begin and end tags
    begin_tag_end = content.find("}", start) + 1
    end_tag_start = content.rfind("\\", start, end)

    # Get raw content between tags (preserving all whitespace for accurate measurement)
    raw_original = content[begin_tag_end:end_tag_start]

    # Wrap with \pvreplaceblock{original}{new} to preserve layout
    new_content = (
        content[:begin_tag_end]
        + "\\pvreplaceblock{"
        + raw_original
        + "}{"
        + new_abstract_escaped
        + "}"
        + content[end_tag_start:]
    )

    return new_content


def has_xcolor_and_pvhighlight(content: str) -> bool:
    """
    Check if the LaTeX content already has xcolor package, default gray color, and \\pvhighlight macro.

    Args:
        content: LaTeX content

    Returns:
        True if xcolor, default gray color, and \\pvhighlight are present
    """
    has_xcolor = bool(re.search(r"\\usepackage(?:\[.*?\])?\{xcolor\}", content))
    has_pvhighlight = bool(re.search(r"\\newcommand\{?\\pvhighlight\}?", content))
    has_default_gray = bool(re.search(r"\\AtBeginDocument\{\\color\{gray\}\}", content))
    has_abstract_black = bool(re.search(r"\\pvabstractblack", content))
    has_replaceblock = bool(
        re.search(
            r"\\(long\\)?def\\pvreplaceblock|\\(long\\)?newcommand\{?\\pvreplaceblock\}?",
            content,
        )
    )

    return (
        has_xcolor
        and has_pvhighlight
        and has_default_gray
        and has_abstract_black
        and has_replaceblock
    )


def inject_preamble(content: str) -> str:
    """
    Inject xcolor package, default gray color, and \\pvhighlight macro into LaTeX preamble if not present.

    The components are injected right before \\begin{document}:
    - xcolor package for color support
    - AtBeginDocument hook to set all text gray by default
    - pvhighlight macro to highlight important content in black
    - Abstract environment override to keep abstract text black

    Args:
        content: LaTeX content

    Returns:
        Modified LaTeX content with injected preamble
    """
    if has_xcolor_and_pvhighlight(content):
        return content

    # Find \begin{document}
    match = re.search(r"\\begin\{document\}", content)
    if not match:
        raise LatexError("Could not find \\begin{document} in LaTeX content")

    inject_pos = match.start()

    # Build injection string
    parts = []

    if not re.search(r"\\usepackage(?:\[.*?\])?\{xcolor\}", content):
        parts.append("\\usepackage{xcolor}")

    if not re.search(r"\\AtBeginDocument\{\\color\{gray\}\}", content):
        parts.append("\\AtBeginDocument{\\color{gray}}")

    if not re.search(r"\\newcommand\{?\\pvhighlight\}?", content):
        parts.append("\\newcommand{\\pvhighlight}[1]{\\textcolor{black}{#1}}")

    # Add replaceblock macro for abstract padding
    # Strategy: Box original in white, overlay new text at top using \vbox to \ht
    # This guarantees identical vertical footprint since original determines height
    if not re.search(
        r"\\(long\\)?def\\pvreplaceblock|\\(long\\)?newcommand\{?\\pvreplaceblock\}?",
        content,
    ):
        parts.append("% Replace block macro: overlay new on white original")
        parts.append("\\newsavebox{\\pvoldbox}")
        parts.append("\\long\\def\\pvreplaceblock#1#2{%")
        # Box the original content (determines final height)
        parts.append(
            "  \\sbox\\pvoldbox{\\begin{minipage}{\\linewidth}\\ignorespaces #1\\unskip\\end{minipage}}%"
        )
        # Create a vbox of exact same height, with new content at top
        parts.append("  \\vbox to \\dimexpr\\ht\\pvoldbox+\\dp\\pvoldbox\\relax{%")
        parts.append("    \\noindent\\ignorespaces #2\\unskip%")
        parts.append("    \\vfill%")
        parts.append("  }%")
        parts.append("}")

    # Keep abstract text black (not gray)
    if not re.search(r"\\pvabstractblack", content):
        parts.append("% Keep abstract text black")
        parts.append("\\let\\pvoldabstract\\abstract")
        parts.append("\\let\\pvendoldabstract\\endabstract")
        parts.append(
            "\\renewenvironment{abstract}{\\pvoldabstract\\color{black}}{\\pvendoldabstract}"
        )
        parts.append("\\newcommand{\\pvabstractblack}{}% marker")

    if not parts:
        return content

    injection = "\n".join(parts) + "\n\n"

    return content[:inject_pos] + injection + content[inject_pos:]


def strip_pvhighlight_wrappers(text: str) -> str:
    """
    Strip all \\pvhighlight{...} wrappers from text, preserving content.

    This is brace-aware and handles nested braces properly.

    Args:
        text: LaTeX text potentially containing \\pvhighlight{...} wrappers

    Returns:
        Text with all \\pvhighlight wrappers removed
    """
    result = []
    i = 0

    while i < len(text):
        # Look for \pvhighlight{
        if text[i : i + 13] == "\\pvhighlight{":
            # Skip the \pvhighlight{ part
            i += 13

            # Extract the content within braces
            brace_level = 1
            content_start = i

            while i < len(text) and brace_level > 0:
                if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                    brace_level += 1
                elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                    brace_level -= 1
                i += 1

            # Add the content (without the closing brace)
            result.append(text[content_start : i - 1])
        else:
            result.append(text[i])
            i += 1

    return "".join(result)


def find_input_files(content: str, base_dir: Path) -> List[Path]:
    """
    Find all files referenced by \\input{} or \\include{} commands.

    Args:
        content: LaTeX file content
        base_dir: Base directory for resolving relative paths

    Returns:
        List of absolute paths to input files
    """
    input_files = []

    # Match \input{filename} and \include{filename}
    patterns = [
        r"\\input\{([^}]+)\}",
        r"\\include\{([^}]+)\}",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content):
            filename = match.group(1).strip()

            # Resolve path relative to base_dir
            # Add .tex extension if not present
            if not filename.endswith(".tex"):
                tex_path = base_dir / f"{filename}.tex"
            else:
                tex_path = base_dir / filename

            # Also try without adding .tex
            alt_path = base_dir / filename

            if tex_path.exists():
                input_files.append(tex_path)
            elif alt_path.exists():
                input_files.append(alt_path)

    return input_files
