"""LaTeX file processing and manipulation."""

import re
from pathlib import Path
from typing import Optional, Tuple, List

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode


class LatexError(Exception):
    """Base exception for LaTeX-related errors."""
    pass


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
        The offsets point to the \\begin{abstract} and \\end{abstract} commands
    """
    # Find \begin{abstract}...\end{abstract}
    pattern = r"(\\begin\{abstract\})(.*?)(\\end\{abstract\})"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not match:
        return None
    
    abstract_text = match.group(2).strip()
    start = match.start()
    end = match.end()
    
    return abstract_text, start, end


def replace_abstract(content: str, new_abstract: str) -> str:
    """
    Replace the abstract in LaTeX content with a new one.
    
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
    
    _, start, end = result
    
    # Preserve the \begin{abstract} and \end{abstract} tags
    # Just replace what's between them
    begin_tag_end = content.find("}", start) + 1
    end_tag_start = content.rfind("\\", start, end)
    
    new_content = (
        content[:begin_tag_end] +
        "\n" + new_abstract + "\n" +
        content[end_tag_start:]
    )
    
    return new_content


def has_xcolor_and_pvgray(content: str) -> bool:
    """
    Check if the LaTeX content already has xcolor package and \\pvgray macro.
    
    Args:
        content: LaTeX content
        
    Returns:
        True if both xcolor and \\pvgray are present
    """
    has_xcolor = bool(re.search(r"\\usepackage(?:\[.*?\])?\{xcolor\}", content))
    has_pvgray = bool(re.search(r"\\newcommand\{?\\pvgray\}?", content))
    
    return has_xcolor and has_pvgray


def inject_preamble(content: str) -> str:
    """
    Inject xcolor package and \\pvgray macro into LaTeX preamble if not present.
    
    The macro is injected right before \\begin{document}.
    
    Args:
        content: LaTeX content
        
    Returns:
        Modified LaTeX content with injected preamble
    """
    if has_xcolor_and_pvgray(content):
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
    
    if not re.search(r"\\newcommand\{?\\pvgray\}?", content):
        parts.append("\\newcommand{\\pvgray}[1]{\\textcolor{gray}{#1}}")
    
    if not parts:
        return content
    
    injection = "\n".join(parts) + "\n\n"
    
    return content[:inject_pos] + injection + content[inject_pos:]


def strip_pvgray_wrappers(text: str) -> str:
    """
    Strip all \\pvgray{...} wrappers from text, preserving content.
    
    This is brace-aware and handles nested braces properly.
    
    Args:
        text: LaTeX text potentially containing \\pvgray{...} wrappers
        
    Returns:
        Text with all \\pvgray wrappers removed
    """
    result = []
    i = 0
    
    while i < len(text):
        # Look for \pvgray{
        if text[i:i+8] == "\\pvgray{":
            # Skip the \pvgray{ part
            i += 8
            
            # Extract the content within braces
            brace_level = 1
            content_start = i
            
            while i < len(text) and brace_level > 0:
                if text[i] == "{" and (i == 0 or text[i-1] != "\\"):
                    brace_level += 1
                elif text[i] == "}" and (i == 0 or text[i-1] != "\\"):
                    brace_level -= 1
                i += 1
            
            # Add the content (without the closing brace)
            result.append(text[content_start:i-1])
        else:
            result.append(text[i])
            i += 1
    
    return "".join(result)
