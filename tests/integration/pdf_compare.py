"""PDF comparison utilities for integration tests."""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF


def extract_pages_text(pdf_path: Path, skip_first: bool = True) -> List[str]:
    """
    Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file
        skip_first: Whether to skip the first page

    Returns:
        List of text content for each page
    """
    doc = fitz.open(pdf_path)
    try:
        pages = []
        for i, page in enumerate(doc):
            if skip_first and i == 0:
                continue
            pages.append(page.get_text())
        return pages
    finally:
        doc.close()


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by collapsing whitespace.

    Args:
        text: Raw text

    Returns:
        Normalized text with collapsed whitespace
    """
    # Collapse all whitespace (spaces, newlines, tabs) into single spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def compare_pages(
    original: Path,
    modified: Path,
    skip_first: bool = True,
) -> List[Tuple[int, str, str, bool]]:
    """
    Compare pages between original and modified PDFs.

    Args:
        original: Path to original PDF
        modified: Path to modified PDF
        skip_first: Whether to skip the first page

    Returns:
        List of (page_num, original_text, modified_text, match) tuples
    """
    orig_pages = extract_pages_text(original, skip_first)
    mod_pages = extract_pages_text(modified, skip_first)

    if len(orig_pages) != len(mod_pages):
        raise ValueError(
            f"Page count mismatch: original has {len(orig_pages) + (1 if skip_first else 0)} pages, "
            f"modified has {len(mod_pages) + (1 if skip_first else 0)} pages"
        )

    results = []
    for i, (orig, mod) in enumerate(zip(orig_pages, mod_pages)):
        page_num = i + 2 if skip_first else i + 1  # 1-indexed
        orig_norm = normalize_text(orig)
        mod_norm = normalize_text(mod)
        match = orig_norm == mod_norm
        results.append((page_num, orig_norm, mod_norm, match))

    return results


def get_page_diff(original_text: str, modified_text: str, context: int = 3) -> str:
    """
    Generate a word-level diff between two page texts.

    Args:
        original_text: Normalized original page text
        modified_text: Normalized modified page text
        context: Number of context words around differences

    Returns:
        Human-readable diff string
    """
    orig_words = original_text.split()
    mod_words = modified_text.split()

    matcher = SequenceMatcher(None, orig_words, mod_words)
    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            diff_parts.append(f"  Changed: '{' '.join(orig_words[i1:i2])}' -> '{' '.join(mod_words[j1:j2])}'")
        elif tag == "delete":
            diff_parts.append(f"  Deleted: '{' '.join(orig_words[i1:i2])}'")
        elif tag == "insert":
            diff_parts.append(f"  Inserted: '{' '.join(mod_words[j1:j2])}'")

    if not diff_parts:
        return "No differences found"

    return "\n".join(diff_parts[:20])  # Limit output
