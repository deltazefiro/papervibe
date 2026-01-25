"""Integration tests for layout preservation.

These tests verify that abstract rewriting preserves the layout of pages 2+.
The abstract should only affect the first page; all subsequent pages should
have identical text content between original and modified PDFs.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from papervibe.compile import compile_latex, check_latexmk_available
from papervibe.latex import find_main_tex_file
from papervibe.process import process_paper

from .conftest import (
    load_paper_ids,
    load_stub_abstract,
    get_paper_source,
    create_stub_rewriter,
)
from .pdf_compare import compare_pages, get_page_diff


# Skip all tests in this module if latexmk is not available
pytestmark = pytest.mark.skipif(
    not check_latexmk_available(),
    reason="latexmk not available"
)


@pytest.mark.integration
@pytest.mark.parametrize("paper_id", load_paper_ids())
@pytest.mark.asyncio
async def test_layout_preserved_after_abstract_rewrite(paper_id: str, work_dir: Path):
    """
    Verify that pages 2+ match between original and modified PDFs.

    This test:
    1. Downloads paper source (cached)
    2. Runs pipeline with stubbed abstract rewriter
    3. Compiles both original and modified sources
    4. Compares text of all pages except page 1
    """
    # Get cached paper source
    source_dir = get_paper_source(paper_id)

    # Load stub abstract
    stub_abstract = load_stub_abstract(paper_id)

    # Setup output directory
    output_dir = work_dir / paper_id.replace("/", "_")
    output_dir.mkdir(parents=True)

    # Copy source to output/original
    original_dir = output_dir / "original"
    shutil.copytree(source_dir, original_dir)

    # Run pipeline with stub abstract rewriter
    stub_rewriter = create_stub_rewriter(stub_abstract)

    with patch("papervibe.process.rewrite_abstract", new=stub_rewriter):
        await process_paper(
            url=paper_id,
            out=output_dir,
            skip_abstract=False,
            skip_highlight=True,
            skip_compile=True,  # We'll compile manually to compare both
            highlight_ratio=0.4,
            concurrency=1,
            dry_run=False,
            llm_timeout=120.0,
            max_chunk_chars=1500,
            validate_chunks=False,
        )

    modified_dir = output_dir / "modified"

    # Verify directories exist
    assert original_dir.exists(), f"Original directory not found: {original_dir}"
    assert modified_dir.exists(), f"Modified directory not found: {modified_dir}"

    # Compile original
    original_main = find_main_tex_file(original_dir)
    original_pdf, _ = compile_latex(original_main, output_dir=original_dir, timeout=300)

    # Compile modified
    modified_main = find_main_tex_file(modified_dir)
    modified_pdf, _ = compile_latex(modified_main, output_dir=modified_dir, timeout=300)

    # Compare all pages except the first
    results = compare_pages(original_pdf, modified_pdf, skip_first=True)

    # Verify all pages match
    failures = []
    for page_num, orig_text, mod_text, match in results:
        if not match:
            diff = get_page_diff(orig_text, mod_text)
            failures.append(f"Page {page_num} differs:\n{diff}")

    if failures:
        pytest.fail(
            f"Layout not preserved for paper {paper_id}:\n" +
            "\n".join(failures)
        )


@pytest.mark.integration
@pytest.mark.parametrize("paper_id", load_paper_ids())
@pytest.mark.asyncio
async def test_page_count_preserved(paper_id: str, work_dir: Path):
    """
    Verify that the number of pages is identical between original and modified PDFs.
    """
    import fitz

    # Get cached paper source
    source_dir = get_paper_source(paper_id)

    # Load stub abstract
    stub_abstract = load_stub_abstract(paper_id)

    # Setup output directory
    output_dir = work_dir / paper_id.replace("/", "_")
    output_dir.mkdir(parents=True)

    # Copy source to output/original
    original_dir = output_dir / "original"
    shutil.copytree(source_dir, original_dir)

    # Run pipeline with stub abstract rewriter
    stub_rewriter = create_stub_rewriter(stub_abstract)

    with patch("papervibe.process.rewrite_abstract", new=stub_rewriter):
        await process_paper(
            url=paper_id,
            out=output_dir,
            skip_abstract=False,
            skip_highlight=True,
            skip_compile=True,
            highlight_ratio=0.4,
            concurrency=1,
            dry_run=False,
            llm_timeout=120.0,
            max_chunk_chars=1500,
            validate_chunks=False,
        )

    modified_dir = output_dir / "modified"

    # Compile both
    original_main = find_main_tex_file(original_dir)
    original_pdf, _ = compile_latex(original_main, output_dir=original_dir, timeout=300)

    modified_main = find_main_tex_file(modified_dir)
    modified_pdf, _ = compile_latex(modified_main, output_dir=modified_dir, timeout=300)

    # Count pages
    orig_doc = fitz.open(original_pdf)
    mod_doc = fitz.open(modified_pdf)

    try:
        assert len(orig_doc) == len(mod_doc), (
            f"Page count mismatch for {paper_id}: "
            f"original has {len(orig_doc)} pages, modified has {len(mod_doc)} pages"
        )
    finally:
        orig_doc.close()
        mod_doc.close()
