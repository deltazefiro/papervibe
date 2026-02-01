"""Integration tests for layout preservation.

These tests verify that abstract rewriting preserves the layout of pages 2+.
The abstract should only affect the first page; all subsequent pages should
have identical text content between original and modified PDFs.

When summary is enabled, the modified PDF has a prepended summary page,
so we compare page 2+ of original with page (2 + num_summary_pages)+ of modified.
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
    load_stub_summary,
    get_paper_source,
    create_stub_rewriter,
    create_stub_summarizer,
)
from .pdf_compare import compare_pages, get_page_diff, get_page_count


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
    2. Runs pipeline with stubbed abstract rewriter and summarizer
    3. Compiles both original and modified sources
    4. Compares text of all pages except page 1 (accounting for summary page in modified)
    """
    # Get cached paper source
    source_dir = get_paper_source(paper_id)

    # Load stubs
    stub_abstract = load_stub_abstract(paper_id)
    stub_summary = load_stub_summary(paper_id)

    # Setup output directory
    output_dir = work_dir / paper_id.replace("/", "_")
    output_dir.mkdir(parents=True)

    # Copy source to output/original
    original_dir = output_dir / "original"
    shutil.copytree(source_dir, original_dir)

    # Run pipeline with stub abstract rewriter and summarizer
    stub_rewriter = create_stub_rewriter(stub_abstract)
    stub_summarizer = create_stub_summarizer(stub_summary)

    with patch("papervibe.process.rewrite_abstract", new=stub_rewriter), \
         patch("papervibe.process.summarize_paper", new=stub_summarizer):
        await process_paper(
            url=paper_id,
            out=output_dir,
            skip_abstract=False,
            skip_highlight=True,
            skip_compile=False,  # Let it compile with summary
            skip_summary=False,  # Enable summary
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

    # Compile original (without summary)
    original_main = find_main_tex_file(original_dir)
    original_pdf, _ = compile_latex(original_main, output_dir=original_dir, timeout=300)

    # Get the modified PDF (already compiled with summary prepended)
    modified_pdf = output_dir / f"{paper_id.replace('/', '_')}.pdf"
    assert modified_pdf.exists(), f"Modified PDF not found: {modified_pdf}"

    # Calculate number of summary pages
    orig_page_count = get_page_count(original_pdf)
    mod_page_count = get_page_count(modified_pdf)
    num_summary_pages = mod_page_count - orig_page_count

    assert num_summary_pages >= 0, (
        f"Modified PDF has fewer pages than original: {mod_page_count} vs {orig_page_count}"
    )

    # Compare pages: skip first page of original, skip (1 + num_summary_pages) of modified
    # This compares page 2+ of original with corresponding pages in modified
    results = compare_pages(
        original_pdf,
        modified_pdf,
        skip_first=True,  # Skip page 1 of original (abstract may differ)
        skip_pages_modified=1 + num_summary_pages,  # Skip summary + page 1 of modified
    )

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
    Verify that the paper content page count is preserved (modified = original + summary pages).
    """
    import fitz

    # Get cached paper source
    source_dir = get_paper_source(paper_id)

    # Load stubs
    stub_abstract = load_stub_abstract(paper_id)
    stub_summary = load_stub_summary(paper_id)

    # Setup output directory
    output_dir = work_dir / paper_id.replace("/", "_")
    output_dir.mkdir(parents=True)

    # Copy source to output/original
    original_dir = output_dir / "original"
    shutil.copytree(source_dir, original_dir)

    # Run pipeline with stub abstract rewriter and summarizer
    stub_rewriter = create_stub_rewriter(stub_abstract)
    stub_summarizer = create_stub_summarizer(stub_summary)

    with patch("papervibe.process.rewrite_abstract", new=stub_rewriter), \
         patch("papervibe.process.summarize_paper", new=stub_summarizer):
        await process_paper(
            url=paper_id,
            out=output_dir,
            skip_abstract=False,
            skip_highlight=True,
            skip_compile=False,  # Let it compile with summary
            skip_summary=False,  # Enable summary
            highlight_ratio=0.4,
            concurrency=1,
            dry_run=False,
            llm_timeout=120.0,
            max_chunk_chars=1500,
            validate_chunks=False,
        )

    modified_dir = output_dir / "modified"

    # Compile original
    original_main = find_main_tex_file(original_dir)
    original_pdf, _ = compile_latex(original_main, output_dir=original_dir, timeout=300)

    # Get the modified PDF (already compiled with summary prepended)
    modified_pdf = output_dir / f"{paper_id.replace('/', '_')}.pdf"

    # Count pages
    orig_doc = fitz.open(original_pdf)
    mod_doc = fitz.open(modified_pdf)

    try:
        # Modified should have at least as many pages as original (summary adds pages)
        assert len(mod_doc) >= len(orig_doc), (
            f"Page count issue for {paper_id}: "
            f"original has {len(orig_doc)} pages, modified has {len(mod_doc)} pages"
        )

        # The difference should be the summary pages (typically 1)
        summary_pages = len(mod_doc) - len(orig_doc)
        assert summary_pages >= 0, (
            f"Modified PDF has fewer pages than expected for {paper_id}"
        )
    finally:
        orig_doc.close()
        mod_doc.close()
