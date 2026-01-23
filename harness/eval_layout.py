"""Evaluation harness for layout preservation: verify second page text matches between original and modified PDFs.

The abstract should only affect the first page layout. By replacing the abstract with
stub lorem text (simulating LLM rewrites of varying lengths), the second page should
remain identical between original and modified PDFs if layout is preserved correctly.

This harness patches the main program's rewrite_abstract function to return stub text,
ensuring the test uses the exact same code path as the main CLI.
"""

import argparse
import asyncio
import difflib
import logging
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papervibe.compile import compile_latex, check_latexmk_available, CompileError
from papervibe.latex import find_main_tex_file
from papervibe.logging import setup_logging
from papervibe.process import process_paper

# Stub lorem text to simulate LLM-rewritten abstract (intentionally different length)
STUB_ABSTRACT = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""


def create_comparison_image(
    original_pdf: Path, modified_pdf: Path, output_jpg: Path
) -> bool:
    """
    Create a side-by-side comparison image from first pages of two PDFs.

    Args:
        original_pdf: Path to the original PDF
        modified_pdf: Path to the modified PDF
        output_jpg: Path for the output comparison JPG

    Returns:
        True if successful, False otherwise
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        doc_orig = fitz.open(original_pdf)
        doc_modified = fitz.open(modified_pdf)

        page_orig = doc_orig[0]
        page_modified = doc_modified[0]

        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix_orig = page_orig.get_pixmap(matrix=mat)
        pix_modified = page_modified.get_pixmap(matrix=mat)

        img_orig = Image.frombytes(
            "RGB", [pix_orig.width, pix_orig.height], pix_orig.samples
        )
        img_modified = Image.frombytes(
            "RGB", [pix_modified.width, pix_modified.height], pix_modified.samples
        )

        width = img_orig.width + img_modified.width
        height = max(img_orig.height, img_modified.height)
        combined = Image.new("RGB", (width, height), (255, 255, 255))

        combined.paste(img_orig, (0, 0))
        combined.paste(img_modified, (img_orig.width, 0))

        combined.save(output_jpg, "JPEG", quality=85)

        doc_orig.close()
        doc_modified.close()

        return True
    except Exception as e:
        print(f"  Image creation error: {e}")
        return False


def extract_second_page_text(pdf_path: Path) -> str:
    """
    Extract text from the second page of a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Text content of the second page, or empty string if PDF has less than 2 pages
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        if len(doc) < 2:
            return ""
        page = doc[1]  # Second page (0-indexed)
        return page.get_text()
    finally:
        doc.close()


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by collapsing whitespace.

    Args:
        text: Raw text

    Returns:
        Normalized text
    """
    # Collapse all whitespace (spaces, newlines, tabs) into single spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


async def stub_rewrite_abstract(llm_client, original_abstract: str) -> str:
    """Stub function that returns lorem ipsum instead of calling LLM."""
    return STUB_ABSTRACT


async def run_pipeline(arxiv_id: str, output_dir: Path, dry_run: bool = False) -> bool:
    """
    Run the papervibe pipeline with skip_highlight.

    Args:
        arxiv_id: arXiv paper ID
        output_dir: Output directory
        dry_run: If True, use stub abstract instead of LLM

    Returns:
        True if successful, False otherwise
    """
    try:
        if dry_run:
            # Patch rewrite_abstract to return stub text instead of calling LLM
            with patch("papervibe.process.rewrite_abstract", new=stub_rewrite_abstract):
                await process_paper(
                    url=arxiv_id,
                    out=output_dir,
                    skip_abstract=False,
                    skip_highlight=True,
                    skip_compile=True,
                    highlight_ratio=0.4,
                    concurrency=1,
                    dry_run=False,  # Not CLI dry_run - we want abstract replacement
                    llm_timeout=120.0,
                    max_chunk_chars=1500,
                    validate_chunks=False,
                )
        else:
            # Real LLM abstract
            await process_paper(
                url=arxiv_id,
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
        return True
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return False


def evaluate_paper(arxiv_id: str, base_output_dir: Path, dry_run: bool = False) -> dict:
    """
    Evaluate layout preservation for a single paper.

    Workflow:
    1. Run pipeline (with real LLM abstract, or stub if dry_run)
    2. Compile both original and modified sources
    3. Compare second page text (should be identical)

    Args:
        arxiv_id: arXiv paper ID
        base_output_dir: Base output directory
        dry_run: If True, use stub abstract instead of LLM

    Returns:
        Dictionary with evaluation results
    """
    # Sanitize arxiv_id for directory name
    safe_id = arxiv_id.replace("/", "_")
    output_dir = base_output_dir / safe_id

    print(f"\nEvaluating {arxiv_id}...")

    # Clear output directory to ensure fresh results
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"  Cleared existing output: {output_dir}")

    # Step 1: Run pipeline
    mode = "stub abstract" if dry_run else "LLM abstract"
    print(f"  Running pipeline with {mode}...")
    if not asyncio.run(run_pipeline(arxiv_id, output_dir, dry_run=dry_run)):
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": "Pipeline failed",
        }

    original_dir = output_dir / "original"
    modified_dir = output_dir / "modified"

    # Check directories exist
    if not original_dir.exists():
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": f"Original directory not found: {original_dir}",
        }

    if not modified_dir.exists():
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": f"Modified directory not found: {modified_dir}",
        }

    # Step 2: Compile original
    print(f"  Compiling original sources...")
    try:
        main_tex = find_main_tex_file(original_dir)
        original_pdf_path, _ = compile_latex(
            main_tex, output_dir=original_dir, timeout=300
        )
        print(f"  Original PDF: {original_pdf_path}")
    except (CompileError, FileNotFoundError) as e:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": f"Original compilation failed: {e}",
        }

    # Step 3: Compile modified
    print(f"  Compiling modified sources...")
    try:
        modified_main = find_main_tex_file(modified_dir)
        modified_pdf_path, _ = compile_latex(
            modified_main, output_dir=modified_dir, timeout=300
        )
        print(f"  Modified PDF: {modified_pdf_path}")
    except (CompileError, FileNotFoundError) as e:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": f"Modified compilation failed: {e}",
        }

    # Step 4: Create side-by-side comparison image
    print(f"  Creating comparison image...")
    comparison_jpg = output_dir / "compare.jpg"
    comparison_created = create_comparison_image(
        original_pdf_path, modified_pdf_path, comparison_jpg
    )
    if comparison_created:
        print(f"  Comparison image: {comparison_jpg}")

    # Step 5: Extract second page text
    print(f"  Extracting second page text...")
    try:
        original_text = extract_second_page_text(original_pdf_path)
        modified_text = extract_second_page_text(modified_pdf_path)
    except Exception as e:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": f"Text extraction failed: {e}",
        }

    # Check if PDFs have at least 2 pages
    if not original_text:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": "Original PDF has less than 2 pages",
        }
    if not modified_text:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": "Modified PDF has less than 2 pages",
        }

    # Step 6: Normalize and compare
    original_normalized = normalize_text(original_text)
    modified_normalized = normalize_text(modified_text)

    if original_normalized == modified_normalized:
        print(f"  PASS: Second page text matches")
        return {
            "arxiv_id": arxiv_id,
            "success": True,
            "match": True,
            "comparison_image": str(comparison_jpg) if comparison_created else None,
        }
    else:
        print(f"  FAIL: Second page text differs")

        # Generate diff for debugging
        original_words = original_normalized.split()
        modified_words = modified_normalized.split()

        diff = list(
            difflib.unified_diff(
                original_words,
                modified_words,
                fromfile="original",
                tofile="modified",
                lineterm="",
                n=3,
            )
        )

        diff_text = "\n".join(diff[:50])  # Limit output
        if len(diff) > 50:
            diff_text += f"\n... ({len(diff) - 50} more lines)"

        return {
            "arxiv_id": arxiv_id,
            "success": True,
            "match": False,
            "diff": diff_text,
            "original_len": len(original_normalized),
            "modified_len": len(modified_normalized),
            "comparison_image": str(comparison_jpg) if comparison_created else None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate layout preservation by comparing second page text between original and modified PDFs"
    )
    parser.add_argument(
        "arxiv_ids",
        nargs="*",
        default=["1706.03762"],
        help="arXiv paper IDs to evaluate (default: 1706.03762)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "out" / "layout",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub lorem abstract instead of LLM-generated abstract",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose logging (use -v for debug, -vv for more detail)",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if not check_latexmk_available():
        print("Error: latexmk not found. Please install TeX Live or similar.")
        return 1

    print(f"Evaluating {len(args.arxiv_ids)} paper(s)...")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'dry-run (stub abstract)' if args.dry_run else 'LLM abstract'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for arxiv_id in args.arxiv_ids:
        result = evaluate_paper(arxiv_id, args.output_dir, dry_run=args.dry_run)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = 0

    for result in results:
        print(f"\n{result['arxiv_id']}:")
        if not result["success"]:
            print(f"  ERROR: {result['error']}")
            errors += 1
        elif result["match"]:
            print(f"  PASS")
            if result.get("comparison_image"):
                print(f"  Comparison: {result['comparison_image']}")
            passed += 1
        else:
            print(f"  FAIL: Text differs")
            print(f"    Original length: {result['original_len']}")
            print(f"    Modified length: {result['modified_len']}")
            if result.get("comparison_image"):
                print(f"    Comparison: {result['comparison_image']}")
            if result.get("diff"):
                print(f"    Diff preview:")
                for line in result["diff"].split("\n")[:20]:
                    print(f"      {line}")
            failed += 1

    print("\n" + "-" * 60)
    print(
        f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Errors: {errors}"
    )

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
