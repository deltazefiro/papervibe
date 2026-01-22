"""Evaluation harness for abstract rewriting: process samples and generate comparison PDFs."""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papervibe.latex import extract_abstract, replace_abstract, inject_preamble
from papervibe.llm import LLMClient, LLMSettings
from papervibe.logging import setup_logging
from papervibe.process import rewrite_abstract

logger = logging.getLogger("papervibe.eval_abstract")


def compile_latex(tex_file: Path, output_pdf: Path) -> bool:
    """
    Compile a LaTeX file to PDF using latexmk.

    Args:
        tex_file: Path to the .tex file
        output_pdf: Path where the PDF should be saved

    Returns:
        True if compilation succeeded, False otherwise
    """
    try:
        logger.debug(f"Compiling {tex_file} with latexmk")
        result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_file.name,
            ],
            cwd=tex_file.parent,
            capture_output=True,
            timeout=60,
        )

        logger.debug(f"latexmk return code: {result.returncode}")
        if result.returncode != 0:
            logger.debug(
                f"latexmk stderr: {result.stderr.decode('utf-8', errors='replace')}"
            )

        pdf_in_dir = tex_file.with_suffix(".pdf")
        if pdf_in_dir.exists():
            logger.debug(f"Moving PDF from {pdf_in_dir} to {output_pdf}")
            pdf_in_dir.rename(output_pdf)
            return True
        logger.debug(f"PDF not created at {pdf_in_dir}")
        return False
    except Exception as e:
        logger.debug(f"Compilation exception: {e}")
        print(f"Compilation error: {e}")
        return False
    finally:
        for ext in [".aux", ".log", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = tex_file.with_suffix(ext)
            if aux_file.exists():
                logger.debug(f"Removing auxiliary file: {aux_file}")
                aux_file.unlink()


def create_comparison_image(
    original_pdf: Path, rewritten_pdf: Path, output_jpg: Path
) -> bool:
    """
    Create a side-by-side comparison image from two PDFs.

    Args:
        original_pdf: Path to the original PDF
        rewritten_pdf: Path to the rewritten PDF
        output_jpg: Path for the output comparison JPG

    Returns:
        True if successful, False otherwise
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        logger.debug(
            f"Creating comparison image from {original_pdf} and {rewritten_pdf}"
        )

        doc_orig = fitz.open(original_pdf)
        doc_rewritten = fitz.open(rewritten_pdf)

        page_orig = doc_orig[0]
        page_rewritten = doc_rewritten[0]

        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix_orig = page_orig.get_pixmap(matrix=mat)
        pix_rewritten = page_rewritten.get_pixmap(matrix=mat)

        logger.debug(f"Original pixmap: {pix_orig.width}x{pix_orig.height}")
        logger.debug(f"Rewritten pixmap: {pix_rewritten.width}x{pix_rewritten.height}")

        img_orig = Image.frombytes(
            "RGB", [pix_orig.width, pix_orig.height], pix_orig.samples
        )
        img_rewritten = Image.frombytes(
            "RGB", [pix_rewritten.width, pix_rewritten.height], pix_rewritten.samples
        )

        width = img_orig.width + img_rewritten.width
        height = max(img_orig.height, img_rewritten.height)
        combined = Image.new("RGB", (width, height), (255, 255, 255))

        combined.paste(img_orig, (0, 0))
        combined.paste(img_rewritten, (img_orig.width, 0))

        combined.save(output_jpg, "JPEG", quality=85)
        logger.debug(f"Saved comparison image to {output_jpg}")

        doc_orig.close()
        doc_rewritten.close()

        return True
    except Exception as e:
        logger.debug(f"Image creation exception: {e}", exc_info=True)
        print(f"Image creation error: {e}")
        return False


async def process_sample(
    sample_path: Path,
    output_dir: Path,
    llm_client: LLMClient,
) -> dict:
    """
    Process a single sample: extract abstract, rewrite it, compile PDFs.

    Args:
        sample_path: Path to the sample .tex file
        output_dir: Output directory for this sample
        llm_client: LLM client for abstract rewriting

    Returns:
        Dictionary with processing stats
    """
    print(f"\nProcessing {sample_path.name}...")
    logger.debug(f"Sample path: {sample_path}")
    logger.debug(f"Output directory: {output_dir}")

    if output_dir.exists():
        import shutil

        logger.debug(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content = sample_path.read_text(encoding="utf-8")
    logger.debug(f"Read {len(content)} characters from sample")

    # Extract abstract
    abstract_result = extract_abstract(content)
    if abstract_result is None:
        print(f"  No abstract found in {sample_path.name}, skipping...")
        return {
            "name": sample_path.stem,
            "original_abstract_len": 0,
            "rewritten_abstract_len": 0,
            "original_pdf": False,
            "rewritten_pdf": False,
            "comparison": False,
            "error": "No abstract found",
        }

    original_abstract, abs_start, abs_end = abstract_result
    print(f"  Found abstract: {len(original_abstract)} chars")
    logger.debug(f"Original abstract: {original_abstract[:100]}...")

    # Rewrite abstract
    print(f"  Rewriting abstract...")
    rewritten_abstract = await rewrite_abstract(llm_client, original_abstract)
    print(f"  Rewritten abstract: {len(rewritten_abstract)} chars")
    logger.debug(f"Rewritten abstract: {rewritten_abstract[:100]}...")

    # Create original and rewritten versions (both with preamble injected)
    original_with_preamble = inject_preamble(content)
    rewritten_content = replace_abstract(content, rewritten_abstract)
    rewritten_with_preamble = inject_preamble(rewritten_content)

    # Write tex files
    original_tex = output_dir / "original.tex"
    rewritten_tex = output_dir / "rewritten.tex"
    original_tex.write_text(original_with_preamble, encoding="utf-8")
    rewritten_tex.write_text(rewritten_with_preamble, encoding="utf-8")
    logger.debug(f"Wrote tex files: {original_tex}, {rewritten_tex}")

    # Also save just the abstract text for easy comparison
    abstracts_txt = output_dir / "abstracts.txt"
    abstracts_txt.write_text(
        f"=== ORIGINAL ABSTRACT ===\n{original_abstract}\n\n"
        f"=== REWRITTEN ABSTRACT ===\n{rewritten_abstract}\n",
        encoding="utf-8",
    )
    logger.debug(f"Wrote abstracts comparison to {abstracts_txt}")

    if hasattr(llm_client, "stats") and any(llm_client.stats.values()):
        print(f"  LLM Stats: {llm_client.stats}")
        logger.debug(f"LLM Stats: {llm_client.stats}")

    print(f"  Compiling original PDF...")
    original_pdf = output_dir / "original.pdf"
    original_success = compile_latex(original_tex, original_pdf)
    logger.debug(
        f"Original PDF compilation: {'success' if original_success else 'failed'}"
    )

    print(f"  Compiling rewritten PDF...")
    rewritten_pdf = output_dir / "rewritten.pdf"
    rewritten_success = compile_latex(rewritten_tex, rewritten_pdf)
    logger.debug(
        f"Rewritten PDF compilation: {'success' if rewritten_success else 'failed'}"
    )

    comparison_success = False
    if original_success and rewritten_success:
        print(f"  Creating comparison image...")
        comparison_jpg = output_dir / "compare.jpg"
        comparison_success = create_comparison_image(
            original_pdf, rewritten_pdf, comparison_jpg
        )
        logger.debug(
            f"Comparison image creation: {'success' if comparison_success else 'failed'}"
        )

    return {
        "name": sample_path.stem,
        "original_abstract_len": len(original_abstract),
        "rewritten_abstract_len": len(rewritten_abstract),
        "original_pdf": original_success,
        "rewritten_pdf": rewritten_success,
        "comparison": comparison_success,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate abstract rewriting on sample LaTeX files"
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).parent / "samples" / "abstract",
        help="Directory containing sample .tex files with abstracts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "out" / "abstract",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Process only this specific sample (by name, without .tex extension)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (skip LLM calls)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="LLM request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(verbose=1 if args.verbose else 0)

    if args.sample:
        sample_files = [args.samples_dir / f"{args.sample}.tex"]
        if not sample_files[0].exists():
            print(f"Error: Sample {args.sample}.tex not found in {args.samples_dir}")
            return 1
    else:
        sample_files = sorted(args.samples_dir.glob("*.tex"))

    if not sample_files:
        print(f"No sample .tex files found in {args.samples_dir}")
        return 1

    print(f"Found {len(sample_files)} sample(s)")
    logger.debug(f"Sample files: {[f.name for f in sample_files]}")

    try:
        settings = LLMSettings()
        settings.request_timeout_seconds = args.timeout
        llm_client = LLMClient(settings=settings, concurrency=1, dry_run=args.dry_run)
        print(f"Using model: {settings.strong_model} (timeout: {args.timeout}s)")
        logger.debug(f"LLM client initialized with dry_run={args.dry_run}")
    except Exception as e:
        logger.debug(f"LLM client initialization exception: {e}", exc_info=True)
        print(f"Error initializing LLM client: {e}")
        return 1

    results = []
    for sample_path in sample_files:
        sample_output_dir = args.output_dir / sample_path.stem
        logger.debug(f"Processing sample: {sample_path} -> {sample_output_dir}")
        result = await process_sample(
            sample_path,
            sample_output_dir,
            llm_client,
        )
        results.append(result)
        logger.debug(f"Result for {sample_path.stem}: {result}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        else:
            print(f"  Original abstract: {result['original_abstract_len']} chars")
            print(f"  Rewritten abstract: {result['rewritten_abstract_len']} chars")
            print(f"  Original PDF: {'OK' if result['original_pdf'] else 'FAILED'}")
            print(f"  Rewritten PDF: {'OK' if result['rewritten_pdf'] else 'FAILED'}")
            print(f"  Comparison: {'OK' if result['comparison'] else 'FAILED'}")

    print(f"\nOutput directory: {args.output_dir}")
    logger.debug("Evaluation complete")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
