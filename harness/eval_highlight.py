"""Evaluation harness for highlighting: process samples and generate comparison PDFs."""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papervibe.latex import inject_preamble
from papervibe.llm import LLMClient, LLMSettings
from papervibe.logging import setup_logging
from papervibe.process import highlight_content_parallel

logger = logging.getLogger("papervibe.eval_highlight")


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
    original_pdf: Path, highlighted_pdf: Path, output_jpg: Path
) -> bool:
    """
    Create a side-by-side comparison image from two PDFs.

    Args:
        original_pdf: Path to the original PDF
        highlighted_pdf: Path to the highlighted PDF
        output_jpg: Path for the output comparison JPG

    Returns:
        True if successful, False otherwise
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        logger.debug(
            f"Creating comparison image from {original_pdf} and {highlighted_pdf}"
        )

        doc_orig = fitz.open(original_pdf)
        doc_high = fitz.open(highlighted_pdf)

        page_orig = doc_orig[0]
        page_high = doc_high[0]

        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix_orig = page_orig.get_pixmap(matrix=mat)
        pix_high = page_high.get_pixmap(matrix=mat)

        logger.debug(f"Original pixmap: {pix_orig.width}x{pix_orig.height}")
        logger.debug(f"Highlighted pixmap: {pix_high.width}x{pix_high.height}")

        img_orig = Image.frombytes(
            "RGB", [pix_orig.width, pix_orig.height], pix_orig.samples
        )
        img_high = Image.frombytes(
            "RGB", [pix_high.width, pix_high.height], pix_high.samples
        )

        width = img_orig.width + img_high.width
        height = max(img_orig.height, img_high.height)
        combined = Image.new("RGB", (width, height), (255, 255, 255))

        combined.paste(img_orig, (0, 0))
        combined.paste(img_high, (img_orig.width, 0))

        combined.save(output_jpg, "JPEG", quality=85)
        logger.debug(f"Saved comparison image to {output_jpg}")

        doc_orig.close()
        doc_high.close()

        return True
    except Exception as e:
        logger.debug(f"Image creation exception: {e}", exc_info=True)
        print(f"Image creation error: {e}")
        return False


async def process_sample(
    sample_path: Path,
    output_dir: Path,
    highlight_ratio: float,
    llm_client: LLMClient,
) -> dict:
    """
    Process a single sample: create original and highlighted versions, compile PDFs.

    Args:
        sample_path: Path to the sample .tex file
        output_dir: Output directory for this sample
        highlight_ratio: Ratio of content to highlight
        llm_client: LLM client for highlighting

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

    print(f"  Highlighting content (ratio={highlight_ratio})...")
    highlighted_content = await highlight_content_parallel(
        content,
        llm_client,
        highlight_ratio=highlight_ratio,
    )
    logger.debug(f"Highlighted content length: {len(highlighted_content)}")

    original_with_preamble = inject_preamble(content)
    highlighted_with_preamble = inject_preamble(highlighted_content)
    logger.debug("Injected preambles to both versions")

    original_tex = output_dir / "original.tex"
    highlighted_tex = output_dir / "highlighted.tex"
    original_tex.write_text(original_with_preamble, encoding="utf-8")
    highlighted_tex.write_text(highlighted_with_preamble, encoding="utf-8")
    logger.debug(f"Wrote tex files: {original_tex}, {highlighted_tex}")

    wrapper_count = highlighted_content.count(r"\pvhighlight{")
    print(f"  Added {wrapper_count} \\pvhighlight{{}} wrappers")
    logger.debug(f"Wrapper count: {wrapper_count}")

    if hasattr(llm_client, "stats") and any(llm_client.stats.values()):
        print(f"  LLM Stats: {llm_client.stats}")
        logger.debug(f"LLM Stats: {llm_client.stats}")

    print(f"  Compiling original PDF...")
    original_pdf = output_dir / "original.pdf"
    original_success = compile_latex(original_tex, original_pdf)
    logger.debug(
        f"Original PDF compilation: {'success' if original_success else 'failed'}"
    )

    print(f"  Compiling highlighted PDF...")
    highlighted_pdf = output_dir / "highlighted.pdf"
    highlighted_success = compile_latex(highlighted_tex, highlighted_pdf)
    logger.debug(
        f"Highlighted PDF compilation: {'success' if highlighted_success else 'failed'}"
    )

    comparison_success = False
    if original_success and highlighted_success:
        print(f"  Creating comparison image...")
        comparison_jpg = output_dir / "compare.jpg"
        comparison_success = create_comparison_image(
            original_pdf, highlighted_pdf, comparison_jpg
        )
        logger.debug(
            f"Comparison image creation: {'success' if comparison_success else 'failed'}"
        )

    return {
        "name": sample_path.stem,
        "wrapper_count": wrapper_count,
        "original_pdf": original_success,
        "highlighted_pdf": highlighted_success,
        "comparison": comparison_success,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate highlighting on sample LaTeX files"
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).parent / "samples",
        help="Directory containing sample .tex files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "out",
        help="Output directory for results",
    )
    parser.add_argument(
        "--highlight-ratio",
        type=float,
        default=0.33,
        help="Ratio of content to highlight (default: 0.33)",
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
        help="LLM request timeout in seconds (default: 60)",
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
        llm_client = LLMClient(settings=settings, concurrency=8, dry_run=args.dry_run)
        print(f"Using model: {settings.light_model} (timeout: {args.timeout}s)")
        logger.debug(
            f"LLM client initialized with concurrency=8, dry_run={args.dry_run}"
        )
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
            args.highlight_ratio,
            llm_client,
        )
        results.append(result)
        logger.debug(f"Result for {sample_path.stem}: {result}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Wrappers: {result['wrapper_count']}")
        print(f"  Original PDF: {'✓' if result['original_pdf'] else '✗'}")
        print(f"  Highlighted PDF: {'✓' if result['highlighted_pdf'] else '✗'}")
        print(f"  Comparison: {'✓' if result['comparison'] else '✗'}")

    print(f"\nOutput directory: {args.output_dir}")
    logger.debug("Evaluation complete")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
