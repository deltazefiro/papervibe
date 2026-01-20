"""Evaluation harness for highlighting: process samples and generate comparison PDFs."""

import argparse
import asyncio
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papervibe.latex import inject_preamble
from papervibe.process import highlight_content_parallel
from papervibe.llm import LLMClient, LLMSettings


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
        # Run latexmk in the same directory as the tex file
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

        # Check if PDF was created
        pdf_in_dir = tex_file.with_suffix(".pdf")
        if pdf_in_dir.exists():
            # Move to output location
            pdf_in_dir.rename(output_pdf)
            return True
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        return False
    finally:
        # Clean up auxiliary files
        for ext in [".aux", ".log", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = tex_file.with_suffix(ext)
            if aux_file.exists():
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

        # Open both PDFs
        doc_orig = fitz.open(original_pdf)
        doc_high = fitz.open(highlighted_pdf)

        # Get first page from each
        page_orig = doc_orig[0]
        page_high = doc_high[0]

        # Render to images (at 2x resolution for better quality)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix_orig = page_orig.get_pixmap(matrix=mat)
        pix_high = page_high.get_pixmap(matrix=mat)

        # Convert to PIL Images
        img_orig = Image.frombytes(
            "RGB", [pix_orig.width, pix_orig.height], pix_orig.samples
        )
        img_high = Image.frombytes(
            "RGB", [pix_high.width, pix_high.height], pix_high.samples
        )

        # Create side-by-side comparison
        width = img_orig.width + img_high.width
        height = max(img_orig.height, img_high.height)
        combined = Image.new("RGB", (width, height), (255, 255, 255))

        combined.paste(img_orig, (0, 0))
        combined.paste(img_high, (img_orig.width, 0))

        # Save as JPG
        combined.save(output_jpg, "JPEG", quality=85)

        doc_orig.close()
        doc_high.close()

        return True
    except Exception as e:
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

    # Clear and create output directory
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read original content
    content = sample_path.read_text(encoding="utf-8")

    # Highlight content FIRST (before preamble injection)
    print(f"  Highlighting content (ratio={highlight_ratio})...")
    highlighted_content = await highlight_content_parallel(
        content,
        llm_client,
        highlight_ratio=highlight_ratio,
    )

    # Then inject preamble to BOTH original and highlighted versions
    original_with_preamble = inject_preamble(content)
    highlighted_with_preamble = inject_preamble(highlighted_content)

    # Write files
    original_tex = output_dir / "original.tex"
    highlighted_tex = output_dir / "highlighted.tex"
    original_tex.write_text(original_with_preamble, encoding="utf-8")
    highlighted_tex.write_text(highlighted_with_preamble, encoding="utf-8")

    # Count wrappers
    wrapper_count = highlighted_content.count(r"\pvhighlight{")
    print(f"  Added {wrapper_count} \\pvhighlight{{}} wrappers")

    # Show LLM stats if available
    if hasattr(llm_client, "stats") and any(llm_client.stats.values()):
        print(f"  LLM Stats: {llm_client.stats}")

    # Compile PDFs
    print(f"  Compiling original PDF...")
    original_pdf = output_dir / "original.pdf"
    original_success = compile_latex(original_tex, original_pdf)

    print(f"  Compiling highlighted PDF...")
    highlighted_pdf = output_dir / "highlighted.pdf"
    highlighted_success = compile_latex(highlighted_tex, highlighted_pdf)

    # Create comparison image
    comparison_success = False
    if original_success and highlighted_success:
        print(f"  Creating comparison image...")
        comparison_jpg = output_dir / "compare.jpg"
        comparison_success = create_comparison_image(
            original_pdf, highlighted_pdf, comparison_jpg
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
        default=Path(__file__).parent / "output",
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
        default=60.0,
        help="LLM request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Find sample files
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

    # Initialize LLM client
    try:
        settings = LLMSettings()
        settings.request_timeout_seconds = args.timeout
        llm_client = LLMClient(settings=settings, concurrency=8, dry_run=args.dry_run)
        print(f"Using model: {settings.light_model} (timeout: {args.timeout}s)")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        return 1

    # Process each sample
    results = []
    for sample_path in sample_files:
        sample_output_dir = args.output_dir / sample_path.stem
        result = await process_sample(
            sample_path,
            sample_output_dir,
            args.highlight_ratio,
            llm_client,
        )
        results.append(result)

    # Print summary
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

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
