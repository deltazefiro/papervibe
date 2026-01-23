"""LaTeX compilation using latexmk."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple


class CompileError(Exception):
    """Exception raised for compilation errors."""
    pass


def check_latexmk_available() -> bool:
    """
    Check if latexmk is available in the system.
    
    Returns:
        True if latexmk is available, False otherwise
    """
    return shutil.which("latexmk") is not None


def compile_latex(
    tex_file: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> Tuple[Path, str]:
    """
    Compile a LaTeX file to PDF using latexmk.
    
    Args:
        tex_file: Path to the main .tex file
        output_dir: Directory for output files (defaults to same as tex_file)
        timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (pdf_path, compilation_log)
        
    Raises:
        CompileError: If compilation fails or latexmk is not available
    """
    if not check_latexmk_available():
        raise CompileError("latexmk not found. Please install TeX Live or similar.")
    
    if not tex_file.exists():
        raise CompileError(f"TeX file not found: {tex_file}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = tex_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run latexmk
    # -pdf: use pdflatex
    # -interaction=nonstopmode: don't stop for errors
    # -file-line-error: better error messages
    # -f: force mode (continue despite errors to complete full compilation cycle)
    #     This is needed because some arXiv sources have malformed LaTeX (e.g., \cite {key}
    #     with space) that causes warnings. Without -f, latexmk stops early and bibliography
    #     references remain undefined [?] in the output PDF.
    # -output-directory: where to put output files
    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-f",
        f"-output-directory={output_dir.absolute()}",
        str(tex_file.absolute()),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=tex_file.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        
        # Collect log
        log = result.stdout + "\n" + result.stderr
        
        # Check for PDF output
        pdf_name = tex_file.stem + ".pdf"
        pdf_path = output_dir / pdf_name
        
        if not pdf_path.exists():
            # Try to extract useful error information from log
            error_lines = [
                line for line in log.split("\n")
                if "error" in line.lower() or "!" in line
            ]
            error_summary = "\n".join(error_lines[-10:]) if error_lines else log[-1000:]
            
            raise CompileError(
                f"PDF compilation failed. LaTeX errors:\n{error_summary}"
            )
        
        return pdf_path, log
        
    except subprocess.TimeoutExpired:
        raise CompileError(f"Compilation timed out after {timeout} seconds")
    
    except Exception as e:
        if isinstance(e, CompileError):
            raise
        raise CompileError(f"Compilation failed: {e}")


def clean_latex_aux_files(directory: Path):
    """
    Remove auxiliary LaTeX files generated during compilation.
    
    Args:
        directory: Directory containing aux files
    """
    aux_extensions = [
        ".aux", ".log", ".out", ".toc", ".lof", ".lot",
        ".fls", ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg",
    ]
    
    for ext in aux_extensions:
        for file in directory.glob(f"*{ext}"):
            try:
                file.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
