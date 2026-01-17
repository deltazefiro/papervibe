"""Tests for compilation module."""

import pytest
from pathlib import Path
from papervibe.compile import (
    check_latexmk_available,
    compile_latex,
    clean_latex_aux_files,
    CompileError,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_check_latexmk_available():
    """Test checking for latexmk availability."""
    # This should return True or False depending on system
    result = check_latexmk_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not check_latexmk_available(),
    reason="latexmk not installed"
)
def test_compile_latex():
    """Test compiling a LaTeX file."""
    tex_file = FIXTURE_DIR / "main.tex"
    
    # Compile to a temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        pdf_path, log = compile_latex(tex_file, output_dir=output_dir, timeout=60)
        
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        assert len(log) > 0
        
        # Clean up
        clean_latex_aux_files(output_dir)


def test_compile_latex_nonexistent_file():
    """Test error when TeX file doesn't exist."""
    with pytest.raises(CompileError, match="not found"):
        compile_latex(Path("/tmp/nonexistent.tex"))


@pytest.mark.skipif(
    not check_latexmk_available(),
    reason="latexmk not installed"
)
def test_compile_latex_error():
    """Test compilation with a broken TeX file."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a broken TeX file with actual syntax error that prevents compilation
        broken_tex = tmpdir / "broken.tex"
        broken_tex.write_text(r"\documentclass{article}\begin{document}\end{document")  # Missing closing brace
        
        with pytest.raises(CompileError, match="failed"):
            compile_latex(broken_tex, timeout=30)


def test_clean_latex_aux_files():
    """Test cleaning auxiliary files."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create some aux files
        (tmpdir / "test.aux").write_text("aux content")
        (tmpdir / "test.log").write_text("log content")
        (tmpdir / "test.pdf").write_text("pdf content")
        
        clean_latex_aux_files(tmpdir)
        
        # Aux files should be removed
        assert not (tmpdir / "test.aux").exists()
        assert not (tmpdir / "test.log").exists()
        
        # PDF should remain
        assert (tmpdir / "test.pdf").exists()
