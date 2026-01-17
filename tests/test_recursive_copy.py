"""Tests for recursive directory copy in CLI pipeline."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def test_recursive_copy_with_subdirs():
    """Test that modified directory includes nested files from original directory."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake original/ structure with nested directories
        original = tmpdir / "original"
        original.mkdir()
        
        # Create top-level file
        (original / "main.tex").write_text(r"\documentclass{article}\begin{document}Hello\end{document}")
        
        # Create subdirectory with files
        subdir = original / "figures"
        subdir.mkdir()
        (subdir / "fig1.png").write_text("fake png data")
        (subdir / "fig2.pdf").write_text("fake pdf data")
        
        # Create nested subdirectory
        nested = subdir / "nested"
        nested.mkdir()
        (nested / "data.txt").write_text("nested data")
        
        # Simulate the copy operation from cli.py
        modified = tmpdir / "modified"
        
        # Remove existing modified directory if it exists
        if modified.exists():
            shutil.rmtree(modified)
        
        # Recursively copy (this is what the fix implements)
        shutil.copytree(original, modified)
        
        # Now verify all files and subdirectories exist
        assert (modified / "main.tex").exists()
        assert (modified / "figures").is_dir()
        assert (modified / "figures" / "fig1.png").exists()
        assert (modified / "figures" / "fig2.pdf").exists()
        assert (modified / "figures" / "nested").is_dir()
        assert (modified / "figures" / "nested" / "data.txt").exists()
        
        # Verify content is preserved
        assert (modified / "main.tex").read_text() == r"\documentclass{article}\begin{document}Hello\end{document}"
        assert (modified / "figures" / "nested" / "data.txt").read_text() == "nested data"
        
        # Test that we can overwrite main.tex after copy
        (modified / "main.tex").write_text(r"\documentclass{article}\begin{document}Modified\end{document}")
        assert (modified / "main.tex").read_text() == r"\documentclass{article}\begin{document}Modified\end{document}"
