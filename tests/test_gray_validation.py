"""Test for gray validation fix."""

import pytest
from papervibe.latex import strip_pvgray_wrappers
from papervibe.gray import validate_grayed_chunk


def test_strip_pvgray_wrappers():
    """Test that strip_pvgray_wrappers correctly removes wrappers."""
    # Simple case
    text = r"This is \pvgray{less important} and this is important."
    expected = "This is less important and this is important."
    assert strip_pvgray_wrappers(text) == expected
    
    # Nested braces
    text = r"This is \pvgray{less \textbf{important}} text."
    expected = r"This is less \textbf{important} text."
    assert strip_pvgray_wrappers(text) == expected
    
    # Multiple wrappers
    text = r"\pvgray{First} middle \pvgray{second} end."
    expected = "First middle second end."
    assert strip_pvgray_wrappers(text) == expected
    
    # Empty wrapper (should still be removed)
    text = r"Text \pvgray{} more text."
    expected = "Text  more text."
    assert strip_pvgray_wrappers(text) == expected


def test_validate_grayed_chunk_basic():
    """Test basic validation."""
    original = "This is a test.\nAnother line."
    
    # Valid: only wrappers added
    grayed = "This is a test.\n\\pvgray{Another line.}"
    assert validate_grayed_chunk(original, grayed) == True
    
    # Invalid: text changed
    grayed = "This is a modified test.\nAnother line."
    assert validate_grayed_chunk(original, grayed) == False
    
    # Invalid: text removed
    grayed = "This is a test."
    assert validate_grayed_chunk(original, grayed) == False


def test_validate_grayed_chunk_whitespace():
    """Test that validation allows CRLF normalization but nothing else."""
    original = "Line 1\r\nLine 2"
    
    # CRLF -> LF is OK
    grayed = "Line 1\nLine 2"
    assert validate_grayed_chunk(original, grayed) == True
    
    # But other whitespace changes are NOT OK
    original = "Line 1\nLine 2"
    grayed = "Line 1 Line 2"  # Changed newline to space
    assert validate_grayed_chunk(original, grayed) == False


def test_validate_preserves_latex():
    """Test that LaTeX commands are preserved during validation."""
    original = r"This is \textbf{bold} and \emph{italic}."
    grayed = r"\pvgray{This is \textbf{bold}} and \emph{italic}."
    assert validate_grayed_chunk(original, grayed) == True
    
    # Changing LaTeX should fail
    grayed = r"\pvgray{This is \textit{bold}} and \emph{italic}."
    assert validate_grayed_chunk(original, grayed) == False
