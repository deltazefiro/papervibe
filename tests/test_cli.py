"""Tests for CLI module."""

from papervibe.cli import app


def test_cli_import():
    """Test that CLI app can be imported."""
    assert app is not None
    assert hasattr(app, "command")
