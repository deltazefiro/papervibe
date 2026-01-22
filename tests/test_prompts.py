"""Tests for Jinja-based prompt templates."""

from papervibe.prompts import get_renderer


def test_highlight_system_contains_pvhighlight_rule():
    renderer = get_renderer()
    system = renderer.render_highlight_system()
    # New snippet-based approach: prompt asks for snippets to highlight, one per line
    assert "one per line" in system.lower()
    assert "snippet" in system.lower()


def test_highlight_user_renders_chunk_and_percent():
    renderer = get_renderer()
    user = renderer.render_highlight_user(chunk="XYZ", highlight_ratio=0.33)
    assert "XYZ" in user
    # Template uses highlight_ratio_percent=highlight_ratio*100; for 0.33 expect 33.0
    assert "33" in user


def test_rewrite_abstract_user_includes_abstract():
    renderer = get_renderer()
    user = renderer.render_rewrite_abstract_user(original_abstract="ABC")
    assert "ABC" in user

