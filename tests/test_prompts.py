"""Tests for Jinja-based prompt templates."""

from papervibe.prompts import get_renderer


def test_gray_out_system_contains_pvgray_rule():
    renderer = get_renderer()
    system = renderer.render_gray_out_system()
    assert "\\pvgray{...}" in system


def test_gray_out_user_renders_chunk_and_percent():
    renderer = get_renderer()
    user = renderer.render_gray_out_user(chunk="XYZ", gray_ratio=0.42)
    assert "XYZ" in user
    # Template uses gray_ratio_percent=gray_ratio*100; for 0.42 expect 42.0
    assert "42" in user


def test_highlight_system_contains_pvhighlight_rule():
    renderer = get_renderer()
    system = renderer.render_highlight_system()
    assert "\\pvhighlight{...}" in system


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

