"""Command-line interface for PaperVibe."""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(help="PaperVibe: Enhance arXiv papers with AI-powered abstract rewrites and smart highlighting")


@app.command()
def arxiv(
    url: str = typer.Argument(..., help="arXiv URL or ID (e.g., 2107.03374 or https://arxiv.org/abs/2107.03374)"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    skip_abstract: bool = typer.Option(False, help="Skip abstract rewriting"),
    skip_gray: bool = typer.Option(False, help="Skip sentence graying"),
    skip_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    gray_ratio: float = typer.Option(0.4, help="Target ratio of sentences to gray out"),
    concurrency: int = typer.Option(8, help="Number of concurrent LLM requests"),
    dry_run: bool = typer.Option(False, help="Dry run mode (skip LLM calls)"),
):
    """Process an arXiv paper: download, enhance abstract, gray out less important sentences, and compile PDF."""
    typer.echo(f"Processing arXiv paper: {url}")
    typer.echo(f"Output directory: {out or 'auto'}")
    typer.echo(f"Options: skip_abstract={skip_abstract}, skip_gray={skip_gray}, skip_compile={skip_compile}")
    typer.echo(f"Gray ratio: {gray_ratio}, Concurrency: {concurrency}, Dry run: {dry_run}")
    # Implementation will be added in later milestones
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
