"""Command-line interface for PaperVibe."""

import asyncio
import logging
import sys
import typer
from pathlib import Path
from typing import Optional

from papervibe.arxiv import ArxivError
from papervibe.latex import LatexError
from papervibe.logging import setup_logging
from papervibe.compile import CompileError
from papervibe.process import process_paper

app = typer.Typer(help="PaperVibe: Enhance arXiv papers with AI-powered abstract rewrites and smart highlighting")
logger = logging.getLogger(__name__)


def _inject_arxiv_command():
    """Inject 'arxiv' command if user provides direct URL (for backward compat)."""
    # If first argument doesn't look like a known command and isn't a flag, assume it's a URL
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ['arxiv', 'main']:
        # Insert 'arxiv' as the subcommand
        sys.argv.insert(1, 'arxiv')


@app.callback(invoke_without_command=True)
def default_command(
    ctx: typer.Context,
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase logging verbosity (repeatable)"),
    quiet: int = typer.Option(0, "--quiet", "-q", count=True, help="Decrease logging verbosity (repeatable)"),
    log_level: Optional[str] = typer.Option(None, help="Set log level (debug, info, warning, error, critical)"),
    log_file: Optional[Path] = typer.Option(None, help="Write full logs to a file"),
):
    """Handle default command routing for backward compatibility."""
    setup_logging(verbose=verbose, quiet=quiet, log_level=log_level, log_file=log_file)
    if ctx.invoked_subcommand is None:
        # No subcommand specified - default to main command
        # This is handled by the ctx system automatically if we don't intervene
        pass


def _process_arxiv_command(
    url: str,
    out: Optional[Path],
    skip_abstract: bool,
    skip_highlight: bool,
    skip_compile: bool,
    highlight_ratio: float,
    concurrency: int,
    dry_run: bool,
    llm_timeout: float,
    max_chunk_chars: int,
    validate_chunks: bool,
):
    """Shared implementation for arXiv processing."""
    try:
        asyncio.run(process_paper(
            url=url,
            out=out,
            skip_abstract=skip_abstract,
            skip_highlight=skip_highlight,
            skip_compile=skip_compile,
            highlight_ratio=highlight_ratio,
            concurrency=concurrency,
            dry_run=dry_run,
            llm_timeout=llm_timeout,
            max_chunk_chars=max_chunk_chars,
            validate_chunks=validate_chunks,
        ))
    except (ArxivError, LatexError, CompileError) as e:
        logger.error("Error: %s", e)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise typer.Exit(code=130)


# Primary command: papervibe arxiv <url>
@app.command("arxiv", help="Process arXiv paper: download, rewrite abstract, highlight important content, compile PDF")
def cmd_arxiv(
    url: str = typer.Argument(..., help="arXiv URL or ID (e.g., 2107.03374 or https://arxiv.org/abs/2107.03374)"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    skip_abstract: bool = typer.Option(False, help="Skip abstract rewriting"),
    skip_highlight: bool = typer.Option(False, help="Skip content highlighting"),
    skip_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    highlight_ratio: float = typer.Option(0.4, help="Target ratio of content to highlight"),
    concurrency: int = typer.Option(8, help="Number of concurrent LLM requests"),
    dry_run: bool = typer.Option(False, help="Dry run mode (skip LLM calls)"),
    llm_timeout: float = typer.Option(30.0, help="Timeout per LLM request in seconds"),
    max_chunk_chars: int = typer.Option(1500, help="Max characters per chunk for highlighting"),
    validate_chunks: bool = typer.Option(False, help="Validate highlighted chunks match originals"),
):
    _process_arxiv_command(url, out, skip_abstract, skip_highlight, skip_compile, highlight_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars, validate_chunks)


# Compat alias: papervibe <url> (default command for backward compatibility)
@app.command()
def main(
    url: str = typer.Argument(..., help="arXiv URL or ID (e.g., 2107.03374 or https://arxiv.org/abs/2107.03374)"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    skip_abstract: bool = typer.Option(False, help="Skip abstract rewriting"),
    skip_highlight: bool = typer.Option(False, help="Skip content highlighting"),
    skip_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    highlight_ratio: float = typer.Option(0.4, help="Target ratio of content to highlight"),
    concurrency: int = typer.Option(8, help="Number of concurrent LLM requests"),
    dry_run: bool = typer.Option(False, help="Dry run mode (skip LLM calls)"),
    llm_timeout: float = typer.Option(30.0, help="Timeout per LLM request in seconds"),
    max_chunk_chars: int = typer.Option(1500, help="Max characters per chunk for highlighting"),
    validate_chunks: bool = typer.Option(False, help="Validate highlighted chunks match originals"),
):
    _process_arxiv_command(url, out, skip_abstract, skip_highlight, skip_compile, highlight_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars, validate_chunks)


def main_entry():
    """Entry point that handles backward compatibility for direct URL invocation."""
    _inject_arxiv_command()
    app()


if __name__ == "__main__":
    main_entry()
