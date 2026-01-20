"""Command-line interface for PaperVibe."""

import asyncio
import logging
import shutil
import sys
import typer
from pathlib import Path
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from papervibe.arxiv import parse_arxiv_id, download_arxiv_source, ArxivError
from papervibe.latex import (
    find_main_tex_file,
    find_references_cutoff,
    extract_abstract,
    get_abstract_span,
    replace_abstract,
    inject_preamble,
    find_input_files,
    LatexError,
)
from papervibe.llm import LLMClient
from papervibe.logging import setup_logging, get_console
from papervibe.highlight import highlight_content_parallel, count_chunks
from papervibe.compile import compile_latex, check_latexmk_available, CompileError

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
):
    """Shared implementation for arXiv processing."""
    try:
        asyncio.run(_process_arxiv_paper(
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
):
    _process_arxiv_command(url, out, skip_abstract, skip_highlight, skip_compile, highlight_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars)


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
):
    _process_arxiv_command(url, out, skip_abstract, skip_highlight, skip_compile, highlight_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars)


async def _process_arxiv_paper(
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
):
    """Internal async function to process an arXiv paper."""
    
    # Step 1: Parse arXiv ID
    logger.info("Parsing arXiv ID from: %s", url)
    arxiv_id, version = parse_arxiv_id(url)
    logger.info("  arXiv ID: %s%s", arxiv_id, version or "")
    
    # Step 2: Determine output directory
    if out is None:
        out = Path(f"out/{arxiv_id.replace('/', '_')}")
    else:
        out = Path(out)
    
    out.mkdir(parents=True, exist_ok=True)
    logger.info("  Output directory: %s", out)
    
    # Step 3: Download source
    logger.info("Downloading arXiv source...")
    source_dir = out / "original"
    source_dir.mkdir(exist_ok=True)
    download_arxiv_source(arxiv_id, version, source_dir)
    logger.info("  Downloaded to: %s", source_dir)
    
    # Step 4: Find main .tex file
    logger.info("Finding main .tex file...")
    main_tex = find_main_tex_file(source_dir)
    logger.info("  Main file: %s", main_tex.name)
    
    # Step 5: Read original content
    original_content = main_tex.read_text(encoding="utf-8", errors="ignore")
    modified_content = original_content
    
    # Step 6: Initialize LLM client
    from papervibe.llm import LLMSettings
    settings = LLMSettings()
    settings.request_timeout_seconds = llm_timeout
    llm_client = LLMClient(settings=settings, concurrency=concurrency, dry_run=dry_run)
    
    # Initialize modified input files dictionary (used for abstract rewrite and gray stage)
    modified_input_files = {}
    
    # Step 7: Rewrite abstract
    abstract_file_path = None
    abstract_found_in_main = False
    
    if not skip_abstract:
        logger.info("Rewriting abstract...")
        
        # First, try to find abstract in main file
        abstract_result = extract_abstract(modified_content)
        
        if abstract_result:
            # Abstract found in main file
            abstract_found_in_main = True
            original_abstract, _, _ = abstract_result
            logger.info("  Found abstract in %s: %s chars", main_tex.name, len(original_abstract))
            
            new_abstract = await llm_client.rewrite_abstract(original_abstract)
            logger.info("  New abstract: %s chars", len(new_abstract))
            
            modified_content = replace_abstract(modified_content, new_abstract)
        else:
            # Try to find abstract in included files
            input_files = find_input_files(modified_content, source_dir)
            for input_file in input_files:
                try:
                    input_content = input_file.read_text(encoding="utf-8", errors="ignore")
                    abstract_result = extract_abstract(input_content)
                    
                    if abstract_result:
                        original_abstract, _, _ = abstract_result
                        logger.info("  Found abstract in %s: %s chars", input_file.name, len(original_abstract))
                        
                        new_abstract = await llm_client.rewrite_abstract(original_abstract)
                        logger.info("  New abstract: %s chars", len(new_abstract))
                        
                        # Store the file path and modified content for later
                        abstract_file_path = input_file
                        modified_input_files[input_file] = replace_abstract(input_content, new_abstract)
                        break
                except Exception as e:
                    continue
            
            if abstract_file_path is None and not abstract_found_in_main:
                logger.warning("No abstract found in main or included files, skipping rewrite")
    
    # Step 8: Inject preamble (xcolor + default gray + \pvhighlight macro)
    logger.info("Injecting preamble...")
    modified_content = inject_preamble(modified_content)

    # Step 9: Highlight important content
    if not skip_highlight:
        logger.info("Highlighting important content (ratio=%s)...", highlight_ratio)
        
        # Find all input files referenced by main.tex
        input_files = find_input_files(modified_content, source_dir)
        
        # Prepare list of files to process (excluding already-processed and abstract-containing files)
        files_to_process = []
        for input_file in input_files:
            # Skip if we already processed this file during abstract rewrite
            if input_file in modified_input_files:
                logger.info("  Skipping %s (already processed during abstract rewrite)", input_file.name)
                continue
                
            try:
                input_content = input_file.read_text(encoding="utf-8", errors="ignore")
                
                # Check if this is the abstract file (skip graying)
                if extract_abstract(input_content):
                    logger.info("  Skipping %s (contains abstract)", input_file.name)
                    continue
                
                files_to_process.append((input_file, input_content))
            except Exception as e:
                logger.warning("Failed to read %s: %s", input_file.name, e)
        
        # Count total chunks for progress bar
        total_chunks = 0
        
        # Count chunks from input files
        for _, content in files_to_process:
            total_chunks += count_chunks(content, max_chunk_chars=max_chunk_chars)
        
        # Count chunks from main file
        # Find references cutoff to exclude references section
        cutoff = find_references_cutoff(modified_content)
        
        # Find abstract span to exclude from graying
        abstract_span = get_abstract_span(modified_content)
        
        # Determine what content from main file to process
        main_content_to_process = modified_content
        if cutoff is not None:
            main_content_to_process = modified_content[:cutoff]
        
        # Exclude abstract if present
        if abstract_span is not None:
            abs_start, abs_end = abstract_span
            if abs_end <= len(main_content_to_process):
                # Process content before and after abstract
                before_abstract = main_content_to_process[:abs_start]
                after_abstract = main_content_to_process[abs_end:]
                main_parts_to_gray = [before_abstract, after_abstract]
            else:
                # Abstract extends beyond cutoff, just process up to cutoff
                main_parts_to_gray = [main_content_to_process]
        else:
            main_parts_to_gray = [main_content_to_process]
        
        # Count chunks from main parts
        for part in main_parts_to_gray:
            if part.strip():
                total_chunks += count_chunks(part, max_chunk_chars=max_chunk_chars)
        
        logger.info(
            "Processing %s chunks across %s input files + main file...",
            total_chunks,
            len(files_to_process),
        )

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=False,
            console=get_console(),
        ) as progress:
            task = progress.add_task("Highlighting chunks", total=total_chunks)
            
            def update_progress(advance: int = 1):
                """Callback to update progress bar."""
                progress.update(task, advance=advance)
            
            # Process all input files in parallel
            async def process_input_file(file_path, content):
                """Process a single input file."""
                try:
                    highlighted = await highlight_content_parallel(
                        content,
                        llm_client,
                        highlight_ratio=highlight_ratio,
                        max_chunk_chars=max_chunk_chars,
                        progress_callback=update_progress,
                    )
                    return (file_path, highlighted, len(content))
                except Exception as e:
                    logger.warning("Failed to process %s: %s", file_path.name, e)
                    return (file_path, content, 0)  # Return original on error

            # Process all files in parallel
            if files_to_process:
                results = await asyncio.gather(*[
                    process_input_file(fp, content) for fp, content in files_to_process
                ])

                total_chars_processed = 0
                for file_path, highlighted_content, chars_processed in results:
                    modified_input_files[file_path] = highlighted_content
                    total_chars_processed += chars_processed
            else:
                total_chars_processed = 0

            # Highlight main file content
            highlighted_main_parts = []
            main_chars_processed = 0
            for part in main_parts_to_gray:
                if part.strip():
                    highlighted_part = await highlight_content_parallel(
                        part,
                        llm_client,
                        highlight_ratio=highlight_ratio,
                        max_chunk_chars=max_chunk_chars,
                        progress_callback=update_progress,
                    )
                    highlighted_main_parts.append(highlighted_part)
                    main_chars_processed += len(part)
                else:
                    highlighted_main_parts.append(part)
        
        # Reconstruct modified_content
        post_refs = modified_content[cutoff:] if cutoff is not None else ""

        if abstract_span is not None:
            abs_start, abs_end = abstract_span
            if abs_end <= len(main_content_to_process):
                abstract_region = modified_content[abs_start:abs_end]
                if len(highlighted_main_parts) == 2:
                    modified_content = highlighted_main_parts[0] + abstract_region + highlighted_main_parts[1] + post_refs
                else:
                    modified_content = highlighted_main_parts[0] + post_refs
            else:
                if len(highlighted_main_parts) > 0:
                    modified_content = highlighted_main_parts[0] + post_refs
        else:
            if len(highlighted_main_parts) > 0:
                modified_content = highlighted_main_parts[0] + post_refs
        
        if modified_input_files:
            logger.info(
                "Processed %s input files (%s chars) + main file (%s chars)",
                len(modified_input_files),
                total_chars_processed,
                main_chars_processed,
            )
        elif main_chars_processed > 0:
            logger.info("Processed main file (%s chars)", main_chars_processed)

        # Step 9.1: Diagnostic summary
        wrapper_count = modified_content.count(r"\pvhighlight{")
        for content in modified_input_files.values():
            wrapper_count += content.count(r"\pvhighlight{")

        if dry_run:
            logger.info("[Dry Run] No \\pvhighlight{} wrappers actually added.")
        elif not skip_highlight and highlight_ratio > 0:
            if wrapper_count == 0:
                logger.warning("No highlighting was applied (wrapper count: 0)")
                if any(llm_client.stats.values()):
                    logger.warning("LLM stats: %s", llm_client.stats)
                    logger.info(
                        "Hint: Some requests timed out or failed. Try increasing --llm-timeout or check LLM config."
                    )
                else:
                    logger.info(
                        "Hint: The LLM might have decided not to highlight any content, or all edits failed validation."
                    )
            else:
                logger.info("Applied %s \\pvhighlight{} wrappers.", wrapper_count)

    # Step 10: Write modified files
    logger.info("Writing modified files...")
    modified_dir = out / "modified"
    
    # Remove existing modified directory if it exists to ensure clean state
    if modified_dir.exists():
        shutil.rmtree(modified_dir)
    
    # Recursively copy all files from original to modified
    shutil.copytree(source_dir, modified_dir)
    
    # Overwrite main .tex file with modified content
    modified_main = modified_dir / main_tex.name
    modified_main.write_text(modified_content, encoding="utf-8")
    
    # Overwrite modified input files
    if not skip_highlight:
        for input_file, highlighted_content in modified_input_files.items():
            # Compute relative path and write to modified directory
            rel_path = input_file.relative_to(source_dir)
            output_file = modified_dir / rel_path
            output_file.write_text(highlighted_content, encoding="utf-8")
    
    logger.info("Modified files in: %s", modified_dir)
    
    # Step 11: Compile PDF
    if not skip_compile:
        if not check_latexmk_available():
            logger.warning("latexmk not found, skipping compilation")
        else:
            logger.info("Compiling PDF...")
            pdf_path, log = compile_latex(
                modified_main,
                output_dir=modified_dir,
                timeout=300,
            )
            logger.info("PDF compiled: %s", pdf_path)
            
            # Copy PDF to output root for convenience
            final_pdf = out / f"{arxiv_id.replace('/', '_')}.pdf"
            shutil.copy2(pdf_path, final_pdf)
            logger.info("Final PDF: %s", final_pdf)

    logger.info("Processing complete!")
    logger.info("Original sources: %s", source_dir)
    logger.info("Modified sources: %s", modified_dir)
    if not skip_compile and check_latexmk_available():
        final_pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
        logger.info("Final PDF: %s", out / final_pdf_name)


def main_entry():
    """Entry point that handles backward compatibility for direct URL invocation."""
    _inject_arxiv_command()
    app()


if __name__ == "__main__":
    main_entry()
