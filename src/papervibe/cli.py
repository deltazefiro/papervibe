"""Command-line interface for PaperVibe."""

import asyncio
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
from papervibe.gray import gray_out_content_parallel, count_chunks
from papervibe.compile import compile_latex, check_latexmk_available, CompileError

app = typer.Typer(help="PaperVibe: Enhance arXiv papers with AI-powered abstract rewrites and smart highlighting")


def _inject_arxiv_command():
    """Inject 'arxiv' command if user provides direct URL (for backward compat)."""
    # If first argument doesn't look like a known command and isn't a flag, assume it's a URL
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ['arxiv', 'main']:
        # Insert 'arxiv' as the subcommand
        sys.argv.insert(1, 'arxiv')


@app.callback(invoke_without_command=True)
def default_command(ctx: typer.Context):
    """Handle default command routing for backward compatibility."""
    if ctx.invoked_subcommand is None:
        # No subcommand specified - default to main command
        # This is handled by the ctx system automatically if we don't intervene
        pass


def _process_arxiv_command(
    url: str,
    out: Optional[Path],
    skip_abstract: bool,
    skip_gray: bool,
    skip_compile: bool,
    gray_ratio: float,
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
            skip_gray=skip_gray,
            skip_compile=skip_compile,
            gray_ratio=gray_ratio,
            concurrency=concurrency,
            dry_run=dry_run,
            llm_timeout=llm_timeout,
            max_chunk_chars=max_chunk_chars,
        ))
    except (ArxivError, LatexError, CompileError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user", err=True)
        raise typer.Exit(code=130)


# Primary command: papervibe arxiv <url>
@app.command("arxiv", help="Process an arXiv paper: download, enhance abstract, gray out less important sentences, and compile PDF.")
def cmd_arxiv(
    url: str = typer.Argument(..., help="arXiv URL or ID (e.g., 2107.03374 or https://arxiv.org/abs/2107.03374)"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    skip_abstract: bool = typer.Option(False, help="Skip abstract rewriting"),
    skip_gray: bool = typer.Option(False, help="Skip sentence graying"),
    skip_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    gray_ratio: float = typer.Option(0.4, help="Target ratio of sentences to gray out"),
    concurrency: int = typer.Option(8, help="Number of concurrent LLM requests"),
    dry_run: bool = typer.Option(False, help="Dry run mode (skip LLM calls)"),
    llm_timeout: float = typer.Option(30.0, help="Timeout per LLM request in seconds"),
    max_chunk_chars: int = typer.Option(1500, help="Max characters per chunk for graying"),
):
    _process_arxiv_command(url, out, skip_abstract, skip_gray, skip_compile, gray_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars)


# Compat alias: papervibe <url> (default command for backward compatibility)
@app.command()
def main(
    url: str = typer.Argument(..., help="arXiv URL or ID (e.g., 2107.03374 or https://arxiv.org/abs/2107.03374)"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    skip_abstract: bool = typer.Option(False, help="Skip abstract rewriting"),
    skip_gray: bool = typer.Option(False, help="Skip sentence graying"),
    skip_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    gray_ratio: float = typer.Option(0.4, help="Target ratio of sentences to gray out"),
    concurrency: int = typer.Option(8, help="Number of concurrent LLM requests"),
    dry_run: bool = typer.Option(False, help="Dry run mode (skip LLM calls)"),
    llm_timeout: float = typer.Option(30.0, help="Timeout per LLM request in seconds"),
    max_chunk_chars: int = typer.Option(1500, help="Max characters per chunk for graying"),
):
    _process_arxiv_command(url, out, skip_abstract, skip_gray, skip_compile, gray_ratio, concurrency, dry_run, llm_timeout, max_chunk_chars)


async def _process_arxiv_paper(
    url: str,
    out: Optional[Path],
    skip_abstract: bool,
    skip_gray: bool,
    skip_compile: bool,
    gray_ratio: float,
    concurrency: int,
    dry_run: bool,
    llm_timeout: float,
    max_chunk_chars: int,
):
    """Internal async function to process an arXiv paper."""
    
    # Step 1: Parse arXiv ID
    typer.echo(f"Parsing arXiv ID from: {url}")
    arxiv_id, version = parse_arxiv_id(url)
    typer.echo(f"   arXiv ID: {arxiv_id}{version or ''}")
    
    # Step 2: Determine output directory
    if out is None:
        out = Path(f"out/{arxiv_id.replace('/', '_')}")
    else:
        out = Path(out)
    
    out.mkdir(parents=True, exist_ok=True)
    typer.echo(f"   Output directory: {out}")
    
    # Step 3: Download source
    typer.echo(f"Downloading arXiv source...")
    source_dir = out / "original"
    source_dir.mkdir(exist_ok=True)
    download_arxiv_source(arxiv_id, version, source_dir)
    typer.echo(f"   Downloaded to: {source_dir}")
    
    # Step 4: Find main .tex file
    typer.echo(f"Finding main .tex file...")
    main_tex = find_main_tex_file(source_dir)
    typer.echo(f"   Main file: {main_tex.name}")
    
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
        typer.echo(f"Rewriting abstract...")
        
        # First, try to find abstract in main file
        abstract_result = extract_abstract(modified_content)
        
        if abstract_result:
            # Abstract found in main file
            abstract_found_in_main = True
            original_abstract, _, _ = abstract_result
            typer.echo(f"   Found abstract in {main_tex.name}: {len(original_abstract)} chars")
            
            new_abstract = await llm_client.rewrite_abstract(original_abstract)
            typer.echo(f"   New abstract: {len(new_abstract)} chars")
            
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
                        typer.echo(f"   Found abstract in {input_file.name}: {len(original_abstract)} chars")
                        
                        new_abstract = await llm_client.rewrite_abstract(original_abstract)
                        typer.echo(f"   New abstract: {len(new_abstract)} chars")
                        
                        # Store the file path and modified content for later
                        abstract_file_path = input_file
                        modified_input_files[input_file] = replace_abstract(input_content, new_abstract)
                        break
                except Exception as e:
                    continue
            
            if abstract_file_path is None and not abstract_found_in_main:
                typer.echo(f"   Warning: No abstract found in main or included files, skipping rewrite")
    
    # Step 8: Inject preamble (xcolor + \pvgray macro)
    typer.echo(f"Injecting preamble...")
    modified_content = inject_preamble(modified_content)
    
    # Step 9: Gray out sentences
    if not skip_gray:
        typer.echo(f"Graying out less important sentences (ratio={gray_ratio})...")
        
        # Find all input files referenced by main.tex
        input_files = find_input_files(modified_content, source_dir)
        
        # Prepare list of files to process (excluding already-processed and abstract-containing files)
        files_to_process = []
        for input_file in input_files:
            # Skip if we already processed this file during abstract rewrite
            if input_file in modified_input_files:
                typer.echo(f"   Skipping {input_file.name} (already processed during abstract rewrite)")
                continue
                
            try:
                input_content = input_file.read_text(encoding="utf-8", errors="ignore")
                
                # Check if this is the abstract file (skip graying)
                if extract_abstract(input_content):
                    typer.echo(f"   Skipping {input_file.name} (contains abstract)")
                    continue
                
                files_to_process.append((input_file, input_content))
            except Exception as e:
                typer.echo(f"   Warning: Failed to read {input_file.name}: {e}")
        
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
        
        typer.echo(f"   Processing {total_chunks} chunks across {len(files_to_process)} input files + main file...")
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Graying chunks", total=total_chunks)
            
            def update_progress(advance: int = 1):
                """Callback to update progress bar."""
                progress.update(task, advance=advance)
            
            # Process all input files in parallel
            async def process_input_file(file_path, content):
                """Process a single input file."""
                try:
                    grayed = await gray_out_content_parallel(
                        content,
                        llm_client,
                        gray_ratio=gray_ratio,
                        max_chunk_chars=max_chunk_chars,
                        progress_callback=update_progress,
                    )
                    return (file_path, grayed, len(content))
                except Exception as e:
                    typer.echo(f"   Warning: Failed to process {file_path.name}: {e}")
                    return (file_path, content, 0)  # Return original on error
            
            # Process all files in parallel
            if files_to_process:
                results = await asyncio.gather(*[
                    process_input_file(fp, content) for fp, content in files_to_process
                ])
                
                total_chars_processed = 0
                for file_path, grayed_content, chars_processed in results:
                    modified_input_files[file_path] = grayed_content
                    total_chars_processed += chars_processed
            else:
                total_chars_processed = 0
            
            # Gray out main file content
            grayed_main_parts = []
            main_chars_processed = 0
            for part in main_parts_to_gray:
                if part.strip():
                    grayed_part = await gray_out_content_parallel(
                        part,
                        llm_client,
                        gray_ratio=gray_ratio,
                        max_chunk_chars=max_chunk_chars,
                        progress_callback=update_progress,
                    )
                    grayed_main_parts.append(grayed_part)
                    main_chars_processed += len(part)
                else:
                    grayed_main_parts.append(part)
        
        # Reconstruct modified_content
        post_refs = modified_content[cutoff:] if cutoff is not None else ""
        
        if abstract_span is not None:
            abs_start, abs_end = abstract_span
            if abs_end <= len(main_content_to_process):
                abstract_region = modified_content[abs_start:abs_end]
                if len(grayed_main_parts) == 2:
                    modified_content = grayed_main_parts[0] + abstract_region + grayed_main_parts[1] + post_refs
                else:
                    modified_content = grayed_main_parts[0] + post_refs
            else:
                if len(grayed_main_parts) > 0:
                    modified_content = grayed_main_parts[0] + post_refs
        else:
            if len(grayed_main_parts) > 0:
                modified_content = grayed_main_parts[0] + post_refs
        
        if modified_input_files:
            typer.echo(f"   Processed {len(modified_input_files)} input files ({total_chars_processed} chars) + main file ({main_chars_processed} chars)")
        elif main_chars_processed > 0:
            typer.echo(f"   Processed main file ({main_chars_processed} chars)")
        
        # Step 9.1: Diagnostic summary
        wrapper_count = modified_content.count(r"\pvgray{")
        for content in modified_input_files.values():
            wrapper_count += content.count(r"\pvgray{")
        
        if dry_run:
            typer.echo(f"   [Dry Run] No \\pvgray{{}} wrappers actually added.")
        elif not skip_gray and gray_ratio > 0:
            if wrapper_count == 0:
                typer.echo(f"   Warning: No graying was applied! (wrapper count: 0)")
                if any(llm_client.stats.values()):
                    typer.echo(f"   LLM Stats: {llm_client.stats}")
                    typer.echo(f"   Hint: Some requests timed out or failed. Try increasing --llm-timeout or check LLM config.")
                else:
                    typer.echo(f"   Hint: The LLM might have decided not to gray out any sentences, or all edits failed validation.")
            else:
                typer.echo(f"   Applied {wrapper_count} \\pvgray{{}} wrappers.")

    # Step 10: Write modified files
    typer.echo(f"Writing modified files...")
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
    if not skip_gray:
        for input_file, grayed_content in modified_input_files.items():
            # Compute relative path and write to modified directory
            rel_path = input_file.relative_to(source_dir)
            output_file = modified_dir / rel_path
            output_file.write_text(grayed_content, encoding="utf-8")
    
    typer.echo(f"   Modified files in: {modified_dir}")
    
    # Step 11: Compile PDF
    if not skip_compile:
        if not check_latexmk_available():
            typer.echo(f"   Warning: latexmk not found, skipping compilation")
        else:
            typer.echo(f"Compiling PDF...")
            pdf_path, log = compile_latex(
                modified_main,
                output_dir=modified_dir,
                timeout=300,
            )
            typer.echo(f"   Success: PDF compiled: {pdf_path}")
            
            # Copy PDF to output root for convenience
            final_pdf = out / f"{arxiv_id.replace('/', '_')}.pdf"
            shutil.copy2(pdf_path, final_pdf)
            typer.echo(f"   Final PDF: {final_pdf}")
    
    typer.echo(f"\nProcessing complete!")
    typer.echo(f"   Original sources: {source_dir}")
    typer.echo(f"   Modified sources: {modified_dir}")
    if not skip_compile and check_latexmk_available():
        final_pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
        typer.echo(f"   Final PDF: {out / final_pdf_name}")


def main_entry():
    """Entry point that handles backward compatibility for direct URL invocation."""
    _inject_arxiv_command()
    app()


if __name__ == "__main__":
    main_entry()
