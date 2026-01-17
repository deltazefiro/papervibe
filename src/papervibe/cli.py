"""Command-line interface for PaperVibe."""

import asyncio
import shutil
import sys
import typer
from pathlib import Path
from typing import Optional

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
from papervibe.gray import gray_out_content_parallel
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
):
    _process_arxiv_command(url, out, skip_abstract, skip_gray, skip_compile, gray_ratio, concurrency, dry_run)


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
):
    _process_arxiv_command(url, out, skip_abstract, skip_gray, skip_compile, gray_ratio, concurrency, dry_run)


async def _process_arxiv_paper(
    url: str,
    out: Optional[Path],
    skip_abstract: bool,
    skip_gray: bool,
    skip_compile: bool,
    gray_ratio: float,
    concurrency: int,
    dry_run: bool,
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
    llm_client = LLMClient(concurrency=concurrency, dry_run=dry_run)
    
    # Step 7: Rewrite abstract
    if not skip_abstract:
        typer.echo(f"Rewriting abstract...")
        abstract_result = extract_abstract(modified_content)
        
        if abstract_result:
            original_abstract, _, _ = abstract_result
            typer.echo(f"   Original abstract: {len(original_abstract)} chars")
            
            new_abstract = await llm_client.rewrite_abstract(original_abstract)
            typer.echo(f"   New abstract: {len(new_abstract)} chars")
            
            modified_content = replace_abstract(modified_content, new_abstract)
        else:
            typer.echo(f"   Warning: No abstract found, skipping rewrite")
    
    # Step 8: Inject preamble (xcolor + \pvgray macro)
    typer.echo(f"Injecting preamble...")
    modified_content = inject_preamble(modified_content)
    
    # Initialize modified input files dictionary (used later in Step 10)
    modified_input_files = {}
    
    # Step 9: Gray out sentences
    if not skip_gray:
        typer.echo(f"Graying out less important sentences (ratio={gray_ratio})...")
        
        # Find all input files referenced by main.tex
        input_files = find_input_files(modified_content, source_dir)
        
        # Process each input file separately
        total_chars_processed = 0
        for input_file in input_files:
            try:
                input_content = input_file.read_text(encoding="utf-8", errors="ignore")
                
                # Check if this is the abstract file (skip graying)
                if extract_abstract(input_content):
                    typer.echo(f"   Skipping {input_file.name} (contains abstract)")
                    continue
                
                # Gray out this file's content
                grayed_input = await gray_out_content_parallel(
                    input_content,
                    llm_client,
                    gray_ratio=gray_ratio,
                )
                
                modified_input_files[input_file] = grayed_input
                total_chars_processed += len(input_content)
                
            except Exception as e:
                typer.echo(f"   Warning: Failed to process {input_file.name}: {e}")
        
        if modified_input_files:
            typer.echo(f"   Processed {len(modified_input_files)} input files ({total_chars_processed} chars)")
        else:
            # Fallback to old behavior if no input files found
            # Find references cutoff to exclude references section
            cutoff = find_references_cutoff(modified_content)
            
            # Find abstract span to exclude from graying
            abstract_span = get_abstract_span(modified_content)
            
            if cutoff is not None:
                # Process only pre-references content
                pre_refs = modified_content[:cutoff]
                post_refs = modified_content[cutoff:]
                
                # Exclude abstract from graying
                if abstract_span is not None:
                    abs_start, abs_end = abstract_span
                    if abs_end <= cutoff:
                        # Abstract is before references, split content into 3 parts
                        before_abstract = pre_refs[:abs_start]
                        abstract_region = pre_refs[abs_start:abs_end]
                        after_abstract = pre_refs[abs_end:]
                        
                        # Gray out only before and after abstract
                        grayed_before = await gray_out_content_parallel(
                            before_abstract,
                            llm_client,
                            gray_ratio=gray_ratio,
                        ) if before_abstract.strip() else before_abstract
                        
                        grayed_after = await gray_out_content_parallel(
                            after_abstract,
                            llm_client,
                            gray_ratio=gray_ratio,
                        ) if after_abstract.strip() else after_abstract
                        
                        modified_content = grayed_before + abstract_region + grayed_after + post_refs
                        typer.echo(f"   Processed {len(before_abstract) + len(after_abstract)} chars (excluded abstract and references)")
                    else:
                        # Abstract is after references (unlikely), just exclude references
                        grayed_pre_refs = await gray_out_content_parallel(
                            pre_refs,
                            llm_client,
                            gray_ratio=gray_ratio,
                        )
                        modified_content = grayed_pre_refs + post_refs
                        typer.echo(f"   Processed {len(pre_refs)} chars (excluded references)")
                else:
                    # No abstract found, just exclude references
                    grayed_pre_refs = await gray_out_content_parallel(
                        pre_refs,
                        llm_client,
                        gray_ratio=gray_ratio,
                    )
                    modified_content = grayed_pre_refs + post_refs
                    typer.echo(f"   Processed {len(pre_refs)} chars (excluded references)")
            else:
                # No references found
                if abstract_span is not None:
                    abs_start, abs_end = abstract_span
                    before_abstract = modified_content[:abs_start]
                    abstract_region = modified_content[abs_start:abs_end]
                    after_abstract = modified_content[abs_end:]
                    
                    # Gray out only before and after abstract
                    grayed_before = await gray_out_content_parallel(
                        before_abstract,
                        llm_client,
                        gray_ratio=gray_ratio,
                    ) if before_abstract.strip() else before_abstract
                    
                    grayed_after = await gray_out_content_parallel(
                        after_abstract,
                        llm_client,
                        gray_ratio=gray_ratio,
                    ) if after_abstract.strip() else after_abstract
                    
                    modified_content = grayed_before + abstract_region + grayed_after
                    typer.echo(f"   Processed entire document (excluded abstract)")
                else:
                    # No abstract found, process entire content
                    modified_content = await gray_out_content_parallel(
                        modified_content,
                        llm_client,
                        gray_ratio=gray_ratio,
                    )
                    typer.echo(f"   Processed entire document")
    
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
