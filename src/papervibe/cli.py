"""Command-line interface for PaperVibe."""

import asyncio
import shutil
import typer
from pathlib import Path
from typing import Optional

from papervibe.arxiv import parse_arxiv_id, download_arxiv_source, ArxivError
from papervibe.latex import (
    find_main_tex_file,
    find_references_cutoff,
    extract_abstract,
    replace_abstract,
    inject_preamble,
    LatexError,
)
from papervibe.llm import LLMClient
from papervibe.gray import gray_out_content_parallel
from papervibe.compile import compile_latex, check_latexmk_available, CompileError

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
    typer.echo(f"📄 Parsing arXiv ID from: {url}")
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
    typer.echo(f"⬇️  Downloading arXiv source...")
    source_dir = out / "original"
    source_dir.mkdir(exist_ok=True)
    download_arxiv_source(arxiv_id, version, source_dir)
    typer.echo(f"   Downloaded to: {source_dir}")
    
    # Step 4: Find main .tex file
    typer.echo(f"🔍 Finding main .tex file...")
    main_tex = find_main_tex_file(source_dir)
    typer.echo(f"   Main file: {main_tex.name}")
    
    # Step 5: Read original content
    original_content = main_tex.read_text(encoding="utf-8", errors="ignore")
    modified_content = original_content
    
    # Step 6: Initialize LLM client
    llm_client = LLMClient(concurrency=concurrency, dry_run=dry_run)
    
    # Step 7: Rewrite abstract
    if not skip_abstract:
        typer.echo(f"✍️  Rewriting abstract...")
        abstract_result = extract_abstract(modified_content)
        
        if abstract_result:
            original_abstract, _, _ = abstract_result
            typer.echo(f"   Original abstract: {len(original_abstract)} chars")
            
            new_abstract = await llm_client.rewrite_abstract(original_abstract)
            typer.echo(f"   New abstract: {len(new_abstract)} chars")
            
            modified_content = replace_abstract(modified_content, new_abstract)
        else:
            typer.echo(f"   ⚠️  No abstract found, skipping rewrite")
    
    # Step 8: Inject preamble (xcolor + \pvgray macro)
    typer.echo(f"🎨 Injecting preamble...")
    modified_content = inject_preamble(modified_content)
    
    # Step 9: Gray out sentences
    if not skip_gray:
        typer.echo(f"🖍️  Graying out less important sentences (ratio={gray_ratio})...")
        
        # Find references cutoff to exclude references section
        cutoff = find_references_cutoff(modified_content)
        
        if cutoff is not None:
            # Process only pre-references content
            pre_refs = modified_content[:cutoff]
            post_refs = modified_content[cutoff:]
            
            # Gray out pre-references content
            grayed_pre_refs = await gray_out_content_parallel(
                pre_refs,
                llm_client,
                gray_ratio=gray_ratio,
            )
            
            modified_content = grayed_pre_refs + post_refs
            typer.echo(f"   Processed {len(pre_refs)} chars (excluded references)")
        else:
            # No references found, process entire content
            modified_content = await gray_out_content_parallel(
                modified_content,
                llm_client,
                gray_ratio=gray_ratio,
            )
            typer.echo(f"   Processed entire document")
    
    # Step 10: Write modified files
    typer.echo(f"💾 Writing modified files...")
    modified_dir = out / "modified"
    modified_dir.mkdir(exist_ok=True)
    
    # Copy all files from original to modified
    for file in source_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, modified_dir / file.name)
    
    # Overwrite main .tex file with modified content
    modified_main = modified_dir / main_tex.name
    modified_main.write_text(modified_content, encoding="utf-8")
    typer.echo(f"   Modified files in: {modified_dir}")
    
    # Step 11: Compile PDF
    if not skip_compile:
        if not check_latexmk_available():
            typer.echo(f"   ⚠️  latexmk not found, skipping compilation")
        else:
            typer.echo(f"📦 Compiling PDF...")
            pdf_path, log = compile_latex(
                modified_main,
                output_dir=modified_dir,
                timeout=300,
            )
            typer.echo(f"   ✅ PDF compiled: {pdf_path}")
            
            # Copy PDF to output root for convenience
            final_pdf = out / f"{arxiv_id.replace('/', '_')}.pdf"
            shutil.copy2(pdf_path, final_pdf)
            typer.echo(f"   📄 Final PDF: {final_pdf}")
    
    typer.echo(f"\n✅ Processing complete!")
    typer.echo(f"   Original sources: {source_dir}")
    typer.echo(f"   Modified sources: {modified_dir}")
    if not skip_compile and check_latexmk_available():
        final_pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
        typer.echo(f"   Final PDF: {out / final_pdf_name}")


if __name__ == "__main__":
    app()
