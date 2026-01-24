"""Paper processing pipeline and domain logic."""

import asyncio
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .arxiv import parse_arxiv_id, download_arxiv_source
from .latex import (
    find_main_tex_file,
    find_references_cutoff,
    extract_abstract,
    get_abstract_span,
    replace_abstract,
    inject_preamble,
    find_input_files,
)
from .llm import LLMClient
from .logging import get_console
from .compile import compile_latex, check_latexmk_available

logger = logging.getLogger(__name__)


def extract_footnotes(text: str) -> Tuple[str, List[str]]:
    """
    Extract \\footnote{...} commands from text.

    Args:
        text: LaTeX text potentially containing footnotes

    Returns:
        Tuple of (text_without_footnotes, list_of_footnote_commands)
    """
    footnotes = []
    result = []
    i = 0

    while i < len(text):
        # Look for \footnote{
        if text[i:i+10] == "\\footnote{":
            # Find the matching closing brace
            start = i
            i += 10
            brace_level = 1

            while i < len(text) and brace_level > 0:
                if text[i] == "{" and (i == 0 or text[i-1] != "\\"):
                    brace_level += 1
                elif text[i] == "}" and (i == 0 or text[i-1] != "\\"):
                    brace_level -= 1
                i += 1

            # Extract the full footnote command
            footnotes.append(text[start:i])
        else:
            result.append(text[i])
            i += 1

    return "".join(result), footnotes


async def rewrite_abstract(
    llm_client: LLMClient,
    original_abstract: str,
) -> str:
    """
    Rewrite an abstract to be more clear and engaging.

    Footnotes from the original abstract are preserved and appended to the
    rewritten abstract to maintain page layout (footnotes affect page bottom space).

    Args:
        llm_client: LLM client for processing
        original_abstract: Original abstract text

    Returns:
        Rewritten abstract text with original footnotes preserved

    Raises:
        Exception: On non-retryable errors
    """
    from .prompts import get_renderer
    from pydantic import BaseModel, Field

    class RewrittenAbstract(BaseModel):
        """Structured output for abstract rewriting."""
        abstract: str = Field(description="The rewritten abstract text")
        reasoning: Optional[str] = Field(default=None, description="Brief explanation of changes made")

    # Handle dry-run mode early
    if llm_client.dry_run:
        return original_abstract

    # Extract footnotes to preserve them (they affect page layout)
    abstract_without_footnotes, footnotes = extract_footnotes(original_abstract)

    renderer = get_renderer()
    system_prompt = renderer.render_rewrite_abstract_system()
    # Send abstract without footnotes to LLM (cleaner input)
    user_prompt = renderer.render_rewrite_abstract_user(abstract_without_footnotes)

    try:
        result = await llm_client.complete_structured(
            model_type="strong",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RewrittenAbstract,
            temperature=0.7,
        )
        rewritten = result.abstract
        # Append original footnotes to preserve page layout
        if footnotes:
            rewritten = rewritten.rstrip() + " " + " ".join(footnotes)
        return rewritten
    except asyncio.TimeoutError:
        llm_client.stats["abstract_timeouts"] += 1
        logger.warning(
            "Abstract rewrite timed out after %ss, using original",
            llm_client.settings.request_timeout_seconds,
        )
        return original_abstract
    except Exception as e:
        llm_client.stats["abstract_errors"] += 1
        from .llm import print_error
        print_error(e, context="Failed to rewrite abstract")
        raise


async def highlight_chunk(
    llm_client: LLMClient,
    chunk: str,
    highlight_ratio: float = 0.4,
) -> str:
    """
    Highlight important keywords and sentences in a text chunk.

    The LLM outputs a list of snippets to highlight (one per line),
    then we search for each snippet in the original text and wrap it.

    Args:
        llm_client: LLM client for processing
        chunk: Text chunk to process
        highlight_ratio: Target ratio of content to highlight (0.0 to 1.0)

    Returns:
        Chunk with \\pvhighlight{} wrappers around important content

    Raises:
        Exception: If LLM call fails
    """
    from .prompts import get_renderer
    from .highlight import parse_snippets, apply_highlights

    # Handle dry-run mode early
    if llm_client.dry_run:
        return chunk

    renderer = get_renderer()
    system_prompt = renderer.render_highlight_system()
    user_prompt = renderer.render_highlight_user(chunk, highlight_ratio)

    try:
        llm_output = await llm_client.complete(
            model_type="light",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=4000,  # Snippets list is much shorter than full chunk
        )

        # Parse snippets from LLM output
        snippets = parse_snippets(llm_output)

        # Apply highlights to original chunk
        result, matched, unmatched = apply_highlights(chunk, snippets)

        logger.debug(
            "Chunk highlighting: %d snippets, %d matched, %d unmatched",
            len(snippets), matched, unmatched
        )

        return result
    except asyncio.TimeoutError:
        llm_client.stats["highlight_timeouts"] += 1
        return chunk
    except Exception as e:
        # Check for token limit errors (these are non-retryable)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['token', 'length', 'context_length', 'too long']):
            raise Exception(f"Token limit exceeded: chunk is too large ({len(chunk)} chars)")
        llm_client.stats["highlight_errors"] += 1
        from .llm import print_error
        print_error(e, context=f"Failed to highlight chunk ({len(chunk)} chars)")
        raise


async def highlight_chunks_parallel(
    llm_client: LLMClient,
    chunks: List[str],
    highlight_ratio: float = 0.4,
) -> List[str]:
    """
    Highlight multiple chunks in parallel with concurrency control.

    Args:
        llm_client: LLM client for processing
        chunks: List of text chunks to process
        highlight_ratio: Target ratio of content to highlight

    Returns:
        List of processed chunks with highlighting applied (or original on error)
    """
    from .llm import print_error

    async def process_chunk_safe(chunk: str, index: int) -> str:
        """Process a single chunk with error handling."""
        try:
            return await highlight_chunk(llm_client, chunk, highlight_ratio)
        except Exception as e:
            llm_client.stats["highlight_errors"] += 1
            print_error(e, context=f"Chunk {index + 1}/{len(chunks)} failed")
            return chunk

    tasks = [process_chunk_safe(chunk, i) for i, chunk in enumerate(chunks)]
    return await asyncio.gather(*tasks)


def count_chunks(content: str, max_chunk_chars: int = 1500) -> int:
    """
    Count the number of chunks that will be created from content.

    Args:
        content: LaTeX content to count chunks for
        max_chunk_chars: Maximum characters per chunk

    Returns:
        Number of chunks that will be created
    """
    from .highlight import chunk_content
    chunks = chunk_content(content, max_chunk_size=max_chunk_chars)
    return len(chunks)


async def highlight_content_parallel(
    content: str,
    llm_client: LLMClient,
    highlight_ratio: float = 0.4,
    max_retries: int = 2,
    max_chunk_chars: int = 1500,
    progress_callback: Optional[Callable[[int], None]] = None,
    validate: bool = False,
) -> str:
    """
    Highlight content with parallel chunk processing.

    Args:
        content: LaTeX content to process
        llm_client: LLM client for processing
        highlight_ratio: Target ratio of content to highlight
        max_retries: (unused, kept for API compatibility)
        max_chunk_chars: Maximum characters per chunk
        progress_callback: Optional callback to call after each chunk is processed
        validate: (unused, kept for API compatibility)

    Returns:
        Content with \\pvhighlight{} wrappers applied
    """
    from .highlight import chunk_content_with_seps, rejoin_chunks
    from .llm import print_error

    # Split into chunks with separators preserved
    chunks_with_seps = chunk_content_with_seps(content, max_chunk_size=max_chunk_chars)

    if not chunks_with_seps:
        return content

    # Extract just the chunks for processing
    chunks = [chunk for chunk, _ in chunks_with_seps]

    # Process all chunks in parallel
    try:
        highlighted_chunks = await highlight_chunks_parallel(llm_client, chunks, highlight_ratio)
    except Exception as e:
        print_error(e, context="Parallel highlight processing failed entirely")
        return content

    # Call progress callback for each chunk processed
    if progress_callback:
        for _ in highlighted_chunks:
            progress_callback(1)

    # Rejoin with original separators
    highlighted_with_seps = [
        (highlighted, sep)
        for highlighted, (_, sep) in zip(highlighted_chunks, chunks_with_seps)
    ]
    return rejoin_chunks(highlighted_with_seps)


async def process_paper(
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
    """
    Process an arXiv paper: download, rewrite abstract, highlight content, compile PDF.

    Args:
        url: arXiv URL or ID
        out: Output directory (None for default)
        skip_abstract: Skip abstract rewriting
        skip_highlight: Skip content highlighting
        skip_compile: Skip PDF compilation
        highlight_ratio: Target ratio of content to highlight
        concurrency: Number of concurrent LLM requests
        dry_run: Dry run mode (skip LLM calls)
        llm_timeout: Timeout per LLM request in seconds
        max_chunk_chars: Max characters per chunk for highlighting
        validate_chunks: Validate highlighted chunks match originals
    """
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
    from .llm import LLMSettings
    settings = LLMSettings()
    settings.request_timeout_seconds = llm_timeout
    llm_client = LLMClient(settings=settings, concurrency=concurrency, dry_run=dry_run)

    # Initialize modified input files dictionary
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

            new_abstract = await rewrite_abstract(llm_client, original_abstract)
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

                        new_abstract = await rewrite_abstract(llm_client, original_abstract)
                        logger.info("  New abstract: %s chars", len(new_abstract))

                        # Store the file path and modified content for later
                        abstract_file_path = input_file
                        modified_input_files[input_file] = replace_abstract(input_content, new_abstract)
                        break
                except Exception:
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

        # Prepare list of files to process
        files_to_process = []
        for input_file in input_files:
            # Skip if already processed during abstract rewrite
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
        cutoff = find_references_cutoff(modified_content)
        abstract_span = get_abstract_span(modified_content)

        # Determine what content from main file to process
        main_content_to_process = modified_content
        if cutoff is not None:
            main_content_to_process = modified_content[:cutoff]

        # Exclude abstract if present
        if abstract_span is not None:
            abs_start, abs_end = abstract_span
            if abs_end <= len(main_content_to_process):
                before_abstract = main_content_to_process[:abs_start]
                after_abstract = main_content_to_process[abs_end:]
                main_parts_to_gray = [before_abstract, after_abstract]
            else:
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
                        validate=validate_chunks,
                    )
                    return (file_path, highlighted, len(content))
                except Exception as e:
                    logger.warning("Failed to process %s: %s", file_path.name, e)
                    return (file_path, content, 0)

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
                        validate=validate_chunks,
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

        # Diagnostic summary
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

    # Remove existing modified directory if it exists
    if modified_dir.exists():
        shutil.rmtree(modified_dir)

    # Recursively copy all files from original to modified
    shutil.copytree(source_dir, modified_dir)

    # Overwrite main .tex file with modified content
    modified_main = modified_dir / main_tex.name
    modified_main.write_text(modified_content, encoding="utf-8")

    # Overwrite modified input files (abstract rewrites and/or highlights)
    for input_file, content in modified_input_files.items():
        rel_path = input_file.relative_to(source_dir)
        output_file = modified_dir / rel_path
        output_file.write_text(content, encoding="utf-8")

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
