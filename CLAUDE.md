# AGENTS.md (repo guidance)

## Overview

`papervibe` is a Python (uv-managed) CLI tool that:
1) Downloads an arXiv paper TeX source bundle
2) Rewrites the LaTeX abstract using a "strong" OpenAI model
3) Highlights important keywords and sentences by inserting `\\pvhighlight{...}` wrappers
4) Compiles the modified sources to PDF using `latexmk`

The design goal is to keep LaTeX diffs minimal and mechanical.

## Repository layout

- `pyproject.toml`: uv project config and CLI entry point
- `src/papervibe/`
  - `cli.py`: Typer CLI and end-to-end pipeline orchestration
  - `arxiv.py`: arXiv ID parsing + source download/unpack
  - `latex.py`: LaTeX helpers (main file detection, references cutoff, abstract span/replace, preamble injection, wrapper stripping)
  - `llm.py`: AsyncOpenAI wrapper + settings (reads `.env`) + concurrency semaphore
  - `highlight.py`: chunking + snippet-based highlighting (parse_snippets, apply_highlights, chunk_content_with_seps)
  - `compile.py`: `latexmk` compilation wrapper
- `tests/`: pytest suite + LaTeX fixtures (automated, no LLM calls)
  - `fixtures/papers.json`: paper IDs for integration tests
  - `fixtures/responses/<id>/`: pre-computed LLM responses (stub abstracts, future highlight snippets)
  - `integration/`: integration tests (layout preservation, require latexmk)
- `harness/`: manual LLM evaluation with side-by-side PDF comparison
  - `samples/`: LaTeX samples (highlight/, abstract/)
  - `eval_highlight.py`, `eval_abstract.py`, `eval_layout.py`: evaluation scripts
  - `out/`: evaluation outputs

## Key invariants

- Minimal LaTeX edits only:
  - abstract replacement (using `\pvreplaceblock{old}{new}` to preserve original layout/spacing)
  - one-time preamble injection (`xcolor` + `\AtBeginDocument{\color{gray}}` + `\pvhighlight` + `\pvreplaceblock` + abstract black override)
  - `\pvhighlight{...}` wrappers around important keywords and sentences
- Color scheme:
  - Default text color: gray (set via `\AtBeginDocument{\color{gray}}`)
  - Highlighted content: black (via `\pvhighlight{...}` wrapper)
  - Abstract: always black (via `\renewenvironment{abstract}`)
- Highlight approach (snippet-based):
  - LLM outputs only the text snippets to highlight, one per line
  - Code searches for each snippet in the original chunk and wraps first match with `\pvhighlight{...}`
  - Unmatched snippets are logged at debug level and skipped
- Abstract is excluded from the highlight stage (but always rendered in black).
- Chunking must preserve original whitespace: use `chunk_content_with_seps()` + `rejoin_chunks()` to avoid corrupting LaTeX tables/environments.
- `\pvreplaceblock{old}{new}` boxes the original content to get its exact height, then creates a vbox of that height with new content at top. This guarantees identical vertical footprint.
- Footnotes in abstracts: `rewrite_abstract()` extracts footnotes from original and appends them to the rewritten abstract to preserve page layout.
- `.env` contains secrets and must never be committed.

## Common commands

- Run tests: `uv run pytest`
- Run unit tests only: `uv run pytest --ignore=tests/integration/`
- Run integration tests only: `uv run pytest tests/integration/ -v`
- Show CLI help: `uv run papervibe --help`
- Process a paper (dry run): `uv run papervibe arxiv 2107.03374 --dry-run`
- Process a paper (real): `uv run papervibe arxiv 2107.03374 --concurrency 8`
- Evaluate highlighting on samples: `uv run python harness/eval_highlight.py`
- Evaluate abstract rewriting: `uv run python harness/eval_abstract.py`
- Evaluate layout preservation: `uv run python harness/eval_layout.py [arxiv_ids...]`

## Outputs

By default, results are written under `out/<id>/`:
- `original/`: downloaded/unpacked sources
- `modified/`: modified sources used for compilation
- `<id>.pdf`: compiled PDF (if compilation succeeds)

## Conventions
- LLM timeout should be at least 120 seconds. Long outputs may need even more.
- Keep CLAUDE.md concise.