# AGENTS.md (repo guidance)

## Overview

`papervibe` is a Python (uv-managed) CLI tool that:
1) Downloads an arXiv paper TeX source bundle
2) Rewrites the LaTeX abstract using a "strong" OpenAI model
3) Highlights important keywords and sentences by inserting `\\pvhighlight{...}` wrappers (validated to avoid any other content changes)
4) Compiles the modified sources to PDF using `latexmk`

The design goal is to keep LaTeX diffs minimal and mechanical.

## Repository layout

- `pyproject.toml`: uv project config and CLI entry point
- `src/papervibe/`
  - `cli.py`: Typer CLI and end-to-end pipeline orchestration
  - `arxiv.py`: arXiv ID parsing + source download/unpack
  - `latex.py`: LaTeX helpers (main file detection, references cutoff, abstract span/replace, preamble injection, wrapper stripping)
  - `llm.py`: AsyncOpenAI wrapper + settings (reads `.env`) + concurrency semaphore
  - `highlight.py`: chunking + strict validation of highlight edits
  - `compile.py`: `latexmk` compilation wrapper
- `tests/`: pytest suite + LaTeX fixtures (automated, no LLM calls)
- `harness/`: scripts for testing specific modules with real LLM calls that require manual evaluation

## Key invariants

- Minimal LaTeX edits only:
  - abstract replacement
  - one-time preamble injection (`xcolor` + `\\AtBeginDocument{\\color{gray}}` + `\\pvhighlight` macro)
  - `\\pvhighlight{...}` wrappers around important keywords and sentences
- Color scheme:
  - Default text color: gray (set via `\\AtBeginDocument{\\color{gray}}`)
  - Highlighted content: black (via `\\pvhighlight{...}` wrapper)
- Highlight-stage validation is strict:
  - strip all `\\pvhighlight{...}` wrappers (brace-aware)
  - remaining text must match the original chunk exactly (only CRLF->LF and trailing whitespace normalization allowed)
  - on failure: retry, then fall back to leaving the chunk unchanged
- Abstract is excluded from the highlight stage.
- `.env` contains secrets and must never be committed.

## Common commands

- Run tests: `uv run pytest`
- Show CLI help: `uv run papervibe --help`
- Process a paper (dry run): `uv run papervibe arxiv 2107.03374 --dry-run`
- Process a paper (real): `uv run papervibe arxiv 2107.03374 --highlight-ratio 0.4 --concurrency 8`
- Evaluate highlighting on samples: `uv run python harness/eval_highlight.py`

## Outputs

By default, results are written under `out/<id>/`:
- `original/`: downloaded/unpacked sources
- `modified/`: modified sources used for compilation
- `<id>.pdf`: compiled PDF (if compilation succeeds)
