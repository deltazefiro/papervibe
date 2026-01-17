# PaperVibe

Python tool that downloads arXiv papers, rewrites abstracts with AI, and highlights less important content by graying it out.

## Features

- **arXiv Integration**: Download LaTeX source from arXiv using URL or ID
- **Abstract Rewriting**: Uses GPT-4o to improve abstract clarity and engagement
- **Smart Highlighting**: Grays out less important sentences using GPT-4o-mini
- **LaTeX Compilation**: Automatically compiles modified papers to PDF
- **Minimal Diffs**: Only modifies abstract, adds one macro, and wraps sentences
- **Validation**: Ensures grayed text matches original when wrappers are removed

## Requirements

- Python 3.11+
- TeX Live with latexmk (for PDF compilation)
- OpenAI API key

## Installation

```bash
# Clone repository
cd papervibe

# Install with uv
uv sync

# Or install in development mode
uv pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # optional, for custom endpoints
```

## Usage

### Basic Usage

```bash
# Process a paper by arXiv ID
uv run papervibe 2107.03374

# Process using arXiv URL
uv run papervibe https://arxiv.org/abs/2107.03374

# Specify output directory
uv run papervibe 2107.03374 --out output/my-paper
```

### Options

```bash
# Skip abstract rewriting
uv run papervibe 2107.03374 --skip-abstract

# Skip graying out sentences
uv run papervibe 2107.03374 --skip-gray

# Skip PDF compilation
uv run papervibe 2107.03374 --skip-compile

# Adjust gray ratio (0.0 to 1.0)
uv run papervibe 2107.03374 --gray-ratio 0.3

# Control concurrency for LLM requests
uv run papervibe 2107.03374 --concurrency 4

# Dry run (no actual LLM calls)
uv run papervibe 2107.03374 --dry-run
```

## Output Structure

```
out/2107.03374/
├── original/          # Original downloaded sources
│   ├── main.tex
│   └── ...
├── modified/          # Modified sources
│   ├── main.tex       # With rewritten abstract and grayed sentences
│   ├── main.pdf       # Compiled PDF
│   └── ...
└── 2107.03374.pdf    # Copy of final PDF
```

## How It Works

1. **Download**: Fetches LaTeX source from arXiv
2. **Find Main File**: Identifies the main .tex file using heuristics
3. **Rewrite Abstract**: Uses GPT-4o to rewrite the abstract
4. **Inject Preamble**: Adds `xcolor` package and `\pvgray{}` macro
5. **Gray Out**: Processes content in chunks, marking less important sentences
6. **Validate**: Ensures modifications preserve original content
7. **Compile**: Builds PDF using latexmk

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_arxiv.py -v
```

## Development

### Project Structure

```
papervibe/
├── src/papervibe/
│   ├── cli.py         # Command-line interface
│   ├── arxiv.py       # arXiv downloading
│   ├── latex.py       # LaTeX processing
│   ├── llm.py         # OpenAI integration
│   ├── gray.py        # Graying pipeline
│   └── compile.py     # LaTeX compilation
├── tests/             # Test suite
│   ├── fixtures/      # Test LaTeX files
│   └── test_*.py      # Test modules
├── pyproject.toml     # Project configuration
└── .env               # API keys (not committed)
```

### Running Tests

All tests verify the acceptance criteria:
- URL parsing covers new-style, old-style, and versioned IDs
- LaTeX processing validates abstract replacement and macro injection
- Wrapper validation ensures content preservation
- Compilation tests check latexmk integration

## Constraints

- **Minimal LaTeX diffs**: Only abstract, preamble macro, and sentence wrappers
- **Content preservation**: Validation ensures no content loss
- **Abstract exclusion**: Gray stage never processes abstract
- **References exclusion**: Content after bibliography markers is skipped

## License

MIT
