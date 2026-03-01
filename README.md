# Models Tool

A command-line tool for fetching and displaying AI model information from the models.dev API.

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd models
   ```

2. Install the package in development mode:
   ```bash
   uv tool install -e .
   ```

## Usage

### Running the tool

After installation, you can run the tool using:

```bash
uv tool run models
```

You can also run it directly if it's on your PATH:

```bash
models
```

### Command-line options

```
uv tool run models [COMMAND] [OPTIONS]

Options:
  -c, --column TEXT     Columns to display (can be specified multiple times)
  -f, --filter TEXT     Filter conditions (e.g., "provider=openai")
  -l, --limit INTEGER   Maximum number of results to show (default: 10)
  --all-columns         Show all available columns
  -s, --sort TEXT       Sort by column (e.g., "context_window:desc")
  --style TEXT           Rich table style: simple, rounded, minimal, square, ascii.
  --version             Show the version and exit.
  --help                Show this message and exit.
```

### Examples

Show specific columns:
```bash
models -c model_name -c provider
```

Filter results:
```bash
models -f "provider=openai"
```

Show all available columns:
```bash
models --all-columns
```

Use a different table style:
```bash
models --style rounded
```

List providers (default: table with counts):
```bash
models providers
```

Disable counts:
```bash
models providers --no-count
```

Comma-separated list:
```bash
models providers --format comma --no-count
```

Table style:
```bash
models providers --style rounded
```

Fuzzy search:
```bash
models search gpt --limit 20
```

List models for a provider:
```bash
models provider openai
models list --provider openai --provider-partial
```

Launch TUI:
```bash
models tui
models --tui
```

In the TUI:

- Press Enter in the search box to show details for the top result.
- Press Enter on a selected table row to show details for that model.
- The detail view includes `input_cost` and `output_cost`.
- A status bar shows the current provider/query/row counts and `advanced_fuzzy` mode.
- A split-view preview panel shows a quick summary for the highlighted row (toggle with `p`).
- Typing `/` in the search box shows a small command palette with matching commands.
- The chips line below the search box shows the active provider filter and other toggles (search target, sort, filters, compare, preview).

TUI keyboard shortcuts:

- `?` show help
- `Esc` clear the search input (when focused)
- `r` refresh table
- `Tab` focus next widget
- `p` toggle preview split view
- `s` cycle sort key
- `S` toggle sort order
- `i` cycle search target (id/name/both)
- `a` toggle supports_attachments filter
- `R` toggle supports_reasoning filter
- `t` toggle supports_temperature filter
- `c` add/remove current row to compare (shows compare when 2 selected)
- `C` clear compare selection

TUI slash commands (type in the search box and press Enter):

- `/help`
- `/clear`
- `/provider NAME` (e.g. `/provider openai`)
- `/provider` (no args: open provider picker UI)
- `/in id|name|both`
- `/sort KEY[:asc|:desc]` (e.g. `/sort context_window:desc`)
- `/filter attachments|reasoning|temperature on|off`
- `/compare add|show|clear`
- `/quit` (or `/exit`)
- `/af on|off` (toggle advanced fuzzy scoring)

Advanced fuzzy search:

- `--advanced-fuzzy` enables field-specific fuzzy scoring in search.

For more examples, refer to [[models.md]] file

## Development

### Rebuilding after changes

After making changes to the source code, you'll need to reinstall the package:

```bash
uv tool install -e .
```

### Running tests

To run tests (if available):

```bash
uv run pytest
```

## License

[Add your license information here]