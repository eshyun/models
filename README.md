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

### Data caching

The tool fetches model data from `https://models.dev/api.json` and caches it locally to avoid repeated network requests.

- Default TTL: 24 hours
- Cache path: `~/.cache/models/api.json` (configurable)
- On network failure, the tool will fall back to the cached data if available.

Environment variables:

- `MODELS_CACHE_PATH`: Override the cache file path.
- `MODELS_CACHE_TTL_SECONDS`: Override TTL in seconds (set to `0` to disable TTL and always fetch remotely).

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
  -C, --add-column TEXT Additional columns to add to the default output (can be specified multiple times)
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

Add extra columns to the default output:
```bash
models -C updated -C release
models search gpt -C updated_at -l 20
```

Filter results:
```bash
models -f "provider=openai"
```

Filter syntax notes:

- Operators support optional spaces (e.g. `provider ~= openrou`).
- Supported operators: `=`, `==` (alias of `=`), `!=`, `~=`, `~`, `>`, `>=`, `<`, `<=`.
- Numeric comparisons support human-readable suffixes for numeric columns (e.g. `4K`, `1.0M`, `2B`, `1T`) and commas (e.g. `4,096`).
- Date comparisons are supported on date-like columns (e.g. `release_date`, `last_updated`) using `dateparser.parse()` (e.g. `release_date >= 2026-03-01`, `release_date >= 1 month ago`).
- Common column aliases can be used in `--column`, `--sort`, and `--filter`:
  - `id` -> `model_id`
  - `name` -> `model_name`
  - `model` -> `model_id`, `model_name`
  - `updated` -> `last_updated`
    - Also accepts common variants like `updated_at`, `last_updated_at`.
  - `release` -> `release_date`
    - Also accepts common variants like `released`, `released_at`, `release_at`.
  - `supports` -> `supports_attachments`, `supports_reasoning`, `supports_temperature`
  - `cost` -> `input_cost`, `output_cost`
  - `input` -> `input_cost`
  - `output` -> `output_cost`
  - `context` -> `context_window`
  - `tokens` / `max_tokens` -> `max_output_tokens`

- Boolean filters support a short form:
  - `-f reasoning` is treated as `reasoning=true`
  - `-f !reasoning` is treated as `reasoning=false`

Show all available columns:
```bash
models --all-columns
```

Note:

- `--all-columns` hides nested flattened columns (those containing `__`) for readability.
- Those hidden columns are still available for `--sort`, `--filter`, and can be displayed explicitly via `--column`.

Use a different table style:
```bash
models --style rounded
```

Alias examples:
```bash
models -c id -c provider -c input -c output
models --sort context:desc
models -f "id ~= gpt" -f "context >= 100000"
models list -p openrouter --sort release_date:desc --filter "context_window=1.0M"
models list -p openrouter --sort release_date:desc --filter "context >= 4096"
models list -p openrouter --sort release_date:desc --filter "release_date >= 2026-03-01"
```

Sort by a non-default (raw) column:
```bash
models -c model,last_updated,release_date --sort last_updated:desc -l 10
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
models providers --style plain
```

Fuzzy search:
```bash
models search gpt --limit 20
models search --min-score 50 gemini --limit 20
models search gemini --limit 0
models search gpt --sort updated:desc -C updated
```

List models for a provider:
```bash
models provider openai
models list --provider openai --provider-partial
models list --limit 0
models list -c provider,model_id,model_name -l 20
models search -p openrouter -p google gemini -l 10
models search -p openrouter,google gemini -l 10
models search -p open --provider-partial gemini -l 10
```

Show raw model details by model_id:
```bash
models show google/gemini-3-flash
models show google/gemini-3-flash --format lines
models show google/gemini-3-flash --format table
models show gemini --provider google
```

Launch TUI:
```bash
models tui
models --tui
```

In the TUI:

- Search results apply a minimum fuzzy score cutoff (default: 50) to reduce low-relevance noise.

- Press Enter in the search box to show details for the top result.
- Press Enter on a selected table row to show details for that model.
- The detail view includes `input_cost` and `output_cost`.
- The detail view shows a summary on the left and raw JSON on the right.
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