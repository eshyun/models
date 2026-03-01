# Implementation Details

## CLI (Typer)

- The CLI is implemented in `src/models/main.py`.
- A `typer.Typer` instance named `app` is the primary command entry.
- The console script entry point is configured to call `models.main:app`.
 - The `cli()` wrapper appends `list` when no subcommand is provided so that `models` behaves like `models list`.

## Command behavior

- Commands:
  - `models` (defaults to `models list`)
  - `models list` (table view with filters/sort/columns)
  - `models provider PROVIDER` (shortcut for provider-scoped listing)
  - `models providers` (comma-separated provider list; optional counts)
  - `models search QUERY` (fuzzy search by `model_id`/`model_name`, sorted by fuzzy score desc)
  - `models tui` (Textual-based interactive TUI)

- Common options (list/provider):
  - `--column/-c` (repeatable)
  - `--filter/-f` (repeatable; supports `=`, `==`, `!=`, `~=`, `~`, `>`, `>=`, `<`, `<=`; operator whitespace is allowed)
  - `--limit/-l`
  - `--all-columns`
  - `--sort/-s` (supports `:asc`/`:desc`)
  - Column aliases are supported in `--column`, `--sort`, and `--filter` (e.g. `id`, `name`, `input`, `output`, `context`, `tokens`, `max_tokens`).
  - `--version`

- Provider selection:
  - `-p/--provider` is repeatable and also supports comma-separated lists (e.g. `-p openrouter -p google` or `-p openrouter,google`).
  - Exact matching is the default; `--provider-partial/-pp` enables case-insensitive substring matching.

## Data flow

- `ModelDataFetcher` fetches JSON from `https://models.dev/api.json`.
 - `ModelDataFetcher.fetch_data()` caches the fetched JSON on disk (default: `~/.cache/models/api.json`) with a configurable TTL (default: 24 hours). If the remote fetch fails, it falls back to the cached file when available.
   - Env: `MODELS_CACHE_PATH`, `MODELS_CACHE_TTL_SECONDS`
- Data is flattened into records and converted into a DataFrame for filtering/sorting.
- Results are rendered using `rich.Table`.
 - The selected Rich table style (`--style`) is threaded into rendering explicitly so it remains consistent even if filtering/sorting creates a new DataFrame.
 - Fuzzy search uses `rapidfuzz.fuzz.WRatio` over `model_id` and/or `model_name`.
 - TUI uses Textual widgets (`DataTable`, `Select`, `Input`) and refreshes the view on input changes.
 - TUI details view is shown via a `ModalScreen` when Enter is pressed in the search input (top result) or on the table (cursor row).
 - The TUI shows a status bar with provider/query/row counts and advanced-fuzzy mode.
 - The TUI defines key bindings for help (`?`), clear search (`Esc`), refresh (`r`), and focus cycling (`Tab`).
 - TUI table cost columns use fixed-decimal formatting for consistent scanning.
 - TUI supports keyboard-driven exploration features (sorting, search target selection, capability filters) and a compare mode for two models.
 - TUI slash commands include `/quit` and `/exit` for exiting the app.
 - TUI layout uses a split view (table + preview panel) that can be toggled with `p`, and shows a simple command palette when typing `/`.
 - The detail screen can render in two columns (summary on the left, raw JSON on the right).
 - Advanced fuzzy scoring can be enabled via `--advanced-fuzzy` (CLI) or `/af on|off` in the TUI.
