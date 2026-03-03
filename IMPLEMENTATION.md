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
  - `--column/-c` (repeatable; also supports comma-separated lists)
  - `--filter/-f` (repeatable; supports `=`, `==`, `!=`, `~=`, `~`, `>`, `>=`, `<`, `<=`; operator whitespace is allowed)
  - `--limit/-l` (`0` means no limit)
  - `--all-columns`
  - `--sort/-s` (supports `:asc`/`:desc`)
  - Column aliases are supported in `--column`, `--sort`, and `--filter` (e.g. `id`, `name`, `input`, `output`, `context`, `tokens`, `max_tokens`).
  - `--version`

- Provider selection:
  - `-p/--provider` is repeatable and also supports comma-separated lists (e.g. `-p openrouter -p google` or `-p openrouter,google`).
  - Exact matching is the default; `--provider-partial/-pp` enables case-insensitive substring matching.

- Fuzzy search cutoff:
  - `models search` supports `--min-score` (default: 50) to exclude low-relevance fuzzy matches.
  - The TUI applies the same default minimum score cutoff (50) when a query is present.

## Data flow

- `ModelDataFetcher` fetches JSON from `https://models.dev/api.json`.
 - `ModelDataFetcher.fetch_data()` caches the fetched JSON on disk (default: `~/.cache/models/api.json`) with a configurable TTL (default: 24 hours). If the remote fetch fails, it falls back to the cached file when available.
   - Env: `MODELS_CACHE_PATH`, `MODELS_CACHE_TTL_SECONDS`
- Data is flattened into records and converted into a DataFrame for filtering/sorting.
 - Flattening strategy:
   - `ModelDataFetcher._flatten_model_data()` keeps the stable “core columns” (`model_id`, `model_name`, `provider`, costs, limits, supports_*).
   - It additionally flattens the raw model JSON object into extra columns using a `__` separator for nested keys (best-effort), so that raw fields can be used with `--column`, `--filter`, and `--sort` (e.g. `last_updated`, `release_date`, `modalities__input`).
   - Non-scalar values (lists/dicts) are serialized to JSON strings for safe table display and consistent filtering.
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
 - TUI sorting:
   - The TUI sort key supports all available columns in the loaded DataFrame (including nested flattened columns containing `__`).
   - `/sort` with no args opens a picker UI for selecting a sort key.
 - TUI table columns:
   - `/columns add SPEC` appends columns to the current table.
   - `/columns remove SPEC` removes columns from the current table.
   - `/columns reset` restores the default table column set.
- TUI layout uses a split view (table + preview panel) that can be toggled with `p`, and shows a simple command palette when typing `/`.
 - The detail screen renders in two columns (summary on the left, raw JSON on the right).
 - Advanced fuzzy scoring can be enabled via `--advanced-fuzzy` (CLI) or `/af on|off` in the TUI.

## Column selection / aliases

- `--column` parsing supports repeated flags and comma-separated lists.
- `--column/-c` replaces the default output columns with the user-specified set.
- `--add-column/-C` adds columns to the default output without replacing it (useful for “show me everything plus X”).
- Column specs support both:
  - single-column aliases (e.g. `id -> model_id`)
  - multi-column expansions (e.g. `model -> model_id,model_name`, `supports -> supports_attachments,supports_reasoning,supports_temperature`, `cost -> input_cost,output_cost`).
- Prefix/suffix normalization (conservative, no fuzzy matching): common variants like `updated_at`, `last_updated_at`, `released_at` are normalized to the stable alias tokens `updated`/`release` before applying alias/expansion rules.

## Default output columns

- Default table output columns are: `provider`, `model_id`, `cost(in/out)`, `context_window`, `model_name`, `last_updated`.
- When a provider filter is applied via `--provider/-p`, the default output omits the `provider` column for readability.
- `cost(in/out)` is a synthetic display column rendered as `input_cost / output_cost` (e.g. `0.6 / 3.6`).

## Sorting

- `--sort` accepts raw column names or aliases.
- If an alias expands to multiple columns, the sort key uses the first column of the expansion (e.g. `--sort model` sorts by `model_id`).
- `models search` uses fuzzy scoring to rank matches by default, but also supports `--sort` to re-sort the final filtered result set by a concrete column.

## Filtering

- Filter expressions support operators: `=`, `==`, `!=`, `~=`, `~`, `>`, `>=`, `<`, `<=`.
- Boolean short-hands are supported:
  - `reasoning` is treated as `reasoning=true`
  - `!reasoning` is treated as `reasoning=false`

- Numeric parsing:
  - For numeric columns (int/float), filter values accept human-readable suffixes: `K`, `M`, `B`/`G`, `T`.
  - Commas are ignored (e.g. `4,096` -> `4096`).

- Date parsing:
  - Date-like columns (e.g. `release_date`, `last_updated`, and common `*_date`/`*_at` patterns) support comparisons with `=`, `!=`, `<`, `<=`, `>`, `>=`.
  - Values are parsed with `dateparser.parse()` (supports absolute dates like `2026-03-01` and relative expressions like `1 month ago`).
  - If parsing fails for the filter value, filtering falls back to string comparison and emits a warning.
