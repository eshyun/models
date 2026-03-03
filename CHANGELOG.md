# Changelog

## Unreleased

- Changed default table output columns across list/search to: provider, id, cost(in/out), context, name, updated.
- When a provider filter is applied via `--provider/-p`, the default output omits the `provider` column.
- Combined `input_cost` and `output_cost` into a synthetic `cost(in/out)` column in the default output.

- Expanded the underlying DataFrame schema to include (best-effort) flattened raw model JSON fields as columns, enabling sorting/filtering/selection on non-default fields (e.g. `last_updated`, `release_date`).
- Enhanced `--filter` to parse human-readable numeric values (e.g. `1.0M`, `4K`, comma-separated digits) for all numeric columns and to support date comparisons on date-like columns using `dateparser`.
- Enhanced `--column` to support multi-column expansion aliases (e.g. `model`, `supports`, `cost`) in addition to single-column aliases.
- Enhanced `--sort` to accept expanded aliases by using the first column of the expansion (e.g. `--sort model` sorts by `model_id`).
- Added `--sort/-s` to `models search` to optionally re-sort the final filtered result set by a concrete column (default remains fuzzy score ranking).
- Added boolean filter short-hands: `-f reasoning` (true) and `-f !reasoning` (false).
- Added additional convenience aliases for common raw fields: `updated -> last_updated`, `release -> release_date`.
- Added conservative prefix/suffix normalization for aliases (no fuzzy matching), e.g. `updated_at`/`last_updated_at` and `released_at`/`release_at`.
- Added `--add-column/-C` to append columns to the default output without replacing it (complements `--column/-c`).
- Improved `--all-columns` output readability by hiding nested flattened columns (those containing `__`) while keeping them available for `--sort`, `--filter`, and explicit `--column` display.
- Enhanced TUI sorting and table customization: all DataFrame columns are available as sort keys (including `__`), `/sort` opens a sort-key picker UI, and `/columns add|remove|reset` adjusts visible columns.

- Switched CLI implementation from Click to Typer.
- Updated console script entry point to use Typer app.
- Removed Click dependency from packaging metadata.
- Added CLI subcommands: `providers`, `search`, `list`, `provider`.
- Added fuzzy search for `model_id`/`model_name` using RapidFuzz.
- Added an interactive Textual-based TUI (`models tui` / `models --tui`).
- Added model detail dialog in TUI (Enter shows selected/top model details).
- Added TUI slash commands in the search box (`/help`, `/clear`, `/provider NAME`).
- Improved TUI UX with a status bar (provider/query/row counts/advanced-fuzzy) and keyboard shortcuts (`?`, `Esc`, `r`, `Tab`).
- Standardized TUI table cost formatting to fixed decimals for consistent scanning.
- Added TUI exploration features: keyboard sorting, search-target selection, boolean capability filters, compare mode, and `/quit`/`/exit` commands.
- Improved TUI detail modal content structure while keeping raw JSON.
- Upgraded TUI layout with a split-view preview panel (`p`), a command palette for slash commands, and a 2-column detail view (summary + raw JSON).
- Added `--style` option to Rich table outputs to select box styles (simple, rounded, minimal, square, ascii).
- Fixed `models list --style` so the selected table style is applied consistently.
- Improved CLI `--filter` parsing to allow spaces around operators and added `==` as an alias of `=`.
- Added common column aliases for CLI options (`--column`, `--sort`, `--filter`), e.g. `id`, `name`, `input`, `output`, `context`, `tokens`.
- Added `--advanced-fuzzy` option for field-specific fuzzy scoring and TUI toggle via `/af on|off`.
- Improved provider filtering across CLI commands: repeatable `-p/--provider`, comma-separated provider lists, and shared exact/partial filtering logic.
- Added `--provider-partial/-pp` to `models search` for case-insensitive substring provider matching.
- Added `--style` to `models providers` when using `--format table`.
- Added a 24h local cache for remote `api.json` fetches with configurable TTL/path and stale-cache fallback on network errors.
- Added `--min-score` to `models search` (default: 50) and applied the same minimum fuzzy score cutoff in the TUI search results.
- Interpreted `--limit 0` as "no limit" across CLI commands that support `--limit`.
- Added `--style plain` for rendering tables without borders/grids.
- Updated the TUI model detail view layout (summary + raw JSON).
- Added `models show MODEL_ID` to display full raw API model details with selectable output format (json/lines/table).
- Added a header to the TUI raw JSON panel in the detail view.
