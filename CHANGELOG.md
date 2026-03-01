# Changelog

## Unreleased

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
