import json
from typing import Dict, Any, List, Optional, Tuple
import typer
import click
import fireducks.pandas as fd
import pandas as pd
from urllib.parse import urljoin
import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rapidfuzz import fuzz
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Select, Footer, Header, Static, Button, OptionList

# Import version information from our utility module
from ._version import __version__


STYLE_CHOICES = ["simple", "rounded", "minimal", "square", "ascii"]

class ModelDataFetcher:
    """Fetches and processes AI model data from the models.dev API."""
    
    BASE_URL = "https://models.dev/"
    API_ENDPOINT = "api.json"
    
    def __init__(self):
        self._raw_data: Optional[dict] = None
        self._df: Optional[fd.DataFrame] = None
    
    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetches model data from the API.
        
        Returns:
            Dict containing the raw model data.
            
        Raises:
            requests.RequestException: If the API request fails.
        """
        url = urljoin(self.BASE_URL, self.API_ENDPOINT)

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        self._raw_data = response.json()
        

        
        return self._raw_data
    
    def _flatten_model_data(self, provider: str, model_id: str, model_data: Dict) -> Dict:
        """Flatten the model data structure for easier analysis.
        
        Args:
            provider: The provider of the model (e.g., 'google', 'openai').
            model_id: The unique identifier for the model.
            model_data: The raw model data from the API.
            
        Returns:
            A flattened dictionary with model information. Missing cost data will be set to None.
        """
        # Initialize default values
        result = {
            'model_id': model_id,
            'model_name': '',
            'provider': provider,
            'supports_attachments': False,
            'supports_reasoning': False,
            'supports_temperature': False,
            'input_cost': None,  # Use None for missing values
            'output_cost': None,  # Use None for missing values
            'cost_cache_read_per_million': None,  # Use None for missing values
            'context_window': 0,
            'max_output_tokens': 0,
        }
        
        if not isinstance(model_data, dict):
            print(f"Warning: model_data is not a dictionary for {model_id}")
            return result
        
        # Get model name
        result['model_name'] = model_data.get('name', model_data.get('model_name', ''))
        
        # Get boolean flags
        result['supports_attachments'] = bool(model_data.get('attachment', False))
        result['supports_reasoning'] = bool(model_data.get('reasoning', False))
        result['supports_temperature'] = bool(model_data.get('temperature', False))
        

        
        # Check for cost data in different possible locations
        cost_data = None
        
        # Try to get cost data from the model_data directly
        if 'cost' in model_data and isinstance(model_data['cost'], dict):
            cost_data = model_data['cost']
        # Check if cost data is nested under 'pricing' or 'price' keys
        elif 'pricing' in model_data and isinstance(model_data['pricing'], dict):
            cost_data = model_data['pricing']
        elif 'price' in model_data and isinstance(model_data['price'], dict):
            cost_data = model_data['price']
        # Check if cost data is at the top level with different keys
        elif any(key in model_data for key in ['input_cost', 'output_cost', 'cost_per_token']):
            cost_data = {
                'input': model_data.get('input_cost', 0),
                'output': model_data.get('output_cost', 0),
                'cache_read': model_data.get('cost_cache_read_per_million', 0)
            }

        
        # Process cost data if found
        if cost_data:
            # Handle different possible key names for input and output costs
            input_cost = cost_data.get('input') or cost_data.get('input_cost') or cost_data.get('cost_per_token', {}).get('input', 0)
            output_cost = cost_data.get('output') or cost_data.get('output_cost') or cost_data.get('cost_per_token', {}).get('output', 0)
            cache_read = cost_data.get('cache_read') or cost_data.get('cost_cache_read_per_million', 0)
            
            # Convert to float, keeping None for missing values
            try:
                if input_cost is not None and input_cost != "":
                    result['input_cost'] = float(input_cost)
                if output_cost is not None and output_cost != "":
                    result['output_cost'] = float(output_cost)
                if cache_read is not None and cache_read != "":
                    result['cost_cache_read_per_million'] = float(cache_read)
            except (ValueError, TypeError):
                # Keep the values as None if conversion fails
                pass
        
        # Get limits
        if 'limit' in model_data and isinstance(model_data['limit'], dict):
            limits = model_data['limit']
            result['context_window'] = int(limits.get('context', 0))
            result['max_output_tokens'] = int(limits.get('output', 0))
        

        return result
    
    def to_dataframe(self) -> fd.DataFrame:
        """
        Converts the fetched model data to a FireDucks DataFrame.
        
        Returns:
            FireDucks DataFrame containing the model data.
            
        Raises:
            ValueError: If no data has been fetched yet.
        """
        if self._raw_data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        records = []
        
        # Process models from each provider
        for provider, provider_data in self._raw_data.items():
            models = provider_data.get('models', {})
            for model_id, model_data in models.items():
                record = self._flatten_model_data(provider, model_id, model_data)
                records.append(record)
        
        # Create a pandas DataFrame first, then convert to FireDucks
        pdf = pd.DataFrame(records)
        
        self._df = fd.DataFrame(pdf)
        
        return self._df
    
    def get_models_by_provider(self, provider: str) -> fd.DataFrame:
        """
        Get all models from a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'google', 'azure')
            
        Returns:
            FireDucks DataFrame with models from the specified provider.
        """
        if self._df is None:
            self.to_dataframe()
        return self._df[self._df['provider'] == provider]
    
    def get_most_affordable_models(self, n: int = 10, min_context: int = 0) -> fd.DataFrame:
        """
        Get the most cost-effective models based on input cost per token.
        
        Args:
            n: Number of models to return
            min_context: Minimum context window size to consider
            
        Returns:
            Sorted FireDucks DataFrame with the most affordable models
        """
        if self._df is None:
            self.to_dataframe()
            
        filtered = self._df[self._df['context_window'] >= min_context]
        return filtered.sort_values('input_cost').head(n)


def get_model_data() -> fd.DataFrame:
    """
    Fetches model data from the API and returns it as a FireDucks DataFrame.
    
    Returns:
        FireDucks DataFrame containing the model data.
    """
    fetcher = ModelDataFetcher()
    fetcher.fetch_data()
    return fetcher.to_dataframe()


def get_available_columns() -> List[str]:
    """Return a list of available column names in the model data."""
    return [
        'model_id', 'model_name', 'provider', 'supports_attachments',
        'supports_reasoning', 'supports_temperature', 'input_cost',
        'output_cost', 'cost_cache_read_per_million',
        'context_window', 'max_output_tokens'
    ]

def resolve_column_alias(column: str, available_columns: List[str]) -> str:
    """Resolve column aliases to their actual column names.
    
    Args:
        column: The column name or alias to resolve
        available_columns: List of available column names in the DataFrame
        
    Returns:
        The resolved column name if found, otherwise the original column name
    """
    # Define aliases mapping
    aliases = {
        'cost_input_per_million': 'input_cost',
        'cost_output_per_million': 'output_cost',
        'cost_cache_read': 'cost_cache_read_per_million',
        'name': 'model_name',
        'id': 'model_id'
    }
    
    # Check if the column is an alias
    if column in aliases and aliases[column] in available_columns:
        return aliases[column]
    return column

def format_context_window(value: int) -> str:
    """Format context window value to human-readable format (K or M)."""
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.0f}K"
    return str(value)


def format_int_with_commas(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and pd.isna(value):
            return "N/A"
        return f"{int(value):,}"
    except Exception:
        return str(value)


def format_cost(value: Optional[float]) -> str:
    """Format cost value, showing N/A for None values."""
    if value is None or value == "":
        return "N/A"
    try:
        # Format to display up to 4 decimal places, removing trailing zeros
        return f"{float(value):.4g}"
    except (ValueError, TypeError):
        return "N/A"


def format_cost_fixed(value: Any, decimals: int = 4) -> str:
    """Format cost value with fixed decimals for consistent table display."""
    if value is None or value == "":
        return "N/A"
    try:
        if isinstance(value, float) and pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "", "none", "nan"}:
        return False
    return True


def resolve_rich_table_box(style: Optional[str]) -> Optional[box.Box]:
    if not style:
        return None
    key = style.strip().lower()
    mapping = {
        "simple": box.SIMPLE,
        "rounded": box.ROUNDED,
        "minimal": box.MINIMAL,
        "square": box.SQUARE,
        "ascii": box.ASCII,
    }
    if key not in mapping:
        raise typer.BadParameter(
            f"Unknown --style '{style}'. Supported: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[key]


def _normalize_query(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _match_rank(query: str, text: str) -> Tuple[int, int]:
    """Return (tier, score) where lower tier is better, higher score is better.

    Tiers:
      0: exact match
      1: prefix match
      2: substring match
      3: all tokens contained (space separated)
      4: fuzzy fallback (rapidfuzz)
    """
    q = _normalize_query(query)
    t = _normalize_query(text)

    if not q or not t:
        return (99, 0)

    if t == q:
        return (0, 100)
    if t.startswith(q):
        return (1, 95)
    if q in t:
        return (2, 90)

    tokens = [tok for tok in q.split(" ") if tok]
    if tokens and all(tok in t for tok in tokens):
        return (3, 85)

    return (4, fuzz.WRatio(t, q))


def _match_rank_for_field(query: str, text: str, field: str, advanced_fuzzy: bool) -> Tuple[int, int]:
    q = _normalize_query(query)
    t = _normalize_query(text)

    if not q or not t:
        return (99, 0)

    if t == q:
        return (0, 100)
    if t.startswith(q):
        return (1, 95)
    if q in t:
        return (2, 90)

    tokens = [tok for tok in q.split(" ") if tok]
    if tokens and all(tok in t for tok in tokens):
        return (3, 85)

    if not advanced_fuzzy:
        return (4, fuzz.WRatio(t, q))

    if field == "model_id":
        return (4, fuzz.QRatio(t, q))
    if field == "model_name":
        return (4, fuzz.token_set_ratio(t, q))
    return (4, fuzz.WRatio(t, q))


def _row_best_rank(query: str, row: Any, fields: List[str], advanced_fuzzy: bool) -> Tuple[int, int]:
    best: Tuple[int, int] = (99, 0)
    for f in fields:
        cand = str(row.get(f, "") if hasattr(row, "get") else getattr(row, f, ""))
        rank = _match_rank_for_field(query, cand, f, advanced_fuzzy)
        # Prefer lower tier; for same tier prefer higher score
        if rank[0] < best[0] or (rank[0] == best[0] and rank[1] > best[1]):
            best = rank
    return best

def display_results(data: List[Dict], columns: List[str], title: str = "Model Data", style: Optional[str] = None):
    """Display results in a formatted table.
    
    Args:
        data: List of dictionaries containing the model data
        columns: List of column names to display
        title: Title of the table (default: "Model Data")
    """
    if not data:
        print("No data to display.")
        return

    # Import pandas here to avoid making it a required dependency
    try:
        import pandas as pd
    except ImportError:
        pd = None

    console = Console()
    resolved_box = resolve_rich_table_box(style)
    if resolved_box is not None:
        table = Table(title=title, show_header=True, header_style="bold magenta", box=resolved_box)
    else:
        table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # Add columns
    for col in columns:
        table.add_column(col, overflow="fold")
    
    # Add rows
    for row in data:
        formatted_row = []
        for col in columns:
            value = row.get(col, None)
            
            # Handle NaN/None values
            if value is None or (pd and pd.isna(value)) or value == '':
                formatted_row.append("N/A")
                continue
                
            # Format specific columns
            if col == 'context_window' and isinstance(value, (int, float)):
                formatted_row.append(format_context_window(value))
            elif col in ['input_cost', 'output_cost', 'cost_cache_read_per_million']:
                formatted_row.append(format_cost(value))
            else:
                formatted_row.append(str(value))
                
        table.add_row(*formatted_row)
    
    console.print(table)

app = typer.Typer(add_completion=False)


def _version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    column: Optional[List[str]] = typer.Option(
        None,
        "--column",
        "-c",
        help="Column(s) to display. Can be specified multiple times.",
    ),
    filters: Optional[List[str]] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter conditions.",
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of rows to display."),
    all_columns: bool = typer.Option(False, "--all-columns", help="Show all columns in the output."),
    sort: Optional[str] = typer.Option(
        None,
        "--sort",
        "-s",
        help='Sort by column (e.g., "context_window:desc").',
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="List models for a provider (exact match by default).",
    ),
    provider_partial: bool = typer.Option(False, "--provider-partial", help="Use partial provider match (provider~=...)."),
    tui: bool = typer.Option(False, "--tui", help="Launch the interactive TUI (Textual)."),
    advanced_fuzzy: bool = typer.Option(False, "--advanced-fuzzy/--no-advanced-fuzzy", help="Use advanced fuzzy scoring for search (field-specific)."),
    style: str = typer.Option(
        "simple",
        "--style",
        click_type=click.Choice(STYLE_CHOICES, case_sensitive=False),
        help="Rich table style: simple, rounded, minimal, square, ascii.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    if ctx.invoked_subcommand is not None:
        return
    if tui:
        ModelsTUI(advanced_fuzzy=advanced_fuzzy).run()
        return
    _list_models(
        column=column,
        filters=filters,
        limit=limit,
        all_columns=all_columns,
        sort=sort,
        provider=provider,
        provider_partial=provider_partial,
        style=style,
    )


def _validate_columns(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return values
    available = get_available_columns()
    normalized = {c.lower(): c for c in available}
    resolved: List[str] = []
    invalid: List[str] = []

    for v in values:
        key = v.lower()
        if key in normalized:
            resolved.append(normalized[key])
        else:
            invalid.append(v)

    if invalid:
        raise typer.BadParameter(
            f"Unknown column(s): {', '.join(invalid)}. Available columns: {', '.join(available)}"
        )
    return resolved
def _load_df() -> fd.DataFrame:
    fetcher = ModelDataFetcher()
    fetcher.fetch_data()
    return fetcher.to_dataframe()


def _load_pdf() -> pd.DataFrame:
    fetcher = ModelDataFetcher()
    fetcher.fetch_data()
    raw = fetcher._raw_data or {}

    records: List[Dict[str, Any]] = []
    for provider, provider_data in raw.items():
        models = provider_data.get('models', {}) if isinstance(provider_data, dict) else {}
        for model_id, model_data in (models or {}).items():
            records.append(fetcher._flatten_model_data(provider, model_id, model_data))

    return pd.DataFrame(records)


def _apply_filters(df: fd.DataFrame, filters: Optional[List[str]]) -> fd.DataFrame:
    if not filters:
        return df

    import re

    for f in filters:
        # Check for different filter types
        if '~=' in f:  # Partial match
            col, val = f.split('~=', 1)
            col = col.strip()
            # Resolve column alias
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns:
                df = df[df[resolved_col].astype(str).str.contains(re.escape(val), case=False, na=False)]
            else:
                typer.echo(f"Warning: Unknown column '{col}' in filter", err=True)
        elif '~' in f and not f.startswith('~'):  # Regex match
            col, pattern = f.split('~', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns:
                try:
                    df = df[df[resolved_col].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
                except re.error as e:
                    typer.echo(f"Error in regex pattern '{pattern}': {e}", err=True)
            else:
                typer.echo(f"Warning: Unknown column '{col}' in filter", err=True)
        elif '>=' in f:
            col, val = f.split('>=', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns and df[resolved_col].dtype in ('int64', 'float64'):
                try:
                    val_f = float(val)
                    df = df[df[resolved_col] >= val_f]
                except (ValueError, TypeError):
                    typer.echo(f"Warning: Invalid value '{val}' for column '{col}'", err=True)
        elif '<=' in f:
            col, val = f.split('<=', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns and df[resolved_col].dtype in ('int64', 'float64'):
                try:
                    val_f = float(val)
                    df = df[df[resolved_col] <= val_f]
                except (ValueError, TypeError):
                    typer.echo(f"Warning: Invalid value '{val}' for column '{col}'", err=True)
        elif '!=' in f:
            col, val = f.split('!=', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns:
                df = df[df[resolved_col].astype(str) != str(val)]
            else:
                typer.echo(f"Warning: Unknown column '{col}' in filter", err=True)
        elif '=' in f:  # Exact match
            col, val = f.split('=', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns:
                # Try to convert value to the same type as the column
                try:
                    if df[resolved_col].dtype == 'int64':
                        val = int(val)
                    elif df[resolved_col].dtype == 'float64':
                        val = float(val)
                    elif df[resolved_col].dtype == 'bool':
                        val = str(val).lower() in ('true', '1', 'yes')
                except (ValueError, TypeError):
                    pass
                df = df[df[resolved_col] == val]
            else:
                typer.echo(f"Warning: Unknown column '{col}' in filter", err=True)
        elif '>' in f:  # Greater than
            col, val = f.split('>', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns and df[resolved_col].dtype in ('int64', 'float64'):
                try:
                    val_f = float(val)
                    df = df[df[resolved_col] > val_f]
                except (ValueError, TypeError):
                    typer.echo(f"Warning: Invalid value '{val}' for column '{col}'", err=True)
        elif '<' in f:
            col, val = f.split('<', 1)
            col = col.strip()
            resolved_col = resolve_column_alias(col, df.columns.tolist())
            if resolved_col in df.columns and df[resolved_col].dtype in ('int64', 'float64'):
                try:
                    val_f = float(val)
                    df = df[df[resolved_col] < val_f]
                except (ValueError, TypeError):
                    typer.echo(f"Warning: Invalid value '{val}' for column '{col}'", err=True)

    return df


def _select_columns(df: fd.DataFrame, column: Optional[List[str]], all_columns: bool) -> List[str]:
    if all_columns:
        columns = get_available_columns()
    elif column:
        column = _validate_columns(column)
        columns = [resolve_column_alias(col, get_available_columns()) for col in column]
    else:
        columns = [
            'provider',
            'model_id',
            'input_cost',
            'output_cost',
            'context_window',
            'max_output_tokens',
            'model_name',
        ]

    available_cols = set(df.columns)
    valid_columns = [col for col in columns if col in available_cols]

    invalid_columns = [col for col in columns if col not in available_cols]
    if invalid_columns:
        typer.echo(
            f"Warning: The following columns are not available and will be ignored: {', '.join(invalid_columns)}",
            err=True,
        )

    if not valid_columns and columns:
        typer.echo("Warning: None of the specified columns are available. Showing all columns.", err=True)
        valid_columns = df.columns.tolist()

    return valid_columns


def _apply_sort(df: fd.DataFrame, sort: Optional[str]) -> fd.DataFrame:
    if not sort:
        return df

    sort_column = sort
    ascending = True

    if ':' in sort:
        sort_column, direction = sort.split(':', 1)
        ascending = direction.lower() != 'desc'

    sort_column = resolve_column_alias(sort_column, df.columns.tolist())

    if sort_column in df.columns:
        return df.sort_values(by=sort_column, ascending=ascending)

    typer.echo(
        f"Warning: Cannot sort by unknown column '{sort_column}'. Available columns: {', '.join(sorted(df.columns))}",
        err=True,
    )
    return df


def _render_table(df: fd.DataFrame, columns: List[str], limit: int, title_prefix: str = "Model Data"):
    display_df = df[columns].head(limit).copy()

    for col in display_df.columns:
        if display_df[col].dtype == 'float64':
            display_df[col] = display_df[col].apply(lambda x: round(x, 6) if pd.notnull(x) else x)
        elif display_df[col].dtype == 'int64':
            display_df[col] = display_df[col].astype('Int64')

    display_data = display_df.to_dict('records')
    display_results(
        display_data,
        columns,
        f"{title_prefix} (showing {len(display_data)} of {len(df)} models)",
        style=getattr(df, "_rich_table_style", None),
    )


def _list_models(
    *,
    column: Optional[List[str]],
    filters: Optional[List[str]],
    limit: int,
    all_columns: bool,
    sort: Optional[str],
    provider: Optional[str] = None,
    provider_partial: bool = False,
    style: Optional[str] = None,
):
    df = _load_df()

    # Attach style to df instance for downstream rendering
    try:
        setattr(df, "_rich_table_style", style)
    except Exception:
        pass

    # Provider shortcut
    if provider:
        if provider_partial:
            filters = list(filters or []) + [f"provider~={provider}"]
        else:
            filters = list(filters or []) + [f"provider={provider}"]

    df = _apply_filters(df, filters)
    df = _apply_sort(df, sort)

    columns = _select_columns(df, column, all_columns)
    _render_table(df, columns, limit)


class ModelDetailScreen(ModalScreen):
    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, title: str, content: str, right_content: Optional[str] = None):
        super().__init__()
        self._title = title
        self._content = content
        self._right_content = right_content

    def compose(self) -> ComposeResult:
        yield Static(self._title)
        if self._right_content is None:
            with VerticalScroll():
                yield Static(self._content)
        else:
            with Horizontal():
                with VerticalScroll():
                    yield Static(self._content)
                with VerticalScroll():
                    yield Static(self._right_content)
        yield Button("Close", id="close")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss()

    def action_dismiss(self) -> None:
        self.dismiss()


class ProviderPickerScreen(ModalScreen):
    CSS = """
    ProviderPickerScreen {
        layout: vertical;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    #provider_picker {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, providers: List[str]):
        super().__init__()
        self._providers = providers
        self._all_options: List[str] = ["(all)"] + [p for p in providers]
        self._current_options: List[str] = list(self._all_options)

    def compose(self) -> ComposeResult:
        yield Static(f"Select provider ({len(self._providers)})")
        yield Input(placeholder="Filter providers (substring)", id="provider_filter")
        yield OptionList(*self._all_options, id="provider_picker")
        with Horizontal():
            yield Button("Apply", id="apply")
            yield Button("Clear", id="clear")
            yield Button("Cancel", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.query_one("#provider_filter", Input).focus()
        except Exception:
            pass
        try:
            picker = self.query_one("#provider_picker", OptionList)
            picker.highlighted = 0
        except Exception:
            pass
        try:
            # Keep keyboard focus on the filter input so users can type immediately.
            self.query_one("#provider_filter", Input).focus()
        except Exception:
            pass

    def _set_picker_options(self, options: List[str]) -> None:
        self._current_options = list(options)
        picker = self.query_one("#provider_picker", OptionList)
        if hasattr(picker, "clear_options"):
            picker.clear_options()
            picker.add_options(self._current_options)
        else:
            # Fallback for older versions: replace widget contents
            try:
                picker.remove()
            except Exception:
                pass
            try:
                parent = self.query_one("#provider_filter", Input).parent
                if parent is not None:
                    parent.mount(OptionList(*self._current_options, id="provider_picker"))
            except Exception:
                pass
        try:
            picker.highlighted = 0
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "provider_filter":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self._set_picker_options(self._all_options)
            return
        filtered = [opt for opt in self._all_options if q in opt.lower()]
        if not filtered:
            filtered = ["(no matches)"]
        self._set_picker_options(filtered)

    def action_apply(self) -> None:
        value = ""
        try:
            picker = self.query_one("#provider_picker", OptionList)
            idx = getattr(picker, "highlighted", None)
            if idx is None:
                idx = 0
            if 0 <= int(idx) < len(self._current_options):
                chosen = self._current_options[int(idx)]
                if chosen == "(all)" or chosen == "(no matches)":
                    value = ""
                else:
                    value = chosen
        except Exception:
            value = ""
        try:
            widget = self.app.query_one("#provider", Select)
            widget.value = value
        except Exception:
            pass
        try:
            if hasattr(self.app, "_refresh_table"):
                self.app._refresh_table()
        except Exception:
            pass
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
            return
        if event.button.id == "clear":
            self._set_picker_options(self._all_options)
            self.action_apply()
            return
        if event.button.id == "cancel":
            self.dismiss()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_apply()


class ModelsTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #toolbar1 {
        height: auto;
        min-height: 4;
        layout: horizontal;
    }
    #provider {
        height: 3;
    }
    #query {
        height: 3;
    }
    #toolbar2 {
        height: auto;
        min-height: 1;
    }
    #chips {
        height: 1;
    }
    #palette {
        height: auto;
    }
    #main {
        height: 1fr;
        layout: horizontal;
    }
    #table {
        height: 1fr;
    }
    #preview {
        width: 45;
        border: round $primary;
    }
    #status {
        height: 1;
        dock: bottom;
    }
    """

    BINDINGS = [
        ("?", "help", "Help"),
        ("escape", "escape", "Clear"),
        ("r", "refresh", "Refresh"),
        ("tab", "focus_next", "Focus next"),
        ("p", "preview_toggle", "Preview"),
        ("s", "sort_cycle", "Sort key"),
        ("S", "sort_toggle", "Sort order"),
        ("i", "search_in_cycle", "Search in"),
        ("a", "toggle_attachments", "Filter: attachments"),
        ("R", "toggle_reasoning", "Filter: reasoning"),
        ("t", "toggle_temperature", "Filter: temperature"),
        ("c", "compare_toggle", "Compare"),
        ("C", "compare_clear", "Compare clear"),
    ]

    def __init__(self, advanced_fuzzy: bool = False):
        super().__init__()
        self._df: Optional[pd.DataFrame] = None
        self._providers: List[str] = []
        self._applied_query: str = ""
        self._table_columns: List[str] = []
        self._table_column_specs: List[Tuple[str, str]] = []
        self._current_view: Optional[pd.DataFrame] = None
        self._suppress_next_submit: bool = False
        self._advanced_fuzzy: bool = advanced_fuzzy
        self._last_filtered_count: int = 0
        self._last_total_count: int = 0
        self._last_shown_count: int = 0
        self._search_fields: List[str] = ["model_id", "model_name"]
        self._sort_keys: List[str] = [
            "provider",
            "model_id",
            "model_name",
            "context_window",
            "input_cost",
            "output_cost",
            "max_output_tokens",
        ]
        self._sort_key: Optional[str] = None
        self._sort_desc: bool = True
        self._filter_supports_attachments: bool = False
        self._filter_supports_reasoning: bool = False
        self._filter_supports_temperature: bool = False
        self._compare_rows: List[Dict[str, Any]] = []
        self._preview_enabled: bool = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar1"):
            yield Select([("(all)", "")], id="provider")
            yield Input(placeholder="Search (fuzzy) in model_id/model_name... (Ctrl+U to clear)", id="query")
        with Horizontal(id="toolbar2"):
            yield Static("", id="chips")
        yield Static("", id="palette")
        with Horizontal(id="main"):
            yield DataTable(id="table")
            with VerticalScroll(id="preview"):
                yield Static("", id="preview_text")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        pdf = _load_pdf()
        self._df = pdf
        self._providers = sorted(set([str(x) for x in pdf['provider'].dropna().tolist()]))

        provider_widget = self.query_one("#provider", Select)
        provider_options = [("(all)", "")] + [(p, p) for p in self._providers]
        if hasattr(provider_widget, "set_options"):
            provider_widget.set_options(provider_options)
        else:
            setattr(provider_widget, "options", provider_options)
        provider_widget.value = ""

        table = self.query_one("#table", DataTable)
        table.cursor_type = "row"

        self._table_columns = [
            "provider",
            "model_id",
            "input_cost",
            "output_cost",
            "context_window",
            "max_output_tokens",
            "model_name",
        ]

        # (header_label, source_column)
        label_map = {
            "input_cost": "input",
            "output_cost": "output",
        }
        self._table_column_specs = []
        for c in self._table_columns:
            if self._df is not None and c in self._df.columns:
                label = label_map.get(c, c)
                self._table_column_specs.append((label, c))
                table.add_column(label)

        self.query_one("#query", Input).focus()
        self._refresh_table()

    def _update_status(self) -> None:
        provider_value = self.query_one("#provider", Select).value or ""
        provider_label = provider_value if provider_value else "(all)"
        query_value = (self._applied_query or "").strip()
        query_label = query_value if query_value else "(none)"
        af = "on" if self._advanced_fuzzy else "off"
        search_in = "+".join([f.replace("model_", "") for f in self._search_fields]) if self._search_fields else "(none)"
        sort_label = "(none)"
        if self._sort_key:
            direction = "desc" if self._sort_desc else "asc"
            sort_label = f"{self._sort_key}:{direction}"
        filters: List[str] = []
        if self._filter_supports_attachments:
            filters.append("attachments")
        if self._filter_supports_reasoning:
            filters.append("reasoning")
        if self._filter_supports_temperature:
            filters.append("temperature")
        filter_label = ",".join(filters) if filters else "(none)"
        shown = self._last_shown_count
        filtered = self._last_filtered_count
        total = self._last_total_count
        suffix = ""
        if shown < filtered:
            suffix = f" (shown {shown}/{filtered})"
        else:
            suffix = f" (shown {shown})"
        text = f"provider={provider_label} | query={query_label} | in={search_in} | sort={sort_label} | filters={filter_label} | rows={filtered}/{total}{suffix} | advanced_fuzzy={af}"
        self.query_one("#status", Static).update(text)

        chips = f"  provider={provider_label}  in={search_in}  sort={sort_label}  filters={filter_label}  compare={len(self._compare_rows)}/2  preview={'on' if self._preview_enabled else 'off'}"
        self.query_one("#chips", Static).update(chips)

    def _update_palette(self) -> None:
        inp = self.query_one("#query", Input)
        raw = (inp.value or "").strip()
        palette = self.query_one("#palette", Static)
        if not raw.startswith("/"):
            palette.update("")
            return
        prefix = raw[1:].strip().lower()
        cmds = [
            "/help (/h)",
            "/clear (/c)",
            "/provider (/p) NAME",
            "/in id|name|both",
            "/sort KEY[:asc|:desc]",
            "/filter attachments|reasoning|temperature on|off",
            "/compare add|show|clear",
            "/af on|off",
            "/quit (/exit)",
        ]
        filtered = [c for c in cmds if c[1:].lower().startswith(prefix) or prefix == ""]
        if not filtered:
            palette.update("(no matching commands)")
            return
        palette.update("Commands:\n" + "\n".join(["  " + c for c in filtered[:8]]))

    def action_preview_toggle(self) -> None:
        self._preview_enabled = not self._preview_enabled
        preview = self.query_one("#preview")
        try:
            preview.display = self._preview_enabled
        except Exception:
            pass
        self._update_status()
        self._refresh_preview()

    def _show_help(self) -> None:
        help_text = "\n".join(
            [
                "Keyboard shortcuts:",
                "  ?            Show help",
                "  Enter        Show details (top result / cursor row)",
                "  Esc          Clear search input",
                "  r            Refresh table",
                "  Tab          Focus next widget",
                "  p            Toggle preview split view",
                "  s            Cycle sort key",
                "  S            Toggle sort order (desc/asc)",
                "  i            Cycle search target (id/name/both)",
                "  a            Toggle filter: supports_attachments",
                "  R            Toggle filter: supports_reasoning",
                "  t            Toggle filter: supports_temperature",
                "  c            Add/remove current row to compare (shows compare when 2 selected)",
                "  C            Clear compare selection",
                "",
                "Slash commands:",
                "  /help (/h)                 Show this help",
                "  /clear (/c)                Clear the search query and provider",
                "  /provider (/p) NAME        Set provider filter (empty = all)",
                "  /in id|name|both            Set search target",
                "  /sort KEY[:asc|:desc]       Set sort key (context_window,input_cost,output_cost,max_output_tokens)",
                "  /filter NAME on|off         Toggle boolean filters (attachments, reasoning, temperature)",
                "  /compare add|show|clear     Compare workflow",
                "  /quit (/exit)               Quit the TUI",
                "  /af on|off                 Toggle advanced fuzzy scoring",
                "  /advanced-fuzzy on|off     Same as /af",
                "",
                "Tips:",
                "- Typing '/' in the search box shows a command palette with matching commands.",
            ]
        )
        self.push_screen(ModelDetailScreen(title="TUI Help", content=help_text))

    def action_help(self) -> None:
        self._show_help()

    def action_escape(self) -> None:
        focused = self.focused
        if isinstance(focused, Input) and focused.id == "query":
            focused.value = ""
            self._applied_query = ""
            try:
                focused.cursor_position = 0
            except Exception:
                pass
            focused.focus()
            self._refresh_table()

    def action_refresh(self) -> None:
        self._refresh_table()

    def action_focus_next(self) -> None:
        try:
            self.screen.focus_next()
        except Exception:
            pass

    def action_sort_cycle(self) -> None:
        if not self._sort_keys:
            return
        if self._sort_key is None:
            self._sort_key = self._sort_keys[0]
        else:
            try:
                idx = self._sort_keys.index(self._sort_key)
            except ValueError:
                idx = -1
            nxt = idx + 1
            if nxt >= len(self._sort_keys):
                self._sort_key = None
            else:
                self._sort_key = self._sort_keys[nxt]
        self._refresh_table()

    def action_sort_toggle(self) -> None:
        if self._sort_key is None:
            self._sort_key = self._sort_keys[0] if self._sort_keys else None
        self._sort_desc = not self._sort_desc
        self._refresh_table()

    def action_search_in_cycle(self) -> None:
        cur = "+".join(self._search_fields)
        if cur == "model_id+model_name":
            self._search_fields = ["model_id"]
        elif cur == "model_id":
            self._search_fields = ["model_name"]
        else:
            self._search_fields = ["model_id", "model_name"]
        self._refresh_table()

    def action_toggle_attachments(self) -> None:
        self._filter_supports_attachments = not self._filter_supports_attachments
        self._refresh_table()

    def action_toggle_reasoning(self) -> None:
        self._filter_supports_reasoning = not self._filter_supports_reasoning
        self._refresh_table()

    def action_toggle_temperature(self) -> None:
        self._filter_supports_temperature = not self._filter_supports_temperature
        self._refresh_table()

    def action_compare_clear(self) -> None:
        self._compare_rows = []
        self._update_status()
        self._refresh_preview()

    def _compare_key(self, row: Dict[str, Any]) -> str:
        provider = str(row.get("provider", "") or "")
        model_id = str(row.get("model_id", "") or "")
        return f"{provider}::{model_id}"

    def action_compare_toggle(self) -> None:
        if self._current_view is None or len(self._current_view) == 0:
            return
        table = self.query_one("#table", DataTable)
        cursor = getattr(table, "cursor_row", None)
        if cursor is None:
            return
        idx = int(cursor)
        if not (0 <= idx < len(self._current_view)):
            return
        row = self._current_view.iloc[idx].to_dict()
        key = self._compare_key(row)
        existing_keys = [self._compare_key(r) for r in self._compare_rows]
        if key in existing_keys:
            self._compare_rows = [r for r in self._compare_rows if self._compare_key(r) != key]
            self._update_status()
            return
        self._compare_rows.append(row)
        if len(self._compare_rows) > 2:
            self._compare_rows = self._compare_rows[-2:]
        self._update_status()
        if len(self._compare_rows) == 2:
            self._show_compare(self._compare_rows[0], self._compare_rows[1])
        self._refresh_preview()

    def _refresh_table(self) -> None:
        if self._df is None:
            return

        provider_value = self.query_one("#provider", Select).value or ""
        query_value = self._applied_query.strip()

        pdf = self._df
        total_count = int(len(pdf))
        if provider_value:
            pdf = pdf[pdf["provider"].astype(str) == str(provider_value)]

        # Boolean capability filters (if the column exists)
        if self._filter_supports_attachments and "supports_attachments" in pdf.columns:
            pdf = pdf[pdf["supports_attachments"].apply(_truthy)]
        if self._filter_supports_reasoning and "supports_reasoning" in pdf.columns:
            pdf = pdf[pdf["supports_reasoning"].apply(_truthy)]
        if self._filter_supports_temperature and "supports_temperature" in pdf.columns:
            pdf = pdf[pdf["supports_temperature"].apply(_truthy)]

        if query_value:
            q = query_value

            def score_row(row: pd.Series) -> int:
                tier, score = _row_best_rank(q, row, list(self._search_fields or ["model_id", "model_name"]), self._advanced_fuzzy)
                # Encode tier into the score so that lower tier always wins.
                # Larger is better for sort desc.
                return (1000 - min(tier, 99)) * 1000 + score

            scored = pdf.copy()
            scored["_score"] = scored.apply(score_row, axis=1)
            scored = scored.sort_values("_score", ascending=False)
            scored = scored[scored["_score"] > 0]
            pdf = scored.drop(columns=["_score"], errors="ignore")

        # Sorting (after filtering/search)
        if self._sort_key and self._sort_key in pdf.columns:
            try:
                pdf = pdf.sort_values(self._sort_key, ascending=not self._sort_desc, na_position="last")
            except Exception:
                pass

        filtered_count = int(len(pdf))

        table = self.query_one("#table", DataTable)
        table.clear()

        specs = [(label, src) for (label, src) in getattr(self, "_table_column_specs", []) if src in pdf.columns]
        existing_sources = [src for (_, src) in specs]
        max_rows = 200
        view = pdf[existing_sources].head(max_rows)
        self._current_view = view
        self._last_total_count = total_count
        self._last_filtered_count = filtered_count
        self._last_shown_count = int(len(view))
        for _, row in view.iterrows():
            rendered: List[str] = []
            for _, src in specs:
                val = row[src]
                if src in {"context_window", "max_output_tokens"}:
                    rendered.append(format_int_with_commas(val))
                elif src in {"input_cost", "output_cost", "cost_cache_read_per_million"}:
                    rendered.append(format_cost_fixed(val, decimals=4))
                else:
                    rendered.append("" if pd.isna(val) else str(val))
            table.add_row(*rendered)

        self._update_status()

        # Keep preview in sync with the current view
        self._refresh_preview()

    def _render_row_summary(self, row: Dict[str, Any]) -> str:
        provider = str(row.get("provider", "") or "")
        model_id = str(row.get("model_id", "") or "")
        model_name = str(row.get("model_name", "") or "")
        ctx = format_int_with_commas(row.get("context_window", None))
        max_out = format_int_with_commas(row.get("max_output_tokens", None))
        input_cost = format_cost_fixed(row.get("input_cost", None), decimals=4)
        output_cost = format_cost_fixed(row.get("output_cost", None), decimals=4)
        cache_read = format_cost_fixed(row.get("cost_cache_read_per_million", None), decimals=4)

        lines = [
            "Preview",
            "-------",
            f"provider: {provider}",
            f"model_id: {model_id}",
            f"model_name: {model_name}",
            f"context_window: {ctx}",
            f"max_output_tokens: {max_out}",
            f"input_cost: {input_cost}",
            f"output_cost: {output_cost}",
            f"cache_read_per_million: {cache_read}",
            "",
            "Tips",
            "----",
            "- Enter: details",
            "- c: compare select",
            "- p: toggle preview",
        ]
        return "\n".join(lines)

    def _refresh_preview(self) -> None:
        if not getattr(self, "_preview_enabled", False):
            return

        try:
            widget = self.query_one("#preview_text", Static)
        except Exception:
            return

        if self._current_view is None or len(self._current_view) == 0:
            widget.update("(no rows)")
            return

        idx = 0
        try:
            table = self.query_one("#table", DataTable)
            cursor = getattr(table, "cursor_row", None)
            if cursor is not None:
                idx = int(cursor)
        except Exception:
            idx = 0

        if not (0 <= idx < len(self._current_view)):
            idx = 0

        row = self._current_view.iloc[idx].to_dict()
        widget.update(self._render_row_summary(row))

    def _show_details_for_row(self, row: pd.Series) -> None:
        provider = str(row.get('provider', '') or '')
        model_id = str(row.get('model_id', '') or '')
        model_name = str(row.get('model_name', '') or '')
        title = f"{provider} / {model_id}"

        input_cost = format_cost_fixed(row.get('input_cost', None), decimals=4)
        output_cost = format_cost_fixed(row.get('output_cost', None), decimals=4)
        cache_read = format_cost_fixed(row.get('cost_cache_read_per_million', None), decimals=4)
        ctx = format_int_with_commas(row.get('context_window', None))
        max_out = format_int_with_commas(row.get('max_output_tokens', None))

        summary_lines = [
            "Summary",
            "-------",
            f"provider: {provider}",
            f"model_id: {model_id}",
            f"model_name: {model_name}",
            f"context_window: {ctx}",
            f"max_output_tokens: {max_out}",
            f"input_cost: {input_cost}",
            f"output_cost: {output_cost}",
            f"cost_cache_read_per_million: {cache_read}",
        ]
        raw_json = json.dumps(row.to_dict(), indent=2, ensure_ascii=False, default=str)
        self.push_screen(ModelDetailScreen(title=title, content="\n".join(summary_lines), right_content=raw_json))

    def _show_compare(self, left: Dict[str, Any], right: Dict[str, Any]) -> None:
        def _fmt(v: Any) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float) and pd.isna(v):
                return "N/A"
            return str(v)

        left_key = self._compare_key(left)
        right_key = self._compare_key(right)

        fields = [
            "provider",
            "model_id",
            "model_name",
            "context_window",
            "max_output_tokens",
            "input_cost",
            "output_cost",
            "cost_cache_read_per_million",
            "supports_attachments",
            "supports_reasoning",
            "supports_temperature",
        ]
        lines: List[str] = [
            "Compare",
            "-------",
            f"left:  {left_key}",
            f"right: {right_key}",
            "",
            "Differences",
            "-----------",
        ]
        any_diff = False
        for f in fields:
            lv = left.get(f, None)
            rv = right.get(f, None)
            if f in {"context_window", "max_output_tokens"}:
                lvs = format_int_with_commas(lv)
                rvs = format_int_with_commas(rv)
            elif f in {"input_cost", "output_cost", "cost_cache_read_per_million"}:
                lvs = format_cost_fixed(lv, decimals=4)
                rvs = format_cost_fixed(rv, decimals=4)
            else:
                lvs = _fmt(lv)
                rvs = _fmt(rv)
            if lvs != rvs:
                any_diff = True
                lines.append(f"{f}:\n  left : {lvs}\n  right: {rvs}")
        if not any_diff:
            lines.append("(no differences in the compared fields)")

        left_json = json.dumps(left, indent=2, ensure_ascii=False, default=str)
        right_json = json.dumps(right, indent=2, ensure_ascii=False, default=str)
        right_panel = "\n".join(["Raw JSON (left)", "--------------", left_json, "", "Raw JSON (right)", "---------------", right_json])
        self.push_screen(ModelDetailScreen(title="Compare", content="\n".join(lines), right_content=right_panel))

    def _handle_slash_command(self, text: str) -> bool:
        raw = text.strip()
        if not raw.startswith('/'):
            return False

        def _clear_query_input() -> None:
            input_widget = self.query_one("#query", Input)
            input_widget.value = ""
            self._applied_query = ""
            try:
                input_widget.cursor_position = 0
            except Exception:
                pass
            input_widget.focus()

        parts = raw[1:].strip().split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:]

        # Aliases
        if cmd == "p":
            cmd = "provider"
        elif cmd == "c":
            cmd = "clear"
        elif cmd == "h":
            cmd = "help"
        elif cmd == "q":
            cmd = "quit"
        elif cmd in {"af", "advanced-fuzzy"}:
            cmd = "advanced_fuzzy"

        if cmd in {"help", "h", "?"}:
            self._show_help()
            _clear_query_input()
            return True

        if cmd in {"quit", "exit"}:
            self.exit()
            return True

        if cmd == "clear":
            self._applied_query = ""
            input_widget = self.query_one("#query", Input)
            input_widget.value = ""
            try:
                input_widget.cursor_position = 0
            except Exception:
                pass
            input_widget.focus()
            provider_widget = self.query_one("#provider", Select)
            provider_widget.value = ""
            self._refresh_table()
            return True

        if cmd == "provider":
            name = " ".join(args).strip() if args else ""
            provider_widget = self.query_one("#provider", Select)
            # Empty means '(all)'
            if not name:
                # Defer screen push so the Enter key event that triggered the slash command
                # doesn't interfere with the modal's first render.
                picker = ProviderPickerScreen(self._providers)
                try:
                    self.call_after_refresh(lambda: self.push_screen(picker))
                except Exception:
                    try:
                        self.call_later(lambda: self.push_screen(picker))
                    except Exception:
                        self.push_screen(picker)
                return True

            # Exact match first, then case-insensitive match
            if name in self._providers:
                provider_widget.value = name
                self._refresh_table()
                _clear_query_input()
                return True
            lowered = {p.lower(): p for p in self._providers}
            if name.lower() in lowered:
                provider_widget.value = lowered[name.lower()]
                self._refresh_table()
                _clear_query_input()
                return True

            self.push_screen(
                ModelDetailScreen(
                    title="Unknown provider",
                    content=f"Provider '{name}' not found. Try: /provider openai",
                )
            )
            _clear_query_input()
            return True

        if cmd == "in":
            value = (args[0].lower().strip() if args else "").strip()
            if value in {"id", "model_id"}:
                self._search_fields = ["model_id"]
                self._refresh_table()
                _clear_query_input()
                return True
            if value in {"name", "model_name"}:
                self._search_fields = ["model_name"]
                self._refresh_table()
                _clear_query_input()
                return True
            if value in {"both", "id+name", "all"}:
                self._search_fields = ["model_id", "model_name"]
                self._refresh_table()
                _clear_query_input()
                return True
            self.push_screen(ModelDetailScreen(title="Usage", content="/in id|name|both"))
            _clear_query_input()
            return True

        if cmd == "sort":
            value = (" ".join(args).strip() if args else "").strip()
            if not value:
                self._sort_key = None
                self._refresh_table()
                _clear_query_input()
                return True
            key = value
            order = None
            if ":" in value:
                key, order = value.split(":", 1)
                key = key.strip()
                order = order.strip().lower()
            allowed = set(self._sort_keys)
            if self._df is not None:
                try:
                    allowed |= set([str(c) for c in self._df.columns])
                except Exception:
                    pass
            if key not in allowed:
                self.push_screen(
                    ModelDetailScreen(
                        title="Unknown sort key",
                        content=f"Unknown sort key: {key}\nSupported: {', '.join(self._sort_keys)}",
                    )
                )
                _clear_query_input()
                return True
            self._sort_key = key
            if order in {"asc", "desc"}:
                self._sort_desc = order == "desc"
            self._refresh_table()
            _clear_query_input()
            return True

        if cmd == "filter":
            name = (args[0].lower().strip() if args else "").strip()
            value = (args[1].lower().strip() if len(args) > 1 else "").strip()
            if name in {"attachments", "attachment"}:
                if value in {"on", "true", "1", "yes"}:
                    self._filter_supports_attachments = True
                elif value in {"off", "false", "0", "no"}:
                    self._filter_supports_attachments = False
                else:
                    self._filter_supports_attachments = not self._filter_supports_attachments
                self._refresh_table()
                _clear_query_input()
                return True
            if name in {"reasoning", "reason"}:
                if value in {"on", "true", "1", "yes"}:
                    self._filter_supports_reasoning = True
                elif value in {"off", "false", "0", "no"}:
                    self._filter_supports_reasoning = False
                else:
                    self._filter_supports_reasoning = not self._filter_supports_reasoning
                self._refresh_table()
                _clear_query_input()
                return True
            if name in {"temperature", "temp"}:
                if value in {"on", "true", "1", "yes"}:
                    self._filter_supports_temperature = True
                elif value in {"off", "false", "0", "no"}:
                    self._filter_supports_temperature = False
                else:
                    self._filter_supports_temperature = not self._filter_supports_temperature
                self._refresh_table()
                _clear_query_input()
                return True
            self.push_screen(
                ModelDetailScreen(
                    title="Usage",
                    content="/filter attachments on|off\n/filter reasoning on|off\n/filter temperature on|off",
                )
            )
            _clear_query_input()
            return True

        if cmd == "compare":
            sub = (args[0].lower().strip() if args else "show").strip()
            if sub in {"clear", "reset"}:
                self._compare_rows = []
                self._update_status()
                _clear_query_input()
                return True
            if sub in {"show"}:
                if len(self._compare_rows) == 2:
                    self._show_compare(self._compare_rows[0], self._compare_rows[1])
                else:
                    self.push_screen(
                        ModelDetailScreen(
                            title="Compare",
                            content=f"Selected for compare: {len(self._compare_rows)}\nUse 'c' on two rows (table) or /compare add",
                        )
                    )
                _clear_query_input()
                return True
            if sub in {"add"}:
                self.action_compare_toggle()
                _clear_query_input()
                return True
            self.push_screen(ModelDetailScreen(title="Usage", content="/compare add|show|clear"))
            _clear_query_input()
            return True

        if cmd == "advanced_fuzzy":
            value = (args[0].lower().strip() if args else "").strip()
            if value in {"on", "true", "1", "yes"}:
                self._advanced_fuzzy = True
                self._refresh_table()
                _clear_query_input()
                return True
            if value in {"off", "false", "0", "no"}:
                self._advanced_fuzzy = False
                self._refresh_table()
                _clear_query_input()
                return True

            self.push_screen(
                ModelDetailScreen(
                    title="Usage",
                    content="/af on|off\n/advanced-fuzzy on|off",
                )
            )
            _clear_query_input()
            return True

        self.push_screen(
            ModelDetailScreen(
                title="Unknown command",
                content=f"Unknown command: {raw}\nTry /help",
            )
        )
        _clear_query_input()
        return True

    def _show_details_top_result(self) -> None:
        if self._current_view is None or len(self._current_view) == 0:
            return
        self._show_details_for_row(self._current_view.iloc[0])

    def _show_details_cursor_row(self) -> None:
        if self._current_view is None or len(self._current_view) == 0:
            return
        table = self.query_one("#table", DataTable)
        cursor = getattr(table, "cursor_row", None)
        if cursor is None:
            return
        if 0 <= int(cursor) < len(self._current_view):
            self._show_details_for_row(self._current_view.iloc[int(cursor)])

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider":
            self._refresh_table()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "query":
            self._applied_query = event.value
            self._refresh_table()
            self._update_palette()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._refresh_preview()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query":
            if self._suppress_next_submit:
                self._suppress_next_submit = False
                return
            if self._handle_slash_command(event.value):
                return
            self._show_details_top_result()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._show_details_cursor_row()

    def on_key(self, event) -> None:
        if getattr(event, "key", None) == "enter":
            focused = self.focused
            if isinstance(focused, Input) and focused.id == "query":
                if self._handle_slash_command(focused.value):
                    # Some Textual versions may emit both Key and Submitted events.
                    # Suppress the subsequent submit handler to avoid opening details.
                    self._suppress_next_submit = True
                    return
                self._show_details_top_result()
                return
            if isinstance(focused, DataTable) and focused.id == "table":
                self._show_details_cursor_row()
                return


@app.command("tui")
def tui(
    advanced_fuzzy: bool = typer.Option(False, "--advanced-fuzzy/--no-advanced-fuzzy", help="Use advanced fuzzy scoring for search (field-specific)."),
):
    """Launch the interactive TUI (Textual)."""
    ModelsTUI(advanced_fuzzy=advanced_fuzzy).run()


@app.command("providers")
def providers(
    search: Optional[str] = typer.Option(None, "--search", help="Filter providers by substring (case-insensitive)."),
    count: bool = typer.Option(True, "--count/--no-count", help="Show model counts per provider."),
    format: str = typer.Option(
        "table",
        "--format",
        click_type=click.Choice(["comma", "lines", "table", "json"], case_sensitive=False),
        help="Output format: comma, lines, table, json.",
    ),
    style: str = typer.Option(
        "simple",
        "--style",
        click_type=click.Choice(STYLE_CHOICES, case_sensitive=False),
        help="Rich table style (only used when --format table).",
    ),
):
    """List all providers."""
    pdf = _load_pdf()
    providers_list = sorted(set([str(x) for x in pdf['provider'].dropna().tolist()]))

    if search:
        s = search.lower()
        providers_list = [p for p in providers_list if s in p.lower()]

    counts: Dict[str, int] = {}
    if count:
        counts = pdf['provider'].value_counts().to_dict()

    fmt = format.strip().lower()
    if fmt == "comma":
        if count:
            rendered = [f"{p}:{counts.get(p, 0)}" for p in providers_list]
            typer.echo(",".join(rendered))
        else:
            typer.echo(",".join(providers_list))
        return

    if fmt == "lines":
        if count:
            for p in providers_list:
                typer.echo(f"{p}\t{counts.get(p, 0)}")
        else:
            for p in providers_list:
                typer.echo(p)
        return

    if fmt == "json":
        if count:
            payload = [{"provider": p, "count": int(counts.get(p, 0))} for p in providers_list]
        else:
            payload = providers_list
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if fmt == "table":
        console = Console()
        resolved_box = resolve_rich_table_box(style)
        if resolved_box is not None:
            table = Table(title="Providers", show_header=True, header_style="bold magenta", box=resolved_box)
        else:
            table = Table(title="Providers", show_header=True, header_style="bold magenta")

        table.add_column("provider")
        if count:
            table.add_column("count", justify="right")

        for p in providers_list:
            if count:
                table.add_row(p, str(counts.get(p, 0)))
            else:
                table.add_row(p)

        console.print(table)
        return

    raise typer.BadParameter("--format must be one of: comma, lines, table, json")


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query for model_id/model_name."),
    in_: str = typer.Option("both", "--in", help="Search field: id|name|both."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Restrict to a provider (exact match)."),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of rows to display."),
    column: Optional[List[str]] = typer.Option(None, "--column", "-c", help="Column(s) to display."),
    all_columns: bool = typer.Option(False, "--all-columns", help="Show all columns in the output."),
    filters: Optional[List[str]] = typer.Option(None, "--filter", "-f", help="Additional filters to apply."),
    advanced_fuzzy: bool = typer.Option(False, "--advanced-fuzzy/--no-advanced-fuzzy", help="Use advanced fuzzy scoring for search (field-specific)."),
    style: str = typer.Option(
        "simple",
        "--style",
        click_type=click.Choice(STYLE_CHOICES, case_sensitive=False),
        help="Rich table style: simple, rounded, minimal, square, ascii.",
    ),
):
    """Fuzzy search models by model_id/model_name (sorted by fuzzy score desc)."""
    pdf = _load_pdf()
    if provider:
        pdf = pdf[pdf['provider'].astype(str) == str(provider)]

    # Apply extra filters using the same filter syntax by temporarily converting to fd.DataFrame
    if filters:
        df_tmp = fd.DataFrame(pdf)
        df_tmp = _apply_filters(df_tmp, filters)
        pdf = df_tmp.to_pandas() if hasattr(df_tmp, 'to_pandas') else pd.DataFrame(df_tmp)
    q = query.strip()
    if not q:
        raise typer.BadParameter("query must be non-empty")

    in_norm = in_.lower().strip()
    if in_norm not in {"id", "name", "both"}:
        raise typer.BadParameter("--in must be one of: id, name, both")

    def score_row(row: pd.Series) -> int:
        fields: List[str] = []
        if in_norm in {"id", "both"}:
            fields.append("model_id")
        if in_norm in {"name", "both"}:
            fields.append("model_name")
        tier, score = _row_best_rank(q, row, fields, advanced_fuzzy)
        return (1000 - min(tier, 99)) * 1000 + score

    pdf = pdf.copy()
    pdf['_score'] = pdf.apply(score_row, axis=1)
    pdf = pdf.sort_values('_score', ascending=False)
    pdf = pdf[pdf['_score'] > 0]

    # back to display columns
    display_pdf = pdf.drop(columns=['_score'], errors='ignore')
    display_df = fd.DataFrame(display_pdf)
    try:
        setattr(display_df, "_rich_table_style", style)
    except Exception:
        pass

    columns = _select_columns(display_df, column, all_columns)
    _render_table(display_df, columns, limit, title_prefix="Search Results")


@app.command("list")
def list_cmd(
    column: Optional[List[str]] = typer.Option(None, "--column", "-c", help="Column(s) to display."),
    filters: Optional[List[str]] = typer.Option(None, "--filter", "-f", help="Filter conditions."),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of rows to display."),
    all_columns: bool = typer.Option(False, "--all-columns", help="Show all columns in the output."),
    sort: Optional[str] = typer.Option(None, "--sort", "-s", help='Sort by column (e.g., "context_window:desc").'),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="List models for a provider (exact match by default)."),
    provider_partial: bool = typer.Option(False, "--provider-partial", help="Use partial provider match (provider~=...)."),
    style: str = typer.Option(
        "simple",
        "--style",
        click_type=click.Choice(STYLE_CHOICES, case_sensitive=False),
        help="Rich table style: simple, rounded, minimal, square, ascii.",
    ),
):
    """List models (same as default command)."""
    _list_models(
        column=column,
        filters=filters,
        limit=limit,
        all_columns=all_columns,
        sort=sort,
        provider=provider,
        provider_partial=provider_partial,
        style=style,
    )


@app.command("provider")
def provider_cmd(
    provider: str = typer.Argument(..., help="Provider name."),
    partial: bool = typer.Option(True, "--partial/--exact", help="Use partial match by default."),
    column: Optional[List[str]] = typer.Option(None, "--column", "-c", help="Column(s) to display."),
    filters: Optional[List[str]] = typer.Option(None, "--filter", "-f", help="Additional filters."),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of rows to display."),
    all_columns: bool = typer.Option(False, "--all-columns", help="Show all columns in the output."),
    sort: Optional[str] = typer.Option(None, "--sort", "-s", help='Sort by column (e.g., "context_window:desc").'),
    style: str = typer.Option(
        "simple",
        "--style",
        click_type=click.Choice(STYLE_CHOICES, case_sensitive=False),
        help="Rich table style: simple, rounded, minimal, square, ascii.",
    ),
):
    """List models for a given provider (shortcut for list --provider...)."""
    _list_models(
        column=column,
        filters=filters,
        limit=limit,
        all_columns=all_columns,
        sort=sort,
        provider=provider,
        provider_partial=partial,
        style=style,
    )


def cli():
    """CLI entry point."""
    app()

if __name__ == "__main__":
    cli()