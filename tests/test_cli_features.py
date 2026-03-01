import sys
from types import SimpleNamespace

import pandas as pd

import json
from pathlib import Path
from typer.testing import CliRunner


def _import_main():
    from models import main as m

    return m


def test_apply_filters_supports_gte_lte_ne():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "openai", "context_window": 100, "model_id": "a", "model_name": "A"},
            {"provider": "openai", "context_window": 200, "model_id": "b", "model_name": "B"},
            {"provider": "google", "context_window": 300, "model_id": "c", "model_name": "C"},
        ]
    )

    df = m.fd.DataFrame(pdf)
    out = m._apply_filters(df, ["provider=openai", "context_window>=200"])

    out_pdf = out.to_pandas() if hasattr(out, "to_pandas") else pd.DataFrame(out)
    assert set(out_pdf["model_id"].tolist()) == {"b"}

    out2 = m._apply_filters(df, ["provider!=openai"])
    out2_pdf = out2.to_pandas() if hasattr(out2, "to_pandas") else pd.DataFrame(out2)
    assert set(out2_pdf["provider"].tolist()) == {"google"}


def test_apply_filters_allows_operator_whitespace_and_eqeq():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "openai", "context_window": 100, "model_id": "gpt-4", "model_name": "GPT-4"},
            {"provider": "openai", "context_window": 200, "model_id": "gpt-4.1", "model_name": "GPT-4.1"},
            {"provider": "google", "context_window": 300, "model_id": "gemini", "model_name": "Gemini"},
        ]
    )

    df = m.fd.DataFrame(pdf)
    out = m._apply_filters(df, ["provider == openai", "context_window >= 200"])
    out_pdf = out.to_pandas() if hasattr(out, "to_pandas") else pd.DataFrame(out)
    assert set(out_pdf["model_id"].tolist()) == {"gpt-4.1"}


def test_multi_provider_filter_union_semantics():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "openrouter", "model_id": "a", "model_name": "A", "context_window": 100},
            {"provider": "google", "model_id": "b", "model_name": "B", "context_window": 200},
            {"provider": "openai", "model_id": "c", "model_name": "C", "context_window": 300},
        ]
    )

    # mirror CLI logic: exact-match filter on provider list
    providers = ["openrouter", "google"]
    lowered = {p.lower() for p in providers}
    filtered = pdf[pdf["provider"].astype(str).str.lower().isin(lowered)]
    assert set(filtered["provider"].tolist()) == {"openrouter", "google"}


def test_parse_provider_values_supports_commas_and_dedup():
    m = _import_main()
    assert m._parse_provider_values(["openrouter,google"]) == ["openrouter", "google"]
    assert m._parse_provider_values(["openrouter, google", "OpenRouter"]) == ["openrouter", "google"]


def test_parse_column_values_supports_commas_and_dedup():
    m = _import_main()
    assert m._parse_column_values(["provider,model_id"]) == ["provider", "model_id"]
    assert m._parse_column_values(["provider, model_id", "provider"]) == ["provider", "model_id"]


def test_filter_pdf_by_providers_partial_match():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "openrouter", "model_id": "a"},
            {"provider": "openai", "model_id": "b"},
            {"provider": "google", "model_id": "c"},
        ]
    )
    out = m._filter_pdf_by_providers(pdf, ["open"], partial=True)
    assert set(out["provider"].tolist()) == {"openrouter", "openai"}


def test_column_alias_resolution_in_filter_and_sort():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "openai", "context_window": 100, "model_id": "b", "model_name": "B"},
            {"provider": "openai", "context_window": 200, "model_id": "a", "model_name": "A"},
        ]
    )
    df = m.fd.DataFrame(pdf)

    # alias: id -> model_id
    out = m._apply_filters(df, ["id ~= a"])
    out_pdf = out.to_pandas() if hasattr(out, "to_pandas") else pd.DataFrame(out)
    assert out_pdf["model_id"].tolist() == ["a"]

    # alias: context -> context_window
    sorted_df = m._apply_sort(df, "context:desc")
    sorted_pdf = sorted_df.to_pandas() if hasattr(sorted_df, "to_pandas") else pd.DataFrame(sorted_df)
    assert sorted_pdf["context_window"].tolist() == [200, 100]


def test_column_alias_prefix_suffix_normalization_updated_release():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {
                "model_id": "a",
                "model_name": "A",
                "provider": "p",
                "last_updated": "2025-01-01",
                "release_date": "2024-01-01",
            },
            {
                "model_id": "b",
                "model_name": "B",
                "provider": "p",
                "last_updated": "2026-01-01",
                "release_date": "2023-01-01",
            },
        ]
    )
    df = m.fd.DataFrame(pdf)

    # updated_at -> last_updated
    sorted_df = m._apply_sort(df, "updated_at:desc")
    sorted_pdf = sorted_df.to_pandas() if hasattr(sorted_df, "to_pandas") else pd.DataFrame(sorted_df)
    assert sorted_pdf["model_id"].tolist() == ["b", "a"]

    # released_at -> release_date (ascending default)
    sorted_df2 = m._apply_sort(df, "released_at")
    sorted_pdf2 = sorted_df2.to_pandas() if hasattr(sorted_df2, "to_pandas") else pd.DataFrame(sorted_df2)
    assert sorted_pdf2["release_date"].tolist() == ["2023-01-01", "2024-01-01"]

    # Column spec expansion should preserve canonical column names
    cols = m._select_columns(df, ["updated_at", "released"], add_column=None, all_columns=False)
    assert cols == ["last_updated", "release_date"]


def test_add_column_appends_to_default_output_and_allows_hidden_nested_cols():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {
                "provider": "p",
                "model_id": "a",
                "model_name": "A",
                "input_cost": 1.0,
                "output_cost": 2.0,
                "context_window": 100,
                "max_output_tokens": 10,
                "last_updated": "2025-01-01",
                "modalities__input": "[\"text\"]",
            }
        ]
    )
    df = m.fd.DataFrame(pdf)

    # Default columns + add updated (alias) and a hidden nested column
    cols = m._select_columns(df, column=None, add_column=["updated", "modalities__input"], all_columns=False)
    # Must start with defaults
    assert cols[: len(m.get_default_display_columns())] == m.get_default_display_columns()
    assert "last_updated" in cols
    assert "modalities__input" in cols

    # If --column is used, it still replaces the full set (add_column appends to that set)
    cols2 = m._select_columns(df, column=["model"], add_column=["updated"], all_columns=False)
    assert cols2 == ["model_id", "model_name", "last_updated"]


def test_normalize_query_llm_naming_separators_and_digits():
    m = _import_main()
    assert m._normalize_query("google/gemini-3-flash") == "google gemini 3 flash"
    assert m._normalize_query("Gemini3Flash") == "gemini 3 flash"
    assert m._normalize_query("gemini3 flash") == "gemini 3 flash"


def test_match_rank_penalizes_numeric_only_accidental_hits():
    m = _import_main()
    # Query has alphabetic tokens gemini/flash. Candidate only shares the numeric token.
    tier, score = m._match_rank_for_field("gemini3 flash", "Intellect 3", "model_name", advanced_fuzzy=False)
    assert tier == 4
    assert score == 0


def test_match_rank_weights_primary_and_version_over_flash_only():
    m = _import_main()
    # Candidate with primary token (gemini) + version should outrank flash-only candidates.
    _, score_gemini31 = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-3.1-pro-preview",
        "model_id",
        advanced_fuzzy=False,
    )
    _, score_flash_only = m._match_rank_for_field(
        "gemini3 flash",
        "stepfun/step-3.5-flash:free",
        "model_id",
        advanced_fuzzy=False,
    )
    assert score_gemini31 > score_flash_only


def test_match_rank_version_proximity_bonus_orders_versions():
    m = _import_main()
    # Same primary token + same variant; only version differs.
    _, score_v3 = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-3-flash-preview",
        "model_id",
        advanced_fuzzy=False,
    )
    _, score_v31 = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-3.1-flash-preview",
        "model_id",
        advanced_fuzzy=False,
    )
    _, score_v25 = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-2.5-flash-preview",
        "model_id",
        advanced_fuzzy=False,
    )

    assert score_v3 >= score_v31
    assert score_v31 >= score_v25


def test_match_rank_variant_preference_flash_penalizes_non_flash():
    m = _import_main()
    # Query explicitly specifies the variant token 'flash'.
    _, score_flash = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-3-flash-preview",
        "model_id",
        advanced_fuzzy=False,
    )
    _, score_pro = m._match_rank_for_field(
        "gemini3 flash",
        "google/gemini-3.1-pro-preview",
        "model_id",
        advanced_fuzzy=False,
    )
    assert score_flash > score_pro


def test_cli_default_appends_list_subcommand(monkeypatch):
    m = _import_main()

    # emulate `models` with no args
    monkeypatch.setattr(sys, "argv", ["models"])

    called = SimpleNamespace(value=False)

    def fake_app():
        called.value = True

    monkeypatch.setattr(m, "app", fake_app)
    m.cli()

    assert called.value is True


def test_fetch_data_uses_cache_when_fresh(monkeypatch, tmp_path):
    m = _import_main()

    cache_path = tmp_path / "api.json"
    cache_path.write_text(json.dumps({"cached": True}), encoding="utf-8")

    monkeypatch.setenv("MODELS_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("MODELS_CACHE_TTL_SECONDS", "86400")

    def fail_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called when cache is fresh")

    monkeypatch.setattr(m.requests, "get", fail_get)
    fetcher = m.ModelDataFetcher()
    out = fetcher.fetch_data()
    assert out == {"cached": True}


def test_cli_show_outputs_json_for_model_id(monkeypatch):
    m = _import_main()

    fake_raw = {
        "google": {
            "models": {
                "gemini": {"name": "Gemini", "family": "llama", "attachment": False},
            }
        }
    }

    monkeypatch.setattr(m.ModelDataFetcher, "fetch_data", lambda self: fake_raw)

    runner = CliRunner()
    result = runner.invoke(m.app, ["show", "gemini", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["provider"] == "google"
    assert payload["model_id"] == "gemini"
    assert payload["model"]["family"] == "llama"


def test_cli_show_requires_provider_when_ambiguous(monkeypatch):
    m = _import_main()

    fake_raw = {
        "google": {"models": {"x": {"name": "X"}}},
        "openai": {"models": {"x": {"name": "X2"}}},
    }

    monkeypatch.setattr(m.ModelDataFetcher, "fetch_data", lambda self: fake_raw)

    runner = CliRunner()
    result = runner.invoke(m.app, ["show", "x"])
    assert result.exit_code != 0
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "") + (result.output or "")
    assert "exists in multiple providers" in combined


def test_cli_search_min_score_filters_low_score_rows():
    m = _import_main()
    pdf = pd.DataFrame(
        [
            {"provider": "google", "model_id": "gemini-3-flash", "model_name": "Gemini 3 Flash"},
            {"provider": "google", "model_id": "totally-unrelated", "model_name": "Something Else"},
        ]
    )

    q = "gemini3 flash"
    fields = ["model_id", "model_name"]

    def base_score(row: pd.Series) -> int:
        _, s = m._row_best_rank(q, row, fields, advanced_fuzzy=False)
        return int(s)

    scores = pdf.apply(base_score, axis=1)
    # sanity: the relevant row should score higher than the unrelated row
    assert scores.iloc[0] > scores.iloc[1]

    # With a high min_score, only the relevant row remains
    min_score = max(0, int(scores.iloc[0]))
    filtered = pdf[scores >= min_score]
    assert set(filtered["model_id"].tolist()) == {"gemini-3-flash"}


def test_limit_zero_means_no_limit(monkeypatch):
    m = _import_main()

    df = m.fd.DataFrame(
        pd.DataFrame(
            [
                {"provider": "a", "model_id": "1", "model_name": "One"},
                {"provider": "b", "model_id": "2", "model_name": "Two"},
            ]
        )
    )
    columns = ["provider", "model_id", "model_name"]

    captured = {}

    def fake_display_results(data, cols, title, style=None):
        captured["data"] = data
        captured["cols"] = cols
        captured["title"] = title

    monkeypatch.setattr(m, "display_results", fake_display_results)
    m._render_table(df, columns, limit=0, title_prefix="Test")
    assert len(captured["data"]) == 2


def test_style_plain_is_supported_and_renders_without_box_chars():
    m = _import_main()
    assert "plain" in m.STYLE_CHOICES

    captured = {}

    real_table = m.Table

    def fake_table(*args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return real_table(*args, **kwargs)

    # display_results imports Table at module scope, so patching m.Table is enough.
    m.Table = fake_table
    try:
        m.display_results(
            data=[{"provider": "a", "model_id": "1"}],
            columns=["provider", "model_id"],
            title="T",
            style="plain",
        )
    finally:
        m.Table = real_table

    assert captured["kwargs"].get("box") is None
    assert captured["kwargs"].get("show_edge") is False


def test_fetch_data_ttl_zero_forces_remote(monkeypatch, tmp_path):
    m = _import_main()

    cache_path = tmp_path / "api.json"
    cache_path.write_text(json.dumps({"cached": True}), encoding="utf-8")

    monkeypatch.setenv("MODELS_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("MODELS_CACHE_TTL_SECONDS", "0")

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"remote": True}

    monkeypatch.setattr(m.requests, "get", lambda *a, **k: FakeResp())
    fetcher = m.ModelDataFetcher()
    out = fetcher.fetch_data()
    assert out == {"remote": True}


def test_fetch_data_falls_back_to_stale_cache_on_error(monkeypatch, tmp_path):
    m = _import_main()

    cache_path = tmp_path / "api.json"
    cache_path.write_text(json.dumps({"cached": True}), encoding="utf-8")

    monkeypatch.setenv("MODELS_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("MODELS_CACHE_TTL_SECONDS", "0")

    def raise_req(*args, **kwargs):
        raise m.requests.RequestException("boom")

    monkeypatch.setattr(m.requests, "get", raise_req)
    fetcher = m.ModelDataFetcher()
    out = fetcher.fetch_data()
    assert out == {"cached": True}


def test_truthy_parsing():
    m = _import_main()
    assert m._truthy(True) is True
    assert m._truthy(False) is False
    assert m._truthy(1) is True
    assert m._truthy(0) is False
    assert m._truthy("true") is True
    assert m._truthy("FALSE") is False
    assert m._truthy("yes") is True
    assert m._truthy("off") is False
    assert m._truthy(None) is False


def test_format_cost_fixed():
    m = _import_main()
    assert m.format_cost_fixed(None) == "N/A"
    assert m.format_cost_fixed(0.1) == "0.1000"
    assert m.format_cost_fixed("0.25") == "0.2500"
    assert m.format_cost_fixed(1, decimals=2) == "1.00"
