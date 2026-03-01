import sys
from types import SimpleNamespace

import pandas as pd


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
    assert sys.argv[1] == "list"


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
