import pandas as pd

from src.preprocessor import DataPreprocessor


def test_clean_columns_headers():
    df = pd.DataFrame({"Monto $$": [10], "Nombre Cliente": ["Ana"]})
    prep = DataPreprocessor()
    df_clean = prep.clean_columns(df)

    assert "monto" in df_clean.columns
    assert "nombre_cliente" in df_clean.columns

    for col in df_clean.columns:
        assert col == col.lower()
        assert " " not in col


def test_add_null_flags_creates_flag_column():
    df = pd.DataFrame({"monto": [1, None]})
    prep = DataPreprocessor()
    df_flag = prep.add_null_flags(df)

    assert "monto_nan" in df_flag.columns
    assert df_flag.loc[0, "monto_nan"] == 0
    assert df_flag.loc[1, "monto_nan"] == 1


def test_quality_gate_fails_if_critical_nulls_exceed_threshold():
    df = pd.DataFrame({
        "monto": [None, None, 10, None, None],
        "es_fraude": [1, None, 0, None, None],
    })

    prep = DataPreprocessor()
    ok = prep.analyze_quality(df, umbral=0.30)

    assert ok is False
