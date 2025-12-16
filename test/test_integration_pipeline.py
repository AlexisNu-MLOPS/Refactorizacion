import pandas as pd

from src.preprocessor import DataPreprocessor


def test_full_pipeline_runs_end_to_end():
    df = pd.DataFrame({
        "Monto $$": [10, None, 5],
        "Es Fraude?": [1, 0, None],
        "Score Cliente": [800, None, 650],
    })

    prep = DataPreprocessor()

    df = prep.clean_columns(df)
    df = prep.add_null_flags(df)

    assert "monto_nan" in df.columns
    assert "es_fraude_nan" in df.columns

    decision = prep.analyze_quality(df, umbral=0.50)
    assert isinstance(decision, bool)

