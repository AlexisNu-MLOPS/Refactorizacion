# -------------------------------
# Clase para el Data PreProcessor
# -------------------------------
import pandas as pd
import unicodedata
import re

class DataPreprocessor:
    """
    Clase responsable de la limpieza, generación de banderas de nulos
    y análisis de calidad del dataset.
    """

    def __init__(self):
        pass

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza los nombres de las columnas:
        - Minúsculas
        - Sin acentos
        - Reemplaza espacios por _
        - Elimina caracteres especiales
        """
        df = df.copy()

        def limpiar(col):
            col = str(col)
            col = unicodedata.normalize("NFKD", col)
            col = col.encode("ascii", "ignore").decode("utf-8")
            col = col.lower()
            col = re.sub(r"\s+", "_", col)
            col = re.sub(r"[^a-z0-9_]", "", col)
            col = re.sub(r"_+", "_", col).strip("_")
            return col

        df.columns = [limpiar(c) for c in df.columns]
        return df

    def add_null_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea columnas *_nan indicando presencia de valores nulos.
        """
        df = df.copy()
        for col in df.columns:
            df[f"{col}_nan"] = df[col].isna().astype(int)
        return df

    def analyze_quality(self, df: pd.DataFrame, umbral: float = 0.1) -> bool:
        """
        Evalúa calidad del dataset según porcentaje de nulos
        en columnas críticas.
        """
        if "monto" not in df.columns or "es_fraude" not in df.columns:
            raise ValueError("Columnas críticas faltantes")

        total = len(df)
        pct_monto = df["monto"].isna().sum() / total
        pct_fraude = df["es_fraude"].isna().sum() / total

        if pct_monto > umbral or pct_fraude > umbral:
            print("⚠ Dataset debe descartarse por alta cantidad de nulos")
            return False

        return True
