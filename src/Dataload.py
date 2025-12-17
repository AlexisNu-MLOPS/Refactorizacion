import pandas as pd
from pathlib import Path
import unicodedata
import re
import tabulate

# -------------------------------
# Excepción de calidad de datos
# -------------------------------

class DataQualityException(Exception):
    """Excepción para problemas críticos de calidad de datos."""
    pass


# -------------------------------
# Loaders
# -------------------------------

class DataLoader:
    """
    Clase base para cargar datos desde un archivo.
    
    Parametros:
    -----------
    filepath : str o Path
        Ruta al archivo que se desea cargar.
    """
    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def read_file(self, **kwargs):
        """
        Metodo base que debe ser sobresrito por cada loader especifico.
        """
        raise NotImplementedError("Este metodo debe ser implementado por subclases")


class CSVLoader(DataLoader):
    """
    Carga archivos CSV.
    """
    def read_file(self, **kwargs):
        return pd.read_csv(self.filepath, **kwargs)


class ExcelLoader(DataLoader):
    """
    Carga archivos Excel (.xls y .xlsx).
    """
    def read_file(self, **kwargs):
        return pd.read_excel(self.filepath, **kwargs)


class JSONLoader(DataLoader):
    """
    Carga archivos JSON.
    """
    def read_file(self, **kwargs):
        return pd.read_json(self.filepath, **kwargs)


# Asociacion entre extension y Loader
LOADER_FACTORY = {
    ".csv": CSVLoader,
    ".xlsx": ExcelLoader,
    ".xls": ExcelLoader,
    ".json": JSONLoader,
    ".txt": CSVLoader
}

# -------------------------------
# Clase para el Data PreProcessor
# -------------------------------
