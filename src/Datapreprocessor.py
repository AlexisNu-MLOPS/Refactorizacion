class DataPreprocessor:
    """
    Clase para preprocesamiento de DataFrames.
    
    Parametros:
    -----------
    df : pd.DataFrame
        DataFrame a procesar.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_columns(self):
        """
        Normaliza los nombres de los columnas:
        - Convierte a minusculas
        - Reemplaza espacios por guiones bajos
        - Elimina caracteres especiales
        - Renombra columnas especificas a nombres estandar
        """
        def limpiar_caracteres(col):
            col = str(col)
            col = unicodedata.normalize("NFKD", col)
            col = col.encode("ascii", "ignore").decode("utf-8")
            col = col.lower()
            col = re.sub(r"\s+", "_", col)
            col = col.replace("@", "a").replace("/", "_")
            col = re.sub(r"[^a-z0-9_]", "", col)
            col = re.sub(r"_+", "_", col).strip("_")
            return col

        self.df.columns = [limpiar_caracteres(c) for c in self.df.columns]

        # Renombrar columnas especificas si existen
        rename_map = {
            "score_15": "score",
            "es_fraude": "fraude",
            "notes_comments": "comentarios"
        }
        self.df.rename(
            columns={k: v for k, v in rename_map.items() if k in self.df.columns},
            inplace=True
        )
        return self.df

    def vacios_a_nulos(self):
        """
        Convierte valores vacios, espacios en blanco o string 'nan' a valores NaN.
        """
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                # Convertimos a string y reemplazamos espacios vacios por NaN
                self.df[col] = self.df[col].astype("string").replace(
                    r"^\s*$", pd.NA, regex=True
                )
                # Convertimos string "nan" a NaN
                self.df[col] = self.df[col].replace("nan", pd.NA)
        return self.df

    def limpieza_monto(self, col_name="monto"):
        """
        Limpieza de columnas numericas que representan montos:
        - Elimina simbolos de dolar y comas
        - Convierte a tipo numerico
        """
        if col_name in self.df.columns:
            self.df[col_name] = (
                self.df[col_name].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            self.df[col_name] = pd.to_numeric(self.df[col_name], errors="coerce")
        return self.df

    def limpieza_num_cliente(self, col_name="nombre_cliente_raw"):
        """
        Extrae numeros de columnas de clientes y los convierte a numerico.
        """
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].astype(str).str.extract(r"(\d+)")
            self.df[col_name] = pd.to_numeric(self.df[col_name], errors="coerce")
        return self.df

    def limpieza_score(self, col_name="score"):
        """
        Convierte columnas de score de texto a numerico (uno=1, dos=2, etc.).
        """
        if col_name in self.df.columns:
            mapping = {"uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
            self.df[col_name] = self.df[col_name].astype(str).str.lower().str.strip()
            self.df[col_name] = self.df[col_name].replace(mapping)
            self.df[col_name] = pd.to_numeric(self.df[col_name], errors="coerce")
        return self.df

    def eliminar_acentos(self):
        """
        Elimina acentos de todas las columnas de tipo string.
        """
        for col in self.df.select_dtypes(include=["string"]).columns:
            self.df[col] = self.df[col].apply(
                lambda x: unicodedata.normalize("NFKD", x)
                .encode("ascii", "ignore")
                .decode("utf-8")
                if pd.notna(x) else x
            )
        return self.df

    def limpieza_fecha_registro(self, col_name="fecha_registro"):
        """
        Convierte columnas de fecha a datetime.
        """
        if col_name in self.df.columns:
            self.df[col_name] = pd.to_datetime(self.df[col_name], errors="coerce")
        return self.df

    def ban_columnas_nulas(self):
        """
        Crea columnas bandera indicando la presencia de valores nulos.
        """
        for col in self.df.columns:
            self.df[f"{col}_nan"] = self.df[col].isna().astype(int)
        return self.df

    def validar_nulos_criticos(
        self,
        target_col="fraude",
        monto_col="monto",
        umbral=0.10
    ):
        """
        Valida si columnas criticas superan un umbral de nulos.
        """
        alertas = []

        for col in [target_col, monto_col]:
            if col in self.df.columns:
                porcentaje_nulos = self.df[col].isna().mean()

                if porcentaje_nulos > umbral:
                    alertas.append(
                        f"- Columna '{col}' tiene {porcentaje_nulos:.2%} de nulos "
                        f"(umbral permitido: {umbral:.0%})"
                    )

        if alertas:
            print(
                "\nEL SET DE DATOS NO ES APTO PARA MODELOS DE ML\n"
                "Se recomienda DESCARTAR este set de datos."
            )
            print("\n".join(alertas))
            raise DataQualityException("Falla en la calidad de datos")

        print("\n<Tras la evaluacion del contenido del archivo, se concluye que:")
        print("\nEL SET DE DATOS ES APTO PARA MODELOS DE ML>")
        return True

    def calidad_df(self):
        """
        Retorna informacion basica de calidad del DataFrame:
        - resumen de valores nulos
        - omite columnas que terminan en '_nan' (columnas bandera)
        """
        # Filtrar columnas que no terminan en '_nan'
        return self.df[[col for col in self.df.columns if not col.endswith("_nan")]].isna().sum()


# -------------------------------
# Funcion principal
# -------------------------------

def carga_procesa(filepath, **kwargs):
    """
    Carga un archivo y aplica procesamiento estandar:
    - Limpieza de columnas
    - Conversion de vacios a NaN
    - Limpieza de montos, scores y fechas
    - Eliminacion de acentos
    - Creacion de columnas bandera de nulos
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext not in LOADER_FACTORY:
        raise ValueError(f"Formato no soportado: {ext}")

    # Carga del archivo usando el loader adecuado
    loader = LOADER_FACTORY[ext](filepath)
    df = loader.read_file(**kwargs)

    # Procesamiento de datos
    processor = DataPreprocessor(df)
    processor.clean_columns()
    processor.vacios_a_nulos()
    processor.limpieza_monto()
    processor.limpieza_num_cliente()
    processor.limpieza_score()
    processor.limpieza_fecha_registro()
    processor.eliminar_acentos()
    processor.ban_columnas_nulas()

    # Validacion critica de calidad de datos
    processor.validar_nulos_criticos(
        target_col="fraude",
        monto_col="monto",
        umbral=0.10
    )

    return processor.df, processor.calidad_df()


# -------------------------------
# Informacion global del DF
# -------------------------------

if __name__ == "__main__":
    ruta_archivo = input("Ingrese la ruta del archivo: ").strip()
    try:
        df, quality = carga_procesa(ruta_archivo, sep=",", encoding="utf-8")

        # Imprimir calidad de datos
        print("\nCalidad de datos (valores nulos por columna):")
        print(quality)

        # Muestra el DF tabulado
        print("\nVista del DataFrame final (primeras 5 filas):")
        print(df.head(5).to_markdown(index=False))

    except DataQualityException:
        print("\nProceso detenido por falla critica de calidad de datos.")
    except Exception as e:
        print(f"\nError: {e}")