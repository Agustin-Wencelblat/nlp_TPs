"""Carga de IMDB, de la submuestra fija y del challenge set."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR


def cargar_imdb(incluir_unsup: bool = False) -> dict[str, pd.DataFrame]:
    """
    Devuelve {"train", "test"[, "unsup"]} como DataFrames con columnas `text` y `label`
    (0 = negativo, 1 = positivo). El índice de cada fila en "test" es el que usan
    todos los archivos de predicciones.

    Para pruebas locales, NLP_IMDB_LOCAL puede apuntar a una carpeta
    con train.parquet, test.parquet y (opcional) unsupervised.parquet.
    """
    splits = {"train": "train", "test": "test"}
    if incluir_unsup:
        splits["unsup"] = "unsupervised"

    local = os.environ.get("NLP_IMDB_LOCAL")
    if local:
        return {k: pd.read_parquet(Path(local) / f"{v}.parquet") for k, v in splits.items()}

    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    return {k: ds[v].to_pandas()[["text", "label"]] for k, v in splits.items()}


def indices_submuestra() -> np.ndarray:
    """
    1000 índices del test set elegidos al azar (seed 42) y guardados en
    data/submuestra_idx.csv. Sobre esta submuestra se evalúan TODOS los modelos,
    incluido el LLM, para poder compararlos en igualdad de condiciones.
    """
    return pd.read_csv(DATA_DIR / "submuestra_idx.csv")["idx"].to_numpy()


def cargar_challenge() -> pd.DataFrame:
    """Reseñas escritas a mano para casos difíciles. Columnas: id, categoria, text, label."""
    return pd.read_csv(DATA_DIR / "challenge_set.csv")
