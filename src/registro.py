"""Guardado y lectura de predicciones y tiempos de cada modelo."""

import json

import numpy as np
import pandas as pd

from .config import PRED_DIR, TIEMPOS_DIR, crear_directorios


def guardar_predicciones(slug, y_test, p_test, idx_test, y_chal, p_chal):
    """
    Guarda una fila por ejemplo con la probabilidad de clase positiva (`p_pos`).
    """
    crear_directorios()
    test = pd.DataFrame({"split": "test", "idx": idx_test, "y_true": y_test, "p_pos": p_test})
    chal = pd.DataFrame(
        {"split": "challenge", "idx": np.arange(len(y_chal)), "y_true": y_chal, "p_pos": p_chal}
    )
    pd.concat([test, chal]).to_csv(PRED_DIR / f"{slug}.csv", index=False, float_format="%.6f")


def guardar_tiempos(slug, entrenamiento_s, inferencia_s, n_inferencia, hardware, extra=None):
    crear_directorios()
    datos = {
        "entrenamiento_s": round(float(entrenamiento_s), 2),
        "inferencia_s": round(float(inferencia_s), 2),
        "n_inferencia": int(n_inferencia),
        "inferencia_s_por_1000": round(1000 * float(inferencia_s) / max(int(n_inferencia), 1), 3),
        "hardware": hardware,
    }
    if extra:
        datos.update(extra)
    (TIEMPOS_DIR / f"{slug}.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False))


def existe(slug) -> bool:
    return (PRED_DIR / f"{slug}.csv").exists()


def cargar_predicciones(slug) -> pd.DataFrame:
    return pd.read_csv(PRED_DIR / f"{slug}.csv")


def cargar_tiempos(slug) -> dict | None:
    ruta = TIEMPOS_DIR / f"{slug}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else None


def hardware_actual() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass
    return "CPU"
