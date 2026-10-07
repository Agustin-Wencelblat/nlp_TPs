"""Rutas, seeds y nombres de los modelos, compartidos por todo el proyecto."""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"

# Los resultados se pueden redirigir (por ejemplo, a Google Drive en Colab)
# con la variable de entorno NLP_RESULTS_DIR.
RESULTS_DIR = Path(os.environ.get("NLP_RESULTS_DIR", RAIZ / "results"))
PRED_DIR = RESULTS_DIR / "predicciones"
TIEMPOS_DIR = RESULTS_DIR / "tiempos"
FIG_DIR = RESULTS_DIR / "figuras"

SEED = 42

# Modelos en el orden de la "evolución" que cuenta el repo.
# slug (nombre de archivo) -> nombre para mostrar en tablas y gráficos.
MODELOS = {
    "bow_nb": "BoW + Naive Bayes",
    "tfidf_lr": "TF-IDF + Reg. logística",
    "w2v_lr": "Word2Vec + Reg. logística",
    "sst2_transfer": "DistilBERT SST-2 (sin ajuste)",
    "bert_ft": "BERT fine-tuneado",
    "llm_zeroshot": "LLM zero-shot",
    "llm_fewshot": "LLM few-shot",
}


def crear_directorios():
    for d in (PRED_DIR, TIEMPOS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
