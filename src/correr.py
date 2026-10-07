"""
Corre las etapas del proyecto desde la terminal.

    python -m src.correr                       # todas las etapas que falten
    python -m src.correr --etapas bow tfidf    # solo algunas
    python -m src.correr --reentrenar          # rehace aunque ya haya predicciones

Las etapas cuyo archivo de predicciones ya existe se saltean.
"""

import argparse
import gc

from .config import MODELOS
from .datos import cargar_challenge, cargar_imdb, indices_submuestra
from .registro import existe

ETAPAS = ["bow", "tfidf", "w2v", "sst2", "bert", "llm"]
SLUGS = {
    "bow": ["bow_nb"],
    "tfidf": ["tfidf_lr"],
    "w2v": ["w2v_lr"],
    "sst2": ["sst2_transfer"],
    "bert": ["bert_ft"],
    "llm": ["llm_zeroshot", "llm_fewshot"],
}


def correr(etapas=ETAPAS, reentrenar=False, llm_modelo=None):
    pendientes = [e for e in etapas if reentrenar or not all(existe(s) for s in SLUGS[e])]
    if not pendientes:
        print("Todas las etapas pedidas ya tienen predicciones guardadas.")
        return

    datos = cargar_imdb(incluir_unsup="w2v" in pendientes)
    challenge = cargar_challenge()

    for etapa in pendientes:
        print(f"\n{'═' * 70}\n  Etapa: {', '.join(MODELOS[s] for s in SLUGS[etapa])}\n{'═' * 70}")
        if etapa in ("bow", "tfidf", "w2v"):
            from .modelos import clasicos

            {"bow": clasicos.correr_bow, "tfidf": clasicos.correr_tfidf, "w2v": clasicos.correr_word2vec}[
                etapa
            ](datos, challenge)
        elif etapa == "sst2":
            from .modelos.transformers_gpu import correr_sst2

            correr_sst2(datos, challenge)
        elif etapa == "bert":
            from .modelos.transformers_gpu import correr_bert

            correr_bert(datos, challenge)
        elif etapa == "llm":
            from .modelos.llm import MODELO_POR_DEFECTO, correr_llm

            correr_llm(datos, challenge, indices_submuestra(), modelo_id=llm_modelo or MODELO_POR_DEFECTO)
        _liberar_memoria()


def _liberar_memoria():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--etapas", nargs="+", choices=ETAPAS, default=ETAPAS)
    p.add_argument("--reentrenar", action="store_true")
    p.add_argument("--llm-modelo", default=None, help="ID de Hugging Face del LLM a usar")
    a = p.parse_args()
    correr(a.etapas, a.reentrenar, a.llm_modelo)
