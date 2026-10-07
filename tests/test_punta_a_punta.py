"""
Prueba de punta a punta con datos sintéticos: corre las etapas clásicas y el
análisis completo, sin descargar nada de internet salvo los recursos de NLTK.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent

POS = "great excellent wonderful loved amazing brilliant".split()
NEG = "bad awful boring terrible waste worst".split()
NEU = "the movie film plot actor scene story was is and it this".split()


def _resena(rng, label):
    palabras = list(rng.choice(NEU, 6)) + list(rng.choice(POS if label else NEG, 2))
    rng.shuffle(palabras)
    return " ".join(palabras) + ". It wasn't long."


def _split(rng, n):
    labels = np.repeat([0, 1], n // 2)  # ordenado por clase, como en Hugging Face
    return pd.DataFrame({"text": [_resena(rng, l) for l in labels], "label": labels})


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    import nltk

    for r in ["stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
        nltk.download(r, quiet=True)

    base = tmp_path_factory.mktemp("imdb")
    rng = np.random.default_rng(0)
    _split(rng, 400).to_parquet(base / "train.parquet")
    _split(rng, 25_000).to_parquet(base / "test.parquet")  # la submuestra fija asume 25 000
    _split(rng, 200).to_parquet(base / "unsupervised.parquet")

    env = dict(os.environ, NLP_IMDB_LOCAL=str(base), NLP_RESULTS_DIR=str(base / "results"))
    return env, base / "results"


def _correr(env, *args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=RAIZ, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_etapas_clasicas_y_analisis(entorno, tmp_path):
    env, results = entorno
    _correr(env, "src.correr", "--etapas", "bow", "tfidf", "w2v")
    for slug in ["bow_nb", "tfidf_lr", "w2v_lr"]:
        df = pd.read_csv(results / "predicciones" / f"{slug}.csv")
        assert (df["split"] == "test").sum() == 25_000
        assert (df["split"] == "challenge").sum() == 50
        assert df["p_pos"].between(0, 1).all()

    # El análisis no debe tocar el README real del repo durante el test
    readme = RAIZ / "README.md"
    original = readme.read_text(encoding="utf-8")
    try:
        salida = _correr(env, "src.analizar")
    finally:
        readme.write_text(original, encoding="utf-8")

    assert "Test completo" in salida
    for fig in ["accuracy", "calibracion", "challenge", "costo"]:
        assert (results / "figuras" / f"{fig}.png").exists()

    # Los datos sintéticos son fáciles: TF-IDF tiene que andar bien
    tfidf = pd.read_csv(results / "predicciones" / "tfidf_lr.csv").query("split == 'test'")
    assert ((tfidf["p_pos"] >= 0.5) == tfidf["y_true"]).mean() > 0.9


def test_mcnemar_y_ece():
    sys.path.insert(0, str(RAIZ))
    from src.evaluacion import ece, mcnemar

    y = np.array([1, 1, 0, 0, 1, 0])
    r = mcnemar(y, y, 1 - y)
    assert r["b"] == 6 and r["c"] == 0 and r["p_valor"] < 0.05
    assert ece(y, y.astype(float)) == pytest.approx(0.0)
