"""
Etapas 1 a 3: modelos que corren en CPU.

1. Bag of Words + Naive Bayes, comparando variantes de preprocesamiento.
2. TF-IDF (unigramas y bigramas) + regresión logística.
3. Word2Vec entrenado sobre IMDB, promedio de vectores + regresión logística.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from ..config import RESULTS_DIR, SEED
from ..preprocesamiento import VARIANTES, solo_limpieza, tokenizar
from ..registro import guardar_predicciones, guardar_tiempos


def _bow_nb():
    return make_pipeline(CountVectorizer(min_df=2, max_df=0.95), MultinomialNB(alpha=1.0))


# ─── Etapa 1 ──────────────────────────────────────────────────────────────────
def correr_bow(datos, challenge, verbose=True):
    """
    Compara las variantes de preprocesamiento con validación cruzada de 5 folds
    SOBRE TRAIN y elige la mejor según ese criterio. El test set se usa solo para
    reportar: elegir la variante mirando el test sería sobreajustar al test.
    """
    train, test = datos["train"], datos["test"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    filas, cache = [], {}

    for nombre, fn in VARIANTES.items():
        t0 = time.time()
        x_train = [fn(t) for t in train["text"]]
        x_test = [fn(t) for t in test["text"]]
        t_prep = time.time() - t0
        cache[nombre] = (x_train, x_test)

        scores = cross_val_score(_bow_nb(), x_train, train["label"], cv=cv, n_jobs=-1)
        modelo = _bow_nb().fit(x_train, train["label"])
        acc_test = (modelo.predict(x_test) == test["label"].to_numpy()).mean()
        vocab = len(modelo.named_steps["countvectorizer"].vocabulary_)
        filas.append(
            {
                "variante": nombre,
                "vocabulario": vocab,
                "cv_accuracy": scores.mean(),
                "cv_desvio": scores.std(),
                "test_accuracy": acc_test,
                "preprocesamiento_s": round(t_prep, 1),
            }
        )
        if verbose:
            print(f"  {nombre:<40} CV={scores.mean():.4f}±{scores.std():.4f}  test={acc_test:.4f}")

    tabla = pd.DataFrame(filas)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTS_DIR / "bow_variantes.csv", index=False, float_format="%.4f")

    mejor = tabla.loc[tabla["cv_accuracy"].idxmax(), "variante"]
    if verbose:
        print(f"\n  Variante elegida por CV: {mejor}")

    fn = VARIANTES[mejor]
    x_train, x_test = cache[mejor]
    t0 = time.time()
    modelo = _bow_nb().fit(x_train, train["label"])
    t_train = time.time() - t0
    t0 = time.time()
    p_test = modelo.predict_proba(x_test)[:, 1]
    t_inf = time.time() - t0
    p_chal = modelo.predict_proba([fn(t) for t in challenge["text"]])[:, 1]

    guardar_predicciones("bow_nb", test["label"], p_test, test.index, challenge["label"], p_chal)
    guardar_tiempos("bow_nb", t_train, t_inf, len(test), "CPU", {"variante": mejor})
    return tabla


# ─── Etapa 2 ──────────────────────────────────────────────────────────────────
def correr_tfidf(datos, challenge, verbose=True):
    train, test = datos["train"], datos["test"]
    modelo = make_pipeline(
        TfidfVectorizer(
            preprocessor=solo_limpieza,
            max_features=100_000,
            ngram_range=(1, 2),  # los bigramas capturan "not good", "the worst", etc.
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # 1 + log(tf): una palabra repetida 10 veces no pesa 10 veces más
        ),
        LogisticRegression(C=1.0, max_iter=1000),
    )
    t0 = time.time()
    modelo.fit(train["text"], train["label"])
    t_train = time.time() - t0
    t0 = time.time()
    p_test = modelo.predict_proba(test["text"])[:, 1]
    t_inf = time.time() - t0
    p_chal = modelo.predict_proba(challenge["text"])[:, 1]

    guardar_predicciones("tfidf_lr", test["label"], p_test, test.index, challenge["label"], p_chal)
    guardar_tiempos("tfidf_lr", t_train, t_inf, len(test), "CPU")

    # Atributos con mayor peso hacia cada clase
    vec, clf = modelo.named_steps["tfidfvectorizer"], modelo.named_steps["logisticregression"]
    nombres, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
    orden = np.argsort(coef)
    top = pd.DataFrame(
        {
            "positivo": nombres[orden[-15:][::-1]],
            "coef_pos": coef[orden[-15:][::-1]].round(3),
            "negativo": nombres[orden[:15]],
            "coef_neg": coef[orden[:15]].round(3),
        }
    )
    top.to_csv(RESULTS_DIR / "tfidf_top_atributos.csv", index=False)
    if verbose:
        print(f"  accuracy test = {((p_test >= 0.5) == test['label']).mean():.4f}")
        print(top.head(10).to_string(index=False))
    return top


# ─── Etapa 3 ──────────────────────────────────────────────────────────────────
def _vector_documento(tokens, kv):
    vecs = [kv[w] for w in tokens if w in kv.key_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size, dtype=np.float32)


def correr_word2vec(datos, challenge, verbose=True, dim=100, epocas=5):
    """
    Entrena Word2Vec (skip-gram) sobre las reseñas de train y las 50 000 sin etiqueta
    del split "unsupervised" (usar texto sin etiquetas es legítimo: no hay fuga del
    test). Cada reseña se representa con el promedio de los vectores de sus palabras.
    """
    from gensim.models import Word2Vec

    train, test = datos["train"], datos["test"]
    textos_w2v = list(train["text"]) + list(datos.get("unsup", pd.DataFrame({"text": []}))["text"])

    t0 = time.time()
    oraciones = [tokenizar(t) for t in textos_w2v]
    w2v = Word2Vec(
        oraciones,
        vector_size=dim,
        window=5,
        min_count=5,
        sg=1,
        epochs=epocas,
        workers=os.cpu_count() or 2,
        seed=SEED,
    )
    kv = w2v.wv
    x_train = np.vstack([_vector_documento(tokenizar(t), kv) for t in train["text"]])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(x_train, train["label"])
    t_train = time.time() - t0

    t0 = time.time()
    x_test = np.vstack([_vector_documento(tokenizar(t), kv) for t in test["text"]])
    p_test = clf.predict_proba(x_test)[:, 1]
    t_inf = time.time() - t0
    x_chal = np.vstack([_vector_documento(tokenizar(t), kv) for t in challenge["text"]])
    p_chal = clf.predict_proba(x_chal)[:, 1]

    guardar_predicciones("w2v_lr", test["label"], p_test, test.index, challenge["label"], p_chal)
    guardar_tiempos("w2v_lr", t_train, t_inf, len(test), "CPU", {"dimension": dim})

    # Vecinos más cercanos: muestran por qué el promedio pierde información de
    # sentimiento. Word2Vec aprende de contexto, y "good" y "bad" aparecen en
    # contextos casi idénticos, así que quedan cerca.
    filas = []
    for palabra in ["good", "bad", "boring", "masterpiece"]:
        if palabra in kv.key_to_index:
            for vecino, sim in kv.most_similar(palabra, topn=5):
                filas.append({"palabra": palabra, "vecino": vecino, "similitud": round(sim, 3)})
    vecinos = pd.DataFrame(filas)
    vecinos.to_csv(RESULTS_DIR / "w2v_vecinos.csv", index=False)
    if verbose:
        print(f"  accuracy test = {((p_test >= 0.5) == test['label']).mean():.4f}")
        if len(vecinos):
            print(vecinos.groupby("palabra")["vecino"].apply(", ".join).to_string())
    return vecinos
