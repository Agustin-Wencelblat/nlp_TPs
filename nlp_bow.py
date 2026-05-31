# ── Imports ───────────────────────────────────────────────────────────────────
import re
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
 
def load_imdb():
    """
    Carga IMDB desde HuggingFace datasets.
    Retorna (train_texts, train_labels, test_texts, test_labels).
    Labels: 0 = negativo, 1 = positivo.
    """
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train_texts  = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts   = ds["test"]["text"]
    test_labels  = ds["test"]["label"]
    return train_texts, train_labels, test_texts, test_labels
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
 
STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
 
 
def clean(text: str) -> list[str]:
    """Paso base: lowercase + eliminar caracteres no alfabéticos."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)   # eliminar HTML tags (IMDB los tiene)
    text = re.sub(r"[^a-z\s]", " ", text)  # conservar solo letras
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()
 
 
def preprocess_none(text: str) -> str:
    """Solo limpieza básica. Sin stopwords, sin normalización morfológica."""
    return " ".join(clean(text))
 
 
def preprocess_stopwords(text: str) -> str:
    """Limpieza + remoción de stopwords."""
    tokens = [t for t in clean(text) if t not in STOP_WORDS]
    return " ".join(tokens)
 
 
def preprocess_stem(text: str) -> str:
    """
    Limpieza + stopwords + stemming (Porter Stemmer).
    Reduce a raíz heurística: 'running' → 'run', 'loved' → 'love'.
    Rápido pero las raíces no son palabras reales: 'happiness' → 'happi'.
    """
    tokens = [stemmer.stem(t)
              for t in clean(text)
              if t not in STOP_WORDS]
    return " ".join(tokens)
 
 
def preprocess_lemma(text: str) -> str:
    """
    Limpieza + stopwords + lemmatization (WordNet).
    Reduce al lema canónico: 'running' → 'run', 'better' → 'good'.
    Más preciso que stemming; los tokens resultantes son palabras reales.
    """
    tokens = [lemmatizer.lemmatize(t)
              for t in clean(text)
              if t not in STOP_WORDS]
    return " ".join(tokens)
 
 
# Diccionario con todas las variantes a comparar
PREPROS = {
    "Sin preprocesar":        preprocess_none,
    "+ Stopwords removidas":  preprocess_stopwords,
    "+ Stemming":             preprocess_stem,
    "+ Lemmatization":        preprocess_lemma,
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE: BOW + NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
 
def run_experiment(name, preprocess_fn, train_texts, train_labels,
                   test_texts, test_labels):
    """
    Ejecuta un experimento completo con una función de preprocesamiento dada.
 
    Pasos internos:
      1. Aplicar preprocesamiento a todos los textos.
      2. CountVectorizer: construye el vocabulario y genera la matriz
         documento×término (cada celda = conteo de la palabra en el doc).
      3. MultinomialNB: estima P(clase) y P(palabra|clase) con MLE.
         alpha=1.0 aplica Laplace smoothing para manejar palabras no vistas.
      4. Predecir y evaluar sobre test set.
 
    Retorna un dict con los resultados.
    """
    t0 = time.time()
 
    # ── Preprocesamiento ──────────────────────────────────────────────────────
    X_train = [preprocess_fn(t) for t in train_texts]
    X_test  = [preprocess_fn(t) for t in test_texts]
 
    # ── Vectorización (Bag of Words) ──────────────────────────────────────────
    #   min_df=2: ignora tokens que aparecen en menos de 2 documentos (ruido)
    #   max_df=0.95: ignora tokens en >95% de los docs (palabras vacías que
    #                escaparon al filtro de stopwords)
    vectorizer = CountVectorizer(min_df=2, max_df=0.95)
    X_tr = vectorizer.fit_transform(X_train)  # aprende vocabulario + transforma
    X_te = vectorizer.transform(X_test)       # solo transforma (sin re-aprender)
 
    # ── Naive Bayes ───────────────────────────────────────────────────────────
    #   alpha=1.0  →  Laplace smoothing: P(w|c) = (count(w,c)+1) / (count(c)+|V|)
    #   Esto evita que palabras no vistas en entrenamiento tengan P=0
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_tr, train_labels)
 
    # ── Evaluación ────────────────────────────────────────────────────────────
    preds = clf.predict(X_te)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average="binary")
    vocab = len(vectorizer.vocabulary_)
    elapsed = time.time() - t0
 
    return {
        "name": name,
        "acc": acc,
        "f1": f1,
        "vocab": vocab,
        "elapsed": elapsed,
        "vectorizer": vectorizer,
        "clf": clf,
        "preprocess_fn": preprocess_fn,
        "preds": preds,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. ANÁLISIS DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
 
def print_results_table(results: list[dict]):
    """Imprime tabla comparativa de todas las variantes."""
    print("\n" + "═" * 66)
    print(f"  {'Configuración':<28} {'Vocab':>7} {'Acc':>7} {'F1':>7} {'Time':>7}")
    print("═" * 66)
    for r in results:
        print(f"  {r['name']:<28} {r['vocab']:>7,} "
              f"{r['acc']:>7.2%} {r['f1']:>7.3f} {r['elapsed']:>6.1f}s")
    print("═" * 66)
 
 
def print_top_features(result: dict, n: int = 12):
    """
    Muestra las palabras más discriminativas según log P(w|pos) - log P(w|neg).
    Un log-ratio alto → la palabra predice sentimiento positivo.
    Un log-ratio bajo  → la palabra predice sentimiento negativo.
    """
    vec  = result["vectorizer"]
    clf  = result["clf"]
    feat = np.array(vec.get_feature_names_out())
 
    # log_prob_ tiene forma (n_clases, vocab): log P(token | clase)
    diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
 
    top_pos = np.argsort(diff)[-n:][::-1]
    top_neg = np.argsort(diff)[:n]
 
    print(f"\n  Top features — {result['name']}")
    print(f"  {'PREDICEN POSITIVO':<35}  {'PREDICEN NEGATIVO'}")
    print(f"  {'─'*35}  {'─'*35}")
    for i, j in zip(top_pos, top_neg):
        print(f"  {feat[i]:<20} log-ratio {diff[i]:+.2f}   "
              f"{feat[j]:<20} log-ratio {diff[j]:+.2f}")
 
 
def print_confusion_matrix(result: dict, test_labels):
    """Imprime matriz de confusión formateada."""
    cm = confusion_matrix(test_labels, result["preds"])
    print(f"\n  Matriz de confusión — {result['name']}")
    print(f"  {'':20} Pred NEG   Pred POS")
    print(f"  {'Real NEG':<20} {cm[0,0]:>8,}   {cm[0,1]:>8,}")
    print(f"  {'Real POS':<20} {cm[1,0]:>8,}   {cm[1,1]:>8,}")
 
 
def predict_example(result: dict, texts: list[str]):
    """Clasifica ejemplos nuevos y muestra la probabilidad de cada clase."""
    fn  = result["preprocess_fn"]
    vec = result["vectorizer"]
    clf = result["clf"]
 
    print(f"\n  Ejemplos de predicción — {result['name']}")
    print(f"  {'─'*60}")
    for text in texts:
        prep  = fn(text)
        x     = vec.transform([prep])
        pred  = clf.predict(x)[0]
        proba = clf.predict_proba(x)[0]
        label = "POSITIVO ✓" if pred == 1 else "NEGATIVO ✗"
        print(f"\n  Texto:  {text}")
        print(f"  Tokens: {prep}")
        print(f"  → {label}  (P_neg={proba[0]:.3f}, P_pos={proba[1]:.3f})")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
 
    print("\n  Cargando dataset IMDB...")
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    print(f"  Train: {len(train_texts):,} | Test: {len(test_texts):,}")
 
    # ── Experimentos: una variante por preprocesador ──────────────────────────
    print("\n  Entrenando variantes...")
    results = []
    for name, fn in PREPROS.items():
        r = run_experiment(name, fn, train_texts, train_labels,
                           test_texts, test_labels)
        results.append(r)
        print(f"  ✓ {name}")
 
    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print_results_table(results)
 
    # ── Análisis de la mejor variante ─────────────────────────────────────────
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  Mejor variante: '{best['name']}' (F1={best['f1']:.3f})")
 
    print_top_features(best, n=10)
    print_confusion_matrix(best, test_labels)
 
    # ── Predicción sobre ejemplos nuevos ─────────────────────────────────────
    ejemplos = [
        "This was a brilliant and moving film, truly unforgettable",
        "Terrible movie, completely boring and a waste of time",
        "Not bad but not great either, somewhat disappointing overall",
        "I didn't like it at first but it grew on me eventually",
    ]
    predict_example(best, ejemplos)
 
    # ── Reporte completo de clasificación ────────────────────────────────────
    print(f"\n  Reporte completo — {best['name']}")
    print(classification_report(test_labels, best["preds"],
                                target_names=["Negativo", "Positivo"]))
