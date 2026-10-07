"""
Métricas e inferencia estadística sobre las predicciones guardadas.

- Intervalos de confianza por bootstrap no paramétrico (percentil).
- Test de McNemar para comparar dos modelos evaluados sobre los MISMOS ejemplos.
- Calibración: Expected Calibration Error (ECE) y curva de confiabilidad.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score


def _acc(y, yhat):
    return (y == yhat).mean(axis=-1)


def _f1(y, yhat):
    tp = ((yhat == 1) & (y == 1)).sum(axis=-1)
    fp = ((yhat == 1) & (y == 0)).sum(axis=-1)
    fn = ((yhat == 0) & (y == 1)).sum(axis=-1)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def bootstrap_ic(y, yhat, metrica="accuracy", B=2000, nivel=0.95, seed=42):
    """
    IC percentil para accuracy o F1: se remuestrean los ejemplos con reposición
    B veces y se toma el cuantil (1-nivel)/2 y 1-(1-nivel)/2 de la métrica.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    fn = {"accuracy": _acc, "f1": _f1}[metrica]
    rng = np.random.default_rng(seed)
    n = len(y)
    valores = []
    bloque = max(1, 2_000_000 // n)  # remuestras por bloque, para no agotar memoria
    for inicio in range(0, B, bloque):
        b = min(bloque, B - inicio)
        idx = rng.integers(0, n, size=(b, n))
        valores.append(fn(y[idx], yhat[idx]))
    valores = np.concatenate(valores)
    alfa = (1 - nivel) / 2
    return float(np.quantile(valores, alfa)), float(np.quantile(valores, 1 - alfa))


def mcnemar(y, pred_a, pred_b):
    """
    Test de McNemar. Solo importan los ejemplos donde los modelos discrepan:
      b = A acierta y B falla,  c = A falla y B acierta.
    Bajo H0 (misma tasa de error), b ~ Binomial(b + c, 1/2).
    Se usa el test binomial exacto si b + c < 25 y, si no, el estadístico
    chi-cuadrado con corrección de continuidad: (|b - c| - 1)^2 / (b + c).
    """
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    ok_a, ok_b = pred_a == y, pred_b == y
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    if b + c == 0:
        p = 1.0
    elif b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, df=1)
    return {"b": b, "c": c, "p_valor": float(p)}


def ece(y, p_pos, n_bins=10):
    """
    Expected Calibration Error. Se agrupan las predicciones por confianza
    (max(p, 1-p), entre 0.5 y 1) y se promedia |accuracy - confianza| en cada grupo,
    ponderado por la cantidad de ejemplos del grupo.
    """
    y, p_pos = np.asarray(y), np.asarray(p_pos)
    conf = np.maximum(p_pos, 1 - p_pos)
    acierto = ((p_pos >= 0.5).astype(int) == y).astype(float)
    bordes = np.linspace(0.5, 1.0, n_bins + 1)
    grupo = np.clip(np.digitize(conf, bordes[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for g in range(n_bins):
        m = grupo == g
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def curva_confiabilidad(y, p_pos, n_bins=10):
    """Para el diagrama de confiabilidad: P(y=1) observada vs. p_pos promedio por bin."""
    y, p_pos = np.asarray(y), np.asarray(p_pos)
    bordes = np.linspace(0, 1, n_bins + 1)
    grupo = np.clip(np.digitize(p_pos, bordes[1:-1], right=True), 0, n_bins - 1)
    filas = []
    for g in range(n_bins):
        m = grupo == g
        if m.sum() > 0:
            filas.append((p_pos[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(filas)  # columnas: predicho, observado, n


def resumen(y, p_pos, con_ic=True):
    y, p_pos = np.asarray(y), np.asarray(p_pos)
    yhat = (p_pos >= 0.5).astype(int)
    r = {
        "n": len(y),
        "accuracy": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat),
        "ece": ece(y, p_pos),
        "brier": brier_score_loss(y, p_pos),
    }
    if con_ic:
        r["acc_ic_inf"], r["acc_ic_sup"] = bootstrap_ic(y, yhat)
    return r
