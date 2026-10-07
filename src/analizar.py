"""
Análisis final: lee las predicciones guardadas de todos los modelos y genera
tablas, gráficos y la sección de resultados del README.

Uso:  python -m src.analizar
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .config import FIG_DIR, MODELOS, RAIZ, RESULTS_DIR, crear_directorios
from .datos import cargar_challenge, indices_submuestra
from .evaluacion import curva_confiabilidad, mcnemar, resumen
from .registro import cargar_predicciones, cargar_tiempos, existe

N_TEST = 25_000

# Paleta (validada para daltonismo; ver README) y tinta de texto
AZUL, NARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TEXTO_2,
        "axes.titlecolor": TEXTO,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)

CATEGORIAS = {
    "negacion_de_positivo": "Negación de positivo",
    "negacion_de_negativo": "Negación de negativo",
    "contraste": "Contraste (pero...)",
    "sarcasmo": "Sarcasmo",
    "sentimiento_implicito": "Sentimiento implícito",
}


def _cargar_todo():
    preds = {}
    for slug in MODELOS:
        if existe(slug):
            df = cargar_predicciones(slug)
            preds[slug] = {
                "test": df[df["split"] == "test"].set_index("idx"),
                "challenge": df[df["split"] == "challenge"].set_index("idx"),
                "tiempos": cargar_tiempos(slug),
            }
    if not preds:
        raise SystemExit("No hay predicciones en results/predicciones/. Corré primero los modelos.")
    return preds


def _fmt_ic(r):
    return f"{r['accuracy']:.4f} [{r['acc_ic_inf']:.4f}, {r['acc_ic_sup']:.4f}]"


def _tabla_md(df):
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


# ─── Tablas ───────────────────────────────────────────────────────────────────
def tablas(preds):
    sub = indices_submuestra()
    completo, submuestra = [], []
    res_completo, res_sub = {}, {}

    for slug, p in preds.items():
        t = p["test"]
        nombre = MODELOS[slug]
        tiempos = p["tiempos"] or {}
        if len(t) == N_TEST:
            r = resumen(t["y_true"], t["p_pos"])
            res_completo[slug] = r
            completo.append(
                {
                    "Modelo": nombre,
                    "Accuracy [IC 95%]": _fmt_ic(r),
                    "F1": f"{r['f1']:.4f}",
                    "ECE": f"{r['ece']:.3f}",
                    "Inferencia (s / 1000 reseñas)": f"{tiempos.get('inferencia_s_por_1000', float('nan')):.2f}",
                    "Entrenamiento (s)": f"{tiempos.get('entrenamiento_s', float('nan')):.0f}",
                    "Hardware": tiempos.get("hardware", "?"),
                }
            )
        ts = t.loc[t.index.intersection(sub)]
        if len(ts) == len(sub):
            r = resumen(ts["y_true"], ts["p_pos"])
            res_sub[slug] = r
            submuestra.append(
                {
                    "Modelo": nombre,
                    "Accuracy [IC 95%]": _fmt_ic(r),
                    "F1": f"{r['f1']:.4f}",
                    "ECE": f"{r['ece']:.3f}",
                }
            )
    return pd.DataFrame(completo), pd.DataFrame(submuestra), res_completo, res_sub


def comparaciones(preds):
    """McNemar entre modelos consecutivos de la evolución (test completo) y del LLM contra el resto (submuestra)."""
    sub = indices_submuestra()
    filas = []
    completos = [s for s in preds if len(preds[s]["test"]) == N_TEST]
    pares = [(completos[i], completos[i + 1], "Test completo") for i in range(len(completos) - 1)]
    for llm in ("llm_zeroshot", "llm_fewshot"):
        if llm in preds:
            for otro in ("tfidf_lr", "bert_ft"):
                if otro in preds:
                    pares.append((otro, llm, "Submuestra"))
    if "llm_zeroshot" in preds and "llm_fewshot" in preds:
        pares.append(("llm_zeroshot", "llm_fewshot", "Submuestra"))

    for a, b, conjunto in pares:
        ta, tb = preds[a]["test"], preds[b]["test"]
        if conjunto == "Submuestra":
            ta, tb = ta.loc[sub], tb.loc[sub]
        else:
            tb = tb.loc[ta.index]
        r = mcnemar(ta["y_true"], ta["p_pos"] >= 0.5, tb["p_pos"] >= 0.5)
        filas.append(
            {
                "Comparación": f"{MODELOS[a]} vs. {MODELOS[b]}",
                "Conjunto": conjunto,
                "Solo acierta el 1º": r["b"],
                "Solo acierta el 2º": r["c"],
                "p-valor": "< 1e-16" if r["p_valor"] < 1e-16 else f"{r['p_valor']:.2g}",
            }
        )
    return pd.DataFrame(filas)


def tabla_challenge(preds):
    ch = cargar_challenge()
    filas = {}
    for slug, p in preds.items():
        c = p["challenge"].sort_index()
        acierto = ((c["p_pos"] >= 0.5).astype(int).to_numpy() == ch["label"].to_numpy())
        por_cat = pd.Series(acierto, index=ch["categoria"]).groupby(level=0).mean()
        por_cat["Total"] = acierto.mean()
        filas[MODELOS[slug]] = por_cat
    tabla = pd.DataFrame(filas).T
    tabla = tabla[[c for c in CATEGORIAS if c in tabla.columns] + ["Total"]]
    return tabla.rename(columns=CATEGORIAS)


# ─── Gráficos ─────────────────────────────────────────────────────────────────
def fig_accuracy(res_completo, res_sub):
    slugs = [s for s in MODELOS if s in res_completo or s in res_sub]
    y = np.arange(len(slugs))[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(slugs) + 1.2))
    for desplaz, res, color, etiqueta in (
        (0.12, res_completo, AZUL, f"Test completo ({N_TEST:,} reseñas)".replace(",", ".")),
        (-0.12, res_sub, NARANJA, "Submuestra común (1.000 reseñas)"),
    ):
        pts = [(yi + desplaz, res[s]) for yi, s in zip(y, slugs) if s in res]
        if not pts:
            continue
        ys, rs = zip(*pts)
        acc = np.array([r["accuracy"] for r in rs])
        err = np.array([[r["accuracy"] - r["acc_ic_inf"] for r in rs], [r["acc_ic_sup"] - r["accuracy"] for r in rs]])
        ax.errorbar(acc, ys, xerr=err, fmt="o", ms=5, color=color, ecolor=color, elinewidth=2, capsize=0, label=etiqueta)
    ax.set_yticks(y, [MODELOS[s] for s in slugs], color=TEXTO)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy (punto = estimación, barra = IC 95% bootstrap)")
    ax.set_title("Accuracy por modelo", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.savefig(FIG_DIR / "accuracy.png")
    plt.close(fig)


def fig_calibracion(preds):
    slugs = list(preds)
    cols = min(4, len(slugs))
    filas = int(np.ceil(len(slugs) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(2.3 * cols, 2.4 * filas), sharex=True, sharey=True, squeeze=False)
    for ax, slug in zip(axes.flat, slugs):
        t = preds[slug]["test"]
        curva = curva_confiabilidad(t["y_true"], t["p_pos"])
        ax.plot([0, 1], [0, 1], color=TEXTO_2, lw=1, ls="--")
        ax.plot(curva[:, 0], curva[:, 1], color=AZUL, lw=2, marker="o", ms=4)
        ax.set_title(MODELOS[slug], fontsize=8.5, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(slugs):]:
        ax.set_visible(False)
    fig.supxlabel("Probabilidad predicha de reseña positiva", color=TEXTO_2, fontsize=9)
    fig.supylabel("Proporción observada de positivas", color=TEXTO_2, fontsize=9)
    fig.suptitle("Calibración: sobre la diagonal = bien calibrado", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibracion.png")
    plt.close(fig)


def fig_challenge(tabla):
    datos = tabla.drop(columns="Total")
    fig, ax = plt.subplots(figsize=(1.25 * datos.shape[1] + 2.5, 0.42 * len(datos) + 1.4))
    ax.imshow(datos.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(datos.shape[1]), datos.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(datos)), datos.index, color=TEXTO)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = datos.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else TEXTO)
    ax.set_title("Challenge set: accuracy por tipo de caso difícil", loc="left")
    fig.savefig(FIG_DIR / "challenge.png")
    plt.close(fig)


def fig_costo(preds, res_completo, res_sub):
    """Accuracy en la submuestra común vs. tiempo de inferencia (escala log)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (slug, p) in enumerate(preds.items()):
        tiempos = p["tiempos"]
        if not tiempos or slug not in res_sub:
            continue
        x = tiempos["inferencia_s_por_1000"]
        gpu = tiempos["hardware"].startswith("GPU")
        ax.scatter(x, res_sub[slug]["accuracy"], s=55, color=AZUL, marker="s" if gpu else "o",
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(MODELOS[slug], (x, res_sub[slug]["accuracy"]), xytext=(7, 5 if i % 2 == 0 else -11),
                    textcoords="offset points", fontsize=8, color=TEXTO)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Segundos de inferencia por cada 1000 reseñas (escala log)")
    ax.set_ylabel("Accuracy (submuestra común)")
    ax.scatter([], [], marker="o", color=AZUL, label="CPU")
    ax.scatter([], [], marker="s", color=AZUL, label="GPU")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax.margins(x=0.25)
    ax.set_title("Accuracy vs. costo de inferencia", loc="left")
    fig.savefig(FIG_DIR / "costo.png")
    plt.close(fig)


# ─── README ───────────────────────────────────────────────────────────────────
INICIO, FIN = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIN -->"


def generar_markdown(t_completo, t_sub, t_mcnemar, t_chal):
    partes = ["*Sección generada automáticamente por `python -m src.analizar`.*\n"]
    if len(t_completo):
        partes += ["### Test completo (25.000 reseñas)\n", _tabla_md(t_completo), ""]
    if len(t_sub):
        partes += [
            "### Submuestra común (1.000 reseñas, todos los modelos)\n",
            "El LLM se evalúa solo sobre esta submuestra por costo. Para compararlo en "
            "igualdad de condiciones, todos los modelos se evalúan también sobre las "
            "mismas 1.000 reseñas.\n",
            _tabla_md(t_sub),
            "",
            "![Accuracy por modelo](results/figuras/accuracy.png)\n",
        ]
    if len(t_mcnemar):
        partes += [
            "### ¿Las diferencias son significativas? (test de McNemar)\n",
            "Como todos los modelos se evalúan sobre las mismas reseñas, la comparación "
            "correcta es pareada: solo importan los casos en los que un modelo acierta y el "
            "otro no.\n",
            _tabla_md(t_mcnemar),
            "",
        ]
    partes += [
        "### Calibración\n",
        "¿Cuando un modelo dice 90%, acierta el 90% de las veces? El ECE (Expected "
        "Calibration Error) resume la distancia a la diagonal: más bajo es mejor.\n",
        "![Calibración](results/figuras/calibracion.png)\n",
        "### Costo\n",
        "![Accuracy vs. costo](results/figuras/costo.png)\n",
    ]
    if len(t_chal):
        chal = t_chal.copy()
        for c in chal.columns:
            chal[c] = chal[c].map(lambda v: f"{v:.0%}")
        chal.insert(0, "Modelo", chal.index)
        partes += [
            "### Challenge set (50 reseñas escritas a mano)\n",
            _tabla_md(chal),
            "",
            "![Challenge set](results/figuras/challenge.png)\n",
        ]
    bow = RESULTS_DIR / "bow_variantes.csv"
    if bow.exists():
        v = pd.read_csv(bow)
        v = pd.DataFrame(
            {
                "Preprocesamiento": v["variante"],
                "Vocabulario": v["vocabulario"].map(lambda x: f"{x:,}".replace(",", ".")),
                "Accuracy CV (train)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(v["cv_accuracy"], v["cv_desvio"])],
                "Accuracy test": v["test_accuracy"].map(lambda x: f"{x:.4f}"),
            }
        )
        partes += ["### Etapa 1 en detalle: efecto del preprocesamiento en BoW + Naive Bayes\n", _tabla_md(v), ""]
    return "\n".join(partes)


def actualizar_readme(md):
    readme = RAIZ / "README.md"
    if not readme.exists():
        return False
    texto = readme.read_text(encoding="utf-8")
    if INICIO not in texto or FIN not in texto:
        return False
    nuevo = re.sub(
        re.escape(INICIO) + r".*?" + re.escape(FIN),
        lambda _: f"{INICIO}\n{md}\n{FIN}",
        texto,
        flags=re.S,
    )
    readme.write_text(nuevo, encoding="utf-8")
    return True


def main():
    crear_directorios()
    preds = _cargar_todo()
    t_completo, t_sub, res_completo, res_sub = tablas(preds)
    t_mcnemar = comparaciones(preds)
    t_chal = tabla_challenge(preds)

    fig_accuracy(res_completo, res_sub)
    fig_calibracion(preds)
    fig_challenge(t_chal)
    fig_costo(preds, res_completo, res_sub)

    md = generar_markdown(t_completo, t_sub, t_mcnemar, t_chal)
    (RESULTS_DIR / "resultados.md").write_text(md, encoding="utf-8")
    actualizado = actualizar_readme(md)

    print(md)
    print(f"\nFiguras en {FIG_DIR}")
    print("README actualizado." if actualizado else "No se encontró la sección de resultados en el README.")
    return t_completo, t_sub, t_mcnemar, t_chal


if __name__ == "__main__":
    main()
