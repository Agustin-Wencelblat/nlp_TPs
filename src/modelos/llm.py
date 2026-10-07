"""
Etapa 5: un LLM instruccional usado como clasificador, sin entrenamiento.

En lugar de generar texto y buscar "POSITIVE" en la respuesta (donde cualquier
respuesta rara se cuenta mal), se mira directamente la distribución del modelo
sobre el PRIMER token de la respuesta y se comparan las probabilidades de
"positive" y "negative":

    p_pos = P(positive) / (P(positive) + P(negative))

Esto nos da una probabilidad para cada reseña (necesaria para medir calibración) y
elimina las respuestas imposibles de interpretar. Como control, se registra
también cuánta probabilidad total pone el modelo en esas dos palabras: si es
baja, el modelo "quería" responder otra cosa.

Por defecto vamos a usar un modelo abierto chico que entra en una T4 de Colab, así no hace
falta ninguna API key.
"""

import time

import numpy as np
import torch

from ..config import SEED
from ..registro import guardar_predicciones, guardar_tiempos, hardware_actual
from .transformers_gpu import DEVICE

MODELO_POR_DEFECTO = "Qwen/Qwen2.5-1.5B-Instruct"

INSTRUCCION = (
    "You are a sentiment classifier for movie reviews. "
    "Read the review and answer with exactly one word: positive or negative."
)


def _mensajes(texto, ejemplos, max_chars):
    msgs = [{"role": "system", "content": INSTRUCCION}]
    for ej_texto, ej_label in ejemplos:
        msgs.append({"role": "user", "content": f"Review: {ej_texto}"})
        msgs.append({"role": "assistant", "content": "positive" if ej_label == 1 else "negative"})
    msgs.append({"role": "user", "content": f"Review: {texto[:max_chars]}"})
    return msgs


def _ids_etiqueta(tokenizer, palabra):
    """Primer token de las formas habituales de la palabra ("positive", " Positive", ...)."""
    ids = set()
    for forma in (palabra, palabra.capitalize(), " " + palabra, " " + palabra.capitalize()):
        tok = tokenizer.encode(forma, add_special_tokens=False)
        if tok:
            ids.add(tok[0])
    return sorted(ids)


def elegir_ejemplos_few_shot(train, k_por_clase=2, max_chars=600, seed=SEED):
    """
    Ejemplos balanceados (k por clase) elegidos al azar de TRAIN. Ojo: tomar las
    primeras filas daría solo negativos, porque el split viene ordenado por clase.
    """
    rng = np.random.default_rng(seed)
    elegidos = []
    for label in (1, 0):
        candidatos = train.index[(train["label"] == label) & (train["text"].str.len() < max_chars)]
        for i in rng.choice(candidatos, k_por_clase, replace=False):
            elegidos.append((train.loc[i, "text"], int(train.loc[i, "label"])))
    rng.shuffle(elegidos)
    return elegidos


@torch.no_grad()
def clasificar(modelo, tokenizer, textos, ejemplos=(), batch_size=8, max_chars=2000):
    """Devuelve (p_pos, masa_etiquetas) para cada texto."""
    modelo.eval()
    ids_pos = _ids_etiqueta(tokenizer, "positive")
    ids_neg = _ids_etiqueta(tokenizer, "negative")
    prompts = [
        tokenizer.apply_chat_template(
            _mensajes(t, ejemplos, max_chars), tokenize=False, add_generation_prompt=True
        )
        for t in textos
    ]
    orden = np.argsort([len(p) for p in prompts])
    p_pos = np.empty(len(prompts), dtype=np.float32)
    masa = np.empty(len(prompts), dtype=np.float32)
    for i in range(0, len(prompts), batch_size):
        idx = orden[i : i + batch_size]
        enc = tokenizer(
            [prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(DEVICE)
        try:
            # Solo hacen falta los logits de la última posición (padding a la izquierda).
            # Pedir solo esos ahorra mucha memoria con vocabularios grandes.
            salida = modelo(**enc, logits_to_keep=1)
        except TypeError:
            salida = modelo(**enc)
        logits = salida.logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        lp_pos = torch.logsumexp(logp[:, ids_pos], dim=-1)
        lp_neg = torch.logsumexp(logp[:, ids_neg], dim=-1)
        p_pos[idx] = torch.sigmoid(lp_pos - lp_neg).cpu().numpy()
        masa[idx] = (lp_pos.exp() + lp_neg.exp()).cpu().numpy()
    return p_pos, masa


def cargar_llm(modelo_id=MODELO_POR_DEFECTO):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(modelo_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id, dtype=dtype).to(DEVICE)
    return modelo, tokenizer


def correr_llm(
    datos,
    challenge,
    idx_submuestra,
    modelo_id=MODELO_POR_DEFECTO,
    modelo=None,
    tokenizer=None,
    batch_size=8,
):
    """Evalúa zero-shot y few-shot sobre la submuestra fija del test y el challenge set."""
    if modelo is None:
        modelo, tokenizer = cargar_llm(modelo_id)
    test = datos["test"].loc[idx_submuestra]
    ejemplos = elegir_ejemplos_few_shot(datos["train"])

    for slug, ej, bs in (
        ("llm_zeroshot", [], batch_size),
        ("llm_fewshot", ejemplos, max(1, batch_size // 2)),
    ):
        t0 = time.time()
        p_test, masa = clasificar(modelo, tokenizer, test["text"], ej, batch_size=bs)
        t_inf = time.time() - t0
        p_chal, _ = clasificar(modelo, tokenizer, challenge["text"], ej, batch_size=bs)
        guardar_predicciones(slug, test["label"], p_test, test.index, challenge["label"], p_chal)
        guardar_tiempos(
            slug,
            0,
            t_inf,
            len(test),
            hardware_actual(),
            {
                "modelo": modelo_id,
                "n_ejemplos_few_shot": len(ej),
                "masa_en_etiquetas_media": round(float(masa.mean()), 4),
            },
        )
        acc = ((p_test >= 0.5) == test["label"].to_numpy()).mean()
        print(f"  {slug}: accuracy submuestra = {acc:.4f} | masa media en etiquetas = {masa.mean():.3f}")
