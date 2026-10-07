"""
Etapas 4 y 5: modelos basados en transformers (pensados para una GPU de Colab).

4a. DistilBERT ya entrenado en SST-2, aplicado a IMDB sin ningún ajuste
    (mide cuánto se transfiere un clasificador de sentimiento a otro dominio).
4b. BERT fine-tuneado sobre IMDB.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import SEED
from ..registro import guardar_predicciones, guardar_tiempos, hardware_actual

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fijar_semillas(seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predecir_proba(modelo, tokenizer, textos, indice_positivo, max_len=512, batch_size=64):
    """
    Probabilidad de la clase positiva para cada texto. Ordena por longitud para que
    cada batch tenga textos parecidos y el padding sea mínimo (mucho más rápido),
    y después devuelve los resultados en el orden original.
    """
    modelo.eval()
    textos = list(textos)
    orden = np.argsort([len(t) for t in textos])
    probs = np.empty(len(textos), dtype=np.float32)
    usar_amp = DEVICE == "cuda"
    for i in range(0, len(textos), batch_size):
        idx = orden[i : i + batch_size]
        enc = tokenizer(
            [textos[j] for j in idx],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            logits = modelo(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1)[:, indice_positivo].cpu().numpy()
    return probs


# ─── Etapa 4a ─────────────────────────────────────────────────────────────────
def correr_sst2(datos, challenge, modelo_id="distilbert-base-uncased-finetuned-sst-2-english"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    test = datos["test"]
    tokenizer = AutoTokenizer.from_pretrained(modelo_id)
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_id).to(DEVICE)
    etiquetas = {v.upper(): k for k, v in modelo.config.id2label.items()}
    pos = etiquetas["POSITIVE"]

    t0 = time.time()
    p_test = predecir_proba(modelo, tokenizer, test["text"], pos)
    t_inf = time.time() - t0
    p_chal = predecir_proba(modelo, tokenizer, challenge["text"], pos)

    guardar_predicciones("sst2_transfer", test["label"], p_test, test.index, challenge["label"], p_chal)
    guardar_tiempos("sst2_transfer", 0, t_inf, len(test), hardware_actual(), {"modelo": modelo_id})
    print(f"  accuracy test = {((p_test >= 0.5) == test['label']).mean():.4f}")


# ─── Etapa 4b ─────────────────────────────────────────────────────────────────
def correr_bert(
    datos,
    challenge,
    modelo_id="bert-base-uncased",
    max_len=256,
    batch_size=16,
    epocas=2,
    lr=2e-5,
    modelo=None,
    tokenizer=None,
):
    """
    Fine-tuning completo de BERT con AdamW, warmup lineal del 10% y clipping de
    gradiente. Usa precisión mixta (float16) en GPU, que en una T4 reduce el tiempo
    a menos de la mitad sin cambiar en la práctica la accuracy, y padding dinámico
    (cada batch se rellena solo hasta su texto más largo).

    `modelo` y `tokenizer` se pueden pasar ya construidos (se usa en los tests).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    _fijar_semillas()
    train, test = datos["train"], datos["test"]
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(modelo_id)
    modelo = modelo or AutoModelForSequenceClassification.from_pretrained(modelo_id, num_labels=2)
    modelo.to(DEVICE)

    pares = list(zip(train["text"], train["label"]))

    def collate(batch):
        textos, etiquetas = zip(*batch)
        enc = tokenizer(
            list(textos), truncation=True, max_length=max_len, padding=True, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(etiquetas, dtype=torch.long)
        return enc

    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(pares, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=gen)
    pasos = len(loader) * epocas
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    usar_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    print(f"  Entrenando {modelo_id}: {epocas} épocas, {pasos:,} pasos, dispositivo={DEVICE}")
    t0 = time.time()
    for epoca in range(epocas):
        modelo.train()
        perdida = 0.0
        for paso, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
                loss = modelo(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            perdida += loss.item()
            if paso % 200 == 0:
                print(
                    f"    época {epoca + 1}/{epocas} | paso {paso:>5}/{len(loader)} "
                    f"| loss {perdida / (paso + 1):.4f} | {time.time() - t0:.0f}s"
                )
    t_train = time.time() - t0

    t0 = time.time()
    p_test = predecir_proba(modelo, tokenizer, test["text"], 1, max_len=max_len)
    t_inf = time.time() - t0
    p_chal = predecir_proba(modelo, tokenizer, challenge["text"], 1, max_len=max_len)

    guardar_predicciones("bert_ft", test["label"], p_test, test.index, challenge["label"], p_chal)
    guardar_tiempos(
        "bert_ft",
        t_train,
        t_inf,
        len(test),
        hardware_actual(),
        {"modelo": modelo_id, "max_len": max_len, "epocas": epocas, "lr": lr},
    )
    print(f"  accuracy test = {((p_test >= 0.5) == test['label']).mean():.4f}")
    return modelo
