*Sección generada automáticamente por `python -m src.analizar`.*

### Test completo (25.000 reseñas)

| Modelo | Accuracy [IC 95%] | F1 | ECE | Inferencia (s / 1000 reseñas) | Entrenamiento (s) | Hardware |
|---|---|---|---|---|---|---|
| BoW + Naive Bayes | 0.8303 [0.8258, 0.8349] | 0.8206 | 0.138 | 0.09 | 2 | CPU |
| TF-IDF + Reg. logística | 0.9004 [0.8968, 0.9042] | 0.9009 | 0.122 | 0.51 | 20 | CPU |
| Word2Vec + Reg. logística | 0.8458 [0.8412, 0.8499] | 0.8440 | 0.058 | 0.48 | 431 | CPU |
| DistilBERT SST-2 (sin ajuste) | 0.8908 [0.8870, 0.8947] | 0.8876 | 0.084 | 3.01 | 0 | GPU (Tesla T4) |
| BERT fine-tuneado | 0.9232 [0.9200, 0.9266] | 0.9236 | 0.051 | 4.16 | 787 | GPU (Tesla T4) |

### Submuestra común (1.000 reseñas, todos los modelos)

El LLM se evalúa solo sobre esta submuestra por costo. Para compararlo en igualdad de condiciones, todos los modelos se evalúan también sobre las mismas 1.000 reseñas.

| Modelo | Accuracy [IC 95%] | F1 | ECE |
|---|---|---|---|
| BoW + Naive Bayes | 0.8480 [0.8270, 0.8700] | 0.8390 | 0.123 |
| TF-IDF + Reg. logística | 0.9170 [0.9000, 0.9340] | 0.9177 | 0.135 |
| Word2Vec + Reg. logística | 0.8650 [0.8450, 0.8870] | 0.8629 | 0.081 |
| DistilBERT SST-2 (sin ajuste) | 0.8830 [0.8620, 0.9020] | 0.8802 | 0.092 |
| BERT fine-tuneado | 0.9350 [0.9200, 0.9500] | 0.9355 | 0.042 |
| LLM zero-shot | 0.9380 [0.9240, 0.9520] | 0.9375 | 0.052 |
| LLM few-shot | 0.9420 [0.9280, 0.9560] | 0.9415 | 0.042 |

![Accuracy por modelo](results/figuras/accuracy.png)

### ¿Las diferencias son significativas? (test de McNemar)

Como todos los modelos se evalúan sobre las mismas reseñas, la comparación correcta es pareada: solo importan los casos en los que un modelo acierta y el otro no.

| Comparación | Conjunto | Solo acierta el 1º | Solo acierta el 2º | p-valor |
|---|---|---|---|---|
| BoW + Naive Bayes vs. TF-IDF + Reg. logística | Test completo | 785 | 2538 | < 1e-16 |
| TF-IDF + Reg. logística vs. Word2Vec + Reg. logística | Test completo | 2066 | 699 | < 1e-16 |
| Word2Vec + Reg. logística vs. DistilBERT SST-2 (sin ajuste) | Test completo | 1524 | 2650 | < 1e-16 |
| DistilBERT SST-2 (sin ajuste) vs. BERT fine-tuneado | Test completo | 871 | 1682 | < 1e-16 |
| TF-IDF + Reg. logística vs. LLM zero-shot | Submuestra | 35 | 56 | 0.036 |
| BERT fine-tuneado vs. LLM zero-shot | Submuestra | 30 | 33 | 0.8 |
| TF-IDF + Reg. logística vs. LLM few-shot | Submuestra | 31 | 56 | 0.01 |
| BERT fine-tuneado vs. LLM few-shot | Submuestra | 26 | 33 | 0.43 |
| LLM zero-shot vs. LLM few-shot | Submuestra | 7 | 11 | 0.48 |

### Calibración

¿Cuando un modelo dice 90%, acierta el 90% de las veces? El ECE (Expected Calibration Error) resume la distancia a la diagonal: más bajo es mejor.

![Calibración](results/figuras/calibracion.png)

### Costo

![Accuracy vs. costo](results/figuras/costo.png)

### Challenge set (50 reseñas escritas a mano)

| Modelo | Negación de positivo | Negación de negativo | Contraste (pero...) | Sarcasmo | Sentimiento implícito | Total |
|---|---|---|---|---|---|---|
| BoW + Naive Bayes | 90% | 10% | 50% | 70% | 80% | 60% |
| TF-IDF + Reg. logística | 90% | 10% | 90% | 50% | 90% | 66% |
| Word2Vec + Reg. logística | 50% | 0% | 50% | 50% | 90% | 48% |
| DistilBERT SST-2 (sin ajuste) | 90% | 70% | 80% | 20% | 80% | 68% |
| BERT fine-tuneado | 90% | 70% | 100% | 30% | 90% | 76% |
| LLM zero-shot | 100% | 100% | 100% | 30% | 100% | 86% |
| LLM few-shot | 100% | 100% | 100% | 40% | 100% | 88% |

![Challenge set](results/figuras/challenge.png)

### Etapa 1 en detalle: efecto del preprocesamiento en BoW + Naive Bayes

| Preprocesamiento | Vocabulario | Accuracy CV (train) | Accuracy test |
|---|---|---|---|
| Solo limpieza | 44.200 | 0.8462 ± 0.0031 | 0.8140 |
| Sin stopwords (lista NLTK completa) | 44.064 | 0.8606 ± 0.0037 | 0.8253 |
| Sin stopwords, conservando negaciones | 44.067 | 0.8600 ± 0.0035 | 0.8255 |
| + Stemming | 29.172 | 0.8544 ± 0.0042 | 0.8167 |
| + Lematización con POS | 36.107 | 0.8548 ± 0.0021 | 0.8193 |
| + Marcado de negación | 55.290 | 0.8612 ± 0.0022 | 0.8303 |
