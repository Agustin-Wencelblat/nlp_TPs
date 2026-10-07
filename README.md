# De Bag of Words a LLMs: análisis de sentimiento en IMDB

¿Cuánto se gana realmente con cada salto en la historia del procesamiento del lenguaje natural, y a qué costo?

Este proyecto resuelve **una misma tarea** (clasificar reseñas de películas como positivas o negativas) con siete modelos que recorren esa historia: desde contar palabras hasta un LLM sin entrenamiento. Todos se evalúan **sobre las mismas reseñas**, con intervalos de confianza, tests pareados, calibración y un conjunto de casos difíciles escrito a mano.

**Datos:** [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (Maas et al., 2011). 25.000 reseñas para entrenar, 25.000 para evaluar, clases balanceadas, y 50.000 reseñas adicionales sin etiqueta.

**Stack:** Python · scikit-learn · NLTK · gensim · PyTorch · Hugging Face Transformers

\---

## Las etapas

|#|Modelo|Representación del texto|Qué muestra|
|-|-|-|-|
|1|BoW + Naive Bayes|Conteo de palabras|El baseline, y cuánto importa el preprocesamiento|
|2|TF-IDF + regresión logística|Términos ponderados, con bigramas|Qué tan lejos llega un modelo lineal bien armado|
|3|Word2Vec + regresión logística|Promedio de vectores densos|La semántica ayuda, pero promediar pierde el orden y la negación|
|4a|DistilBERT SST-2, sin ajuste|Embeddings contextuales|Cuánto se transfiere un clasificador entrenado en otro dataset|
|4b|BERT fine-tuneado|Embeddings contextuales|El salto que da el contexto cuando se entrena sobre la tarea|
|5|LLM zero-shot y few-shot|Prompt a un LLM instruccional|Clasificar sin entrenar nada, con otro perfil de costo|

## Cómo se evalúa

La mayoría de las comparaciones de este tipo terminan en una tabla de accuracy. Acá se agregan cinco cosas:

* **Submuestra común.** El LLM se evalúa sobre 1.000 reseñas por costo. Para compararlo de forma justa, *todos* los modelos se evalúan también sobre esas mismas 1.000 reseñas (fijadas con semilla en [`data/submuestra\_idx.csv`](data/submuestra_idx.csv)).
* **Intervalos de confianza por bootstrap.** Con 1.000 reseñas y una accuracy cerca de 0.9, el intervalo del 95% mide unos ±2 puntos: muchas diferencias que parecen claras no lo son.
* **Test de McNemar.** Como todos los modelos clasifican las mismas reseñas, la comparación correcta es pareada: solo cuentan los casos donde un modelo acierta y el otro no.
* **Calibración.** Además de acertar, ¿las probabilidades significan algo? Si un modelo dice 90%, debería acertar el 90% de esas veces. Se mide con el ECE (*Expected Calibration Error*) y diagramas de confiabilidad.
* **Challenge set.** [50 reseñas escritas a mano](data/challenge_set.csv) en cinco categorías difíciles: negación de algo positivo (*"not good at all"*), negación de algo negativo (*"not bad at all"*), contraste (*"…but…"*), sarcasmo y sentimiento implícito sin palabras de opinión (*"I walked out after forty minutes"*). Con 10 casos por categoría sirve para ver patrones, no para sacar conclusiones estadísticas.

Además se registra el **costo**: tiempo de entrenamiento y de inferencia, y en qué hardware corrió cada modelo.

## Resultados

<!-- RESULTADOS:INICIO -->

*Esta sección se genera automáticamente al correr el notebook* [*`notebooks/experimentos.ipynb`*](notebooks/experimentos.ipynb) *(o `python -m src.analizar`).*

<!-- RESULTADOS:FIN -->

## Decisiones de diseño

Algunas decisiones que no se ven en la tabla final pero cambian los resultados:

* **Las negaciones no son stopwords.** La lista de stopwords de NLTK incluye *not*, *no* y *nor*. Sacarlas convierte "not good" en "good", lo contrario de lo que se quiere en análisis de sentimiento. La etapa 1 compara ambas versiones.
* **Las contracciones se expanden antes de limpiar.** Si no, al borrar la puntuación "didn't" queda partido en "didn" + "t" y la negación se pierde.
* **El lematizador recibe la etiqueta gramatical.** Sin ella, `WordNetLemmatizer` trata todo como sustantivo: "running" no pasa a "run" ni "better" a "good".
* **El preprocesamiento se elige con validación cruzada sobre train.** Elegirlo mirando el test sería sobreajustar al test.
* **El LLM no genera texto: se leen sus probabilidades.** En vez de buscar "positive" en la respuesta (donde cualquier respuesta rara se cuenta como negativa), se compara la probabilidad que el modelo le asigna a *positive* y a *negative* como primera palabra. Así no hay respuestas imposibles de interpretar y se obtiene una probabilidad para medir calibración.
* **Los ejemplos few-shot están balanceados y son aleatorios.** El split de train viene ordenado por clase (primero los 12.500 negativos), así que tomar "los primeros 4" daría cuatro ejemplos negativos.
* **Se guardan las probabilidades de cada reseña, no solo las métricas.** Todo el análisis estadístico se hace después sobre esos archivos ([`results/predicciones/`](results/predicciones/)), sin volver a entrenar.

## Estructura del repo

```
nlp\_TPs/
├── notebooks/
│   └── experimentos.ipynb      # corre todo en Colab y actualiza este README
├── src/
│   ├── datos.py                # carga de IMDB, submuestra fija y challenge set
│   ├── preprocesamiento.py     # limpieza y variantes (stopwords, stemming, lematización, negación)
│   ├── modelos/
│   │   ├── clasicos.py         # etapas 1 a 3
│   │   ├── transformers\_gpu.py # etapas 4a y 4b
│   │   └── llm.py              # etapa 5
│   ├── evaluacion.py           # bootstrap, McNemar, ECE, curva de confiabilidad
│   ├── analizar.py             # tablas, gráficos y sección de resultados del README
│   └── correr.py               # corre las etapas desde la terminal
├── data/
│   ├── challenge\_set.csv       # 50 reseñas "difíciles" escritas a mano
│   └── submuestra\_idx.csv      # índices de la submuestra común de 1.000 reseñas
├── results/                    # predicciones, tiempos, tablas y figuras
└── tests/                      # prueba de punta a punta con datos sintéticos
```

## Cómo reproducirlo

**En Colab (recomendado):** abrí [`notebooks/experimentos.ipynb`](notebooks/experimentos.ipynb) en Colab, elegí una GPU T4 y corré todas las celdas. 

**Localmente:**

```bash
pip install -r requirements.txt
python -m src.correr --etapas bow tfidf w2v   # modelos que corren en CPU
python -m src.correr --etapas sst2 bert llm   # requieren GPU para tiempos razonables
python -m src.analizar                        # tablas, gráficos y README
pytest                                        # prueba de punta a punta con datos sintéticos
```

## Limitaciones

* BERT ve como máximo 256 tokens por reseña; el resto se trunca. Algunas reseñas de IMDB son bastante más largas.
* Cada modelo se entrena con una sola semilla, así que los intervalos reflejan la variabilidad del test, no la del entrenamiento.
* El LLM es un modelo abierto chico (1.500 millones de parámetros) para que entre en una GPU gratuita. Un modelo más grande probablemente rinda más.
* El challenge set tiene 10 reseñas por categoría: muestra tendencias, no diferencias significativas.

