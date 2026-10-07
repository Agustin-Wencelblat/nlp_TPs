"""
Limpieza y variantes de preprocesamiento para los modelos clásicos.
"""

import re
from functools import lru_cache

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

NEGACIONES = {"not", "no", "nor", "never"}
FIN_DE_FRASE = "<p>"  # marcador interno de puntuación, se usa para el alcance de la negación


@lru_cache(maxsize=1)
def _stopwords():
    completa = set(stopwords.words("english"))
    return completa, completa - NEGACIONES


_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def tokenizar(texto: str, conservar_puntuacion: bool = False) -> list[str]:
    t = texto.lower()
    t = re.sub(r"<[^>]+>", " ", t)  # IMDB tiene <br /> en el texto
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"n't\b", " not", t)
    if conservar_puntuacion:
        t = re.sub(r"[.,!?;:()]", f" {FIN_DE_FRASE} ", t)
        t = re.sub(r"[^a-z<>\s]", " ", t)
        return [w for w in t.split() if w.isalpha() or w == FIN_DE_FRASE]
    t = re.sub(r"[^a-z\s]", " ", t)
    return t.split()


def _pos_wordnet(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def marcar_negacion(tokens: list[str]) -> list[str]:
    """
    Después de una negación, se agrega el prefijo NOT_ a cada palabra hasta el siguiente signo
    de puntuación. Así "not good" y "good" pasan a ser atributos distintos.
    """
    salida, negando = [], False
    for w in tokens:
        if w == FIN_DE_FRASE:
            negando = False
        elif w in NEGACIONES:
            negando = True
            salida.append(w)
        else:
            salida.append(f"NOT_{w}" if negando else w)
    return salida


def solo_limpieza(texto):
    return " ".join(tokenizar(texto))


def stop_completa(texto):
    completa, _ = _stopwords()
    return " ".join(w for w in tokenizar(texto) if w not in completa)


def stop_sin_neg(texto):
    _, sin_neg = _stopwords()
    return " ".join(w for w in tokenizar(texto) if w not in sin_neg)


def stemming(texto):
    _, sin_neg = _stopwords()
    return " ".join(_stemmer.stem(w) for w in tokenizar(texto) if w not in sin_neg)


def lematizacion(texto):
    # Se etiqueta la oración completa (el tagger necesita el contexto) y después
    # se filtran las stopwords.
    _, sin_neg = _stopwords()
    etiquetados = pos_tag(tokenizar(texto))
    return " ".join(
        _lemmatizer.lemmatize(w, _pos_wordnet(tag)) for w, tag in etiquetados if w not in sin_neg
    )


def negacion_marcada(texto):
    _, sin_neg = _stopwords()
    tokens = [w for w in tokenizar(texto, conservar_puntuacion=True) if w not in sin_neg]
    return " ".join(marcar_negacion(tokens))


# Nombre para mostrar -> función. El orden es el de la tabla del README.
VARIANTES = {
    "Solo limpieza": solo_limpieza,
    "Sin stopwords (lista NLTK completa)": stop_completa,
    "Sin stopwords, conservando negaciones": stop_sin_neg,
    "+ Stemming": stemming,
    "+ Lematización con POS": lematizacion,
    "+ Marcado de negación": negacion_marcada,
}
