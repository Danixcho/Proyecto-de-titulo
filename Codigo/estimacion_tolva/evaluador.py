"""
evaluador.py  -  Modulo 4: metricas y estadistica
==================================================
Exactitud:    RMSE y MAE por corrida, despues del quemado.
Consistencia: NEES (requiere la verdad), NIS y autocorrelacion de la
              innovacion normalizada (calculables en planta).
Estadistica:  media con intervalo de confianza bootstrap; comparacion pareada
              entre dos estimadores (reduccion del cociente de medias con IC
              bootstrap pareado, mediana de la reduccion por corrida, fraccion
              de corridas ganadas y prueba de Wilcoxon).

El remuestreo bootstrap se siembra con los propios datos comparados
(_rng_de_datos): una misma comparacion produce siempre el mismo intervalo,
aunque aparezca en mas de un experimento. El argumento `rng` de resumen() y
comparar() se conserva por compatibilidad y no se usa.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import stats


def rmse(est: np.ndarray, verdad: np.ndarray, q: int = 0) -> np.ndarray:
    """RMSE por corrida sobre los pasos k >= q. Formas (M, N) -> (M,)."""
    err = np.atleast_2d(est)[:, q:] - np.atleast_2d(verdad)[:, q:]
    return np.sqrt(np.mean(err ** 2, axis=1))


def mae(est: np.ndarray, verdad: np.ndarray, q: int = 0) -> np.ndarray:
    err = np.atleast_2d(est)[:, q:] - np.atleast_2d(verdad)[:, q:]
    return np.mean(np.abs(err), axis=1)


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """NEES por corrida y paso: err (M, N, 2), P (N, 2, 2) -> (M, N).
    Valor esperado para un filtro consistente: 2 (dimension del estado)."""
    Pinv = np.linalg.inv(P)
    return np.einsum("mni,nij,mnj->mn", err, Pinv, err)


def nis(nu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """NIS por corrida y paso: nu (M, N), S (N,) -> (M, N). Valor esperado: 1."""
    return nu ** 2 / np.asarray(S)[None, :]


def autocorrelacion(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """Autocorrelacion muestral de rezago `lag` de cada fila. (M, n) -> (M,)."""
    x = np.atleast_2d(x)
    xc = x - x.mean(axis=1, keepdims=True)
    num = np.sum(xc[:, lag:] * xc[:, :-lag], axis=1)
    den = np.sum(xc ** 2, axis=1)
    return num / den


def _rng_de_datos(*arreglos) -> np.random.Generator:
    """Generador sembrado con el contenido de los datos (CRC32)."""
    h = 0
    for a in arreglos:
        h = zlib.crc32(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes(), h)
    return np.random.default_rng(h)


def resumen(x: np.ndarray, rng: np.random.Generator | None = None, B: int = 2000) -> dict:
    """Media con IC 95 % bootstrap percentil."""
    x = np.asarray(x, dtype=float)
    g = _rng_de_datos(x)
    idx = g.integers(0, len(x), size=(B, len(x)))
    medias = x[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return {"media": float(x.mean()), "ic_inf": float(lo), "ic_sup": float(hi),
            "mediana": float(np.median(x)), "n": int(len(x))}


def comparar(a: np.ndarray, b: np.ndarray, rng: np.random.Generator | None = None,
             B: int = 2000) -> dict:
    """Compara el estimador a (filtro) con el b (referencia) en las mismas corridas.

    reduccion    1 - media(a) / media(b): positiva si el filtro tiene menor error medio
    red_ic_*     IC 95 % bootstrap pareado (se remuestrean corridas, conservando el par)
    red_mediana  mediana de 1 - a/b por corrida: la corrida tipica
    gana_filtro  fraccion de corridas con a < b
    p_wilcoxon   prueba de rangos con signo, pareada, bilateral
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    g = _rng_de_datos(a, b)
    idx = g.integers(0, len(a), size=(B, len(a)))
    red_b = 1.0 - a[idx].mean(axis=1) / b[idx].mean(axis=1)
    lo, hi = np.percentile(red_b, [2.5, 97.5])
    dif = a - b
    p = float(stats.wilcoxon(a, b).pvalue) if np.any(dif != 0) else 1.0
    return {"reduccion": float(1.0 - a.mean() / b.mean()),
            "red_ic_inf": float(lo), "red_ic_sup": float(hi),
            "red_mediana": float(np.median(1.0 - a / b)),
            "gana_filtro": float(np.mean(a < b)),
            "p_wilcoxon": p, "n": int(len(a))}
