"""
referencias.py  -  Modulo 3 (continuacion): Estimadores de referencia
=====================================================================
Estimadores SIN modelo dinamico, contra los que se compara el filtro. Todos son
causales (en el paso k solo usan mediciones hasta k). La regresion local y las
diferencias finitas usan la descarga conocida de camiones u, como el filtro; la
medicion cruda y la media movil no la usan, y por eso sirven solo como
referencias simples. La auditoria del anteproyecto mostro que comparar contra un
metodo que ignora informacion disponible infla artificialmente la ventaja del
filtro: la regresion local es el competidor exigente.

Volumen:
  - medicion cruda          y[k]
  - media movil             promedio de las ultimas W mediciones. Es el
                            "amortiguamiento" que ofrece cualquier transmisor.
  - regresion local         recta de minimos cuadrados sobre las ultimas W
                            mediciones, evaluada en k. Sigue las rampas sin el
                            retardo de la media movil.

Tasa de extraccion (no medida):
  - diferencias finitas     e[k] = u[k-1] - (y[k] - y[k-1]) / dt
  - regresion local         e[k] = -pendiente de la recta anterior

Para la regresion se descuenta primero el volumen aportado por los camiones:

    z[k] = y[k] - dt * sum_{j<k} u[j]

de modo que z solo cambia por la extraccion y la pendiente de z es -e.

Al comienzo de cada corrida, mientras k + 1 < W, las ventanas usan las k + 1
mediciones disponibles. Las ventanas W no se eligen a mano: se ajustan sobre
corridas de calibracion separadas de las corridas de prueba (ver
experimentos.calibrar_ventanas).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def _suma_ventana(C: np.ndarray, W: int) -> np.ndarray:
    """Dada la suma acumulada C (con C[:, 0] = 0), devuelve la suma de cada
    ventana causal de largo min(W, k + 1) que termina en k."""
    N = C.shape[1] - 1
    k = np.arange(N)
    ini = np.maximum(k + 1 - W, 0)
    return C[:, k + 1] - C[:, ini]


def media_movil(Y: np.ndarray, W: int) -> np.ndarray:
    """Promedio causal de las ultimas W mediciones."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    C = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Y, axis=1)], axis=1)
    n = np.minimum(np.arange(Y.shape[1]) + 1, W)
    return _suma_ventana(C, W) / n[None, :]


def aporte_camiones(U: np.ndarray, dt: float) -> np.ndarray:
    """Volumen aportado por los camiones antes de cada paso: dt * sum_{j<k} u[j]."""
    U = np.atleast_2d(U)
    return dt * np.concatenate([np.zeros((U.shape[0], 1)), np.cumsum(U, axis=1)[:, :-1]], axis=1)


def _pesos_recta(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pesos de la recta de minimos cuadrados sobre n puntos, indexados por el
    retardo i = 0..n-1 (i = 0 es la medicion actual). Devuelven el valor de la
    recta en el instante actual y su pendiente por paso."""
    if n == 1:
        return np.array([1.0]), np.array([0.0])
    s = -np.arange(n, dtype=float)          # abscisa relativa al instante actual
    sc = s - s.mean()
    pend = sc / np.sum(sc ** 2)
    nivel = 1.0 / n - s.mean() * pend
    return nivel, pend


def regresion_local(Y: np.ndarray, U: np.ndarray, dt: float, W: int):
    """Recta de minimos cuadrados sobre las ultimas W mediciones de z.

    Devuelve (V_est, e_est), ambos con forma (M, N)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(U)
    ac = aporte_camiones(U, dt)
    z = Y - ac
    M, N = z.shape
    nivel, pend = _pesos_recta(W)
    z_est = lfilter(nivel, [1.0], z, axis=1)
    p_est = lfilter(pend, [1.0], z, axis=1)
    # Primeros pasos: ventana con las k + 1 mediciones disponibles.
    for k in range(min(W - 1, N)):
        nv, pd = _pesos_recta(k + 1)
        tramo = z[:, k::-1]                  # z[k], z[k-1], ..., z[0]
        z_est[:, k] = tramo @ nv
        p_est[:, k] = tramo @ pd
    return z_est + ac, -p_est / dt


def diferencias_finitas(Y: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    """e[k] = u[k-1] - (y[k] - y[k-1]) / dt; en k = 0 no hay diferencia y se
    deja 0 (queda dentro del quemado)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(U)
    e = np.zeros_like(Y)
    e[:, 1:] = U[:, :-1] - (Y[:, 1:] - Y[:, :-1]) / dt
    return e
