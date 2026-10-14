"""
sensor.py  -  Modulo 2: sensor de nivel (la medicion)
======================================================
Transforma el volumen verdadero en la medicion que recibe el estimador
(Seccion 4.4 de la tesis):

    y[k] = V[k] + b[k] + sigma[k] n[k]

con tres mecanismos de discrepancia que se activan por separado:

  - Ruido coloreado: n es AR(1) con varianza marginal unitaria,
        n[k] = rho n[k-1] + sqrt(1 - rho^2) xi[k],  n[0] = xi[0].
    Con rho = 0 es ruido blanco. Mantener la varianza marginal fija aisla el
    efecto de la correlacion del de la magnitud.
  - Ruido intensificado en las descargas: sigma[k] = f_d sigma_v mientras un
    camion descarga y sigma_v en otro caso.
  - Deriva lineal: b[k] = b_N k / N.

En la condicion nominal (rho = 0, f_d = 1, b_N = 0) el error es blanco,
gaussiano y de desviacion sigma_v, como supone el filtro. La salida no se acota
al rango fisico.

Las normales xi se generan una sola vez por corrida y se reutilizan en todas
las condiciones de un experimento (numeros aleatorios comunes): las
diferencias entre condiciones se deben solo a la condicion.
"""

from __future__ import annotations

import numpy as np

from parametros import ParametrosSensor


def medir_desde_eps(V: np.ndarray, descargando: np.ndarray, eps: np.ndarray,
                    ps: ParametrosSensor, dt: float = 1.0) -> np.ndarray:
    """Transformacion determinista de las normales `eps` en la medicion y.

    V, descargando y eps tienen forma (M, N) o (N,)."""
    V = np.atleast_2d(V)
    D = np.atleast_2d(descargando)
    eps = np.atleast_2d(eps)
    M, N = V.shape

    if ps.rho == 0.0:
        n = eps
    else:
        n = np.empty_like(eps)
        c = np.sqrt(1.0 - ps.rho**2)
        n[:, 0] = eps[:, 0]                  # arranque estacionario: varianza 1
        for k in range(1, N):
            n[:, k] = ps.rho * n[:, k - 1] + c * eps[:, k]

    sigma = np.where(D, ps.sigma_v * ps.factor_descarga, ps.sigma_v)
    sesgo = ps.deriva_total * (np.arange(N) / N)[None, :]
    return V + sesgo + sigma * n


def medir(V: np.ndarray, descargando: np.ndarray, ps: ParametrosSensor,
          rng: np.random.Generator, dt: float = 1.0) -> np.ndarray:
    """Conveniencia: genera las normales y aplica la transformacion."""
    eps = rng.standard_normal(np.atleast_2d(V).shape)
    return medir_desde_eps(V, descargando, eps, ps, dt)
