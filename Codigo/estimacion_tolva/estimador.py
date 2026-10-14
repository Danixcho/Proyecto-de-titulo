"""
estimador.py  -  Modulo 3: filtro de Kalman lineal de dos estados
==================================================================
Estado x = [V, e] (volumen y tasa de extraccion). Modelo del filtro
(Seccion 4.5 de la tesis):

    x[k] = A x[k-1] + B u[k-1] + w[k-1],   w ~ N(0, Q)
    y[k] = H x[k] + v[k],                  v ~ N(0, R)

En k = 0 el filtro incorpora y[0] sin prediccion previa, a partir de (x0, P0).
En cada paso aplica la prediccion y la actualizacion con la forma de Joseph de
la covarianza, que conserva la simetria y la definicion positiva.

Dos implementaciones con las mismas operaciones:

  FiltroKalman        por lotes: filtra M corridas a la vez. Como A, H, Q y R
                      son constantes, la secuencia P[k], K[k] no depende de las
                      mediciones y se calcula una sola vez.
  EstimadorEnLinea    una medicion por llamada, como operaria conectado a un
                      flujo de datos (por ejemplo, un SCADA). La prueba H del
                      Experimento EV comprueba que ambas coinciden.

El filtro solo recibe la medicion y la descarga conocida u; nunca la verdad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parametros import ParametrosEstimador


@dataclass
class ResultadoFiltro:
    x: np.ndarray        # (M, N, 2) estimacion a posteriori
    P: np.ndarray        # (N, 2, 2) covarianza a posteriori
    P_pred: np.ndarray   # (N, 2, 2) covarianza a priori
    nu: np.ndarray       # (M, N) innovacion y[k] - H x_pred[k]
    S: np.ndarray        # (N,) varianza de la innovacion
    K: np.ndarray        # (N, 2) ganancia


def _actualizar_cov(P_pred: np.ndarray, H: np.ndarray, R: float):
    """Ganancia y covarianza a posteriori (forma de Joseph)."""
    S = float((H @ P_pred @ H.T)[0, 0]) + R
    K = (P_pred @ H.T).ravel() / S
    IKH = np.eye(2) - np.outer(K, H.ravel())
    P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
    return K, S, P


class FiltroKalman:
    """Filtro de Kalman por lotes."""

    def __init__(self, pe: ParametrosEstimador):
        self.pe = pe

    def covarianzas(self, N: int):
        """Secuencias de ganancia, covarianzas y varianza de la innovacion."""
        pe = self.pe
        A, H, Q, R = pe.A, pe.H, pe.Q, pe.R
        K = np.empty((N, 2))
        S = np.empty(N)
        P = np.empty((N, 2, 2))
        P_pred = np.empty((N, 2, 2))
        Pp = pe.P0.copy()
        for k in range(N):
            if k > 0:
                Pp = A @ P[k - 1] @ A.T + Q
            P_pred[k] = Pp
            K[k], S[k], P[k] = _actualizar_cov(Pp, H, R)
        return K, S, P, P_pred

    def filtrar(self, Y: np.ndarray, U: np.ndarray) -> ResultadoFiltro:
        """Y, U con forma (M, N) o (N,). u[k] actua entre k y k+1."""
        Y = np.atleast_2d(Y)
        U = np.atleast_2d(U)
        M, N = Y.shape
        pe = self.pe
        A, B = pe.A, pe.B
        K, S, P, P_pred = self.covarianzas(N)
        x = np.empty((M, N, 2))
        nu = np.empty((M, N))
        xp = np.tile(pe.x0, (M, 1))
        for k in range(N):
            if k > 0:
                xp = x[:, k - 1] @ A.T + U[:, k - 1, None] * B[None, :]
            nu[:, k] = Y[:, k] - xp[:, 0]
            x[:, k] = xp + nu[:, k, None] * K[k][None, :]
        return ResultadoFiltro(x=x, P=P, P_pred=P_pred, nu=nu, S=S, K=K)


class EstimadorEnLinea:
    """Interfaz en linea: una medicion por llamada.

    Uso:
        est = EstimadorEnLinea(pe)
        for k, y in enumerate(mediciones):
            r = est.actualizar(y, u_anterior=u[k-1] if k > 0 else None)
            r["V"], r["e"], r["de_V"], r["de_e"], r["innovacion"], r["nis"]

    `u_anterior` es la descarga conocida del paso anterior, que actuo entre la
    medicion previa y esta. En la primera llamada no hay prediccion.
    """

    def __init__(self, pe: ParametrosEstimador):
        self.pe = pe
        self.x = pe.x0.copy()
        self.P = pe.P0.copy()
        self.k = 0

    def actualizar(self, y: float, u_anterior: float | None = None) -> dict:
        pe = self.pe
        A, B, H, Q, R = pe.A, pe.B, pe.H, pe.Q, pe.R
        if self.k == 0:
            xp, Pp = self.x, self.P
        else:
            xp = self.x @ A.T + (0.0 if u_anterior is None else u_anterior) * B
            Pp = A @ self.P @ A.T + Q
        K, S, P = _actualizar_cov(Pp, H, R)
        nu = float(y) - xp[0]
        self.x = xp + nu * K
        self.P = P
        self.k += 1
        return {"V": float(self.x[0]), "e": float(self.x[1]),
                "de_V": float(np.sqrt(P[0, 0])), "de_e": float(np.sqrt(P[1, 1])),
                "innovacion": float(nu), "nis": float(nu * nu / S)}
