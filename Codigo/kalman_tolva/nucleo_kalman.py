"""
nucleo_kalman.py
================
Nucleo algoritmico del modulo de software analitico: Filtro de Kalman Lineal
Discreto en su formulacion predictor-corrector.

Este modulo es deliberadamente generico y autocontenido: no conoce nada del
proceso de chancado primario. Recibe las matrices del sistema y opera. Esa
independencia es lo que permite reutilizarlo sobre distintos subsistemas
ciberfisicos (volumen de tolva, potencia activa, temperatura, desgaste) sin
modificar una sola linea del estimador, y es la evidencia del caracter
desacoplado del modulo declarado en el alcance del proyecto.

Proyecto : Diseno y validacion mediante simulacion computacional de un modulo
           de software analitico para la estimacion de estados en un gemelo
           digital de chancado primario.
Autor    : Daniel Alejandro Prado Correa
Asignatura: Proyecto de Investigacion (ACIF3002) - Universidad Andres Bello
"""

from __future__ import annotations

import numpy as np


class FiltroKalman:
    """Filtro de Kalman Lineal Discreto.

    Implementa el ciclo recursivo de dos etapas sobre el sistema en espacio de
    estados en tiempo discreto:

        x_{k+1} = A x_k + B u_k + w_k ,   w_k ~ N(0, Q)
        y_k     = H x_k + v_k         ,   v_k ~ N(0, R)

    Prediccion (a priori):
        x_{k|k-1} = A x_{k-1|k-1} + B u_k
        P_{k|k-1} = A P_{k-1|k-1} A^T + Q

    Correccion (a posteriori):
        S_k = H P_{k|k-1} H^T + R                (covarianza de innovacion)
        K_k = P_{k|k-1} H^T S_k^{-1}             (ganancia de Kalman)
        x_{k|k} = x_{k|k-1} + K_k (y_k - H x_{k|k-1})
        P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T

    La actualizacion de la covarianza usa la forma de Joseph en lugar de la
    forma simplificada (I - K H) P. Ambas son algebraicamente equivalentes,
    pero la de Joseph preserva la simetria y la definicion positiva de P frente
    a errores de redondeo en aritmetica de punto flotante, lo que garantiza la
    estabilidad numerica del filtro en horizontes largos.

    Parametros
    ----------
    A : (n, n) matriz de transicion de estados.
    B : (n, m) matriz de entrada de control.
    H : (p, n) matriz de observacion.
    Q : (n, n) covarianza del ruido de proceso.
    R : (p, p) covarianza del ruido de medicion.
    x0 : (n, 1) estimacion inicial del vector de estado.
    P0 : (n, n) covarianza inicial del error de estimacion.
    """

    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))

        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.atleast_2d(np.asarray(P0, dtype=float))

        n = self.A.shape[0]
        self.I = np.eye(n)

        self._validar_dimensiones(n)

        # Trazabilidad: se registran ganancia e innovacion de cada ciclo para
        # poder evidenciar la convergencia asintotica exigida por los criterios
        # de calidad del estudio.
        self.historial_K: list[np.ndarray] = []
        self.historial_P: list[np.ndarray] = []
        self.historial_innovacion: list[float] = []

    def _validar_dimensiones(self, n: int) -> None:
        if self.A.shape != (n, n):
            raise ValueError(f"A debe ser cuadrada (n,n); se recibio {self.A.shape}")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q debe ser (n,n)={n,n}; se recibio {self.Q.shape}")
        if self.H.shape[1] != n:
            raise ValueError(f"H debe tener {n} columnas; se recibio {self.H.shape}")
        if self.P.shape != (n, n):
            raise ValueError(f"P0 debe ser (n,n)={n,n}; se recibio {self.P.shape}")
        if self.x.shape != (n, 1):
            raise ValueError(f"x0 debe ser (n,1)={n,1}; se recibio {self.x.shape}")
        p = self.H.shape[0]
        if self.R.shape != (p, p):
            raise ValueError(f"R debe ser (p,p)={p,p}; se recibio {self.R.shape}")

    def prediccion(self, u=None) -> np.ndarray:
        """Etapa de prediccion: propaga estado y covarianza mediante el modelo fisico."""
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        u = np.asarray(u, dtype=float).reshape(-1, 1)

        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def correccion(self, y) -> np.ndarray:
        """Etapa de correccion: fusiona la prediccion con la medicion ruidosa."""
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        innovacion = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovacion

        # Forma de Joseph: numericamente estable, preserva simetria y P > 0.
        IKH = self.I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        self.historial_K.append(K.copy())
        self.historial_P.append(self.P.copy())
        self.historial_innovacion.append(float(innovacion.ravel()[0]))
        return self.x

    def paso(self, y, u=None) -> np.ndarray:
        """Ejecuta un ciclo completo predictor-corrector y devuelve x_{k|k}."""
        self.prediccion(u)
        return self.correccion(y)
