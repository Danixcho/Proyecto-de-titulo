"""
simulador.py  -  Modulo 1: proceso simulado (la verdad)
========================================================
Genera la trayectoria verdadera del volumen de la tolva y de la extraccion
efectiva del chancador. Es deliberadamente DISTINTO del modelo que supone el
filtro de Kalman (Seccion 4.3 de la tesis):

  - Capacidad de extraccion e_cap[k] = max(e_nom[k] + d[k], 0), donde d es una
    desviacion de Ornstein-Uhlenbeck (dureza variable) iniciada en su
    distribucion estacionaria. El filtro la modela como caminata aleatoria.
  - Llegadas de camiones con intervalos gamma (media T_a, coef. de variacion
    c_v); la primera llegada es uniforme en [0, T_a). Los camiones forman una
    cola. Un camion comienza a descargar solo si no hay otro descargando y si
    su carga completa cabe en la tolva
    (V + carga <= V_max); si no cabe, espera. Asi el volumen nunca excede la
    capacidad.
  - La descarga dura T_d segundos a caudal constante carga / T_d.
  - Saturacion: la extraccion efectiva es la capacidad limitada por el
    material disponible, e[k] = min(e_cap[k], V[k]/dt + u[k]). Cuando la tolva
    se vacia, la extraccion cae bruscamente. El filtro no conoce esta no
    linealidad.
  - Balance de masa con limites fisicos:
        V[k+1] = min(max(V[k] + dt (u[k] - e[k]), 0), V_max).

V[k] es el volumen al comienzo del paso k; u[k] y e[k] actuan entre k y k+1.

Toda la aleatoriedad se consume al comienzo (llegadas y desviacion OU), de modo
que el mismo generador produce las mismas llegadas y la misma capacidad con o
sin saturacion (ablacion E2c).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parametros import ParametrosEstimador, ParametrosProceso


@dataclass
class TrayectoriaProceso:
    t: np.ndarray            # [s] instantes
    V: np.ndarray            # [m3] volumen verdadero al comienzo de cada paso
    e: np.ndarray            # [m3/s] extraccion efectiva (la verdad de la tasa)
    e_capacidad: np.ndarray  # [m3/s] capacidad de extraccion (sin saturar)
    u: np.ndarray            # [m3/s] descarga conocida de camiones
    descargando: np.ndarray  # hay un camion descargando
    n_camiones: int          # camiones que descargaron en la corrida
    espera_total: float      # [s] tiempo acumulado de camiones esperando
    pasos_vacia: int         # pasos con la extraccion saturada


def llegadas_camiones(pp: ParametrosProceso, rng: np.random.Generator) -> np.ndarray:
    """Instantes de llegada en [0, N dt). Intervalos gamma con media
    intervalo_medio y coeficiente de variacion cv_llegadas (forma 1/cv^2)."""
    forma = 1.0 / pp.cv_llegadas ** 2
    escala = pp.intervalo_medio / forma
    horizonte = pp.N * pp.dt
    llegadas = [rng.uniform(0.0, pp.intervalo_medio)]
    while True:
        siguiente = llegadas[-1] + rng.gamma(forma, escala)
        if siguiente >= horizonte:
            break
        llegadas.append(siguiente)
    return np.array(llegadas)


def capacidad_extraccion(pp: ParametrosProceso, rng: np.random.Generator) -> np.ndarray:
    """e_cap[k] = max(e_nom[k] + d[k], 0) con d de Ornstein-Uhlenbeck.

    d[0] ~ N(0, sigma_e^2) (arranque estacionario) y
    d[k+1] = a d[k] + sigma_e sqrt(1 - a^2) xi[k+1], a = exp(-dt/tau_e).
    La capacidad nominal e_nom[k] es constante, salvo en E4, donde se
    multiplica por factor_escalon desde t_escalon.
    """
    a = np.exp(-pp.dt / pp.tau_e)
    xi = rng.standard_normal(pp.N)
    d = np.empty(pp.N)
    d[0] = pp.sigma_e * xi[0]
    c = pp.sigma_e * np.sqrt(1.0 - a ** 2)
    for k in range(1, pp.N):
        d[k] = a * d[k - 1] + c * xi[k]
    t = np.arange(pp.N) * pp.dt
    e_nom_t = np.full(pp.N, pp.e_nom)
    if pp.t_escalon is not None:
        e_nom_t[t >= pp.t_escalon] *= pp.factor_escalon
    return np.maximum(e_nom_t + d, 0.0)


def simular_proceso(pp: ParametrosProceso, rng: np.random.Generator) -> TrayectoriaProceso:
    """Una corrida del proceso simulado."""
    # e_cap[k] = e_nom[k] + d[k], donde d es una desviacion de Ornstein-Uhlenbeck
    # y e_nom cae en t_escalon solo en el Experimento E4.
    e_cap = capacidad_extraccion(pp, rng)
    llegadas = llegadas_camiones(pp, rng)

    dt, N = pp.dt, pp.N
    t = np.arange(N) * dt
    V = np.empty(N)
    e_ef = np.empty(N)
    u = np.zeros(N)
    descargando = np.zeros(N, dtype=bool)

    caudal = pp.carga_camion / pp.duracion_descarga
    v = pp.V0
    resto = 0.0               # carga que resta por descargar del camion en curso
    j = 0                     # proxima llegada por procesar
    cola: list[float] = []    # instantes de llegada de los camiones en espera
    espera_total = 0.0
    n_camiones = 0
    pasos_vacia = 0

    for k in range(N):
        # Camiones que llegaron hasta el instante t[k] se suman a la cola.
        while j < len(llegadas) and llegadas[j] <= t[k]:
            cola.append(llegadas[j])
            j += 1
        # Un camion comienza a descargar si no hay otro descargando y su carga cabe.
        if resto <= 0.0 and cola and v + pp.carga_camion <= pp.V_max:
            espera_total += t[k] - cola.pop(0)
            resto = pp.carga_camion
            n_camiones += 1
        if resto > 0.0:
            u[k] = min(caudal, resto / dt)
            resto -= u[k] * dt
            if resto < 1e-9:
                resto = 0.0
            descargando[k] = True

        if pp.saturacion:
            disponible = v / dt + u[k]
            e_ef[k] = min(e_cap[k], disponible)
            if e_ef[k] < e_cap[k]:
                pasos_vacia += 1
        else:
            e_ef[k] = e_cap[k]           # ablacion: sin saturacion (contrafactual)

        V[k] = v
        v = v + dt * (u[k] - e_ef[k])
        # El redondeo numerico no debe producir volumenes fuera del rango fisico.
        v = min(max(v, 0.0), pp.V_max) if pp.saturacion else min(v, pp.V_max)

    return TrayectoriaProceso(t=t, V=V, e=e_ef, e_capacidad=e_cap, u=u,
                              descargando=descargando, n_camiones=n_camiones,
                              espera_total=espera_total, pasos_vacia=pasos_vacia)


def descarga_fija(N: int, dt: float, caudal: float, duracion: float, periodo: float,
                  inicio: float) -> np.ndarray:
    """Perfil determinista de descarga del Experimento EV: `caudal` durante
    `duracion` segundos cada `periodo` segundos a partir de `inicio`."""
    t = np.arange(N) * dt
    fase = t - inicio
    return np.where((fase >= 0) & (np.mod(fase, periodo) < duracion), caudal, 0.0)


def simular_modelo_coincidente_lote(pe: ParametrosEstimador, u: np.ndarray, rngs) -> np.ndarray:
    """Version vectorizada de simular_modelo_coincidente para varias corridas.
    Cada corrida consume su generador en el mismo orden que la version por
    corrida (estado inicial y luego ruido de proceso), de modo que ambas dan el
    mismo resultado. Devuelve X con forma (M, N, 2)."""
    N, M = len(u), len(rngs)
    A, B, Q = pe.A, pe.B, pe.Q
    x0 = np.array([g.multivariate_normal(pe.x0, pe.P0) for g in rngs])
    w = np.array([g.standard_normal((N, 2)) for g in rngs]) * np.sqrt(np.diag(Q))[None, None, :]
    X = np.empty((M, N, 2))
    X[:, 0] = x0
    for k in range(1, N):
        X[:, k] = X[:, k - 1] @ A.T + u[k - 1] * B[None, :] + w[:, k - 1]
    return X


def simular_modelo_coincidente(pe: ParametrosEstimador, u: np.ndarray,
                               rng: np.random.Generator) -> TrayectoriaProceso:
    """Proceso generado con EXACTAMENTE el modelo del filtro (Experimento EV):
    lineal, gaussiano, con las mismas Q y estado inicial muestreado de
    N(x0, P0). No respeta limites fisicos: solo sirve para verificar la
    implementacion, nunca para concluir sobre la tolva."""
    N = len(u)
    A, B, Q = pe.A, pe.B, pe.Q
    x = rng.multivariate_normal(pe.x0, pe.P0)
    w = rng.standard_normal((N, 2)) * np.sqrt(np.diag(Q))[None, :]
    X = np.empty((N, 2))
    X[0] = x
    for k in range(1, N):
        X[k] = A @ X[k - 1] + B * u[k - 1] + w[k - 1]
    t = np.arange(N) * pe.dt
    return TrayectoriaProceso(t=t, V=X[:, 0], e=X[:, 1], e_capacidad=X[:, 1], u=u,
                              descargando=u > 0, n_camiones=0, espera_total=0.0,
                              pasos_vacia=0)
