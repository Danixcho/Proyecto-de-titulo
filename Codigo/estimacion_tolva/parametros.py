"""
parametros.py  -  Parametros del estudio
========================================
Todos los parametros son SUPUESTOS DE DISENO: se eligieron para que los ordenes
de magnitud sean coherentes con una estacion de chancado primario de gran
mineria, pero ninguno proviene de mediciones de una planta (Capitulo 4 de la
tesis).

Los parametros se separan en cuatro grupos que no se mezclan:

  ParametrosProceso      la verdad simulada (capa 1)
  ParametrosSensor       la medicion (capa 2)
  ParametrosEstimador    lo que SUPONE el filtro (capa 3)
  ParametrosExperimento  el protocolo (corridas, quemado, semillas, grillas)

El filtro nunca lee ParametrosProceso ni ParametrosSensor durante la
estimacion: sus supuestos se derivan una sola vez, al construir
ParametrosEstimador, y desde ahi se tratan como un modelo aparte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ParametrosProceso:
    """Proceso simulado: la tolva con limites fisicos (la verdad)."""
    dt: float = 1.0                  # [s] periodo de muestreo
    N: int = 3600                    # pasos por corrida (una hora)
    V_max: float = 450.0             # [m3] capacidad viva de la tolva
    V0: float = 225.0                # [m3] volumen inicial
    e_nom: float = 0.8               # [m3/s] capacidad nominal de extraccion
    tau_e: float = 600.0             # [s] constante de tiempo de la dureza (OU)
    sigma_e: float = 0.08            # [m3/s] desviacion estacionaria de la capacidad
    carga_camion: float = 180.0      # [m3] carga por camion
    duracion_descarga: float = 45.0  # [s] duracion de una descarga
    intervalo_medio: float = 210.0   # [s] intervalo medio entre llegadas
    cv_llegadas: float = 0.4         # coeficiente de variacion de los intervalos (gamma)
    # --- Cambio brusco de la dureza (Experimento E4) ---
    t_escalon: float | None = None   # [s] instante en que cae la capacidad nominal
    factor_escalon: float = 1.0      # capacidad nominal despues del cambio / antes
    densidad_aparente: float = 2.0   # [t/m3] solo para conversiones informativas

    # --- Ablacion (Experimento E2c) ---
    # Con saturacion=False la extraccion efectiva es siempre la capacidad y el
    # volumen puede hacerse negativo. Es FISICAMENTE IMPOSIBLE: solo sirve como
    # contrafactual para aislar el efecto de la saturacion por vaciado.
    saturacion: bool = True


@dataclass(frozen=True)
class ParametrosSensor:
    """Sensor de nivel convertido a volumen: y = V + b + sigma * n."""
    sigma_v: float = 10.0            # [m3] desviacion del error de medicion
    rho: float = 0.0                 # autocorrelacion AR(1) del error (0 = blanco)
    factor_descarga: float = 1.0     # multiplicador de sigma_v mientras un camion descarga
    deriva_total: float = 0.0        # [m3] sesgo acumulado al final de la hora (deriva lineal)


@dataclass(frozen=True)
class ParametrosEstimador:
    """Supuestos del filtro de Kalman (modelo del estimador).

    Estado x = [V, e]: volumen y tasa de extraccion.
        x[k] = A x[k-1] + B u[k-1] + w[k-1],   w ~ N(0, Q),  Q = diag(q_V, q_e)
        y[k] = H x[k] + v[k],                  v ~ N(0, R)
    con A = [[1, -dt], [0, 1]], B = [dt, 0], H = [1, 0].
    """
    dt: float
    q_V: float
    q_e: float
    R: float
    V0_prior: float
    e0_prior: float
    P0_V: float
    P0_e: float

    @property
    def A(self) -> np.ndarray:
        return np.array([[1.0, -self.dt], [0.0, 1.0]])

    @property
    def B(self) -> np.ndarray:
        return np.array([self.dt, 0.0])

    @property
    def H(self) -> np.ndarray:
        return np.array([[1.0, 0.0]])

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.q_V, self.q_e])

    @property
    def x0(self) -> np.ndarray:
        return np.array([self.V0_prior, self.e0_prior])

    @property
    def P0(self) -> np.ndarray:
        return np.diag([self.P0_V, self.P0_e])


def estimador_nominal(pp: ParametrosProceso, ps: ParametrosSensor) -> ParametrosEstimador:
    """Supuestos nominales del filtro (Seccion 4.5 de la tesis).

    - q_V = 0: el balance de masa es exacto y la descarga es conocida.
    - q_e = sigma_e^2 (1 - a^2), a = exp(-dt/tau_e): varianza por paso de las
      innovaciones del proceso de Ornstein-Uhlenbeck, que el filtro modela como
      una caminata aleatoria.
    - R = sigma_v^2: la especificacion nominal del sensor.
    - x0 = [V0, e_nom]; P0 = diag(V_max^2/12, sigma_e^2): varianza de una
      uniforme sobre la capacidad para el volumen.
    """
    a = math.exp(-pp.dt / pp.tau_e)
    return ParametrosEstimador(
        dt=pp.dt,
        q_V=0.0,
        q_e=pp.sigma_e ** 2 * (1.0 - a ** 2),
        R=ps.sigma_v ** 2,
        V0_prior=pp.V0,
        e0_prior=pp.e_nom,
        P0_V=pp.V_max ** 2 / 12.0,
        P0_e=pp.sigma_e ** 2,
    )


@dataclass(frozen=True)
class ParametrosExperimento:
    """Protocolo experimental."""
    semilla_maestra: int = 2026
    M: int = 100                     # corridas de prueba por condicion
    M_calibracion: int = 30          # corridas de calibracion de las lineas base
    quemado: int = 300               # pasos iniciales excluidos de las metricas
    B_bootstrap: int = 2000          # remuestreos bootstrap
    ventanas: tuple = (2, 3, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600)
    factores_qe: tuple = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass(frozen=True)
class ParametrosVerificacion:
    """Experimento EV: proceso identico al modelo del filtro."""
    M: int = 200                     # corridas
    N: int = 1000                    # pasos por corrida
    quemado: int = 60                # pasos omitidos en las pruebas de banda y en la figura
    caudal_descarga: float = 4.0     # [m3/s] perfil fijo de descarga
    duracion_descarga: float = 45.0  # [s]
    periodo_descarga: float = 210.0  # [s]
    inicio_descarga: float = 30.0    # [s]
    corridas_en_linea: int = 5       # corridas para comparar la interfaz en linea
