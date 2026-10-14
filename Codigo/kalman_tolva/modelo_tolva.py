"""
modelo_tolva.py
===============
Modelo fisico de referencia (Truth Model) de la tolva de recepcion de un
chancador primario, y generador de telemetria SCADA degradada.

Unidad de analisis
------------------
Subsistema ciberfisico tolva-chancador, representado por el vector de estado
dinamico bidimensional en tiempo discreto:

    x_k = [ V_k , Vdot_k ]^T

    V_k    [m^3]   volumen de mineral retenido en la camara de recepcion.
    Vdot_k [m^3/s] tasa neta instantanea de variacion de flujo volumetrico,
                   dominada por la extraccion continua del chancador.

Fundamento fisico
-----------------
El modelo expresa la conservacion de masa de la tolva discretizada por Euler:

    V_{k+1}    = V_k + dt * Vdot_k + dt * u_k
    Vdot_{k+1} = Vdot_k + w_k

donde u_k [m^3/s] es el caudal de descarga de camion, conocido y registrado por
el SCADA (evento operacional discreto), y Vdot_k es la componente NO sensada de
la dinamica: la tasa de extraccion del chancador, que varia con la dureza del
mineral y no cuenta con instrumentacion directa. Esta es precisamente la
variable latente que sustenta el problema de observabilidad parcial: el sensor
solo observa el volumen, nunca la tasa.

    H = [1, 0]   ->   y_k = V_k + v_k

Observabilidad
--------------
El par (A, H) es observable: la matriz de observabilidad
    O = [H ; H A] = [[1, 0], [1, dt]]
tiene rango 2 para dt != 0. Es decir, Vdot es reconstruible a partir de la
secuencia de mediciones de V pese a no ser medido directamente. Esta propiedad
es la que habilita matematicamente al estimador.

Calibracion de parametros
-------------------------
Los valores corresponden a una estacion de chancado primario giratorio de gran
mineria a cielo abierto:

  * Extraccion nominal 0.8 m^3/s. Equivale a ~5760 t/h con densidad aparente
    2.0 t/m^3, coherente con un chancador giratorio de gran capacidad.
  * Descarga de camion 4.5 m^3/s durante 45 s -> ~202 m^3 por evento, coherente
    con un CAEX de 300-400 t a 2.0 t/m^3.
  * Cuatro ciclos de camion en el horizonte de 1000 s, cadencia consistente con
    mantener el nivel de la tolva en regimen.

NOTA DE RECALIBRACION (Q22)
---------------------------
Las entregas de Fase 3 declaraban Q = diag(0.01, 0.05). Se verifico
numericamente que una covarianza de proceso de 0.05 por paso sobre la tasa
produce, en un horizonte de N=1000 pasos, una caminata aleatoria con desviacion
acumulada sqrt(N * 0.05) = 7.07 m^3/s, es decir casi nueve veces la tasa de
extraccion nominal de 0.8 m^3/s. Ese regimen es fisicamente inadmisible: la
tolva se llenaria o vaciaria a velocidades imposibles para el activo.

Se recalibra Q22 = 1.0e-4, que produce una dispersion acumulada de 0.32 m^3/s
en torno a la tasa nominal (aproximadamente +-40% de variabilidad por dureza
variable del mineral), coherente con la literatura de operacion de chancado.
Q11 = 0.01 se mantiene sin cambios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ParametrosTolva:
    """Parametrizacion completa y auditable del experimento."""

    # --- Discretizacion temporal ---
    dt: float = 1.0            # [s]  paso de tiempo discreto
    N: int = 1000              # [-]  pasos por corrida
    M: int = 50                # [-]  corridas independientes de Monte Carlo
    semilla: int = 42          # [-]  semilla determinista (reproducibilidad)

    # --- Covarianzas de ruido ---
    q_volumen: float = 0.01     # [m^6]      ruido de proceso sobre V
    q_tasa: float = 1.0e-4      # [m^6/s^2]  ruido de proceso sobre Vdot (recalibrado)
    r_sensor: float = 5.0       # [m^6]      ruido de medicion del sensor de nivel

    # --- Condiciones iniciales ---
    V0: float = 15.0            # [m^3]    tolva casi vacia al inicio de turno
    Vdot0: float = -0.8         # [m^3/s]  tasa nominal de extraccion del chancador

    V0_filtro: float = 15.0     # [m^3]    estimacion inicial del filtro
    Vdot0_filtro: float = 0.0   # [m^3/s]  el filtro NO conoce la tasa: debe inferirla
    p0_volumen: float = 10.0    # [m^6]
    p0_tasa: float = 1.0        # [m^6/s^2]

    # --- Regimen operacional ---
    caudal_camion: float = 4.5              # [m^3/s] durante la descarga
    ventanas_camion: tuple = ((10, 55), (255, 300), (505, 550), (755, 800))

    # ------------------------------------------------------------------ #
    # Matrices del sistema en espacio de estados
    # ------------------------------------------------------------------ #
    @property
    def A(self) -> np.ndarray:
        return np.array([[1.0, self.dt],
                         [0.0, 1.0]])

    @property
    def B(self) -> np.ndarray:
        return np.array([[self.dt],
                         [0.0]])

    @property
    def H(self) -> np.ndarray:
        return np.array([[1.0, 0.0]])

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.q_volumen, self.q_tasa])

    @property
    def R(self) -> np.ndarray:
        return np.array([[self.r_sensor]])

    @property
    def x0_real(self) -> np.ndarray:
        return np.array([[self.V0], [self.Vdot0]])

    @property
    def x0_filtro(self) -> np.ndarray:
        return np.array([[self.V0_filtro], [self.Vdot0_filtro]])

    @property
    def P0(self) -> np.ndarray:
        return np.diag([self.p0_volumen, self.p0_tasa])

    def perfil_control(self) -> np.ndarray:
        """Secuencia u_k [m^3/s] de descarga de camiones (entrada conocida)."""
        u = np.zeros(self.N)
        for inicio, fin in self.ventanas_camion:
            u[inicio:fin] = self.caudal_camion
        return u

    def es_observable(self) -> bool:
        """Verifica el rango de la matriz de observabilidad del par (A, H)."""
        O = np.vstack([self.H, self.H @ self.A])
        return int(np.linalg.matrix_rank(O)) == self.A.shape[0]


def generar_escenario(par: ParametrosTolva, rng: np.random.Generator):
    """Genera una realizacion del Truth Model y su telemetria SCADA degradada.

    Devuelve
    --------
    x_real : (N, 2) trayectoria real del estado [V, Vdot].
    y_scada : (N,)  lectura degradada del sensor de nivel (volumen) [m^3].
    u : (N,)        entrada de control conocida [m^3/s].
    """
    A, B, H = par.A, par.B, par.H
    u = par.perfil_control()

    x_real = np.zeros((par.N, 2))
    y_scada = np.zeros(par.N)

    x = par.x0_real.copy()

    # Factores de Cholesky para muestrear los ruidos gaussianos multivariados.
    L_Q = np.linalg.cholesky(par.Q)
    sigma_R = float(np.sqrt(par.R[0, 0]))

    for k in range(par.N):
        # Perturbacion de proceso w_k ~ N(0, Q)
        w = L_Q @ rng.standard_normal((2, 1))
        x = A @ x + B @ np.array([[u[k]]]) + w
        x_real[k] = x.ravel()

        # Degradacion telemetrica v_k ~ N(0, R): vibracion severa y polvo abrasivo
        v = rng.normal(0.0, sigma_R)
        y_scada[k] = float((H @ x).ravel()[0]) + v

    return x_real, y_scada, u


def estimar(par: ParametrosTolva, y_scada: np.ndarray, u: np.ndarray,
            Q_filtro: np.ndarray | None = None,
            R_filtro: np.ndarray | None = None):
    """Ejecuta el modulo Filtro de Kalman sobre la telemetria degradada.

    Parametros opcionales
    ---------------------
    Q_filtro, R_filtro : permiten sintonizar el filtro con covarianzas
        distintas a las que generaron el proceso real. Se usan en el analisis
        de sensibilidad para cuantificar la degradacion del desempeno ante
        desajustes intencionales de la razon Q/R, es decir cuando el ingeniero
        de planta calibra mal el estimador respecto de la fisica real.

    Devuelve
    --------
    x_est : (N, 2) trayectoria estimada [V_hat, Vdot_hat].
    filtro : instancia con el historial de ganancias y covarianzas.
    """
    from nucleo_kalman import FiltroKalman

    Q_f = par.Q if Q_filtro is None else Q_filtro
    R_f = par.R if R_filtro is None else R_filtro

    filtro = FiltroKalman(par.A, par.B, par.H, Q_f, R_f,
                          par.x0_filtro, par.P0)

    x_est = np.zeros((par.N, 2))
    for k in range(par.N):
        x_k = filtro.paso(y=y_scada[k], u=np.array([[u[k]]]))
        x_est[k] = x_k.ravel()

    return x_est, filtro


def tasa_por_diferencias_finitas(y_scada: np.ndarray, dt: float) -> np.ndarray:
    """Linea base ingenua para la tasa: derivada numerica de la senal cruda.

    Es el unico procedimiento disponible para un SCADA convencional que quiera
    estimar la tasa de llenado sin un estimador de estados. Amplifica el ruido
    por un factor sqrt(2 R)/dt, y sirve como termino de comparacion honesto
    para dimensionar el aporte del modulo analitico sobre la variable latente.
    """
    tasa = np.zeros_like(y_scada)
    tasa[1:] = (y_scada[1:] - y_scada[:-1]) / dt
    tasa[0] = tasa[1]
    return tasa
