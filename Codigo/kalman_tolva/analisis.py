"""
analisis.py
===========
Estrategia de analisis de datos: metricas de error, simulacion de Monte Carlo e
inferencia estadistica.

Coherencia dimensional
----------------------
Se distingue de forma explicita entre metricas cuadraticas y metricas lineales,
observacion levantada en la retroalimentacion de la Formativa 4:

    MSE  [m^6]   error cuadratico medio del volumen (V en m^3 -> error^2 en m^6)
    RMSE [m^3]   raiz del error cuadratico medio, en las unidades del estado
    IC   [m^3]   intervalo de confianza sobre la diferencia de RMSE

Toda inferencia estadistica (prueba t, intervalo de confianza, tamano del
efecto) se realiza sobre el RMSE, de modo que los resultados inferenciales
quedan expresados en metros cubicos y son dimensionalmente homogeneos con la
variable fisica estimada. El MSE se reporta como metrica descriptiva.

Diseno de la prueba de hipotesis
--------------------------------
Cada corrida de Monte Carlo es una realizacion independiente del proceso
estocastico y aporta un par (RMSE_scada, RMSE_kf). Con M = 50 pares se aplica
una prueba t de Student para muestras pareadas sobre las diferencias:

    H0: mu_d = 0   (el modulo analitico no reduce el error)
    H1: mu_d > 0   (el modulo analitico reduce el error)

    con d_m = RMSE_scada,m - RMSE_kf,m ,  alpha = 0.05

El supuesto de normalidad de las diferencias se verifica mediante la prueba de
Shapiro-Wilk. Se reporta ademas la prueba no parametrica de Wilcoxon de rangos
con signo como contraste de robustez, de modo que la conclusion no dependa del
cumplimiento del supuesto parametrico.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

from modelo_tolva import (
    ParametrosTolva,
    generar_escenario,
    estimar,
    tasa_por_diferencias_finitas,
)


# --------------------------------------------------------------------------- #
# Metricas de error
# --------------------------------------------------------------------------- #
def mse(referencia: np.ndarray, estimacion: np.ndarray) -> float:
    """Error cuadratico medio. Unidades: cuadrado de las de la variable."""
    return float(np.mean((referencia - estimacion) ** 2))


def rmse(referencia: np.ndarray, estimacion: np.ndarray) -> float:
    """Raiz del error cuadratico medio. Unidades: las de la variable."""
    return float(np.sqrt(mse(referencia, estimacion)))


def reduccion_porcentual(base: float, mejorado: float) -> float:
    """Reduccion relativa del error respecto de la linea base, en porcentaje."""
    return float((base - mejorado) / base * 100.0)


@dataclass
class MetricasCorrida:
    """Metricas de una corrida individual."""

    # Volumen V [m^3] - variable observada con ruido
    mse_scada_V: float      # [m^6]
    mse_kf_V: float         # [m^6]
    rmse_scada_V: float     # [m^3]
    rmse_kf_V: float        # [m^3]

    # Tasa Vdot [m^3/s] - variable latente, no sensada
    rmse_dif_finitas: float  # [m^3/s]
    rmse_kf_Vdot: float      # [m^3/s]


def metricas_de_corrida(par: ParametrosTolva, rng: np.random.Generator):
    """Ejecuta una corrida completa y devuelve sus metricas y series."""
    x_real, y_scada, u = generar_escenario(par, rng)
    x_est, filtro = estimar(par, y_scada, u)

    V_real, Vdot_real = x_real[:, 0], x_real[:, 1]
    V_est, Vdot_est = x_est[:, 0], x_est[:, 1]
    Vdot_dif = tasa_por_diferencias_finitas(y_scada, par.dt)

    m = MetricasCorrida(
        mse_scada_V=mse(V_real, y_scada),
        mse_kf_V=mse(V_real, V_est),
        rmse_scada_V=rmse(V_real, y_scada),
        rmse_kf_V=rmse(V_real, V_est),
        rmse_dif_finitas=rmse(Vdot_real, Vdot_dif),
        rmse_kf_Vdot=rmse(Vdot_real, Vdot_est),
    )

    series = {
        "x_real": x_real,
        "y_scada": y_scada,
        "x_est": x_est,
        "u": u,
        "Vdot_dif": Vdot_dif,
        "filtro": filtro,
    }
    return m, series


# --------------------------------------------------------------------------- #
# Simulacion de Monte Carlo
# --------------------------------------------------------------------------- #
def monte_carlo(par: ParametrosTolva, verbose: bool = True):
    """Ejecuta M corridas independientes con semilla maestra determinista.

    Cada corrida usa un flujo (stream) independiente derivado de la semilla
    maestra mediante SeedSequence.spawn(), lo que garantiza independencia
    estadistica entre corridas y reproducibilidad exacta del conjunto.
    """
    semilla_maestra = np.random.SeedSequence(par.semilla)
    hijas = semilla_maestra.spawn(par.M)

    filas: list[MetricasCorrida] = []
    primera_serie = None

    for i, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        m, series = metricas_de_corrida(par, rng)
        filas.append(m)
        if i == 0:
            primera_serie = series

    if verbose:
        print(f"  Monte Carlo completado: M = {par.M} corridas de N = {par.N} pasos.")

    return filas, primera_serie


# --------------------------------------------------------------------------- #
# Inferencia estadistica
# --------------------------------------------------------------------------- #
def inferencia_pareada(base: np.ndarray, mejorado: np.ndarray, alpha: float = 0.05) -> dict:
    """Prueba t pareada unilateral, IC al 95%, normalidad y contraste robusto.

    Parametros
    ----------
    base : RMSE de la linea base por corrida.
    mejorado : RMSE del modulo analitico por corrida.
    """
    base = np.asarray(base, dtype=float)
    mejorado = np.asarray(mejorado, dtype=float)
    d = base - mejorado
    M = d.size

    media_d = float(np.mean(d))
    desv_d = float(np.std(d, ddof=1))
    error_estandar = desv_d / np.sqrt(M)

    # Prueba t pareada unilateral (H1: la reduccion es positiva)
    t_stat, p_bilateral = stats.ttest_rel(base, mejorado)
    p_unilateral = p_bilateral / 2.0 if t_stat > 0 else 1.0 - p_bilateral / 2.0

    # Intervalo de confianza bilateral al 95% sobre la diferencia media
    t_critico = stats.t.ppf(1.0 - alpha / 2.0, df=M - 1)
    ic_inf = media_d - t_critico * error_estandar
    ic_sup = media_d + t_critico * error_estandar

    # Verificacion del supuesto de normalidad de las diferencias
    sw_stat, sw_p = stats.shapiro(d)

    # Contraste no parametrico de robustez
    w_stat, w_p = stats.wilcoxon(base, mejorado, alternative="greater")

    # Tamano del efecto (d de Cohen para muestras pareadas)
    cohen_d = media_d / desv_d if desv_d > 0 else float("inf")

    return {
        "M": int(M),
        "media_diferencia": media_d,
        "desv_diferencia": desv_d,
        "error_estandar": float(error_estandar),
        "t_estadistico": float(t_stat),
        "gl": int(M - 1),
        "p_unilateral": float(p_unilateral),
        "ic95_inferior": float(ic_inf),
        "ic95_superior": float(ic_sup),
        "shapiro_W": float(sw_stat),
        "shapiro_p": float(sw_p),
        "normalidad_no_rechazada": bool(sw_p > alpha),
        "wilcoxon_W": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohen_d": float(cohen_d),
        "alpha": alpha,
        "rechaza_H0": bool(p_unilateral < alpha),
    }


def consolidar(filas: list[MetricasCorrida]) -> dict:
    """Agrega las metricas de las M corridas y ejecuta la inferencia."""
    arr = {campo: np.array([getattr(f, campo) for f in filas])
           for campo in asdict(filas[0]).keys()}

    resumen = {
        # --- Volumen: metricas descriptivas ---
        "mse_scada_V_medio": float(np.mean(arr["mse_scada_V"])),
        "mse_kf_V_medio": float(np.mean(arr["mse_kf_V"])),
        "mse_scada_V_desv": float(np.std(arr["mse_scada_V"], ddof=1)),
        "mse_kf_V_desv": float(np.std(arr["mse_kf_V"], ddof=1)),
        "rmse_scada_V_medio": float(np.mean(arr["rmse_scada_V"])),
        "rmse_kf_V_medio": float(np.mean(arr["rmse_kf_V"])),
        "rmse_scada_V_desv": float(np.std(arr["rmse_scada_V"], ddof=1)),
        "rmse_kf_V_desv": float(np.std(arr["rmse_kf_V"], ddof=1)),
        "reduccion_mse_V_pct": reduccion_porcentual(
            float(np.mean(arr["mse_scada_V"])), float(np.mean(arr["mse_kf_V"]))),
        "reduccion_rmse_V_pct": reduccion_porcentual(
            float(np.mean(arr["rmse_scada_V"])), float(np.mean(arr["rmse_kf_V"]))),

        # --- Tasa latente: metricas descriptivas ---
        "rmse_dif_finitas_medio": float(np.mean(arr["rmse_dif_finitas"])),
        "rmse_kf_Vdot_medio": float(np.mean(arr["rmse_kf_Vdot"])),
        "reduccion_rmse_Vdot_pct": reduccion_porcentual(
            float(np.mean(arr["rmse_dif_finitas"])), float(np.mean(arr["rmse_kf_Vdot"]))),
    }

    resumen["inferencia_V"] = inferencia_pareada(arr["rmse_scada_V"], arr["rmse_kf_V"])
    resumen["inferencia_Vdot"] = inferencia_pareada(arr["rmse_dif_finitas"], arr["rmse_kf_Vdot"])

    return resumen, arr
