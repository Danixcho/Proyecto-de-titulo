"""
sensibilidad.py
===============
Analisis de sensibilidad y robustez parametrica del modulo analitico.

Este analisis operacionaliza el criterio de calidad "Sensibilidad y Robustez"
declarado en el diseno metodologico del proyecto, que exige evaluar el
comportamiento del estimador ante desajustes intencionales en las razones de
covarianza Q/R para verificar los margenes de estabilidad parametrica.

Se ejecutan dos experimentos complementarios:

1. BARRIDO DE DEGRADACION (R real creciente)
   Se incrementa la varianza real del ruido de medicion desde condiciones
   benignas hasta degradacion severa, manteniendo el filtro correctamente
   sintonizado. Responde a la pregunta: como escala el aporte del modulo con
   la hostilidad del entorno? Es la evidencia cuantitativa que sustenta la
   afirmacion de que el modulo es tanto mas necesario cuanto mas degradado
   esta el entorno minero.

2. MALLA DE DESAJUSTE (Q/R mal sintonizados)
   El filtro se sintoniza con covarianzas escaladas por factores
   (alpha, beta) respecto de las reales: Q_filtro = alpha * Q_real,
   R_filtro = beta * R_real. Representa el escenario realista en que el
   ingeniero de planta calibra el estimador con parametros imprecisos.
   Cuantifica cuanta degradacion tolera el modulo antes de perder utilidad.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from modelo_tolva import ParametrosTolva, generar_escenario, estimar
from analisis import rmse, mse, reduccion_porcentual


def barrido_degradacion(par_base: ParametrosTolva,
                        valores_R: tuple = (1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 400.0),
                        M: int = 50) -> list[dict]:
    """Escala la varianza real del sensor y mide el aporte del modulo.

    El numero de replicas por punto (M) es identico al del experimento
    principal de Monte Carlo, de modo que el punto R = r_sensor del barrido
    reproduce exactamente la metrica consolidada del experimento central. Esto
    hace que ambas tablas del informe sean mutuamente reconciliables sin
    necesidad de nota aclaratoria, y se declara explicitamente en el registro
    de salida.
    """
    resultados = []

    for r in valores_R:
        par = replace(par_base, r_sensor=r, M=M)
        semillas = np.random.SeedSequence(par.semilla).spawn(M)

        rmse_scada, rmse_kf = [], []
        mse_scada, mse_kf = [], []

        for hija in semillas:
            rng = np.random.default_rng(hija)
            x_real, y_scada, u = generar_escenario(par, rng)
            x_est, _ = estimar(par, y_scada, u)

            V_real = x_real[:, 0]
            rmse_scada.append(rmse(V_real, y_scada))
            rmse_kf.append(rmse(V_real, x_est[:, 0]))
            mse_scada.append(mse(V_real, y_scada))
            mse_kf.append(mse(V_real, x_est[:, 0]))

        resultados.append({
            "R": r,
            "replicas": M,
            "sigma_sensor_m3": float(np.sqrt(r)),
            "mse_scada_m6": float(np.mean(mse_scada)),
            "mse_kf_m6": float(np.mean(mse_kf)),
            "rmse_scada_m3": float(np.mean(rmse_scada)),
            "rmse_kf_m3": float(np.mean(rmse_kf)),
            "reduccion_mse_pct": reduccion_porcentual(float(np.mean(mse_scada)),
                                                      float(np.mean(mse_kf))),
            "reduccion_rmse_pct": reduccion_porcentual(float(np.mean(rmse_scada)),
                                                       float(np.mean(rmse_kf))),
        })

    return resultados


def malla_desajuste(par_base: ParametrosTolva,
                    factores: tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
                    M: int = 50) -> dict:
    """Desajusta Q y R del filtro respecto de la fisica real y mide el impacto.

    Devuelve una malla de RMSE del volumen indexada por (alpha, beta), junto al
    valor de referencia con sintonizacion perfecta (alpha = beta = 1).

    El numero de replicas por celda (M) coincide con el del experimento
    principal, y queda declarado en el registro de salida para que la malla sea
    auditable y comparable con las metricas consolidadas.
    """
    par = replace(par_base, M=M)
    semillas = np.random.SeedSequence(par.semilla).spawn(M)

    # Se generan los escenarios una sola vez: todas las sintonizaciones se
    # evaluan sobre exactamente la misma realidad fisica, de modo que las
    # diferencias observadas sean atribuibles solo a la calibracion del filtro.
    escenarios = []
    for hija in semillas:
        rng = np.random.default_rng(hija)
        escenarios.append(generar_escenario(par, rng))

    malla = np.zeros((len(factores), len(factores)))

    for i, alpha in enumerate(factores):
        for j, beta in enumerate(factores):
            Q_f = par.Q * alpha
            R_f = par.R * beta

            errores = []
            for x_real, y_scada, u in escenarios:
                x_est, _ = estimar(par, y_scada, u, Q_filtro=Q_f, R_filtro=R_f)
                errores.append(rmse(x_real[:, 0], x_est[:, 0]))
            malla[i, j] = float(np.mean(errores))

    # Linea base: RMSE de la telemetria cruda sobre los mismos escenarios.
    rmse_crudo = float(np.mean([rmse(x_real[:, 0], y_scada)
                                for x_real, y_scada, _ in escenarios]))

    idx_uno = factores.index(1.0)
    return {
        "factores": list(factores),
        "replicas": M,
        "malla_rmse_m3": malla.tolist(),
        "rmse_sintonizacion_perfecta_m3": float(malla[idx_uno, idx_uno]),
        "rmse_scada_crudo_m3": rmse_crudo,
        "peor_caso_m3": float(malla.max()),
        "mejor_caso_m3": float(malla.min()),
        "combinaciones_peores_que_crudo": int(np.sum(malla >= rmse_crudo)),
        "combinaciones_totales": int(malla.size),
    }
