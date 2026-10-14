"""
verificacion.py  -  Experimento EV: verificacion de la implementacion
======================================================================
El proceso se genera con EXACTAMENTE el modelo del filtro (lineal, gaussiano,
mismas Q y R, estado inicial muestreado de N(x0, P0)) y un perfil fijo de
descarga. En esa condicion el filtro es optimo y debe cumplir propiedades
teoricas conocidas. Si alguna prueba falla, main.py se detiene sin ejecutar
los experimentos.

  A  P[k] simetrica y definida positiva
  B  la ganancia converge a la solucion de la ecuacion de Riccati, calculada
     de forma independiente con SciPy
  C  error insesgado (prueba t de media cero sobre el error medio de cada
     corrida, por componente)
  D  NEES promedio entre corridas dentro de la banda chi^2 del 95 % en al
     menos el 90 % de los pasos
  E  NIS promedio entre corridas dentro de la banda chi^2 del 95 % en al
     menos el 90 % de los pasos
  F  innovacion blanca (autocorrelacion de rezago 1 dentro de +-1,96/sqrt(N'))
  G  escalar Q, R y P0 por un mismo factor no cambia la ganancia
  H  la interfaz en linea (una medicion por llamada) coincide con la
     implementacion por lotes

Los criterios D y E son conservadores: el NEES promedio esta autocorrelacionado
en el tiempo (el error de la tasa tiene una memoria de alrededor de un minuto),
de modo que la fraccion de pasos en banda varia entre lotes y puede quedar bajo
el 90 % aun con un filtro exactamente consistente: un fallo de D puede detener
el programa aunque el codigo este correcto. analisis_criterio_bandas() estima
la frecuencia de esos falsos rechazos con lotes independientes. Que las pruebas
se cumplan aporta evidencia, pero no demuestra que no haya errores.

Estas trayectorias no respetan los limites fisicos de la tolva y no se usan
para ninguna conclusion sobre ella.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import linalg, stats

from estimador import EstimadorEnLinea, FiltroKalman
from evaluador import autocorrelacion, nees, nis
from parametros import ParametrosEstimador, ParametrosSensor, ParametrosVerificacion
from sensor import medir_desde_eps
from simulador import descarga_fija, simular_modelo_coincidente_lote


def ganancia_riccati(pe: ParametrosEstimador) -> np.ndarray:
    """Ganancia estacionaria a partir de la ecuacion algebraica discreta de
    Riccati (SciPy), independiente de la recursion del filtro."""
    A, H, Q, R = pe.A, pe.H, pe.Q, np.array([[pe.R]])
    Pp = linalg.solve_discrete_are(A.T, H.T, Q, R)
    return (Pp @ H.T / (H @ Pp @ H.T + R)).ravel()


def _lote(pe, sigma_v, pv, rng_para, fase=0):
    """Trayectorias del modelo coincidente y sus mediciones."""
    dt, N, M = pe.dt, pv.N, pv.M
    u = descarga_fija(N, dt, pv.caudal_descarga, pv.duracion_descarga, pv.periodo_descarga,
                      pv.inicio_descarga)
    X = simular_modelo_coincidente_lote(pe, u, [rng_para(m, 0, fase) for m in range(M)])
    eps = np.array([rng_para(m, 1, fase).standard_normal(N) for m in range(M)])
    Y = medir_desde_eps(X[:, :, 0], np.zeros((M, N), bool), eps, ParametrosSensor(sigma_v=sigma_v), dt)
    return u, X, Y


def _bandas(M):
    return (stats.chi2.ppf([0.025, 0.975], 2 * M) / M, stats.chi2.ppf([0.025, 0.975], M) / M)


def ejecutar(pe: ParametrosEstimador, sigma_v: float, rng_para, pv: ParametrosVerificacion | None = None):
    """Ejecuta las ocho pruebas. `rng_para(corrida, flujo, fase)` entrega el
    generador de cada corrida. Devuelve (resultados, serie_para_figura)."""
    pv = pv or ParametrosVerificacion()
    N, M, q = pv.N, pv.M, pv.quemado
    u, X, Y = _lote(pe, sigma_v, pv, rng_para)
    U = np.tile(u, (M, 1))
    kf = FiltroKalman(pe).filtrar(Y, U)
    err = X - kf.x
    res = {}

    # A: simetria y definicion positiva
    asim = float(np.max(np.abs(kf.P - np.transpose(kf.P, (0, 2, 1)))))
    lam_min = float(np.min(np.linalg.eigvalsh(kf.P)))
    res["A_simetria_P"] = {"asimetria_max": asim, "autovalor_min": lam_min,
                           "ok": bool(asim < 1e-10 and lam_min > 0)}

    # B: ganancia final frente a la solucion de Riccati
    K_inf = ganancia_riccati(pe)
    dif = float(np.max(np.abs(kf.K[-1] - K_inf)))
    res["B_riccati"] = {"K_filtro": kf.K[-1].tolist(), "K_riccati": K_inf.tolist(),
                        "diferencia": dif, "ok": bool(dif < 1e-8)}

    # C: insesgamiento (media temporal del error de cada corrida; prueba t)
    pV = float(stats.ttest_1samp(err[:, q:, 0].mean(axis=1), 0.0).pvalue)
    pe_ = float(stats.ttest_1samp(err[:, q:, 1].mean(axis=1), 0.0).pvalue)
    res["C_insesgamiento"] = {"p_volumen": pV, "p_tasa": pe_, "ok": bool(pV > 0.01 and pe_ > 0.01)}

    # D y E: consistencia del NEES y del NIS
    bD, bE = _bandas(M)
    NEES = nees(err, kf.P)[:, q:]
    NIS = nis(kf.nu, kf.S)[:, q:]
    for clave, Z, esperado, banda in (("D_nees", NEES, 2.0, bD), ("E_nis", NIS, 1.0, bE)):
        zk = Z.mean(axis=0)
        en = float(np.mean((zk >= banda[0]) & (zk <= banda[1])))
        res[clave] = {"media_global": float(Z.mean()), "esperado": esperado, "banda": banda.tolist(),
                      "fraccion_pasos_en_banda": en, "ok": bool(en >= 0.90)}

    # F: blancura de la innovacion normalizada
    inn = kf.nu[:, q:] / np.sqrt(kf.S[None, q:])
    r1 = autocorrelacion(inn, 1)
    lim = 1.96 / np.sqrt(inn.shape[1])
    enF = float(np.mean(np.abs(r1) < lim))
    res["F_blancura"] = {"r1_medio": float(r1.mean()), "limite": float(lim),
                         "fraccion_corridas_en_banda": enF, "ok": bool(enF >= 0.90)}

    # G: escala comun de Q, R y P0
    difs = []
    for c in (0.1, 10.0):
        pc = replace(pe, q_V=c * pe.q_V, q_e=c * pe.q_e, R=c * pe.R, P0_V=c * pe.P0_V, P0_e=c * pe.P0_e)
        Kc = FiltroKalman(pc).covarianzas(N)[0]
        difs.append(float(np.max(np.abs(Kc[-1] - kf.K[-1]))))
    res["G_escala_comun"] = {"factores": [0.1, 10.0], "diferencia_max_K": max(difs),
                             "ok": bool(max(difs) < 1e-10)}

    # H: interfaz en linea frente a la implementacion por lotes
    dmax = 0.0
    for m in range(pv.corridas_en_linea):
        est = EstimadorEnLinea(pe)
        for k in range(N):
            r = est.actualizar(Y[m, k], u[k - 1] if k > 0 else None)
            dmax = max(dmax, abs(r["V"] - kf.x[m, k, 0]), abs(r["e"] - kf.x[m, k, 1]))
    res["H_en_linea"] = {"diferencia_max": float(dmax), "corridas": pv.corridas_en_linea,
                         "ok": bool(dmax < 1e-9)}

    res["_parametros"] = {"M": M, "N": N, "quemado": q}
    serie = {"k": np.arange(N)[q:].tolist(), "nees": NEES.mean(axis=0).tolist(),
             "nis": NIS.mean(axis=0).tolist(), "banda_nees": bD.tolist(), "banda_nis": bE.tolist()}
    return res, serie


def analisis_criterio_bandas(pe: ParametrosEstimador, sigma_v: float, rng_para,
                             pv: ParametrosVerificacion | None = None, lotes: int = 100) -> dict:
    """Con un filtro exactamente consistente, que fraccion de lotes EV
    independientes (semillas distintas de las del lote de verificacion) no
    cumpliria los criterios D y E. Como comparacion se informa tambien una
    prueba t sobre el NEES y el NIS medios de cada corrida, que son
    independientes entre corridas."""
    pv = pv or ParametrosVerificacion()
    M, q = pv.M, pv.quemado
    bD, bE = _bandas(M)
    filas = []
    kfc = FiltroKalman(pe)
    for lote in range(lotes):
        u, X, Y = _lote(pe, sigma_v, pv, rng_para, fase=10 + lote)
        kf = kfc.filtrar(Y, np.tile(u, (M, 1)))
        NE = nees(X - kf.x, kf.P)[:, q:]
        NI = nis(kf.nu, kf.S)[:, q:]
        fD = np.mean((NE.mean(0) >= bD[0]) & (NE.mean(0) <= bD[1]))
        fE = np.mean((NI.mean(0) >= bE[0]) & (NI.mean(0) <= bE[1]))
        pD = stats.ttest_1samp(NE.mean(1), 2.0).pvalue
        pE = stats.ttest_1samp(NI.mean(1), 1.0).pvalue
        filas.append((fD, fE, NE.mean(), NI.mean(), pD, pE))
    r = np.array(filas)
    return {"lotes": lotes,
            "nees": {"fraccion_en_banda_media": float(r[:, 0].mean()),
                     "fraccion_en_banda_min": float(r[:, 0].min()),
                     "lotes_bajo_90": float(np.mean(r[:, 0] < 0.90)),
                     "media_global_media": float(r[:, 2].mean()),
                     "lotes_prueba_t_p_menor_001": float(np.mean(r[:, 4] < 0.01))},
            "nis": {"fraccion_en_banda_media": float(r[:, 1].mean()),
                    "fraccion_en_banda_min": float(r[:, 1].min()),
                    "lotes_bajo_90": float(np.mean(r[:, 1] < 0.90)),
                    "media_global_media": float(r[:, 3].mean()),
                    "lotes_prueba_t_p_menor_001": float(np.mean(r[:, 5] < 0.01))}}
