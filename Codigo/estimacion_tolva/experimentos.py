"""
experimentos.py  -  Protocolo experimental
==========================================
Define y ejecuta los experimentos de la tesis. Cada experimento responde a una
parte de la pregunta de investigacion:

  EV  Verificacion      El proceso coincide EXACTAMENTE con el modelo del filtro.
                        No mide desempeno: comprueba que la implementacion cumple
                        lo que predice la teoria (ver verificacion.py).
  E1  Ruido             Nivel de ruido del sensor: sigma_v de 2 a 40 m3.
  E2  Comparacion       Condicion nominal en detalle: filtro frente a medicion
                        cruda y estimadores de referencia, con estadistica pareada.
  E2b Filtro calibrado  Factor sobre q_e elegido en las corridas de calibracion,
                        con el mismo privilegio que las ventanas de las lineas base.
  E2c Ablacion          Mismas semillas con el proceso SIN saturacion
                        (saturacion=False): contrafactual fisicamente imposible
                        para aislar el efecto del vaciado. En las corridas que se
                        vacian cambian la trayectoria y, a veces, las descargas u.
  E3a Sintonia          Error en la covarianza de proceso supuesta por el filtro.
  E3b Estructura        Discrepancias en el sensor que el filtro no modela:
                        ruido coloreado, ruido intensificado en descargas, deriva.
  E4  Perturbacion      Cambio brusco de la dureza del mineral: la capacidad de
                        extraccion cae 25 % a mitad de la hora.

Semillas y numeros aleatorios comunes
-------------------------------------
Toda la aleatoriedad deriva de una semilla maestra mediante
numpy.random.SeedSequence con claves (experimento, fase, corrida, flujo). El
proceso y el sensor usan flujos separados. Dentro de un experimento, todas las
condiciones reutilizan las mismas trayectorias del proceso y las mismas normales
del sensor, de modo que las diferencias entre condiciones se deben solo a la
condicion. E4 usa un lote propio, porque su proceso es distinto. Las corridas de
calibracion (fase 1) usan semillas distintas de las de prueba (fase 0), para que
el ajuste de las lineas base no se haga sobre los mismos datos con que se
evaluan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from estimador import EstimadorEnLinea, FiltroKalman
from evaluador import autocorrelacion, comparar, mae, nees, nis, resumen, rmse
from parametros import (ParametrosEstimador, ParametrosExperimento,
                        ParametrosProceso, ParametrosSensor, estimador_nominal)
from referencias import diferencias_finitas, media_movil, regresion_local
from sensor import medir_desde_eps
from simulador import simular_proceso

# Claves de las semillas. E1, E2, E3a y E3b comparten el lote BASE; las claves
# 2 y 3 no se usan.
EXPERIMENTOS = {"EV": 0, "BASE": 1, "E4": 4}
PRUEBA, CALIBRACION = 0, 1
FLUJO_PROCESO, FLUJO_SENSOR = 0, 1
CLAVE_NULA = 98            # Monte Carlo de la tasa de falsas alarmas de referencia


def rng_para(pexp: ParametrosExperimento, experimento: str, fase: int, corrida: int,
             flujo: int) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=pexp.semilla_maestra,
                                spawn_key=(EXPERIMENTOS[experimento], fase, corrida, flujo))
    return np.random.default_rng(ss)


@dataclass
class Lote:
    V: np.ndarray     # (M, N) volumen verdadero
    E: np.ndarray     # (M, N) extraccion efectiva verdadera
    U: np.ndarray     # (M, N) descarga conocida
    D: np.ndarray     # (M, N) hay descarga en curso
    eps: np.ndarray   # (M, N) normales estandar del sensor
    camiones: np.ndarray
    espera: np.ndarray
    vacia: np.ndarray
    Ecap: np.ndarray | None = None   # (M, N) capacidad de extraccion (sin saturar)


def generar_lote(pp: ParametrosProceso, pexp: ParametrosExperimento,
                 experimento: str, fase: int, M: int) -> Lote:
    tr = [simular_proceso(pp, rng_para(pexp, experimento, fase, m, FLUJO_PROCESO))
          for m in range(M)]
    eps = np.array([rng_para(pexp, experimento, fase, m, FLUJO_SENSOR).standard_normal(pp.N)
                    for m in range(M)])
    return Lote(V=np.array([t.V for t in tr]), E=np.array([t.e for t in tr]),
                U=np.array([t.u for t in tr]), D=np.array([t.descargando for t in tr]),
                eps=eps, camiones=np.array([t.n_camiones for t in tr]),
                espera=np.array([t.espera_total for t in tr]),
                vacia=np.array([t.pasos_vacia for t in tr]),
                Ecap=np.array([t.e_capacidad for t in tr]))


# ------------------------------------------------------------------ estimadores
def calibrar_ventanas(lote: Lote, Y: np.ndarray, dt: float, ventanas, q: int) -> dict:
    """Elige la ventana de cada linea base minimizando su RMSE medio en las
    corridas de calibracion, con acceso a la verdad.

    Esto les da a las lineas base acceso a la verdad para ajustarse, algo
    imposible en una planta: su desempeno es una cota optimista de lo que
    lograrian en operacion. El filtro tampoco esta en condiciones de planta
    (usa los parametros verdaderos del sensor); la comparacion con el mismo
    privilegio es la del Experimento E2b."""
    err = {"mm": [], "rl_V": [], "rl_e": []}
    for W in ventanas:
        err["mm"].append(rmse(media_movil(Y, W), lote.V, q).mean())
        Vr, er = regresion_local(Y, lote.U, dt, W)
        err["rl_V"].append(rmse(Vr, lote.V, q).mean())
        err["rl_e"].append(rmse(er, lote.E, q).mean())
    return {k: int(ventanas[int(np.argmin(v))]) for k, v in err.items()}


def estimar_todo(lote: Lote, Y: np.ndarray, pe: ParametrosEstimador, W: dict, dt: float):
    """Todos los estimadores sobre el mismo lote y las mismas mediciones."""
    kf = FiltroKalman(pe).filtrar(Y, lote.U)
    V_rl, _ = regresion_local(Y, lote.U, dt, W["rl_V"])
    _, e_rl = regresion_local(Y, lote.U, dt, W["rl_e"])
    est = {"V": {"medicion": Y, "media_movil": media_movil(Y, W["mm"]),
                 "regresion_local": V_rl, "kalman": kf.x[:, :, 0]},
           "e": {"dif_finitas": diferencias_finitas(Y, lote.U, dt),
                 "regresion_local": e_rl, "kalman": kf.x[:, :, 1]}}
    return est, kf


def metricas(est: dict, lote: Lote, q: int) -> dict:
    verdad = {"V": lote.V, "e": lote.E}
    return {var: {n: {"rmse": rmse(x, verdad[var], q), "mae": mae(x, verdad[var], q)}
                  for n, x in est[var].items()} for var in est}


def fila_condicion(met: dict, kf, lote: Lote, q: int, rng, B: int) -> dict:
    """Resumen de una condicion: error de cada estimador, comparacion del
    filtro con la regresion local (todas las corridas y separando las corridas
    con y sin vaciado) y diagnosticos de consistencia del filtro."""
    con = lote.vacia > 0
    F = {}
    for var in ("V", "e"):
        F[var] = {n: {"rmse": resumen(m["rmse"], rng, B), "mae": resumen(m["mae"], rng, B)}
                  for n, m in met[var].items()}
        a, b = met[var]["kalman"]["rmse"], met[var]["regresion_local"]["rmse"]
        F[var]["_kf_vs_ref"] = comparar(a, b, rng, B)
        F[var]["_kf_vs_ref_sin_vaciado"] = comparar(a[~con], b[~con], rng, B) if (~con).sum() >= 5 else None
        F[var]["_kf_vs_ref_con_vaciado"] = comparar(a[con], b[con], rng, B) if con.sum() >= 5 else None
    verdad = np.stack([lote.V, lote.E], axis=-1)
    F["nees_medio"] = float(nees(verdad - kf.x, kf.P)[:, q:].mean())
    NIS = nis(kf.nu, kf.S)[:, q:]
    F["nis_medio"] = float(NIS.mean())
    F["nis_fuera_banda"] = float(np.mean(NIS > stats.chi2.ppf(0.95, 1)))
    inn = kf.nu[:, q:] / np.sqrt(kf.S[None, q:])
    F["autocorr_innov"] = float(autocorrelacion(inn, 1).mean())
    return F


def _regimen(pe: ParametrosEstimador, kf) -> dict:
    """Dinamica del error en regimen, A (I - K H), expresada como un sistema
    continuo de segundo orden."""
    K = kf.K[-1]
    F = pe.A @ (np.eye(2) - np.outer(K, pe.H.ravel()))
    lam = np.linalg.eigvals(F)
    lam = lam[np.argsort(-lam.imag)]
    s = np.log(lam.astype(complex)) / pe.dt
    wn = float(np.abs(s[0]))
    return {"autovalores_re": lam.real.tolist(), "autovalores_im": lam.imag.tolist(),
            "modulo": float(np.abs(lam[0])),
            "constante_tiempo_s": float(-pe.dt / np.log(np.abs(lam[0]))),
            "wn_rad_s": wn, "zeta": float(-s[0].real / wn),
            "periodo_amortiguado_s": float(2 * np.pi / abs(s[0].imag)) if abs(s[0].imag) > 0 else None,
            "wn_aprox_continuo": float((pe.q_e / (pe.R * pe.dt ** 2)) ** 0.25),
            "S_estacionaria": float(kf.S[-1])}


# ------------------------------------------------------------------ principal
def ejecutar(pp: ParametrosProceso, ps0: ParametrosSensor, pexp: ParametrosExperimento, log=print):
    """Ejecuta E1 a E4. Devuelve (R, series)."""
    dt, q, B = pp.dt, pexp.quemado, pexp.B_bootstrap
    rng_est = np.random.default_rng(np.random.SeedSequence(entropy=pexp.semilla_maestra,
                                                           spawn_key=(99,)))
    pe = estimador_nominal(pp, ps0)
    R: dict = {}
    series: dict = {}

    log("  Generando el lote comun y las corridas de calibracion ...")
    base = generar_lote(pp, pexp, "BASE", PRUEBA, pexp.M)
    base_cal = generar_lote(pp, pexp, "BASE", CALIBRACION, pexp.M_calibracion)
    con = base.vacia > 0

    R["proceso"] = {
        "camiones_por_hora": resumen(base.camiones * 3600.0 / (pp.N * dt), rng_est, B),
        "espera_por_camion_s": resumen(base.espera / np.maximum(base.camiones, 1), rng_est, B),
        "fraccion_tiempo_vacia": resumen(np.mean(base.V <= 1e-9, axis=1), rng_est, B),
        "corridas_con_vaciado": int(con.sum()),
        "pasos_saturados_por_corrida": base.vacia.tolist(),
        "V_min": float(base.V.min()), "V_medio": float(base.V.mean()), "V_max": float(base.V.max()),
        "e_media": float(base.E.mean()),
    }

    # -------------------------------------------------------- E2 nominal
    log("  E2: condicion nominal ...")
    Y0 = medir_desde_eps(base.V, base.D, base.eps, ps0, dt)
    Y0_cal = medir_desde_eps(base_cal.V, base_cal.D, base_cal.eps, ps0, dt)
    W0 = calibrar_ventanas(base_cal, Y0_cal, dt, pexp.ventanas, q)
    est0, kf0 = estimar_todo(base, Y0, pe, W0, dt)
    met0 = metricas(est0, base, q)
    R["E2"] = fila_condicion(met0, kf0, base, q, rng_est, B)
    R["E2"]["ventanas"] = W0
    R["E2"]["comparaciones"] = {
        "V": {n: comparar(met0["V"]["kalman"]["rmse"], met0["V"][n]["rmse"], rng_est, B)
              for n in ("medicion", "media_movil", "regresion_local")},
        "e": {n: comparar(met0["e"]["kalman"]["rmse"], met0["e"][n]["rmse"], rng_est, B)
              for n in ("dif_finitas", "regresion_local")}}
    Err0 = np.stack([base.V, base.E], axis=-1) - kf0.x
    NEES0 = nees(Err0, kf0.P)[:, q:]
    NIS0 = nis(kf0.nu, kf0.S)[:, q:]
    R["E2"]["por_vaciado"] = {}
    for etiqueta, sel in (("con_vaciado", con), ("sin_vaciado", ~con)):
        d = {"n_corridas": int(sel.sum())}
        for var in ("V", "e"):
            d[var] = {n: resumen(met0[var][n]["rmse"][sel], rng_est, B)
                      for n in ("kalman", "regresion_local")}
            d[var]["kf_vs_rl"] = comparar(met0[var]["kalman"]["rmse"][sel],
                                          met0[var]["regresion_local"]["rmse"][sel], rng_est, B)
        d["nees_medio"] = float(NEES0[sel].mean())
        d["nis_medio"] = float(NIS0[sel].mean())
        R["E2"]["por_vaciado"][etiqueta] = d

    R["E2"]["K_estacionaria"] = kf0.K[-1].tolist()
    R["E2"]["P_estacionaria_de"] = np.sqrt(np.diag(kf0.P[-1])).tolist()
    R["E2"]["regimen"] = _regimen(pe, kf0)
    Vkf = est0["V"]["kalman"][:, q:]
    R["E2"]["V_estimado_min"] = float(Vkf.min())
    R["E2"]["fraccion_V_estimado_negativo"] = float(np.mean(Vkf < 0))

    # Pasos hasta que la ganancia queda a menos de 0,5 % de su valor de regimen
    relK = np.max(np.abs(kf0.K - kf0.K[-1]) / np.abs(kf0.K[-1]), axis=1)
    fuera = np.flatnonzero(relK >= 0.005)
    R["E2"]["pasos_convergencia_ganancia"] = int(fuera[-1] + 1) if fuera.size else 0
    R["E2"]["ganancia_rel_en_60"] = (np.abs(kf0.K[60] - kf0.K[-1]) / np.abs(kf0.K[-1])).tolist()

    # Estratificacion (exploratoria) por la duracion de los vaciados: numero de
    # pasos con la extraccion saturada en la corrida.
    cortes = ((0, 1, "0"), (1, 30, "1-29"), (30, 120, "30-119"), (120, pp.N + 1, ">=120"))
    R["E2"]["por_duracion_vaciado"] = []
    for a_, b_, etq in cortes:
        sel = (base.vacia >= a_) & (base.vacia < b_)
        fila = {"pasos": etq, "n_corridas": int(sel.sum())}
        for var in ("V", "e"):
            a_r, b_r = met0[var]["kalman"]["rmse"][sel], met0[var]["regresion_local"]["rmse"][sel]
            if sel.sum() == 0:
                fila[var] = {"reduccion": None, "red_mediana": None, "gana_filtro": None}
                continue
            fila[var] = {"reduccion": float(1 - a_r.mean() / b_r.mean()),
                         "red_mediana": float(np.median(1 - a_r / b_r)),
                         "gana_filtro": float(np.mean(a_r < b_r))}
        R["E2"]["por_duracion_vaciado"].append(fila)

    # Sensibilidad al periodo de quemado
    R["E2"]["sensibilidad_quemado"] = {}
    for qq in (60, 300, 600):
        mq = metricas(est0, base, qq)
        dq = {}
        for var in ("V", "e"):
            r_kf, r_rl = mq[var]["kalman"]["rmse"], mq[var]["regresion_local"]["rmse"]
            dq[var] = {"todas": float(1 - r_kf.mean() / r_rl.mean()),
                       "sin_vaciado": float(1 - r_kf[~con].mean() / r_rl[~con].mean()),
                       "con_vaciado": float(1 - r_kf[con].mean() / r_rl[con].mean())}
        R["E2"]["sensibilidad_quemado"][str(qq)] = dq

    # Serie representativa: la primera corrida, en orden de indice, en que la
    # tolva se vacio (permite ver los dos regimenes). Se guarda la hora completa.
    m0 = int(np.flatnonzero(con)[0]) if con.any() else 0
    sd = np.sqrt(kf0.P[:, [0, 1], [0, 1]])
    series["serie"] = {
        "corrida": m0, "t": (np.arange(pp.N) * dt).tolist(),
        "V": base.V[m0].tolist(), "y": Y0[m0].tolist(),
        "V_mm": est0["V"]["media_movil"][m0].tolist(), "V_rl": est0["V"]["regresion_local"][m0].tolist(),
        "V_kf": est0["V"]["kalman"][m0].tolist(), "de_V": sd[:, 0].tolist(),
        "e": base.E[m0].tolist(), "e_cap": base.Ecap[m0].tolist(),
        "e_rl": est0["e"]["regresion_local"][m0].tolist(), "e_kf": est0["e"]["kalman"][m0].tolist(),
        "de_e": sd[:, 1].tolist(), "descargando": base.D[m0].astype(int).tolist()}

    # -------------------------------------------------------- E2b calibrado
    log("  E2b: filtro calibrado en las corridas de calibracion ...")
    cal = {"V": [], "e": []}
    for f in pexp.factores_qe:
        kf_c = FiltroKalman(replace(pe, q_e=pe.q_e * f)).filtrar(Y0_cal, base_cal.U)
        cal["V"].append(rmse(kf_c.x[:, :, 0], base_cal.V, q).mean())
        cal["e"].append(rmse(kf_c.x[:, :, 1], base_cal.E, q).mean())
    fV = float(pexp.factores_qe[int(np.argmin(cal["V"]))])
    fe = float(pexp.factores_qe[int(np.argmin(cal["e"]))])
    R["E2b"] = {"factor_V": fV, "factor_e": fe, "rmse_calibracion": cal,
                "corridas_calibracion_con_vaciado": int((base_cal.vacia > 0).sum())}
    for var, f, j in (("V", fV, 0), ("e", fe, 1)):
        kf_b = FiltroKalman(replace(pe, q_e=pe.q_e * f)).filtrar(Y0, base.U)
        verdad = base.V if var == "V" else base.E
        r_kf = rmse(kf_b.x[:, :, j], verdad, q)
        r_rl = met0[var]["regresion_local"]["rmse"]
        R["E2b"][var] = {"todas": comparar(r_kf, r_rl, rng_est, B),
                         "sin_vaciado": comparar(r_kf[~con], r_rl[~con], rng_est, B),
                         "con_vaciado": comparar(r_kf[con], r_rl[con], rng_est, B)}

    # -------------------------------------------------------- E2c ablacion
    # Ablacion de la saturacion. Mismas semillas, pero la extraccion nunca se
    # satura (contrafactual fisicamente imposible). Se comparan las mismas
    # corridas, separadas segun si la tolva se vacio en el proceso real.
    log("  E2c: ablacion de la saturacion ...")
    lote_abl = generar_lote(replace(pp, saturacion=False), pexp, "BASE", PRUEBA, pexp.M)
    Y_abl = medir_desde_eps(lote_abl.V, lote_abl.D, lote_abl.eps, ps0, dt)
    est_abl, kf_abl = estimar_todo(lote_abl, Y_abl, pe, W0, dt)
    met_abl = metricas(est_abl, lote_abl, q)
    NEES_abl = nees(np.stack([lote_abl.V, lote_abl.E], axis=-1) - kf_abl.x, kf_abl.P)[:, q:]
    NIS_abl = nis(kf_abl.nu, kf_abl.S)[:, q:]
    # La regla de descarga depende del volumen (la carga debe caber), de modo
    # que en algunas corridas con vaciado cambian tambien las descargas u.
    u_distinta = [not np.array_equal(lote_abl.U[m], base.U[m]) for m in np.flatnonzero(con)]
    R["E2c"] = {"V_min_verdad": float(lote_abl.V.min()),
                "corridas_identicas_sin_vaciado": bool(np.array_equal(lote_abl.V[~con], base.V[~con])
                                                       and np.array_equal(lote_abl.U[~con], base.U[~con])),
                "corridas_con_vaciado_u_distinta": int(np.sum(u_distinta)),
                "espera_por_camion_con_vaciado_s": {
                    "proceso": float(base.espera[con].sum() / base.camiones[con].sum()),
                    "ablacion": float(lote_abl.espera[con].sum() / lote_abl.camiones[con].sum())}}
    for etq, sel in (("todas", np.ones_like(con)), ("sin_vaciado", ~con), ("con_vaciado", con)):
        dd = {"n_corridas": int(sel.sum())}
        for var in ("V", "e"):
            dd[var] = {n: resumen(met_abl[var][n]["rmse"][sel], rng_est, B)
                       for n in ("kalman", "regresion_local")}
            dd[var]["kf_vs_rl"] = comparar(met_abl[var]["kalman"]["rmse"][sel],
                                           met_abl[var]["regresion_local"]["rmse"][sel], rng_est, B)
        dd["nees_medio"] = float(NEES_abl[sel].mean())
        dd["nis_medio"] = float(NIS_abl[sel].mean())
        R["E2c"][etq] = dd

    # Costo computacional: filtro corrida por corrida, como operaria en linea.
    # Es una medicion de tiempo, no un resultado determinista: se guarda aparte
    # (rendimiento.json) y queda fuera del manifiesto de reproducibilidad.
    # Se mide con la interfaz en linea, una medicion por llamada, que es como
    # operaria conectado a un flujo de datos.
    rep = 20
    t0 = time.perf_counter()
    for m in range(rep):
        est = EstimadorEnLinea(pe)
        for k in range(pp.N):
            est.actualizar(Y0[m, k], base.U[m, k - 1] if k > 0 else None)
    R["_rendimiento"] = {"us_por_paso_en_linea": 1e6 * (time.perf_counter() - t0) / (rep * pp.N),
                         "corridas_medidas": rep, "pasos_por_corrida": pp.N}

    # -------------------------------------------------------- E1 ruido
    log("  E1: nivel de ruido del sensor ...")
    R["E1"] = {"sigma_v": [], "ventanas": [], "filas": []}
    for s in (2.0, 5.0, 10.0, 20.0, 40.0):
        ps = replace(ps0, sigma_v=s)
        Y = medir_desde_eps(base.V, base.D, base.eps, ps, dt)
        Yc = medir_desde_eps(base_cal.V, base_cal.D, base_cal.eps, ps, dt)
        W = calibrar_ventanas(base_cal, Yc, dt, pexp.ventanas, q)
        est, kf = estimar_todo(base, Y, replace(pe, R=s ** 2), W, dt)
        R["E1"]["sigma_v"].append(s)
        R["E1"]["ventanas"].append(W)
        R["E1"]["filas"].append(fila_condicion(metricas(est, base, q), kf, base, q, rng_est, B))

    # ------------------------------------------------------ E3a sintonia
    log("  E3a: error de sintonia ...")
    R["E3a"] = {"factor_qe": [], "filas": [], "escala_comun": [], "filas_escala": []}
    for f in pexp.factores_qe:
        est, kf = estimar_todo(base, Y0, replace(pe, q_e=pe.q_e * f), W0, dt)
        R["E3a"]["factor_qe"].append(float(f))
        R["E3a"]["filas"].append(fila_condicion(metricas(est, base, q), kf, base, q, rng_est, B))
    for c in (0.1, 1.0, 10.0):
        pc = replace(pe, q_e=c * pe.q_e, R=c * pe.R, P0_V=c * pe.P0_V, P0_e=c * pe.P0_e)
        est, kf = estimar_todo(base, Y0, pc, W0, dt)
        F = fila_condicion(metricas(est, base, q), kf, base, q, rng_est, B)
        F["K_estacionaria"] = kf.K[-1].tolist()
        R["E3a"]["escala_comun"].append(c)
        R["E3a"]["filas_escala"].append(F)

    # ------------------------------------------------- E3b discrepancias
    log("  E3b: discrepancias del sensor ...")
    R["E3b"] = {}
    barridos = {"rho": (0.0, 0.5, 0.8, 0.9, 0.95),
                "factor_descarga": (1.0, 2.0, 3.0, 5.0),
                "deriva_total": (0.0, 10.0, 20.0, 40.0)}
    for nombre, valores in barridos.items():
        R["E3b"][nombre] = {"valores": list(valores), "filas": []}
        for val in valores:
            ps = replace(ps0, **{nombre: val})
            Y = medir_desde_eps(base.V, base.D, base.eps, ps, dt)
            est, kf = estimar_todo(base, Y, pe, W0, dt)
            R["E3b"][nombre]["filas"].append(fila_condicion(metricas(est, base, q), kf, base, q, rng_est, B))

    # Sintonia bajo ruido coloreado: con rho = 0,5 y 0,95, que sucede si se
    # sigue la autocorrelacion positiva de la innovacion y se aumenta q_e. Los
    # mismos factores estan en la grilla de E3a (ruido blanco).
    R["E3b"]["sintonia_coloreado"] = []
    for rho_ in (0.5, 0.95):
        Y_rho = medir_desde_eps(base.V, base.D, base.eps, replace(ps0, rho=rho_), dt)
        dd = {"rho": rho_, "factor_qe": [], "rmse_V": [], "rmse_e": [], "nis": [], "r1": []}
        for f in (0.1, 0.3, 1.0, 3.0, 10.0):
            kf = FiltroKalman(replace(pe, q_e=pe.q_e * f)).filtrar(Y_rho, base.U)
            inn = kf.nu[:, q:] / np.sqrt(kf.S[None, q:])
            dd["factor_qe"].append(f)
            dd["rmse_V"].append(float(rmse(kf.x[:, :, 0], base.V, q).mean()))
            dd["rmse_e"].append(float(rmse(kf.x[:, :, 1], base.E, q).mean()))
            dd["nis"].append(float(nis(kf.nu, kf.S)[:, q:].mean()))
            dd["r1"].append(float(autocorrelacion(inn, 1).mean()))
        R["E3b"]["sintonia_coloreado"].append(dd)

    # -------------------------------------------------- E4 perturbacion
    log("  E4: cambio de dureza ...")
    t_e = pp.N * dt / 2
    pp4 = replace(pp, t_escalon=t_e, factor_escalon=0.75)
    lote4 = generar_lote(pp4, pexp, "E4", PRUEBA, pexp.M)
    Y4 = medir_desde_eps(lote4.V, lote4.D, lote4.eps, ps0, dt)
    est4, kf4 = estimar_todo(lote4, Y4, pe, W0, dt)
    k_e = int(t_e / dt)
    salto = pp.e_nom * (1.0 - 0.75)                 # 0,2 m3/s
    umbral = 0.10 * salto                           # banda de 10 % del salto
    sostener = 120                                  # s dentro de la banda
    tramos = {"antes": (q, k_e),
              "transitorio_0_5min": (k_e, k_e + 300),
              "recuperado_10_30min": (k_e + 600, pp.N)}
    R["E4"] = {"t_escalon": t_e, "factor": 0.75, "salto_m3s": salto,
               "umbral_asentamiento_m3s": umbral, "sostener_s": sostener,
               "tramos": {k: [int(a), int(b)] for k, (a, b) in tramos.items()},
               "rmse_e": {}, "rmse_V": {}, "asentamiento_s": {},
               "fraccion_antes_dentro_banda": {}, "comparaciones_e": {}}
    rm = {}
    for nom in ("kalman", "regresion_local"):
        err_e = est4["e"][nom] - lote4.E
        err_V = est4["V"][nom] - lote4.V
        rm[nom] = {tr: np.sqrt(np.mean(err_e[:, a:b] ** 2, axis=1)) for tr, (a, b) in tramos.items()}
        R["E4"]["rmse_e"][nom] = {tr: resumen(x, rng_est, B) for tr, x in rm[nom].items()}
        R["E4"]["rmse_V"][nom] = {
            tr: resumen(np.sqrt(np.mean(err_V[:, a:b] ** 2, axis=1)), rng_est, B)
            for tr, (a, b) in tramos.items()}
        # Asentamiento sobre la respuesta TIPICA: mediana entre corridas del
        # error de extraccion. Es el primer instante tras el escalon desde el
        # cual esa mediana queda dentro de +-10 % del salto durante 120 s.
        # Antes del escalon se registra que fraccion del tiempo la mediana ya
        # estaba dentro de la banda, para comprobar que el criterio discrimina.
        med = np.median(err_e, axis=0)
        dentro = np.abs(med[k_e:]) < umbral
        corrida = np.convolve(dentro.astype(int), np.ones(sostener, int), "valid") == sostener
        idx = np.flatnonzero(corrida)
        R["E4"]["asentamiento_s"][nom] = float(idx[0] * dt) if idx.size else None
        R["E4"]["fraccion_antes_dentro_banda"][nom] = float(np.mean(np.abs(med[q:k_e]) < umbral))
    for tr in tramos:
        R["E4"]["comparaciones_e"][tr] = comparar(rm["kalman"][tr], rm["regresion_local"][tr],
                                                  rng_est, B)

    # ---- Prediccion teorica del efecto del escalon ----------------------
    # (a) Forma cerrada del lazo continuo de segundo orden (Capitulo 2): el
    #     sesgo de prediccion del volumen alcanza 0,456 * salto / wn unos
    #     1,11 / wn s despues del escalon.
    reg = R["E2"]["regimen"]
    wn_ = reg["wn_rad_s"]; z_ = reg["zeta"]; wd_ = wn_ * np.sqrt(1 - z_**2)
    t_pico = float(np.arctan2(wd_, z_ * wn_) / wd_)
    sesgo = float(salto / wd_ * np.exp(-z_ * wn_ * t_pico) * np.sin(wd_ * t_pico))
    # (b) Recursion exacta del error del filtro discreto en regimen, sin ruido:
    #     a posteriori, en el instante del cambio el error de la tasa es -salto;
    #     luego x-[k+1] = A x[k] y x[k+1] = (I - K H) x-[k+1]. La media de la
    #     innovacion es la componente de volumen de x-, y el NIS esperado es
    #     1 + media^2 / S (el filtro sigue siendo consistente en lo demas).
    Kss = np.array(R["E2"]["K_estacionaria"])
    IKH = np.eye(2) - np.outer(Kss, pe.H.ravel())
    S_ss = reg["S_estacionaria"]
    L4 = pp.N - k_e
    xt = np.array([0.0, -salto])
    sesgo_seq = np.zeros(L4)                 # indices k_e+1 ... : media de la innovacion
    err_e_teo = np.zeros(L4 + 1)             # indices k_e ... : error de la tasa (est - verdad)
    err_e_teo[0] = salto
    for j in range(L4):
        xp = pe.A @ xt
        sesgo_seq[j] = xp[0]
        xt = IKH @ xp
        err_e_teo[j + 1] = -xt[1]
    enis = np.ones(pp.N)
    enis[k_e + 1:] = 1 + sesgo_seq[:pp.N - k_e - 1] ** 2 / S_ss
    Wd = 60
    enis60 = media_movil(enis, Wd)[0]
    j60 = int(np.argmax(enis60[k_e:k_e + 300])); ji = int(np.argmax(enis[k_e:k_e + 300]))
    err_teo_full = np.zeros(pp.N); err_teo_full[k_e:] = err_e_teo[:pp.N - k_e]
    R["E4"]["prediccion_teorica"] = {
        "t_pico_s": t_pico, "sesgo_pico_V_m3": sesgo,
        "nis_pico": float(1 + sesgo**2 / S_ss),
        "nis_pico_discreto": float(enis[k_e + ji]), "t_pico_discreto_s": float(ji * dt),
        "nis60_pico": float(enis60[k_e + j60]), "t_nis60_pico_s": float(j60 * dt),
        "absorbido_60s": float(1 - err_e_teo[60] / salto),
        "absorbido_120s": float(1 - err_e_teo[120] / salto),
        # sesgo_seq[j] corresponde al paso k_e + 1 + j
        "sesgo_pico_discreto_m3": float(np.max(np.abs(sesgo_seq[:300]))),
        "t_sesgo_pico_discreto_s": float((int(np.argmax(np.abs(sesgo_seq[:300]))) + 1) * dt)}

    # ---- NIS como detector en linea -------------------------------------
    # Media movil causal de 60 s del NIS comparada con el cuantil 99 % de la
    # media de 60 variables chi2(1). Se cuenta en que fraccion de corridas hay
    # al menos una alarma en los 300 s previos al escalon y en los 300 s
    # posteriores. La tasa de referencia (hipotesis nula) es la de un filtro
    # perfectamente consistente: NIS independiente chi2(1) en cada paso.
    NIS4 = nis(kf4.nu, kf4.S)
    umbral_nis = float(stats.chi2.ppf(0.99, Wd) / Wd)
    nis_mm = media_movil(NIS4, Wd)
    ventanas_nis = {"antes_5min": (k_e - 300, k_e), "despues_0_5min": (k_e, k_e + 300),
                    "despues_10_30min": (k_e + 600, pp.N)}
    rng_nulo = np.random.default_rng(np.random.SeedSequence(entropy=pexp.semilla_maestra,
                                                            spawn_key=(CLAVE_NULA,)))
    nulo = rng_nulo.chisquare(1, size=(20000, 300 + Wd - 1))
    cs = np.concatenate([np.zeros((nulo.shape[0], 1)), np.cumsum(nulo, axis=1)], axis=1)
    mm_nulo = (cs[:, Wd:] - cs[:, :-Wd]) / Wd          # 300 medias moviles completas
    fa_nula = float(np.mean(np.any(mm_nulo > umbral_nis, axis=1)))
    media_mm = nis_mm.mean(axis=0)
    jo = int(np.argmax(media_mm[k_e:k_e + 300]))
    med_err_kf = np.median(est4["e"]["kalman"] - lote4.E, axis=0)
    R["E4"]["nis"] = {
        "medio": {k: float(NIS4[:, a:b].mean()) for k, (a, b) in ventanas_nis.items()},
        "ventana_detector_s": Wd, "umbral_detector": umbral_nis,
        "fraccion_corridas_con_alarma": {
            k: float(np.mean(np.any(nis_mm[:, a:b] > umbral_nis, axis=1)))
            for k, (a, b) in ventanas_nis.items() if k != "despues_10_30min"},
        "fraccion_alarma_nula_300s": fa_nula,
        "nis60_media_antes": float(media_mm[k_e - 300:k_e].mean()),
        "nis60_media_pico": float(media_mm[k_e + jo]), "t_nis60_media_pico_s": float(jo * dt)}
    R["E4"]["absorbido_observado"] = {"60s": float(1 - med_err_kf[k_e + 60] / salto),
                                      "120s": float(1 - med_err_kf[k_e + 120] / salto)}
    # Vaciados antes y despues del cambio: fraccion de pasos con la extraccion
    # saturada y NIS de las corridas que nunca se vaciaron. Sirven para
    # interpretar el nivel del NIS antes y despues del cambio.
    sat4 = (lote4.Ecap - lote4.E) > 1e-12
    nunca = lote4.vacia == 0
    R["E4"]["vaciado"] = {
        "fraccion_pasos_saturados": {k: float(sat4[:, a:b].mean()) for k, (a, b) in ventanas_nis.items()},
        "fraccion_corridas_con_vaciado": {k: float(np.mean(np.any(sat4[:, a:b], axis=1)))
                                          for k, (a, b) in ventanas_nis.items()},
        "corridas_sin_vaciado": int(nunca.sum()),
        "nis_antes_5min_corridas_sin_vaciado": float(NIS4[nunca, k_e - 300:k_e].mean()) if nunca.any() else None,
        "nis_teorico_0_5min": float(enis[k_e:k_e + 300].mean())}
    ventana = slice(k_e - 300, k_e + 900)
    series["serie_E4"] = {
        "t": (np.arange(pp.N)[ventana] * dt).tolist(),
        "e_med": np.median(lote4.E[:, ventana], axis=0).tolist(),
        "kf_med": np.median(est4["e"]["kalman"][:, ventana], axis=0).tolist(),
        "kf_p10": np.percentile(est4["e"]["kalman"][:, ventana], 10, axis=0).tolist(),
        "kf_p90": np.percentile(est4["e"]["kalman"][:, ventana], 90, axis=0).tolist(),
        "rl_med": np.median(est4["e"]["regresion_local"][:, ventana], axis=0).tolist(),
        "rl_p10": np.percentile(est4["e"]["regresion_local"][:, ventana], 10, axis=0).tolist(),
        "rl_p90": np.percentile(est4["e"]["regresion_local"][:, ventana], 90, axis=0).tolist(),
        "err_kf_med": np.median((est4["e"]["kalman"] - lote4.E)[:, ventana], axis=0).tolist(),
        "err_rl_med": np.median((est4["e"]["regresion_local"] - lote4.E)[:, ventana], axis=0).tolist(),
        "nis_mm_med": np.median(nis_mm[:, ventana], axis=0).tolist(),
        "nis_mm_media": nis_mm[:, ventana].mean(axis=0).tolist(),
        "nis_mm_p90": np.percentile(nis_mm[:, ventana], 90, axis=0).tolist(),
        "nis60_teo": enis60[ventana].tolist(),
        "err_teo": err_teo_full[ventana].tolist(),
        "umbral_nis": umbral_nis, "umbral_asentamiento": umbral,
        "t_escalon": t_e, "salto": salto}
    return R, series
