"""
tablas.py
=========
Genera las tablas LaTeX de la tesis (espanol, coma decimal) y del articulo
(ingles, punto decimal) directamente desde resultados/resultados.json.

Ningun numero de las tablas se transcribe a mano: si se vuelve a ejecutar el
estudio, basta correr este script para actualizar todas las tablas. Cada archivo
contiene solo el entorno tabular; el titulo y la etiqueta se escriben en el
documento que lo incluye.

Uso:  python3 tablas.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RAIZ = Path(__file__).parent
SALIDA = RAIZ / "resultados"
L = "es"


# ----------------------------------------------------------------- formato
def num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s.replace(".", "{,}") if L == "es" else s


def sig(x: float, cifras: int = 3) -> str:
    """Cifras significativas, sin notacion cientifica para rangos usuales."""
    if x == 0:
        return num(0.0, cifras - 1)
    nd = max(0, cifras - 1 - int(math.floor(math.log10(abs(x)))))
    return num(x, nd)


def cient(x: float, nd: int = 1) -> str:
    """Notacion cientifica legible en LaTeX (sin signos de dolar)."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    ms = f"{m:.{nd}f}"
    if L == "es":
        ms = ms.replace(".", "{,}")
    return f"{ms}\\times10^{{{e}}}"


def pct(x: float, nd: int = 1, signo: bool = True) -> str:
    v = round(100 * x, nd)
    if v == 0:
        v, signo = 0.0, False       # evita "-0,0"
    s = f"{v:+.{nd}f}" if signo else f"{v:.{nd}f}"
    s = s.replace("-", "$-$").replace("+", "$+$")
    if L == "es":
        s = s.replace(".", "{,}")
    return s


def ic(lo: float, hi: float, f=pct) -> str:
    sep = ";\\," if L == "es" else ",\\,"
    return f"[{f(lo)}{sep}{f(hi)}]"


def pval(p: float) -> str:
    if p < 1e-4:
        e = int(math.floor(math.log10(p)))
        return f"$<10^{{{e + 1}}}$"
    return num(p, 3) if p < 0.1 else num(p, 2)


def media_ic(r: dict, cifras: int = 3) -> str:
    return f"{sig(r['media'], cifras)} [{sig(r['ic_inf'], cifras)}{';' if L == 'es' else ','} {sig(r['ic_sup'], cifras)}]"


def tr(es: str, en: str) -> str:
    return es if L == "es" else en


def tabular(col: str, cabecera: list[str], filas: list[list[str]], notas: str | None = None,
            reglas: list[int] | None = None) -> str:
    """Tabular con booktabs. `reglas` son indices de fila tras los que va \\midrule."""
    out = [f"\\begin{{tabular}}{{{col}}}", "\\toprule"]
    for c in cabecera:
        if isinstance(c, list):
            out.append(" & ".join(c) + " \\\\")
        elif c.startswith("\\cmidrule") or c.rstrip().endswith("\\\\"):
            out.append(c)
        else:
            out.append(c + " \\\\")
    out.append("\\midrule")
    for i, f in enumerate(filas):
        out.append(" & ".join(f) + " \\\\")
        if reglas and i in reglas:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n"


NOMBRE = {
    "medicion": ("Medición cruda", "Raw measurement"),
    "media_movil": ("Media móvil", "Moving average"),
    "regresion_local": ("Regresión local", "Local regression"),
    "kalman": ("Filtro de Kalman", "Kalman filter"),
    "dif_finitas": ("Diferencias finitas", "Finite differences"),
}


def nombre(k: str) -> str:
    return NOMBRE[k][0 if L == "es" else 1]


# ------------------------------------------------------------------ tablas
def t_parametros(P: dict) -> str:
    pp, ps, pe, px = P["proceso"], P["sensor"], P["estimador"], P["experimento"]
    F = [
        [tr("\\multicolumn{4}{l}{\\textit{Proceso simulado (la verdad)}}", "\\multicolumn{4}{l}{\\textit{Simulated process (ground truth)}}")],
        ["$\\Delta t$", tr("Período de muestreo", "Sampling period"), num(pp["dt"], 0), "s"],
        ["$N$", tr("Pasos por corrida (1 h)", "Steps per run (1 h)"), f"{pp['N']}", "---"],
        ["$V_{\\max}$", tr("Capacidad viva de la tolva", "Live hopper capacity"), num(pp["V_max"], 0), "m$^3$"],
        ["$V_0$", tr("Volumen inicial", "Initial volume"), num(pp["V0"], 0), "m$^3$"],
        ["$\\bar e$", tr("Capacidad nominal de extracción", "Nominal extraction capacity"), num(pp["e_nom"], 2), "m$^3$/s"],
        ["$\\tau_e$", tr("Constante de tiempo de la dureza (OU)", "Hardness time constant (OU)"), num(pp["tau_e"], 0), "s"],
        ["$\\sigma_e$", tr("Desviación estacionaria de la extracción", "Stationary extraction std."), num(pp["sigma_e"], 2), "m$^3$/s"],
        ["$C$", tr("Carga por camión", "Load per truck"), num(pp["carga_camion"], 0), "m$^3$"],
        ["$T_d$", tr("Duración de la descarga", "Dumping duration"), num(pp["duracion_descarga"], 0), "s"],
        ["$\\bar T_a$", tr("Intervalo medio entre llegadas", "Mean inter-arrival time"), num(pp["intervalo_medio"], 0), "s"],
        ["$c_v$", tr("Coef. de variación de llegadas (gamma)", "Inter-arrival CV (gamma)"), num(pp["cv_llegadas"], 1), "---"],
        [tr("\\multicolumn{4}{l}{\\textit{Sensor (condición nominal)}}", "\\multicolumn{4}{l}{\\textit{Sensor (nominal condition)}}")],
        ["$\\sigma_v$", tr("Desviación del error de medición", "Measurement error std."), num(ps["sigma_v"], 0), "m$^3$"],
        ["$\\rho$", tr("Autocorrelación AR(1)", "AR(1) autocorrelation"), num(ps["rho"], 0), "---"],
        ["$f_d$", tr("Factor de ruido durante descargas", "Noise factor while dumping"), num(ps["factor_descarga"], 0), "---"],
        ["$b_{N}$", tr("Deriva acumulada al final de la hora", "Drift accumulated over the hour"), num(ps["deriva_total"], 0), "m$^3$"],
        [tr("\\multicolumn{4}{l}{\\textit{Estimador (lo que supone el filtro)}}", "\\multicolumn{4}{l}{\\textit{Estimator (what the filter assumes)}}")],
        ["$q_V$", tr("Ruido de proceso en el volumen", "Process noise on volume"), num(pe["q_V"], 0), "m$^6$"],
        ["$q_e$", tr("Ruido de proceso en la extracción", "Process noise on extraction"),
         (f"{pe['q_e']:.3e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"), "m$^6$/s$^2$"],
        ["$R$", tr("Varianza supuesta de la medición", "Assumed measurement variance"), num(pe["R"], 0), "m$^6$"],
        ["$\\hat x_0$", tr("Estado inicial supuesto", "Assumed initial state"), f"({num(pe['V0_prior'], 0)}{'; ' if L == 'es' else ', '}{num(pe['e0_prior'], 1)})", "---"],
        ["$P_0$", tr("Covarianza inicial (diagonal)", "Initial covariance (diagonal)"), f"({num(pe['P0_V'], 0)}{'; ' if L == 'es' else ', '}{num(pe['P0_e'], 4)})", "---"],
        [tr("\\multicolumn{4}{l}{\\textit{Protocolo experimental}}", "\\multicolumn{4}{l}{\\textit{Experimental protocol}}")],
        ["$M$", tr("Corridas de prueba por condición", "Test runs per condition"), f"{px['M']}", "---"],
        ["$M_c$", tr("Corridas de calibración de las líneas base", "Baseline calibration runs"), f"{px['M_calibracion']}", "---"],
        ["$k_q$", tr("Pasos de quemado excluidos", "Burn-in steps excluded"), f"{px['quemado']}", tr("pasos", "steps")],
        ["$B$", tr("Remuestreos bootstrap", "Bootstrap resamples"), f"{px['B_bootstrap']}", "---"],
        ["---", tr("Semilla maestra", "Master seed"), f"{px['semilla_maestra']}", "---"],
    ]
    # q_e: formato cientifico legible
    qe = pe["q_e"]
    mant, exp = f"{qe:.3e}".split("e")
    mant = mant.replace(".", "{,}") if L == "es" else mant
    F[19][2] = f"${mant}\\times10^{{{int(exp)}}}$"
    filas = []
    for f in F:
        if len(f) == 1:
            filas.append([f[0]])
        else:
            filas.append(f)
    out = ["\\begin{tabular}{@{}llrl@{}}", "\\toprule",
           tr("Símbolo & Descripción & Valor & Unidad \\\\", "Symbol & Description & Value & Unit \\\\"),
           "\\midrule"]
    for f in filas:
        out.append((f[0] if len(f) == 1 else " & ".join(f)) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def t_verificacion(V: dict) -> str:
    ok = lambda b: tr("Cumple", "Pass") if b else tr("No cumple", "Fail")
    A, Bq, C, D, E, Fb, G, Hl = (V[k] for k in ("A_simetria_P", "B_riccati", "C_insesgamiento",
                                                "D_nees", "E_nis", "F_blancura", "G_escala_comun",
                                                "H_en_linea"))
    sep = "; " if L == "es" else ", "
    filas = [
        ["A", tr("$P$ simétrica y definida positiva", "$P$ symmetric positive definite"),
         tr(f"asimetría máx. ${cient(A['asimetria_max'])}$; $\\lambda_{{\\min}}={sig(A['autovalor_min'], 2)}$",
            f"max. asymmetry ${cient(A['asimetria_max'])}$; $\\lambda_{{\\min}}={sig(A['autovalor_min'], 2)}$"), ok(A["ok"])],
        ["B", tr("Ganancia = solución de Riccati (SciPy)", "Gain = Riccati solution (SciPy)"),
         tr(f"diferencia ${cient(Bq['diferencia'])}$", f"difference ${cient(Bq['diferencia'])}$"), ok(Bq["ok"])],
        ["C", tr("Error insesgado ($t$ de una muestra)", "Unbiased error (one-sample $t$)"),
         f"$p_V={num(C['p_volumen'], 2)}$, $p_e={num(C['p_tasa'], 2)}$", ok(C["ok"])],
        ["D", tr("NEES en la banda $\\chi^2$ del 95\\,\\%", "NEES within 95\\,\\% $\\chi^2$ band"),
         tr(f"media {num(D['media_global'], 3)}; {pct(D['fraccion_pasos_en_banda'], 1, False)}\\,\\% de los pasos en banda",
            f"mean {num(D['media_global'], 3)}; {pct(D['fraccion_pasos_en_banda'], 1, False)}\\,\\% of steps in band"), ok(D["ok"])],
        ["E", tr("NIS en la banda $\\chi^2$ del 95\\,\\%", "NIS within 95\\,\\% $\\chi^2$ band"),
         tr(f"media {num(E['media_global'], 3)}; {pct(E['fraccion_pasos_en_banda'], 1, False)}\\,\\% de los pasos en banda",
            f"mean {num(E['media_global'], 3)}; {pct(E['fraccion_pasos_en_banda'], 1, False)}\\,\\% of steps in band"), ok(E["ok"])],
        ["F", tr("Innovación blanca (rezago 1)", "White innovation (lag 1)"),
         tr(f"{pct(Fb['fraccion_corridas_en_banda'], 1, False)}\\,\\% de las corridas con $|r_1|<{num(Fb['limite'], 3)}$",
            f"{pct(Fb['fraccion_corridas_en_banda'], 1, False)}\\,\\% of runs with $|r_1|<{num(Fb['limite'], 3)}$"), ok(Fb["ok"])],
        ["G", tr("Escalar $Q$ y $R$ por $c$ no cambia $K_\\infty$", "Scaling $Q$ and $R$ by $c$ leaves $K_\\infty$ unchanged"),
         tr(f"diferencia máx. ${cient(G['diferencia_max_K'])}$", f"max. difference ${cient(G['diferencia_max_K'])}$"), ok(G["ok"])],
        ["H", tr("Interfaz en línea = implementación por lotes", "Online interface = batch implementation"),
         tr(f"diferencia máx. ${cient(Hl['diferencia_max'])}$ ({Hl['corridas']} corridas)",
            f"max. difference ${cient(Hl['diferencia_max'])}$ ({Hl['corridas']} runs)"), ok(Hl["ok"])],
    ]
    return tabular("@{}clll@{}", [tr("Prueba & Propiedad verificada & Resultado & Estado", "Test & Property & Result & Status")], filas)


def t_proceso(Pr: dict) -> str:
    filas = [
        [tr("Camiones descargados por hora", "Trucks dumped per hour"), media_ic(Pr["camiones_por_hora"], 3)],
        [tr("Espera por camión [s]", "Waiting time per truck [s]"), media_ic(Pr["espera_por_camion_s"], 3)],
        [tr("Fracción del tiempo con la tolva vacía [\\%]", "Fraction of time with empty hopper [\\%]"),
         f"{num(100 * Pr['fraccion_tiempo_vacia']['media'], 2)} [{num(100 * Pr['fraccion_tiempo_vacia']['ic_inf'], 2)}{'; ' if L == 'es' else ', '}{num(100 * Pr['fraccion_tiempo_vacia']['ic_sup'], 2)}]"],
        [tr("Corridas con al menos un vaciado", "Runs with at least one emptying"), f"{Pr['corridas_con_vaciado']} / 100"],
        [tr("Volumen mínimo / medio / máximo [m$^3$]", "Min. / mean / max. volume [m$^3$]"),
         f"{num(Pr['V_min'], 1)} / {num(Pr['V_medio'], 1)} / {num(Pr['V_max'], 1)}"],
        [tr("Extracción efectiva media [m$^3$/s]", "Mean effective extraction [m$^3$/s]"), num(Pr["e_media"], 3)],
    ]
    return tabular("@{}lr@{}", [tr("Indicador & Media [IC 95\\,\\%]", "Indicator & Mean [95\\,\\% CI]")], filas)


def t_e2(E2: dict) -> str:
    filas = []
    for k in ("medicion", "media_movil", "regresion_local", "kalman"):
        r = E2["V"][k]
        filas.append([nombre(k), media_ic(r["rmse"]), media_ic(r["mae"])])
    filas_e = []
    for k in ("dif_finitas", "regresion_local", "kalman"):
        r = E2["e"][k]
        filas_e.append([nombre(k), media_ic(r["rmse"]), media_ic(r["mae"])])
    out = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule",
           tr("Estimador & RMSE [IC 95\\,\\%] & MAE [IC 95\\,\\%] \\\\", "Estimator & RMSE [95\\,\\% CI] & MAE [95\\,\\% CI] \\\\"),
           "\\midrule",
           tr("\\multicolumn{3}{@{}l}{\\textit{Volumen $V$ [m$^3$]}} \\\\", "\\multicolumn{3}{@{}l}{\\textit{Volume $V$ [m$^3$]}} \\\\")]
    out += [" & ".join(f) + " \\\\" for f in filas]
    out += ["\\midrule", tr("\\multicolumn{3}{@{}l}{\\textit{Extracción $e$ [m$^3$/s]}} \\\\", "\\multicolumn{3}{@{}l}{\\textit{Extraction $e$ [m$^3$/s]}} \\\\")]
    out += [" & ".join(f) + " \\\\" for f in filas_e]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def t_e2_comparaciones(E2: dict) -> str:
    filas = []
    for var, lab in (("V", tr("Volumen", "Volume")), ("e", tr("Extracción", "Extraction"))):
        for k, c in E2["comparaciones"][var].items():
            filas.append([lab, nombre(k), f"{pct(c['reduccion'])} {ic(c['red_ic_inf'], c['red_ic_sup'])}",
                          pct(c["gana_filtro"], 0, False), pval(c["p_wilcoxon"])])
    return tabular("@{}llrrr@{}",
                   [tr("Variable & Filtro frente a & Reducción del RMSE [IC 95\\,\\%] & Gana [\\%] & $p$ \\\\",
                       "Variable & KF vs. & RMSE reduction [95\\,\\% CI] & Wins [\\%] & $p$ \\\\")],
                   filas, reglas=[2])


def t_e2_vaciado(E2: dict, E2c: dict | None = None) -> str:
    """Corridas con y sin vaciado (estratificacion exploratoria) y, si se da
    E2c, la misma comparacion en las corridas con vaciado con el proceso sin
    saturacion (ablacion, mismas semillas)."""
    filas = []
    casos = [(E2["por_vaciado"]["sin_vaciado"], tr("Sin vaciado", "No emptying")),
             (E2["por_vaciado"]["con_vaciado"], tr("Con vaciado", "With emptying"))]
    if E2c is not None:
        casos.append((E2c["con_vaciado"], tr("Con vaciado, sin saturación$^{a}$", "With emptying, no saturation$^{a}$")))
    for d, lab in casos:
        cV, ce = d["V"]["kf_vs_rl"], d["e"]["kf_vs_rl"]
        filas.append([lab, f"{d['n_corridas']}",
                      sig(d["V"]["kalman"]["media"]), sig(d["V"]["regresion_local"]["media"]),
                      f"{pct(cV['reduccion'])} {ic(cV['red_ic_inf'], cV['red_ic_sup'])}",
                      sig(d["e"]["kalman"]["media"]), sig(d["e"]["regresion_local"]["media"]),
                      f"{pct(ce['reduccion'])} {ic(ce['red_ic_inf'], ce['red_ic_sup'])}",
                      num(d["nees_medio"], 2), num(d["nis_medio"], 3)])
    cab = [tr("& & \\multicolumn{3}{c}{RMSE de $V$ [m$^3$]} & \\multicolumn{3}{c}{RMSE de $e$ [m$^3$/s]} & & \\\\",
              "& & \\multicolumn{3}{c}{RMSE of $V$ [m$^3$]} & \\multicolumn{3}{c}{RMSE of $e$ [m$^3$/s]} & & \\\\"),
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
           tr("Corridas & $n$ & KF & RL & Reducción & KF & RL & Reducción & NEES & NIS \\\\",
              "Runs & $n$ & KF & LR & Reduction & KF & LR & Reduction & NEES & NIS \\\\")]
    out = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule"] + cab + ["\\midrule"]
    out += [" & ".join(f) + " \\\\" for f in filas]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def t_e2_gradiente(E2: dict) -> str:
    filas = []
    for f in E2["por_duracion_vaciado"]:
        et = f["pasos"].replace(">=", "$\\ge$").replace("-", "--")
        filas.append([et, f"{f['n_corridas']}",
                      pct(f["V"]["reduccion"]), pct(f["V"]["red_mediana"]), pct(f["V"]["gana_filtro"], 0, False),
                      pct(f["e"]["reduccion"]), pct(f["e"]["red_mediana"]), pct(f["e"]["gana_filtro"], 0, False)])
    cab = [tr("& & \\multicolumn{3}{c}{Volumen} & \\multicolumn{3}{c}{Tasa} \\\\",
              "& & \\multicolumn{3}{c}{Volume} & \\multicolumn{3}{c}{Rate} \\\\"),
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
           tr("Pasos saturados & $n$ & Reducción & Mediana & Gana [\\%] & Reducción & Mediana & Gana [\\%] \\\\",
              "Saturated steps & $n$ & Reduction & Median & Wins [\\%] & Reduction & Median & Wins [\\%] \\\\")]
    return tabular("@{}lrrrrrrr@{}", cab, filas)


def t_e2b(R: dict) -> str:
    E2b = R["E2b"]
    filas = []
    for var, f, lab in (("V", E2b["factor_V"], tr("Volumen", "Volume")), ("e", E2b["factor_e"], tr("Tasa", "Rate"))):
        c, cs, cc = E2b[var]["todas"], E2b[var]["sin_vaciado"], E2b[var]["con_vaciado"]
        filas.append([lab, f"{_factor(f)}\\,$q_e$", f"{pct(c['reduccion'])} {ic(c['red_ic_inf'], c['red_ic_sup'])}",
                      pct(c["gana_filtro"], 0, False), pct(cs["reduccion"]), pct(cc["reduccion"])])
    cab = tr("Variable & Sintonía elegida & Reducción [IC 95\\,\\%] & Gana [\\%] & Sin vac. & Con vac.",
             "Variable & Selected tuning & Reduction [95\\,\\% CI] & Wins [\\%] & No empt. & Empt.")
    return tabular("@{}llrrrr@{}", [cab], filas)


def t_quemado(E2: dict) -> str:
    filas = []
    for qq, d in E2["sensibilidad_quemado"].items():
        filas.append([qq, pct(d["V"]["todas"]), pct(d["V"]["sin_vaciado"]), pct(d["V"]["con_vaciado"]),
                      pct(d["e"]["todas"]), pct(d["e"]["sin_vaciado"]), pct(d["e"]["con_vaciado"])])
    cab = [tr("& \\multicolumn{3}{c}{Volumen} & \\multicolumn{3}{c}{Tasa} \\\\", "& \\multicolumn{3}{c}{Volume} & \\multicolumn{3}{c}{Rate} \\\\"),
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           tr("Quemado [pasos] & Todas & Sin vac. & Con vac. & Todas & Sin vac. & Con vac. \\\\",
              "Burn-in [steps] & All & No empt. & Empt. & All & No empt. & Empt. \\\\")]
    return tabular("@{}lrrrrrr@{}", cab, filas)


def t_e1(E1: dict, var: str) -> str:
    """Nivel de ruido. Ademas de la reduccion del cociente de medias se da la
    mediana de la reduccion por corrida y la reduccion separando las corridas
    con y sin vaciado (estratificacion exploratoria)."""
    filas = []
    for s_, W, F in zip(E1["sigma_v"], E1["ventanas"], E1["filas"]):
        c = F[var]["_kf_vs_ref"]
        cs, cc = F[var]["_kf_vs_ref_sin_vaciado"], F[var]["_kf_vs_ref_con_vaciado"]
        if var == "V":
            fila = [num(s_, 0), f"{W['mm']} / {W['rl_V']}",
                    sig(F["V"]["medicion"]["rmse"]["media"]), sig(F["V"]["media_movil"]["rmse"]["media"]),
                    sig(F["V"]["regresion_local"]["rmse"]["media"]), sig(F["V"]["kalman"]["rmse"]["media"])]
        else:
            fila = [num(s_, 0), f"{W['rl_e']}",
                    sig(F["e"]["dif_finitas"]["rmse"]["media"]),
                    sig(F["e"]["regresion_local"]["rmse"]["media"]), sig(F["e"]["kalman"]["rmse"]["media"])]
        fila += [f"{pct(c['reduccion'])} {ic(c['red_ic_inf'], c['red_ic_sup'])}", pct(c["red_mediana"]),
                 pct(c["gana_filtro"], 0, False), pct(cs["reduccion"]), pct(cc["reduccion"])]
        filas.append(fila)
    if var == "V":
        cab = tr("$\\sigma_v$ & $W_{\\mathrm{MM}}$/$W_{\\mathrm{RL}}$ & Medición & MM & RL & KF & Reducción [IC 95\\,\\%] & Mediana & Gana [\\%] & Sin vac. & Con vac.",
                 "$\\sigma_v$ & $W_{\\mathrm{MA}}$/$W_{\\mathrm{LR}}$ & Meas. & MA & LR & KF & Reduction [95\\,\\% CI] & Median & Wins [\\%] & No empt. & Empt.")
        col = "@{}rrrrrrrrrrr@{}"
    else:
        cab = tr("$\\sigma_v$ & $W_{\\mathrm{RL}}$ & DF & RL & KF & Reducción [IC 95\\,\\%] & Mediana & Gana [\\%] & Sin vac. & Con vac.",
                 "$\\sigma_v$ & $W_{\\mathrm{LR}}$ & FD & LR & KF & Reduction [95\\,\\% CI] & Median & Wins [\\%] & No empt. & Empt.")
        col = "@{}rrrrrrrrrr@{}"
    return tabular(col, [cab], filas)


def _factor(f: float) -> str:
    return num(f, 2) if f < 1 else num(f, 0)


def t_e3a(E3a: dict) -> str:
    """Sintonia: volumen, con resumen robusto y regimenes."""
    filas = []
    for f, F in zip(E3a["factor_qe"], E3a["filas"]):
        c, cs, cc = (F["V"][k] for k in ("_kf_vs_ref", "_kf_vs_ref_sin_vaciado", "_kf_vs_ref_con_vaciado"))
        filas.append([_factor(f), sig(F["V"]["kalman"]["rmse"]["media"]), pct(c["reduccion"]),
                      pct(c["red_mediana"]), pct(c["gana_filtro"], 0, False),
                      pct(cs["reduccion"]), pct(cc["reduccion"])])
    cab = tr("Factor sobre $q_e$ & RMSE $V$ [m$^3$] & Reducción & Mediana & Gana [\\%] & Sin vac. & Con vac.",
             "Factor on $q_e$ & RMSE $V$ [m$^3$] & Reduction & Median & Wins [\\%] & No empt. & Empt.")
    return tabular("@{}rrrrrrr@{}", [cab], filas)


def t_e3a_diag(E3a: dict) -> str:
    """Sintonia: tasa de extraccion y diagnosticos de consistencia."""
    filas = []
    for f, F in zip(E3a["factor_qe"], E3a["filas"]):
        c = F["e"]["_kf_vs_ref"]
        filas.append([_factor(f), sig(F["e"]["kalman"]["rmse"]["media"]), pct(c["reduccion"]),
                      pct(c["red_mediana"]), pct(c["gana_filtro"], 0, False),
                      num(F["nis_medio"], 3), num(F["autocorr_innov"], 3), sig(F["nees_medio"], 3)])
    cab = tr("Factor sobre $q_e$ & RMSE $e$ [m$^3$/s] & Reducción & Mediana & Gana [\\%] & NIS & $r_1$ & NEES",
             "Factor on $q_e$ & RMSE $e$ [m$^3$/s] & Reduction & Median & Wins [\\%] & NIS & $r_1$ & NEES")
    return tabular("@{}rrrrrrrr@{}", [cab], filas)


def t_e3a_escala(E3a: dict) -> str:
    filas = []
    for c, F in zip(E3a["escala_comun"], E3a["filas_escala"]):
        K = F["K_estacionaria"]
        filas.append([num(c, 1) if c < 1 else num(c, 0), num(K[0], 5), num(K[1], 6),
                      sig(F["V"]["kalman"]["rmse"]["media"], 4), sig(F["e"]["kalman"]["rmse"]["media"], 4),
                      sig(F["nis_medio"], 4), sig(F["nees_medio"], 4)])
    cab = tr("$c$ & $K_{\\infty,1}$ & $K_{\\infty,2}$ & RMSE $V$ & RMSE $e$ & NIS & NEES",
             "$c$ & $K_{\\infty,1}$ & $K_{\\infty,2}$ & RMSE $V$ & RMSE $e$ & NIS & NEES")
    return tabular("@{}rrrrrrr@{}", [cab], filas)


def _filas_e3b(E3b: dict):
    """Condiciones de E3b en orden; la nominal aparece una sola vez."""
    etiquetas = {"rho": "$\\rho$", "factor_descarga": "$f_d$", "deriva_total": "$b_N$"}
    bloques = []
    for clave in ("rho", "factor_descarga", "deriva_total"):
        d = E3b[clave]
        filas = []
        for v, F in zip(d["valores"], d["filas"]):
            if clave != "rho" and v == d["valores"][0]:
                continue
            filas.append((f"{etiquetas[clave]} = {num(v, 2) if clave == 'rho' else num(v, 0)}", F))
        bloques.append(filas)
    return bloques


def _con_reglas(col: str, cab: list[str], bloques: list[list[list[str]]]) -> str:
    out = [f"\\begin{{tabular}}{{{col}}}", "\\toprule"] + cab + ["\\midrule"]
    for j, b in enumerate(bloques):
        out += [" & ".join(f) + " \\\\" for f in b]
        if j < len(bloques) - 1:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def t_e3b(E3b: dict) -> str:
    """Error de los estimadores frente a las discrepancias del sensor, con la
    reduccion del cociente de medias, la mediana por corrida y las corridas
    ganadas."""
    bloques = []
    for b in _filas_e3b(E3b):
        filas = []
        for lab, F in b:
            fila = [lab]
            for var in ("V", "e"):
                c = F[var]["_kf_vs_ref"]
                fila += [sig(F[var]["kalman"]["rmse"]["media"]), sig(F[var]["regresion_local"]["rmse"]["media"]),
                         pct(c["reduccion"]), pct(c["red_mediana"]), pct(c["gana_filtro"], 0, False)]
            filas.append(fila)
        bloques.append(filas)
    cab = [tr("& \\multicolumn{5}{c}{Volumen $V$ [m$^3$]} & \\multicolumn{5}{c}{Tasa $e$ [m$^3$/s]} \\\\",
              "& \\multicolumn{5}{c}{Volume $V$ [m$^3$]} & \\multicolumn{5}{c}{Rate $e$ [m$^3$/s]} \\\\"),
           "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
           tr("Condición & KF & RL & Red. & Med. & Gana & KF & RL & Red. & Med. & Gana \\\\",
              "Condition & KF & LR & Red. & Med. & Wins & KF & LR & Red. & Med. & Wins \\\\")]
    return _con_reglas("@{}lrrrrrrrrrr@{}", cab, bloques)


def t_e3b_diag(E3b: dict) -> str:
    """Diagnosticos de consistencia frente a las discrepancias del sensor."""
    bloques = []
    for b in _filas_e3b(E3b):
        bloques.append([[lab, num(F["nis_medio"], 2), pct(F["nis_fuera_banda"], 1, False),
                         num(F["autocorr_innov"], 2), sig(F["nees_medio"], 3)] for lab, F in b])
    cab = [tr("& \\multicolumn{3}{c}{Calculable en planta} & Verdad \\\\",
              "& \\multicolumn{3}{c}{Computable in plant} & Truth \\\\"),
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
           tr("Condición & NIS & $>\\chi^2_{0,95}$ [\\%] & $r_1$ & NEES \\\\",
              "Condition & NIS & $>\\chi^2_{0.95}$ [\\%] & $r_1$ & NEES \\\\")]
    return _con_reglas("@{}lrrrr@{}", cab, bloques)


def t_e3b_sintonia(E3a: dict, E3b: dict) -> str:
    """El mismo cambio de q_e con ruido blanco (E3a) y con ruido coloreado
    (E3b, rho = 0,5 y 0,95): NIS, autocorrelacion de la innovacion y error."""
    bloques = []
    blanco = []
    factores = E3b["sintonia_coloreado"][0]["factor_qe"]
    for f in factores:
        F = E3a["filas"][E3a["factor_qe"].index(f)]
        blanco.append([tr("Blanco", "White"), _factor(f), num(F["nis_medio"], 3), num(F["autocorr_innov"], 3),
                       sig(F["V"]["kalman"]["rmse"]["media"]), sig(F["e"]["kalman"]["rmse"]["media"])])
    bloques.append(blanco)
    for d in E3b["sintonia_coloreado"]:
        rho_ = num(d["rho"], 2)
        bloques.append([[tr(f"Coloreado ($\\rho = {rho_}$)", f"Colored ($\\rho = {rho_}$)"), _factor(f),
                         num(n_, 3), num(r_, 3), sig(v_), sig(e_)]
                        for f, n_, r_, v_, e_ in zip(d["factor_qe"], d["nis"], d["r1"], d["rmse_V"], d["rmse_e"])])
    cab = [tr("Ruido de medición & Factor sobre $q_e$ & NIS & $r_1$ & RMSE $V$ [m$^3$] & RMSE $e$ [m$^3$/s] \\\\",
              "Measurement noise & Factor on $q_e$ & NIS & $r_1$ & RMSE $V$ [m$^3$] & RMSE $e$ [m$^3$/s] \\\\")]
    return _con_reglas("@{}lrrrrr@{}", cab, bloques)


def t_e4(E4: dict) -> str:
    nomb = {"antes": tr("Antes del cambio", "Before the change"),
            "transitorio_0_5min": tr("0--5 min después", "0--5 min after"),
            "recuperado_10_30min": tr("10--30 min después", "10--30 min after")}
    filas = []
    for trm, lab in nomb.items():
        k, r, c = E4["rmse_e"]["kalman"][trm], E4["rmse_e"]["regresion_local"][trm], E4["comparaciones_e"][trm]
        filas.append([lab, sig(k["media"]), sig(r["media"]),
                      f"{pct(c['reduccion'])} {ic(c['red_ic_inf'], c['red_ic_sup'])}", pct(c["gana_filtro"], 0, False)])
    cab = tr("Tramo & RMSE $e$ KF & RMSE $e$ RL & Reducción [IC 95\\,\\%] & Gana [\\%]",
             "Window & RMSE $e$ KF & RMSE $e$ LR & Reduction [95\\,\\% CI] & Wins [\\%]")
    return tabular("@{}lrrrr@{}", [cab], filas)


def t_e4_resumen(E4: dict) -> str:
    a, n, pt, ob = E4["asentamiento_s"], E4["nis"], E4["prediccion_teorica"], E4["absorbido_observado"]
    filas = [
        [tr("Tiempo de asentamiento de la mediana [s]", "Settling time of the median [s]"),
         num(a["kalman"], 0), num(a["regresion_local"], 0), "---"],
        [tr("Fracción del salto absorbida por el filtro a los 60 s [\\%]", "Fraction of the step absorbed by the KF at 60 s [\\%]"),
         pct(ob["60s"], 0, False), "---", pct(pt["absorbido_60s"], 0, False)],
        [tr("Fracción del salto absorbida por el filtro a los 120 s [\\%]", "Fraction of the step absorbed by the KF at 120 s [\\%]"),
         pct(ob["120s"], 0, False), "---", pct(pt["absorbido_120s"], 0, False)],
        [tr("NIS (media móvil de 60 s): nivel antes del cambio", "NIS (60 s moving mean): level before the change"),
         num(n["nis60_media_antes"], 3), "---", num(1.0, 3)],
        [tr("NIS (media móvil de 60 s): máximo después del cambio", "NIS (60 s moving mean): peak after the change"),
         num(n["nis60_media_pico"], 3), "---", num(pt["nis60_pico"], 3)],
        [tr("Instante de ese máximo [s después del cambio]", "Time of that peak [s after the change]"),
         num(n["t_nis60_media_pico_s"], 0), "---", num(pt["t_nis60_pico_s"], 0)],
        [tr("Corridas con alarma en los 5 min anteriores [\\%]", "Runs with an alarm in the 5 min before [\\%]"),
         pct(n["fraccion_corridas_con_alarma"]["antes_5min"], 0, False), "---",
         pct(n["fraccion_alarma_nula_300s"], 0, False)],
        [tr("Corridas con alarma en los 5 min posteriores [\\%]", "Runs with an alarm in the 5 min after [\\%]"),
         pct(n["fraccion_corridas_con_alarma"]["despues_0_5min"], 0, False), "---", "---"],
    ]
    return tabular("@{}lrrr@{}", [tr("Indicador & KF & RL & Teoría", "Indicator & KF & LR & Theory")], filas)


def t_paper_nominal(E2: dict, E2b: dict, E2c: dict) -> str:
    """Tabla compacta para el articulo."""
    rows = []
    for k in ("medicion", "media_movil", "regresion_local", "kalman"):
        rV = E2["V"][k]["rmse"]["media"]
        re_ = E2["e"]["dif_finitas" if k == "medicion" else k]["rmse"]["media"] if k != "media_movil" else None
        rows.append([nombre(k) + (" / FD" if k == "medicion" else ""), sig(rV), sig(re_) if re_ is not None else "---"])
    out = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Estimator & RMSE $V$ [m$^3$] & RMSE $e$ [m$^3$/s] \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\midrule", "\\multicolumn{3}{@{}l}{\\textit{KF vs.\\ LR: RMSE reduction [\\%] (runs won [\\%])}} \\\\"]
    cV, ce = E2["V"]["_kf_vs_ref"], E2["e"]["_kf_vs_ref"]
    out.append(f"All runs, ratio of means ($n=100$) & {pct(cV['reduccion'])} ({pct(cV['gana_filtro'], 0, False)}) & {pct(ce['reduccion'])} ({pct(ce['gana_filtro'], 0, False)}) \\\\")
    out.append(f"All runs, median per run & {pct(cV['red_mediana'])} & {pct(ce['red_mediana'])} \\\\")
    for etq, lab in (("sin_vaciado", "Runs without emptying"), ("con_vaciado", "Runs with emptying")):
        d = E2["por_vaciado"][etq]
        out.append(f"{lab} ($n={d['n_corridas']}$) & {pct(d['V']['kf_vs_rl']['reduccion'])} ({pct(d['V']['kf_vs_rl']['gana_filtro'], 0, False)}) & {pct(d['e']['kf_vs_rl']['reduccion'])} ({pct(d['e']['kf_vs_rl']['gana_filtro'], 0, False)}) \\\\")
    d = E2c["con_vaciado"]
    out.append(f"Same runs, saturation disabled$^\\ddagger$ & {pct(d['V']['kf_vs_rl']['reduccion'])} ({pct(d['V']['kf_vs_rl']['gana_filtro'], 0, False)}) & {pct(d['e']['kf_vs_rl']['reduccion'])} ({pct(d['e']['kf_vs_rl']['gana_filtro'], 0, False)}) \\\\")
    bV, be = E2b["V"]["todas"], E2b["e"]["todas"]
    out.append(f"KF tuned on calibration runs$^\\dagger$ & {pct(bV['reduccion'])} ({pct(bV['gana_filtro'], 0, False)}) & {pct(be['reduccion'])} ({pct(be['gana_filtro'], 0, False)}) \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def t_paper_mismatch(E3b: dict, E2: dict) -> str:
    def celdas(FV, Fe, F):
        cV, ce = FV["_kf_vs_ref"], Fe["_kf_vs_ref"]
        return [f"{pct(cV['reduccion'])} ({pct(cV['gana_filtro'], 0, False)})", pct(cV["red_mediana"]),
                f"{pct(ce['reduccion'])} ({pct(ce['gana_filtro'], 0, False)})",
                num(F["nis_medio"], 2), num(F["autocorr_innov"], 2), sig(F["nees_medio"], 3)]
    rows = [["Nominal"] + celdas(E2["V"], E2["e"], E2)]
    sel = (("rho", 0.9, "$\\rho=0.9$"), ("rho", 0.95, "$\\rho=0.95$"), ("factor_descarga", 5.0, "$f_d=5$"),
           ("deriva_total", 40.0, "$b_N=40$ m$^3$"))
    for clave, v, lab in sel:
        d = E3b[clave]
        F = d["filas"][d["valores"].index(v)]
        rows.append([lab] + celdas(F["V"], F["e"], F))
    out = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
           "& \\multicolumn{3}{c}{KF vs.\\ LR [\\%]} & \\multicolumn{2}{c}{In-plant} & Truth \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}",
           "Condition & $V$ (wins) & $V$ med. & $e$ (wins) & NIS & $r_1$ & NEES \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------- valores
def valores(P: dict, V: dict, R: dict, S: dict) -> str:
    """Macros LaTeX con cada numero que el texto cita. El documento los usa
    como \\val{clave}, de modo que tampoco los numeros del texto se transcriben
    a mano. Una clave inexistente aparece en rojo al compilar."""
    d: dict[str, str] = {}

    def red(pref, c):
        if c is None:
            return
        d[pref] = pct(c["reduccion"])
        d[pref + ".ic"] = ic(c["red_ic_inf"], c["red_ic_sup"])
        d[pref + ".gana"] = pct(c["gana_filtro"], 0, False)
        d[pref + ".p"] = pval(c["p_wilcoxon"])
        d[pref + ".med"] = pct(c["red_mediana"])
        d[pref + ".u"] = pct(abs(c["reduccion"]), 1, False)     # sin signo: "reduce en un X %"

    def regimenes(pref, F, var):
        red(pref + ".sin", F[var].get("_kf_vs_ref_sin_vaciado"))
        red(pref + ".con", F[var].get("_kf_vs_ref_con_vaciado"))

    Pr = R["proceso"]
    d["proc.camiones"] = sig(Pr["camiones_por_hora"]["media"], 3)
    d["proc.espera"] = num(Pr["espera_por_camion_s"]["media"], 0)
    d["proc.vacia"] = num(100 * Pr["fraccion_tiempo_vacia"]["media"], 1)
    d["proc.corridasvaciado"] = f"{Pr['corridas_con_vaciado']}"
    d["proc.Vmax"] = num(Pr["V_max"], 0)
    d["proc.Vmedio"] = num(Pr["V_medio"], 0)
    d["proc.emedia"] = num(Pr["e_media"], 3)

    E2 = R["E2"]
    for var in ("V", "e"):
        for k, r in E2[var].items():
            if k.startswith("_"):
                continue
            d[f"e2.{var}.{k}.rmse"] = sig(r["rmse"]["media"])
            d[f"e2.{var}.{k}.mae"] = sig(r["mae"]["media"])
        for k, c in E2["comparaciones"][var].items():
            red(f"e2.red.{var}.{k}", c)
    d["e2.nis"] = num(E2["nis_medio"], 3)
    d["e2.nisfuera"] = pct(E2["nis_fuera_banda"], 1, False)
    d["e2.nees"] = num(E2["nees_medio"], 1)
    d["e2.r1"] = num(E2["autocorr_innov"], 3)
    for w in ("mm", "rl_V", "rl_e"):
        d[f"e2.W.{w.replace('_', '')}"] = f"{E2['ventanas'][w]}"
    for etq in ("sin_vaciado", "con_vaciado"):
        e = etq.replace("_", "")
        dd = E2["por_vaciado"][etq]
        d[f"e2.{e}.n"] = f"{dd['n_corridas']}"
        for var in ("V", "e"):
            d[f"e2.{e}.{var}.kf"] = sig(dd[var]["kalman"]["media"])
            d[f"e2.{e}.{var}.rl"] = sig(dd[var]["regresion_local"]["media"])
            red(f"e2.{e}.red.{var}", dd[var]["kf_vs_rl"])
        d[f"e2.{e}.nees"] = num(dd["nees_medio"], 2 if dd["nees_medio"] < 10 else 1)
        d[f"e2.{e}.nis"] = num(dd["nis_medio"], 3)
    for var in ("V", "e"):
        red(f"e2.global.{var}", E2[var]["_kf_vs_ref"])
    for f in E2["por_duracion_vaciado"]:
        etq = {"0": "b0", "1-29": "b1", "30-119": "b30", ">=120": "b120"}[f["pasos"]]
        d[f"e2.dur.{etq}.n"] = f"{f['n_corridas']}"
        for var in ("V", "e"):
            d[f"e2.dur.{etq}.{var}"] = pct(f[var]["reduccion"])
            d[f"e2.dur.{etq}.{var}.gana"] = pct(f[var]["gana_filtro"], 0, False)
    for qq, dq in E2["sensibilidad_quemado"].items():
        for var in ("V", "e"):
            for reg_ in ("todas", "sin_vaciado", "con_vaciado"):
                d[f"e2.quemado.{qq}.{var}.{reg_.replace('_', '')}"] = pct(dq[var][reg_])
    d["e2.convK"] = f"{E2['pasos_convergencia_ganancia']}"
    d["e2.K60.1"] = pct(E2["ganancia_rel_en_60"][0], 0, False)
    d["e2.K60.2"] = pct(E2["ganancia_rel_en_60"][1], 0, False)
    E2b = R["E2b"]
    d["e2b.fV"] = _factor(E2b["factor_V"]); d["e2b.fe"] = _factor(E2b["factor_e"])
    d["e2b.calvac"] = f"{E2b['corridas_calibracion_con_vaciado']}"
    d["e2b.calvacpct"] = num(100 * E2b["corridas_calibracion_con_vaciado"] / P["experimento"]["M_calibracion"], 0)
    d["proc.corridasvaciadopct"] = num(100 * Pr["corridas_con_vaciado"] / P["experimento"]["M"], 0)
    for var in ("V", "e"):
        red(f"e2b.{var}", E2b[var]["todas"])
        red(f"e2b.{var}.sin", E2b[var]["sin_vaciado"])
        red(f"e2b.{var}.con", E2b[var]["con_vaciado"])
    E2c = R["E2c"]
    d["e2c.Vmin"] = num(E2c["V_min_verdad"], 0)
    d["e2c.udistinta"] = f"{E2c['corridas_con_vaciado_u_distinta']}"
    d["e2c.espera.proc"] = num(E2c["espera_por_camion_con_vaciado_s"]["proceso"], 0)
    d["e2c.espera.abl"] = num(E2c["espera_por_camion_con_vaciado_s"]["ablacion"], 0)
    for etq in ("todas", "sin_vaciado", "con_vaciado"):
        e = etq.replace("_", "")
        dd = E2c[etq]
        d[f"e2c.{e}.n"] = f"{dd['n_corridas']}"
        for var in ("V", "e"):
            d[f"e2c.{e}.{var}.kf"] = sig(dd[var]["kalman"]["media"])
            d[f"e2c.{e}.{var}.rl"] = sig(dd[var]["regresion_local"]["media"])
            red(f"e2c.{e}.red.{var}", dd[var]["kf_vs_rl"])
        d[f"e2c.{e}.nees"] = num(dd["nees_medio"], 2)
        d[f"e2c.{e}.nis"] = num(dd["nis_medio"], 3)
    d["e2.Vmin"] = num(E2["V_estimado_min"], 1)
    d["e2.Vneg"] = num(100 * E2["fraccion_V_estimado_negativo"], 2)
    K = E2["K_estacionaria"]; sd = E2["P_estacionaria_de"]; rg = E2["regimen"]
    d["reg.K1"] = num(K[0], 4); d["reg.K2"] = num(K[1], 6)
    d["reg.sdV"] = num(sd[0], 2); d["reg.sde"] = num(sd[1], 4)
    d["reg.lamre"] = num(rg["autovalores_re"][0], 4); d["reg.lamim"] = num(abs(rg["autovalores_im"][0]), 4)
    d["reg.modulo"] = num(rg["modulo"], 4); d["reg.tau"] = num(rg["constante_tiempo_s"], 1)
    d["reg.wn"] = num(rg["wn_rad_s"], 5); d["reg.wnaprox"] = num(rg["wn_aprox_continuo"], 5)
    d["reg.zeta"] = num(rg["zeta"], 4); d["reg.periodo"] = num(rg["periodo_amortiguado_s"], 0)
    d["reg.S"] = num(rg["S_estacionaria"], 1)
    d["reg.difwn"] = cient(abs(rg["wn_rad_s"] - rg["wn_aprox_continuo"]), 0)

    E1 = R["E1"]
    for s_, W, F in zip(E1["sigma_v"], E1["ventanas"], E1["filas"]):
        k = f"{int(s_)}"
        for var in ("V", "e"):
            red(f"e1.{k}.red.{var}", F[var]["_kf_vs_ref"])
            regimenes(f"e1.{k}.red.{var}", F, var)
        d[f"e1.{k}.nis"] = num(F["nis_medio"], 3)
        d[f"e1.{k}.nees"] = num(F["nees_medio"], 1)
        d[f"e1.{k}.WV"] = f"{W['rl_V']}"; d[f"e1.{k}.We"] = f"{W['rl_e']}"; d[f"e1.{k}.Wmm"] = f"{W['mm']}"
        d[f"e1.{k}.V.mm"] = sig(F["V"]["media_movil"]["rmse"]["media"])
        d[f"e1.{k}.V.med"] = sig(F["V"]["medicion"]["rmse"]["media"])

    E3a = R["E3a"]
    for f, F in zip(E3a["factor_qe"], E3a["filas"]):
        k = f"{f:g}".replace(".", "p")
        d[f"e3a.{k}.V"] = sig(F["V"]["kalman"]["rmse"]["media"])
        d[f"e3a.{k}.e"] = sig(F["e"]["kalman"]["rmse"]["media"])
        red(f"e3a.{k}.red.V", F["V"]["_kf_vs_ref"]); red(f"e3a.{k}.red.e", F["e"]["_kf_vs_ref"])
        regimenes(f"e3a.{k}.red.V", F, "V"); regimenes(f"e3a.{k}.red.e", F, "e")
        d[f"e3a.{k}.nis"] = num(F["nis_medio"], 3)
        d[f"e3a.{k}.nees"] = sig(F["nees_medio"], 3)
        d[f"e3a.{k}.r1"] = num(F["autocorr_innov"], 3)
    for c, F in zip(E3a["escala_comun"], E3a["filas_escala"]):
        k = f"{c:g}".replace(".", "p")
        d[f"e3a.c{k}.nis"] = sig(F["nis_medio"], 4)
        d[f"e3a.c{k}.nees"] = sig(F["nees_medio"], 4)
        d[f"e3a.c{k}.V"] = sig(F["V"]["kalman"]["rmse"]["media"], 4)

    E3b = R["E3b"]
    for clave, abrev in (("rho", "rho"), ("factor_descarga", "fd"), ("deriva_total", "b")):
        for v, F in zip(E3b[clave]["valores"], E3b[clave]["filas"]):
            k = f"{abrev}{v:g}".replace(".", "p")
            for var in ("V", "e"):
                red(f"e3b.{k}.red.{var}", F[var]["_kf_vs_ref"])
                regimenes(f"e3b.{k}.red.{var}", F, var)
                d[f"e3b.{k}.{var}.kf"] = sig(F[var]["kalman"]["rmse"]["media"])
                d[f"e3b.{k}.{var}.rl"] = sig(F[var]["regresion_local"]["rmse"]["media"])
            d[f"e3b.{k}.nis"] = num(F["nis_medio"], 2)
            d[f"e3b.{k}.nisfuera"] = pct(F["nis_fuera_banda"], 1, False)
            d[f"e3b.{k}.r1"] = num(F["autocorr_innov"], 2)
            d[f"e3b.{k}.nees"] = sig(F["nees_medio"], 3)

    # Sintonia con ruido coloreado: prefijo e3b.rhot. para rho = 0,95 y
    # e3b.rhot50. para rho = 0,5.
    for ds in E3b["sintonia_coloreado"]:
        pref = "e3b.rhot." if abs(ds["rho"] - 0.95) < 1e-9 else "e3b.rhot50."
        if pref == "e3b.rhot.":
            d["e3b.rhot.rho"] = num(ds["rho"], 2)
        for f, n_, r_, v_, e_ in zip(ds["factor_qe"], ds["nis"], ds["r1"], ds["rmse_V"], ds["rmse_e"]):
            k = f"{f:g}".replace(".", "p")
            d[f"{pref}{k}.nis"] = num(n_, 3); d[f"{pref}{k}.r1"] = num(r_, 3)
            d[f"{pref}{k}.V"] = sig(v_); d[f"{pref}{k}.e"] = sig(e_)

    E4 = R["E4"]
    for tr_ in ("antes", "transitorio_0_5min", "recuperado_10_30min"):
        k = tr_.split("_")[0]
        d[f"e4.{k}.kf"] = sig(E4["rmse_e"]["kalman"][tr_]["media"])
        d[f"e4.{k}.rl"] = sig(E4["rmse_e"]["regresion_local"][tr_]["media"])
        red(f"e4.{k}.red", E4["comparaciones_e"][tr_])
    d["e4.asent.kf"] = num(E4["asentamiento_s"]["kalman"], 0)
    d["e4.asent.rl"] = num(E4["asentamiento_s"]["regresion_local"], 0)
    n4 = E4["nis"]
    d["e4.nis.antes"] = num(n4["medio"]["antes_5min"], 3)
    d["e4.nis.despues"] = num(n4["medio"]["despues_0_5min"], 3)
    d["e4.nis.umbral"] = num(n4["umbral_detector"], 3)
    d["e4.alarma.antes"] = pct(n4["fraccion_corridas_con_alarma"]["antes_5min"], 0, False)
    d["e4.alarma.despues"] = pct(n4["fraccion_corridas_con_alarma"]["despues_0_5min"], 0, False)
    pt = E4["prediccion_teorica"]
    d["e4.pred.t"] = num(pt["t_pico_s"], 0); d["e4.pred.sesgo"] = num(pt["sesgo_pico_V_m3"], 2)
    d["e4.pred.nis"] = num(pt["nis_pico"], 3)
    d["e4.pred.nisdisc"] = num(pt["nis_pico_discreto"], 3); d["e4.pred.tdisc"] = num(pt["t_pico_discreto_s"], 0)
    d["e4.pred.nis60"] = num(pt["nis60_pico"], 3); d["e4.pred.t60"] = num(pt["t_nis60_pico_s"], 0)
    d["e4.pred.abs60"] = pct(pt["absorbido_60s"], 0, False); d["e4.pred.abs120"] = pct(pt["absorbido_120s"], 0, False)
    ob = E4["absorbido_observado"]
    d["e4.obs.abs60"] = pct(ob["60s"], 0, False); d["e4.obs.abs120"] = pct(ob["120s"], 0, False)
    d["e4.nis60.antes"] = num(n4["nis60_media_antes"], 3)
    d["e4.nis60.pico"] = num(n4["nis60_media_pico"], 3); d["e4.nis60.t"] = num(n4["t_nis60_media_pico_s"], 0)
    d["e4.nis60.subida"] = num(n4["nis60_media_pico"] - n4["nis60_media_antes"], 2)
    d["e4.pred.subida"] = num(pt["nis60_pico"] - 1.0, 2)
    d["e4.alarma.nula"] = pct(n4["fraccion_alarma_nula_300s"], 0, False)
    # Tras el cambio la extraccion media queda muy por debajo del suministro
    # medio de los camiones y la tolva se vacia mucho menos: el NIS por paso baja.
    d["e4.nis.recup"] = num(n4["medio"]["despues_10_30min"], 3)
    v4 = E4["vaciado"]
    d["e4.sat.antes"] = num(100 * v4["fraccion_pasos_saturados"]["antes_5min"], 2)
    d["e4.sat.recup"] = num(100 * v4["fraccion_pasos_saturados"]["despues_10_30min"], 2)
    d["e4.corrvac.recup"] = pct(v4["fraccion_corridas_con_vaciado"]["despues_10_30min"], 0, False)
    d["e4.corrvac.antes"] = pct(v4["fraccion_corridas_con_vaciado"]["antes_5min"], 0, False)
    d["e4.sinvac.n"] = f"{v4['corridas_sin_vaciado']}"
    d["e4.sinvac.nis"] = num(v4["nis_antes_5min_corridas_sin_vaciado"], 3)
    d["e4.pred.nis05"] = num(v4["nis_teorico_0_5min"], 3)
    d["e4.pred.sesgodisc"] = num(pt["sesgo_pico_discreto_m3"], 2)
    d["e4.pred.tsesgodisc"] = num(pt["t_sesgo_pico_discreto_s"], 0)
    d["e4.enom"] = num(P["proceso"]["e_nom"], 1)
    d["e4.nis60.subida.recup"] = num(n4["nis60_media_pico"] - n4["medio"]["despues_10_30min"], 2)
    pp4 = P["proceso"]
    d["e4.efin"] = num(pp4["e_nom"] * E4["factor"], 1)
    d["proc.suministro"] = num(pp4["carga_camion"] / pp4["intervalo_medio"], 3)
    # Pico observado de la mediana del NIS (media movil de 60 s) tras el escalon
    import numpy as _np
    s4 = S["serie_E4"]
    t4 = _np.array(s4["t"]); n = _np.array(s4["nis_mm_med"]); p90 = _np.array(s4["nis_mm_p90"])
    m = (t4 >= E4["t_escalon"]) & (t4 < E4["t_escalon"] + 300)
    i = int(_np.argmax(_np.where(m, n, -_np.inf)))
    d["e4.obs.nis"] = num(float(n[i]), 3); d["e4.obs.t"] = num(float(t4[i] - E4["t_escalon"]), 0)
    d["e4.obs.p90"] = num(float(p90[m].max()), 2)

    try:
        rend = json.loads((SALIDA / "rendimiento.json").read_text(encoding="utf-8"))
        d["rend.us"] = num(rend["us_por_paso_en_linea"], 0)
    except (OSError, KeyError):
        pass
    d["param.quemado"] = f"{P['experimento']['quemado']}"
    pe0 = P["estimador"]
    d["param.K0V"] = num(pe0["P0_V"] / (pe0["P0_V"] + pe0["R"]), 3)
    Vf = V
    d["ev.nees"] = num(Vf["D_nees"]["media_global"], 3); d["ev.neesbanda"] = pct(Vf["D_nees"]["fraccion_pasos_en_banda"], 1, False)
    d["ev.nis"] = num(Vf["E_nis"]["media_global"], 3); d["ev.nisbanda"] = pct(Vf["E_nis"]["fraccion_pasos_en_banda"], 1, False)
    d["ev.difK"] = cient(Vf["B_riccati"]["diferencia"], 1)
    cb = Vf.get("_criterio_bandas")
    if cb:
        d["ev.bandas.lotes"] = f"{cb['lotes']}"
        d["ev.bandas.nees.media"] = pct(cb["nees"]["fraccion_en_banda_media"], 1, False)
        d["ev.bandas.nees.min"] = pct(cb["nees"]["fraccion_en_banda_min"], 1, False)
        d["ev.bandas.nees.bajo"] = pct(cb["nees"]["lotes_bajo_90"], 0, False)
        d["ev.bandas.nis.bajo"] = pct(cb["nis"]["lotes_bajo_90"], 0, False)
        d["ev.bandas.nis.min"] = pct(cb["nis"]["fraccion_en_banda_min"], 1, False)

    # serie representativa (corrida 0)
    s0 = S["serie"]; V0 = _np.array(s0["V"])
    d["serie.vacia"] = f"{int(_np.sum(V0 <= 1e-9))}"
    d["serie.vaciapct"] = num(100 * float(_np.mean(V0 <= 1e-9)), 1)
    d["serie.Vkfmin"] = num(float(_np.min(_np.array(s0["V_kf"])[60:])), 0)

    # Cantidades derivadas de la aproximacion analitica (Capitulo 2): memoria
    # efectiva sqrt(2)/wn con wn = (q_e / (R dt^2))^(1/4).
    pe_, pp_ = P["estimador"], P["proceso"]
    def memoria(qe, Rv):
        return math.sqrt(2) / (qe / (Rv * pp_["dt"] ** 2)) ** 0.25
    d["teo.tau.s2"] = num(memoria(pe_["q_e"], 4.0), 0)
    d["teo.tau.s40"] = num(memoria(pe_["q_e"], 1600.0), 0)
    d["teo.tau.f3"] = num(memoria(3 * pe_["q_e"], pe_["R"]), 0)
    d["teo.tau.f10"] = num(memoria(10 * pe_["q_e"], pe_["R"]), 0)
    # Deriva lineal b_k = b_N k/N: equivale a un error constante de la tasa
    # b_N/(N dt). Si el filtro la sigue por completo, el error del volumen es la
    # propia rampa; su RMSE en la ventana evaluada [quemado, N) es el valor
    # esperado, y sumado en cuadratura con el error nominal da una referencia
    # para el RMSE observado.
    bN = max(R["E3b"]["deriva_total"]["valores"])
    Nn, qq_ = pp_["N"], P["experimento"]["quemado"]
    kk = _np.arange(qq_, Nn)
    rampa = math.sqrt(float(_np.mean((bN * kk / Nn) ** 2)))
    d["teo.beta"] = num(bN / (Nn * pp_["dt"]), 4)
    d["teo.rmsederiva.hora"] = num(bN / math.sqrt(3), 1)
    d["teo.rmsederiva"] = num(rampa, 1)
    d["teo.rmsederiva.tot"] = num(math.sqrt(rampa ** 2 + E2["V"]["kalman"]["rmse"]["media"] ** 2), 1)

    lineas = ["% Generado por tablas.py desde resultados.json y series.json. No editar."]
    lineas += [f"\\defval{{{k}}}{{{v}}}" for k, v in sorted(d.items())]
    return "\n".join(lineas) + "\n"


def main(salida: Path = SALIDA) -> int:
    global L
    reg = json.loads((salida / "resultados.json").read_text(encoding="utf-8"))
    P, V, R = reg["parametros"], reg["verificacion"], reg["resultados"]
    ser = json.loads((salida / "series.json").read_text(encoding="utf-8"))
    for idioma in ("es", "en"):
        L = idioma
        d = salida / f"tablas_{idioma}"
        d.mkdir(parents=True, exist_ok=True)
        tablas = {
            "tab_parametros": t_parametros(P),
            "tab_verificacion": t_verificacion(V),
            "tab_proceso": t_proceso(R["proceso"]),
            "tab_e2": t_e2(R["E2"]),
            "tab_e2_comparaciones": t_e2_comparaciones(R["E2"]),
            "tab_e2_vaciado": t_e2_vaciado(R["E2"], R["E2c"]),
            "tab_e1_V": t_e1(R["E1"], "V"),
            "tab_e1_e": t_e1(R["E1"], "e"),
            "tab_e3a": t_e3a(R["E3a"]),
            "tab_e3a_diag": t_e3a_diag(R["E3a"]),
            "tab_e2_gradiente": t_e2_gradiente(R["E2"]),
            "tab_e2b": t_e2b(R),
            "tab_quemado": t_quemado(R["E2"]),
            "tab_e3a_escala": t_e3a_escala(R["E3a"]),
            "tab_e3b": t_e3b(R["E3b"]),
            "tab_e3b_diag": t_e3b_diag(R["E3b"]),
            "tab_e3b_sintonia": t_e3b_sintonia(R["E3a"], R["E3b"]),
            "tab_e4": t_e4(R["E4"]),
            "tab_e4_resumen": t_e4_resumen(R["E4"]),
        }
        if idioma == "en":
            tablas["tab_paper_nominal"] = t_paper_nominal(R["E2"], R["E2b"], R["E2c"])
            tablas["tab_paper_mismatch"] = t_paper_mismatch(R["E3b"], R["E2"])
        tablas["valores"] = valores(P, V, R, ser)
        man = (salida / "manifiesto_md5.txt").read_text(encoding="utf-8")
        tablas["manifiesto"] = "\\begin{verbatim}\n" + man + "\\end{verbatim}\n"
        ent = json.loads((salida / "entorno.json").read_text(encoding="utf-8"))
        filas_ent = [["Python", ent["python"]], ["NumPy", ent["numpy"]], ["SciPy", ent["scipy"]],
                     ["Matplotlib", ent["matplotlib"]],
                     [tr("Plataforma", "Platform"), ent["plataforma"].replace("_", "\\_")],
                     [tr("Procesador", "Processor"), f"{ent['procesador']}, {ent.get('nucleos_logicos', '?')} "
                      + tr("núcleos lógicos", "logical cores")]]
        tablas["entorno"] = tabular("@{}ll@{}", [tr("Componente & Versión o descripción", "Component & Version or description")], filas_ent)
        for nombre_t, contenido in tablas.items():
            (d / f"{nombre_t}.tex").write_text(contenido, encoding="utf-8")
        if idioma == "en":
            # Copia con prefijo "en." para el abstract en ingles de la tesis
            pref = tablas["valores"].replace("\\defval{", "\\defval{en.")
            (salida / "tablas_es" / "valores_en.tex").write_text(pref, encoding="utf-8")
    print(f"Tablas escritas en {salida}/tablas_es y {salida}/tablas_en")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
