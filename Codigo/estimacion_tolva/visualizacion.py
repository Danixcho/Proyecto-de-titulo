"""
visualizacion.py  -  Modulo 5: figuras
======================================
Figuras de la tesis (espanol, coma decimal, ancho de pagina) y del articulo
(ingles, punto decimal, ancho de columna o de pagina IEEE). Todas se generan
desde los resultados guardados, sin volver a simular (ver figuras.py).

Codificacion fija de colores en todas las figuras:
  filtro de Kalman  azul      regresion local  naranja
  media movil       aguamarina, linea discontinua
  verdad            negro     medicion         gris
Los subconjuntos de corridas (todas, sin vaciado, con vaciado) se distinguen
ademas por el marcador. Paleta validada para daltonismo (dataviz).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


class _FormatoComa(ScalarFormatter):
    """Mismo formato que matplotlib, con coma decimal."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

C_KF = "#2a78d6"
C_RL = "#eb6834"
C_MM = "#1baf7a"
C_VERDAD = "#0b0b0b"
C_MED = "#898781"
C_SIN = "#1baf7a"
C_CON = "#6250d6"
C_TEO = "#0b0b0b"
C_GRIS = "#52514e"
C_REJILLA = "#e1e0d9"
C_DESCARGA = "#ecebe6"

TXT = {
    "es": {
        "t": "Tiempo [s]", "V": "Volumen [m³]", "e": "Extracción [m³/s]",
        "verdad": "Verdad (simulación)", "med": "Medición", "mm": "Media móvil",
        "rl": "Regresión local", "kf": "Filtro de Kalman", "kf_banda": "Filtro de Kalman (±2σ)",
        "descarga": "Descarga de camión", "vacia": "Tolva vacía:\nla extracción se satura",
        "sigma": "σ_v [m³]", "rmse_V": "RMSE de V [m³]", "rmse_e": "RMSE de e [m³/s]",
        "red_V": "Reducción frente a RL, V [%]", "red_e": "Reducción frente a RL, e [%]",
        "todas": "Todas (IC 95 %)", "sin": "Sin vaciado", "con": "Con vaciado",
        "factor": "Factor sobre q_e", "nis": "NIS promedio", "esperado": "Valor esperado",
        "nominal": "Nominal", "nees": "NEES promedio", "r1": "Autocorr. innov.",
        "rho": "Autocorrelación ρ", "fd": "Factor de ruido en descarga", "bN": "Deriva acumulada [m³]",
        "e_med": "Extracción [m³/s]\n(mediana)", "err_med": "Error (mediana)\n[m³/s]",
        "nis60": "NIS\n(media 60 s)", "cambio": "Cambio de dureza", "banda10": "±10 % del salto",
        "teo": "Teoría (régimen)", "p90": "Percentil 90", "media": "Media", "umbral": "Umbral de alarma",
        "paso": "Paso k", "banda95": "Banda χ² del 95 %",
    },
    "en": {
        "t": "Time [s]", "V": "Volume [m³]", "e": "Extraction [m³/s]",
        "verdad": "Ground truth (simulation)", "med": "Measurement", "mm": "Moving average",
        "rl": "Local regression", "kf": "Kalman filter", "kf_banda": "Kalman filter (±2σ)",
        "descarga": "Truck dumping", "vacia": "Empty hopper:\nextraction saturates",
        "sigma": "σ_v [m³]", "rmse_V": "RMSE of V [m³]", "rmse_e": "RMSE of e [m³/s]",
        "red_V": "Reduction vs. LR, V [%]", "red_e": "Reduction vs. LR, e [%]",
        "todas": "All (95 % CI)", "sin": "No emptying", "con": "With emptying",
        "factor": "Factor on q_e", "nis": "Average NIS", "esperado": "Expected value",
        "nominal": "Nominal", "nees": "Average NEES", "r1": "Innov. autocorr.",
        "rho": "Autocorrelation ρ", "fd": "Noise factor while dumping", "bN": "Accumulated drift [m³]",
        "e_med": "Extraction [m³/s]\n(median)", "err_med": "Error (median)\n[m³/s]",
        "nis60": "NIS\n(60 s mean)", "cambio": "Hardness change", "banda10": "±10 % of step",
        "teo": "Theory (steady state)", "p90": "90th percentile", "media": "Mean", "umbral": "Alarm threshold",
        "paso": "Step k", "banda95": "95 % χ² band",
    },
}

ANCHO = {"es": 6.3, "en": 7.16}       # pulgadas: pagina de la tesis / pagina IEEE
COLUMNA_EN = 3.5


def _estilo(idioma: str):
    plt.rcParams.update({
        "font.size": 8.5 if idioma == "es" else 7.5,
        "axes.titlesize": 8.5 if idioma == "es" else 7.5,
        "axes.labelsize": 8.5 if idioma == "es" else 7.5,
        "legend.fontsize": 7.5 if idioma == "es" else 6.5,
        "xtick.labelsize": 7.5 if idioma == "es" else 6.5,
        "ytick.labelsize": 7.5 if idioma == "es" else 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#898781", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
        "axes.grid": True, "grid.color": C_REJILLA, "grid.linewidth": 0.5,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": False, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def _coma(ax, idioma: str, ejes: str = "xy"):
    """Coma decimal en los ejes lineales de las figuras en espanol."""
    if idioma != "es":
        return
    for eje in ejes:
        a = ax.xaxis if eje == "x" else ax.yaxis
        if a.get_scale() == "linear":
            a.set_major_formatter(_FormatoComa(useOffset=False))


def _guardar(fig, salida: Path, nombre: str):
    salida.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida / f"{nombre}.pdf")
    fig.savefig(salida / f"{nombre}.png", dpi=200)
    plt.close(fig)


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


# ------------------------------------------------------------------ figuras
def fig_serie(S: dict, idioma: str, salida: Path, ancho: float):
    T = TXT[idioma]
    s = S["serie"]
    t = np.array(s["t"])
    V = np.array(s["V"])
    vacia = np.flatnonzero(V <= 1e-9)
    # Ventana de 30 minutos que contiene el primer vaciado (o desde el quemado).
    ini = 300.0 if vacia.size == 0 else max(300.0, t[vacia[0]] - 600.0)
    ini = min(ini, t[-1] - 1800.0)
    sel = (t >= ini) & (t < ini + 1800.0)
    tt = t[sel]
    D = np.array(s["descargando"])[sel].astype(bool)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ancho, 0.48 * ancho), sharex=True)
    for ax in (a1, a2):
        # franjas de descarga
        bordes = np.flatnonzero(np.diff(np.r_[0, D.astype(int), 0]))
        for i0, i1 in zip(bordes[::2], bordes[1::2]):
            ax.axvspan(tt[i0], tt[min(i1, len(tt) - 1)], color=C_DESCARGA, lw=0, zorder=0)
    a1.plot(tt, np.array(s["y"])[sel], ".", ms=1.6, color=C_MED, label=T["med"], zorder=1)
    a1.plot(tt, np.array(s["V_mm"])[sel], "--", color=C_MM, lw=1.0, label=T["mm"], zorder=2)
    a1.plot(tt, np.array(s["V_rl"])[sel], color=C_RL, lw=1.0, label=T["rl"], zorder=3)
    vk = np.array(s["V_kf"])[sel]; dv = np.array(s["de_V"])[sel]
    a1.fill_between(tt, vk - 2 * dv, vk + 2 * dv, color=C_KF, alpha=0.18, lw=0, zorder=2)
    a1.plot(tt, vk, color=C_KF, lw=1.2, label=T["kf"], zorder=4)
    a1.plot(tt, V[sel], color=C_VERDAD, lw=1.0, label=T["verdad"], zorder=5)
    a1.set_ylabel(T["V"])
    a1.fill_between([], [], color=C_DESCARGA, label=T["descarga"])
    h, l = a1.get_legend_handles_labels()
    orden = [l.index(x) for x in (T["descarga"], T["med"], T["mm"], T["rl"], T["kf"], T["verdad"])]
    a1.legend([h[i] for i in orden], [l[i] for i in orden], ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.32), handlelength=1.8)

    ek = np.array(s["e_kf"])[sel]; de = np.array(s["de_e"])[sel]
    a2.fill_between(tt, ek - 2 * de, ek + 2 * de, color=C_KF, alpha=0.18, lw=0)
    a2.plot(tt, np.array(s["e_rl"])[sel], color=C_RL, lw=1.0, label=T["rl"])
    a2.plot(tt, ek, color=C_KF, lw=1.2, label=T["kf_banda"])
    a2.plot(tt, np.array(s["e"])[sel], color=C_VERDAD, lw=1.0, label=T["verdad"])
    a2.set_ylabel(T["e"]); a2.set_xlabel(T["t"])
    a2.legend(ncol=3, loc="lower right")
    vs = np.flatnonzero((V[sel] <= 1e-9))
    if vs.size:
        a2.annotate(T["vacia"], xy=(tt[vs[0]] + 5, 0.02), xytext=(tt[vs[0]] + 60, 0.35),
                    fontsize=plt.rcParams["legend.fontsize"], color=C_GRIS,
                    arrowprops=dict(arrowstyle="->", color=C_GRIS, lw=0.7))
    a1.set_xlim(tt[0], tt[-1])
    for ax in (a1, a2):
        _coma(ax, idioma)
    _guardar(fig, salida, "fig_serie_representativa")


def _red(F: dict, var: str, clave: str):
    c = F[var].get(clave)
    return np.nan if c is None else 100 * c["reduccion"]


def fig_e1(R: dict, idioma: str, salida: Path, ancho: float):
    T = TXT[idioma]
    E1 = R["E1"]
    s = np.array(E1["sigma_v"])
    F = E1["filas"]
    fig, ax = plt.subplots(1, 3, figsize=(ancho, 0.34 * ancho))
    for nombre, col, est, lab in (("medicion", C_MED, ":", T["med"]), ("media_movil", C_MM, "--", T["mm"]),
                                  ("regresion_local", C_RL, "-", T["rl"]), ("kalman", C_KF, "-", T["kf"])):
        ax[0].plot(s, [f["V"][nombre]["rmse"]["media"] for f in F], est, marker="o", color=col, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks(s); ax[0].set_xticklabels([f"{v:g}" for v in s])
    ax[0].set_xlabel(T["sigma"]); ax[0].set_ylabel(T["rmse_V"]); ax[0].legend(loc="upper left")
    for j, var in ((1, "V"), (2, "e")):
        r = [100 * f[var]["_kf_vs_ref"]["reduccion"] for f in F]
        lo = [100 * f[var]["_kf_vs_ref"]["red_ic_inf"] for f in F]
        hi = [100 * f[var]["_kf_vs_ref"]["red_ic_sup"] for f in F]
        ax[j].fill_between(s, lo, hi, color=C_GRIS, alpha=0.15, lw=0)
        ax[j].plot(s, r, "-", marker="o", color=C_VERDAD, label=T["todas"])
        ax[j].plot(s, [_red(f, var, "_kf_vs_ref_sin_vaciado") for f in F], "--", marker="s", color=C_SIN, label=T["sin"])
        ax[j].plot(s, [_red(f, var, "_kf_vs_ref_con_vaciado") for f in F], ":", marker="^", color=C_CON, label=T["con"])
        ax[j].axhline(0, color=C_GRIS, lw=0.7)
        ax[j].set_xscale("log"); ax[j].set_xticks(s); ax[j].set_xticklabels([f"{v:g}" for v in s])
        ax[j].set_xlabel(T["sigma"]); ax[j].set_ylabel(T["red_V"] if var == "V" else T["red_e"])
        _coma(ax[j], idioma, "y")
    ax[1].legend(loc="lower right")
    _guardar(fig, salida, "fig_e1_ruido")


def fig_e3a(R: dict, idioma: str, salida: Path, ancho: float):
    T = TXT[idioma]
    E3a = R["E3a"]
    f = np.array(E3a["factor_qe"])
    F = E3a["filas"]
    fig, ax = plt.subplots(1, 3, figsize=(ancho, 0.3 * ancho))
    for j, (var, lab) in enumerate((("V", T["rmse_V"]), ("e", T["rmse_e"]))):
        ax[j].plot(f, [x[var]["kalman"]["rmse"]["media"] for x in F], "-o", color=C_KF, label=T["kf"])
        ax[j].axhline(F[0][var]["regresion_local"]["rmse"]["media"], color=C_RL, lw=1.1, label=T["rl"])
        ax[j].axvline(1.0, color=C_GRIS, ls=":", lw=0.8, label=T["nominal"])
        ax[j].set_xscale("log"); ax[j].set_xlabel(T["factor"]); ax[j].set_ylabel(lab)
        _coma(ax[j], idioma, "y")
    ax[0].legend(loc="upper right")
    ax[2].plot(f, [x["nis_medio"] for x in F], "-o", color=C_KF, label=T["kf"])
    ax[2].axhline(1.0, color=C_VERDAD, ls="--", lw=0.9, label=T["esperado"])
    ax[2].axvline(1.0, color=C_GRIS, ls=":", lw=0.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel(T["factor"]); ax[2].set_ylabel(T["nis"])
    ax[2].legend(loc="upper right")
    _coma(ax[2], idioma, "y")
    _guardar(fig, salida, "fig_e3a_sintonia")


def fig_e3b(R: dict, idioma: str, salida: Path, ancho: float):
    T = TXT[idioma]
    E3b = R["E3b"]
    claves = (("rho", T["rho"]), ("factor_descarga", T["fd"]), ("deriva_total", T["bN"]))
    # Error de los estimadores
    fig, ax = plt.subplots(2, 3, figsize=(ancho, 0.56 * ancho))
    for j, (cl, xlab) in enumerate(claves):
        v = np.array(E3b[cl]["valores"]); F = E3b[cl]["filas"]
        ax[0, j].plot(v, [x["V"]["media_movil"]["rmse"]["media"] for x in F], "--o", color=C_MM, label=T["mm"])
        for i, var in enumerate(("V", "e")):
            ax[i, j].plot(v, [x[var]["regresion_local"]["rmse"]["media"] for x in F], "-o", color=C_RL, label=T["rl"])
            ax[i, j].plot(v, [x[var]["kalman"]["rmse"]["media"] for x in F], "-o", color=C_KF, label=T["kf"])
            _coma(ax[i, j], idioma)
        ax[1, j].set_xlabel(xlab)
    ax[0, 0].set_ylabel(T["rmse_V"]); ax[1, 0].set_ylabel(T["rmse_e"])
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3)
    _guardar(fig, salida, "fig_e3b_estructura")

    # Diagnosticos de consistencia
    fig, ax = plt.subplots(3, 3, figsize=(ancho, 0.72 * ancho), sharey="row")
    filas = (("nees_medio", T["nees"], 2.0), ("nis_medio", T["nis"], 1.0), ("autocorr_innov", T["r1"], 0.0))
    for j, (cl, xlab) in enumerate(claves):
        v = np.array(E3b[cl]["valores"]); F = E3b[cl]["filas"]
        for i, (k, ylab, ref) in enumerate(filas):
            ax[i, j].plot(v, [x[k] for x in F], "-o", color=C_KF, label=T["kf"])
            ax[i, j].axhline(ref, color=C_VERDAD, ls="--", lw=0.8, label=T["esperado"])
            if i == 0:
                ax[i, j].set_yscale("log")
            else:
                _coma(ax[i, j], idioma, "y")
            _coma(ax[i, j], idioma, "x")
            if j == 0:
                ax[i, j].set_ylabel(ylab)
        ax[2, j].set_xlabel(xlab)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2)
    _guardar(fig, salida, "fig_e3b_diagnosticos")


def fig_e4(S: dict, idioma: str, salida: Path, ancho: float, alto: float):
    T = TXT[idioma]
    s = S["serie_E4"]
    t = np.array(s["t"]); te = s["t_escalon"]; salto = s["salto"]
    fig, ax = plt.subplots(3, 1, figsize=(ancho, alto), sharex=True)
    ax[0].plot(t, s["rl_med"], color=C_RL, lw=1.1, label=T["rl"])
    ax[0].plot(t, s["kf_med"], color=C_KF, lw=1.2, label=T["kf"])
    ax[0].plot(t, s["e_med"], color=C_VERDAD, lw=1.0, label=T["verdad"])
    ax[0].set_ylabel(T["e_med"])
    ax[1].axhspan(-0.1 * salto, 0.1 * salto, color=C_GRIS, alpha=0.15, lw=0, label=T["banda10"])
    ax[1].plot(t, s["err_rl_med"], color=C_RL, lw=1.1, label=T["rl"])
    ax[1].plot(t, s["err_kf_med"], color=C_KF, lw=1.2, label=T["kf"])
    ax[1].plot(t, s["err_teo"], "--", color=C_TEO, lw=0.9, label=T["teo"])
    ax[1].set_ylabel(T["err_med"])
    ax[2].plot(t, s["nis_mm_p90"], ":", color=C_KF, lw=1.0, label=T["p90"])
    ax[2].plot(t, s["nis_mm_media"], color=C_KF, lw=1.2, label=T["media"])
    ax[2].plot(t, s["nis60_teo"], "--", color=C_TEO, lw=0.9, label=T["teo"])
    ax[2].axhline(s["umbral_nis"], color=C_GRIS, ls="-.", lw=0.8, label=T["umbral"])
    ax[2].set_ylabel(T["nis60"]); ax[2].set_xlabel(T["t"])
    for a in ax:
        a.axvline(te, color=C_GRIS, ls=":", lw=0.8)
        _coma(a, idioma)
    ax[0].plot([], [], ":", color=C_GRIS, lw=0.8, label=T["cambio"])
    for a in ax:
        a.legend(loc="upper right", ncol=2 if a is not ax[2] else 2)
    ax[0].set_xlim(t[0], t[-1])
    _guardar(fig, salida, "fig_e4_perturbacion")


def fig_verificacion(Sv: dict, idioma: str, salida: Path, ancho: float):
    T = TXT[idioma]
    k = np.array(Sv["k"])
    fig, ax = plt.subplots(1, 2, figsize=(ancho, 0.3 * ancho))
    for a, clave, banda, ref in ((ax[0], "nees", Sv["banda_nees"], 2.0), (ax[1], "nis", Sv["banda_nis"], 1.0)):
        a.axhspan(banda[0], banda[1], color=C_GRIS, alpha=0.15, lw=0, label=T["banda95"])
        a.plot(k, Sv[clave], color=C_KF, lw=0.8, label=T["kf"])
        a.axhline(ref, color=C_VERDAD, ls="--", lw=0.8, label=T["esperado"])
        a.set_xlabel(T["paso"]); a.set_ylabel(T["nees"] if clave == "nees" else T["nis"])
        _coma(a, idioma)
    ax[0].legend(loc="upper right", ncol=1)
    _guardar(fig, salida, "fig_verificacion")


def generar(R: dict, S: dict, raiz: Path):
    """Todas las figuras, en espanol (tesis) y en ingles (articulo)."""
    for idioma in ("es", "en"):
        _estilo(idioma)
        salida = raiz / f"figuras_{idioma}"
        w = ANCHO[idioma]
        fig_serie(S, idioma, salida, w)
        fig_e1(R, idioma, salida, w)
        fig_e3a(R, idioma, salida, w)
        fig_e3b(R, idioma, salida, w)
        if idioma == "es":
            fig_e4(S, idioma, salida, w, 0.72 * w)
        else:
            fig_e4(S, idioma, salida, COLUMNA_EN, 4.3)
        if "verificacion" in S:
            fig_verificacion(S["verificacion"], idioma, salida, w)
