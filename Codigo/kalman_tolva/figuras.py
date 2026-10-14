"""
figuras.py
==========
Generacion de las figuras del articulo cientifico y del informe final.

Criterios de diseno
-------------------
Las figuras se producen para reproduccion IEEE a dos columnas, por lo que se
dimensionan en pulgadas de ancho de columna (3.5 in simple, 7.16 in doble) y se
exportan en PDF vectorial para LaTeX y en PNG de 300 ppp para la presentacion.

La identidad de cada serie se codifica de forma redundante mediante color,
estilo de linea y grosor, de modo que las figuras siguen siendo legibles
impresas en escala de grises y para lectores con deficiencia de vision
cromatica. La paleta categorica fue verificada con un validador de separacion
perceptual: el par azul-naranja alcanza una diferencia CVD de 24.7 y de vision
normal de 33.6 en el espacio OKLab, holgadamente sobre los umbrales exigidos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from textos_figuras import t as _T, set_idioma  # noqa: F401

# --------------------------------------------------------------------------- #
# Paleta y estilo
# --------------------------------------------------------------------------- #
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

AZUL = "#2a78d6"      # slot categorico 1 - modulo Filtro de Kalman
NARANJA = "#eb6834"   # slot categorico 2 - telemetria SCADA cruda

# Rampa secuencial azul (magnitud continua) para el mapa de calor
RAMPA_AZUL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

ANCHO_SIMPLE = 3.5
ANCHO_DOBLE = 7.16


def _aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": EJE,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_TENUE,
        "ytick.color": TINTA_TENUE,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _limpiar(ax) -> None:
    """Retira los bordes superior y derecho: la rejilla ya delimita el area."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _guardar(fig, destino: Path, nombre: str) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino / f"{nombre}.pdf")
    fig.savefig(destino / f"{nombre}.png")
    plt.close(fig)


def _sombrear_camiones(ax, par) -> None:
    for i, (ini, fin) in enumerate(par.ventanas_camion):
        ax.axvspan(ini, fin, color=TINTA_PRIMARIA, alpha=0.05, lw=0,
                   label=_T("f1_camion") if i == 0 else None)


# --------------------------------------------------------------------------- #
# Figura 1 - Reconstruccion del volumen
# --------------------------------------------------------------------------- #
def fig_reconstruccion(par, series, destino: Path) -> None:
    _aplicar_estilo()
    t = np.arange(par.N) * par.dt
    V_real = series["x_real"][:, 0]
    y_scada = series["y_scada"]
    V_est = series["x_est"][:, 0]

    fig, ax = plt.subplots(figsize=(ANCHO_DOBLE, 3.0))
    _sombrear_camiones(ax, par)

    ax.plot(t, y_scada, color=NARANJA, lw=0.6, alpha=0.55,
            label=_T("f1_scada"))
    ax.plot(t, V_real, color=TINTA_PRIMARIA, lw=1.8, ls="--",
            label=_T("f1_real"))
    ax.plot(t, V_est, color=AZUL, lw=1.6, ls="-",
            label=_T("f1_kf"))

    ax.set_xlabel(_T("tiempo"))
    ax.set_ylabel(_T("f1_ylabel"))
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper left", ncol=2)
    _limpiar(ax)
    _guardar(fig, destino, "fig1_reconstruccion_volumen")


# --------------------------------------------------------------------------- #
# Figura 2 - Recuperacion de la variable latente
# --------------------------------------------------------------------------- #
def fig_tasa_latente(par, series, destino: Path) -> None:
    _aplicar_estilo()
    t = np.arange(par.N) * par.dt
    Vdot_real = series["x_real"][:, 1]
    Vdot_dif = series["Vdot_dif"]
    Vdot_est = series["x_est"][:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ANCHO_DOBLE, 3.6),
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1],
                                                "hspace": 0.18})

    # Panel superior: la linea base ingenua, en su escala real
    ax1.plot(t, Vdot_dif, color=NARANJA, lw=0.5, alpha=0.7,
             label=_T("f2_base"))
    ax1.plot(t, Vdot_real, color=TINTA_PRIMARIA, lw=1.6, ls="--",
             label=_T("f2_real"))
    ax1.set_ylabel(_T("f2_ylabel"))
    ax1.legend(loc="upper right", ncol=1, frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.88)
    _limpiar(ax1)

    # Panel inferior: la estimacion del modulo, en escala fisica
    ax2.plot(t, Vdot_real, color=TINTA_PRIMARIA, lw=1.6, ls="--",
             label=_T("f2_real"))
    ax2.plot(t, Vdot_est, color=AZUL, lw=1.4, ls="-",
             label=_T("f2_kf"))
    ax2.set_xlabel(_T("tiempo"))
    ax2.set_ylabel(_T("f2_ylabel"))
    ax2.set_xlim(0, t[-1])
    ax2.legend(loc="lower right", ncol=2, frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.88)
    _limpiar(ax2)

    _guardar(fig, destino, "fig2_tasa_latente")


# --------------------------------------------------------------------------- #
# Figura 3 - Convergencia asintotica (Riccati)
# --------------------------------------------------------------------------- #
def fig_convergencia(par, series, destino: Path) -> None:
    _aplicar_estilo()
    filtro = series["filtro"]
    P = np.array(filtro.historial_P)      # (N, 2, 2)
    K = np.array(filtro.historial_K)      # (N, 2, 1)
    t = np.arange(len(P)) * par.dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.4))

    # Desviacion estandar a posteriori de cada estado
    ax1.plot(t, np.sqrt(P[:, 0, 0]), color=AZUL, lw=1.6, ls="-",
             label=_T("f3_p11"))
    ax1.plot(t, np.sqrt(P[:, 1, 1]), color=NARANJA, lw=1.6, ls="-.",
             label=_T("f3_p22"))
    ax1.set_yscale("log")
    ax1.set_xlabel(_T("tiempo"))
    ax1.set_ylabel(_T("f3_ylabel_p"))
    ax1.set_xlim(0, t[-1])
    ax1.legend(loc="upper right")
    ax1.set_title(_T("f3_titulo_p"), color=TINTA_SECUNDARIA)
    _limpiar(ax1)

    ax2.plot(t, K[:, 0, 0], color=AZUL, lw=1.6, ls="-", label=_T("f3_k1"))
    ax2.plot(t, K[:, 1, 0], color=NARANJA, lw=1.6, ls="-.", label=_T("f3_k2"))
    ax2.set_xlabel(_T("tiempo"))
    ax2.set_ylabel(_T("f3_ylabel_k"))
    ax2.set_xlim(0, t[-1])
    ax2.legend(loc="upper right")
    ax2.set_title(_T("f3_titulo_k"), color=TINTA_SECUNDARIA)
    _limpiar(ax2)

    # Anotacion del regimen permanente
    k_est = K[-1, 0, 0]
    ax2.annotate(_T("f3_regimen", v=k_est),
                 xy=(t[-1] * 0.62, k_est), xytext=(t[-1] * 0.30, k_est + 0.12),
                 fontsize=6.5, color=TINTA_SECUNDARIA,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=TINTA_TENUE))

    _guardar(fig, destino, "fig3_convergencia")


# --------------------------------------------------------------------------- #
# Figura 4 - Monte Carlo e inferencia
# --------------------------------------------------------------------------- #
def fig_montecarlo(arr, resumen, destino: Path) -> None:
    _aplicar_estilo()
    rmse_s = arr["rmse_scada_V"]
    rmse_k = arr["rmse_kf_V"]
    d = rmse_s - rmse_k
    inf = resumen["inferencia_V"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.5))

    # Panel A: comparacion pareada corrida a corrida
    x = np.arange(1, len(rmse_s) + 1)
    ax1.vlines(x, rmse_k, rmse_s, color=TINTA_TENUE, lw=0.6, alpha=0.6)
    ax1.scatter(x, rmse_s, s=14, color=NARANJA, marker="o", zorder=3,
                edgecolor="white", linewidth=0.5, label=_T("scada_crudo"))
    ax1.scatter(x, rmse_k, s=14, color=AZUL, marker="s", zorder=3,
                edgecolor="white", linewidth=0.5, label=_T("filtro_kalman"))
    ax1.set_xlabel(_T("f4_xlabel"))
    ax1.set_ylabel(_T("rmse_volumen"))
    ax1.set_ylim(0, max(rmse_s) * 1.18)
    ax1.set_xlim(0, len(rmse_s) + 1)
    ax1.legend(loc="upper center", ncol=2)
    ax1.set_title(_T("f4_titulo_izq", m=len(rmse_s)), color=TINTA_SECUNDARIA)
    _limpiar(ax1)

    # Panel B: distribucion de las diferencias e intervalo de confianza
    conteos, _, _ = ax2.hist(d, bins=12, color=RAMPA_AZUL[3],
                             edgecolor="white", linewidth=0.8)
    ax2.axvline(inf["media_diferencia"], color=AZUL, lw=1.8,
                label=_T("f4_media", v=inf["media_diferencia"]))
    ax2.axvline(0.0, color=TINTA_PRIMARIA, lw=1.2, ls=":",
                label=_T("f4_h0"))

    tope = conteos.max() * 1.42
    ax2.set_xlabel(_T("f4_xlabel_der"))
    ax2.set_ylabel(_T("f4_ylabel_der"))
    ax2.set_xlim(-0.09, d.max() * 1.06)
    ax2.set_ylim(0, tope)
    ax2.legend(loc="upper left")
    ax2.set_title(_T("f4_titulo_der"), color=TINTA_SECUNDARIA)

    # El IC al 95% es mas estrecho que un pixel a esta escala: se rotula.
    ax2.annotate(
        _T("f4_anotacion", lo=inf["ic95_inferior"], hi=inf["ic95_superior"],
          gl=inf["gl"], t=inf["t_estadistico"]),
        xy=(inf["media_diferencia"], conteos.max()),
        xytext=(0.97, 0.62), textcoords="axes fraction",
        ha="right", va="top", fontsize=6.5, color=TINTA_SECUNDARIA,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=TINTA_TENUE,
                        connectionstyle="arc3,rad=0.2"))
    _limpiar(ax2)

    _guardar(fig, destino, "fig4_montecarlo")


# --------------------------------------------------------------------------- #
# Figura 5 - Barrido de degradacion
# --------------------------------------------------------------------------- #
def fig_degradacion(barrido, destino: Path) -> None:
    _aplicar_estilo()
    sigma = np.array([b["sigma_sensor_m3"] for b in barrido])
    rmse_s = np.array([b["rmse_scada_m3"] for b in barrido])
    rmse_k = np.array([b["rmse_kf_m3"] for b in barrido])
    red = np.array([b["reduccion_mse_pct"] for b in barrido])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.4))

    # Panel A: ambas magnitudes en la misma unidad, un solo eje
    ax1.plot(sigma, rmse_s, color=NARANJA, lw=1.6, ls="-", marker="o", ms=4,
             label=_T("scada_crudo"))
    ax1.plot(sigma, rmse_k, color=AZUL, lw=1.6, ls="-", marker="s", ms=4,
             label=_T("filtro_kalman"))
    ax1.set_xlabel(_T("f5_xlabel"))
    ax1.set_ylabel(_T("rmse_volumen"))
    ax1.legend(loc="upper left")
    ax1.set_title(_T("f5_titulo_izq"), color=TINTA_SECUNDARIA)
    _limpiar(ax1)

    # Panel B: aporte relativo del modulo
    ax2.plot(sigma, red, color=AZUL, lw=1.8, ls="-", marker="D", ms=4)
    for xs, ys, dx, ha in ((sigma[0], red[0], 10, "left"),
                           (sigma[-1], red[-1], -6, "right")):
        ax2.annotate(f"{ys:.1f}%", xy=(xs, ys), xytext=(dx, 8),
                     textcoords="offset points", ha=ha, fontsize=6.5,
                     color=TINTA_SECUNDARIA)
    ax2.set_xlabel(_T("f5_xlabel"))
    ax2.set_ylabel(_T("f5_ylabel_der"))
    ax2.set_ylim(min(red) - 4, 101)
    ax2.set_title(_T("f5_titulo_der"), color=TINTA_SECUNDARIA)
    _limpiar(ax2)

    _guardar(fig, destino, "fig5_degradacion")


# --------------------------------------------------------------------------- #
# Figura 6 - Sensibilidad al desajuste Q/R
# --------------------------------------------------------------------------- #
def fig_sensibilidad(malla, destino: Path) -> None:
    _aplicar_estilo()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    factores = malla["factores"]
    Z = np.array(malla["malla_rmse_m3"])
    crudo = malla["rmse_scada_crudo_m3"]

    cmap = LinearSegmentedColormap.from_list("azul_secuencial", RAMPA_AZUL)

    fig, ax = plt.subplots(figsize=(ANCHO_SIMPLE + 0.9, 3.1))
    im = ax.imshow(Z, cmap=cmap, norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
                   origin="upper", aspect="auto")

    ax.set_xticks(range(len(factores)),
                  [f"{b:g}" for b in factores])
    ax.set_yticks(range(len(factores)),
                  [f"{a:g}" for a in factores])
    ax.set_xlabel(_T("f6_xlabel"))
    ax.set_ylabel(_T("f6_ylabel"))
    ax.grid(False)

    # Etiquetas directas: el valor de cada celda, con tinta legible sobre el fondo
    # Tinta blanca solo sobre las celdas realmente oscuras del extremo superior
    # de la rampa; el resto conserva tinta primaria para no perder contraste.
    lz = np.log(Z)
    umbral = lz.min() + 0.72 * (lz.max() - lz.min())
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            claro = lz[i, j] > umbral
            peor = Z[i, j] >= crudo
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.8,
                    color="white" if claro else TINTA_PRIMARIA,
                    fontweight="bold" if peor else "normal")

    # Marca de la sintonizacion perfecta
    k = factores.index(1.0)
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor=TINTA_PRIMARIA, lw=1.6))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(_T("f6_cbar"), fontsize=7.5,
                   color=TINTA_SECUNDARIA)
    cbar.ax.tick_params(labelsize=6.5, color=TINTA_TENUE)
    cbar.outline.set_edgecolor(EJE)

    ax.set_title(_T("f6_titulo"), color=TINTA_SECUNDARIA)

    leyenda = [
        Line2D([0], [0], color=TINTA_PRIMARIA, lw=1.6,
               label=_T("f6_nominal")),
        Patch(facecolor="none", edgecolor="none",
              label=_T("f6_negrita", v=crudo)),
    ]
    ax.legend(handles=leyenda, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=6.5, ncol=1)

    _guardar(fig, destino, "fig6_sensibilidad_QR")


def generar_todas(par, series, arr, resumen, barrido, malla, destino: Path,
                  idioma: str = "es") -> list[str]:
    """Genera el conjunto completo de figuras y devuelve sus nombres."""
    set_idioma(idioma)
    fig_reconstruccion(par, series, destino)
    fig_tasa_latente(par, series, destino)
    fig_convergencia(par, series, destino)
    fig_montecarlo(arr, resumen, destino)
    fig_degradacion(barrido, destino)
    fig_sensibilidad(malla, destino)
    return ["fig1_reconstruccion_volumen", "fig2_tasa_latente", "fig3_convergencia",
            "fig4_montecarlo", "fig5_degradacion", "fig6_sensibilidad_QR"]
