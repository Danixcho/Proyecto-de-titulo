"""
textos_figuras.py
=================
Cadenas de rotulacion de las figuras en espanol e ingles.

El informe final ABP se redacta en espanol y el articulo cientifico en formato
IEEE se redacta en ingles. Ambos productos comparten exactamente los mismos
datos y el mismo codigo de graficacion; solo cambia el idioma de los rotulos.
Centralizar las cadenas en este modulo evita que las dos versiones de una misma
figura diverjan en contenido.
"""

from __future__ import annotations

TEXTOS = {
    # ---------------------------------------------------------------- comunes
    "tiempo": {
        "es": "Tiempo [s]",
        "en": "Time [s]",
    },
    "scada_crudo": {
        "es": "SCADA crudo",
        "en": "Raw SCADA",
    },
    "filtro_kalman": {
        "es": "Filtro de Kalman",
        "en": "Kalman filter",
    },
    "rmse_volumen": {
        "es": "RMSE del volumen [m$^3$]",
        "en": "Volume RMSE [m$^3$]",
    },

    # ------------------------------------------------------------- figura 1
    "f1_ylabel": {
        "es": "Volumen de mineral $V$ [m$^3$]",
        "en": "Retained ore volume $V$ [m$^3$]",
    },
    "f1_camion": {
        "es": "Descarga de camión",
        "en": "Truck discharge",
    },
    "f1_scada": {
        "es": "Telemetría SCADA degradada",
        "en": "Degraded SCADA telemetry",
    },
    "f1_real": {
        "es": "Volumen real (verdad de referencia)",
        "en": "True volume (ground truth)",
    },
    "f1_kf": {
        "es": "Estimación del módulo (Filtro de Kalman)",
        "en": "Module estimate (Kalman filter)",
    },

    # ------------------------------------------------------------- figura 2
    "f2_ylabel": {
        "es": "$\\dot{V}$ [m$^3$/s]",
        "en": "$\\dot{V}$ [m$^3$/s]",
    },
    "f2_base": {
        "es": "Línea base: diferencias finitas sobre SCADA",
        "en": "Baseline: finite differences over raw SCADA",
    },
    "f2_real": {
        "es": "Tasa real $\\dot{V}$",
        "en": "True rate $\\dot{V}$",
    },
    "f2_kf": {
        "es": "Estimación del módulo $\\hat{\\dot{V}}$",
        "en": "Module estimate $\\hat{\\dot{V}}$",
    },

    # ------------------------------------------------------------- figura 3
    "f3_p11": {
        "es": "$\\sqrt{P_{11}}$  volumen [m$^3$]",
        "en": "$\\sqrt{P_{11}}$  volume [m$^3$]",
    },
    "f3_p22": {
        "es": "$\\sqrt{P_{22}}$  tasa [m$^3$/s]",
        "en": "$\\sqrt{P_{22}}$  rate [m$^3$/s]",
    },
    "f3_ylabel_p": {
        "es": "Incertidumbre a posteriori",
        "en": "Posterior uncertainty",
    },
    "f3_titulo_p": {
        "es": "Convergencia de la covarianza del error",
        "en": "Error covariance convergence",
    },
    "f3_k1": {
        "es": "$K_1$ (volumen)",
        "en": "$K_1$ (volume)",
    },
    "f3_k2": {
        "es": "$K_2$ (tasa)",
        "en": "$K_2$ (rate)",
    },
    "f3_ylabel_k": {
        "es": "Ganancia de Kalman $K_k$",
        "en": "Kalman gain $K_k$",
    },
    "f3_titulo_k": {
        "es": "Convergencia de la ganancia",
        "en": "Gain convergence",
    },
    "f3_regimen": {
        "es": "régimen permanente\n$K_1 \\to$ {v:.4f}",
        "en": "steady state\n$K_1 \\to$ {v:.4f}",
    },

    # ------------------------------------------------------------- figura 4
    "f4_xlabel": {
        "es": "Corrida de Monte Carlo",
        "en": "Monte Carlo run",
    },
    "f4_titulo_izq": {
        "es": "Comparación pareada ($M$ = {m})",
        "en": "Paired comparison ($M$ = {m})",
    },
    "f4_media": {
        "es": "media = {v:.3f} m$^3$",
        "en": "mean = {v:.3f} m$^3$",
    },
    "f4_h0": {
        "es": "$H_0:\\ \\mu_d = 0$",
        "en": "$H_0:\\ \\mu_d = 0$",
    },
    "f4_xlabel_der": {
        "es": "Reducción del RMSE por corrida $d_m$ [m$^3$]",
        "en": "Per-run RMSE reduction $d_m$ [m$^3$]",
    },
    "f4_ylabel_der": {
        "es": "Frecuencia",
        "en": "Frequency",
    },
    "f4_titulo_der": {
        "es": "Distribución de la reducción del error",
        "en": "Distribution of the error reduction",
    },
    "f4_anotacion": {
        "es": "IC 95\\% = [{lo:.3f}, {hi:.3f}] m$^3$\n$t$({gl}) = {t:.1f},  $p$ < 0.001",
        "en": "95\\% CI = [{lo:.3f}, {hi:.3f}] m$^3$\n$t$({gl}) = {t:.1f},  $p$ < 0.001",
    },

    # ------------------------------------------------------------- figura 5
    "f5_xlabel": {
        "es": "Desviación del ruido del sensor $\\sigma_v$ [m$^3$]",
        "en": "Sensor noise standard deviation $\\sigma_v$ [m$^3$]",
    },
    "f5_titulo_izq": {
        "es": "Error absoluto según degradación",
        "en": "Absolute error vs. degradation",
    },
    "f5_ylabel_der": {
        "es": "Reducción del MSE [\\%]",
        "en": "MSE reduction [\\%]",
    },
    "f5_titulo_der": {
        "es": "Aporte del módulo según degradación",
        "en": "Module contribution vs. degradation",
    },

    # ------------------------------------------------------------- figura 6
    "f6_xlabel": {
        "es": "Factor de desajuste en $R$  ($\\beta$)",
        "en": "Mismatch factor on $R$  ($\\beta$)",
    },
    "f6_ylabel": {
        "es": "Factor de desajuste en $Q$  ($\\alpha$)",
        "en": "Mismatch factor on $Q$  ($\\alpha$)",
    },
    "f6_cbar": {
        "es": "RMSE del volumen [m$^3$]",
        "en": "Volume RMSE [m$^3$]",
    },
    "f6_titulo": {
        "es": "Robustez ante desajuste de covarianzas",
        "en": "Robustness to covariance mismatch",
    },
    "f6_nominal": {
        "es": "Sintonización nominal ($\\alpha=\\beta=1$)",
        "en": "Nominal tuning ($\\alpha=\\beta=1$)",
    },
    "f6_negrita": {
        "es": "Negrita: peor que SCADA crudo ({v:.2f} m$^3$)",
        "en": "Bold: worse than raw SCADA ({v:.2f} m$^3$)",
    },
}

_IDIOMA = "es"


def set_idioma(codigo: str) -> None:
    """Fija el idioma activo de rotulacion ('es' o 'en')."""
    global _IDIOMA
    if codigo not in ("es", "en"):
        raise ValueError(f"Idioma no soportado: {codigo}")
    _IDIOMA = codigo


def t(clave: str, **kwargs) -> str:
    """Devuelve la cadena rotulada en el idioma activo, con formato opcional."""
    plantilla = TEXTOS[clave][_IDIOMA]
    return plantilla.format(**kwargs) if kwargs else plantilla
