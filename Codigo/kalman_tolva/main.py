"""
main.py
=======
Punto de entrada unico del experimento. Ejecuta el pipeline completo y produce
todos los artefactos de evidencia del proyecto:

    resultados/
        logs/serie_temporal_corrida01.csv    series temporales completas
        logs/metricas_montecarlo.csv         metricas de las M corridas
        logs/barrido_degradacion.csv         barrido de R real
        logs/malla_desajuste_QR.csv          malla de sensibilidad
        resultados.json                      resumen consolidado
        figuras/fig1..fig6 (.pdf y .png)     figuras del articulo

Uso:
    python main.py

Reproducibilidad
----------------
Todo el experimento queda determinado por la semilla maestra declarada en
ParametrosTolva.semilla (42). Ejecutar este archivo dos veces en cualquier
maquina produce byte a byte los mismos CSV y las mismas metricas.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from modelo_tolva import ParametrosTolva
from analisis import monte_carlo, consolidar
from sensibilidad import barrido_degradacion, malla_desajuste
import figuras

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "resultados"
LOGS = SALIDA / "logs"
FIGURAS = SALIDA / "figuras"


# --------------------------------------------------------------------------- #
# Serializacion de evidencia
# --------------------------------------------------------------------------- #
def escribir_serie_temporal(par, series, ruta: Path) -> None:
    """Exporta la serie temporal completa de la corrida de referencia."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    x_real, y, x_est, u = (series["x_real"], series["y_scada"],
                           series["x_est"], series["u"])
    Vdot_dif = series["Vdot_dif"]
    K = np.array(series["filtro"].historial_K)
    P = np.array(series["filtro"].historial_P)
    innov = np.array(series["filtro"].historial_innovacion)

    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "t_s",
            "u_m3s",
            "V_real_m3", "Vdot_real_m3s",
            "y_scada_m3",
            "V_est_m3", "Vdot_est_m3s",
            "Vdot_dif_finitas_m3s",
            "error_scada_m3", "error_kf_m3",
            "K1", "K2", "P11_m6", "P22_m6s2", "innovacion_m3",
        ])
        for k in range(par.N):
            w.writerow([
                k, round(k * par.dt, 4),
                round(float(u[k]), 6),
                round(float(x_real[k, 0]), 6), round(float(x_real[k, 1]), 6),
                round(float(y[k]), 6),
                round(float(x_est[k, 0]), 6), round(float(x_est[k, 1]), 6),
                round(float(Vdot_dif[k]), 6),
                round(float(x_real[k, 0] - y[k]), 6),
                round(float(x_real[k, 0] - x_est[k, 0]), 6),
                round(float(K[k, 0, 0]), 8), round(float(K[k, 1, 0]), 8),
                round(float(P[k, 0, 0]), 8), round(float(P[k, 1, 1]), 8),
                round(float(innov[k]), 6),
            ])


def escribir_metricas(arr, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(arr.keys())
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corrida"] + campos)
        for i in range(len(arr[campos[0]])):
            w.writerow([i + 1] + [round(float(arr[c][i]), 8) for c in campos])


def escribir_barrido(barrido, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(barrido[0].keys()))
        w.writeheader()
        for fila in barrido:
            w.writerow({k: round(v, 8) for k, v in fila.items()})


def escribir_malla(malla, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    factores = malla["factores"]
    Z = malla["malla_rmse_m3"]
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha_Q \\ beta_R"] + [f"{b:g}" for b in factores])
        for i, a in enumerate(factores):
            w.writerow([f"{a:g}"] + [round(v, 6) for v in Z[i]])


# --------------------------------------------------------------------------- #
# Informe por consola
# --------------------------------------------------------------------------- #
def imprimir_resumen(par, resumen, barrido, malla) -> None:
    r = resumen
    iV, iD = r["inferencia_V"], r["inferencia_Vdot"]
    linea = "=" * 74

    print(linea)
    print("  RESULTADOS DEL MODULO DE ESTIMACION DE ESTADOS - CHANCADO PRIMARIO")
    print(linea)
    print(f"  Configuracion : N = {par.N} pasos | dt = {par.dt} s | "
          f"M = {par.M} corridas | semilla = {par.semilla}")
    print(f"  Q = diag({par.q_volumen}, {par.q_tasa})   R = {par.r_sensor} m^6   "
          f"sigma_v = {np.sqrt(par.r_sensor):.3f} m^3")
    print()

    print("  1. VOLUMEN DE MINERAL V (variable observada con ruido)")
    print(f"     MSE  telemetria SCADA cruda   : {r['mse_scada_V_medio']:9.4f} m^6"
          f"   (sd {r['mse_scada_V_desv']:.4f})")
    print(f"     MSE  modulo Filtro de Kalman  : {r['mse_kf_V_medio']:9.4f} m^6"
          f"   (sd {r['mse_kf_V_desv']:.4f})")
    print(f"     Reduccion del MSE             : {r['reduccion_mse_V_pct']:9.2f} %")
    print(f"     RMSE telemetria SCADA cruda   : {r['rmse_scada_V_medio']:9.4f} m^3")
    print(f"     RMSE modulo Filtro de Kalman  : {r['rmse_kf_V_medio']:9.4f} m^3")
    print(f"     Reduccion del RMSE            : {r['reduccion_rmse_V_pct']:9.2f} %")
    print()

    print("  2. TASA DE LLENADO Vdot (variable latente, no sensada)")
    print(f"     RMSE diferencias finitas      : {r['rmse_dif_finitas_medio']:9.4f} m^3/s")
    print(f"     RMSE modulo Filtro de Kalman  : {r['rmse_kf_Vdot_medio']:9.4f} m^3/s")
    print(f"     Reduccion del RMSE            : {r['reduccion_rmse_Vdot_pct']:9.2f} %")
    print()

    print("  3. INFERENCIA ESTADISTICA SOBRE EL RMSE DEL VOLUMEN [m^3]")
    print(f"     Prueba t pareada  : t({iV['gl']}) = {iV['t_estadistico']:.3f} , "
          f"p = {iV['p_unilateral']:.3e}  (unilateral, alpha = {iV['alpha']})")
    print(f"     Reduccion media   : {iV['media_diferencia']:.4f} m^3   "
          f"IC 95% = [{iV['ic95_inferior']:.4f} , {iV['ic95_superior']:.4f}] m^3")
    print(f"     Tamano del efecto : d de Cohen = {iV['cohen_d']:.2f}")
    print(f"     Normalidad (Shapiro-Wilk) : W = {iV['shapiro_W']:.4f} , "
          f"p = {iV['shapiro_p']:.4f} -> "
          f"{'no se rechaza' if iV['normalidad_no_rechazada'] else 'SE RECHAZA'}")
    print(f"     Contraste robusto (Wilcoxon) : W = {iV['wilcoxon_W']:.1f} , "
          f"p = {iV['wilcoxon_p']:.3e}")
    print(f"     Decision : {'se rechaza H0' if iV['rechaza_H0'] else 'no se rechaza H0'}")
    print()

    print("  4. INFERENCIA SOBRE LA TASA LATENTE [m^3/s]")
    print(f"     t({iD['gl']}) = {iD['t_estadistico']:.3f} , p = {iD['p_unilateral']:.3e} , "
          f"IC 95% = [{iD['ic95_inferior']:.4f} , {iD['ic95_superior']:.4f}] m^3/s")
    print()

    print("  5. SENSIBILIDAD A LA DEGRADACION DEL SENSOR")
    print(f"     {'R [m^6]':>9} {'sigma [m^3]':>12} {'MSE SCADA':>12} "
          f"{'MSE KF':>10} {'red. MSE':>10}")
    for b in barrido:
        print(f"     {b['R']:9.0f} {b['sigma_sensor_m3']:12.2f} "
              f"{b['mse_scada_m6']:12.3f} {b['mse_kf_m6']:10.3f} "
              f"{b['reduccion_mse_pct']:9.2f} %")
    print()

    print("  6. ROBUSTEZ ANTE DESAJUSTE DE COVARIANZAS Q/R")
    print(f"     RMSE con sintonizacion nominal : "
          f"{malla['rmse_sintonizacion_perfecta_m3']:.4f} m^3")
    print(f"     RMSE de la telemetria cruda    : "
          f"{malla['rmse_scada_crudo_m3']:.4f} m^3")
    print(f"     Peor caso de la malla          : {malla['peor_caso_m3']:.4f} m^3")
    print(f"     Combinaciones peores que el crudo : "
          f"{malla['combinaciones_peores_que_crudo']} de {malla['combinaciones_totales']}")
    print(linea)


# --------------------------------------------------------------------------- #
def main() -> int:
    par = ParametrosTolva()

    print("\n[1/5] Verificando observabilidad del par (A, H) ...")
    if not par.es_observable():
        print("      ERROR: el sistema no es observable. Abortando.")
        return 1
    print("      Rango de la matriz de observabilidad = 2. Sistema observable.")

    print("[2/5] Ejecutando simulacion de Monte Carlo ...")
    filas, series = monte_carlo(par)
    resumen, arr = consolidar(filas)

    print("[3/5] Ejecutando analisis de sensibilidad ...")
    barrido = barrido_degradacion(par)
    malla = malla_desajuste(par)

    print("[4/5] Escribiendo registros de trazabilidad ...")
    escribir_serie_temporal(par, series, LOGS / "serie_temporal_corrida01.csv")
    escribir_metricas(arr, LOGS / "metricas_montecarlo.csv")
    escribir_barrido(barrido, LOGS / "barrido_degradacion.csv")
    escribir_malla(malla, LOGS / "malla_desajuste_QR.csv")

    entorno = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "plataforma": platform.platform(),
    }
    consolidado = {
        "parametros": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in asdict(par).items()},
        "entorno": entorno,
        "resultados": resumen,
        "sensibilidad_degradacion": barrido,
        "sensibilidad_desajuste_QR": malla,
    }
    (SALIDA / "resultados.json").write_text(
        json.dumps(consolidado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[5/5] Generando figuras ...")
    nombres = figuras.generar_todas(par, series, arr, resumen, barrido, malla, FIGURAS)
    for n in nombres:
        print(f"      {n}.pdf / .png")

    print()
    imprimir_resumen(par, resumen, barrido, malla)
    print(f"\nArtefactos escritos en: {SALIDA}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
