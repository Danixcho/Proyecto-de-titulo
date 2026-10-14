"""
auditoria.py
============
Bateria de verificacion matematica del estimador.

No comprueba que el codigo "corra": comprueba que satisface las propiedades
que la teoria exige. Bajo la condicion actual del estudio —la planta que genera
los datos y el modelo que usa el filtro son el mismo— el Filtro de Kalman debe
cumplir cuatro propiedades verificables. Si alguna falla, hay un error de
implementacion, y ninguna cantidad de resultados bonitos lo compensa.

  A. Simetria y definicion positiva de P en todo instante (forma de Joseph).
  B. Convergencia de la ganancia al valor de regimen permanente que predice la
     ecuacion algebraica de Riccati, resuelta de forma independiente.
  C. Insesgamiento: el error medio de estimacion debe ser estadisticamente
     indistinguible de cero.
  D. Consistencia: el NEES promedio debe caer dentro de las bandas chi-cuadrado.
     Es la prueba mas exigente, porque verifica que la covarianza declarada por
     el filtro corresponde al error que realmente comete.

Uso:  python3 auditoria.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from modelo_tolva import ParametrosTolva, generar_escenario, estimar
from nucleo_kalman import FiltroKalman

OK = "OK "
FALLA = "FALLA"


def encabezado(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


# ---------------------------------------------------------------------------
def prueba_A_covarianza(par):
    """P debe ser simetrica y definida positiva en cada paso."""
    rng = np.random.default_rng(np.random.SeedSequence(par.semilla).spawn(1)[0])
    x_real, y, u = generar_escenario(par, rng)
    _, filtro = estimar(par, y, u)
    P = np.array(filtro.historial_P)

    asim = max(float(np.max(np.abs(Pk - Pk.T))) for Pk in P)
    autoval_min = min(float(np.min(np.linalg.eigvalsh(Pk))) for Pk in P)

    ok_sim = asim < 1e-12
    ok_pd = autoval_min > 0.0
    print(f"  asimetria maxima de P      : {asim:.3e}   "
          f"{OK if ok_sim else FALLA} (umbral 1e-12)")
    print(f"  autovalor minimo de P      : {autoval_min:.6e}   "
          f"{OK if ok_pd else FALLA} (debe ser > 0)")
    return ok_sim and ok_pd


# ---------------------------------------------------------------------------
def prueba_B_riccati(par):
    """La ganancia debe converger a la solucion de la Riccati algebraica."""
    A, H, Q, R = par.A, par.H, par.Q, np.atleast_2d(par.R)

    # Iteracion de la ecuacion de Riccati en tiempo discreto, independiente
    # del codigo del filtro.
    P = np.eye(2) * 1.0
    for _ in range(200_000):
        P_pri = A @ P @ A.T + Q
        S = H @ P_pri @ H.T + R
        K = P_pri @ H.T @ np.linalg.inv(S)
        P_nuevo = (np.eye(2) - K @ H) @ P_pri
        if np.max(np.abs(P_nuevo - P)) < 1e-15:
            P = P_nuevo
            break
        P = P_nuevo
    K_teorico = (A @ P @ A.T + Q) @ H.T @ np.linalg.inv(
        H @ (A @ P @ A.T + Q) @ H.T + R)

    rng = np.random.default_rng(np.random.SeedSequence(par.semilla).spawn(1)[0])
    x_real, y, u = generar_escenario(par, rng)
    _, filtro = estimar(par, y, u)
    K_final = np.array(filtro.historial_K[-1]).reshape(2, 1)

    err = float(np.max(np.abs(K_final - K_teorico)))
    ok = err < 1e-6
    print(f"  K de Riccati (independiente): "
          f"[{K_teorico[0,0]:.6f}, {K_teorico[1,0]:.6f}]")
    print(f"  K del filtro al paso final  : "
          f"[{K_final[0,0]:.6f}, {K_final[1,0]:.6f}]")
    print(f"  diferencia maxima           : {err:.3e}   "
          f"{OK if ok else FALLA} (umbral 1e-6)")
    return ok


# ---------------------------------------------------------------------------
def _errores_por_corrida(par, M=None):
    """Devuelve el error de estimacion y la covarianza de cada corrida."""
    M = M or par.M
    semillas = np.random.SeedSequence(par.semilla).spawn(M)
    errores, covs = [], []
    for hija in semillas:
        rng = np.random.default_rng(hija)
        x_real, y, u = generar_escenario(par, rng)
        x_est, filtro = estimar(par, y, u)
        errores.append(x_real - x_est)
        covs.append(np.array(filtro.historial_P))
    return np.array(errores), np.array(covs)


def prueba_C_insesgamiento(par, errores, quemado=200):
    """El error medio, en regimen permanente, debe ser indistinguible de cero."""
    e = errores[:, quemado:, :]
    ok_total = True
    for j, nombre in enumerate(["volumen  [m^3]  ", "tasa V'  [m^3/s]"]):
        por_corrida = e[:, :, j].mean(axis=1)
        t, p = stats.ttest_1samp(por_corrida, 0.0)
        ok = p > 0.01
        ok_total &= ok
        print(f"  error medio {nombre}: {por_corrida.mean():+.5f}   "
              f"t = {t:+7.3f}  p = {p:.4f}   "
              f"{OK if ok else FALLA} (se espera p > 0,01)")
    return ok_total


# ---------------------------------------------------------------------------
def prueba_D_nees(par, errores, covs, quemado=200):
    """NEES promedio dentro de las bandas chi-cuadrado al 95 %.

    Es la prueba de consistencia: verifica que la incertidumbre que el filtro
    declara corresponde al error que efectivamente comete. Un filtro que
    subestima su covarianza es un filtro sobreconfiado, y eso no se detecta
    mirando el RMSE.
    """
    M, N, n = errores.shape[0], errores.shape[1], 2
    e = errores[:, quemado:, :]
    P = covs[:, quemado:, :, :]

    nees = np.einsum('mti,mtij,mtj->mt', e, np.linalg.inv(P), e)
    nees_prom = float(nees.mean())

    lo = stats.chi2.ppf(0.025, M * n) / M
    hi = stats.chi2.ppf(0.975, M * n) / M
    ok = lo <= nees_prom <= hi

    print(f"  NEES promedio ({M} corridas) : {nees_prom:.4f}")
    print(f"  banda chi2 al 95 %          : [{lo:.4f} , {hi:.4f}]")
    print(f"  valor esperado (dim. estado): {n}")
    print(f"  {'dentro de banda' if ok else 'FUERA DE BANDA'}          : "
          f"{OK if ok else FALLA}")
    if not ok:
        sesgo = "sobreconfiado (P subestima el error)" if nees_prom > hi \
                else "subconfiado (P sobreestima el error)"
        print(f"  diagnostico                 : filtro {sesgo}")
    return ok


# ---------------------------------------------------------------------------
def prueba_E_innovacion(par, quemado=200):
    """La innovacion debe ser blanca: sin autocorrelacion significativa."""
    semillas = np.random.SeedSequence(par.semilla).spawn(par.M)
    r1 = []
    for hija in semillas:
        rng = np.random.default_rng(hija)
        x_real, y, u = generar_escenario(par, rng)
        _, filtro = estimar(par, y, u)
        inn = np.array(filtro.historial_innovacion).ravel()[quemado:]
        inn = inn - inn.mean()
        r1.append(float(np.sum(inn[1:] * inn[:-1]) / np.sum(inn**2)))
    r1 = np.array(r1)
    limite = 1.96 / np.sqrt(par.N - quemado)
    frac = float(np.mean(np.abs(r1) < limite))
    ok = frac >= 0.90
    print(f"  autocorrelacion lag-1 media : {r1.mean():+.5f}")
    print(f"  banda de blancura al 95 %   : +/- {limite:.5f}")
    print(f"  corridas dentro de banda    : {frac*100:.1f} %   "
          f"{OK if ok else FALLA} (se espera >= 90 %)")
    return ok


# ---------------------------------------------------------------------------
def main() -> int:
    par = ParametrosTolva()
    print(f"Auditoria del estimador | M = {par.M}, N = {par.N}, "
          f"semilla = {par.semilla}")

    encabezado("A. Simetria y definicion positiva de la covarianza")
    a = prueba_A_covarianza(par)

    encabezado("B. Convergencia a la solucion de Riccati")
    b = prueba_B_riccati(par)

    print("\n  (calculando errores y covarianzas de todas las corridas...)")
    errores, covs = _errores_por_corrida(par)

    encabezado("C. Insesgamiento del estimador")
    c = prueba_C_insesgamiento(par, errores)

    encabezado("D. Consistencia estadistica (NEES)")
    d = prueba_D_nees(par, errores, covs)

    encabezado("E. Blancura de la innovacion")
    e = prueba_E_innovacion(par)

    encabezado("RESULTADO")
    todas = {"A simetria/definicion positiva": a, "B Riccati": b,
             "C insesgamiento": c, "D consistencia NEES": d,
             "E blancura innovacion": e}
    for k, v in todas.items():
        print(f"  {k:34s} {OK if v else FALLA}")
    print()
    if all(todas.values()):
        print("  Las cinco propiedades se cumplen. La implementacion es")
        print("  correcta en el sentido teorico, no solo ejecutable.")
        return 0
    print("  Hay al menos una propiedad incumplida: revisar antes de")
    print("  reportar cualquier resultado.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
