"""
main.py  -  Orquestador del estudio
===================================
Ejecuta el estudio completo con un solo comando:

  1. Verificacion de la implementacion (Experimento EV). Si alguna de las ocho
     pruebas falla, el programa se detiene sin ejecutar los experimentos.
  2. Experimentos E1 a E4 (incluidos E2b y E2c).
  3. Guarda los resultados (resultados.json, series.json), el entorno de
     ejecucion (entorno.json) y el costo computacional (rendimiento.json).
  4. Genera las figuras en espanol (tesis) y en ingles (articulo).
  5. Escribe el manifiesto MD5 de resultados.json y series.json.

Uso:
    python3 main.py                      # todo, en ./resultados
    python3 main.py --salida DIR         # otra carpeta de salida
    python3 main.py --sin-figuras        # sin figuras (verificar_reproducibilidad.py)

Las tablas LaTeX se generan despues con `python3 tablas.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import scipy

import experimentos
import verificacion
from parametros import (ParametrosExperimento, ParametrosProceso, ParametrosSensor,
                        ParametrosVerificacion, estimador_nominal)

RAIZ = Path(__file__).parent


def _a_json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"No serializable: {type(o)}")


def guardar_json(ruta: Path, obj) -> None:
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_a_json, allow_nan=False),
                    encoding="utf-8")


def entorno() -> dict:
    proc = platform.processor() or ""
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                proc = linea.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "plataforma": platform.platform(),
            "procesador": proc or "desconocido", "nucleos_logicos": os.cpu_count()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args(argv)
    salida: Path = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    pp, ps0, pexp, pv = ParametrosProceso(), ParametrosSensor(), ParametrosExperimento(), ParametrosVerificacion()
    pe = estimador_nominal(pp, ps0)

    print("[1/5] Verificacion de la implementacion (Experimento EV) ...")
    rng_ev = lambda m, f, fase=experimentos.PRUEBA: experimentos.rng_para(pexp, "EV", fase, m, f)  # noqa: E731
    V, serie_ev = verificacion.ejecutar(pe, ps0.sigma_v, rng_ev, pv)
    ok = True
    for k, v in V.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s} {'cumple' if v['ok'] else 'NO CUMPLE'}")
        ok &= v["ok"]
    if not ok:
        print("La verificacion fallo: no se ejecutan los experimentos.")
        return 1
    # Tasa de falsas fallas de un criterio de bandas por paso (ver verificacion.py)
    V["_criterio_bandas"] = verificacion.analisis_criterio_bandas(pe, ps0.sigma_v, rng_ev, pv)

    print("[2/5] Experimentos ...")
    R, S = experimentos.ejecutar(pp, ps0, pexp, log=print)
    S["verificacion"] = serie_ev
    rend = R.pop("_rendimiento")

    print("[3/5] Guardando resultados ...")
    P = {"proceso": asdict(pp), "sensor": asdict(ps0), "estimador": asdict(pe),
         "experimento": asdict(pexp), "verificacion": asdict(pv)}
    guardar_json(salida / "resultados.json", {"parametros": P, "verificacion": V, "resultados": R})
    guardar_json(salida / "series.json", S)
    guardar_json(salida / "entorno.json", entorno())
    guardar_json(salida / "rendimiento.json", rend)

    if not args.sin_figuras:
        print("[4/5] Figuras ...")
        import visualizacion
        visualizacion.generar(R, S, salida)
    else:
        print("[4/5] Figuras omitidas (--sin-figuras)")

    print("[5/5] Escribiendo manifiesto MD5 ...")
    lineas = [f"{hashlib.md5((salida / n).read_bytes()).hexdigest()}  {n}"
              for n in ("resultados.json", "series.json")]
    (salida / "manifiesto_md5.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nListo en {time.perf_counter() - t0:.1f} s. Resultados en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
