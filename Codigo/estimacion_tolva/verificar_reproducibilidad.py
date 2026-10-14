"""
verificar_reproducibilidad.py  -  Doble ejecucion y comparacion MD5
===================================================================
Copia el paquete en dos directorios temporales independientes, ejecuta el
estudio completo en cada uno (sin figuras, que no forman parte del manifiesto)
y compara los manifiestos MD5 de resultados.json y series.json. Si existe el
manifiesto de la ejecucion reportada (resultados/manifiesto_md5.txt), indica
ademas si coincide con el.

Uso:  python3 verificar_reproducibilidad.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).parent
MODULOS = ("parametros.py", "simulador.py", "sensor.py", "estimador.py", "referencias.py",
           "evaluador.py", "verificacion.py", "experimentos.py", "visualizacion.py", "main.py")


def ejecutar_en(destino: Path) -> str:
    for m in MODULOS:
        shutil.copy2(RAIZ / m, destino / m)
    r = subprocess.run([sys.executable, "main.py", "--sin-figuras"], cwd=destino,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return (destino / "resultados" / "manifiesto_md5.txt").read_text(encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        m1 = ejecutar_en(Path(d1))
        m2 = ejecutar_en(Path(d2))
    print("Manifiesto 1:")
    print(m1, end="")
    print("Manifiesto 2:")
    print(m2, end="")
    reportado = RAIZ / "resultados" / "manifiesto_md5.txt"
    if reportado.exists():
        igual = reportado.read_text(encoding="utf-8") == m1
        print(f"\nCoincide con el manifiesto de la ejecucion reportada: {'si' if igual else 'NO'}")
    if m1 == m2:
        print("\nREPRODUCIBLE: las dos ejecuciones produjeron salidas identicas.")
        return 0
    print("\nNO REPRODUCIBLE: las salidas difieren.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
