"""
figuras.py  -  Regenera las figuras sin volver a simular
=======================================================
Lee resultados/resultados.json y resultados/series.json y vuelve a escribir
resultados/figuras_es y resultados/figuras_en.

Uso:  python3 figuras.py
"""

from __future__ import annotations

import json
from pathlib import Path

import visualizacion

RAIZ = Path(__file__).parent


def main() -> int:
    salida = RAIZ / "resultados"
    R = json.loads((salida / "resultados.json").read_text(encoding="utf-8"))["resultados"]
    S = json.loads((salida / "series.json").read_text(encoding="utf-8"))
    visualizacion.generar(R, S, salida)
    print(f"Figuras escritas en {salida}/figuras_es y {salida}/figuras_en")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
