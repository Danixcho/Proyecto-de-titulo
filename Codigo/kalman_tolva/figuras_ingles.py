"""
figuras_ingles.py
=================
Regenera el conjunto completo de figuras con rotulacion en ingles, para su uso
en el articulo cientifico en formato IEEE.

Reejecuta exactamente el mismo experimento que main.py (misma semilla, mismos
parametros), de modo que las figuras en espanol y en ingles representan los
mismos datos. Escribe en resultados/figuras_en/.

Uso:  python3 figuras_ingles.py
"""

from __future__ import annotations

from pathlib import Path

import figuras
from modelo_tolva import ParametrosTolva
from analisis import monte_carlo, consolidar
from sensibilidad import barrido_degradacion, malla_desajuste

DESTINO = Path(__file__).parent / "resultados" / "figuras_en"


def main() -> int:
    par = ParametrosTolva()

    print("[1/3] Reejecutando Monte Carlo ...")
    filas, series = monte_carlo(par)
    resumen, arr = consolidar(filas)

    print("[2/3] Reejecutando analisis de sensibilidad ...")
    barrido = barrido_degradacion(par)
    malla = malla_desajuste(par)

    print("[3/3] Generando figuras con rotulacion en ingles ...")
    DESTINO.mkdir(parents=True, exist_ok=True)
    nombres = figuras.generar_todas(par, series, arr, resumen, barrido, malla,
                                    DESTINO, idioma="en")
    for n in nombres:
        print(f"      {n}.pdf")

    print(f"\nFiguras en ingles escritas en: {DESTINO}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
