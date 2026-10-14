from pathlib import Path
import runpy
import matplotlib.pyplot as plt

# Carpeta donde está este archivo maestro
BASE_DIR = Path(__file__).resolve().parent

# Simulaciones a ejecutar
SIMULACIONES = [
    "simulacion_kalman.py",
    "simulacion_potencia.py",
    "simulacion_termica.py",
    "simulacion_desgaste.py",
]


def ejecutar_todas():
    
   
    show_original = plt.show

    # Evita que cada script detenga la ejecución al llegar a plt.show().
    # Las figuras sí se crean y quedan abiertas en memoria.
    plt.show = lambda *args, **kwargs: None

    try:
        for archivo in SIMULACIONES:
            ruta = BASE_DIR / archivo

            if not ruta.exists():
                print(f"[ERROR] No se encontró: {ruta.name}")
                continue

            print("\n" + "=" * 70)
            print(f"EJECUTANDO: {ruta.name}")
            print("=" * 70)

            # Cada archivo se ejecuta en un espacio de nombres independiente.
            runpy.run_path(str(ruta), run_name=f"__sim_{ruta.stem}__")

    finally:
        # Restauramos el comportamiento normal de matplotlib.
        plt.show = show_original

    print("\n" + "=" * 70)
    print("SIMULACIONES COMPLETADAS")
    print("Se mostrarán todos los gráficos generados.")
    print("=" * 70)

    # Una sola llamada final: muestra simultáneamente todas las figuras.
    plt.show()


if __name__ == "__main__":
    ejecutar_todas()
