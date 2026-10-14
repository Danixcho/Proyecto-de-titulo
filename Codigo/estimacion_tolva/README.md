# estimacion_tolva

Módulo de software para la estimación de estados en la tolva de alimentación de un
chancador primario, evaluado mediante simulación Monte Carlo. Es el anexo digital de la
tesis «Diseño y evaluación mediante simulación computacional de un módulo de software
para la estimación de estados en la tolva de alimentación de un chancador primario, como
componente de un gemelo digital conceptual» (Daniel Alejandro Prado Correa, Ingeniería
Civil Informática, Universidad Andrés Bello, 2026).

**Alcance.** Todo lo que produce este paquete proviene de simulación. Ningún parámetro
fue medido en una planta: son supuestos de diseño declarados en `parametros.py` y en el
Capítulo 4 de la tesis. Los resultados no constituyen evidencia de desempeño industrial.

## Estructura

| Archivo | Capa | Qué hace |
|---|---|---|
| `parametros.py` | — | Parámetros del proceso, del sensor, del estimador y del protocolo, en grupos separados que no se mezclan. |
| `simulador.py` | 1. Proceso | Genera la trayectoria verdadera: balance de masa con límites físicos (tolva vacía y llena), capacidad de extracción Ornstein-Uhlenbeck, llegadas de camiones gamma. Con `saturacion=False` (solo para la ablación E2c) la extracción no se satura: es un contrafactual físicamente imposible. |
| `sensor.py` | 2. Sensor | Convierte el volumen verdadero en medición: ruido blanco, coloreado AR(1), intensificado en descargas y deriva. |
| `estimador.py` | 3. Estimador | Filtro de Kalman lineal de dos estados (volumen y tasa de extracción) con actualización de Joseph. |
| `referencias.py` | 3. Estimador | Estimadores sin modelo dinámico: media móvil, regresión local y diferencias finitas. La regresión local y las diferencias finitas usan la descarga conocida; la media móvil no. |
| `evaluador.py` | 4. Evaluación | RMSE, MAE, NEES, NIS, autocorrelación, bootstrap y prueba de Wilcoxon. |
| `verificacion.py` | 4. Evaluación | Experimento EV: ocho pruebas de que la implementación cumple la teoría, y una estimación de cuántas veces los criterios de banda D y E fallarían con un filtro exactamente consistente. |
| `experimentos.py` | 4. Evaluación | Protocolo de los experimentos E1 a E4 (incluidos el filtro calibrado E2b y la ablación de la saturación E2c), semillas y números aleatorios comunes. |
| `visualizacion.py` | 5. Visualización | Figuras en español (tesis) y en inglés (artículo). |
| `main.py` | — | Orquestador: verifica, ejecuta, guarda, grafica y escribe el manifiesto MD5. |
| `figuras.py` | — | Regenera las figuras desde los resultados guardados, sin volver a simular. |
| `tablas.py` | — | Genera las tablas LaTeX de la tesis y del artículo desde `resultados.json`. |
| `verificar_reproducibilidad.py` | — | Ejecuta el estudio dos veces y compara las sumas MD5. |

## Ejecución

Requiere Python 3.12 o superior (lo exigen las versiones de NumPy y SciPy fijadas en
`requirements.txt`). Versiones con que se generaron los resultados de la
tesis (ver `resultados/entorno.json`): Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2.

```bash
pip install -r requirements.txt
python3 main.py                        # estudio completo, ~45 s
python3 tablas.py                      # tablas LaTeX desde los resultados
python3 verificar_reproducibilidad.py  # dos ejecuciones y comparación MD5
```

`main.py` se detiene sin ejecutar los experimentos si alguna prueba de verificación falla.

## Salidas (`resultados/`)

| Archivo | Contenido |
|---|---|
| `resultados.json` | Parámetros, verificación y todos los resultados numéricos. |
| `series.json` | Series temporales para las figuras. |
| `manifiesto_md5.txt` | Sumas MD5 de los dos archivos anteriores. |
| `entorno.json` | Versiones de software y plataforma (fuera del manifiesto). |
| `rendimiento.json` | Tiempo de cómputo por paso (fuera del manifiesto: varía entre ejecuciones). |
| `figuras_es/`, `figuras_en/` | Figuras en PDF y PNG. |
| `tablas_es/`, `tablas_en/` | Tablas LaTeX. |

## Reproducibilidad

Toda la aleatoriedad deriva de la semilla maestra 2026 mediante
`numpy.random.SeedSequence`, con claves (experimento, fase, corrida, flujo). Las corridas
de calibración de las líneas base usan semillas distintas de las de prueba. Con las
versiones indicadas, `verificar_reproducibilidad.py` produjo salidas idénticas byte a
byte. Con otras versiones de NumPy o SciPy pueden cambiar los últimos decimales; las
conclusiones no deberían cambiar, pero eso se comprueba ejecutando, no suponiendo.

Manifiesto de la ejecución reportada en la tesis:

```
0f0f42a23309c30c27f312c09e900d5b  resultados.json
9137ea8b9440680b1b4559c4823569c9  series.json
```

## Historial

- **Octubre de 2026.** El paquete se reconstruyó después de perderse el espacio de
  trabajo en que se había desarrollado. La reconstrucción reproduce exactamente las
  trayectorias simuladas y las estimaciones puntuales de la ejecución anterior (por
  ejemplo, el volumen mínimo de la ablación, −625,871 m³, coincide en todos sus
  decimales). Los intervalos bootstrap difieren en algunas décimas de punto porcentual,
  porque el remuestreo se reimplementó. Se agregó la estimación de la tasa de fallas de
  los criterios D y E (`verificacion.analisis_criterio_bandas`).
