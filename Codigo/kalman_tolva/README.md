# Módulo de estimación de estados para gemelo digital de chancado primario

Implementación de referencia, reproducible y auditable, del módulo de software
analítico basado en el Filtro de Kalman Lineal Discreto.

**Autor:** Daniel Alejandro Prado Correa
**Asignatura:** Proyecto de Investigación (ACIF3002) — Universidad Andrés Bello
**Docente guía:** Omar Iván Salinas Silva

---

## Ejecución

```bash
pip install numpy scipy matplotlib
python main.py
```

Genera el árbol completo de evidencia en `resultados/`.

## Estructura

| Archivo | Rol |
|---|---|
| `nucleo_kalman.py` | Núcleo algorítmico genérico. No conoce el proceso minero. |
| `modelo_tolva.py` | Truth model de la tolva, parametrización y degradación SCADA. |
| `analisis.py` | Métricas de error, Monte Carlo e inferencia estadística. |
| `sensibilidad.py` | Barrido de degradación y malla de desajuste Q/R. |
| `figuras.py` | Figuras del artículo, formato IEEE. |
| `main.py` | Orquestador único del experimento. |

El núcleo del estimador está deliberadamente desacoplado del modelo físico: es
la misma clase la que puede operar sobre volumen de tolva, potencia activa,
temperatura de estator o desgaste de revestimiento sin modificación alguna. Esa
independencia es la evidencia del carácter autocontenido y desacoplado del
módulo declarado en el alcance del proyecto.

## Modelo

Vector de estado dinámico bidimensional en tiempo discreto:

```
x_k = [ V_k , V̇_k ]ᵀ

V_{k+1}  = V_k + Δt·V̇_k + Δt·u_k        A = [[1, Δt], [0, 1]]
V̇_{k+1} = V̇_k + w_k                     B = [Δt, 0]ᵀ
y_k      = V_k + v_k                     H = [1, 0]
```

`V` [m³] es el volumen retenido en la cámara de recepción; `V̇` [m³/s] la tasa
neta de variación, dominada por la extracción continua del chancador; `u_k`
[m³/s] el caudal de descarga de camión, evento operacional conocido y
registrado por el SCADA.

`V̇` **no está instrumentada**: es la variable latente que sustenta el problema
de observabilidad parcial. El par `(A, H)` es observable —la matriz de
observabilidad tiene rango 2— por lo que la tasa es reconstruible a partir de
la secuencia de mediciones de volumen pese a no medirse nunca de forma directa.

## Parametrización

| Parámetro | Valor | Unidad | Fundamento |
|---|---|---|---|
| `Δt` | 1.0 | s | Frecuencia de muestreo SCADA de 1 Hz |
| `N` | 1000 | pasos | Horizonte de ~16.7 min de operación |
| `M` | 50 | corridas | Monte Carlo |
| semilla | 42 | — | Reproducibilidad determinista |
| `Q` | diag(0.01, 1.0e-4) | m⁶, m⁶/s² | Ruido de proceso |
| `R` | 5.0 | m⁶ | Ruido de medición (σ = 2.236 m³) |
| `x₀` real | [15.0, −0.8] | m³, m³/s | Tolva casi vacía a inicio de turno |
| `x̂₀` filtro | [15.0, 0.0] | m³, m³/s | El filtro desconoce la tasa |
| `P₀` | diag(10.0, 1.0) | — | Incertidumbre inicial |
| `u` camión | 4.5 | m³/s | ~202 m³ por evento de 45 s |

**Calibración física.** Extracción nominal de 0.8 m³/s equivale a ~5760 t/h con
densidad aparente 2.0 t/m³, coherente con un chancador giratorio de gran
capacidad. Cada descarga de camión aporta ~202 m³, consistente con un CAEX de
300–400 t. Cuatro ciclos de camión en el horizonte mantienen la tolva en una
banda operativa de 7 a 300 m³.

## Nota de recalibración de Q₂₂

Las entregas de Fase 3 declaraban `Q = diag(0.01, 0.05)`. Se verificó
numéricamente que una covarianza de proceso de 0.05 por paso sobre la tasa
produce, en un horizonte de N = 1000, una caminata aleatoria con desviación
acumulada de `sqrt(1000 × 0.05) = 7.07` m³/s, casi nueve veces la tasa de
extracción nominal de 0.8 m³/s. Ese régimen es físicamente inadmisible.

Se recalibró `Q₂₂ = 1.0e-4`, que produce una dispersión acumulada de 0.32 m³/s
en torno a la tasa nominal —aproximadamente ±40% de variabilidad atribuible a
la dureza variable del mineral—, coherente con la operación real. `Q₁₁ = 0.01`
se mantiene sin cambios.

## Coherencia dimensional

Se distingue explícitamente entre métricas cuadráticas y lineales:

- **MSE** en m⁶ — métrica descriptiva del error cuadrático.
- **RMSE** en m³ — en las unidades de la variable física.
- **Intervalos de confianza** en m³ — toda la inferencia estadística se realiza
  sobre el RMSE, de modo que los resultados inferenciales son dimensionalmente
  homogéneos con el estado estimado.

## Validación estadística

Cada corrida de Monte Carlo aporta un par (RMSE_SCADA, RMSE_KF). Sobre los
M = 50 pares se aplica una prueba t de Student pareada unilateral con α = 0.05.
El supuesto de normalidad de las diferencias se verifica con Shapiro-Wilk y se
reporta además el contraste no paramétrico de Wilcoxon de rangos con signo,
de modo que la conclusión no dependa del cumplimiento del supuesto paramétrico.

## Salidas

```
resultados/
├── resultados.json                        resumen consolidado + entorno de ejecución
├── logs/
│   ├── serie_temporal_corrida01.csv       16 columnas × 1000 pasos
│   ├── metricas_montecarlo.csv            métricas de las 50 corridas
│   ├── barrido_degradacion.csv            sensibilidad a R real
│   └── malla_desajuste_QR.csv             malla de robustez 5×5
└── figuras/                               fig1..fig6 en PDF vectorial y PNG 300 ppp
```

## Reproducibilidad

Todo el experimento queda determinado por la semilla maestra. Las corridas de
Monte Carlo usan flujos independientes derivados mediante `SeedSequence.spawn()`,
lo que garantiza independencia estadística entre corridas y reproducibilidad
exacta del conjunto. Dos ejecuciones sucesivas producen archivos CSV idénticos
byte a byte, verificado por suma MD5.

## Limitación de madurez tecnológica

El módulo se valida exclusivamente sobre datos sintéticos paramétricos, en
razón de las restricciones de confidencialidad industrial (NDA) que impiden el
acceso a telemetría SCADA real de faenas operativas. Su nivel de madurez
tecnológica corresponde a laboratorio y simulación. Cualquier despliegue sobre
un gemelo digital operativo exige pilotaje físico in situ y calibración empírica
de seguridad previos.
