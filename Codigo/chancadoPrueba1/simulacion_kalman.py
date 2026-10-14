import numpy as np
import matplotlib.pyplot as plt

# 1. NÚCLEO ALGORÍTMICO (Filtro de Kalman Genérico)

class FiltroKalman:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.I = np.eye(H.shape[1]) 

    def prediccion(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def correccion(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y_tilde = y - (self.H @ self.x)
        self.x = self.x + K @ y_tilde
        self.P = (self.I - K @ self.H) @ self.P
        
        return self.x

# 2. CONFIGURACIÓN DEL ESCENARIO FÍSICO Y PARAMETRIZACIÓN

iteraciones = 600       # 10 minutos de simulación (si dt = 1s)
dt = 1.0                # Paso de tiempo discreto (1 segundo)
area_tolva = 25.0       # Área de la tolva en m^2

# Matrices del Sistema Físico
A = np.array([[1.0]])
B = np.array([[dt, -dt]])       # Dimensión 1x2 (Entrada suma, Salida resta)
H = np.array([[1.0 / area_tolva]]) # Dimensión 1x1 (Convierte Volumen a Nivel en metros)

# Matrices de Covarianza Estadísticas
Q = np.array([[0.05]])          # Incertidumbre física del flujo de mineral
R = np.array([[0.15]])          # Varianza del ruido del sensor ultrasónico (en metros^2)

# --- GENERACIÓN DE PERFILES DE FLUJO DINÁMICOS (u_k) ---
flujo_entrada = np.zeros(iteraciones)
# Simular la llegada de un camión entre el segundo 150 y el 350
flujo_entrada[150:350] = 2.5    # El camión descarga a una tasa de 2.5 m^3/s
flujo_descarga = np.full(iteraciones, 0.8) # El chancador extrae constantemente 0.8 m^3/s

# 3. GENERADOR DE DATOS SCADA (Física Real + Ruido)

estado_real_volumen = np.zeros(iteraciones)
medicion_ruidosa_nivel = np.zeros(iteraciones)

# Estado inicial real de la tolva (comienza con 30 m^3 de mineral)
x_actual = np.array([[30.0]])

for k in range(iteraciones):
    u_k = np.array([[flujo_entrada[k]], [flujo_descarga[k]]])
    
    # Inyección de ruidos gaussianos
    w_k = np.random.normal(0, np.sqrt(Q[0, 0]))
    v_k = np.random.normal(0, np.sqrt(R[0, 0]))
    
    # Física real del activo (Volumen)
    x_actual = A @ x_actual + B @ u_k + w_k
    estado_real_volumen[k] = x_actual[0, 0]
    
    # Lo que lee el SCADA (Nivel en metros con ruido)
    y_k = H @ x_actual + v_k
    medicion_ruidosa_nivel[k] = y_k[0, 0]

# 4. EJECUCIÓN DEL MÓDULO FILTRO DE KALMAN

# El filtro arranca con cierta incertidumbre sobre el volumen inicial
x_estimado_inicial = np.array([[15.0]]) 
P_inicial = np.array([[50.0]])

filtro = FiltroKalman(A, B, H, Q, R, x_estimado_inicial, P_inicial)
volumen_estimado = np.zeros(iteraciones)

for k in range(iteraciones):
    u_k = np.array([[flujo_entrada[k]], [flujo_descarga[k]]])
    y_k = np.array([[medicion_ruidosa_nivel[k]]]) # El filtro recibe el nivel ruidoso en metros
    
    # Ciclo Recursivo
    filtro.prediccion(u_k)
    x_optimo = filtro.correccion(y_k)
    
    volumen_estimado[k] = x_optimo[0, 0]

# Convertimos las lecturas del SCADA a volumen (usando la geometría) para poder comparar
volumen_crudo_scada = medicion_ruidosa_nivel * area_tolva

# 5. CÁLCULO DE MÉTRICAS Y VISUALIZACIÓN TÉCNICA
mse_scada = np.mean((estado_real_volumen - volumen_crudo_scada)**2)
mse_filtro = np.mean((estado_real_volumen - volumen_estimado)**2)

print("--- VALIDACIÓN DEL MODELO CINEMÁTICO DINÁMICO ---")
print(f"MSE del SCADA crudo (calculado desde nivel): {mse_scada:.2f} m^6")
print(f"MSE del Módulo Filtro de Kalman             : {mse_filtro:.2f} m^6")
print(f"Eficiencia de mitigación del ruido           : {((mse_scada - mse_filtro)/mse_scada)*100:.2f}%")

# Gráfico de resultados
plt.figure(figsize=(12, 6))
tiempo = np.arange(iteraciones)

plt.plot(tiempo, volumen_crudo_scada, label='Volumen interpretado por SCADA (Ruidoso)', color='red', alpha=0.25)
plt.plot(tiempo, estado_real_volumen, label='Volumen Real Interno (Física de la Tolva)', color='black', linewidth=2, linestyle='--')
plt.plot(tiempo, volumen_estimado, label='Estimación del Módulo Analítico (Kalman)', color='blue', linewidth=2)

plt.title('Reconstrucción Dinámica del Volumen de Carga en Tolva de Recepción', fontsize=14, fontweight='bold')
plt.xlabel('Tiempo (Segundos)', fontsize=12)
plt.ylabel('Volumen de Mineral ($m^3$)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()