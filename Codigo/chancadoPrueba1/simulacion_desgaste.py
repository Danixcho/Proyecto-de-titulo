import numpy as np
import matplotlib.pyplot as plt

# 1. NÚCLEO ALGORÍTMICO (Filtro de Kalman)

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

# 2. MODELO FÍSICO DE DESGASTE ESTRUCTURAL

def generar_modelo_desgaste(iteraciones, Q, R, flujo_entrada):
    estado_real_mm = np.zeros(iteraciones)
    medicion_scada_mm = np.zeros(iteraciones)
    
    A_mec = np.array([[1.0]])    
    B_mec = np.array([[-0.05]])  # Pérdida de 0.05 mm por unidad de flujo
    H_mec = np.array([[1.0]])
    
    x_actual = np.array([[120.0]]) # Grosor inicial del liner (120 mm)
    
    for k in range(iteraciones):
        u_k = np.array([[flujo_entrada[k]]])
        
        w_k = np.random.normal(0, np.sqrt(Q[0,0]))
        v_k = np.random.normal(0, np.sqrt(R[0,0]))
        
        x_actual = A_mec @ x_actual + B_mec @ u_k + w_k
        estado_real_mm[k] = x_actual[0, 0]
        
        y_k = H_mec @ x_actual + v_k
        medicion_scada_mm[k] = y_k[0, 0]
        
    return estado_real_mm, medicion_scada_mm, A_mec, B_mec, H_mec

# 3. EJECUCIÓN Y VALIDACIÓN

iteraciones = 1000
Q = np.array([[0.01]])  # Incertidumbre baja (desgaste constante)
R = np.array([[1.5]])   # Ruido alto en sensor acústico/indirecto

# Perfil de entrada continuo con variaciones
flujo = np.random.uniform(0.5, 2.5, iteraciones)

estado_real, mediciones, A, B, H = generar_modelo_desgaste(iteraciones, Q, R, flujo)

# Inicialización del Filtro
filtro = FiltroKalman(A, B, H, Q, R, np.array([[120.0]]), np.array([[10.0]]))
estimacion_mm = np.zeros(iteraciones)

for k in range(iteraciones):
    u_k = np.array([[flujo[k]]])
    y_k = np.array([[mediciones[k]]])
    
    filtro.prediccion(u_k)
    x_opt = filtro.correccion(y_k)
    estimacion_mm[k] = x_opt[0,0]

# 4. GRÁFICOS Y RESULTADOS
print(f"MSE SCADA (Desgaste) : {np.mean((estado_real - mediciones)**2):.2f}")
print(f"MSE Filtro Kalman    : {np.mean((estado_real - estimacion_mm)**2):.2f}")

plt.figure(figsize=(10, 5))
plt.plot(mediciones, label='Lectura Sensor Acústico (Ruido)', color='gray', alpha=0.5)
plt.plot(estado_real, label='Desgaste Estructural Real', color='black', linestyle='--')
plt.plot(estimacion_mm, label='Estimación Larga Duración', color='purple')
plt.title('Tendencia de Desgaste Estructural a Largo Plazo')
plt.xlabel('Iteraciones (Toneladas Procesadas)')
plt.ylabel('Grosor del Revestimiento (mm)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()