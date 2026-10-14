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


# 2. MODELO FÍSICO TÉRMICO (Temperatura del Motor)

def generar_modelo_termico(iteraciones, Q, R, carga_trabajo):
    estado_real_temp = np.zeros(iteraciones)
    medicion_scada_temp = np.zeros(iteraciones)
    
    # Parámetros físicos térmicos
    A_term = np.array([[0.95]]) # Inercia térmica alta: retiene 95% del calor anterior (disipa 5%)
    B_term = np.array([[2.5]])  # Se calienta 2.5 °C por cada unidad de carga procesada
    H_term = np.array([[1.0]])
    
    x_actual = np.array([[20.0]]) # Temperatura inicial (delta sobre ambiente)
    
    for k in range(iteraciones):
        u_k = np.array([[carga_trabajo[k]]])
        
        # Ruido de proceso (ventilación variable) y sensor (ruido electromagnético)
        w_k = np.random.normal(0, np.sqrt(Q[0,0]))
        v_k = np.random.normal(0, np.sqrt(R[0,0]))
        
        # Dinámica de calentamiento y enfriamiento
        x_actual = A_term @ x_actual + B_term @ u_k + w_k
        estado_real_temp[k] = x_actual[0, 0]
        
        # Lectura ruidosa de la termocupla
        y_k = H_term @ x_actual + v_k
        medicion_scada_temp[k] = y_k[0, 0]
        
    return estado_real_temp, medicion_scada_temp, A_term, B_term, H_term


# 3. EJECUCIÓN Y VALIDACIÓN

iteraciones = 600
Q = np.array([[0.5]])   # Incertidumbre física baja (la termodinámica es estable)
R = np.array([[12.0]])  # Ruido eléctrico/inductivo moderado en el sensor

# Perfil de entrada: El motor trabaja intenso, descansa y vuelve a trabajar
carga = np.zeros(iteraciones)
carga[50:250] = 4.0   # Operación a plena carga
carga[250:400] = 0.5  # Operación casi en vacío (enfriamiento)
carga[400:550] = 3.5  # Nueva tanda de operación

estado_real, mediciones, A, B, H = generar_modelo_termico(iteraciones, Q, R, carga)

# Inicialización del Filtro
filtro = FiltroKalman(A, B, H, Q, R, np.array([[20.0]]), np.array([[50.0]]))
estimacion_temp = np.zeros(iteraciones)

for k in range(iteraciones):
    u_k = np.array([[carga[k]]])
    y_k = np.array([[mediciones[k]]])
    
    filtro.prediccion(u_k)
    x_opt = filtro.correccion(y_k)
    estimacion_temp[k] = x_opt[0,0]


# 4. GRÁFICOS Y RESULTADOS

print(f"MSE Termocupla Cruda  : {np.mean((estado_real - mediciones)**2):.2f}")
print(f"MSE Filtro Kalman     : {np.mean((estado_real - estimacion_temp)**2):.2f}")

plt.figure(figsize=(10, 5))
plt.plot(mediciones, label='Lectura PT100 (Ruido EMI)', color='salmon', alpha=0.5)
plt.plot(estado_real, label='Temperatura Real del Estator', color='darkred', linestyle='--')
plt.plot(estimacion_temp, label='Estimación Filtrada', color='blue')
plt.title('Dinámica Térmica del Motor Principal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Delta de Temperatura (°C)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()