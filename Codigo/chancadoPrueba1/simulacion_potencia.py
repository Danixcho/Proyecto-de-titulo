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


# 2. MODELO FÍSICO ELECTROMECÁNICO (Potencia Activa)

def generar_modelo_potencia(iteraciones, Q, R, flujo_entrada):
    estado_real_kw = np.zeros(iteraciones)
    medicion_scada_kw = np.zeros(iteraciones)
    
    A_elec = np.array([[0.85]]) # Inercia del motor
    B_elec = np.array([[12.0]]) # Demanda de kW por flujo de mineral
    H_elec = np.array([[1.0]])  # Lectura directa
    
    x_actual = np.array([[50.0]]) # Potencia base en vacío
    
    for k in range(iteraciones):
        u_k = np.array([[flujo_entrada[k]]])
        
        w_k = np.random.normal(0, np.sqrt(Q[0,0]))
        v_k = np.random.normal(0, np.sqrt(R[0,0]))
        
        # Dinámica real y lectura ruidosa
        x_actual = A_elec @ x_actual + B_elec @ u_k + w_k
        estado_real_kw[k] = x_actual[0, 0]
        
        y_k = H_elec @ x_actual + v_k
        medicion_scada_kw[k] = y_k[0, 0]
        
    return estado_real_kw, medicion_scada_kw, A_elec, B_elec, H_elec


# 3. EJECUCIÓN Y VALIDACIÓN

iteraciones = 500
Q = np.array([[2.0]])   # Incertidumbre por dureza variable de la roca
R = np.array([[25.0]])  # Ruido eléctrico/armónicos en la medición del SCADA

# Perfil de entrada: picos de mineral ingresando
flujo = np.zeros(iteraciones)
flujo[100:200] = 3.5
flujo[300:400] = 2.0

estado_real, mediciones, A, B, H = generar_modelo_potencia(iteraciones, Q, R, flujo)

# Inicialización del Filtro
filtro = FiltroKalman(A, B, H, Q, R, np.array([[0.0]]), np.array([[100.0]]))
estimacion_kw = np.zeros(iteraciones)

for k in range(iteraciones):
    u_k = np.array([[flujo[k]]])
    y_k = np.array([[mediciones[k]]])
    
    filtro.prediccion(u_k)
    x_opt = filtro.correccion(y_k)
    estimacion_kw[k] = x_opt[0,0]

# 4. GRÁFICOS Y RESULTADOS

print(f"MSE SCADA (Potencia): {np.mean((estado_real - mediciones)**2):.2f}")
print(f"MSE Filtro Kalman   : {np.mean((estado_real - estimacion_kw)**2):.2f}")

plt.figure(figsize=(10, 5))
plt.plot(mediciones, label='Lectura SCADA (kW ruidosos)', color='orange', alpha=0.4)
plt.plot(estado_real, label='Demanda Mecánica Real', color='black', linestyle='--')
plt.plot(estimacion_kw, label='Estimación Óptima (Kalman)', color='green')
plt.title('Dinámica Electromecánica: Consumo de Potencia Activa')
plt.xlabel('Tiempo (s)')
plt.ylabel('Potencia (kW)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()