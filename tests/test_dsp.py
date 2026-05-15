import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine.filters import apply_distort, apply_lowpass
import matplotlib.pyplot as plt
from src.engine.filters import apply_distort, apply_lowpass

# 1. Generar una señal senoidal pura (Entrada limpia)
fs = 44100
t = np.linspace(0, 0.01, int(fs * 0.01)) # 10ms de sonido
frecuencia = 440 # Nota LA
entrada = 0.5 * np.sin(2 * np.pi * frecuencia * t)

# 2. Procesar la señal con tus filtros
# Aplicamos distorsión fuerte para que se note el cambio visual
distorsionada = apply_distort(entrada, gain=5.0)
# Aplicamos el filtro pasa-bajos a la distorsionada
procesada = apply_lowpass(distorsionada)

# 3. Graficar los resultados
plt.figure(figsize=(10, 6))

plt.plot(t, entrada, label="Entrada Limpia (Seno)", color="blue", linestyle="--")
plt.plot(t, distorsionada, label="Efecto Distorsión (Clipping)", color="red")
plt.plot(t, procesada, label="Salida Final (Filtrada)", color="green", linewidth=2)

plt.title("Prueba de Concepto: Análisis de Forma de Onda")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# Guardar la imagen para el reporte de LaTeX
plt.savefig("grafica_audio.png")
plt.show()