import os
# 1. Mantenemos el truco que funcionó (Debe ir al principio del todo)
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd
import numpy as np

# 2. Configuración de audio para Guitarra (Tiempo Real)
fs = 44100         # Frecuencia de muestreo estándar
blocksize = 128    # Buffer muy bajo = latencia imperceptible (puedes probar 64 si tu PC es potente)
device_idx = 26    # ¡Tu índice mágico de Focusrite ASIO!

# 3. La función de procesamiento (Callback)
def callback(indata, outdata, frames, time, status):
    if status:
        print(f"Alerta de Audio: {status}")
    
    # indata[:, 0] -> Entrada 1 (Micrófono)
    # indata[:, 1] -> Entrada 2 (Guitarra)
    
    # 1. Extraemos solo la señal de la guitarra (columna 1)
    guitarra = indata[:, 1]
    
    # 2. Copiamos la guitarra tanto al canal izquierdo como al derecho de la salida
    outdata[:, 0] = guitarra  # Salida Izquierda (Audífono L)
    outdata[:, 1] = guitarra  # Salida Derecha (Audífono R)

# 4. Encender el motor de audio ASIO
try:
    with sd.Stream(device=device_idx,
                   samplerate=fs,
                   blocksize=blocksize,
                   channels=2,  # Focusrite Solo maneja Entrada 1 (Micro) y Entrada 2 (Instrumento)
                   callback=callback):
        
        print("-" * 50)
        print("¡ASIO ACTIVADO EXITOSAMENTE!")
        print("Tocando en tiempo real... Presiona Ctrl+C para detener el programa.")
        print("-" * 50)
        
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nPrograma detenido por el usuario.")
except Exception as e:
    print(f"\nError al iniciar el stream de audio: {e}") 