# src/engine/processor.py
import pyaudio
import numpy as np
from .filters import apply_distort, apply_delay, apply_lowpass

class AudioEngine:
    def __init__(self):
        # Recursos del Sistema (Etapa 2)
        self.CHUNK = 1024
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        
        # Inicialización de la memoria para el Delay (Procedimiento)
        # Creamos un buffer de 1 segundo de duración
        self.delay_buffer = np.zeros(self.RATE, dtype=np.float32)
        self.delay_ptr = 0 # Puntero para recorrer el buffer circular

    def start(self):
        # Adquisición de Audio (Entrada ADC)
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK
        )
        print("* Motor DSP en línea. Procesando señal...")
        self.run()

    def run(self):
        try:
            while True:
                # 1. Entrada: Leer datos del ADC
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32).copy()

                # 2. PROCESAMIENTO (Motor DSP - Etapa 2)
                # Aplicamos los filtros de forma lineal (uno tras otro)
                
                # A. Distorsión
                samples = apply_distort(samples, gain=2.5)
                
                # B. Delay (Usando el buffer circular)
                # Extraemos el fragmento de audio del pasado
                past_samples = self.delay_buffer[self.delay_ptr : self.delay_ptr + self.CHUNK]
                
                # Aplicamos la lógica de mezcla
                samples = apply_delay(samples, past_samples, feedback=0.3)
                
                # Guardamos la señal procesada actual en el buffer para el futuro
                self.delay_buffer[self.delay_ptr : self.delay_ptr + self.CHUNK] = samples
                
                # C. Filtro Pasa-bajos (Limpieza de señal)
                samples = apply_lowpass(samples)

                # Actualizamos el puntero del buffer circular
                self.delay_ptr = (self.delay_ptr + self.CHUNK) % (self.RATE - self.CHUNK)

                # 3. Salida: Enviar al DAC (Audífonos)
                self.stream.write(samples.tobytes())

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("* Deteniendo sistema...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()