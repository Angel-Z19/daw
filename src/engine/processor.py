import pyaudio
import numpy as np
from src.engine.filters import apply_distort, apply_lowpass

class AudioEngine:
    def __init__(self):
        self.RATE = 44100
        self.CHUNK = 256
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

    def start(self):
            # IDs: 1 (Focusrite), 8 (Realtek)
            try:
                # ENTRADA: Mantenemos Float32 porque la Focusrite es pro
                self.input_stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=1, 
                    rate=self.RATE,
                    input=True,
                    input_device_index=1,
                    frames_per_buffer=self.CHUNK
                )

                # SALIDA: Cambiamos a paInt16 para máxima compatibilidad con Realtek
                self.output_stream = self.p.open(
                    format=pyaudio.paInt16, # <--- CAMBIO CLAVE
                    channels=2,
                    rate=self.RATE,
                    output=True,
                    output_device_index=7,
                    frames_per_buffer=self.CHUNK
                )
                self.run()
            except Exception as e:
                print(f"Error hardware: {e}")

    def run(self):
        print("* DSP Activo: Escuchando guitarra en Mono...")
        try:
            while True:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                # 1. Convertimos entrada a array de números
                samples = np.frombuffer(data, dtype=np.float32)

                # 2. PROCESAMIENTO MUY BAJO (Evitamos estática)
                # Usamos 0.1 para que el sonido sea suave
                processed = samples * 0.3
                processed = np.clip(processed, -1.0, 1.0)

                # 3. CONVERSIÓN A INT16 (Para que los audífonos entiendan)
                # Pasamos de rango -1.0 a 1.0 al rango de 16 bits (-32768 a 32767)
                int_samples = (processed * 32767).astype(np.int16)

                # 4. DUPLICAR A ESTÉREO (L y R para audífonos)
                output_stereo = np.column_stack((int_samples, int_samples)).flatten()

                # 5. ESCRIBIR
                output_stereo = np.repeat(int_samples, 2)
                self.output_stream.write(output_stereo.tobytes())

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()