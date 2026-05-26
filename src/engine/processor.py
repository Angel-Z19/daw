import pyaudio
import numpy as np
from src.engine.filters import apply_distort, apply_lowpass

class AudioEngine:
    def __init__(self):
        self.RATE = 44100
        self.CHUNK = 512
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # --- VARIABLES DE CONTROL (Enlazadas a la GUI) ---
        self.master_gain = 0.30
        
        # Distorsión
        self.distort_active = False
        self.distort_gain = 2.0
        self.distort_threshold = 0.70
        
        # Delay
        self.delay_active = False
        self.delay_feedback = 0.40
        
        # Filtro Pasa-Bajos
        self.filter_active = False
        self.filter_alpha = 0.50
        
        # NUEVO: Reverb
        self.reverb_active = False
        self.reverb_size = 0.60
        self.reverb_damping = 0.45
        self.reverb_wet = 0.35
        
        # NUEVO: EQ (Valores lineales derivados de los dB de la interfaz)
        self.eq_active = False
        self.eq_low = 1.0   # Corresponde a 0 dB (sin cambio)
        self.eq_mid = 1.0
        self.eq_high = 1.0
        
        # NUEVO: Compresor
        self.comp_active = False
        self.comp_threshold = -20.0
        self.comp_ratio = 4.0

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
        print("* Motor DSP v0.4 en línea. Procesando cadena de efectos...")
        try:
            while True:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)

                # --- CADENA DE EFECTOS (AUDIO SERIAL) ---
                processed = np.copy(samples)
                
                # 1. Compresor (Va primero para nivelar la señal de la guitarra)
                if self.comp_active:
                    processed = apply_compressor(processed, self.comp_threshold, self.comp_ratio)
                
                # 2. Ecualizador
                if self.eq_active:
                    processed = apply_eq(processed, self.eq_low, self.eq_mid, self.eq_high)

                # 3. Distorsión
                if self.distort_active:
                    processed = apply_distort(processed, self.distort_gain, self.distort_threshold)
                
                # 4. Filtro Pasa-Bajos
                if self.filter_active:
                    processed = apply_lowpass(processed, self.filter_alpha)
                    
                # 5. Reverb (Efecto espacial, va casi al final)
                if self.reverb_active:
                    processed = apply_reverb(processed, self.reverb_size, self.reverb_damping, self.reverb_wet)

                # 6. Ganancia Máster Final
                processed = processed * self.master_gain
                processed = np.clip(processed, -1.0, 1.0)

                # --- CONVERSIÓN Y SALIDA A 16 BITS ---
                int_samples = (processed * 32767).astype(np.int16)
                output_stereo = np.repeat(int_samples, 2)
                self.output_stream.write(output_stereo.tobytes())

        except Exception as e:
            print(f"Error en cadena DSP: {e}")
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