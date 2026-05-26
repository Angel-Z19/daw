import pyaudio
import numpy as np
import os
from src.engine.filters import apply_distort, apply_lowpass, apply_compressor, apply_delay, apply_eq, apply_reverb


class AudioEngine:
    def __init__(self):
        self.RATE = 44100
        self.CHUNK = 256
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # --- VARIABLES DE CONTROL (Enlazadas a la GUI) ---
        self.master_gain = 0.30
        
        self.stream = None
        
        # Valores iniciales por defecto para que no se queje el editor
        self.comp_on = False
        self.c_thr = -20.0
        self.c_rat = 4.0
        
        self.eq_on = False
        self.eq_l = 1.0
        self.eq_m = 1.0
        self.eq_h = 1.0
        
        self.gain = 1.0
        self.dist_on = False
        self.dgain = 2.0
        self.dthresh = 0.5
        
        self.delay_on = False
        self.delay_buf = np.zeros(44100) # Buffer para un segundo de delay
        self.dfb = 0.5
        
        self.lp_on = False
        self.alpha = 0.5

    def start(self):
        try:
            import os
            # 1. Activamos la variable de entorno obligatoria antes de importar sounddevice
            os.environ["SD_ENABLE_ASIO"] = "1"
            
            import sounddevice as sd
            import numpy as np

            # 2. Tu configuración exacta que ya demostró tener CERO delay
            fs = 44100         
            blocksize = 128    # Forzamos el buffer ultra bajo de tu prueba exitosa
            device_idx = 26    # ¡Tu índice mágico que conecta directo a Focusrite ASIO!

            # 3. Encendemos el stream ASIO nativo usando tus funciones internas
            self.stream = sd.Stream(
                device=device_idx,
                samplerate=fs,
                blocksize=blocksize,
                channels=2,        # Configuración estéreo idéntica a tu test.py
                callback=self._callback
            )
            
            self.stream.start()
            print("-" * 50)
            print("¡MOTOR PRINCIPAL CONECTADO A FOCUSRITE ASIO (CERO DELAY)!")
            print("-" * 50)
            return True, None
            
        except Exception as e:
            return False, str(e)
            
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

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Alerta de Audio: {status}")
        
        # 1. Entrada de la guitarra en canal ASIO correcto
        guitarra = indata[:, 1]
        proc = guitarra.copy()
        
        # =========================================================================
        # ─── CADENA DSP CON ADVERTENCIAS CORREGIDAS (Añadiendo 'self.') ──────────
        # =========================================================================
        
        # A. Compresor (Usa los estados guardados en la instancia del motor)
        if self.comp_on:
            proc = apply_compressor(proc, self.c_thr, self.c_rat)
            
        # B. Ecualizador
        if self.eq_on:
            proc = apply_eq(proc, self.eq_l, self.eq_m, self.eq_h)
            
        # C. Ganancia Maestra + Distorsión
        proc = proc * self.gain
        if self.dist_on:
            proc = apply_distort(proc, self.dgain, self.dthresh)
            
        # D. Delay
        if self.delay_on:
            proc = apply_delay(proc, self.delay_buf, self.dfb)
            self.delay_buf = proc.copy()
            
        # E. Filtro pasa-bajos
        if self.lp_on:
            filt = np.empty_like(proc)
            prev = proc[0]
            for i in range(len(proc)):
                # Cambiamos alpha por self.alpha
                prev = self.alpha * proc[i] + (1.0 - self.alpha) * prev
                filt[i] = prev
            proc = filt

        # Enviar salida estéreo limpia a la Focusrite
        outdata[:, 0] = proc  
        outdata[:, 1] = proc

    def stop(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()