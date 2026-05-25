# src/engine/filters.py
import numpy as np


def apply_distort(samples, gain=2.0, threshold=0.7):
    """
    Hard clipping: multiplica la amplitud y recorta los picos
    para generar armónicos de distorsión.
    """
    return np.clip(samples * gain, -threshold, threshold)


def apply_lowpass(samples, alpha=0.5):
    """
    Filtro pasa-bajos FIR básico (para uso fuera del callback).
    En el callback de audio se usa la versión IIR inline con estado.
    """
    return np.convolve(samples, np.ones(3) / 3, mode='same')


class DelayLine:
    """
    Línea de delay circular con feedback configurable.

    Implementa un eco tipo "tape delay":
      y[n] = x[n] + buf[pos] * feedback
      buf[pos] = y[n]          ← escribe la señal mezclada (feedback real)
      pos = (pos + 1) % delay_samples

    Con delay_ms = 300 ms el primer eco se escucha a los 300 ms.
    Feedback < 1.0 garantiza que las repeticiones decaigan.
    """

    def __init__(self, delay_ms: float = 300.0, sample_rate: int = 44100):
        self.delay_samples = int(sample_rate * delay_ms / 1000.0)
        self.buf = np.zeros(self.delay_samples, dtype=np.float32)
        self.pos = 0

    def process(self, samples: np.ndarray, feedback: float = 0.4) -> np.ndarray:
        """
        Procesa un bloque de muestras.  El bucle Python es inevitable para
        este delay con feedback, pero con CHUNK=128 solo son 128 iteraciones
        (~25 µs) — perfectamente dentro del presupuesto de 2.9 ms del callback.
        """
        n   = len(samples)
        L   = self.delay_samples
        out = np.empty(n, dtype=np.float32)
        pos = self.pos
        buf = self.buf

        for i in range(n):
            delayed  = buf[pos]
            mixed    = samples[i] + delayed * feedback
            # Clip inline (más rápido que np.clip para escalares)
            if   mixed >  1.0: mixed =  1.0
            elif mixed < -1.0: mixed = -1.0
            buf[pos] = mixed
            out[i]   = mixed
            pos += 1
            if pos == L:
                pos = 0

        self.pos = pos
        return out

    def reset(self):
        """Limpia el buffer (útil al reactivar el efecto)."""
        self.buf[:] = 0.0
        self.pos    = 0
