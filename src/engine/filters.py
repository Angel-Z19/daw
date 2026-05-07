# src/engine/filters.py
import numpy as np

def apply_distort(samples, gain=2.0, threshold=0.7):
    """
    Procedimiento: Algoritmo de Distorsión/Saturación.
    Multiplica la amplitud y recorta los picos para generar armónicos.
    """
    # Aplicamos la ganancia (Control/Regla de la Etapa 2)
    processed = samples * gain
    # Hard Clipping: limitamos la señal para crear el efecto de distorsión
    return np.clip(processed, -threshold, threshold)

def apply_delay(samples, delay_buffer, feedback=0.4):
    """
    Procedimiento: Línea de Retraso (Delay Line).
    Mezcla la señal actual con una señal previa guardada en un buffer.
    """
    # Mezclamos la señal original con la repetida (Anidamiento de señales)
    out = samples + (delay_buffer * feedback)
    # Normalizamos para evitar que el volumen suba infinitamente
    return np.clip(out, -1.0, 1.0)

def apply_lowpass(samples, alpha=0.5):
    """
    Procedimiento: Filtro Digital (Pasa-bajos).
    Suaviza los cambios bruscos de la señal para quitar 'brillo' o ruido.
    """
    # Implementación simple de un filtro de primer orden (Método)
    # out[n] = alpha * in[n] + (1 - alpha) * out[n-1]
    # Por simplicidad en este avance, usamos una media móvil básica
    return np.convolve(samples, np.ones(3)/3, mode='same')