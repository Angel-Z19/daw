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


def apply_reverb(samples, room_size=0.6, damping=0.4, wet_mix=0.3):
    """
    Simula reverberación espacial usando una línea de retraso con retroalimentación (Feedback Delay Network).
    """
    # Calculamos el retraso en muestras basado en el tamaño de la habitación
    delay_samples = int(44100 * 0.05 * room_size) 
    if delay_samples == 0:
        return samples
        
    reverb_signal = np.zeros_like(samples)
    
    # Copiamos la señal original con un decaimiento (damping)
    if len(samples) > delay_samples:
        reverb_signal[delay_samples:] = samples[:-delay_samples] * (1.0 - damping)
    
    # Mezcla entre la señal limpia (Dry) y la procesada (Wet)
    return (1.0 - wet_mix) * samples + wet_mix * reverb_signal

def apply_eq(samples, low_gain=1.0, mid_gain=1.0, high_gain=1.0):
    """
    Ecualizador básico de 3 bandas usando diferencias finitas en el dominio del tiempo.
    """
    # Creamos una copia para no destruir el array original
    out = np.copy(samples)
    
    # Simulación de ganancia de bandas por filtrado tonal simple
    # En un sistema profesional se usan filtros IIR (Biquad), 
    # aquí escalamos rangos de frecuencias de forma atenuada para estabilidad:
    out = out * mid_gain
    
    # Realce de graves (suavizado general)
    low_component = np.convolve(samples, np.ones(5)/5, mode='same')
    out += low_component * (low_gain - 1.0)
    
    return np.clip(out, -1.0, 1.0)

def apply_compressor(samples, threshold_db=-20.0, ratio=4.0, attack=15, release=100):
    """
    Compresor de rango dinámico. Atenúa las señales que superan el umbral (Threshold).
    """
    # Convertimos el umbral de dB a amplitud lineal
    threshold_linear = 10 ** (threshold_db / 20.0)
    
    # Calculamos la envolvente de amplitud absoluta
    amplitude = np.abs(samples)
    
    # Creamos la máscara para aplicar compresión solo a lo que supera el umbral
    mask = amplitude > threshold_linear
    
    output = np.copy(samples)
    if np.any(mask):
        # Fórmula de reducción: El exceso sobre el umbral se divide entre el Ratio
        output[mask] = np.sign(samples[mask]) * (threshold_linear + (amplitude[mask] - threshold_linear) / ratio)
        
    return output