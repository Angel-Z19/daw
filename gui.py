#!/usr/bin/env python3
"""
DAW GUI — Estación de Trabajo de Audio Digital
Interfaz gráfica: PyQt6 + QPainter | Audio: sounddevice
Funciona en: Windows, Linux y Mac
Ejecución: python gui.py
"""

import sys
import os
import threading
import numpy as np
import sounddevice as sd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QSizePolicy,
    QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF

# ── Ruta para encontrar el módulo src ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine.filters import apply_distort, apply_delay

def apply_compressor(samples, threshold_db=-20.0, ratio=4.0):
    """Atenúa los picos de volumen que superan el umbral (Threshold)."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    amplitude = np.abs(samples)
    mask = amplitude > threshold_linear
    output = np.copy(samples)
    if np.any(mask):
        output[mask] = np.sign(samples[mask]) * (threshold_linear + (amplitude[mask] - threshold_linear) / ratio)
    return output

def apply_eq(samples, low_gain=1.0, mid_gain=1.0, high_gain=1.0):
    """Ecualizador básico de tres bandas (filtro tonal por escalado)."""
    out = np.copy(samples) * mid_gain
    # Componente de graves (suavizado por promedio móvil)
    low_component = np.convolve(samples, np.ones(5)/5, mode='same')
    out += low_component * (low_gain - 1.0)
    return np.clip(out, -1.0, 1.0)

def apply_reverb(samples, reverb_buf, room_size=0.6, damping=0.4, wet_mix=0.3):
    """Simula una estela espacial manteniendo los ecos en un buffer persistente."""
    delay_samples = int(44100 * 0.25 * room_size) # Aumentamos el tamaño máximo del espacio
    if delay_samples == 0:
        return samples
        
    n = len(samples)
    # Rotamos el buffer de memoria para hacer espacio a las nuevas muestras
    reverb_buf[:-n] = reverb_buf[n:]
    reverb_buf[-n:] = samples
    
    # Extraemos el eco del pasado basado en el tamaño de la habitación
    echo = reverb_buf[-n - delay_samples : -delay_samples] * (1.0 - damping)
    
    # Mezclamos la guitarra limpia con la estela acumulada
    return (1.0 - wet_mix) * samples + wet_mix * echo

# ── Constantes de audio ───────────────────────────────────────
RATE     = 44100
CHUNK    = 256
DISP_LEN = RATE // 3   # ~0.33 s de historia en la forma de onda

# ── Paleta de colores (dark DAW) ──────────────────────────────
BG_DARK  = '#0d1117'
BG_PANEL = '#161b22'
BG_INPUT = '#21262d'
BORDER   = '#30363d'
TEXT_DIM = '#8b949e'
TEXT_ON  = '#e6edf3'
ACCENT   = '#58a6ff'
GREEN    = '#3fb950'
RED      = '#f78166'
YELLOW   = '#d29922'
PURPLE   = '#d2a8ff'

QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_ON};
}}
QLabel#section-lbl {{
    color: {TEXT_DIM};
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 2px;
}}
QLabel#param-lbl {{
    color: {TEXT_ON};
    font-size: 10px;
}}
QLabel#value-lbl {{
    color: {ACCENT};
    font-size: 10px;
    font-family: monospace;
}}
QLabel#dim-lbl {{
    color: {TEXT_DIM};
    font-size: 10px;
}}
QLabel#header-title {{
    color: {ACCENT};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#header-sub {{
    color: {TEXT_DIM};
    font-size: 10px;
}}
QSlider::groove:horizontal {{
    background: {BG_INPUT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 13px;
    height: 13px;
    border-radius: 6px;
    margin: -5px 0;
}}
QSlider::handle:horizontal:hover {{
    background: #79c0ff;
}}
QSlider::sub-page:horizontal {{
    background: #1c3a5c;
    border-radius: 2px;
}}
QPushButton#start-btn {{
    background: #238636;
    color: #ffffff;
    font-weight: bold;
    font-size: 12px;
    border-radius: 6px;
    padding: 9px 30px;
    border: none;
    min-width: 140px;
}}
QPushButton#start-btn:hover {{ background: #2ea043; }}
QPushButton#start-btn:disabled {{ background: #1a4020; color: #4a7a50; }}
QPushButton#stop-btn {{
    background: #da3633;
    color: #ffffff;
    font-weight: bold;
    font-size: 12px;
    border-radius: 6px;
    padding: 9px 30px;
    border: none;
    min-width: 140px;
}}
QPushButton#stop-btn:hover {{ background: #f85149; }}
QPushButton#stop-btn:disabled {{ background: #4a1515; color: #7a3030; }}
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_ON};
    border: 1px solid {BORDER};
    border-radius: 5px;
    font-size: 10px;
    padding: 4px 8px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_ON};
    selection-background-color: {BG_INPUT};
    border: 1px solid {BORDER};
}}
QFrame#separator {{
    color: {BORDER};
    background-color: {BORDER};
    max-height: 1px;
}}
"""


# ═══════════════════════════════════════════════════════════════
#  Estado compartido (hilo audio ↔ hilo GUI)
# ═══════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.gain           = 0.30
        self.distort_on     = False
        self.distort_gain   = 2.0
        self.distort_thresh = 0.7
        self.delay_on       = False
        self.delay_feedback = 0.4
        self.lowpass_on     = False
        self.lowpass_alpha  = 0.5
        self.wave_buf       = np.zeros(DISP_LEN, dtype=np.float32)
        self.wave_pos       = 0
        self.rms            = 0.0

        # ── NUEVAS VARIABLES PARA V0.4 ─────────────────────────
        self.reverb_on      = False
        self.reverb_size    = 0.60
        self.reverb_damping = 0.45
        self.reverb_wet     = 0.35
        
        self.eq_on          = False
        self.eq_low         = 1.0
        self.eq_mid         = 1.0
        self.eq_high        = 1.0
        
        self.comp_on        = False
        self.comp_thresh    = -20.0
        self.comp_ratio     = 4.0


# ═══════════════════════════════════════════════════════════════
#  Motor de audio (sounddevice callback)
# ═══════════════════════════════════════════════════════════════
class AudioEngine:
    def __init__(self, state: SharedState, in_idx: int, out_idx: int):
        self.state     = state
        self.in_idx    = in_idx
        self.out_idx   = out_idx
        self.stream    = None
        self.delay_buf = np.zeros(CHUNK, dtype=np.float32)

        self.reverb_buf = np.zeros(RATE, dtype=np.float32) # Un segundo completo de memoria

    def start(self):
        try:
            self.stream = sd.Stream(
                samplerate=RATE,
                blocksize=CHUNK,
                dtype='float32',
                channels=(1, 2),
                device=(self.in_idx, self.out_idx),
                callback=self._callback,
                latency='low',
            )
            self.stream.start()
            return True, None
        except Exception as e:
            return False, str(e)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, outdata, frames, time, status):
        s       = self.state
        samples = indata[:, 0].copy()

        with s.lock:
            gain     = s.gain
            dist_on  = s.distort_on
            dgain    = s.distort_gain
            dthresh  = s.distort_thresh
            delay_on = s.delay_on
            dfb      = s.delay_feedback
            lp_on    = s.lowpass_on
            alpha    = s.lowpass_alpha

            # Nuevas variables leídas de forma segura desde el hilo de audio:
            rev_on   = s.reverb_on
            r_size   = s.reverb_size
            r_damp   = s.reverb_damping
            r_wet    = s.reverb_wet
            eq_on    = s.eq_on
            eq_l     = s.eq_low
            eq_m     = s.eq_mid
            eq_h     = s.eq_high
            comp_on  = s.comp_on
            c_thr    = s.comp_thresh
            c_rat    = s.comp_ratio

        # 2. Copia inicial de la señal limpia
        proc = samples.copy()

        # ── CADENA DSP EN CASCADA (AUDIO COMPLETO v0.4) ───────────
        
        # A. Compresor (Primero en la cadena para estabilizar la guitarra)
        if comp_on:
            proc = apply_compressor(proc, c_thr, c_rat)
            
        # B. Ecualizador (Modifica el tono básico)
        if eq_on:
            proc = apply_eq(proc, eq_l, eq_m, eq_h)

        # C. Ganancia Maestra + Distorsión original
        proc = proc * gain
        if dist_on:
            proc = apply_distort(proc, dgain, dthresh)

        # D. Delay original
        if delay_on:
            proc = apply_delay(proc, self.delay_buf, dfb)
            self.delay_buf = proc.copy()

        # E. Filtro pasa-bajos original
        if lp_on:
            filt = np.empty_like(proc)
            prev = proc[0]
            for i in range(len(proc)):
                prev    = alpha * proc[i] + (1.0 - alpha) * prev
                filt[i] = prev
            proc = filt
            
        # F. Reverb (Efecto espacial, va al final antes de la salida)
        if rev_on:
            proc = apply_reverb(proc, self.reverb_buf, r_size, r_damp, r_wet)

        PRE_AMP = 3.5
        proc = proc * PRE_AMP * gain


        # Actualizar buffer circular de visualización gráfica (Líneas originales)
        n   = len(proc)
        end = s.wave_pos + n
        if end <= DISP_LEN:
            s.wave_buf[s.wave_pos:end] = proc
        else:
            first = DISP_LEN - s.wave_pos
            s.wave_buf[s.wave_pos:] = proc[:first]
            s.wave_buf[:end - DISP_LEN] = proc[first:]
        s.wave_pos = end % DISP_LEN

        s.rms = float(np.sqrt(np.mean(proc ** 2)))

        # 4. Enviar señal limpia a ambos audífonos (Salida estéreo)
        outdata[:, 0] = proc
        outdata[:, 1] = proc


# ═══════════════════════════════════════════════════════════════
#  Widgets con QPainter (equivalente a Cairo en GTK)
# ═══════════════════════════════════════════════════════════════
class WaveformWidget(QWidget):
    """Visualizador de forma de onda dibujado con QPainter."""

    def __init__(self, state: SharedState):
        super().__init__()
        self.state = state
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid  = h / 2
        amp  = h / 2 - 8

        # Fondo
        painter.fillRect(0, 0, w, h, QColor(BG_DARK))

        # Líneas de cuadrícula
        grid_pen = QPen(QColor(BORDER))
        grid_pen.setWidthF(0.5)
        painter.setPen(grid_pen)
        for level in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
            y = int(mid - level * amp)
            painter.drawLine(0, y, w, y)

        # Línea cero
        zero_pen = QPen(QColor(TEXT_DIM))
        zero_pen.setWidthF(0.8)
        painter.setPen(zero_pen)
        painter.drawLine(0, int(mid), w, int(mid))

        # Etiquetas dB
        painter.setPen(QColor(TEXT_DIM))
        painter.setFont(QFont('monospace', 7))
        for level, text in ((0.75, '0 dB'), (0.0, '-∞ dB'), (-0.75, '0 dB')):
            y = int(mid - level * amp)
            painter.drawText(4, y + 3, text)

        # Tomar snapshot del buffer (thread-safe)
        with self.state.lock:
            buf = np.roll(self.state.wave_buf.copy(), -self.state.wave_pos)

        if len(buf) == 0:
            return

        step = max(1, len(buf) // w)
        pts  = buf[::step][:w]
        n    = len(pts)

        # Línea de la onda
        wave_pen = QPen(QColor(ACCENT))
        wave_pen.setWidthF(1.0)
        painter.setPen(wave_pen)

        poly = QPolygonF()
        for i in range(n):
            poly.append(QPointF(i * w / n, mid - float(pts[i]) * amp))
        painter.drawPolyline(poly)

        # Área rellena bajo la curva
        fill_color = QColor(ACCENT)
        fill_color.setAlpha(20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill_color)

        fill_poly = QPolygonF()
        fill_poly.append(QPointF(0.0, mid))
        for i in range(n):
            fill_poly.append(QPointF(i * w / n, mid - float(pts[i]) * amp))
        fill_poly.append(QPointF(float(w), mid))
        painter.drawPolygon(fill_poly)


class VUMeterWidget(QWidget):
    """Barra VU con gradiente verde→amarillo→rojo."""

    def __init__(self, state: SharedState):
        super().__init__()
        self.state = state
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(BG_INPUT))

        fill  = min(1.0, self.state.rms * 3.0)
        bar_w = int(fill * w)

        color = QColor(GREEN if fill < 0.6 else YELLOW if fill < 0.85 else RED)
        painter.fillRect(0, 0, bar_w, h, color)

        border_pen = QPen(QColor(BORDER))
        border_pen.setWidthF(0.5)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, w - 1, h - 1)


# ═══════════════════════════════════════════════════════════════
#  Ventana principal
# ═══════════════════════════════════════════════════════════════
class DAWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAW — Audio Digital")
        self.resize(1050, 680)

        self.state   = SharedState()
        self.engine  = None
        self.running = False

        self._build_ui()

        # Actualizar waveform + VU meter a 25 fps
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    # ── Construcción de la UI ──────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._build_header())

        body_w = QWidget()
        body   = QHBoxLayout(body_w)
        body.setContentsMargins(10, 8, 10, 8)
        body.setSpacing(10)
        body.addWidget(self._build_left_panel())

        right_w = QWidget()
        right   = QVBoxLayout(right_w)
        right.setSpacing(8)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self._build_waveform())
        right.addWidget(self._build_vu_meter())
        right.addWidget(self._build_devices())
        right.addWidget(self._build_transport())

        body.addWidget(right_w, stretch=1)
        root.addWidget(body_w, stretch=1)

    # ── Cabecera ───────────────────────────────────────────────
    def _build_header(self):
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {BG_PANEL}; border-bottom: 1px solid {BORDER};"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)

        for text, name in (("⬡", "header-title"), ("DAW", "header-title")):
            lbl = QLabel(text)
            lbl.setObjectName(name)
            layout.addWidget(lbl)

        sub = QLabel("Estación de Trabajo de Audio Digital · Prototipo v0.4")
        sub.setObjectName("header-sub")
        layout.addWidget(sub)
        layout.addStretch()

        self.status_lbl = QLabel("⏹  DETENIDO")
        self.status_lbl.setStyleSheet(
            f"color: {RED}; font-size: 10px; font-weight: bold;"
        )
        layout.addWidget(self.status_lbl)
        return bar

    # ── Panel izquierdo (efectos) ──────────────────────────────
    def _build_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet(
            f"QWidget {{ background-color: {BG_PANEL}; border: 1px solid {BORDER};"
            f" border-radius: 8px; }}"
        )
        panel.setFixedWidth(280)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("CONTROLES"))
        layout.addWidget(self._separator())

        layout.addWidget(self._effect_section(
            "MASTER", None,
            [("Ganancia", 0.0, 6.0, 1.00, 'gain', '{:.2f}')], 
        ))
        layout.addWidget(self._effect_section(
            "DISTORSIÓN", ('distort_on', f"background:#3d1a1a; color:{RED}; border-color:{RED};", "Activar"),
            [("Gain",      1.0, 10.0, 2.0, 'distort_gain',   '{:.1f}'),
             ("Threshold", 0.1,  1.0, 0.7, 'distort_thresh',  '{:.2f}')],
        ))
        layout.addWidget(self._effect_section(
            "DELAY", ('delay_on', f"background:#1c3a5c; color:{ACCENT}; border-color:{ACCENT};", "Activar"),
            [("Feedback", 0.0, 0.9, 0.4, 'delay_feedback', '{:.2f}')],
        ))
        layout.addWidget(self._effect_section(
            "FILTRO PASA-BAJOS", ('lowpass_on', f"background:#271a3d; color:{PURPLE}; border-color:{PURPLE};", "Activar"),
            [("Alpha", 0.05, 1.0, 0.5, 'lowpass_alpha', '{:.2f}')],
        ))

        # ── NUEVOS PANELES V0.4 ────────────────────────────────
        layout.addWidget(self._effect_section(
            "REVERB", ('reverb_on', f"background:#1a3d37; color:#3fb950; border-color:#3fb950;", "Activar"),
            [("Size",    0.1, 1.0, 0.60, 'reverb_size',    '{:.2f}'),
             ("Damping", 0.0, 1.0, 0.45, 'reverb_damping', '{:.2f}'),
             ("Wet Mix", 0.0, 1.0, 0.35, 'reverb_wet',     '{:.2f}')],
        ))

        layout.addWidget(self._effect_section(
            "EQUALIZADOR (EQ)", ('eq_on', f"background:#3d361a; color:{YELLOW}; border-color:{YELLOW};", "Activar"),
            [("Low",  0.1, 2.0, 1.0, 'eq_low',  '{:.2f}'),
             ("Mid",  0.1, 2.0, 1.0, 'eq_mid',  '{:.2f}'),
             ("High", 0.1, 2.0, 1.0, 'eq_high', '{:.2f}')],
        ))

        layout.addWidget(self._effect_section(
            "COMPRESOR", ('comp_on', f"background:#2a1a3d; color:#d2a8ff; border-color:#d2a8ff;", "Activar"),
            [("Threshold", -40.0, 0.0, -20.0, 'comp_thresh', '{:.1f} dB'),
             ("Ratio",       1.0, 8.0,   4.0, 'comp_ratio',  '{:.1f}:1')],
        ))


        layout.addStretch()
        return panel

    def _effect_section(self, title, toggle, rows):
        box    = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fila título + botón toggle
        header_row = QHBoxLayout()
        header_row.addWidget(self._section_label(title))
        header_row.addStretch()

        if toggle:
            key, checked_style, btn_lbl = toggle
            btn = QPushButton(btn_lbl)
            btn.setCheckable(True)
            base_style = (
                f"QPushButton {{ background-color: {BG_INPUT}; color: {TEXT_DIM};"
                f" border-radius: 5px; border: 1px solid {BORDER};"
                f" padding: 4px 12px; font-size: 10px; font-weight: bold; }}"
                f"QPushButton:checked {{ {checked_style} }}"
            )
            btn.setStyleSheet(base_style)
            btn.toggled.connect(lambda checked, k=key: self._set_state(k, checked))
            header_row.addWidget(btn)

        layout.addLayout(header_row)

        # Sliders de parámetros
        for (lbl_text, lo, hi, init, key, fmt) in rows:
            row = QHBoxLayout()
            row.setContentsMargins(4, 0, 0, 0)
            row.setSpacing(4)

            lbl_w = QLabel(lbl_text)
            lbl_w.setObjectName('param-lbl')
            lbl_w.setFixedWidth(80)
            row.addWidget(lbl_w)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(int((init - lo) / (hi - lo) * 1000))
            row.addWidget(sl, stretch=1)

            val_lbl = QLabel(fmt.format(init))
            val_lbl.setObjectName('value-lbl')
            val_lbl.setFixedWidth(38)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val_lbl)

            def on_change(v, lo=lo, hi=hi, k=key, vl=val_lbl, f=fmt):
                real_v = lo + (v / 1000.0) * (hi - lo)
                vl.setText(f.format(real_v))
                self._set_state(k, real_v)

            sl.valueChanged.connect(on_change)
            layout.addLayout(row)

        return box

    def _set_state(self, key, value):
        with self.state.lock:
            setattr(self.state, key, value)

    # ── Visualizador de forma de onda ──────────────────────────
    def _build_waveform(self):
        frame  = QWidget()
        frame.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 8px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(10, 0, 10, 0)
        header.addWidget(self._section_label("FORMA DE ONDA"))
        layout.addLayout(header)

        self.wave_widget = WaveformWidget(self.state)
        layout.addWidget(self.wave_widget)
        return frame

    # ── VU meter ───────────────────────────────────────────────
    def _build_vu_meter(self):
        frame  = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("VU"))

        self.vu_widget = VUMeterWidget(self.state)
        layout.addWidget(self.vu_widget, stretch=1)

        self.vu_lbl = QLabel("0.000")
        self.vu_lbl.setObjectName('value-lbl')
        self.vu_lbl.setFixedWidth(46)
        self.vu_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.vu_lbl)
        return frame

    # ── Selección de dispositivos ──────────────────────────────
    def _build_devices(self):
        frame  = QWidget()
        frame.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 8px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 6, 0, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(10, 0, 10, 0)
        header.addWidget(self._section_label("DISPOSITIVOS DE AUDIO"))
        layout.addLayout(header)

        row = QHBoxLayout()
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(16)

        inputs, outputs = self._get_devices()

        for label_text, devices, attr in (
            ("Entrada:", inputs, 'in_combo'),
            ("Salida:",  outputs, 'out_combo'),
        ):
            col   = QVBoxLayout()
            lbl   = QLabel(label_text)
            lbl.setObjectName('dim-lbl')
            col.addWidget(lbl)
            combo = QComboBox()
            for idx, name in devices:
                combo.addItem(f"{idx}: {name[:42]}", idx)
            setattr(self, attr, combo)
            col.addWidget(combo)
            row.addLayout(col)

        layout.addLayout(row)
        return frame

    def _get_devices(self):
        inputs, outputs = [], []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                inputs.append((i, d['name']))
            if d['max_output_channels'] > 0:
                outputs.append((i, d['name']))
        return inputs, outputs

    # ── Controles de transporte ────────────────────────────────
    def _build_transport(self):
        bar    = QWidget()
        layout = QHBoxLayout(bar)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self.play_btn = QPushButton("▶  INICIAR DSP")
        self.play_btn.setObjectName('start-btn')
        self.play_btn.clicked.connect(self._on_play)
        layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■  DETENER")
        self.stop_btn.setObjectName('stop-btn')
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        return bar

    def _on_play(self):
        in_id  = self.in_combo.currentData()
        out_id = self.out_combo.currentData()
        if in_id is None or out_id is None:
            QMessageBox.critical(self, "Error de Audio",
                                 "Selecciona dispositivos de entrada y salida.")
            return

        self.engine = AudioEngine(self.state, int(in_id), int(out_id))
        ok, err = self.engine.start()
        if not ok:
            QMessageBox.critical(self, "Error de Audio",
                                 f"Error al abrir audio:\n{err}")
            self.engine = None
            return

        self.running = True
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.in_combo.setEnabled(False)
        self.out_combo.setEnabled(False)
        self.status_lbl.setText("▶  ACTIVO")
        self.status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 10px; font-weight: bold;"
        )

    def _on_stop(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.running    = False
        self.state.rms  = 0.0
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.in_combo.setEnabled(True)
        self.out_combo.setEnabled(True)
        self.status_lbl.setText("⏹  DETENIDO")
        self.status_lbl.setStyleSheet(
            f"color: {RED}; font-size: 10px; font-weight: bold;"
        )

    def _tick(self):
        """Refresco de UI a 25 fps (equivale a GLib.timeout_add)."""
        self.wave_widget.update()
        self.vu_widget.update()
        self.vu_lbl.setText(f"{self.state.rms:.3f}")

    def closeEvent(self, event):
        if self.engine:
            self.engine.stop()
        event.accept()

    # ── Helpers UI ─────────────────────────────────────────────
    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName('section-lbl')
        return lbl

    def _separator(self):
        sep = QFrame()
        sep.setObjectName('separator')
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        return sep


# ═══════════════════════════════════════════════════════════════
#  Aplicación
# ═══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = DAWWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
