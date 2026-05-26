"""
DAW — Estación de Trabajo de Audio Digital
GUI v2.1 — Ondas y efectos corregidos
"""

import tkinter as tk
import threading
import numpy as np
import queue
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.engine.processor import AudioEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("[AVISO] No se pudo importar AudioEngine. Modo demo activo.")

# ═══════════════════════════════════════════════════════════════════════════════
# PALETA DE COLORES
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":            "#0d0f14",
    "panel":         "#13161e",
    "panel2":        "#1a1e28",
    "border":        "#252a38",
    "border_light":  "#2e3447",
    "accent":        "#00e5a0",
    "accent2":       "#00b8ff",
    "danger":        "#ff4466",
    "warn":          "#ffb020",
    "text":          "#e8ecf4",
    "text_dim":      "#6b7590",
    "text_mid":      "#9aa3bc",
    "slider_trough": "#1e2230",
    "btn_on":        "#00e5a0",
    "btn_on_fg":     "#0d0f14",
    "btn_off":       "#1e2230",
    "btn_off_fg":    "#6b7590",
    "vu_green":      "#00e5a0",
    "vu_yellow":     "#ffb020",
    "vu_red":        "#ff4466",
    "wave_in":       "#00e5a0",
    "wave_out":      "#00b8ff",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DEMO  (sin hardware)
# ═══════════════════════════════════════════════════════════════════════════════
class DummyEngine:
    def __init__(self):
        self.master_gain = 0.30
        self.comp_on  = False; self.c_thr = -20.0; self.c_rat = 4.0
        self.eq_on    = False; self.eq_l  = 1.0;   self.eq_m  = 1.0; self.eq_h = 1.0
        self.gain     = 1.0
        self.dist_on  = False; self.dgain = 2.0;   self.dthresh = 0.5
        self.delay_on = False; self.dfb   = 0.5
        self.delay_buf = np.zeros(44100)
        self.lp_on    = False; self.alpha = 0.5
        self._running = False
        self.gui_queue = queue.Queue(maxsize=4)

    def start(self):
        self._running = True
        threading.Thread(target=self._generate, daemon=True).start()
        return True, None

    def _generate(self):
        t = 0.0
        while self._running:
            n = 512
            x = np.linspace(t, t + 0.012, n)
            raw = (0.45 * np.sin(2 * np.pi * 82 * x)
                 + 0.20 * np.sin(2 * np.pi * 246 * x)
                 + 0.10 * np.sin(2 * np.pi * 440 * x)
                 + 0.04 * np.random.randn(n))
            raw = np.clip(raw, -1.0, 1.0).astype(np.float32)

            proc = raw.copy() * self.master_gain * self.gain
            if self.dist_on:
                proc = np.clip(proc * self.dgain, -self.dthresh, self.dthresh)
            if self.lp_on:
                proc = np.convolve(proc, np.ones(5) / 5, mode="same")
            proc = np.clip(proc, -1.0, 1.0).astype(np.float32)

            try:
                self.gui_queue.put_nowait((raw, proc))
            except queue.Full:
                pass
            t += 0.012
            time.sleep(0.033)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGET: SLIDER NEÓN
# ═══════════════════════════════════════════════════════════════════════════════
class NeonSlider(tk.Canvas):
    def __init__(self, parent, from_=0.0, to=1.0, value=0.5,
                 width=180, height=20, command=None, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C["panel2"], highlightthickness=0, **kw)
        self.from_ = from_; self.to = to
        self._value = value; self.command = command
        self.W = width;      self.H = height
        self.bind("<ButtonPress-1>",   self._click)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self._draw()

    def _norm(self):
        return (self._value - self.from_) / (self.to - self.from_)

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete("all")
        pad = 8; cy = self.H // 2
        self._rrect(pad, cy-3, self.W-pad, cy+3, 3, fill=C["slider_trough"], outline="")
        filled = pad + self._norm() * (self.W - 2*pad)
        if filled > pad:
            self._rrect(pad, cy-3, filled, cy+3, 3, fill=C["accent"], outline="")
        tx = pad + self._norm() * (self.W - 2*pad); r = 7
        self.create_oval(tx-r, cy-r, tx+r, cy+r, fill=C["panel"], outline=C["accent"], width=2)
        self.create_oval(tx-3, cy-3, tx+3, cy+3, fill=C["accent"], outline="")

    def _pos_val(self, x):
        pad = 8
        return self.from_ + max(0.0, min(1.0, (x-pad)/(self.W-2*pad))) * (self.to-self.from_)

    def _click(self, e):
        self._value = self._pos_val(e.x); self._draw()
        if self.command: self.command(self._value)
    def _drag(self, e):
        self._value = self._pos_val(e.x); self._draw()
        if self.command: self.command(self._value)
    def _release(self, e): pass
    def get(self): return self._value
    def set(self, v):
        self._value = max(self.from_, min(self.to, v)); self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGET: BOTÓN TOGGLE
# ═══════════════════════════════════════════════════════════════════════════════
class ToggleBtn(tk.Canvas):
    def __init__(self, parent, command=None, **kw):
        super().__init__(parent, width=72, height=22,
                         bg=C["panel2"], highlightthickness=0, **kw)
        self.state = False; self.command = command
        self.bind("<ButtonPress-1>", self._toggle); self._draw()

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete("all")
        bg = C["btn_on"] if self.state else C["btn_off"]
        fg = C["btn_on_fg"] if self.state else C["btn_off_fg"]
        self._rrect(1, 1, 71, 21, 5, fill=bg, outline=C["border_light"])
        label = "● ON" if self.state else "○ OFF"
        self.create_text(36, 11, text=label, fill=fg, font=("Consolas", 8, "bold"))

    def _toggle(self, e=None):
        self.state = not self.state; self._draw()
        if self.command: self.command(self.state)

    def set(self, v): self.state = bool(v); self._draw()
    def get(self): return self.state


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGET: VU METER
# ═══════════════════════════════════════════════════════════════════════════════
class VUMeter(tk.Canvas):
    def __init__(self, parent, width=400, height=14, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C["panel2"], highlightthickness=0, **kw)
        self.W = width; self.H = height
        self._level = 0.0; self._peak = 0.0; self._hold = 0
        self._draw()

    def _draw(self):
        self.delete("all")
        pad = 2; w = self.W - pad*2; segs = 40; sw = w / segs
        for i in range(segs):
            norm = i / segs; filled = norm < self._level
            if norm < 0.6:   col = C["vu_green"]  if filled else "#0d2018"
            elif norm < 0.8: col = C["vu_yellow"] if filled else "#1a1200"
            else:             col = C["vu_red"]    if filled else "#1a0008"
            self.create_rectangle(pad + i*sw + 1, pad,
                                  pad + (i+1)*sw - 1, self.H-pad,
                                  fill=col, outline="")
        if self._peak > 0:
            px = pad + self._peak * w
            self.create_rectangle(px-1, pad, px+1, self.H-pad, fill="white", outline="")

    def set_level(self, level):
        self._level = max(0.0, min(1.0, level))
        if self._level > self._peak:
            self._peak = self._level; self._hold = 50
        else:
            if self._hold > 0: self._hold -= 1
            else: self._peak = max(0.0, self._peak - 0.008)
        self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGET: FORMA DE ONDA
# ═══════════════════════════════════════════════════════════════════════════════
class Waveform(tk.Canvas):
    def __init__(self, parent, label="", color=C["wave_in"],
                 width=500, height=80, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C["panel"], highlightthickness=0, **kw)
        self.W = width; self.H = height
        self.label = label; self.color = color
        self._data = np.zeros(width)
        self._draw()

    def _draw(self):
        self.delete("all")
        cy = self.H // 2
        # Líneas de referencia
        self.create_line(0, cy, self.W, cy, fill=C["border"], width=1)
        for yo in [self.H//4, -self.H//4]:
            self.create_line(0, cy+yo, self.W, cy+yo,
                             fill=C["border"], width=1, dash=(2, 6))
        # Forma de onda
        if len(self._data) > 1:
            pts = []
            for x in range(self.W):
                idx = min(int(x * len(self._data) / self.W), len(self._data)-1)
                y = cy - self._data[idx] * (cy - 4)
                pts.extend([x, y])
            if len(pts) >= 4:
                fill_pts = [0, cy] + pts + [self.W, cy]
                # stipple en lugar de alpha hex (Tkinter no soporta #RRGGBBAA)
                self.create_polygon(fill_pts, fill=self.color,
                                    outline="", stipple="gray25")
                self.create_line(pts, fill=self.color, width=1.5, smooth=False)
        # Etiqueta
        self.create_text(8, 8, text=self.label, anchor="nw",
                         fill=self.color, font=("Consolas", 8))

    def update_data(self, data):
        if len(data) > 0:
            idx = np.linspace(0, len(data)-1, self.W).astype(int)
            self._data = np.clip(data[idx], -1.0, 1.0)
        self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# GUI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class DAW_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DAW — Estación de Trabajo de Audio Digital  ·  v2.1")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)

        self.engine = AudioEngine() if ENGINE_AVAILABLE else DummyEngine()
        # Aseguramos que el engine siempre tenga una cola GUI
        if not hasattr(self.engine, "gui_queue"):
            self.engine.gui_queue = queue.Queue(maxsize=4)

        self._running = False

        # Buffers circulares para las formas de onda (750 muestras = ancho del canvas)
        self._buf_in  = np.zeros(750)
        self._buf_out = np.zeros(750)

        self._build_ui()
        self._update_loop()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # HEADER
        hdr = tk.Frame(self.root, bg=C["panel"], height=46)
        hdr.pack(fill="x", side="top"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◉  DAW", bg=C["panel"], fg=C["accent"],
                 font=("Consolas", 16, "bold")).pack(side="left", padx=14)
        tk.Label(hdr, text="Estación de Trabajo de Audio Digital  ·  v2.1",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Consolas", 9)).pack(side="left", padx=4)
        self.status_dot = tk.Label(hdr, text="● OFFLINE", bg=C["panel"],
                                   fg=C["text_dim"], font=("Consolas", 9, "bold"))
        self.status_dot.pack(side="right", padx=16)
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # PANEL IZQUIERDO
        left = tk.Frame(body, bg=C["panel"], width=270,
                        highlightbackground=C["border"], highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 8)); left.pack_propagate(False)
        self._build_controls(left)

        # PANEL DERECHO
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_waveform(right)
        self._build_vu(right)
        self._build_devices(right)
        self._build_transport(right)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _sec(self, parent, text):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", padx=8, pady=(10, 2))
        tk.Frame(f, bg=C["accent"], width=3, height=14).pack(side="left", padx=(0,6))
        tk.Label(f, text=text, bg=C["panel"], fg=C["text"],
                 font=("Consolas", 8, "bold")).pack(side="left")
        return f

    def _slider_row(self, parent, label, attr, from_, to, init, fmt):
        """Fila con label + NeonSlider + valor. Enlaza directo al engine."""
        f = tk.Frame(parent, bg=C["panel2"]); f.pack(fill="x", padx=8, pady=2)
        tk.Label(f, text=label, bg=C["panel2"], fg=C["text_mid"],
                 font=("Consolas", 8), width=10, anchor="w").pack(side="left", padx=(6,2))
        vv = tk.StringVar(value=fmt.format(init))
        tk.Label(f, textvariable=vv, bg=C["panel2"], fg=C["accent"],
                 font=("Consolas", 8), width=7, anchor="e").pack(side="right", padx=(0,6))
        def cb(v, a=attr, s=fmt, lv=vv):
            if a: setattr(self.engine, a, v)
            lv.set(s.format(v))
        NeonSlider(f, from_=from_, to=to, value=init, command=cb).pack(
            side="left", fill="x", expand=True, padx=4)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        tk.Label(parent, text="CONTROLES", bg=C["panel"], fg=C["text_dim"],
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(10,4))
        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", padx=8)

        # MASTER
        self._sec(parent, "MASTER")
        self._slider_row(parent, "Ganancia", "master_gain", 0.0, 1.0, 0.30, "{:.2f}")

        # DISTORSIÓN
        sf = self._sec(parent, "DISTORSIÓN")
        ToggleBtn(sf, command=lambda v: setattr(self.engine, "dist_on", v)).pack(side="right")
        self._slider_row(parent, "Gain",      "dgain",   1.0, 10.0, 2.0,  "{:.1f}")
        self._slider_row(parent, "Threshold", "dthresh", 0.1,  1.0, 0.5,  "{:.2f}")

        # DELAY
        sd = self._sec(parent, "DELAY")
        ToggleBtn(sd, command=lambda v: setattr(self.engine, "delay_on", v)).pack(side="right")
        self._slider_row(parent, "Feedback", "dfb", 0.0, 0.95, 0.5, "{:.2f}")

        # FILTRO PASA-BAJOS
        slp = self._sec(parent, "FILTRO PASA-BAJOS")
        ToggleBtn(slp, command=lambda v: setattr(self.engine, "lp_on", v)).pack(side="right")
        self._slider_row(parent, "Alpha", "alpha", 0.01, 1.0, 0.5, "{:.2f}")

        # REVERB  (atributos opcionales en el engine)
        srv = self._sec(parent, "REVERB")
        ToggleBtn(srv).pack(side="right")  # reverb_on no está en el engine base aún
        self._slider_row(parent, "Size",    None, 0.1, 1.0, 0.60, "{:.2f}")
        self._slider_row(parent, "Damping", None, 0.0, 1.0, 0.45, "{:.2f}")
        self._slider_row(parent, "Wet Mix", None, 0.0, 1.0, 0.25, "{:.2f}")

        # ECUALIZADOR
        seq = self._sec(parent, "EQUALIZADOR (EQ)")
        ToggleBtn(seq, command=lambda v: setattr(self.engine, "eq_on", v)).pack(side="right")

        for lbl, attr, init in [("Low", "eq_l", 1.0), ("Mid", "eq_m", 0.8), ("High", "eq_h", 1.2)]:
            f = tk.Frame(parent, bg=C["panel2"]); f.pack(fill="x", padx=8, pady=2)
            tk.Label(f, text=lbl, bg=C["panel2"], fg=C["text_mid"],
                     font=("Consolas", 8), width=10, anchor="w").pack(side="left", padx=(6,2))
            vv = tk.StringVar(value="{:+.1f}dB".format(20*np.log10(max(init,0.001))))
            tk.Label(f, textvariable=vv, bg=C["panel2"], fg=C["accent"],
                     font=("Consolas", 8), width=7, anchor="e").pack(side="right", padx=(0,6))
            def _cb(v, a=attr, lv=vv):
                setattr(self.engine, a, v)
                lv.set("{:+.1f}dB".format(20*np.log10(max(v, 0.001))))
            NeonSlider(f, from_=0.0, to=3.0, value=init, command=_cb).pack(
                side="left", fill="x", expand=True, padx=4)

        # COMPRESOR
        sco = self._sec(parent, "COMPRESOR")
        ToggleBtn(sco, command=lambda v: setattr(self.engine, "comp_on", v)).pack(side="right")
        self._slider_row(parent, "Threshold", "c_thr", -60.0,  0.0, -20.0, "{:.0f} dB")
        self._slider_row(parent, "Ratio",     "c_rat",   1.0, 20.0,   4.0, "{:.1f}:1")
        self._slider_row(parent, "Attack",    None,      1.0, 200.0,  15.0, "{:.0f} ms")
        self._slider_row(parent, "Release",   None,     10.0,2000.0, 100.0, "{:.0f} ms")

        tk.Frame(parent, bg=C["panel"]).pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_waveform(self, parent):
        wf = tk.Frame(parent, bg=C["panel"],
                      highlightbackground=C["border"], highlightthickness=1)
        wf.pack(fill="x", pady=(0, 6))
        tk.Label(wf, text="FORMA DE ONDA", bg=C["panel"], fg=C["text_dim"],
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(8,4))
        tk.Frame(wf, bg=C["border"], height=1).pack(fill="x", padx=8)

        inner = tk.Frame(wf, bg=C["panel"]); inner.pack(fill="x", padx=8, pady=6)
        for label, color, attr in [
            ("Entrada Directa",  C["wave_in"],  "_wave_in"),
            ("Salida Procesada", C["wave_out"], "_wave_out"),
        ]:
            row = tk.Frame(inner, bg=C["panel"]); row.pack(fill="x", pady=3)
            lc = tk.Frame(row, bg=C["panel"], width=38); lc.pack(side="left"); lc.pack_propagate(False)
            for db in ["0 dB", "", "-0 dB"]:
                tk.Label(lc, text=db, bg=C["panel"], fg=C["text_dim"],
                         font=("Consolas", 7)).pack(anchor="e")
            w = Waveform(row, label=label, color=color, width=750, height=72)
            w.pack(side="left", padx=(4, 0))
            setattr(self, attr, w)

    def _build_vu(self, parent):
        vuf = tk.Frame(parent, bg=C["panel2"],
                       highlightbackground=C["border"], highlightthickness=1)
        vuf.pack(fill="x", pady=(0, 6))
        inner = tk.Frame(vuf, bg=C["panel2"]); inner.pack(fill="x", padx=10, pady=6)
        self._vu_meters = []
        for lbl, tag in [("VU", "in_l"), ("VR", "in_r"), ("VU", "out")]:
            row = tk.Frame(inner, bg=C["panel2"]); row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, bg=C["panel2"], fg=C["text_dim"],
                     font=("Consolas", 8), width=3).pack(side="left")
            m = VUMeter(row, width=680, height=14); m.pack(side="left", padx=4)
            vv = tk.StringVar(value="0.000")
            tk.Label(row, textvariable=vv, bg=C["panel2"], fg=C["text_mid"],
                     font=("Consolas", 8), width=6).pack(side="left")
            self._vu_meters.append((m, vv))

    def _build_devices(self, parent):
        df = tk.Frame(parent, bg=C["panel"],
                      highlightbackground=C["border"], highlightthickness=1)
        df.pack(fill="x", pady=(0, 6))
        tk.Label(df, text="DISPOSITIVOS DE AUDIO", bg=C["panel"], fg=C["text_dim"],
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(8,4))
        tk.Frame(df, bg=C["border"], height=1).pack(fill="x", padx=8)
        row = tk.Frame(df, bg=C["panel"]); row.pack(fill="x", padx=10, pady=6)
        for side, lbl, default in [
            ("left",  "Entrada:", "1: Focusrite Scarlett Solo (3rd Gen)"),
            ("right", "Salida:",  "8: Realtek Speakers (High Def Audio)"),
        ]:
            col = tk.Frame(row, bg=C["panel"]); col.pack(side=side, fill="x", expand=True, padx=4)
            tk.Label(col, text=lbl, bg=C["panel"], fg=C["text_dim"],
                     font=("Consolas", 8)).pack(anchor="w")
            e = tk.Entry(col, bg=C["panel2"], fg=C["text"], insertbackground=C["accent"],
                         font=("Consolas", 9), relief="flat",
                         highlightbackground=C["border"], highlightthickness=1)
            e.insert(0, default); e.pack(fill="x", ipady=4, pady=(2,0))

    def _build_transport(self, parent):
        tp = tk.Frame(parent, bg=C["bg"]); tp.pack(fill="x", pady=4)
        self._btn_start = tk.Button(
            tp, text="▶  INICIAR DSP", bg=C["accent"], fg=C["bg"],
            font=("Consolas", 11, "bold"), relief="flat", cursor="hand2",
            padx=28, pady=8, activebackground="#00c080", command=self._start_dsp)
        self._btn_start.pack(side="left", padx=6)
        self._btn_stop = tk.Button(
            tp, text="■  DETENER", bg=C["danger"], fg="white",
            font=("Consolas", 11, "bold"), relief="flat", cursor="hand2",
            padx=28, pady=8, activebackground="#cc0033", state="disabled",
            command=self._stop_dsp)
        self._btn_stop.pack(side="left", padx=6)
        self._status_lbl = tk.Label(tp, text="● Motor detenido",
                                    bg=C["bg"], fg=C["text_dim"], font=("Consolas", 9))
        self._status_lbl.pack(side="left", padx=12)

    # ──────────────────────────────────────────────────────────────────────────
    def _start_dsp(self):
        self._status_lbl.config(text="⟳ Iniciando motor...", fg=C["warn"])
        self.root.update()

        # Parchear _callback del engine real para alimentar la gui_queue
        if ENGINE_AVAILABLE and isinstance(self.engine, AudioEngine):
            _orig = self.engine._callback
            _q    = self.engine.gui_queue
            def _patched(indata, outdata, frames, time_info, status):
                _orig(indata, outdata, frames, time_info, status)
                try:
                    _q.put_nowait((indata[:, 1].copy(), outdata[:, 0].copy()))
                except queue.Full:
                    pass
            self.engine._callback = _patched

        ok, err = self.engine.start()
        if ok:
            self._running = True
            self._btn_start.config(state="disabled")
            self._btn_stop.config(state="normal")
            self._status_lbl.config(text="● Motor DSP activo", fg=C["accent"])
            self.status_dot.config(text="● EN LÍNEA", fg=C["accent"])
        else:
            self._status_lbl.config(
                text=f"✗ Error: {(err or 'desconocido')[:60]}", fg=C["danger"])

    def _stop_dsp(self):
        self._running = False
        self.engine.stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_lbl.config(text="● Motor detenido", fg=C["text_dim"])
        self.status_dot.config(text="● OFFLINE", fg=C["text_dim"])

    # ──────────────────────────────────────────────────────────────────────────
    def _update_loop(self):
        """Consume la gui_queue y actualiza ondas + VU meters."""
        raw_d = proc_d = None
        q = getattr(self.engine, "gui_queue", None)
        if q:
            while True:         # vaciamos; nos quedamos con el frame más reciente
                try:
                    raw_d, proc_d = q.get_nowait()
                except queue.Empty:
                    break

        if raw_d is not None and len(raw_d) > 0:
            n = len(raw_d)
            # Buffer circular → scroll continuo de la onda
            self._buf_in  = np.roll(self._buf_in,  -n); self._buf_in[-n:]  = raw_d
            self._buf_out = np.roll(self._buf_out, -n); self._buf_out[-n:] = proc_d
            self._wave_in.update_data(self._buf_in)
            self._wave_out.update_data(self._buf_out)

            lvl_in  = min(1.0, float(np.sqrt(np.mean(raw_d**2)))  * 3.0)
            lvl_out = min(1.0, float(np.sqrt(np.mean(proc_d**2))) * 3.0)
        else:
            lvl_in = lvl_out = 0.0

        for i, (meter, vv) in enumerate(self._vu_meters):
            lvl = lvl_in if i < 2 else lvl_out
            meter.set_level(lvl)
            vv.set(f"{lvl:.3f}")

        self.root.after(33, self._update_loop)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.geometry("1100x720")
    icon = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
    if os.path.exists(icon):
        try: root.iconbitmap(icon)
        except: pass
    DAW_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()