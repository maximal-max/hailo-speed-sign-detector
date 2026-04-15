#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPI_deploy.py
=============
Raspberry Pi 5 + Hailo-8 + Camera Module 3 (imx708) -- HailoRT 4.20.0
Deploy-Version: Kein Webserver, vollbild GUI-Fenster via OpenCV.

Aufruf:  /usr/bin/python3 RPI_deploy.py

Unterschiede zur RPI_application.py:
  - Kein Flask / HTTP-Webserver / MJPEG-Stream
  - Kein Kamerabild, keine Bounding-Box-Anzeige
  - Nur das erkannte Temposchild wird als PNG im Vollbild angezeigt
  - Start-Disclaimer (10 s) mit Countdown-Balken oben links
"""

import sys
import time
import subprocess
from pathlib import Path
import threading
from collections import Counter, deque
from typing import Optional

import numpy as np
import cv2
from picamera2 import Picamera2
from picamera2.devices.hailo import Hailo


# ================================================================
#  KONFIGURATION  -- hier alle Parameter anpassen
# ================================================================

# Erkennungs-Schwellenwerte
CONFIDENCE_THRESHOLD = 0.45   # Mindest-Konfidenz einer Detektion  (0.0 – 1.0)
IOU_THRESHOLD        = 0.45   # Hinweis: wird vom Hailo-Modell intern verwaltet

# Modell-Pfad  (None = automatische Erkennung aus models/active_hef/)
MODEL_PATH: Optional[str] = None

# Kamera
CAMERA_MODE    = "1280x720@60"   # "1280x720@60" | "1536x864@30" | "800x600@90"
ROI_CROP       = False           # untere 30% abschneiden (für Fahrzeug-Montage)
MAX_EXPOSURE_US   = 4000         # Maximale Belichtungszeit in µs
MAX_ANALOGUE_GAIN = 8.0
NOISE_REDUCTION_MODE = 1
SHARPNESS         = 1.5

# Inferenz
INFER_EVERY_N  = 2   # Jeden N-ten Kamera-Frame durch die NPU schicken
DEBOUNCE_COUNT = 3   # Anzahl übereinstimmender Treffer bis zur Bestätigung

# Modell-spezifische Overrides (nach erkannter Eingangsbreite)
_MODEL_PARAMS: dict = {
    512: {"conf_thresh": 0.45, "infer_every": 1, "debounce": 3},
    640: {"conf_thresh": 0.45, "infer_every": 2, "debounce": 3},
    800: {"conf_thresh": 0.50, "infer_every": 4, "debounce": 4},
}

# Anzeige
FULLSCREEN       = True    # True = Vollbild; False = Fenster (800x480)
WINDOW_W         = 800     # Fensterbreite  (nur relevant wenn FULLSCREEN=False)
WINDOW_H         = 480     # Fensterhöhe    (nur relevant wenn FULLSCREEN=False)
SIGN_DISPLAY_FRACTION = 0.65  # Anteil der Bildschirmhöhe für das Schild-PNG

# Disclaimer
DISCLAIMER_SECONDS = 10   # Anzeigedauer des Hinweistexts beim Start


# ================================================================
#  KONSOLEN-HELPER
# ================================================================

def _print_line(msg: str) -> None:
    print(f"\r\033[2K{msg}", flush=True)


# ================================================================
#  BILDSCHIRMAUFLÖSUNG
# ================================================================

def _get_screen_resolution() -> tuple[int, int]:
    """
    Ermittelt die native Bildschirmauflösung vor dem Öffnen des OpenCV-Fensters.

    Reihenfolge:
      1. tkinter  (sauber, keine sichtbaren Fenster)
      2. xrandr   (Fallback für headless / Wayland ohne tkinter)
      3. fbset    (Framebuffer-Fallback, z.B. ohne X11)
      4. Statischer Fallback 1920×1080
    """
    # --- tkinter (bevorzugt) ---
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    # --- xrandr ---
    try:
        out = subprocess.check_output(["xrandr", "--query"],
                                      stderr=subprocess.DEVNULL,
                                      text=True)
        for line in out.splitlines():
            if " connected" in line and "x" in line:
                # Beispiel: "HDMI-1 connected 1920x1080+0+0 ..."
                for token in line.split():
                    if "x" in token and "+" in token:
                        res = token.split("+")[0]
                        w, h = res.split("x")
                        return int(w), int(h)
    except Exception:
        pass

    # --- fbset (Framebuffer) ---
    try:
        out = subprocess.check_output(["fbset", "-s"],
                                      stderr=subprocess.DEVNULL,
                                      text=True)
        for line in out.splitlines():
            if "geometry" in line:
                parts = line.split()
                return int(parts[1]), int(parts[2])
    except Exception:
        pass

    _print_line("[WARN] Auflösung nicht erkannt -- Fallback 1920x1080")
    return 1920, 1080


# ================================================================
#  MODELL-PFAD  (automatische Erkennung aus ./models/active_hef/)
# ================================================================

def _find_hef_path() -> Path:
    if MODEL_PATH is not None:
        p = Path(MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Konfigurieter MODEL_PATH nicht gefunden: {p}")
        return p
    models_dir = Path(__file__).resolve().parent / "models" / "active_hef"
    hef_files  = sorted(models_dir.glob("*.hef"))
    if not hef_files:
        raise FileNotFoundError(
            f"Keine .hef-Datei in '{models_dir}' gefunden.\n"
            f"Bitte eine .hef-Datei in models/active_hef/ ablegen."
        )
    chosen = hef_files[0]
    if len(hef_files) > 1:
        others = ", ".join(f.name for f in hef_files[1:])
        _print_line(f"[HEF] Mehrere Modelle gefunden -- nutze '{chosen.name}' "
                    f"(ignoriert: {others})")
    return chosen

try:
    HEF_PATH = _find_hef_path()
except FileNotFoundError as _hef_err:
    print(f"[FEHLER] {_hef_err}")
    sys.exit(1)


# ================================================================
#  CPU-TEMPERATUR
# ================================================================

_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def get_cpu_temp() -> float:
    try:
        return int(_THERMAL_PATH.read_text()) / 1000.0
    except OSError:
        return 0.0


# ================================================================
#  KAMERA-MODI
# ================================================================

CAMERA_MODES = {
    "1280x720@60":  (1280, 720, 60),
    "1536x864@30":  (1536, 864, 30),
    "800x600@90":   (800,  600, 90),
}

ROI_TOP_FRACTION = 0.70   # Anteil des Frames (von oben), der bei ROI-Crop bleibt


# ================================================================
#  KLASSEN-DEFINITIONEN
# ================================================================

SIGN_CLASSES = {
    0:  {"name": "Tempolimit_20"},
    1:  {"name": "Tempolimit_30"},
    2:  {"name": "Tempolimit_40"},
    3:  {"name": "Tempolimit_50"},
    4:  {"name": "Tempolimit_60"},
    5:  {"name": "Tempolimit_70"},
    6:  {"name": "Tempolimit_80"},
    7:  {"name": "Tempolimit_90"},
    8:  {"name": "Tempolimit_100"},
    9:  {"name": "Tempolimit_110"},
    10: {"name": "Tempolimit_120"},
    11: {"name": "Tempolimit_130"},
    12: {"name": "Spielstrasse"},
    13: {"name": "Ende_Spielstrasse"},
    14: {"name": "Ortsschild"},
    15: {"name": "Ende_Ortsschild"},
    16: {"name": "Aufhebeschild"},
    17: {"name": "Aufhebeschild_Zahl"},
    18: {"name": "Autobahn"},
    19: {"name": "Ende_Autobahn"},
}


# ================================================================
#  SPEED-SIGN PNG-CACHE  (identisch zu RPI_application.py)
# ================================================================

SPEED_SIGN_FOLDER = Path(__file__).resolve().parent / "datasets" / "application_images_dataset"

_sign_png_cache: dict = {}


def _load_sign_png_composited(limit: int, size: int):
    """Lädt und cached ein Schild-PNG als (afg*a, 1-a)-Tuple für Alpha-Compositing."""
    key = (limit, size)
    if key in _sign_png_cache:
        return _sign_png_cache[key]
    path = SPEED_SIGN_FOLDER / f"{limit}.png"
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
    if raw is None:
        _sign_png_cache[key] = None
        return None
    scaled = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    if scaled.ndim == 3 and scaled.shape[2] == 4:
        a      = scaled[:, :, 3:4].astype(np.float32) / 255.0
        fg     = scaled[:, :, :3].astype(np.float32)
        result = (a * fg, 1.0 - a)
    else:
        result = scaled[:, :, :3]
    _sign_png_cache[key] = result
    return result


def _load_sign_png_bgra(limit: int, size: int) -> Optional[np.ndarray]:
    """
    Lädt ein Schild-PNG als BGR-Bild auf transparentem oder weißem Hintergrund
    zurück (für die eigenständige GUI-Anzeige auf schwarzem Hintergrund).
    Gibt None zurück, wenn kein PNG gefunden wird.
    """
    path = SPEED_SIGN_FOLDER / f"{limit}.png"
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    scaled = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    if scaled.ndim == 3 and scaled.shape[2] == 4:
        return scaled   # BGRA mit Alpha
    return scaled       # BGR ohne Alpha


# ================================================================
#  STATE MACHINE  (identisch zu RPI_application.py)
# ================================================================

class SpeedStateMachine:
    def __init__(self) -> None:
        self._limit: Optional[int] = None
        self._context: str         = "unbekannt"
        self.use_blue_circle: bool = False

    @property
    def current_limit(self) -> Optional[int]:
        return self._limit

    def _set(self, limit: int, blue: bool = False) -> None:
        self._limit          = limit
        self.use_blue_circle = blue

    def update(self, class_id: int) -> None:
        if class_id not in SIGN_CLASSES:
            return
        direct = {0: 20,  1: 30,  2: 40,  3: 50,  4: 60,  5: 70,
                  6: 80,  7: 90,  8: 100, 9: 110, 10: 120, 11: 130}
        if class_id in direct:
            self._set(direct[class_id], blue=False)
            return
        if class_id == 12:
            self._context = "innerorts";  self._set(7, blue=True)
        elif class_id == 13:
            self._context = "innerorts";  self._set(50)
        elif class_id == 14:
            self._context = "innerorts";  self._set(50)
        elif class_id == 15:
            self._context = "ausserorts"; self._set(100)
        elif class_id in (16, 17):
            self._set(100 if self._context == "ausserorts" else 30)
        elif class_id == 18:
            self._context = "ausserorts"; self._set(130, blue=True)
        elif class_id == 19:
            self._context = "ausserorts"; self._set(100)


# ================================================================
#  DETEKTION AUSWAEHLEN  (identisch zu RPI_application.py)
# ================================================================

def select_primary_detection(detections: list,
                              cam_w: int,
                              cam_h: int) -> Optional[dict]:
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]
    frame_area = cam_w * cam_h

    def score(d):
        x1, y1, x2, y2 = d["bbox"]
        area_norm = ((x2 - x1) * (y2 - y1)) / frame_area
        return d["conf"] * area_norm

    return max(detections, key=score)


# ================================================================
#  HAILO INFERENZ  (identisch zu RPI_application.py)
# ================================================================

class SpeedSignDetector:
    def __init__(self, hef_path) -> None:
        hef_path = Path(hef_path)
        if not hef_path.exists():
            raise FileNotFoundError(f"HEF nicht gefunden: {hef_path}")
        self.hailo       = Hailo(str(hef_path), output_type="FLOAT32")
        self.input_shape = self.hailo.get_input_shape()
        self.model_h     = self.input_shape[0]
        self.model_w     = self.input_shape[1]

        # Modell-Parameter aus Mapping ableiten
        params = _MODEL_PARAMS.get(self.model_w)
        if params:
            self.conf_threshold = params["conf_thresh"]
            self.infer_every_n  = params["infer_every"]
            self.debounce_count = params["debounce"]
        else:
            _print_line(f"[WARN] Kein Param-Mapping fuer {self.model_w}px -- "
                        f"Defaults beibehalten.")
            self.conf_threshold = CONFIDENCE_THRESHOLD
            self.infer_every_n  = INFER_EVERY_N
            self.debounce_count = DEBOUNCE_COUNT

    def close(self) -> None:
        try:
            time.sleep(0.3)
            self.hailo.close()
            _print_line("[Hailo] Sauber beendet.")
        except Exception as e:
            if "HAILO_STREAM_ABORT" not in str(e) and "system_error" not in str(e):
                _print_line(f"[Hailo] Beenden: {e}")

    def preprocess(self, frame_bgr: np.ndarray, cam_w: int, cam_h: int,
                   apply_roi_crop: bool = False):
        roi_offset_y = 0
        if apply_roi_crop:
            h_orig    = frame_bgr.shape[0]
            roi_h     = int(h_orig * ROI_TOP_FRACTION)
            frame_bgr = frame_bgr[:roi_h, :]
            roi_offset_y = 0
        h, w  = frame_bgr.shape[:2]
        scale = min(self.model_w / w, self.model_h / h)
        nw    = int(w * scale)
        nh    = int(h * scale)
        pad_x = (self.model_w - nw) // 2
        pad_y = (self.model_h - nh) // 2
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded  = cv2.copyMakeBorder(
            resized,
            pad_y, self.model_h - nh - pad_y,
            pad_x, self.model_w - nw - pad_x,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return img_rgb, scale, pad_x, pad_y, roi_offset_y

    def run(self, img_rgb: np.ndarray):
        return self.hailo.run(img_rgb)

    def postprocess(self, result, orig_w: int, orig_h: int,
                    scale: float, pad_x: int, pad_y: int,
                    roi_offset_y: int = 0) -> list:
        detections = []
        if not isinstance(result, list):
            return detections
        for cls_id, dets in enumerate(result):
            if cls_id not in SIGN_CLASSES:
                continue
            if not hasattr(dets, "__len__") or len(dets) == 0:
                continue
            for det in dets:
                if len(det) < 5:
                    continue
                y1_n, x1_n, y2_n, x2_n, score = (
                    float(det[0]), float(det[1]),
                    float(det[2]), float(det[3]), float(det[4])
                )
                if score < self.conf_threshold:
                    continue
                x1_l = x1_n * self.model_w;  y1_l = y1_n * self.model_h
                x2_l = x2_n * self.model_w;  y2_l = y2_n * self.model_h
                rx1 = int(max(0, min((x1_l - pad_x) / scale, orig_w)))
                ry1 = int(max(0, min((y1_l - pad_y) / scale, orig_h)))
                rx2 = int(max(0, min((x2_l - pad_x) / scale, orig_w)))
                ry2 = int(max(0, min((y2_l - pad_y) / scale, orig_h)))
                ry1 += roi_offset_y
                ry2 += roi_offset_y
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                detections.append({
                    "bbox":     [rx1, ry1, rx2, ry2],
                    "conf":     score,
                    "class_id": cls_id,
                })
        return detections


# ================================================================
#  TEMPORAL DEBOUNCER  (identisch zu RPI_application.py)
# ================================================================

class TemporalDebouncer:
    def __init__(self, buffer_size: int = 5, required_hits: int = 3) -> None:
        self.buffer_size   = buffer_size
        self.required_hits = required_hits
        self.buffer        = deque(maxlen=buffer_size)

    def update(self, class_id: Optional[int]) -> Optional[int]:
        self.buffer.append(class_id)
        counts = Counter(v for v in self.buffer if v is not None)
        if not counts:
            return None
        best_id, best_count = counts.most_common(1)[0]
        return best_id if best_count >= self.required_hits else None

    def get_progress(self, class_id: Optional[int]) -> int:
        if class_id is None:
            return 0
        return sum(1 for v in self.buffer if v == class_id)

    def reset(self) -> None:
        self.buffer.clear()

    def resize(self, new_buffer_size: int) -> None:
        self.buffer_size   = new_buffer_size
        self.required_hits = max(1, int(new_buffer_size * 0.6))
        self.buffer        = deque(self.buffer, maxlen=new_buffer_size)


# ================================================================
#  KAMERA-HARDWARE-PARAMETER
# ================================================================

def _build_camera_controls(frame_us: int) -> dict:
    return {
        "AeEnable":            True,
        "FrameDurationLimits": (frame_us, frame_us),
        "ExposureTime":        MAX_EXPOSURE_US,
        "AnalogueGain":        MAX_ANALOGUE_GAIN,
        "NoiseReductionMode":  NOISE_REDUCTION_MODE,
        "Sharpness":           SHARPNESS,
    }


# ================================================================
#  CAMERA STREAM THREAD  (identisch zu RPI_application.py)
# ================================================================

class CameraStream:
    def __init__(self, cam: Picamera2) -> None:
        self._cam    = cam
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._active = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self._active = True
        self._thread.start()
        while self.read() is None:
            time.sleep(0.01)

    def stop(self) -> None:
        self._active = False
        self._thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        while self._active:
            try:
                frame = self._cam.capture_array()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._frame = np.copy(frame)
            except Exception as e:
                if self._active:
                    _print_line(f"[FEHLER] CameraStream: {e}")
                break

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else np.copy(self._frame)


# ================================================================
#  GUI: DISCLAIMER-ANZEIGE
# ================================================================

DISCLAIMER_TEXT = (
    "Das System kann Fehler aufweisen.",
    "Bitte achten Sie weiterhin",
    "aktiv auf die Verkehrsschilder.",
)

# Farben (BGR)
_BG_COLOR       = (20,  20,  20)   # Dunkler Hintergrund
_TEXT_COLOR     = (220, 220, 220)  # Heller Text
_BAR_BG_COLOR   = (60,  60,  60)  # Ladebalken-Hintergrund
_BAR_FG_COLOR   = (0,   200, 100) # Ladebalken-Vordergrund
_ACCENT_COLOR   = (0,   180, 255) # Akzentfarbe (Countdown-Zahl)


def _show_disclaimer(win_name: str, screen_w: int, screen_h: int) -> None:
    """
    Zeigt den Disclaimer-Bildschirm für DISCLAIMER_SECONDS Sekunden.

    Layout:
      - Hintergrund: dunkelgrau
      - Mitte:       Hinweistext (mehrzeilig)
      - Oben links:  animierter Ladebalken + Countdown-Zahl
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    t_start   = time.monotonic()

    # --- Text-Maße einmalig berechnen ---
    fs_main   = screen_h / 400.0      # Schriftgröße skaliert zur Bildhöhe
    fs_small  = screen_h / 650.0
    thickness = max(1, int(screen_h / 250))

    line_heights = []
    line_widths  = []
    for line in DISCLAIMER_TEXT:
        (tw, th), bl = cv2.getTextSize(line, font, fs_main, thickness)
        line_heights.append(th + bl + int(screen_h * 0.015))
        line_widths.append(tw)

    total_text_h = sum(line_heights)
    start_y      = (screen_h - total_text_h) // 2

    # Ladebalken-Geometrie (oben links, Abstand 3% vom Rand)
    bar_x      = int(screen_w * 0.03)
    bar_y      = int(screen_h * 0.04)
    bar_w      = int(screen_w * 0.25)
    bar_h      = int(screen_h * 0.025)
    bar_y_txt  = bar_y - int(screen_h * 0.01)   # Label über dem Balken

    while True:
        elapsed  = time.monotonic() - t_start
        if elapsed >= DISCLAIMER_SECONDS:
            break
        remaining = DISCLAIMER_SECONDS - elapsed
        progress  = elapsed / DISCLAIMER_SECONDS  # 0.0 → 1.0

        frame = np.full((screen_h, screen_w, 3), _BG_COLOR, dtype=np.uint8)

        # --- Ladebalken (oben links) ---
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      _BAR_BG_COLOR, -1)
        if fill_w > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          _BAR_FG_COLOR, -1)
        # Rahmen um Balken
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      _TEXT_COLOR, 1)

        # Countdown-Zahl rechts neben Balken
        countdown_txt = f"{int(remaining) + 1}s"
        (cw, ch), _  = cv2.getTextSize(countdown_txt, font, fs_small, 1)
        cv2.putText(frame, countdown_txt,
                    (bar_x + bar_w + int(screen_w * 0.012),
                     bar_y + bar_h // 2 + ch // 2),
                    font, fs_small, _ACCENT_COLOR, max(1, thickness - 1),
                    cv2.LINE_AA)

        # --- Hinweistext (mittig) ---
        cur_y = start_y
        for i, line in enumerate(DISCLAIMER_TEXT):
            (tw, _), _ = cv2.getTextSize(line, font, fs_main, thickness)
            tx = (screen_w - tw) // 2
            cv2.putText(frame, line, (tx, cur_y + line_heights[i] - int(screen_h * 0.005)),
                        font, fs_main, _TEXT_COLOR, thickness, cv2.LINE_AA)
            cur_y += line_heights[i]

        cv2.imshow(win_name, frame)
        # ESC bricht den Disclaimer vorzeitig ab
        if cv2.waitKey(33) == 27:
            break


# ================================================================
#  GUI: HAUPT-ANZEIGEFRAME AUFBAUEN
# ================================================================

def _draw_no_limit_screen(frame: np.ndarray, screen_w: int, screen_h: int) -> None:
    """Zeigt einen Platzhalter, wenn noch kein Limit erkannt wurde."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = screen_h / 480.0
    th   = max(1, int(screen_h / 300))
    msg  = "Erkennung laeuft ..."
    (tw, txth), _ = cv2.getTextSize(msg, font, fs, th)
    cv2.putText(frame, msg,
                ((screen_w - tw) // 2, (screen_h + txth) // 2),
                font, fs, (100, 100, 100), th, cv2.LINE_AA)


def _draw_sign_png(frame: np.ndarray, limit: int,
                   screen_w: int, screen_h: int,
                   use_blue_circle: bool = False) -> None:
    """
    Zeichnet das Schild-PNG zentriert auf dem (schwarzen) Frame.
    Nutzt Alpha-Compositing wenn PNG einen Alpha-Kanal hat.
    Fallback: gezeichneter Kreis mit Zahl.
    use_blue_circle: True für blaue Schilder (z.B. Autobahn), False für rote Temposchilder.
    """
    size  = int(screen_h * SIGN_DISPLAY_FRACTION)
    size  = size - (size % 2)   # gerade Zahl für sauberes Centering
    x1    = (screen_w - size) // 2
    y1    = (screen_h - size) // 2
    x2    = x1 + size
    y2    = y1 + size

    png = _load_sign_png_bgra(limit, size)

    if png is not None:
        if png.ndim == 3 and png.shape[2] == 4:
            # Alpha-Compositing auf schwarzem Hintergrund
            a    = png[:, :, 3:4].astype(np.float32) / 255.0
            fg   = png[:, :, :3].astype(np.float32)
            bg   = frame[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (a * fg + (1.0 - a) * bg).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = png[:, :, :3]
    else:
        # Fallback: gezeichneter Kreis
        cx     = screen_w // 2
        cy     = screen_h // 2
        radius = size // 2
        ring_w = max(4, radius // 7)

        ring_color = (0, 0, 220) if use_blue_circle else (220, 80, 0)  # blau für Sonderschilder, rot für Tempolimits
        cv2.circle(frame, (cx, cy), radius, ring_color, -1)
        cv2.circle(frame, (cx, cy), radius - ring_w, (255, 255, 255), -1)

        txt  = str(limit)
        fs   = (size / 200.0) * (0.8 if len(txt) >= 3 else 1.1)
        th   = max(2, int(size / 80))
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, txth), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(frame, txt,
                    (cx - tw // 2, cy + txth // 2),
                    font, fs, (20, 20, 20), th, cv2.LINE_AA)


def _draw_debounce_arc(frame: np.ndarray,
                       debounce_progress: int, debounce_total: int,
                       screen_w: int, screen_h: int) -> None:
    """Zeichnet einen grünen Bogen um das Schild, der den Bestätigungs-Fortschritt zeigt."""
    if debounce_total <= 1 or debounce_progress <= 0:
        return
    size   = int(screen_h * SIGN_DISPLAY_FRACTION)
    size   = size - (size % 2)
    cx     = screen_w // 2
    cy     = screen_h // 2
    radius = size // 2
    arc_r  = radius + max(4, int(screen_h * 0.008))
    arc_t  = max(3, int(screen_h * 0.005))
    angle  = int(360 * min(debounce_progress, debounce_total) / debounce_total)
    cv2.ellipse(frame, (cx, cy), (arc_r, arc_r),
                -90, 0, angle, (0, 220, 120), arc_t)


def build_gui_frame(state: SpeedStateMachine,
                    debounce_progress: int,
                    debounce_total: int,
                    screen_w: int,
                    screen_h: int,
                    fps_inf: float = 0.0,
                    cpu_temp: float = 0.0) -> np.ndarray:
    """
    Erstellt den vollständigen Anzeigeframe für das GUI-Fenster.
    - Schwarzer Hintergrund
    - Zentriertes Schild-PNG (oder Platzhalter)
    - Debounce-Bogen
    - Kleine Statuszeile unten rechts
    """
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    limit = state.current_limit
    if limit is None:
        _draw_no_limit_screen(frame, screen_w, screen_h)
    else:
        _draw_sign_png(frame, limit, screen_w, screen_h, state.use_blue_circle)
        _draw_debounce_arc(frame, debounce_progress, debounce_total,
                           screen_w, screen_h)

    # Statuszeile unten rechts (klein, dezent)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = screen_h / 900.0
    th    = max(1, int(screen_h / 600))
    color = (70, 70, 70)
    temp_str = f"{cpu_temp:.1f}C" if cpu_temp > 0 else "--"
    status   = f"FPS:{fps_inf:.0f}  T:{temp_str}"
    (tw, txth), _ = cv2.getTextSize(status, font, fs, th)
    cv2.putText(frame, status,
                (screen_w - tw - int(screen_w * 0.01),
                 screen_h - int(screen_h * 0.015)),
                font, fs, color, th, cv2.LINE_AA)

    return frame


# ================================================================
#  MAIN
# ================================================================

def main() -> None:
    print("Starte Speed Sign Detector (Deploy-Version) ...")

    # --- Hailo-NPU initialisieren ---
    print("  [1/3] Hailo-NPU ...", end="", flush=True)
    try:
        detector = SpeedSignDetector(HEF_PATH)
    except Exception as e:
        print(f" FEHLER\n[FEHLER] Hailo: {e}")
        sys.exit(1)
    print(" OK")

    # --- Picamera2 initialisieren ---
    print("  [2/3] Picamera2 ...", end="", flush=True)
    cam               = Picamera2()
    cam_w, cam_h, fps = CAMERA_MODES.get(CAMERA_MODE, CAMERA_MODES["1280x720@60"])
    frame_us          = int(1_000_000 / fps)
    cfg = cam.create_video_configuration(
        main={"format": "BGR888", "size": (cam_w, cam_h)},
        controls=_build_camera_controls(frame_us),
        display=None,
        encode=None
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2.0)
    print(" OK")

    # --- CameraStream starten ---
    print("  [3/3] CameraStream ...", end="", flush=True)
    cam_stream = CameraStream(cam)
    cam_stream.start()
    print(" OK")

    # --- Startup-Banner ---
    _params = _MODEL_PARAMS.get(detector.model_w, {})
    print(f"\n{'=' * 57}")
    print(f"  Speed Sign Detector  \u00b7  Deploy-Version")
    print(f"  {'=' * 53}")
    print(f"  Modell    : {HEF_PATH.name:<20}  [{detector.model_w}x{detector.model_h} px]")
    print(f"  Konfidenz : {detector.conf_threshold}   every={detector.infer_every_n}   debounce={detector.debounce_count}")
    print(f"  Kamera    : {cam_w}x{cam_h} @ {fps} fps  |  ROI-Crop: {ROI_CROP}")
    print(f"{'=' * 57}\n")

    # --- Bildschirmauflösung ermitteln (vor Fenster-Öffnung) ---
    if FULLSCREEN:
        screen_w, screen_h = _get_screen_resolution()
        _print_line(f"[Display] Auflösung: {screen_w}x{screen_h} (Vollbild)")
    else:
        screen_w, screen_h = WINDOW_W, WINDOW_H

    # --- OpenCV-Fenster einrichten ---
    WIN_NAME = "SpeedDisplay"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, screen_w, screen_h)
    if FULLSCREEN:
        cv2.setWindowProperty(WIN_NAME,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    # --- Disclaimer anzeigen ---
    print(f"[Disclaimer] Zeige Hinweis für {DISCLAIMER_SECONDS} Sekunden ...")
    _show_disclaimer(WIN_NAME, screen_w, screen_h)

    # --- Haupt-Erkennungsschleife ---
    print("[System] Erkennungsschleife gestartet. ESC zum Beenden.")

    state     = SpeedStateMachine()
    debouncer = TemporalDebouncer(buffer_size=5, required_hits=3)

    inf_times: list             = []
    fps_inf                     = 0.0
    frame_count                 = 0
    last_detections: list       = []
    last_primary: Optional[dict]= None
    debounce_progress           = 0
    last_temp_t                 = 0.0
    cpu_temp                    = 0.0
    last_logged_limit: Optional[int] = None

    try:
        while True:
            t0 = time.perf_counter()

            # CPU-Temperatur alle 2 Sekunden
            if t0 - last_temp_t >= 2.0:
                cpu_temp    = get_cpu_temp()
                last_temp_t = t0

            # Frame holen (nicht-blockierend)
            frame_bgr = cam_stream.read()

            # Thread-Tod erkennen und automatisch neu starten
            if frame_bgr is None:
                if not cam_stream._thread.is_alive():
                    _print_line("[WARN] CameraStream-Thread tot -- starte neu ...")
                    cam_stream = CameraStream(cam)
                    cam_stream.start()
                else:
                    time.sleep(0.001)
                continue

            frame_count += 1

            if debouncer.buffer_size != detector.debounce_count:
                debouncer.resize(detector.debounce_count)

            # Inferenz (jeden N-ten Frame)
            if frame_count % detector.infer_every_n == 0:
                t_inf = time.perf_counter()

                img_rgb, scale, pad_x, pad_y, roi_offset_y = detector.preprocess(
                    frame_bgr, cam_w, cam_h, apply_roi_crop=ROI_CROP)
                result          = detector.run(img_rgb)
                last_detections = detector.postprocess(
                    result, cam_w, cam_h, scale, pad_x, pad_y, roi_offset_y)
                last_primary    = select_primary_detection(
                    last_detections, cam_w, cam_h)

                cid               = last_primary["class_id"] if last_primary else None
                confirmed         = debouncer.update(cid)
                debounce_progress = debouncer.get_progress(cid)

                if confirmed is not None:
                    state.update(confirmed)
                    if state.current_limit != last_logged_limit:
                        name = SIGN_CLASSES.get(confirmed, {}).get("name", "?")
                        _print_line(f"[SCHILD] {name} -> {state.current_limit} km/h")
                        last_logged_limit = state.current_limit

                inf_times.append(time.perf_counter() - t_inf)
                if len(inf_times) > 30:
                    inf_times.pop(0)
                fps_inf = 1.0 / (sum(inf_times) / len(inf_times))

            # GUI-Frame aufbauen und anzeigen
            gui_frame = build_gui_frame(
                state, debounce_progress, detector.debounce_count,
                screen_w, screen_h, fps_inf, cpu_temp
            )
            cv2.imshow(WIN_NAME, gui_frame)

            # ESC beendet die Schleife
            if cv2.waitKey(1) == 27:
                break

            # Konsolenstatus
            _limit = state.current_limit
            _line  = (f"FPS-Inf: {fps_inf:.0f}"
                      f"  |  Temp: {cpu_temp:.1f}°C"
                      f"  |  Schild: {f'{_limit} km/h' if _limit else '---'}"
                      f"  |  Det: {len(last_detections)}")
            print(f"\r\033[2K{_line}", end="", flush=True)

    except KeyboardInterrupt:
        pass

    finally:
        print(f"\r{'':<90}", flush=True)
        print("[System] Beende ...")
        cv2.destroyAllWindows()
        try:
            cam_stream.stop()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        detector.close()
        print("[System] Sauber beendet.")


if __name__ == "__main__":
    main()
