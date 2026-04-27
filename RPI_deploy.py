#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPI_deploy.py  —  Speed Sign Detector  |  Produktion
=====================================================
Raspberry Pi 5 + Hailo-8 + Camera Module 3 (IMX708)  —  HailoRT 4.20.0

Ausgabe:   direkt auf HDMI-Display, kein Webserver.
           Die Bildschirmaufloesung wird automatisch erkannt (fbset / xrandr).
Umschalten: TAB-Taste oder Klick auf Toggle-Button oben rechts.

Modi:
  SIGN  — Vollbild-Schildanzeige (schwarzer Hintergrund, Schild-PNG zentriert)
  CAM   — Kamerabild mit Bounding Boxes + Einstellungs-Panel (untere 120 px)

Einstellungen (CAM-Modus, Mausklick oder Tastatur):
  Konfidenz · Infer-N · Debounce · Min. Schildgroesse (px) · ROI-Crop
  Kameramodus-Wechsel (1280x720@60 <-> 800x600@90)

Tastatur (global):
  TAB  — Modus wechseln (SIGN <-> CAM)
  ESC  — Beenden

Tastatur (nur im CAM-Modus):
  Q / W  — Konfidenz         -/+  (Schritt 0.05)
  A / S  — Infer-N           -/+  (jeden N-ten Frame inferieren)
  Y / X  — Debounce          -/+  (Bestaedigungsframes)
  U / I  — Min. Schildgroesse -/+  (Schritt 5 px)
  R      — ROI-Crop           umschalten
  K      — Kameramodus        wechseln (1280x720 <-> 800x600)
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ================================================================
#  ANZEIGE-KONFIGURATION
# ================================================================

DISPLAY_W             = 480
DISPLAY_H             = 320
CAM_VIEW_H            = 200    # Kamerabild-Hoehe im CAM-Modus
FULLSCREEN            = True
WIN_NAME              = "SpeedDisplay"
SIGN_DISPLAY_FRACTION = 0.72   # Anteil der Bildschirmhoehe fuer Schild-PNG
DISCLAIMER_SECONDS    = 10

# Werden in main() auf die echte Bildschirmaufloesung gesetzt
_screen_w = DISPLAY_W
_screen_h = DISPLAY_H


# ================================================================
#  KAMERA
# ================================================================

CAMERA_MODES = {
    "1280x720@60": (1280, 720, 60),
    "800x600@90":  (800,  600, 90),
}
DEFAULT_MODE     = "1280x720@60"
ROI_TOP_FRACTION = 0.70

MAX_EXPOSURE_US      = 4000
MAX_ANALOGUE_GAIN    = 8.0
NOISE_REDUCTION_MODE = 1
SHARPNESS            = 1.5


# ================================================================
#  INFERENZ-DEFAULTS
# ================================================================

CONF_THRESHOLD = 0.50
INFER_EVERY_N  = 4
DEBOUNCE_COUNT = 4
MIN_SIGN_PX    = 20   # Mindestbreite/-hoehe der BBox in Kamera-Pixeln

_MODEL_PARAMS: dict = {
    512: {"conf_thresh": 0.45, "infer_every": 1, "debounce": 3},
    640: {"conf_thresh": 0.45, "infer_every": 2, "debounce": 3},
    800: {"conf_thresh": 0.50, "infer_every": 4, "debounce": 4},
}


# ================================================================
#  LAUFZEIT-PARAMETER  (thread-safe)
# ================================================================

_runtime_lock = threading.Lock()
_runtime: dict = {
    "camera_mode":  DEFAULT_MODE,
    "conf_thresh":  CONF_THRESHOLD,
    "infer_every":  INFER_EVERY_N,
    "debounce":     DEBOUNCE_COUNT,
    "min_sign_px":  MIN_SIGN_PX,
    "roi_crop":     False,
    "display_mode": "SIGN",     # "SIGN" | "CAM"
    "_mode_change": False,
    "hef_name":     "–",
    "model_res":    "–x–",
    "cpu_temp":     0.0,
    "fps_cam":      0.0,
    "fps_inf":      0.0,
    "n_det":        0,
}


def get_runtime(key):
    with _runtime_lock:
        return _runtime[key]


def set_runtime(key, value):
    with _runtime_lock:
        _runtime[key] = value


# ================================================================
#  KONSOLEN-HELPER
# ================================================================

_STATUS_LINE_LEN = 90


def _print_line(msg: str) -> None:
    print(f"\r\033[2K{msg}", flush=True)


# ================================================================
#  BILDSCHIRMAUFLOESUNG ERKENNEN
# ================================================================

def _detect_screen_size(fallback_w: int, fallback_h: int) -> tuple:
    """Erkennt die echte Bildschirmaufloesung (fbset oder xrandr)."""
    try:
        out = subprocess.run(
            ["fbset", "-s"], capture_output=True, text=True, timeout=2
        ).stdout
        for line in out.splitlines():
            if "geometry" in line:
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[1]), int(parts[2])
    except Exception:
        pass
    try:
        import re
        out = subprocess.run(
            ["xrandr"], capture_output=True, text=True, timeout=2
        ).stdout
        for line in out.splitlines():
            m = re.search(r"(\d+)x(\d+).*\*", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return fallback_w, fallback_h


def _fit_to_screen(frame: np.ndarray) -> np.ndarray:
    """Skaliert den Content-Frame auf die echte Bildschirmaufloesung."""
    if _screen_w == DISPLAY_W and _screen_h == DISPLAY_H:
        return frame
    return cv2.resize(frame, (_screen_w, _screen_h), interpolation=cv2.INTER_LINEAR)


# ================================================================
#  CURSOR-STEUERUNG
# ================================================================

def _set_cursor_visible(visible: bool) -> None:
    """Zeigt oder versteckt den Mauszeiger (Xlib/XFixes, Fallback: xdotool)."""
    # Methode 1: Xlib XFixes (python3-xlib muss installiert sein)
    try:
        from Xlib import display as xdisp
        from Xlib.ext import xfixes as xf
        d = xdisp.Display()
        root = d.screen().root
        if visible:
            xf.show_cursor(d, root)
        else:
            xf.hide_cursor(d, root)
        d.sync()
        d.close()
        return
    except Exception:
        pass
    # Methode 2: xdotool – Cursor ausserhalb des Bildschirms verschieben
    try:
        if visible:
            pos = [str(_screen_w // 2), str(_screen_h // 2)]
        else:
            pos = ["99999", "99999"]
        subprocess.run(["xdotool", "mousemove"] + pos,
                       capture_output=True, timeout=1)
    except Exception:
        pass


# ================================================================
#  MODELL-PFAD
# ================================================================

def _find_hef_path() -> Path:
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
        _print_line(f"[HEF] Mehrere Modelle -- nutze '{chosen.name}' (ignoriert: {others})")
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
#  KLASSEN-DEFINITIONEN & FARBEN
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

BBOX_COLORS = {
    **{i: (0, 200, 0) for i in range(12)},
    12: (0,  128, 255),
    13: (0,  220, 255),
    14: (0,    0, 255),
    15: (200,  0, 255),
    16: (255, 220,  0),
    17: (255, 100,  0),
    18: (255,  50,  50),
    19: (128,   0, 128),
}


# ================================================================
#  SPEED-SIGN PNG-CACHE
# ================================================================

SPEED_SIGN_FOLDER = Path(__file__).resolve().parent / "datasets" / "application_images_dataset"
_sign_png_cache: dict = {}


def _load_sign_png_composited(limit: int, size: int):
    """Gibt (afg*a, 1-a)-Tuple fuer Alpha-Compositing-Overlay zurueck."""
    key = (limit, size)
    if key in _sign_png_cache:
        return _sign_png_cache[key]
    path = SPEED_SIGN_FOLDER / f"{limit}.png"
    raw  = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
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
    """Gibt BGRA-Array fuer Vollbild-Schild-Anzeige zurueck."""
    path = SPEED_SIGN_FOLDER / f"{limit}.png"
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)


# ================================================================
#  STATE MACHINE
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
        direct = {0: 20, 1: 30, 2: 40, 3: 50, 4: 60, 5: 70,
                  6: 80, 7: 90, 8: 100, 9: 110, 10: 120, 11: 130}
        if class_id in direct:
            self._set(direct[class_id], blue=False)
            return
        if class_id == 12:
            self._context = "innerorts";   self._set(7, blue=True)
        elif class_id == 13:
            self._context = "innerorts";   self._set(50)
        elif class_id == 14:
            self._context = "innerorts";   self._set(50)
        elif class_id == 15:
            self._context = "ausserorts";  self._set(100)
        elif class_id in (16, 17):
            self._set(100 if self._context == "ausserorts" else 30)
        elif class_id == 18:
            self._context = "ausserorts";  self._set(130, blue=True)
        elif class_id == 19:
            self._context = "ausserorts";  self._set(100)


# ================================================================
#  DETEKTION AUSWAEHLEN
# ================================================================

def select_primary_detection(detections: list,
                              cam_w: int, cam_h: int) -> Optional[dict]:
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]
    frame_area = cam_w * cam_h

    def score(d):
        x1, y1, x2, y2 = d["bbox"]
        return d["conf"] * ((x2 - x1) * (y2 - y1)) / frame_area

    return max(detections, key=score)


# ================================================================
#  HAILO INFERENZ
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
        params = _MODEL_PARAMS.get(self.model_w)
        if params:
            with _runtime_lock:
                _runtime["conf_thresh"] = params["conf_thresh"]
                _runtime["infer_every"] = params["infer_every"]
                _runtime["debounce"]    = params["debounce"]
        else:
            _print_line(f"[WARN] Kein Param-Mapping fuer {self.model_w}px -- Defaults.")
        with _runtime_lock:
            _runtime["hef_name"]  = hef_path.name
            _runtime["model_res"] = f"{self.model_w}x{self.model_h}"

    def close(self) -> None:
        try:
            time.sleep(0.3)
            self.hailo.close()
            print("[Hailo] Sauber beendet.")
        except Exception as e:
            if "HAILO_STREAM_ABORT" not in str(e) and "system_error" not in str(e):
                print(f"[Hailo] Beenden: {e}")

    def preprocess(self, frame_bgr: np.ndarray, cam_w: int, cam_h: int,
                   apply_roi_crop: bool = False):
        roi_offset_y = 0
        if apply_roi_crop:
            roi_h     = int(frame_bgr.shape[0] * ROI_TOP_FRACTION)
            frame_bgr = frame_bgr[:roi_h, :]
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
        detections     = []
        conf_threshold = get_runtime("conf_thresh")
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
                if score < conf_threshold:
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
#  TEMPORAL DEBOUNCER
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

    def resize(self, new_size: int) -> None:
        self.buffer_size   = new_size
        self.required_hits = max(1, int(new_size * 0.6))
        self.buffer        = deque(self.buffer, maxlen=new_size)


# ================================================================
#  CAMERA STREAM THREAD
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
#  KAMERA-HARDWARE
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


def restart_camera(cam: Picamera2, mode_name: str) -> tuple:
    cam_w, cam_h, fps = CAMERA_MODES.get(mode_name, CAMERA_MODES[DEFAULT_MODE])
    frame_us = int(1_000_000 / fps)
    _print_line(f"[Kamera] Modusaenderung -> {mode_name}")
    cam.stop()
    cfg = cam.create_video_configuration(
        main={"format": "BGR888", "size": (cam_w, cam_h)},
        controls=_build_camera_controls(frame_us),
        display=None,
        encode=None
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.5)
    return cam_w, cam_h, fps


# ================================================================
#  DISCLAIMER
# ================================================================

DISCLAIMER_TEXT = (
    "Das System kann Fehler aufweisen.",
    "Bitte achten Sie weiterhin",
    "aktiv auf die Verkehrsschilder.",
)

_BG_COLOR     = (20,  20,  20)
_TEXT_COLOR   = (220, 220, 220)
_BAR_BG_COLOR = (60,  60,  60)
_BAR_FG_COLOR = (0,  200, 100)
_ACCENT_COLOR = (0,  180, 255)


def _show_disclaimer(win_name: str) -> None:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    t_start   = time.monotonic()
    fs_main   = DISPLAY_H / 480.0
    fs_small  = DISPLAY_H / 750.0
    thickness = max(1, int(DISPLAY_H / 350))

    line_heights = []
    for line in DISCLAIMER_TEXT:
        (_, th), bl = cv2.getTextSize(line, font, fs_main, thickness)
        line_heights.append(th + bl + int(DISPLAY_H * 0.018))

    total_h = sum(line_heights)
    start_y = (DISPLAY_H - total_h) // 2

    bar_x = int(DISPLAY_W * 0.04)
    bar_y = int(DISPLAY_H * 0.07)
    bar_w = int(DISPLAY_W * 0.35)
    bar_h = int(DISPLAY_H * 0.04)

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= DISCLAIMER_SECONDS:
            break
        remaining = DISCLAIMER_SECONDS - elapsed
        progress  = elapsed / DISCLAIMER_SECONDS

        frame    = np.full((DISPLAY_H, DISPLAY_W, 3), _BG_COLOR, dtype=np.uint8)
        fill_w   = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), _BAR_BG_COLOR, -1)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), _BAR_FG_COLOR, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), _TEXT_COLOR, 1)

        cnt_txt = f"{int(remaining)+1}s"
        (cw, ch), _ = cv2.getTextSize(cnt_txt, font, fs_small, 1)
        cv2.putText(frame, cnt_txt,
                    (bar_x + bar_w + int(DISPLAY_W * 0.015), bar_y + bar_h//2 + ch//2),
                    font, fs_small, _ACCENT_COLOR, 1, cv2.LINE_AA)

        cur_y = start_y
        for i, line in enumerate(DISCLAIMER_TEXT):
            (tw, _), _ = cv2.getTextSize(line, font, fs_main, thickness)
            tx = (DISPLAY_W - tw) // 2
            cv2.putText(frame, line,
                        (tx, cur_y + line_heights[i] - int(DISPLAY_H * 0.005)),
                        font, fs_main, _TEXT_COLOR, thickness, cv2.LINE_AA)
            cur_y += line_heights[i]

        cv2.imshow(win_name, _fit_to_screen(frame))
        if cv2.waitKey(33) == 27:
            break


# ================================================================
#  BUTTON-SYSTEM
# ================================================================

# Globale Button-Registry: name -> (x1, y1, x2, y2)
# Wird jeden Frame neu befuellt; Klicks via Mouse-Callback ausgewertet.
_buttons: dict = {}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_btn(frame: np.ndarray, name: str, label: str,
              x1: int, y1: int, x2: int, y2: int,
              active: Optional[bool] = None,
              fs: float = 0.35) -> None:
    """Zeichnet einen Button, registriert seine Klickflaeche."""
    _buttons[name] = (x1, y1, x2, y2)
    if active is True:
        bg, fg, brd = (20, 80, 30), (80, 220, 100), (30, 120, 50)
    elif active is False:
        bg, fg, brd = (70, 20, 20), (200, 80,  80), (100, 30, 30)
    else:
        bg, fg, brd = (40, 40, 55), (170, 170, 200), (65, 65, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), brd, 1)
    (tw, th), _ = cv2.getTextSize(label, _FONT, fs, 1)
    tx = x1 + max(0, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, label, (tx, ty), _FONT, fs, fg, 1, cv2.LINE_AA)


# ================================================================
#  CONTROL PANEL  (CAM-Modus, untere 120 px)
# ================================================================
#
#  Layout (y=200..319):
#  ┌────────────────────────────┬────────────────────────────┐
#  │ CONF  [-] 0.50 [+]         │ INFER [-] 4 [+]            │ y≈204
#  │ DEB   [-]  4   [+]         │ MINPX [-] 20px [+]         │ y≈230
#  │ [ROI: AUS/AN]              │ [1280x720 ▶]               │ y≈256
#  │ Statuszeile                                              │ y≈285
#  └─────────────────────────────────────────────────────────┘

_CTRL_BG   = (15, 15, 20)
_CTRL_LINE = (40, 40, 52)
_LBL_COLOR = (110, 110, 135)
_VAL_COLOR = (80,  180, 255)

# Geometrie-Konstanten fuer das Control-Panel
_CP_Y0    = CAM_VIEW_H      # 200
_CP_ROW_H = 26              # Zeilenhoehe
_CP_BTN_H = 20              # Button-Hoehe
_CP_BTN_W = 18              # [-]/[+]-Breite
_CP_LBL_W = 58              # Breite reserviert fuer Label
_CP_VAL_W = 32              # Mindestbreite fuer Wert-Text


def _row_y(row: int) -> int:
    return _CP_Y0 + 4 + row * _CP_ROW_H


def _draw_adj(frame: np.ndarray, btn_name: str, label: str,
              value_str: str, col: int, row: int) -> None:
    """Zeichnet 'LABEL [-] WERT [+]' in einer Panel-Haelfte."""
    cx = 0 if col == 0 else DISPLAY_W // 2
    ry = _row_y(row)

    cv2.putText(frame, label, (cx + 4, ry + 14),
                _FONT, 0.33, _LBL_COLOR, 1, cv2.LINE_AA)

    bx_minus = cx + _CP_LBL_W
    _draw_btn(frame, btn_name + "_minus", "-",
              bx_minus, ry, bx_minus + _CP_BTN_W, ry + _CP_BTN_H, fs=0.40)

    val_x = bx_minus + _CP_BTN_W + 3
    cv2.putText(frame, value_str, (val_x, ry + 14),
                _FONT, 0.34, _VAL_COLOR, 1, cv2.LINE_AA)

    (vw, _), _ = cv2.getTextSize(value_str, _FONT, 0.34, 1)
    bx_plus = val_x + max(vw, _CP_VAL_W) + 3
    _draw_btn(frame, btn_name + "_plus", "+",
              bx_plus, ry, bx_plus + _CP_BTN_W, ry + _CP_BTN_H, fs=0.40)


def _draw_control_panel(frame: np.ndarray, rt: dict) -> None:
    # Hintergrund + Trennlinie
    cv2.rectangle(frame, (0, _CP_Y0), (DISPLAY_W - 1, DISPLAY_H - 1), _CTRL_BG, -1)
    cv2.line(frame, (0, _CP_Y0), (DISPLAY_W, _CP_Y0), _CTRL_LINE, 1)
    cv2.line(frame, (DISPLAY_W // 2, _CP_Y0 + 2),
             (DISPLAY_W // 2, DISPLAY_H - 22), _CTRL_LINE, 1)

    # Zeile 0: CONF | INFER
    _draw_adj(frame, "conf",  "CONF",  f"{rt['conf_thresh']:.2f}", 0, 0)
    _draw_adj(frame, "infer", "INFER", str(rt["infer_every"]),      1, 0)

    # Zeile 1: DEB | MINPX
    _draw_adj(frame, "deb",   "DEB",   str(rt["debounce"]),         0, 1)
    _draw_adj(frame, "minpx", "MINPX", f"{rt['min_sign_px']}px",   1, 1)

    # Zeile 2: ROI-Toggle | Kameramodus-Wechsel
    ry2     = _row_y(2)
    roi_on  = rt["roi_crop"]
    roi_lbl = "ROI:AN " if roi_on else "ROI:AUS"
    _draw_btn(frame, "roi_toggle", roi_lbl,
              4, ry2, 118, ry2 + _CP_BTN_H, active=roi_on, fs=0.34)

    cur_mode = rt["camera_mode"]
    cam_lbl  = cur_mode.split("@")[0]   # z.B. "1280x720" ohne "@60"
    _draw_btn(frame, "cam_mode", cam_lbl,
              DISPLAY_W // 2 + 4, ry2, DISPLAY_W - 4, ry2 + _CP_BTN_H, fs=0.31)

    # Zeile 3: Statuszeile
    fps_c    = rt.get("fps_cam", 0.0)
    fps_i    = rt.get("fps_inf", 0.0)
    temp     = rt.get("cpu_temp", 0.0)
    n_det    = rt.get("n_det",    0)
    hef      = rt.get("hef_name", "?")
    temp_s   = f"{temp:.0f}C" if temp > 0 else "--"
    stat_str = f"C:{fps_c:.0f} I:{fps_i:.0f} T:{temp_s} Det:{n_det}  {hef}"
    cv2.putText(frame, stat_str, (4, DISPLAY_H - 5),
                _FONT, 0.30, (80, 80, 100), 1, cv2.LINE_AA)


# ================================================================
#  TOGGLE-BUTTON  (immer sichtbar, oben rechts)
# ================================================================

def _draw_toggle_btn(frame: np.ndarray, display_mode: str) -> None:
    label = "CAM>" if display_mode == "SIGN" else "<SGN"
    _draw_btn(frame, "toggle_mode", label,
              DISPLAY_W - 52, 2, DISPLAY_W - 2, 20, fs=0.33)


# ================================================================
#  GUI: SIGN-MODUS
# ================================================================

def _draw_sign_frame(state: SpeedStateMachine,
                     debounce_progress: int, debounce_total: int,
                     rt: dict) -> np.ndarray:
    frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    limit = state.current_limit

    if limit is None:
        msg = "Erkennung laeuft ..."
        (tw, th), _ = cv2.getTextSize(msg, _FONT, 0.55, 1)
        cv2.putText(frame, msg,
                    ((DISPLAY_W - tw) // 2, (DISPLAY_H + th) // 2),
                    _FONT, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    else:
        size = int(DISPLAY_H * SIGN_DISPLAY_FRACTION)
        size = size - (size % 2)
        cx   = DISPLAY_W // 2
        cy   = DISPLAY_H // 2
        x1   = cx - size // 2
        y1   = cy - size // 2

        png = _load_sign_png_bgra(limit, size)
        if png is not None:
            if png.ndim == 3 and png.shape[2] == 4:
                a  = png[:, :, 3:4].astype(np.float32) / 255.0
                fg = png[:, :, :3].astype(np.float32)
                bg = frame[y1:y1+size, x1:x1+size].astype(np.float32)
                frame[y1:y1+size, x1:x1+size] = (a * fg + (1.0-a) * bg).astype(np.uint8)
            else:
                frame[y1:y1+size, x1:x1+size] = png[:, :, :3]
        else:
            radius   = size // 2
            ring_w   = max(3, radius // 8)
            ring_col = (0, 0, 200) if state.use_blue_circle else (200, 60, 0)
            cv2.circle(frame, (cx, cy), radius, ring_col, -1)
            cv2.circle(frame, (cx, cy), radius - ring_w, (255, 255, 255), -1)
            txt = str(limit)
            fs  = (size / 200.0) * (0.8 if len(txt) >= 3 else 1.1)
            t   = max(2, int(size / 80))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, t)
            cv2.putText(frame, txt, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (20, 20, 20), t, cv2.LINE_AA)

        # Debounce-Bogen
        if debounce_total > 1 and debounce_progress > 0:
            radius = size // 2
            arc_r  = radius + max(3, int(DISPLAY_H * 0.010))
            arc_t  = max(2, int(DISPLAY_H * 0.006))
            angle  = int(360 * min(debounce_progress, debounce_total) / debounce_total)
            cv2.ellipse(frame, (cx, cy), (arc_r, arc_r),
                        -90, 0, angle, (0, 220, 120), arc_t)

    # Status unten rechts
    fps_i  = rt.get("fps_inf", 0.0)
    temp   = rt.get("cpu_temp", 0.0)
    temp_s = f"{temp:.0f}C" if temp > 0 else "--"
    stat   = f"FPS:{fps_i:.0f} T:{temp_s}"
    (tw, _), _ = cv2.getTextSize(stat, _FONT, 0.30, 1)
    cv2.putText(frame, stat, (DISPLAY_W - tw - 4, DISPLAY_H - 5),
                _FONT, 0.30, (55, 55, 55), 1, cv2.LINE_AA)

    _draw_toggle_btn(frame, "SIGN")
    return frame


# ================================================================
#  GUI: CAM-MODUS
# ================================================================

def _letterbox_frame(frame_bgr: np.ndarray, target_w: int, target_h: int):
    """Letterbox-Resize; gibt (out, scale, x_off, y_off) zurueck."""
    h, w  = frame_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    out   = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    out[y_off:y_off+nh, x_off:x_off+nw] = cv2.resize(
        frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, x_off, y_off


def _scale_detections(detections: list, scale: float,
                      x_off: int, y_off: int) -> list:
    """Skaliert BBox-Koordinaten von Kamera- auf Display-Raum."""
    return [
        {**d, "bbox": [
            int(d["bbox"][0] * scale) + x_off,
            int(d["bbox"][1] * scale) + y_off,
            int(d["bbox"][2] * scale) + x_off,
            int(d["bbox"][3] * scale) + y_off,
        ]}
        for d in detections
    ]


def _draw_detections_small(frame_bgr: np.ndarray, detections: list,
                            primary_idx: Optional[int]) -> None:
    """Zeichnet BBoxen + Konfidenz-Label fuer das kleine Kamerabild."""
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color  = BBOX_COLORS.get(cls_id, (0, 255, 0))
        t      = 2 if i == primary_idx else 1
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, t)
        label = f"{det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.30, 1)
        ly = max(y1, th + 2)
        cv2.rectangle(frame_bgr, (x1, ly - th - 2), (x1 + tw + 3, ly), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, ly - 1),
                    _FONT, 0.30, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_cam_frame(cam_frame: np.ndarray,
                    detections: list,
                    primary: Optional[dict],
                    state: SpeedStateMachine,
                    debounce_progress: int,
                    rt: dict) -> np.ndarray:
    cam_disp, scale, x_off, y_off = _letterbox_frame(cam_frame, DISPLAY_W, CAM_VIEW_H)

    # Detektionen skalieren + zeichnen
    det_scaled = _scale_detections(detections, scale, x_off, y_off)
    primary_idx: Optional[int] = None
    if primary is not None:
        for i, d in enumerate(detections):
            if d is primary:
                primary_idx = i
                break
    _draw_detections_small(cam_disp, det_scaled, primary_idx)

    # Kleines Schild-Overlay (oben rechts, unterhalb Toggle-Button)
    limit = state.current_limit
    if limit is not None:
        sign_sz = min(48, CAM_VIEW_H // 4)
        sx1 = DISPLAY_W - sign_sz - 4
        sy1 = 24
        png = _load_sign_png_composited(limit, sign_sz)
        if png is not None:
            if isinstance(png, tuple):
                afg, one_minus_a = png
                bg = cam_disp[sy1:sy1+sign_sz, sx1:sx1+sign_sz].astype(np.float32)
                cam_disp[sy1:sy1+sign_sz, sx1:sx1+sign_sz] = (
                    afg + one_minus_a * bg).astype(np.uint8)
            else:
                cam_disp[sy1:sy1+sign_sz, sx1:sx1+sign_sz] = png
        # Debounce-Bogen auf kleinem Schild
        deb_total = rt.get("debounce", 4)
        if deb_total > 1 and debounce_progress > 0:
            cxs    = sx1 + sign_sz // 2
            cys    = sy1 + sign_sz // 2
            arc_r  = sign_sz // 2 + 2
            angle  = int(360 * min(debounce_progress, deb_total) / deb_total)
            cv2.ellipse(cam_disp, (cxs, cys), (arc_r, arc_r),
                        -90, 0, angle, (0, 220, 120), 1)

    # Gesamtframe aufbauen
    output = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    output[:CAM_VIEW_H, :] = cam_disp
    _draw_control_panel(output, rt)
    _draw_toggle_btn(output, "CAM")
    return output


# ================================================================
#  MOUSE CALLBACK & BUTTON-HANDLER
# ================================================================

def _on_mouse(event, x: int, y: int, flags, param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # Screen-Koordinaten auf Content-Koordinaten zurueckskalieren
    cx = int(x * DISPLAY_W / _screen_w)
    cy = int(y * DISPLAY_H / _screen_h)
    for name, (bx1, by1, bx2, by2) in list(_buttons.items()):
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            _handle_button(name)
            break


def _handle_button(name: str) -> None:
    with _runtime_lock:
        rt = _runtime.copy()

    if name == "toggle_mode":
        set_runtime("display_mode", "CAM" if rt["display_mode"] == "SIGN" else "SIGN")

    elif name == "conf_minus":
        set_runtime("conf_thresh", max(0.10, round(rt["conf_thresh"] - 0.05, 2)))
    elif name == "conf_plus":
        set_runtime("conf_thresh", min(0.95, round(rt["conf_thresh"] + 0.05, 2)))

    elif name == "infer_minus":
        set_runtime("infer_every", max(1, rt["infer_every"] - 1))
    elif name == "infer_plus":
        set_runtime("infer_every", min(6, rt["infer_every"] + 1))

    elif name == "deb_minus":
        set_runtime("debounce", max(1, rt["debounce"] - 1))
    elif name == "deb_plus":
        set_runtime("debounce", min(8, rt["debounce"] + 1))

    elif name == "minpx_minus":
        set_runtime("min_sign_px", max(5, rt["min_sign_px"] - 5))
    elif name == "minpx_plus":
        set_runtime("min_sign_px", min(200, rt["min_sign_px"] + 5))

    elif name == "roi_toggle":
        set_runtime("roi_crop", not rt["roi_crop"])

    elif name == "cam_mode":
        modes  = list(CAMERA_MODES.keys())
        cur    = rt["camera_mode"]
        idx    = modes.index(cur) if cur in modes else 0
        set_runtime("camera_mode", modes[(idx + 1) % len(modes)])
        set_runtime("_mode_change", True)


# ================================================================
#  MAIN
# ================================================================

def main() -> None:
    print("Starte Speed Sign Detector (Display-Version) ...")

    print("  [1/3] Hailo-NPU ...", end="", flush=True)
    try:
        detector = SpeedSignDetector(HEF_PATH)
    except Exception as e:
        print(f" FEHLER\n[FEHLER] Hailo: {e}")
        sys.exit(1)
    print(" OK")

    print("  [2/3] Picamera2 ...", end="", flush=True)
    cam               = Picamera2()
    cam_w, cam_h, fps = CAMERA_MODES[DEFAULT_MODE]
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

    print("  [3/3] CameraStream ...", end="", flush=True)
    cam_stream = CameraStream(cam)
    cam_stream.start()
    print(" OK")

    _params = _MODEL_PARAMS.get(detector.model_w, {})
    print(f"\n{'=' * 57}")
    print(f"  Speed Sign Detector  \u00b7  Display-Version  (480x320)")
    print(f"  {'=' * 53}")
    print(f"  Modell  : {HEF_PATH.name:<22} [{detector.model_w}x{detector.model_h}px]")
    print(f"  Kamera  : {cam_w}x{cam_h} @ {fps} fps  |  ROI-Crop: AUS")
    print(f"  Display : {DISPLAY_W}x{DISPLAY_H}  Vollbild={FULLSCREEN}")
    print(f"  Inf.    : conf={_params.get('conf_thresh', CONF_THRESHOLD)}"
          f"  every={_params.get('infer_every', INFER_EVERY_N)}"
          f"  debounce={_params.get('debounce', DEBOUNCE_COUNT)}"
          f"  minpx={MIN_SIGN_PX}")
    print(f"{'=' * 57}\n")

    # Echte Bildschirmaufloesung erkennen
    global _screen_w, _screen_h
    _screen_w, _screen_h = _detect_screen_size(DISPLAY_W, DISPLAY_H)
    if _screen_w != DISPLAY_W or _screen_h != DISPLAY_H:
        print(f"  Display : Inhalt {DISPLAY_W}x{DISPLAY_H} -> Bildschirm {_screen_w}x{_screen_h}")

    # OpenCV-Fenster
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, _screen_w, _screen_h)
    if FULLSCREEN:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN_NAME, _on_mouse)

    print(f"[Disclaimer] Zeige Hinweis fuer {DISCLAIMER_SECONDS} Sekunden ...")
    _show_disclaimer(WIN_NAME)

    print("[System] Erkennungsschleife. TAB=Modus, ESC=Beenden.")

    _set_cursor_visible(False)   # Startmodus ist SIGN -> kein Cursor

    state     = SpeedStateMachine()
    debouncer = TemporalDebouncer(buffer_size=5, required_hits=3)

    cam_times: list = []
    inf_times: list = []
    fps_cam = fps_inf = 0.0
    frame_count                          = 0
    prev_disp_mode: str                  = "SIGN"
    last_detections: list                = []
    last_primary: Optional[dict]         = None
    last_inf_frame: Optional[np.ndarray] = None
    debounce_progress                    = 0
    last_temp_t                          = 0.0
    last_logged_limit: Optional[int]     = None

    try:
        while True:
            t0 = time.perf_counter()

            if t0 - last_temp_t >= 2.0:
                set_runtime("cpu_temp", get_cpu_temp())
                last_temp_t = t0

            infer_every = get_runtime("infer_every")
            deb_count   = get_runtime("debounce")
            roi_active  = get_runtime("roi_crop")
            min_sign_px = get_runtime("min_sign_px")

            # Kamera-Modusaenderung
            if get_runtime("_mode_change"):
                set_runtime("_mode_change", False)
                new_mode = get_runtime("camera_mode")
                cam_stream.stop()
                cam_w, cam_h, fps = restart_camera(cam, new_mode)
                last_detections   = []
                last_primary      = None
                last_inf_frame    = None
                debouncer.reset()
                cam_stream = CameraStream(cam)
                cam_stream.start()

            frame_bgr = cam_stream.read()
            if frame_bgr is None:
                if not cam_stream._thread.is_alive():
                    _print_line("[WARN] CameraStream-Thread tot -- starte neu ...")
                    cam_stream = CameraStream(cam)
                    cam_stream.start()
                else:
                    time.sleep(0.001)
                continue

            frame_count += 1

            if debouncer.buffer_size != deb_count:
                debouncer.resize(deb_count)

            # Inferenz (jeden N-ten Frame)
            if frame_count % infer_every == 0:
                t_inf          = time.perf_counter()
                last_inf_frame = frame_bgr

                img_rgb, scale, pad_x, pad_y, roi_off_y = detector.preprocess(
                    last_inf_frame, cam_w, cam_h, apply_roi_crop=roi_active)
                result   = detector.run(img_rgb)
                raw_dets = detector.postprocess(
                    result, cam_w, cam_h, scale, pad_x, pad_y, roi_off_y)

                # Min-Groesse-Filter (in Kamera-Pixel)
                last_detections = [
                    d for d in raw_dets
                    if (d["bbox"][2] - d["bbox"][0]) >= min_sign_px
                    and (d["bbox"][3] - d["bbox"][1]) >= min_sign_px
                ]

                last_primary      = select_primary_detection(last_detections, cam_w, cam_h)
                cid               = last_primary["class_id"] if last_primary else None
                confirmed         = debouncer.update(cid)
                debounce_progress = debouncer.get_progress(cid)

                if confirmed is not None:
                    state.update(confirmed)
                    if state.current_limit != last_logged_limit:
                        sname = SIGN_CLASSES.get(confirmed, {}).get("name", "?")
                        _print_line(f"[SCHILD] {sname} -> {state.current_limit} km/h")
                        last_logged_limit = state.current_limit

                inf_times.append(time.perf_counter() - t_inf)
                if len(inf_times) > 30:
                    inf_times.pop(0)
                fps_inf = 1.0 / (sum(inf_times) / len(inf_times))
                set_runtime("fps_inf", fps_inf)
                set_runtime("n_det",   len(last_detections))

            # FPS messen
            cam_times.append(time.perf_counter() - t0)
            if len(cam_times) > 60:
                cam_times.pop(0)
            fps_cam = 1.0 / (sum(cam_times) / len(cam_times))
            set_runtime("fps_cam", fps_cam)

            # Display-Frame aufbauen
            with _runtime_lock:
                rt_snap = {k: v for k, v in _runtime.items() if not k.startswith("_")}

            disp_mode = rt_snap["display_mode"]
            if disp_mode != prev_disp_mode:
                _set_cursor_visible(disp_mode == "CAM")
                prev_disp_mode = disp_mode
            # Kopie des letzten Inferenz-Frames (verhindert Annotations-Akkumulation)
            enc_frame = last_inf_frame.copy() if last_inf_frame is not None else frame_bgr

            if disp_mode == "CAM":
                output = _draw_cam_frame(
                    enc_frame, last_detections, last_primary,
                    state, debounce_progress, rt_snap)
            else:
                output = _draw_sign_frame(
                    state, debounce_progress, deb_count, rt_snap)

            cv2.imshow(WIN_NAME, _fit_to_screen(output))

            # Tastatur
            key = cv2.waitKey(1) & 0xFF
            if key == 27:    # ESC -> Beenden
                break
            elif key == 9:   # TAB -> Modus wechseln
                _handle_button("toggle_mode")
            elif disp_mode == "CAM":
                if   key == ord('q'): _handle_button("conf_minus")
                elif key == ord('w'): _handle_button("conf_plus")
                elif key == ord('a'): _handle_button("infer_minus")
                elif key == ord('s'): _handle_button("infer_plus")
                elif key == ord('y'): _handle_button("deb_minus")
                elif key == ord('x'): _handle_button("deb_plus")
                elif key == ord('u'): _handle_button("minpx_minus")
                elif key == ord('i'): _handle_button("minpx_plus")
                elif key == ord('r'): _handle_button("roi_toggle")
                elif key == ord('k'): _handle_button("cam_mode")

            # Konsolenstatus
            _limit = state.current_limit
            _line  = (f"FPS: {fps_cam:.0f}/{fps_inf:.0f}"
                      f"  Temp: {get_runtime('cpu_temp'):.0f}C"
                      f"  Schild: {f'{_limit} km/h' if _limit else '---'}"
                      f"  Det: {len(last_detections)}"
                      f"  [{disp_mode}]")
            print(f"\r\033[2K{_line}", end="", flush=True)

    except KeyboardInterrupt:
        pass

    finally:
        print(f"\r{'':<{_STATUS_LINE_LEN}}", flush=True)
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
