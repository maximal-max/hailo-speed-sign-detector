#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speed_sign_detector.py
======================
Raspberry Pi 5 + Hailo-8 + Camera Module 3 (imx708) -- HailoRT 4.20.0

Aufruf:    /usr/bin/python3 speed_sign_detector.py
Stream:    http://localhost:8080

=================================================
DYNAMISCHES MODELL-SWITCHING
=================================================
  MODEL_SIZE  steuert alles:
    MODEL_SIZE = 640  ->  640px.hef  (Standard, ausgewogen)
    MODEL_SIZE = 512  ->  512px.hef  (schneller, etwas weniger Genauigkeit)
    MODEL_SIZE = 800  ->  800px.hef  (groesser, mehr Genauigkeit, mehr CPU)
  HEF_PATH und INPUT_SIZE werden automatisch abgeleitet.

=================================================
BGR/RGB-FARBPIPELINE (bereinigt, stabil)
=================================================
  Picamera2 liefert trotz BGR888-Konfiguration intern RGB-Daten.
  CameraStream-Thread:  capture_array() -> cvtColor(RGB2BGR) -> Puffer
  preprocess():         BGR-Frame -> opt. ROI-Crop -> cvtColor(BGR2RGB) -> Hailo
  _encode_worker():
    Normal:    BGR-Frame -> resize -> imencode (BGR nativ korrekt)
    KI-Auge:   RGB-Tensor -> cvtColor(RGB2BGR) -> imencode

=================================================
KI-AUGE (show_ai_eye)
=================================================
  Streamt den exakten MODEL_SIZE x MODEL_SIZE RGB-Tensor, der in die
  Hailo-NPU geht -- inklusive Letterboxing und grauem Padding.
  Toggle: Web-UI Button oder GET /cmd?ai_eye=1

=================================================
ROI-CROP (roi_crop)
=================================================
  Schneidet die unteren 30% des Kamera-Frames vor der NPU-Inferenz ab.
  Nur die oberen 70% (ROI_TOP_FRACTION) werden an die Hailo-NPU gegeben.
  Die Y-Koordinaten der BBoxen werden im Postprocessing korrigiert.
  Toggle: Web-UI Button oder GET /cmd?roi=1
  Default: AUS (Labortest ohne Fahrzeug-Montage).

=================================================
LIMIT-LOGIK
=================================================
  Schilder  0-11 (Tempolimit)    -> Limit direkt setzen  (roter Ring)
  Schild    12   (Spielstrasse)  -> 7 km/h               (blauer Ring)
  Schild    13   (Ende Spielstr) -> 50 km/h              (roter Ring)
  Schild    14   (Ortsschild)    -> 50 km/h              (roter Ring)
  Schild    15   (Ende Orts)     -> 100 km/h             (roter Ring)
  Schilder 16/17 (Aufhebeschild) -> 30 km/h (innerorts/unbekannt)
                                    100 km/h (ausserorts)
  Schild    18   (Autobahn)      -> 130 km/h             (blauer Ring)
  Schild    19   (Ende Autobahn) -> 100 km/h             (roter Ring)
  Kontext (innerorts/ausserorts) wird nur intern verwaltet.

=================================================
AENDERUNGEN GGU. URSPRUNGSVERSION
=================================================
  Fix  1: JPEG-Encoding in Daemon-Thread ausgelagert (Queue, Drop-on-Full).
  Fix  2: Auto-Exposure aktiviert; ExposureTime als obere Grenze (4 ms).
  Fix  3: select_primary_detection: Score = conf x area_norm statt Minimum.
  Fix  4: postprocess() nutzt self.model_w/h statt globaler INPUT_SIZE.
  Fix  5: CameraStream.stop() wartet mit join() auf Thread-Ende.
  Fix  6: ROI-Crop zur Laufzeit schaltbar (Web-UI + /cmd?roi=).
  Fix  7: Runtime-Lock nur einmal pro Loop-Durchlauf (lokaler Cache).
  Fix  8: Thread-Tod-Erkennung + automatischer Neustart im Main-Loop.
  Fix  9: Counter-Import auf Modulebene verschoben (war im Hotpath).
  Fix 10: np.ascontiguousarray nach cvtColor entfernt (war redundant).
  Fix 11: _stream_running als threading.Event (thread-sicher).
"""

import os
import sys
import time
import queue
import threading
import socketserver
import http.server
import subprocess
from collections import Counter, deque   # Fix 9: Counter auf Modulebene
from urllib.parse import urlparse, parse_qs
from typing import Optional

import numpy as np
import cv2
from picamera2 import Picamera2
from picamera2.devices.hailo import Hailo


# ================================================================
#  MODELL  <-- Hier aendern: 512 | 640 | 800
# ================================================================

MODEL_SIZE = 640   # px  ->  laedt  <MODEL_SIZE>px.hef

# Abgeleitete Konstanten -- NICHT manuell aendern
HEF_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{MODEL_SIZE}px.hef")
INPUT_SIZE = MODEL_SIZE


# ================================================================
#  KAMERA
# ================================================================

CAMERA_MODES = {
    "1280x720@60":  (1280, 720, 60),   # Standard: Binning, niedrige Latenz
    "1536x864@30":  (1536, 864, 30),   # Fernsicht: mehr Detail auf Distanz
    "800x600@90":   (800,  600, 90),   # Schnell: max. FPS, nah am Modell-px
}
DEFAULT_MODE = "1280x720@60"

# Fix 6: Anteil des Frames (von oben), der bei aktivem ROI-Crop erhalten bleibt.
# 0.70 = untere 30% werden abgeschnitten (Motorhaube, irrelevanter Boden).
ROI_TOP_FRACTION = 0.70


# ================================================================
#  INFERENZ-DEFAULTS
# ================================================================

CONF_THRESHOLD = 0.45   # Konfidenz-Schwelle
INFER_EVERY_N  = 2      # Jeden N-ten Frame inferieren
DEBOUNCE_COUNT = 3      # Debouncer: benoetigte Treffer im Puffer


# ================================================================
#  KAMERA-HARDWARE-PARAMETER
# ================================================================

# Fix 2: Max-Belichtungszeit auf 4 ms reduziert (war 8 ms).
# AeEnable=True aktiviert Auto-Exposure; ExposureTime wirkt als obere Grenze.
MAX_EXPOSURE_US      = 4000
MAX_ANALOGUE_GAIN    = 8.0
NOISE_REDUCTION_MODE = 1
SHARPNESS            = 1.5


# ================================================================
#  MJPEG-STREAM
# ================================================================

STREAM_PORT         = 8080
STREAM_JPEG_QUALITY = 75
STREAM_DISPLAY_W    = 960
STREAM_DISPLAY_H    = 540

FFMPEG_MODE  = False
MEDIAMTX_URL = "rtsp://localhost:8554/live"


# ================================================================
#  VIDEO-OVERLAY  (Info-Leiste, feste Pixelgroessen)
# ================================================================

INFO_FONT      = cv2.FONT_HERSHEY_SIMPLEX
INFO_FS        = 0.52
INFO_THICKNESS = 1
INFO_LINE_H    = 22
INFO_PAD_X     = 8
INFO_PAD_Y     = 5
INFO_COLOR     = (0, 220, 0)   # BGR: Gruen
INFO_BG_COLOR  = (0,   0, 0)   # BGR: Schwarz


# ================================================================
#  LAUFZEIT-PARAMETER  (thread-safe dict)
# ================================================================

_runtime_lock = threading.Lock()
_runtime: dict = {
    "camera_mode":  DEFAULT_MODE,
    "conf_thresh":  CONF_THRESHOLD,
    "infer_every":  INFER_EVERY_N,
    "debounce":     DEBOUNCE_COUNT,
    "show_ai_eye":  False,    # KI-Auge: streamt NPU-Eingabe-Tensor
    "roi_crop":     False,    # Fix 6: ROI-Crop (Labortest: Standard AUS)
    "_mode_change": False,    # intern: signalisiert Kamera-Neustart
}


def get_runtime(key):
    with _runtime_lock:
        return _runtime[key]


def set_runtime(key, value):
    with _runtime_lock:
        _runtime[key] = value


# ================================================================
#  STREAM-EVENT  (Fix 11: threading.Event statt bool-Flag)
# ================================================================

# Wird bei Programmende via _stream_event.clear() signalisiert.
# MJPEG-Handler und Encode-Worker pruefen _stream_event.is_set().
_stream_event = threading.Event()
_stream_event.set()


# ================================================================
#  ASYNC JPEG ENCODING  (Fix 1)
# ================================================================

_frame_lock   = threading.Lock()
_current_jpeg = b""
_encode_queue: queue.Queue = queue.Queue(maxsize=2)


def _encode_worker() -> None:
    """
    Daemon-Thread: entnimmt Frames aus _encode_queue und kodiert sie als JPEG.

    Laeuft vollstaendig parallel zum Inferenz-Main-Loop.
    Der Main-Loop ruft nur noch _encode_queue.put_nowait() auf und kehrt
    sofort zurueck -- das Encoding blockiert die Inferenz nicht mehr.

    Drop-on-Full-Strategie: Ist die Queue voll (beide Slots belegt), wird
    der Frame via put_nowait() verworfen (QueueFull). Das ist besser als
    Blockieren, weil die Inferenz-Rate dadurch nicht leidet.
    """
    global _current_jpeg
    while _stream_event.is_set():
        try:
            frame_bgr, ai_eye_rgb = _encode_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        show_ai = get_runtime("show_ai_eye")
        if show_ai and ai_eye_rgb is not None:
            # RGB-Tensor -> BGR fuer imencode
            disp = cv2.cvtColor(ai_eye_rgb, cv2.COLOR_RGB2BGR)
        else:
            disp = cv2.resize(frame_bgr, (STREAM_DISPLAY_W, STREAM_DISPLAY_H))
        _, jpeg = cv2.imencode(".jpg", disp,
                               [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        with _frame_lock:
            _current_jpeg = jpeg.tobytes()


# ================================================================
#  KLASSEN-DEFINITIONEN  &  FARBEN
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
#  SPEED-SIGN PNG-CACHE
# ================================================================

SPEED_SIGN_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "datasets", "application_images_dataset")

# (limit, size) -> (afg_f32, one_minus_a_f32) | bgr_u8 | None
_sign_png_cache: dict = {}


def _load_sign_png(limit: int, size: int):
    """Resizes and pre-converts PNG once; returns compositing-ready arrays or None."""
    key = (limit, size)
    if key in _sign_png_cache:
        return _sign_png_cache[key]
    path = os.path.join(SPEED_SIGN_FOLDER, f"{limit}.png")
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
    if raw is None:
        _sign_png_cache[key] = None
        return None
    scaled = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    if scaled.ndim == 3 and scaled.shape[2] == 4:
        a   = scaled[:, :, 3:4].astype(np.float32) / 255.0
        fg  = scaled[:, :, :3].astype(np.float32)
        result = (a * fg, 1.0 - a)          # precompute both blend terms
    else:
        result = scaled[:, :, :3]
    _sign_png_cache[key] = result
    return result


# Standard-BGR-Farben -- imencode codiert korrekt; Browser zeigt RGB.
BBOX_COLORS = {
    **{i: (0, 200, 0) for i in range(12)},  # Gruen   BGR(0,200,0)
    12: (0,  128, 255),   # Orange  BGR(0,128,255)
    13: (0,  220, 255),   # Gelb    BGR(0,220,255)
    14: (0,    0, 255),   # Rot     BGR(0,0,255)
    15: (200,  0, 255),   # Magenta BGR(200,0,255)
    16: (255, 220,  0),   # Cyan    BGR(255,220,0)
    17: (255, 100,  0),   # Blau    BGR(255,100,0)
    18: (255,  50, 50),   # Hellbl. BGR(255,50,50)
    19: (128,   0, 128),  # Lila    BGR(128,0,128)
}


# ================================================================
#  STATE MACHINE
# ================================================================

class SpeedStateMachine:
    """
    Verwaltet das aktuelle Tempolimit und den internen Fahrkontext.

    current_limit   -- konkreter km/h-Wert oder None (nur am Start).
    _context        -- 'innerorts' | 'ausserorts' | 'unbekannt';
                       wird ausschliesslich intern genutzt um Aufhebeschild
                       korrekt aufzuloesen. Wird NICHT angezeigt.
    use_blue_circle -- True: Autobahn (130) oder Spielstrasse (7).
    """

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

        # Direkte Tempolimits 0-11
        direct = {0: 20,  1: 30,  2: 40,  3: 50,  4: 60,  5: 70,
                  6: 80,  7: 90,  8: 100, 9: 110, 10: 120, 11: 130}
        if class_id in direct:
            self._set(direct[class_id], blue=False)
            return

        if class_id == 12:          # Spielstrasse -> 7 km/h, blauer Ring
            self._context = "innerorts"
            self._set(7, blue=True)
        elif class_id == 13:        # Ende Spielstrasse -> 50 km/h
            self._context = "innerorts"
            self._set(50, blue=False)
        elif class_id == 14:        # Ortsschild -> 50 km/h
            self._context = "innerorts"
            self._set(50, blue=False)
        elif class_id == 15:        # Ende Ortsschild -> 100 km/h
            self._context = "ausserorts"
            self._set(100, blue=False)
        elif class_id in (16, 17):  # Aufhebeschild -- kontextabhaengig
            if self._context == "ausserorts":
                self._set(100, blue=False)
            else:                   # innerorts oder unbekannt -> sicherer Wert
                self._set(30, blue=False)
        elif class_id == 18:        # Autobahn -> 130 km/h, blauer Ring
            self._context = "ausserorts"
            self._set(130, blue=True)
        elif class_id == 19:        # Ende Autobahn -> 100 km/h
            self._context = "ausserorts"
            self._set(100, blue=False)


# ================================================================
#  DETEKTION AUSWAEHLEN
# ================================================================

def select_primary_detection(detections: list,
                              cam_w: int,
                              cam_h: int) -> Optional[dict]:
    """
    Fix 3: Score-basierte Auswahl statt blindem Minimum-Limit.

    Score = conf x area_norm
      conf      -- Konfidenz der Detektion (0..1)
      area_norm -- BBox-Flaeche normiert auf Frame-Flaeche (0..1)

    Grosse, nah gelegene Schilder erzeugen grosse BBoxen und gewinnen
    damit gegenueber kleinen, weit entfernten Schildern auch dann,
    wenn letztere ein niedrigeres Tempolimit anzeigen.

    Die Normierung durch cam_w*cam_h macht den Score aufloesung-
    unabhaengig: ein Modusaenderung von 720p auf 864p aendert
    die Schwellenwert-Semantik nicht.
    """
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
#  HAILO INFERENZ
# ================================================================

class SpeedSignDetector:
    """
    Kapselt Hailo-NPU: Initialisierung, Preprocess, Run, Postprocess.

    Wichtig: Hailo wird OHNE with-Block verwendet, um Segfaults bei
    HailoRT 4.20.0 auf dem Pi 5 zu vermeiden.
    """

    def __init__(self, hef_path: str) -> None:
        print(f"[Hailo] Lade Modell: {hef_path}")
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF nicht gefunden: {hef_path}")
        self.hailo       = Hailo(hef_path, output_type="FLOAT32")
        self.input_shape = self.hailo.get_input_shape()
        self.model_h     = self.input_shape[0]
        self.model_w     = self.input_shape[1]
        print(f"[Hailo] Input Shape: {self.input_shape}")
        print("[Hailo] Bereit.")

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
        """
        Skaliert den BGR-Frame letterbox-artig auf Modell-Eingabegroesse
        und konvertiert ihn zu RGB, da Hailo RGB erwartet.

        Fix  6: Optionaler ROI-Crop vor dem Letterboxing.
                Bei apply_roi_crop=True werden die unteren (1-ROI_TOP_FRACTION)*100%
                des Frames abgeschnitten. Da der Crop vom oberen Rand startet
                (y=0 bleibt y=0), ist der Y-Offset fuer die BBox-Rueckprojektion
                in diesem Fall 0. Die Infrastruktur (roi_offset_y) ist vorhanden
                fuer zukuenftige Crop-Varianten (z.B. Top-Crop fuer Himmelabschnitt).
        Fix 10: np.ascontiguousarray nach cvtColor entfernt -- cvtColor gibt
                bereits ein C-contiguous Array zurueck.

        Rueckgabe: (img_rgb, scale, pad_x, pad_y, roi_offset_y)
          roi_offset_y -- Anzahl abgeschnittener Pixel oben im Original-Frame.
                          0 bei Top-ROI-Crop (Bottom-Abschnitt aendert y=0 nicht).
        """
        roi_offset_y = 0

        if apply_roi_crop:
            h_orig    = frame_bgr.shape[0]
            roi_h     = int(h_orig * ROI_TOP_FRACTION)
            frame_bgr = frame_bgr[:roi_h, :]
            # Bottom-Crop: das Bild beginnt weiterhin bei y=0, daher kein Y-Offset.
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
        # BGR -> RGB: Hailo-Modell wurde mit RGB-Daten trainiert.
        # Fix 10: np.ascontiguousarray entfernt (cvtColor ist bereits C-contiguous).
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return img_rgb, scale, pad_x, pad_y, roi_offset_y

    def run(self, img_rgb: np.ndarray):
        return self.hailo.run(img_rgb)

    def postprocess(self, result, orig_w: int, orig_h: int,
                    scale: float, pad_x: int, pad_y: int,
                    roi_offset_y: int = 0) -> list:
        """
        Dekodiert Hailo-Output zu einer Liste von Detektionen.

        Fix 4: self.model_w/h statt globaler INPUT_SIZE-Konstante.
               Verhindert stille BBox-Fehler bei nicht-quadratischen Modellen
               oder wenn INPUT_SIZE und tatsaechliche Modell-Dimension divergieren.
        Fix 6: roi_offset_y wird zu allen Y-Koordinaten addiert.
               Bei Bottom-Crop (roi_offset_y=0) ist das ein No-Op.
               Bei einem zukuenftigen Top-Crop wuerde hier der Versatz
               der abgeschnittenen Zeilen addiert, damit die BBoxen
               wieder korrekt auf dem Original-Frame sitzen.
        """
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
                # Fix 4: self.model_w/h statt INPUT_SIZE
                x1_l = x1_n * self.model_w;  y1_l = y1_n * self.model_h
                x2_l = x2_n * self.model_w;  y2_l = y2_n * self.model_h
                rx1 = int(max(0, min((x1_l - pad_x) / scale, orig_w)))
                ry1 = int(max(0, min((y1_l - pad_y) / scale, orig_h)))
                rx2 = int(max(0, min((x2_l - pad_x) / scale, orig_w)))
                ry2 = int(max(0, min((y2_l - pad_y) / scale, orig_h)))
                # Fix 6: Y-Koordinaten um ROI-Versatz korrigieren
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
#  TEMPORAL DEBOUNCER  (Counter-basiert, miss-tolerant)
# ================================================================

class TemporalDebouncer:
    """
    Mehrheitsentscheidung via collections.Counter.

    Hailo verliert manchmal fuer einen Frame die BBox (Flackern).
    Striktes Zaehlen wuerde den Counter zuruecksetzen; dieser Debouncer
    toleriert einzelne Luecken (None-Eintraege werden ignoriert).

    Beispiel buffer_size=5, required_hits=3:
      [50, 50, None, 50, 50] -> UPDATE (4 Treffer >= 3 benoetigte)
    """

    def __init__(self, buffer_size: int = 5, required_hits: int = 3) -> None:
        self.buffer_size   = buffer_size
        self.required_hits = required_hits
        self.buffer        = deque(maxlen=buffer_size)

    def update(self, class_id: Optional[int]) -> Optional[int]:
        # Fix 9: Counter kommt aus Modulebene -- kein lokaler Import mehr.
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
        """Passt Puffergroesse an; required_hits wird proportional skaliert (60%)."""
        self.buffer_size   = new_buffer_size
        self.required_hits = max(1, int(new_buffer_size * 0.6))
        self.buffer        = deque(self.buffer, maxlen=new_buffer_size)


# ================================================================
#  MJPEG STREAM  +  WEB-CONTROL-API
# ================================================================

def _build_html(rt: dict) -> str:
    """
    Generiert das komplette HTML der Web-UI.

    Layout:
      - Videostream (oben, max. 960 px breit)
      - Modell-Info-Badge (Zeile)
      - Einstellungs-Panel: dauerhaft sichtbar, 3-spaltiges CSS-Grid
          Spalte 1: Kamera-Aufloesung (Buttons)
          Spalte 2: Inferenz (Konfidenz, Infer-N, Debounce Slider)
          Spalte 3: Diagnose (KI-Auge + ROI-Crop Toggle + JSON-Status)
        Auf schmalen Bildschirmen (<= 720 px) wechselt das Grid auf 1 Spalte.
    """
    conf_pct   = int(rt["conf_thresh"] * 100)
    ai_eye_cls = "" if rt["show_ai_eye"] else "off"
    ai_eye_lbl = "AN"  if rt["show_ai_eye"] else "AUS"
    roi_cls    = "" if rt["roi_crop"] else "off"
    roi_lbl    = "AN"  if rt["roi_crop"]    else "AUS"
    cur_mode   = rt["camera_mode"]

    mode_btns = "".join(
        "<button class='{cls}' onclick='setMode(\"{m}\")'>{m}</button>".format(
            cls=("active" if m == cur_mode else ""), m=m)
        for m in CAMERA_MODES
    )

    return """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speed Sign Detector</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0a0a0a;
  color: #d8d8d8;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  padding: 14px;
}

/* -- Stream -- */
#stream-wrap {
  max-width: 960px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #1e1e1e;
  box-shadow: 0 4px 24px #000a;
}
#stream-wrap img { display: block; width: 100%; height: auto; }

/* -- Modell-Badge -- */
.model-badge {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin: 10px 0 8px 0;
  font-size: 0.78em;
  color: #444;
}
.badge {
  background: #141c22;
  color: #4db8ff;
  border: 1px solid #1e3040;
  border-radius: 4px;
  padding: 2px 9px;
  font-weight: 600;
  letter-spacing: 0.4px;
}

/* -- Settings Panel -- */
.settings {
  max-width: 960px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 2px;
}
@media (max-width: 720px) {
  .settings { grid-template-columns: 1fr; }
}

.card {
  background: #0f0f0f;
  border: 1px solid #1a1a1a;
  border-radius: 8px;
  padding: 14px 16px 16px;
}

.sec-label {
  font-size: 0.67em;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #2d5a78;
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid #181818;
}

/* -- Buttons -- */
.btn-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start; }
button {
  background: #141c22;
  color: #7ab4d0;
  border: 1px solid #1e3040;
  padding: 5px 14px;
  border-radius: 5px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.8em;
  transition: background 0.12s, color 0.12s;
  white-space: nowrap;
}
button:hover  { background: #1c2d3c; color: #a0d0e8; }
button.active { background: #0f2e1a; color: #5fdc90; border-color: #1a5030; }
button.off    { background: #2a1010; color: #d87070; border-color: #4a2020; }

/* -- Sliders -- */
.slider-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 9px;
}
.slider-row:last-child { margin-bottom: 0; }
.slider-row label { min-width: 110px; font-size: 0.77em; color: #555; }
input[type=range] { flex: 1; min-width: 80px; accent-color: #4db8ff; cursor: pointer; }
.val  { color: #4db8ff; min-width: 30px; font-size: 0.82em; text-align: right; }
.hint { font-size: 0.68em; color: #2e2e2e; margin-left: 3px; }

/* -- Warn + Status-Link -- */
.warn {
  font-size: 0.68em;
  color: #8a5010;
  margin-top: 8px;
  display: block;
  line-height: 1.4;
}
.status-link {
  display: inline-block;
  margin-top: 12px;
  font-size: 0.72em;
  color: #2d5a78;
  text-decoration: none;
  border-bottom: 1px dashed #2d4a5a;
}
.status-link:hover { color: #4db8ff; }
</style>
</head>
<body>

<div id="stream-wrap">
  <img src="/stream" alt="Live-Stream">
</div>

<div class="model-badge">
  Modell: <span class="badge">MODEL_SIZE_PXpx.hef</span>
  &middot;
  Modus: <span class="badge">CUR_MODE</span>
</div>

<div class="settings">

  <!-- Spalte 1: Kamera-Aufloesung -->
  <div class="card">
    <div class="sec-label">Kamera-Aufloesung</div>
    <div class="btn-row">MODE_BTNS</div>
    <span class="warn">Modusaenderung: ca. 1.5 s Unterbrechung</span>
  </div>

  <!-- Spalte 2: Inferenz -->
  <div class="card">
    <div class="sec-label">Inferenz</div>

    <div class="slider-row">
      <label>Konfidenz</label>
      <input type="range" min="10" max="95" value="CONF_PCT"
        oninput='s("conf",(this.value/100).toFixed(2));
                this.nextElementSibling.textContent=(this.value/100).toFixed(2)'>
      <span class="val">CONF_VAL</span>
    </div>

    <div class="slider-row">
      <label>Infer every N</label>
      <input type="range" min="1" max="6" value="INFER_VAL"
        oninput='s("infer_every",this.value);
                this.nextElementSibling.textContent=this.value'>
      <span class="val">INFER_VAL</span>
      <span class="hint">Frames</span>
    </div>

    <div class="slider-row">
      <label>Debounce</label>
      <input type="range" min="1" max="8" value="DEB_VAL"
        oninput='s("debounce",this.value);
                this.nextElementSibling.textContent=this.value'>
      <span class="val">DEB_VAL</span>
      <span class="hint">Treffer</span>
    </div>
  </div>

  <!-- Spalte 3: Diagnose -->
  <div class="card">
    <div class="sec-label">Diagnose</div>
    <div class="btn-row">
      <button id="aie" class="AI_EYE_CLS" onclick="toggleAiEye(this)">
        KI-Auge&nbsp;AI_EYE_LBL
      </button>
      <button id="roi" class="ROI_CLS" onclick="toggleRoi(this)">
        ROI-Crop&nbsp;ROI_LBL
      </button>
    </div>
    <span class="hint" style="display:block;margin-top:8px;line-height:1.5;color:#2e2e2e">
      KI-Auge: MODEL_SIZE_PX&times;MODEL_SIZE_PX NPU-Tensor<br>
      ROI-Crop: untere 30% abschneiden (Auto-Modus)
    </span>
    <a class="status-link" href="/status" target="_blank">&#8599; JSON-Status</a>
  </div>

</div><!-- .settings -->

<script>
function s(p, v) { fetch("/cmd?" + p + "=" + v).catch(function(){}); }
function setMode(m) {
  if (!confirm("Aufloesung wechseln zu " + m + "?\\nKurze Stream-Unterbrechung.")) return;
  fetch("/cmd?mode=" + encodeURIComponent(m))
    .then(function(){ setTimeout(function(){ location.reload(); }, 2600); })
    .catch(function(){});
}
function toggleAiEye(btn) {
  var off = btn.classList.contains("off");
  fetch("/cmd?ai_eye=" + (off ? "1" : "0")).then(function() {
    btn.classList.toggle("off", !off);
    btn.textContent = off ? "KI-Auge\u00a0AN" : "KI-Auge\u00a0AUS";
  }).catch(function(){});
}
function toggleRoi(btn) {
  var off = btn.classList.contains("off");
  fetch("/cmd?roi=" + (off ? "1" : "0")).then(function() {
    btn.classList.toggle("off", !off);
    btn.textContent = off ? "ROI-Crop\u00a0AN" : "ROI-Crop\u00a0AUS";
  }).catch(function(){});
}
</script>
</body>
</html>""".replace("MODEL_SIZE_PX", str(MODEL_SIZE)) \
           .replace("CUR_MODE",      cur_mode) \
           .replace("MODE_BTNS",     mode_btns) \
           .replace("CONF_PCT",      str(conf_pct)) \
           .replace("CONF_VAL",      f"{rt['conf_thresh']:.2f}") \
           .replace("INFER_VAL",     str(rt["infer_every"])) \
           .replace("DEB_VAL",       str(rt["debounce"])) \
           .replace("AI_EYE_CLS",    ai_eye_cls) \
           .replace("AI_EYE_LBL",    ai_eye_lbl) \
           .replace("ROI_CLS",       roi_cls) \
           .replace("ROI_LBL",       roi_lbl)


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass  # Konsole sauber halten

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        # /cmd  -- Laufzeit-Parameter aendern
        if path == "/cmd":
            msg = "OK:"
            if "mode" in params:
                mode = params["mode"][0]
                if mode in CAMERA_MODES:
                    set_runtime("camera_mode", mode)
                    set_runtime("_mode_change", True)
                    msg += f" camera_mode={mode}"
                else:
                    msg += f" FEHLER: ungueltig '{mode}'"
            if "conf" in params:
                try:
                    v = max(0.1, min(0.99, float(params["conf"][0])))
                    set_runtime("conf_thresh", v)
                    msg += f" conf={v:.2f}"
                except ValueError:
                    pass
            if "infer_every" in params:
                try:
                    v = max(1, min(6, int(params["infer_every"][0])))
                    set_runtime("infer_every", v)
                    msg += f" infer_every={v}"
                except ValueError:
                    pass
            if "debounce" in params:
                try:
                    v = max(1, min(8, int(params["debounce"][0])))
                    set_runtime("debounce", v)
                    msg += f" debounce={v}"
                except ValueError:
                    pass
            if "ai_eye" in params:
                v = params["ai_eye"][0] == "1"
                set_runtime("show_ai_eye", v)
                msg += f" ai_eye={v}"
            if "roi" in params:
                # Fix 6: ROI-Crop zur Laufzeit schaltbar
                v = params["roi"][0] == "1"
                set_runtime("roi_crop", v)
                msg += f" roi_crop={v}"
            self._send_text(msg)
            return

        # /status  -- JSON-Statusdump
        if path == "/status":
            import json
            with _runtime_lock:
                st = {k: v for k, v in _runtime.items() if not k.startswith("_")}
            st["model_size"]       = MODEL_SIZE
            st["hef_path"]         = HEF_PATH
            st["available_modes"]  = list(CAMERA_MODES.keys())
            st["roi_top_fraction"] = ROI_TOP_FRACTION
            self._send_text(json.dumps(st, indent=2))
            return

        # /stream  -- MJPEG
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                # Fix 11: threading.Event statt bool-Flag
                while _stream_event.is_set():
                    with _frame_lock:
                        jpeg = _current_jpeg
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.016)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # /  -- HTML-UI
        with _runtime_lock:
            rt = {k: v for k, v in _runtime.items() if not k.startswith("_")}
        html = _build_html(rt).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _send_text(self, text: str) -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type",   "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_stream_server(port: int = STREAM_PORT) -> None:
    class TServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    threading.Thread(
        target=TServer(("", port), MJPEGHandler).serve_forever,
        daemon=True
    ).start()
    print(f"[Stream] -> http://localhost:{port}")
    print(f"[Stream] -> http://<PI-IP>:{port}")


# ================================================================
#  FFMPEG  (optional, RTSP-Out)
# ================================================================

class FFmpegPipe:
    def __init__(self, w: int, h: int, fps: int, url: str) -> None:
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "h264_v4l2m2m", "-b:v", "2M",
            "-maxrate", "2M", "-bufsize", "1M", "-g", str(fps * 2),
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> bool:
        try:
            self.proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:
            pass


# ================================================================
#  CAMERA STREAM THREAD
# ================================================================

class CameraStream:
    """
    Entkoppelt Sensor-Capture von der Inferenzschleife.

    Der Thread liest kontinuierlich Frames; die Hauptschleife holt
    den jeweils neuesten Frame nicht-blockierend via read().

    BGR-Korrektur an der Quelle:
      Picamera2 liefert trotz BGR888-Konfiguration intern RGB-Daten.
      Jeder Frame wird sofort nach capture_array() zu echtem BGR
      konvertiert (cv2.COLOR_RGB2BGR), bevor er gepuffert wird.
      Alle nachgelagerten Operationen (OpenCV-Zeichnen, preprocess,
      imencode) arbeiten damit auf korrekten BGR-Daten.

    Thread-Sicherheit: Lock + np.copy() verhindern Race Conditions.

    Fix 5: stop() wartet mit join(timeout) auf Thread-Ende.
           Verhindert Race Conditions beim Kamera-Neustart.
    """

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
        print("[CameraStream] Thread gestartet.")

    def stop(self) -> None:
        """Fix 5: Aktives Warten auf Thread-Ende (join) statt nur Flag setzen."""
        self._active = False
        self._thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        while self._active:
            try:
                frame = self._cam.capture_array()
                # 4-kanaliges Array normieren (XBGR/XRGB -> 3 Kanaele)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                # Picamera2 liefert RGB intern -> zu BGR konvertieren
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._frame = np.copy(frame)
            except Exception as e:
                if self._active:
                    print(f"[CameraStream] Fehler: {e}")
                break

    def read(self) -> Optional[np.ndarray]:
        """Gibt den neuesten Frame nicht-blockierend zurueck (None wenn noch keiner)."""
        with self._lock:
            return None if self._frame is None else np.copy(self._frame)


# ================================================================
#  VISUALISIERUNG
# ================================================================

def _sc(frame_h: int, value: float) -> float:
    """Skaliert proportional zur Frame-Hoehe (Basis 720 px).
    Nur fuer das Geschwindigkeits-Kreis-Overlay -- Info-Leiste ist fest."""
    return value * (frame_h / 720.0)


def draw_detections(frame_bgr: np.ndarray, detections: list,
                    primary: Optional[dict] = None) -> None:
    """Zeichnet BBoxen und Label-Kacheln fuer alle Detektionen."""
    h          = frame_bgr.shape[0]
    font_scale = _sc(h, 0.52)
    t_box      = max(1, int(_sc(h, 2.5)))
    t_txt      = max(1, int(_sc(h, 1.2)))
    pad        = max(3, int(_sc(h, 4)))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id     = det["class_id"]
        sign       = SIGN_CLASSES.get(cls_id)
        if not sign:
            continue
        color      = BBOX_COLORS.get(cls_id, (0, 255, 0))
        is_primary = primary is not None and det is primary
        t          = t_box + 1 if is_primary else t_box
        label      = f"{sign['name']} {det['conf']:.0%}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, t)
        if is_primary:
            cv2.rectangle(frame_bgr,
                          (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          (255, 255, 255), 1)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, t_txt)
        ly = max(y1, th + pad * 2)
        cv2.rectangle(frame_bgr,
                      (x1, ly - th - pad * 2), (x1 + tw + pad * 2, ly),
                      color, -1)
        cv2.putText(frame_bgr, label, (x1 + pad, ly - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), t_txt, cv2.LINE_AA)


def draw_speed_display(frame_bgr: np.ndarray, state: SpeedStateMachine,
                       debounce_progress: int, debounce_total: int) -> None:
    """
    Geschwindigkeitsschild-Overlay oben rechts.

    Bevorzugt PNG aus datasets/application_images_dataset/<limit>.png
    (mit Alpha-Compositing). Fallback: gezeichneter Kreis (BGR).

    Debounce-Bogen wird in beiden Fällen auf das Schild gelegt.
    """
    limit = state.current_limit
    if limit is None:
        return

    fh, fw = frame_bgr.shape[:2]
    radius = int(_sc(fh, 62))
    cx     = fw - radius - int(_sc(fh, 18))
    cy     = radius + int(_sc(fh, 18))
    size   = radius * 2
    x1, y1 = cx - radius, cy - radius
    x2, y2 = x1 + size,   y1 + size

    png = _load_sign_png(limit, size)

    if png is not None and x1 >= 0 and y1 >= 0 and x2 <= fw and y2 <= fh:
        if isinstance(png, tuple):
            afg, one_minus_a = png
            bg = frame_bgr[y1:y2, x1:x2].astype(np.float32)
            frame_bgr[y1:y2, x1:x2] = (afg + one_minus_a * bg).astype(np.uint8)
        else:
            frame_bgr[y1:y2, x1:x2] = png
    else:
        ring_w  = max(3, int(_sc(fh, 9)))
        inner_r = radius - ring_w

        ring_color = (220, 80, 0) if state.use_blue_circle else (0, 0, 220)

        shadow = max(2, int(_sc(fh, 4)))
        cv2.circle(frame_bgr, (cx + shadow, cy + shadow), radius, (0, 0, 0), -1)
        cv2.circle(frame_bgr, (cx, cy), radius, ring_color, -1)
        cv2.circle(frame_bgr, (cx, cy), inner_r, (255, 255, 255), -1)

        txt    = str(limit)
        fs_num = _sc(fh, 1.2) if len(txt) >= 3 else _sc(fh, 1.65)
        t_num  = max(1, int(_sc(fh, 2.5)))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs_num, t_num)
        cv2.putText(frame_bgr, txt,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, fs_num, (20, 20, 20), t_num, cv2.LINE_AA)

        fs_u = _sc(fh, 0.37)
        t_u  = max(1, int(_sc(fh, 1)))
        (uw, _), _ = cv2.getTextSize("km/h", cv2.FONT_HERSHEY_SIMPLEX, fs_u, t_u)
        uy = cy + th // 2 + int(_sc(fh, 17))
        if uy < cy + inner_r - 4:
            cv2.putText(frame_bgr, "km/h",
                        (cx - uw // 2, uy),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_u, (80, 80, 80), t_u, cv2.LINE_AA)

    # Debounce-Bogen -- liegt auf PNG und gezeichnetem Schild
    if debounce_total > 1 and debounce_progress > 0:
        angle = int(360 * min(debounce_progress, debounce_total) / debounce_total)
        arc_r = radius + int(_sc(fh, 5))
        arc_t = max(2, int(_sc(fh, 3)))
        cv2.ellipse(frame_bgr, (cx, cy), (arc_r, arc_r),
                    -90, 0, angle, (0, 220, 120), arc_t)


def draw_info(frame_bgr: np.ndarray, fps_cam: float, fps_inf: float,
              n_det: int, mode: str, roi_active: bool) -> None:
    """
    Info-Leiste unten links.
    Feste Pixelgroesse -- skaliert NICHT mit der Kamera-Aufloesung.
    Zeigt: Kamera-FPS | Inferenz-FPS | Anzahl Detektionen | Modus | ROI-Status.
    """
    fh, fw  = frame_bgr.shape[:2]
    roi_tag = " ROI" if roi_active else ""
    line    = (f"Cam:{fps_cam:.0f}  Inf:{fps_inf:.0f} fps"
               f"  |  Det:{n_det}  |  {mode}{roi_tag}")

    (tw, _), _ = cv2.getTextSize(line, INFO_FONT, INFO_FS, INFO_THICKNESS)
    bar_h = INFO_LINE_H + INFO_PAD_Y * 2
    bar_w = min(tw + INFO_PAD_X * 2, fw)

    cv2.rectangle(frame_bgr, (0, fh - bar_h), (bar_w, fh), INFO_BG_COLOR, -1)
    cv2.putText(frame_bgr, line,
                (INFO_PAD_X, fh - INFO_PAD_Y - 2),
                INFO_FONT, INFO_FS, INFO_COLOR, INFO_THICKNESS, cv2.LINE_AA)


# ================================================================
#  KAMERA-NEUSTART
# ================================================================

def _build_camera_controls(frame_us: int) -> dict:
    """
    Erzeugt das Controls-Dictionary fuer Picamera2-Konfiguration.

    Fix 2: AeEnable=True aktiviert Auto-Exposure-Control (AEC) des imx708.
           ExposureTime=MAX_EXPOSURE_US (4 ms) dient als obere Grenze --
           bei Vollautomatik wuerde 8 ms bei Autobahnfahrt zu Motion-Blur
           auf reflektierenden Schildflaechendetails fuehren.
           AnalogueGain bleibt als Safety-Cap bei 8.0.
    """
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
    print(f"\n[Kamera] Wechsle -> {mode_name}")
    cam.stop()
    cfg = cam.create_video_configuration(
        main={"format": "BGR888", "size": (cam_w, cam_h)},
        controls=_build_camera_controls(frame_us),
        display=None,   # Pflicht: verhindert xrdp-Absturz unter Wayland
        encode=None
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.5)
    print(f"[Kamera] Modus aktiv: {mode_name}\n")
    return cam_w, cam_h, fps


# ================================================================
#  MAIN
# ================================================================

def main() -> None:

    print("=" * 55)
    print("  Speed Sign Detector -- HailoRT 4.20.0")
    print(f"  Modell   : {MODEL_SIZE}px  ({HEF_PATH})")
    print(f"  Startmode: {DEFAULT_MODE}")
    print("=" * 55)

    # Fix 1: Encode-Worker-Thread starten
    threading.Thread(target=_encode_worker, daemon=True,
                     name="EncodeWorker").start()
    print("[Encode] Async JPEG-Worker gestartet.")

    start_stream_server()

    try:
        detector = SpeedSignDetector(HEF_PATH)
    except Exception as e:
        print(f"[FEHLER] Hailo: {e}")
        sys.exit(1)

    ffmpeg_pipe = None
    if FFMPEG_MODE:
        try:
            w0, h0, f0 = CAMERA_MODES[DEFAULT_MODE]
            ffmpeg_pipe = FFmpegPipe(w0, h0, f0, MEDIAMTX_URL)
        except Exception as e:
            print(f"[WARNUNG] FFmpeg: {e}")

    print("\n[Kamera] Starte Picamera2 ...")
    cam                   = Picamera2()
    cam_w, cam_h, fps     = CAMERA_MODES[DEFAULT_MODE]
    frame_us              = int(1_000_000 / fps)

    cfg = cam.create_video_configuration(
        main={"format": "BGR888", "size": (cam_w, cam_h)},
        controls=_build_camera_controls(frame_us),
        display=None,   # Pflicht: verhindert xrdp-Absturz unter Wayland
        encode=None
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2.0)

    cam_stream = CameraStream(cam)
    cam_stream.start()

    print(f"[Kamera] Bereit: {cam_w}x{cam_h}@{fps} fps")
    print(f"[System] Web-UI : http://localhost:{STREAM_PORT}")
    print("[System] Beenden: Ctrl+C\n")

    state     = SpeedStateMachine()
    debouncer = TemporalDebouncer(buffer_size=5, required_hits=3)

    cam_times:  list = []
    inf_times:  list = []
    fps_cam                       = 0.0
    fps_inf                       = 0.0
    frame_count                   = 0
    last_detections: list              = []
    last_primary: Optional[dict]       = None
    last_img_rgb: Optional[np.ndarray] = None   # KI-Auge Cache
    last_inf_frame: Optional[np.ndarray] = None # Frame auf dem Inferenz lief
    last_roi_offset_y                  = 0
    debounce_progress                  = 0
    current_mode                       = DEFAULT_MODE

    try:
        while True:
            t0 = time.perf_counter()

            # -----------------------------------------------------------
            # Fix 7: Runtime-Werte einmal pro Loop-Durchlauf cachen.
            # Spart 6-8 Lock-Acquisitions pro Frame.
            # -----------------------------------------------------------
            infer_every = get_runtime("infer_every")
            deb_count   = get_runtime("debounce")
            roi_active  = get_runtime("roi_crop")

            # Kamera-Modusaenderung
            if get_runtime("_mode_change"):
                set_runtime("_mode_change", False)
                new_mode = get_runtime("camera_mode")
                cam_stream.stop()                          # Fix 5: join() intern
                cam_w, cam_h, fps = restart_camera(cam, new_mode)
                current_mode      = new_mode
                last_detections   = []
                last_primary      = None
                last_img_rgb      = None
                last_inf_frame    = None
                last_roi_offset_y = 0
                debouncer.reset()
                cam_stream = CameraStream(cam)
                cam_stream.start()

            # Frame holen (nicht-blockierend)
            frame_bgr = cam_stream.read()

            # -----------------------------------------------------------
            # Fix 8: Thread-Tod erkennen und automatisch neu starten.
            # Vorher: lautloser Ausfall + endloser busy-wait auf None.
            # -----------------------------------------------------------
            if frame_bgr is None:
                if not cam_stream._thread.is_alive():
                    print("[WARNING] CameraStream-Thread tot -- starte neu ...")
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
                t_inf = time.perf_counter()

                # cam_stream.read() liefert bereits np.copy() -- Referenz reicht.
                last_inf_frame = frame_bgr

                # Fix 6: ROI-Crop-Flag aus gecachtem Laufzeit-Wert
                img_rgb, scale, pad_x, pad_y, roi_offset_y = detector.preprocess(
                    last_inf_frame, cam_w, cam_h, apply_roi_crop=roi_active)
                last_img_rgb      = img_rgb
                last_roi_offset_y = roi_offset_y
                result            = detector.run(img_rgb)

                # Fix 4 + Fix 6: model_w/h + roi_offset_y im Postprocessing
                last_detections = detector.postprocess(
                    result, cam_w, cam_h, scale, pad_x, pad_y, roi_offset_y)

                # Fix 3: Neue Score-Funktion mit cam_w/cam_h
                last_primary      = select_primary_detection(
                    last_detections, cam_w, cam_h)
                cid               = last_primary["class_id"] if last_primary else None
                confirmed         = debouncer.update(cid)
                debounce_progress = debouncer.get_progress(cid)

                if confirmed is not None:
                    state.update(confirmed)
                    name = SIGN_CLASSES.get(confirmed, {}).get("name", "?")
                    print(f"[BESTAETIGT] {name} -> {state.current_limit} km/h")

                inf_times.append(time.perf_counter() - t_inf)
                if len(inf_times) > 30:
                    inf_times.pop(0)
                fps_inf = 1.0 / (sum(inf_times) / len(inf_times))

            # Frische Kopie des Inferenz-Frames pro Iteration verhindert
            # Annotations-Akkumulation bei mehreren Nicht-Inferenz-Frames.
            enc_frame = last_inf_frame.copy() if last_inf_frame is not None else frame_bgr
            draw_detections(enc_frame, last_detections, last_primary)
            draw_speed_display(enc_frame, state, debounce_progress, deb_count)
            draw_info(enc_frame, fps_cam, fps_inf,
                      len(last_detections), current_mode, roi_active)

            # Fix 1: Frame asynchron in Encode-Queue einreihen.
            if ffmpeg_pipe:
                if not ffmpeg_pipe.write(enc_frame):
                    print("[FFmpeg] Pipe unterbrochen -> MJPEG")
                    ffmpeg_pipe = None
            if not ffmpeg_pipe:
                try:
                    _encode_queue.put_nowait((enc_frame, last_img_rgb))
                except queue.Full:
                    pass

            # FPS messen
            cam_times.append(time.perf_counter() - t0)
            if len(cam_times) > 60:
                cam_times.pop(0)
            fps_cam = 1.0 / (sum(cam_times) / len(cam_times))

    except KeyboardInterrupt:
        print("\n[System] Beende ...")

    finally:
        # Fix 11: Event statt bool-Flag signalisieren
        _stream_event.clear()
        try:
            cam_stream.stop()
        except Exception:
            pass
        try:
            cam.stop()
            print("[Kamera] Gestoppt.")
        except Exception:
            pass
        detector.close()
        if ffmpeg_pipe:
            ffmpeg_pipe.close()
        print("[System] Sauber beendet.")


if __name__ == "__main__":
    main() 
