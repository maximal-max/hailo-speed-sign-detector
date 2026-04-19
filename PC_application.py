"""
==============================================================================
PC_application.py  —  Speed Sign Detector  |  Automotive HUD
==============================================================================

Windows-Testanwendung für das trainierte YOLO-Modell.
Quellen: Webcam (OpenCV) oder Bildschirmaufnahme (mss).

Architektur:
  - Inference-Thread:   YOLO läuft asynchron; Display-Loop wird nie blockiert.
  - Queue(maxsize=1):   Immer der aktuellste Frame; veraltete Frames werden
                        verworfen wenn der Inference-Thread noch rechnet.
  - Merged Window:      Kamerabild und Dashboard in einem einzigen cv2.imshow().
  - IMG_SIZE auto:      Wird beim Start aus den Modell-Metadaten gelesen.

Tastaturkürzel:  q / ESC = Beenden   r = Reset   s = Screenshot

==============================================================================
"""

import cv2
import numpy as np
import threading
from queue import Queue, Empty
from mss import mss
from ultralytics import YOLO
import sys
import os
import time
import signal
import math
from collections import Counter, deque
from typing import Optional

# Windows-Konsole auf UTF-8 umstellen
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────
# 0. KONFIGURATION
# ─────────────────────────────────────────────────────────────
SOURCE       = "screen"     # "webcam" oder "screen"
WEBCAM_INDEX = 0

MODEL_PATH        = 'runs/800px_YOLO11s/weights/best.pt'
SPEED_SIGN_FOLDER = 'datasets/application_images_dataset'

YOUTUBE      = {"top": 185, "left": 110, "width": 1280, "height": 720}
FULL_SCREEN  = {"top": 0,   "left": 0,   "width": 1920, "height": 1080}
SCREEN_MONITOR = YOUTUBE

# Ziel-Höhe für den Display-Frame (Camera-Seite wird skaliert)
DISPLAY_H = 720

# IMG_SIZE wird nach dem Modell-Laden automatisch gesetzt (→ Abschnitt 12)
IMG_SIZE = 640  # Fallback; wird überschrieben


# ─────────────────────────────────────────────────────────────
# 1. LIVE-RUNTIME  (per Trackbar änderbar)
# ─────────────────────────────────────────────────────────────
_runtime = {
    "conf_thresh":      0.80,
    "stable_frames":    5,
    "infer_every":      1,
    "min_box_size":     25,
    "require_centered": False,
}

def get_rt(key):       return _runtime[key]
def set_rt(key, val):  _runtime[key] = val

# Slider-Spezifikationen: (runtime_key, label, min, max, fmt_func)
_SLIDER_SPECS = [
    ("conf_thresh",   "Konfidenz",    0.05, 0.95, lambda v: f"{v:.0%}"),
    ("stable_frames", "Stab. Frames", 1,    10,   lambda v: str(int(round(v)))),
    ("infer_every",   "Infer / N",    1,    6,    lambda v: str(int(round(v)))),
    ("min_box_size",  "Min Box px",   1,    100,  lambda v: str(int(round(v)))),
]


# ─────────────────────────────────────────────────────────────
# 2. FARBPALETTE  (BGR)
# ─────────────────────────────────────────────────────────────
C = {
    "bg":         (18,  18,  28),
    "bg2":        (28,  28,  42),
    "bg3":        (38,  38,  58),
    "accent":     (180, 220,  0),
    "accent2":    (255, 180,  0),
    "success":    ( 80, 200,  80),
    "text":       (235, 235, 245),
    "text_dim":   (120, 120, 145),
    "border":     ( 55,  55,  75),
    "bar_bg":     ( 40,  40,  58),
    "sign_red":   ( 30,  30, 200),
    "sign_white": (245, 245, 245),
    "black":      (  0,   0,   0),
}


# ─────────────────────────────────────────────────────────────
# 3. GLOBALS + THREADING
# ─────────────────────────────────────────────────────────────
stop_event   = threading.Event()
_pulse_phase = 0.0
_prev_limit  = None
_fps_cam     = 0.0
_cam_times   = deque(maxlen=60)

# Inference-Thread teilt Ergebnisse über dieses Dict mit dem Main-Thread.
# Nur result_lock schützt es – kein direktes State-Objekt-Zugriff im Main-Thread.
_result_lock = threading.Lock()
_shared = {
    "detections":   [],
    "primary":      None,
    "annotated":    None,
    "fps_inf":      0.0,
    "deb_prog":     0,
    "limit":        None,
    "context":      "unbekannt",
    "use_blue":     False,
    "pulse_reset":  False,   # Inference-Thread signalisiert neue Erkennung
}

# Queue: maxsize=1 → Main-Thread legt immer den neuesten Frame hinein;
# altes Frame wird verworfen, wenn Inference noch läuft.
_infer_queue: Queue = Queue(maxsize=1)

# Maus-Interaktion für Custom-Slider
_slider_hit: dict = {}   # key → (track_x1, track_y_center, track_x2) in Dashboard-Koordinaten
_toggle_hit: dict = {}   # key → (x1, y1, x2, y2) in Dashboard-Koordinaten
_cam_display_w: int = 1280   # wird im Main-Loop aktualisiert
_dragging_key: Optional[str] = None

def _apply_slider_at(key: str, dash_x: int):
    """Berechnet neuen Wert anhand der Mausposition auf dem Track."""
    specs = {s[0]: s for s in _SLIDER_SPECS}
    if key not in specs or key not in _slider_hit:
        return
    _, _, mn, mx, _ = specs[key]
    tx1, _, tx2 = _slider_hit[key]
    t = max(0.0, min(1.0, (dash_x - tx1) / max(1, tx2 - tx1)))
    val = mn + t * (mx - mn)
    if key in ("stable_frames", "infer_every", "min_box_size"):
        val = float(round(val))
    set_rt(key, max(mn, min(mx, val)))

def _on_mouse(event, x, y, flags, param):
    global _dragging_key
    dx = x - _cam_display_w          # Koordinate relativ zum Dashboard

    if event == cv2.EVENT_LBUTTONDOWN:
        _dragging_key = None
        for key, (tx1, tcy, tx2) in _slider_hit.items():
            if tx1 - 12 <= dx <= tx2 + 12 and tcy - 12 <= y <= tcy + 12:
                _dragging_key = key
                _apply_slider_at(key, dx)
                return
        for key, (bx1, by1, bx2, by2) in _toggle_hit.items():
            if bx1 <= dx <= bx2 and by1 <= y <= by2:
                set_rt(key, not get_rt(key))
                return

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if _dragging_key:
            _apply_slider_at(_dragging_key, dx)

    elif event == cv2.EVENT_LBUTTONUP:
        _dragging_key = None

def handle_sigint(sig, frame):
    print("\n⏹  Abbruch – beende sauber...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


# ─────────────────────────────────────────────────────────────
# 4. STATE MACHINE
# ─────────────────────────────────────────────────────────────
class SpeedStateMachine:
    DIRECT = {
        "Tempolimit_20": 20,  "Tempolimit_30": 30,  "Tempolimit_40": 40,
        "Tempolimit_50": 50,  "Tempolimit_60": 60,  "Tempolimit_70": 70,
        "Tempolimit_80": 80,  "Tempolimit_90": 90,  "Tempolimit_100":100,
        "Tempolimit_110":110, "Tempolimit_120":120, "Tempolimit_130":130,
    }

    def __init__(self):
        self._limit:   Optional[int] = None
        self._context: str           = "unbekannt"
        self.use_blue: bool          = False

    @property
    def current_limit(self): return self._limit
    @property
    def context(self):       return self._context

    def _set(self, limit, blue=False):
        self._limit   = limit
        self.use_blue = blue

    def update(self, label: str):
        if label in self.DIRECT:
            self._set(self.DIRECT[label])
            return
        handlers = {
            "Spielstrasse":       lambda: (setattr(self, '_context', 'innerorts'),  self._set(7,   blue=True)),
            "Ende_Spielstrasse":  lambda: (setattr(self, '_context', 'innerorts'),  self._set(50)),
            "Ortsschild":         lambda: (setattr(self, '_context', 'innerorts'),  self._set(50)),
            "Ende_Ortsschild":    lambda: (setattr(self, '_context', 'ausserorts'), self._set(100)),
            "Autobahn":           lambda: (setattr(self, '_context', 'autobahn'),   self._set(130, blue=True)),
            "Ende_Autobahn":      lambda: (setattr(self, '_context', 'ausserorts'), self._set(100)),
        }
        if label in handlers:
            handlers[label]()
            return
        if label.startswith("Aufhebeschild"):
            if self._context == "ausserorts": self._set(100)
            elif self._context == "autobahn": self._set(130, blue=True)
            else:                             self._set(50)

    def reset(self):
        self._limit   = None
        self._context = "unbekannt"
        self.use_blue = False


# ─────────────────────────────────────────────────────────────
# 5. TEMPORAL DEBOUNCER
# ─────────────────────────────────────────────────────────────
class TemporalDebouncer:
    def __init__(self, buffer_size=5, required_hits=3):
        self.buffer_size   = buffer_size
        self.required_hits = required_hits
        self.buffer        = deque(maxlen=buffer_size)

    def update(self, label: Optional[str]) -> Optional[str]:
        self.buffer.append(label)
        counts = Counter(v for v in self.buffer if v is not None)
        if not counts:
            return None
        best, cnt = counts.most_common(1)[0]
        return best if cnt >= self.required_hits else None

    def progress(self, label: Optional[str]) -> int:
        return 0 if label is None else sum(1 for v in self.buffer if v == label)

    def reset(self):
        self.buffer.clear()

    def resize(self, new_size: int):
        self.buffer_size   = new_size
        self.required_hits = max(1, int(new_size * 0.6))
        self.buffer        = deque(self.buffer, maxlen=new_size)


# ─────────────────────────────────────────────────────────────
# 6. PRIMARY DETECTION  (Score = conf × area_norm)
# ─────────────────────────────────────────────────────────────
def select_primary(detections, frame_w, frame_h):
    if not detections:        return None
    if len(detections) == 1: return detections[0]
    total = frame_w * frame_h
    if total <= 0:            return detections[0]
    def score(d):
        x1, y1, x2, y2 = d["bbox"]
        return d["conf"] * ((x2 - x1) * (y2 - y1)) / total
    return max(detections, key=score)


# ─────────────────────────────────────────────────────────────
# 7. ZEICHENHILFEN
# ─────────────────────────────────────────────────────────────
def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
    if thickness == -1:
        cv2.rectangle(img, (x1+r,y1),(x2-r,y2), color,-1)
        cv2.rectangle(img, (x1,y1+r),(x2,y2-r), color,-1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img,(cx,cy),r,color,-1)
    else:
        cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
        cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
        cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
        cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
        for cx,cy,a1,a2 in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),
                             (x1+r,y2-r, 90,180),(x2-r,y2-r,  0, 90)]:
            cv2.ellipse(img,(cx,cy),(r,r),0,a1,a2,color,thickness)

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=0):
    ov = img.copy()
    rounded_rect(ov,x1,y1,x2,y2,radius,color,-1) if radius > 0 \
        else cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def text_center(img, txt, cx, cy, scale, color, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
    sz,_ = cv2.getTextSize(txt,font,scale,thick)
    cv2.putText(img,txt,(cx-sz[0]//2,cy+sz[1]//2),font,scale,color,thick,cv2.LINE_AA)

def progress_bar(img, x, y, w, h, value, maximum, fg, radius=4):
    rounded_rect(img,x,y,x+w,y+h,radius,C["bar_bg"],-1)
    fill = int(w*min(value/max(maximum,1),1.0))
    if fill > radius*2:
        rounded_rect(img,x,y,x+fill,y+h,radius,fg,-1)

def section_hdr(img, x, y, label, w=260):
    sz,_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
    cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.44,C["text_dim"],1,cv2.LINE_AA)
    cv2.line(img,(x+sz[0]+8,y-4),(x+w,y-4),C["border"],1)

def draw_param_slider(img, x, y, w, key, label, fmt_fn):
    """Zeichnet einen Slider-Row (28 px hoch). Gibt (tx1, tcy, tx2) zurück."""
    val = get_rt(key)
    specs = {s[0]: s for s in _SLIDER_SPECS}
    mn, mx = specs[key][2], specs[key][3]
    t = max(0.0, min(1.0, (val - mn) / (mx - mn)))

    # Label
    cv2.putText(img, label, (x, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["text_dim"], 1, cv2.LINE_AA)

    # Wert rechts
    val_txt = fmt_fn(val)
    sz, _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    val_x = x + w - sz[0]
    cv2.putText(img, val_txt, (val_x, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                C["accent"] if t >= 0.5 else C["accent2"], 1, cv2.LINE_AA)

    # Track
    TH = 6
    tx1 = x + 112
    tx2 = val_x - 10
    tcy = y + 19
    rounded_rect(img, tx1, tcy - TH//2, tx2, tcy + TH//2 + 1, 3, C["bar_bg"], -1)

    fill_x = tx1 + int((tx2 - tx1) * t)
    if fill_x > tx1 + 4:
        rounded_rect(img, tx1, tcy - TH//2, fill_x, tcy + TH//2 + 1, 3, C["accent"], -1)

    # Handle (Kreis)
    hx = tx1 + int((tx2 - tx1) * t)
    cv2.circle(img, (hx, tcy), 8, C["bg2"], -1)
    cv2.circle(img, (hx, tcy), 7, C["accent"], -1)

    return tx1, tcy, tx2

def draw_toggle_button(img, x, y, w, key, label):
    """Zeichnet ein Toggle-Button-Row (28 px hoch). Gibt (bx1, by1, bx2, by2) zurück."""
    val = get_rt(key)
    cv2.putText(img, label, (x, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["text_dim"], 1, cv2.LINE_AA)

    bw, bh = 58, 20
    bx1 = x + w - bw
    by1 = y + 4
    bx2 = bx1 + bw
    by2 = by1 + bh
    btn_col = C["accent"] if val else C["bg3"]
    rounded_rect(img, bx1, by1, bx2, by2, 6, btn_col, -1)
    txt = "  AN" if val else " AUS"
    text_center(img, txt, bx1 + bw//2, by1 + bh//2, 0.38,
                C["bg"] if val else C["text_dim"], 1, cv2.FONT_HERSHEY_SIMPLEX)
    return bx1, by1, bx2, by2


# ─────────────────────────────────────────────────────────────
# 8. TEMPOSCHILD  (PNG-first, Fallback gezeichnet)
# ─────────────────────────────────────────────────────────────
def draw_speed_sign(img, cx, cy, radius, limit_str, pulse=0.0, use_blue=False):
    size = radius * 2
    if limit_str:
        png_path = os.path.join(SPEED_SIGN_FOLDER, f"{limit_str}.png")
        ov = cv2.imread(png_path, cv2.IMREAD_UNCHANGED) if os.path.exists(png_path) else None
        if ov is not None:
            ov = cv2.resize(ov, (size, size), interpolation=cv2.INTER_AREA)
            x1,y1 = cx-radius, cy-radius
            x2,y2 = x1+size,   y1+size
            ih,iw = img.shape[:2]
            if x1>=0 and y1>=0 and x2<=iw and y2<=ih:
                if ov.shape[2] == 4:
                    a = ov[:,:,3]/255.0
                    for c in range(3):
                        img[y1:y2,x1:x2,c] = (a*ov[:,:,c]+(1-a)*img[y1:y2,x1:x2,c]).astype(np.uint8)
                else:
                    img[y1:y2,x1:x2] = ov[:,:,:3]
            if pulse > 0.05:
                rw = max(2, int(5*pulse))
                cv2.circle(img,(cx,cy),radius+rw,tuple(int(c*pulse) for c in C["accent"]),rw)
            return
    # Fallback
    rc = (200,80,0) if use_blue else C["sign_red"]
    rw = max(3, int(5+pulse*7))
    cv2.circle(img,(cx,cy),radius,C["sign_white"],-1)
    cv2.circle(img,(cx,cy),radius,rc,rw+4)
    cv2.circle(img,(cx,cy),radius-4,C["sign_white"],4)
    cv2.circle(img,(cx,cy),radius-8,rc,rw)
    if limit_str and limit_str.isdigit():
        n=len(limit_str); fs=1.7 if n<=2 else 1.25; ft=4 if n<=2 else 3
        text_center(img,limit_str,cx,cy+5,fs,C["black"],ft,cv2.FONT_HERSHEY_DUPLEX)
    else:
        for deg in range(0,360,22):
            a1=math.radians(deg); a2=math.radians(deg+11); ri=radius-18
            cv2.line(img,(int(cx+ri*math.cos(a1)),int(cy+ri*math.sin(a1))),
                        (int(cx+ri*math.cos(a2)),int(cy+ri*math.sin(a2))),C["text_dim"],2)


# ─────────────────────────────────────────────────────────────
# 9. HUD  (Overlay auf Kamerabild)
# ─────────────────────────────────────────────────────────────
def draw_hud(frame, limit_str, use_blue, fps_cam, fps_inf, n_det, pulse, infer_every):
    h,w = frame.shape[:2]

    # Top-Bar
    alpha_rect(frame,0,0,w,48,C["bg"],alpha=0.76)
    cv2.line(frame,(0,48),(w,48),C["border"],1)
    cv2.line(frame,(0,50),(210,50),C["accent"],2)
    cv2.putText(frame,"SPEED SIGN DETECTOR",(16,33),cv2.FONT_HERSHEY_DUPLEX,0.68,C["accent"],1,cv2.LINE_AA)

    fps_txt = f"CAM {fps_cam:.0f}  |  INF {fps_inf:.0f} fps"
    text_center(frame,fps_txt,w//2,25,0.48,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)

    src_col = C["accent"] if SOURCE == "webcam" else C["accent2"]
    src_txt = f"SRC {SOURCE.upper()}  N={infer_every}"
    sz,_=cv2.getTextSize(src_txt,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
    cv2.putText(frame,src_txt,(w-sz[0]-14,30),cv2.FONT_HERSHEY_SIMPLEX,0.48,src_col,1,cv2.LINE_AA)

    # Temposchild links unten
    sr=72; scx=56+sr; scy=h-54-sr
    alpha_rect(frame,scx-sr-20,scy-sr-20,scx+sr+20,scy+sr+26,C["bg"],alpha=0.68,radius=14)
    draw_speed_sign(frame,scx,scy,sr,limit_str,pulse,use_blue)
    lbl = "AKTUELLES LIMIT" if limit_str else "KEIN LIMIT"
    text_center(frame,lbl,scx,scy+sr+17,0.40,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)

    # Detektion-Badge rechts unten
    det_col = C["accent"] if n_det > 0 else C["text_dim"]
    alpha_rect(frame,w-128,h-42,w-8,h-8,C["bg"],alpha=0.65,radius=8)
    text_center(frame,f"{n_det} Schild{'er' if n_det!=1 else ''}",
                w-68,h-24,0.48,det_col,1,cv2.FONT_HERSHEY_SIMPLEX)
    return frame


# ─────────────────────────────────────────────────────────────
# 10. DASHBOARD
# ─────────────────────────────────────────────────────────────
DASH_W, DASH_H = 540, 720

def build_dashboard(shared_snap, fps_cam, pulse):
    """shared_snap: Snapshot-Dict aus _shared (kein Lock nötig, da lokale Kopie)."""
    detections  = shared_snap["detections"]
    primary     = shared_snap["primary"]
    deb_prog    = shared_snap["deb_prog"]
    fps_inf     = shared_snap["fps_inf"]
    limit_str   = str(shared_snap["limit"]) if shared_snap["limit"] else None
    context     = shared_snap["context"]
    use_blue    = shared_snap["use_blue"]

    dash = np.full((DASH_H,DASH_W,3),C["bg"],dtype=np.uint8)

    # Titel
    cv2.rectangle(dash,(0,0),(DASH_W,56),C["bg2"],-1)
    cv2.line(dash,(0,56),(DASH_W,56),C["border"],1)
    cv2.putText(dash,"DETECTOR  DASHBOARD",(18,37),cv2.FONT_HERSHEY_DUPLEX,0.70,C["accent"],1,cv2.LINE_AA)
    cv2.line(dash,(18,51),(195,51),C["accent"],2)

    y = 72

    # ── Aktives Limit ───────────────────────────────────
    section_hdr(dash,18,y,"AKTIVES TEMPOLIMIT",w=DASH_W-36); y+=12
    ph=108
    rounded_rect(dash,18,y,DASH_W-18,y+ph,10,C["bg2"],-1)
    rounded_rect(dash,18,y,DASH_W-18,y+ph,10,C["border"],1)
    if limit_str:
        pc = tuple(int(c*(0.4+0.6*pulse)) for c in C["accent"])
        cv2.rectangle(dash,(18,y),(25,y+ph),pc,-1)
        draw_speed_sign(dash,74,y+ph//2,42,limit_str,pulse,use_blue)
        cv2.putText(dash,limit_str+" km/h",(132,y+48),cv2.FONT_HERSHEY_DUPLEX,1.35,C["text"],2,cv2.LINE_AA)
        ctx_col = C["accent2"] if context != "unbekannt" else C["text_dim"]
        cv2.putText(dash,context.upper(),(132,y+76),cv2.FONT_HERSHEY_SIMPLEX,0.42,ctx_col,1,cv2.LINE_AA)
    else:
        text_center(dash,"–  Warte auf Erkennung  –",DASH_W//2,y+ph//2,0.50,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)
    y+=ph+18

    # ── Debounce ────────────────────────────────────────
    section_hdr(dash,18,y,"DEBOUNCE  STABILITAET",w=DASH_W-36); y+=12
    stable = int(get_rt("stable_frames"))
    cand   = primary["label"] if primary else None
    cv2.putText(dash,f"Kandidat:  {cand or '–'}",(18,y+13),cv2.FONT_HERSHEY_SIMPLEX,0.48,C["text_dim"],1,cv2.LINE_AA)
    bc = C["accent"] if deb_prog >= stable else C["accent2"]
    progress_bar(dash,18,y+20,DASH_W-36,18,deb_prog,stable,bc,radius=5)
    bw=DASH_W-36
    for i in range(1,stable):
        tx=18+int(bw*i/stable); cv2.line(dash,(tx,y+20),(tx,y+38),C["bg"],2)
    cv2.putText(dash,f"{min(deb_prog,stable)}/{stable}",(DASH_W-54,y+16),cv2.FONT_HERSHEY_SIMPLEX,0.42,bc,1,cv2.LINE_AA)
    y+=52

    # ── Live Parameter  (interaktive Schieber) ──────────
    section_hdr(dash,18,y,"EINSTELLUNGEN  (Klicken & Ziehen)",w=DASH_W-36); y+=12
    ph2 = len(_SLIDER_SPECS)*28 + 28 + 12   # 4 Slider + 1 Toggle + Padding
    rounded_rect(dash,18,y,DASH_W-18,y+ph2,8,C["bg2"],-1)
    rounded_rect(dash,18,y,DASH_W-18,y+ph2,8,C["border"],1)
    row_w = DASH_W - 18 - 30   # innere Breite (30px = linker Einzug)
    for i, (key, lbl, _, _, fmt) in enumerate(_SLIDER_SPECS):
        ry = y + 8 + i * 28
        tx1, tcy, tx2 = draw_param_slider(dash, 30, ry, row_w, key, lbl, fmt)
        _slider_hit[key] = (tx1, tcy, tx2)
    # Toggle Zentriert
    ry = y + 8 + len(_SLIDER_SPECS) * 28
    bx1, by1, bx2, by2 = draw_toggle_button(dash, 30, ry, row_w, "require_centered", "Zentriert")
    _toggle_hit["require_centered"] = (bx1, by1, bx2, by2)
    y+=ph2+18

    # ── Erkannte Schilder ──────────────────────────────
    section_hdr(dash,18,y,"ERKANNTE SCHILDER  (aktueller Frame)",w=DASH_W-36); y+=12
    row_h=34; list_h=max(len(detections),1)*row_h+14
    rounded_rect(dash,18,y,DASH_W-18,y+list_h,8,C["bg2"],-1)
    rounded_rect(dash,18,y,DASH_W-18,y+list_h,8,C["border"],1)
    if detections:
        for i,d in enumerate(detections):
            ry=y+8+i*row_h
            is_p=(primary is not None and d is primary)
            rc=C["accent"] if is_p else C["text"]
            if is_p:
                rounded_rect(dash,22,ry-1,DASH_W-22,ry+row_h-4,5,(30,40,30),-1)
            cv2.circle(dash,(38,ry+14),7,rc,-1 if is_p else 1)
            cv2.putText(dash,d["label"],(54,ry+19),cv2.FONT_HERSHEY_DUPLEX,0.58,rc,1,cv2.LINE_AA)
            ct=f"{d['conf']:.0%}"; sz,_=cv2.getTextSize(ct,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            cv2.putText(dash,ct,(DASH_W-sz[0]-26,ry+19),cv2.FONT_HERSHEY_SIMPLEX,0.42,C["text_dim"],1,cv2.LINE_AA)
            if is_p:
                bsz,_=cv2.getTextSize("PRIM",cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
                bx=DASH_W-sz[0]-bsz[0]-48
                rounded_rect(dash,bx-4,ry+4,bx+bsz[0]+4,ry+24,3,C["accent"],-1)
                cv2.putText(dash,"PRIM",(bx,ry+19),cv2.FONT_HERSHEY_SIMPLEX,0.34,C["bg"],1,cv2.LINE_AA)
    else:
        text_center(dash,"keine Schilder im Frame",DASH_W//2,y+list_h//2,0.44,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)
    y+=list_h+18

    # ── Performance ────────────────────────────────────
    section_hdr(dash,18,y,"PERFORMANCE",w=DASH_W-36); y+=10
    rounded_rect(dash,18,y,DASH_W-18,y+36,6,C["bg2"],-1)
    rounded_rect(dash,18,y,DASH_W-18,y+36,6,C["border"],1)
    perf=[(f"CAM  {fps_cam:.1f} fps",  C["accent"]),
          (f"INF  {fps_inf:.1f} fps",  C["text"]),
          (f"SRC  {SOURCE.upper()}",   C["accent2"])]
    cw=(DASH_W-36)//3
    for i,(txt,col) in enumerate(perf):
        text_center(dash,txt,18+cw//2+i*cw,y+20,0.42,col,1,cv2.FONT_HERSHEY_SIMPLEX)
    y+=50

    # ── Tastaturkürzel ─────────────────────────────────
    section_hdr(dash,18,y,"TASTATURKUERZEL",w=DASH_W-36); y+=10
    keys=[("q / ESC","Beenden"),("r","Limit zuruecksetzen"),("s","Screenshot speichern")]
    for k,v in keys:
        cv2.putText(dash,k,(28,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C["accent2"],1,cv2.LINE_AA)
        cv2.putText(dash,v,(112,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C["text_dim"],1,cv2.LINE_AA)
        y+=20

    # Footer
    cv2.rectangle(dash,(0,DASH_H-28),(DASH_W,DASH_H),C["bg2"],-1)
    cv2.line(dash,(0,DASH_H-28),(DASH_W,DASH_H-28),C["border"],1)
    text_center(dash,"Schieber im Dashboard  |  Tastenkuerzel links",
                DASH_W//2,DASH_H-10,0.38,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)
    return dash


# ─────────────────────────────────────────────────────────────
# 11. INPUT-QUELLE
# ─────────────────────────────────────────────────────────────
def init_source():
    cap, sct_obj = None, None
    if SOURCE == "webcam":
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print(f"❌ Webcam (Index {WEBCAM_INDEX}) nicht gefunden!"); sys.exit(1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam: {actual_w}x{actual_h} @ {actual_fps:.0f} fps  (Index {WEBCAM_INDEX})")
    elif SOURCE == "screen":
        sct_obj = mss()
        print(f"✔ Screen-Capture: {SCREEN_MONITOR['width']}×{SCREEN_MONITOR['height']}")
    else:
        print(f"❌ Unbekannte SOURCE: '{SOURCE}'"); sys.exit(1)
    return cap, sct_obj

def read_frame(cap, sct_obj):
    """Schnellster Capture-Pfad: frombuffer + BGR-Slice vermeidet cvtColor-Overhead."""
    if SOURCE == "webcam":
        ret, frame = cap.read()
        return (True, frame) if (ret and frame is not None) else (False, None)
    # mss liefert BGRA; [:,:,:3] schneidet Alpha weg (View, kein Copy)
    shot  = sct_obj.grab(SCREEN_MONITOR)
    frame = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
    return True, np.ascontiguousarray(frame[:, :, :3])

def cleanup(cap, sct_obj):
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("✔ Ressourcen freigegeben.")


# ─────────────────────────────────────────────────────────────
# 12. MODELL  +  IMG_SIZE auto-detect
# ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modell nicht gefunden: {MODEL_PATH}"); sys.exit(1)
print(f"Lade Modell: {MODEL_PATH} ...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Modell-Ladefehler: {e}"); sys.exit(1)

_imgsz = model.overrides.get("imgsz", 640)
if isinstance(_imgsz, (list, tuple)):
    _imgsz = _imgsz[0]
IMG_SIZE = int(_imgsz)
print(f"✔ IMG_SIZE automatisch gesetzt: {IMG_SIZE}px")


cap, sct_obj = init_source()


# ─────────────────────────────────────────────────────────────
# 13. INFERENCE-THREAD
# ─────────────────────────────────────────────────────────────
_state    = SpeedStateMachine()
_deb      = TemporalDebouncer(buffer_size=5, required_hits=3)
_inf_times: deque = deque(maxlen=30)

def _inference_worker():
    while not stop_event.is_set():
        # Auf neuen Frame warten (timeout damit stop_event geprüft wird)
        try:
            frame, fw, fh = _infer_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            conf_thresh      = get_rt("conf_thresh")
            min_box_size     = int(get_rt("min_box_size"))
            require_centered = get_rt("require_centered")
            stable_frames    = int(get_rt("stable_frames"))

            t_inf   = time.perf_counter()
            results = model.predict(frame, conf=conf_thresh, verbose=False, imgsz=IMG_SIZE)

            detections = []
            for box in results[0].boxes:
                label        = model.names[int(box.cls[0])]
                conf_val     = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0]
                bw,bh        = float(x2-x1), float(y2-y1)
                if bw < min_box_size or bh < min_box_size:
                    continue
                if require_centered:
                    if abs((x1+x2)/2 - fw/2) > fw*0.35:
                        continue
                detections.append({
                    "label": label, "conf": conf_val,
                    "bbox":  [int(x1),int(y1),int(x2),int(y2)],
                })

            primary    = select_primary(detections, fw, fh)
            cand_label = primary["label"] if primary else None

            if _deb.buffer_size != stable_frames:
                _deb.resize(stable_frames)
            deb_prog  = _deb.progress(cand_label)
            confirmed = _deb.update(cand_label)

            pulse_reset = False
            if confirmed is not None:
                _state.update(confirmed)
                print(f"📌 BESTÄTIGT: {confirmed}  →  {_state.current_limit} km/h  [{_state.context}]")
                pulse_reset = True

            _inf_times.append(time.perf_counter() - t_inf)
            fps_inf = 1.0 / (sum(_inf_times)/len(_inf_times)) if _inf_times else 0.0

            annotated = np.array(results[0].plot(), copy=True)

            with _result_lock:
                _shared["detections"]  = detections
                _shared["primary"]     = primary
                _shared["annotated"]   = annotated
                _shared["fps_inf"]     = fps_inf
                _shared["deb_prog"]    = deb_prog
                _shared["limit"]       = _state.current_limit
                _shared["context"]     = _state.context
                _shared["use_blue"]    = _state.use_blue
                _shared["pulse_reset"] = pulse_reset

        except Exception as e:
            print(f"⚠ Inference-Fehler: {e}")

_infer_thread = threading.Thread(target=_inference_worker, daemon=True, name="InferenceWorker")
_infer_thread.start()
print("✔ Inference-Thread gestartet\n")


# ─────────────────────────────────────────────────────────────
# 14. FENSTER  (ein einzelnes Merged-Window)
# ─────────────────────────────────────────────────────────────
WIN = "SPEED SIGN DETECTOR"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280 + DASH_W, DISPLAY_H)
cv2.moveWindow(WIN, 0, 0)

# Maus-Callback für interaktive Dashboard-Schieber
cv2.setMouseCallback(WIN, _on_mouse)

print("Starte... (q / ESC zum Beenden)")
print("Tastenkuerzel:  r=Reset  c=Zentriert  s=Screenshot\n")


# ─────────────────────────────────────────────────────────────
# 15. HAUPTLOOP  (nur Capture + Display, kein blockierendes YOLO)
# ─────────────────────────────────────────────────────────────
frame_count    = 0
consecutive_err = 0
MAX_ERR         = 10

while not stop_event.is_set():
    t0 = time.perf_counter()

    # Puls-Animation (läuft ausschließlich im Main-Thread)
    _pulse_phase += 0.07
    with _result_lock:
        cur_limit   = _shared["limit"]
        use_blue    = _shared["use_blue"]
        pulse_reset = _shared["pulse_reset"]
        _shared["pulse_reset"] = False  # nach Lesen zurücksetzen
    if pulse_reset:
        _pulse_phase = 0.0
        _prev_limit  = None
    pulse_value = (max(0.0, math.sin(_pulse_phase))
                   if cur_limit != _prev_limit else 0.0)
    if (cur_limit != _prev_limit
            and abs(math.sin(_pulse_phase)) < 0.04 and _pulse_phase > 1.0):
        _prev_limit = cur_limit

    # Frame lesen
    ok, frame = read_frame(cap, sct_obj)
    if not ok:
        consecutive_err += 1
        if consecutive_err >= MAX_ERR:
            print("❌ Zu viele Lesefehler – beende."); break
        time.sleep(0.02); continue
    consecutive_err = 0
    frame_count    += 1
    fh, fw          = frame.shape[:2]

    # Frame an Inference-Thread übergeben (non-blocking, altes Frame wird verworfen)
    infer_every = int(get_rt("infer_every"))
    if frame_count % infer_every == 0:
        try:
            _infer_queue.put_nowait((frame, fw, fh))
        except Exception:
            pass  # Queue voll → Inference noch am Rechnen, Frame überspringen

    # Snapshot der letzten Inference-Ergebnisse holen
    with _result_lock:
        snap = dict(_shared)  # flache Kopie reicht (Listen werden nicht mutiert)

    # Display-Frame: letztes annotiertes Bild oder Roh-Frame
    annotated = snap.get("annotated")
    display_frame = annotated.copy() if annotated is not None else frame.copy()

    # Auf DISPLAY_H skalieren (erhält Aspect Ratio)
    dh, dw = display_frame.shape[:2]
    if dh != DISPLAY_H:
        scale  = DISPLAY_H / dh
        new_w  = int(dw * scale)
        display_frame = cv2.resize(display_frame, (new_w, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

    # HUD auf Camera-Frame
    limit_str = str(snap["limit"]) if snap["limit"] else None
    draw_hud(display_frame, limit_str, snap["use_blue"],
             _fps_cam, snap["fps_inf"], len(snap["detections"]),
             pulse_value, infer_every)

    # Dashboard (wird auf DISPLAY_H gestreckt falls nötig)
    dash = build_dashboard(snap, _fps_cam, pulse_value)
    if dash.shape[0] != DISPLAY_H:
        dash = cv2.resize(dash, (DASH_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

    # Einzel-Frame: Camera | Dashboard
    _cam_display_w = display_frame.shape[1]   # für Mouse-Callback aktuell halten
    merged = np.hstack([display_frame, dash])
    cv2.imshow(WIN, merged)

    _cam_times.append(time.perf_counter() - t0)
    _fps_cam = 1.0 / (sum(_cam_times)/len(_cam_times)) if _cam_times else 0.0

    # Tastatur
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        print("⏹  Benutzer-Abbruch."); stop_event.set(); break
    elif key == ord('r'):
        _state.reset(); _deb.reset()
        with _result_lock:
            _shared.update(detections=[], primary=None, annotated=None,
                           deb_prog=0, limit=None, context="unbekannt",
                           use_blue=False, pulse_reset=False)
        _prev_limit=None; _pulse_phase=0.0
        print("Reset: Limit & Zustand zurueckgesetzt.")
    elif key == ord('s'):
        fn = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fn, merged)
        print(f"📸 Screenshot: {fn}")


# ─────────────────────────────────────────────────────────────
# 16. CLEANUP
# ─────────────────────────────────────────────────────────────
stop_event.set()
_infer_thread.join(timeout=3.0)
cleanup(cap, sct_obj)
print("✅ Programm beendet.")
