"""
================================================================
PC_application.py  –  Speed Sign Detector  |  HUD Edition v2
================================================================
Visuelles Design: Automotive HUD / Dark Cockpit

Neu gegenüber v1:
  - Live-Trackbars: Konfidenz, Stabile Frames, Infer-N, Min-Box
  - TemporalDebouncer (Counter-basiert, miss-tolerant wie RPI-App)
  - SpeedStateMachine (saubere Zustandsverwaltung + Kontext)
  - select_primary_detection (Score = conf x area_norm)
  - Infer-every-N: jeden N-ten Frame inferieren (CPU-Entlastung)
  - Rollende FPS-Messung (Cam + Inferenz getrennt)
  - Tastaturkürzel: q/ESC=Ende  r=Reset  s=Screenshot  c=Centered
  - Sauberes Beenden via Strg+C
================================================================
"""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import sys
import os
import time
import signal
import math
from collections import Counter, deque
from typing import Optional

# Windows-Konsole auf UTF-8 umstellen (Emojis + Sonderzeichen)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────
# 0. KONFIGURATION
# ─────────────────────────────────────────────────────────────
SOURCE       = "webcam"     # "webcam" oder "screen"
WEBCAM_INDEX = 0

MODEL_PATH        = 'runs/s_800px_ohne_130/weights/best.pt'
SPEED_SIGN_FOLDER = 'datasets/application_images_dataset'

YOUTUBE      = {"top": 185, "left": 110, "width": 1280, "height": 720}
STREET_VIEW  = {"top": 0,   "left": 0,   "width": 1920, "height": 1080}
SCREEN_MONITOR = STREET_VIEW

IMG_SIZE = 800


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

def _cb_conf(v):     set_rt("conf_thresh",   max(0.05, v / 100.0))
def _cb_stable(v):   set_rt("stable_frames", max(1, v))
def _cb_infer(v):    set_rt("infer_every",   max(1, v))
def _cb_minbox(v):   set_rt("min_box_size",  max(1, v))


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
# 3. GLOBALS
# ─────────────────────────────────────────────────────────────
running      = True
_cam_times   = deque(maxlen=60)
_inf_times   = deque(maxlen=30)
_fps_cam     = 0.0
_fps_inf     = 0.0
_pulse_phase = 0.0
_prev_limit  = None

def handle_sigint(sig, frame):
    global running
    print("\n⏹  Abbruch – beende sauber...")
    running = False

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
    """
    Counter-basiert mit Miss-Toleranz.
    Einzelne None-Frames (Flackern) setzen den Zähler nicht zurück.
    """
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
# 6. PRIMARY DETECTION  (Score = conf x area_norm)
# ─────────────────────────────────────────────────────────────
def select_primary(detections, frame_w, frame_h):
    if not detections:   return None
    if len(detections) == 1: return detections[0]
    total = frame_w * frame_h
    def score(d):
        x1,y1,x2,y2 = d["bbox"]
        return d["conf"] * ((x2-x1)*(y2-y1)) / total
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
    rounded_rect(ov,x1,y1,x2,y2,radius,color,-1) if radius>0 \
        else cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def text_center(img, txt, cx, cy, scale, color, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
    sz,_ = cv2.getTextSize(txt,font,scale,thick)
    cv2.putText(img,txt,(cx-sz[0]//2,cy+sz[1]//2),font,scale,color,thick,cv2.LINE_AA)

def progress_bar(img, x, y, w, h, value, maximum, fg, radius=4):
    rounded_rect(img,x,y,x+w,y+h,radius,C["bar_bg"],-1)
    fill = int(w*min(value/max(maximum,1),1.0))
    if fill>radius*2:
        rounded_rect(img,x,y,x+fill,y+h,radius,fg,-1)

def section_hdr(img, x, y, label, w=260):
    sz,_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
    cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.44,C["text_dim"],1,cv2.LINE_AA)
    cv2.line(img,(x+sz[0]+8,y-4),(x+w,y-4),C["border"],1)


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
                if ov.shape[2]==4:
                    a = ov[:,:,3]/255.0
                    for c in range(3):
                        img[y1:y2,x1:x2,c] = (a*ov[:,:,c]+(1-a)*img[y1:y2,x1:x2,c]).astype(np.uint8)
                else:
                    img[y1:y2,x1:x2] = ov[:,:,:3]
            if pulse>0.05:
                rw = max(2,int(5*pulse))
                cv2.circle(img,(cx,cy),radius+rw,tuple(int(c*pulse) for c in C["accent"]),rw)
            return
    # Fallback
    rc = (200,80,0) if use_blue else C["sign_red"]
    rw = max(3,int(5+pulse*7))
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
def draw_hud(frame, state, fps_cam, fps_inf, n_det, pulse, infer_every):
    h,w = frame.shape[:2]
    limit_str = str(state.current_limit) if state.current_limit else None

    # Top-Bar
    alpha_rect(frame,0,0,w,48,C["bg"],alpha=0.76)
    cv2.line(frame,(0,48),(w,48),C["border"],1)
    cv2.line(frame,(0,50),(210,50),C["accent"],2)
    cv2.putText(frame,"SPEED SIGN DETECTOR",(16,33),cv2.FONT_HERSHEY_DUPLEX,0.68,C["accent"],1,cv2.LINE_AA)

    fps_txt = f"CAM {fps_cam:.0f}  |  INF {fps_inf:.0f} fps"
    text_center(frame,fps_txt,w//2,25,0.48,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)

    src_col = C["accent"] if SOURCE=="webcam" else C["accent2"]
    src_txt = f"SRC {SOURCE.upper()}  N={infer_every}"
    sz,_=cv2.getTextSize(src_txt,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
    cv2.putText(frame,src_txt,(w-sz[0]-14,30),cv2.FONT_HERSHEY_SIMPLEX,0.48,src_col,1,cv2.LINE_AA)

    # Temposchild links unten
    sr=72; scx=56+sr; scy=h-54-sr
    alpha_rect(frame,scx-sr-20,scy-sr-20,scx+sr+20,scy+sr+26,C["bg"],alpha=0.68,radius=14)
    draw_speed_sign(frame,scx,scy,sr,limit_str,pulse,state.use_blue)
    lbl = "AKTUELLES LIMIT" if limit_str else "KEIN LIMIT"
    text_center(frame,lbl,scx,scy+sr+17,0.40,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)

    # Detektion-Badge rechts unten
    det_col = C["accent"] if n_det>0 else C["text_dim"]
    alpha_rect(frame,w-128,h-42,w-8,h-8,C["bg"],alpha=0.65,radius=8)
    text_center(frame,f"{n_det} Schild{'er' if n_det!=1 else ''}",
                w-68,h-24,0.48,det_col,1,cv2.FONT_HERSHEY_SIMPLEX)
    return frame


# ─────────────────────────────────────────────────────────────
# 10. DASHBOARD
# ─────────────────────────────────────────────────────────────
DASH_W, DASH_H = 540, 720

def build_dashboard(state, detections, primary, deb_prog, fps_cam, fps_inf, pulse):
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
    limit_str = str(state.current_limit) if state.current_limit else None
    if limit_str:
        pc = tuple(int(c*(0.4+0.6*pulse)) for c in C["accent"])
        cv2.rectangle(dash,(18,y),(25,y+ph),pc,-1)
        draw_speed_sign(dash,74,y+ph//2,42,limit_str,pulse,state.use_blue)
        cv2.putText(dash,limit_str+" km/h",(132,y+48),cv2.FONT_HERSHEY_DUPLEX,1.35,C["text"],2,cv2.LINE_AA)
        ctx_col = C["accent2"] if state.context!="unbekannt" else C["text_dim"]
        cv2.putText(dash,state.context.upper(),(132,y+76),cv2.FONT_HERSHEY_SIMPLEX,0.42,ctx_col,1,cv2.LINE_AA)
    else:
        text_center(dash,"–  Warte auf Erkennung  –",DASH_W//2,y+ph//2,0.50,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)
    y+=ph+18

    # ── Debounce ────────────────────────────────────────
    section_hdr(dash,18,y,"DEBOUNCE  STABILITAET",w=DASH_W-36); y+=12
    stable = get_rt("stable_frames")
    cand   = primary["label"] if primary else None
    cv2.putText(dash,f"Kandidat:  {cand or '–'}",(18,y+13),cv2.FONT_HERSHEY_SIMPLEX,0.48,C["text_dim"],1,cv2.LINE_AA)
    bc = C["accent"] if deb_prog>=stable else C["accent2"]
    progress_bar(dash,18,y+20,DASH_W-36,18,deb_prog,stable,bc,radius=5)
    bw=DASH_W-36
    for i in range(1,stable):
        tx=18+int(bw*i/stable); cv2.line(dash,(tx,y+20),(tx,y+38),C["bg"],2)
    cv2.putText(dash,f"{min(deb_prog,stable)}/{stable}",(DASH_W-54,y+16),cv2.FONT_HERSHEY_SIMPLEX,0.42,bc,1,cv2.LINE_AA)
    y+=52

    # ── Live Parameter ──────────────────────────────────
    section_hdr(dash,18,y,"LIVE PARAMETER  (Trackbar oben)",w=DASH_W-36); y+=12
    params=[
        ("Konfidenz",      f"{get_rt('conf_thresh'):.0%}",
         C["accent"] if get_rt("conf_thresh")>=0.70 else C["accent2"]),
        ("Stabile Frames", str(get_rt("stable_frames")),  C["text"]),
        ("Infer alle N",   str(get_rt("infer_every")),    C["text"]),
        ("Min Box px",     str(get_rt("min_box_size")),   C["text"]),
        ("Zentriert [c]",  "AN" if get_rt("require_centered") else "AUS",
         C["accent"] if get_rt("require_centered") else C["text_dim"]),
    ]
    ph2=len(params)*28+12
    rounded_rect(dash,18,y,DASH_W-18,y+ph2,8,C["bg2"],-1)
    rounded_rect(dash,18,y,DASH_W-18,y+ph2,8,C["border"],1)
    for i,(k,v,col) in enumerate(params):
        ry=y+8+i*28
        cv2.putText(dash,k,(30,ry+14),cv2.FONT_HERSHEY_SIMPLEX,0.44,C["text_dim"],1,cv2.LINE_AA)
        cv2.putText(dash,v,(DASH_W-90,ry+14),cv2.FONT_HERSHEY_SIMPLEX,0.50,col,1,cv2.LINE_AA)
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
    perf=[( f"CAM  {fps_cam:.1f} fps",C["accent"]),
          (f"INF  {fps_inf:.1f} fps", C["text"]),
          (f"SRC  {SOURCE.upper()}",  C["accent2"])]
    cw=(DASH_W-36)//3
    for i,(txt,col) in enumerate(perf):
        text_center(dash,txt,18+cw//2+i*cw,y+20,0.42,col,1,cv2.FONT_HERSHEY_SIMPLEX)
    y+=50

    # ── Tastaturkürzel ─────────────────────────────────
    section_hdr(dash,18,y,"TASTATURKUERZEL",w=DASH_W-36); y+=10
    keys=[("q / ESC","Beenden"),("r","Limit zuruecksetzen"),
          ("c","Zentriert an/aus"),("s","Screenshot speichern")]
    for k,v in keys:
        cv2.putText(dash,k,(28,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C["accent2"],1,cv2.LINE_AA)
        cv2.putText(dash,v,(112,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C["text_dim"],1,cv2.LINE_AA)
        y+=20

    # Footer
    cv2.rectangle(dash,(0,DASH_H-28),(DASH_W,DASH_H),C["bg2"],-1)
    cv2.line(dash,(0,DASH_H-28),(DASH_W,DASH_H-28),C["border"],1)
    text_center(dash,"Trackbars oben  |  Tastenkuerzel wie links",
                DASH_W//2,DASH_H-10,0.38,C["text_dim"],1,cv2.FONT_HERSHEY_SIMPLEX)
    return dash


# ─────────────────────────────────────────────────────────────
# 11. INPUT-QUELLE
# ─────────────────────────────────────────────────────────────
def init_source():
    cap,sct_obj=None,None
    if SOURCE=="webcam":
        cap=cv2.VideoCapture(WEBCAM_INDEX,cv2.CAP_DSHOW)
        if not cap.isOpened(): cap=cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print(f"❌ Webcam (Index {WEBCAM_INDEX}) nicht gefunden!"); sys.exit(1)
        # MJPG zuerst setzen: verhindert unkomprimiertes YUY2 (~10fps bei 720p).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam: {actual_w}x{actual_h} @ {actual_fps:.0f} fps  (Index {WEBCAM_INDEX})")
    elif SOURCE=="screen":
        sct_obj=mss()
        print(f"✔ Screen-Capture: {SCREEN_MONITOR['width']}×{SCREEN_MONITOR['height']}")
    else:
        print(f"❌ Unbekannte SOURCE: '{SOURCE}'"); sys.exit(1)
    return cap,sct_obj

def read_frame(cap,sct_obj):
    if SOURCE=="webcam":
        ret,frame=cap.read()
        return (True,frame) if (ret and frame is not None) else (False,None)
    return True,cv2.cvtColor(np.array(sct_obj.grab(SCREEN_MONITOR)),cv2.COLOR_BGRA2BGR)

def cleanup(cap,sct_obj):
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("✔ Ressourcen freigegeben.")


# ─────────────────────────────────────────────────────────────
# 12. MODELL
# ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modell nicht gefunden: {MODEL_PATH}"); sys.exit(1)
print(f"Lade Modell: {MODEL_PATH} ...")
try:
    model=YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Modell-Ladefehler: {e}"); sys.exit(1)

cap,sct_obj=init_source()


# ─────────────────────────────────────────────────────────────
# 13. FENSTER + TRACKBARS
# ─────────────────────────────────────────────────────────────
cv2.namedWindow("SPEED SIGN DETECTOR",cv2.WINDOW_NORMAL)
cv2.resizeWindow("SPEED SIGN DETECTOR",1280,720)
cv2.moveWindow("SPEED SIGN DETECTOR",0,0)

cv2.namedWindow("DASHBOARD",cv2.WINDOW_NORMAL)
cv2.resizeWindow("DASHBOARD",DASH_W,DASH_H)
cv2.moveWindow("DASHBOARD",1282,0)

# Trackbars – erscheinen oberhalb des Dashboard-Bildes
cv2.createTrackbar("Konfidenz  %","DASHBOARD", 80, 95, _cb_conf)
cv2.createTrackbar("Stabile Fr.", "DASHBOARD",  5, 10, _cb_stable)
cv2.createTrackbar("Infer alle N","DASHBOARD",  1,  6, _cb_infer)
cv2.createTrackbar("Min Box px",  "DASHBOARD", 25,100, _cb_minbox)

print("Starte... (q / ESC zum Beenden)\n")
print("Tastenkuerzel:  r=Reset  c=Zentriert  s=Screenshot\n")


# ─────────────────────────────────────────────────────────────
# 14. HAUPTLOOP
# ─────────────────────────────────────────────────────────────
state     = SpeedStateMachine()
debouncer = TemporalDebouncer(buffer_size=5, required_hits=3)

frame_count       = 0
last_detections   = []
last_primary      = None
debounce_progress = 0
last_annotated    = None
consecutive_err   = 0
MAX_ERR           = 10

while running:
    t0 = time.perf_counter()

    # Runtime einmal cachen
    conf_thresh      = get_rt("conf_thresh")
    stable_frames    = get_rt("stable_frames")
    infer_every      = get_rt("infer_every")
    min_box_size     = get_rt("min_box_size")
    require_centered = get_rt("require_centered")

    # Debouncer-Größe anpassen
    if debouncer.buffer_size != stable_frames:
        debouncer.resize(stable_frames)

    # Puls-Animation
    _pulse_phase += 0.07
    pulse_value   = (max(0.0, math.sin(_pulse_phase))
                     if state.current_limit != _prev_limit else 0.0)
    if (state.current_limit != _prev_limit
            and abs(math.sin(_pulse_phase)) < 0.04 and _pulse_phase > 1.0):
        _prev_limit = state.current_limit

    # Frame lesen
    ok, frame = read_frame(cap, sct_obj)
    if not ok:
        consecutive_err += 1
        if consecutive_err >= MAX_ERR:
            print("❌ Zu viele Lesefehler – beende."); break
        time.sleep(0.05); continue
    consecutive_err = 0
    frame_count    += 1
    fh, fw          = frame.shape[:2]

    # Inferenz (jeden N-ten Frame)
    if frame_count % infer_every == 0:
        t_inf   = time.perf_counter()
        results = model.predict(frame, conf=conf_thresh, verbose=False, imgsz=IMG_SIZE)

        last_detections = []
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

            last_detections.append({
                "label": label, "conf": conf_val,
                "bbox":  [int(x1),int(y1),int(x2),int(y2)],
            })

        last_primary      = select_primary(last_detections, fw, fh)
        cand_label        = last_primary["label"] if last_primary else None
        debounce_progress = debouncer.progress(cand_label)
        confirmed         = debouncer.update(cand_label)

        if confirmed is not None:
            state.update(confirmed)
            print(f"📌 BESTÄTIGT: {confirmed}  →  {state.current_limit} km/h  [{state.context}]")
            _pulse_phase = 0.0

        _inf_times.append(time.perf_counter() - t_inf)
        _fps_inf = 1.0/(sum(_inf_times)/len(_inf_times)) if _inf_times else 0.0

        last_annotated = np.array(results[0].plot(), copy=True)

    # Anzeige-Frame: letztes annotiertes Bild oder Roh-Frame
    display_frame = last_annotated.copy() if last_annotated is not None else frame.copy()

    # HUD + Dashboard
    final_frame = draw_hud(display_frame, state, _fps_cam, _fps_inf,
                           len(last_detections), pulse_value, infer_every)
    dash        = build_dashboard(state, last_detections, last_primary,
                                  debounce_progress, _fps_cam, _fps_inf, pulse_value)

    cv2.imshow("SPEED SIGN DETECTOR", final_frame)
    cv2.imshow("DASHBOARD", dash)

    _cam_times.append(time.perf_counter() - t0)
    _fps_cam = 1.0/(sum(_cam_times)/len(_cam_times)) if _cam_times else 0.0

    # Tastatur
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        print("⏹  Benutzer-Abbruch."); break
    elif key == ord('r'):
        state.reset(); debouncer.reset()
        last_detections=[]; last_primary=None; debounce_progress=0
        _prev_limit=None; _pulse_phase=0.0
        print("Reset: Limit & Zustand zurueckgesetzt.")
    elif key == ord('c'):
        new=not get_rt("require_centered"); set_rt("require_centered",new)
        print(f"🎯 Zentriert-Modus: {'AN' if new else 'AUS'}")
    elif key == ord('s'):
        fn=f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fn, final_frame)
        print(f"📸 Screenshot: {fn}")


# ─────────────────────────────────────────────────────────────
# 15. CLEANUP
# ─────────────────────────────────────────────────────────────
cleanup(cap, sct_obj)
print("✅ Programm beendet.")