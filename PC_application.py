"""
================================================================
Anleitung: Finaler PC Test (PC_application.py)
================================================================

Features:
- Safety First (Minimum-Logik)
- Debouncing (X Frames Stabilität nötig)
- Spam-Prevention (Saubere Konsole)
- Robustes Error-Handling für Bilder
- Verbesserte Mehrfach-Limit-Logik
- False-Positive-Reduktion
- Debug-Fenster zusätzlich rechts
- FPS-Anzeige

================================================================

"""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import sys
import os
import time

# -------------------------------------------------
# 0. KONFIGURATION
# -------------------------------------------------
SOURCE = "screen"
MODEL_PATH = 'runs/s_800px_ohne_130/weights/best.pt'
SPEED_SIGN_FOLDER = 'speed_limit_images'

YOUTUBE = {"top": 185, "left": 110, "width": 1280, "height": 720}
STREET_VIEW = {"top": 0, "left": 0, "width": 1920, "height": 1080}

SCREEN_MONITOR = STREET_VIEW

IMG_SIZE = 800
CONF_THRESHOLD = 0.80
STABLE_FRAMES_REQUIRED = 5
MIN_BOX_SIZE = 25
REQUIRE_CENTERED = False

# -------------------------------------------------
# 1. STATUS
# -------------------------------------------------
LAST_ENVIRONMENT = None
CURRENT_DISPLAY_IMG = None
LAST_CANDIDATE = None
STABILITY_COUNTER = 0
LAST_LOGGED_IMG = "INIT"

prev_time = time.time()
FPS = 0

# -------------------------------------------------
# 2. OVERLAY
# -------------------------------------------------
def draw_overlay(frame, img_name, fps):
    if img_name is not None:
        image_path = os.path.join(SPEED_SIGN_FOLDER, f"{img_name}.png")
        overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(image_path) else None

        if overlay is not None:
            overlay = cv2.resize(overlay, (120, 120))
            h, w, _ = frame.shape
            y1 = h - 150; x1 = 30
            y2 = y1 + overlay.shape[0]; x2 = x1 + overlay.shape[1]

            if overlay.shape[2] == 4:
                alpha = overlay[:,:,3]/255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = alpha*overlay[:,:,c] + (1-alpha)*frame[y1:y2, x1:x2, c]
            else:
                frame[y1:y2, x1:x2] = overlay[:,:,:3]
        else:
            cv2.putText(frame, f"LIMIT: {img_name}", (30, frame.shape[0]-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

    # FPS anzeigen
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    return frame


# -------------------------------------------------
# 3. DEBUG WINDOW BUILDER
# -------------------------------------------------
def build_debug_window(detected_list, current_limit, last_env, stability, last_candidate, fps):
    dbg = np.zeros((720, 600, 3), dtype=np.uint8)

    y = 40
    line_h = 28

    cv2.putText(dbg, "DEBUG INFORMATION", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 200, 255), 2)
    y += 50

    cv2.putText(dbg, f"Aktuelles Limit: {current_limit}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    y += line_h

    cv2.putText(dbg, f"Umfeld: {last_env}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    y += line_h

    cv2.putText(dbg, f"Stability: {stability}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    y += line_h

    cv2.putText(dbg, f"Last Candidate: {last_candidate}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
    y += line_h

    cv2.putText(dbg, f"FPS: {int(fps)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    y += 40

    cv2.putText(dbg, "Erkannte Schilder:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 150, 150), 2)
    y += line_h

    for d in detected_list:
        cv2.putText(dbg, f"- {d['img']}  ({d['val']} km/h)", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
        y += line_h

    return dbg


# -------------------------------------------------
# 4. MODELL LADEN
# -------------------------------------------------
try:
    print(f"Lade Modell: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
except:
    print("❌ Modell nicht gefunden!")
    sys.exit()

sct = mss()

# Hauptfenster links
cv2.namedWindow("YOLO_FINAL", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO_FINAL", 1280, 720)
cv2.moveWindow("YOLO_FINAL", 0, 0)

# Debug rechts
cv2.namedWindow("YOLO_DEBUG", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO_DEBUG", 600, 720)
cv2.moveWindow("YOLO_DEBUG", 1280, 0)

print("Starte... (q zum Beenden)")

# -------------------------------------------------
# 5. LOOP
# -------------------------------------------------
while True:

    # FPS berechnen
    now = time.time()
    FPS = 1.0 / (now - prev_time)
    prev_time = now

    screenshot = sct.grab(SCREEN_MONITOR)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=IMG_SIZE)
    debug_frame = np.array(results[0].plot(), copy=True)

    detected_candidates = []
    reset_detected = False

    # -------------------------------------------------
    # 6. DETECTION & FILTER
    # -------------------------------------------------
    for box in results[0].boxes:
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0]
        w = x2 - x1
        h = y2 - y1

        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        if REQUIRE_CENTERED:
            cx = (x1 + x2) / 2
            if abs(cx - frame.shape[1] / 2) > frame.shape[1] * 0.35:
                continue

        # Klassische Limits
        if label.startswith("Tempolimit_"):
            try:
                val = int(label.split("_")[1])
                detected_candidates.append({"val": val, "img": str(val)})
            except: pass

        elif label == "Spielstrasse":
            LAST_ENVIRONMENT = "spielstrasse"
            detected_candidates.append({"val": 7, "img": "7"})

        elif label == "Ende_Spielstrasse":
            LAST_ENVIRONMENT = "innerorts"
            detected_candidates.append({"val": 50, "img": "50"})

        elif label == "Ortsschild":
            LAST_ENVIRONMENT = "innerorts"
            detected_candidates.append({"val": 50, "img": "50"})

        elif label == "Ende_Ortsschild":
            LAST_ENVIRONMENT = "ausserorts"
            detected_candidates.append({"val": 100, "img": "100"})

        elif label == "Autobahn":
            LAST_ENVIRONMENT = "autobahn"
            detected_candidates.append({"val": 130, "img": "130"})

        elif label == "Ende_Autobahn":
            LAST_ENVIRONMENT = "ausserorts"
            detected_candidates.append({"val": 100, "img": "100"})

        elif label.startswith("Aufhebeschild"):
            reset_detected = True

    # -------------------------------------------------
    # 7. ENTSCHEIDUNG
    # -------------------------------------------------
    current_frame_winner = None

    if detected_candidates:
        win = sorted(detected_candidates, key=lambda x: x['val'])[0]
        current_frame_winner = win['img']

    elif reset_detected:

        if LAST_ENVIRONMENT == "spielstrasse":
            CURRENT_DISPLAY_IMG = "50"
        elif LAST_ENVIRONMENT == "innerorts":
            CURRENT_DISPLAY_IMG = "50"
        elif LAST_ENVIRONMENT == "autobahn":
            CURRENT_DISPLAY_IMG = "130"
        elif LAST_ENVIRONMENT == "ausserorts":
            CURRENT_DISPLAY_IMG = "100"
        else:
            CURRENT_DISPLAY_IMG = "50"

        LAST_LOGGED_IMG = CURRENT_DISPLAY_IMG
        print(f"🔄 AUFHEBUNG → neues Limit: {CURRENT_DISPLAY_IMG}")
        continue

    # -------------------------------------------------
    # 8. DEBOUNCE
    # -------------------------------------------------
    if current_frame_winner:
        if current_frame_winner == LAST_CANDIDATE:
            STABILITY_COUNTER += 1
        else:
            STABILITY_COUNTER = 0
            LAST_CANDIDATE = current_frame_winner

        if STABILITY_COUNTER >= STABLE_FRAMES_REQUIRED:
            CURRENT_DISPLAY_IMG = current_frame_winner

    # Logging
    if CURRENT_DISPLAY_IMG != LAST_LOGGED_IMG:
        print(f"📌 Limit gesetzt: {CURRENT_DISPLAY_IMG}")
        LAST_LOGGED_IMG = CURRENT_DISPLAY_IMG

    # -------------------------------------------------
    # 9. ANZEIGE
    # -------------------------------------------------
    final_frame = draw_overlay(debug_frame, CURRENT_DISPLAY_IMG, FPS)
    cv2.imshow("YOLO_FINAL", final_frame)

    dbg_frame = build_debug_window(
        detected_candidates,
        CURRENT_DISPLAY_IMG,
        LAST_ENVIRONMENT,
        STABILITY_COUNTER,
        LAST_CANDIDATE,
        FPS
    )

    cv2.imshow("YOLO_DEBUG", dbg_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
