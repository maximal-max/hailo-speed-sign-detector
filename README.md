# Echtzeit-Verkehrsschilderkennung — Raspberry Pi 5 + Hailo-8

Echtzeit-Erkennung von Tempolimits und Verkehrsschildern mit YOLOv8s, optimiert für den Hailo-8 KI-Beschleuniger. Das System erkennt 20 Schildklassen — von Tempolimits über Ortsschilder bis hin zu Autobahnschildern — bei Geschwindigkeiten bis 130 km/h.

---

## Hardware

| Komponente | Modell |
|---|---|
| Einplatinencomputer | Raspberry Pi 5 |
| KI-Beschleuniger | Hailo-8 (26 TOPS) via AI HAT+ |
| Kamera | IMX708 (Camera Module 3) |
| Trainings-PC | Windows, CUDA-fähige GPU |

---

## Erkannte Schildklassen

| ID | Klasse | Ausgelöstes Limit |
|---|---|---|
| 0–11 | Tempolimit 20–130 | Direkt (20 / 30 / 40 / 50 / 60 / 70 / 80 / 90 / 100 / 110 / 120 / 130 km/h) |
| 12 | Spielstraße | 7 km/h (blauer Ring) |
| 13 | Ende Spielstraße | 50 km/h |
| 14 | Ortsschild | 50 km/h |
| 15 | Ende Ortsschild | 100 km/h |
| 16 | Aufhebeschild (leer) | 30 km/h (innerorts) / 100 km/h (außerorts) |
| 17 | Aufhebeschild (Zahl) | 30 km/h (innerorts) / 100 km/h (außerorts) |
| 18 | Autobahn | 130 km/h (blauer Ring) |
| 19 | Ende Autobahn | 100 km/h |

---

## Projektstruktur

```
hailo-speed-sign-detector/
│
├── tempolimits.yaml                ← Zentrale Konfig: Klassen + Datenpfade
│
├── split_dataset.py                ← Schritt 1: Rohdaten aufteilen (train/val)
├── train_yolo.py                   ← Schritt 2: Training + ONNX-Export
├── generate_universal_calib.py     ← Schritt 3: Kalibrierungsset für Hailo DFC
│
├── compare_models.py               ← Hilfstool: Trainingsläufe vergleichen
├── setup_venv.bat                  ← Windows: Trainingsumgebung einrichten
│
├── RPI_application.py              ← Raspberry Pi: Echtzeit-Inferenz + Web-UI
└── PC_application.py               ← Windows-PC: Testen mit Webcam / Bildschirm
```

---

## Schnellstart

### 1. Trainingsumgebung einrichten (Windows)

```bat
setup_venv.bat
```

Installiert Python 3.11.9, PyTorch mit CUDA 12.1, Ultralytics, Albumentations und den ONNX-Stack in einer isolierten `.venv`.

### 2. Datensatz aufteilen

```
datasets/
└── stable_tempo_dataset/
    ├── images/
    └── labels/        ← YOLO-Format (.txt, eine Zeile pro Schild)
```

```bash
python split_dataset.py
```

Erzeugt `sorted_dataset/` mit `train/` (80%) und `val/` (20%), stratifiziert nach Klasse — seltene Schilder landen garantiert auch im Validierungsset.

### 3. `tempolimits.yaml` anpassen

```yaml
path: D:/Dein/Pfad/datasets/sorted_dataset   # ← anpassen
train: images/train
val:   images/val

names:
  0: Tempolimit_20
  1: Tempolimit_30
  # ...
```

### 4. Training starten

```bash
# Zuerst testen (DRY_RUN = True in train_yolo.py):
python train_yolo.py

# Dann echtes Training (DRY_RUN = False, EPOCHS = 200):
python train_yolo.py
```

Ausgabe: `runs/<NAME>/weights/best.pt` + `best.onnx` (Hailo-ready: Opset 11, FP32, statische Shapes).

### 5. Kalibrierungsset erstellen

```bash
python generate_universal_calib.py
```

Erzeugt pro Modellgröße (512 / 640 / 800 px) einen Ordner mit 1024 Letterbox-skalierten Originalbildern:

```
hailo_calibration/
├── calib_512px/
│   ├── images/          ← JPGs für Hailo Cloud Compiler (Ordner-Upload)
│   └── calib_set.npy    ← Array für lokalen DFC
├── calib_640px/  ...
└── calib_800px/  ...
```

### 6. Hailo-Compiler

1. [Hailo Developer Zone](https://hailo.ai/developer-zone/) öffnen
2. `best.onnx` hochladen
3. `hailo_calibration/calib_640px/images/` als Kalibrierungsordner hochladen
4. Kompiliertes `640px.hef` herunterladen und auf den Pi kopieren

### 7. Testen auf dem Windows-PC

```bash
python PC_application.py
```

Lädt das trainierte `best.pt`-Modell und zeigt eine Echtzeit-Inferenz auf Webcam oder Bildschirmaufnahme — kein Raspberry Pi erforderlich. Die Ergebnisse werden in einem automotive HUD-Design dargestellt.

### 8. Inferenz auf dem Raspberry Pi 5

```bash
/usr/bin/python3 RPI_application.py
```

Web-UI: `http://<PI-IP>:8080`

---

## Pipeline-Details

### Augmentierung (`train_yolo.py`)

Die Offline-Augmentierung simuliert gezielt die Bedingungen bei Autobahnfahrt:

| Effekt | Parameter | Zweck |
|---|---|---|
| Motion Blur | Kernel (13–21), 60% Wahrscheinlichkeit | 130 km/h Fahrtgeschwindigkeit |
| Affine Shear | Y: ±4°, X: ±3°, Rot: ±4°, 60% | Rolling-Shutter-Verzerrung der IMX708 |
| Sensor Noise | Varianz 40–150, 60% | High-ISO / Nacht / Regen |
| Helligkeit/Kontrast | 50% | Tunnel, Gegenlicht |

YOLO-Online-Augmentierungen, die bei Schildern schaden würden, sind deaktiviert: `fliplr=0` (Zahlen werden gespiegelt), `mosaic=0`, `mixup=0`.

### Kalibrierung (`generate_universal_calib.py`)

Kalibrierungsbilder sind **keine** augmentierten Bilder — sie sind die unveränderten Originalbilder aus dem Trainingsdatensatz, nur per Letterbox auf die Zielgröße gebracht. Der Hailo DFC misst anhand dieser Bilder die Aktivierungsverteilungen im FP32-Modell und berechnet daraus die optimale INT8-Skalierung pro Schicht. Verzerrte Kalibrierungsbilder führen zu falschen Skalierungen und messbarem mAP-Verlust auf dem Chip.

### Konsistenz: Training ↔ Kalibrierung ↔ Inferenz

Alle drei Schritte verwenden **dasselbe Letterboxing**:

```
LongestMaxSize(max_size=N) + PadIfNeeded(N×N, border=CONSTANT, value=114)
```

Das ist der einzige Preprocessing-Schritt, der in allen drei Kontexten vorkommt — und er muss pixel-identisch sein.

---

## PC-Anwendung (`PC_application.py`)

Ermöglicht das vollständige Testen des trainierten Modells auf einem Windows-PC ohne Raspberry Pi.

### Eingabemodi

| Modus | Beschreibung |
|---|---|
| Webcam | Kameraindex 0 (über OpenCV) |
| Screen Capture | Bildschirmaufnahme via `mss`-Bibliothek |

### Oberfläche

- **Automotive HUD**: Dunkles Cockpit-Design mit farbkodierten Geschwindigkeitszonen
- **Live-Trackbars**: Konfidenz, Stable Frames, Infer-Every-N, Mindest-Boxgröße
- **FPS-Anzeige**: Getrennte Messung für Kamera und Inferenz
- **Keyboard-Shortcuts**: `q` / `ESC` = Beenden, `r` = Reset, `s` = Screenshot, `c` = Zentrierte Ansicht

### Identische Logik

`PC_application.py` verwendet dieselbe `SpeedStateMachine` und denselben `TemporalDebouncer` wie `RPI_application.py` — Ergebnisse sind direkt vergleichbar.

---

## Echtzeit-Anwendung (`RPI_application.py`)

### Konfiguration

Am Anfang der Datei eine Zeile ändern:

```python
MODEL_SIZE = 640   # 512 | 640 | 800  →  lädt <MODEL_SIZE>px.hef
```

### Kamera-Modi

| Modus | Auflösung | FPS | Empfohlen für |
|---|---|---|---|
| `1280x720@60` | 1280×720 | 60 | Standard, niedrige Latenz |
| `1536x864@30` | 1536×864 | 30 | Fernsicht, mehr Detail |
| `800x600@90` | 800×600 | 90 | Maximale FPS |

### Web-UI Funktionen

- **Kamera-Auflösung** live umschalten (ca. 1,5 s Unterbrechung)
- **Konfidenz-Schwelle** per Slider (Standard: 45%)
- **Infer every N** — nur jeden N-ten Frame inferieren (1–6)
- **Debounce** — benötigte Treffer für eine Bestätigung (1–8)
- **KI-Auge** — streamt den exakten 640×640-Tensor der an den Chip geht
- **ROI-Crop** — schneidet die unteren 30% ab (Motorhaube ausblenden)
- **JSON-Status** unter `/status`

### Limit-Logik (State Machine)

Die `SpeedStateMachine` verwaltet neben dem aktuellen Limit einen internen Fahrkontext (`innerorts` / `außerorts` / `unbekannt`). Der Kontext wird nur intern verwendet — er ist nicht im Stream sichtbar — und dient ausschließlich dazu, Aufhebebilder korrekt aufzulösen:

- Aufhebeschild nach Ortsschild → **30 km/h**
- Aufhebeschild nach Autobahnschild → **100 km/h**

### Temporal Debouncer

Einzelne verlorene Frames (Hailo-Flackern) werden toleriert. Bei `buffer_size=5` und `required_hits=3` reicht das Muster `[50, 50, None, 50, 50]` für eine Bestätigung.

---

## Trainings-Ergebnisse bewerten (`compare_models.py`)

```bash
python compare_models.py
```

Durchsucht `runs/` rekursiv nach allen `results.csv`-Dateien und erstellt ein Ranking nach mAP50-95:

```
RANK  NAME              MODEL       RES   EPOCH  mAP50   mAP50-95   BEWERTUNG
1     prod_s_640px      Small (s)   640   187    0.8821  0.7634     ⭐ Exzellent
2     dry_s_640px       Small (s)   640   1      0.1203  0.0891     ❌ Schwach
```

| mAP50-95 | Bewertung |
|---|---|
| ≥ 0.75 | ⭐ Exzellent — Produktionsreif |
| ≥ 0.65 | ✅ Gut — Solide für den Einsatz |
| ≥ 0.50 | ⚠ Mittelmäßig — Unsicher bei 130 km/h |
| < 0.50 | ❌ Schwach — Mehr Daten oder weniger Augmentierung |

---

## Technische Hinweise

**ONNX-Export für den Hailo DFC**
Der Export erfolgt zwingend mit `opset=11`, `half=False` (FP32) und `dynamic=False` (statische Shapes). Direkt nach dem Export prüft `onnx.checker.check_model()` den Graph — ein fehlerhafter Graph führt im DFC zu kryptischen Fehlermeldungen ohne Zeilenangabe.

**BGR/RGB auf dem Pi**
Picamera2 liefert trotz `BGR888`-Konfiguration intern RGB-Daten. `CameraStream` konvertiert jeden Frame sofort nach `capture_array()` zu echtem BGR (`cv2.COLOR_RGB2BGR`). `preprocess()` konvertiert dann BGR→RGB bevor der Tensor an den Hailo-Chip geht. Das `.npy`-Kalibrierungsset ist ebenfalls in RGB gespeichert — konsistent mit dem was der Chip zur Laufzeit sieht.

**Reproduzierbarkeit**
`set_seed(42)` setzt Python-, NumPy-, PyTorch- und CUDA-Zufallsgeneratoren. `torch.backends.cudnn.deterministic = True` macht das Training vollständig reproduzierbar (auf Kosten von ~5–10% Trainingsgeschwindigkeit).

**Namenskollisionen im Datensatz**
`split_dataset.py` benennt Dateien mit dem relativen Pfad als Präfix um (`Unterordner_00001.jpg`), um stille Überschreibungen bei Datensätzen mit identischen Dateinamen (z.B. GTSDB) zu verhindern.

---

## Abhängigkeiten

Vollständige Installation via `setup_venv.bat`. Kernpakete:

| Paket | Version | Zweck |
|---|---|---|
| torch | 2.5.1+cu121 | Training (CUDA 12.1) |
| ultralytics | 8.3.x | YOLOv8 |
| albumentations | 1.4.18 | Offline-Augmentierung |
| onnx | 1.20.0 | Export + Validierung |
| onnxruntime-gpu | 1.20.1 | ONNX-Inferenz (optional) |
| opencv-python | 4.10.0.84 | Bildverarbeitung |
| numpy | 1.26.4 | Array-Operationen (< 2.0 für ONNX-Kompatibilität) |
| mss | — | Bildschirmaufnahme (`PC_application.py`) |

Auf dem Raspberry Pi: `picamera2`, `hailo` (HailoRT 4.20.0), `opencv-python`, `numpy`.
