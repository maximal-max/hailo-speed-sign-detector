# Echtzeit-Verkehrsschilderkennung — Raspberry Pi 5 + Hailo-8

Echtzeit-Erkennung von Tempolimits und Verkehrsschildern mit YOLO (v8 / v11), optimiert für den Hailo-8 KI-Beschleuniger. Das System erkennt 20 Schildklassen — von Tempolimits über Ortsschilder bis hin zu Autobahnschildern — bei Geschwindigkeiten bis 130 km/h.

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

## Plattform-Architektur

Das Projekt ist bewusst auf zwei Plattformen aufgeteilt. **Nur zwei Dateien laufen auf dem Raspberry Pi** — alles andere gehört auf den Windows-Entwicklungsrechner.

| Schritt | Datei | Plattform |
|---|---|---|
| Umgebung einrichten | `setup_venv.bat` | Windows |
| Datensatz aufteilen | `split_dataset.py` | Windows |
| Modell trainieren + exportieren | `train_yolo.py` | Windows (CUDA) |
| Kalibrierungsset erstellen | `generate_universal_calib.py` | Windows |
| Modell kompilieren (`.hef`) | Hailo Cloud Compiler | Web / Windows |
| Modell auf dem PC testen | `PC_application.py` | Windows |
| Trainingsläufe vergleichen (Terminal) | `compare_models_advanced.py` | Windows |
| Trainingsläufe vergleichen (Grafik) | `compare_models_visual.py` | Windows |
| **Inferenz (Entwicklung)** | **`RPI_debug.py`** | **Raspberry Pi** |
| **Inferenz (Produktion)** | **`RPI_deploy.py`** | **Raspberry Pi** |

### Übergabe PC → Raspberry Pi

Nach dem Training und der Hailo-Kompilierung müssen folgende Dateien auf den Pi übertragen werden:

```
models/active_hef/<modell>.hef    ← kompiliertes Hailo-Modell
assets/                           ← Schild-PNGs für die Anzeige
```

Alle anderen Dateien (Datensatz, Trainings-Checkpoints, ONNX-Modell, Kalibrierungsbilder) verbleiben auf dem Windows-PC.

---

## Projektstruktur

```
hailo-speed-sign-detector/
│
│  ── Windows (Entwicklung & Training) ──────────────────────
├── setup_venv.bat                  ← Trainingsumgebung einrichten
├── tempolimits.yaml                ← Zentrale Konfig: Klassen + Datenpfade
│
├── split_dataset.py                ← Schritt 1: Rohdaten aufteilen (train/val)
├── train_yolo.py                   ← Schritt 2: Training + ONNX-Export
├── generate_universal_calib.py     ← Schritt 3: Kalibrierungsset für Hailo DFC
│
├── PC_application.py               ← Modell auf dem PC testen (Webcam / Screen)
├── compare_models_advanced.py      ← Trainingsläufe vergleichen (Terminal)
├── compare_models_visual.py        ← Trainingsläufe vergleichen (PNG/JPG-Export)
│
│  ── Raspberry Pi (Inferenz) ────────────────────────────────
├── RPI_debug.py                    ← Echtzeit-Inferenz + Web-UI (Entwicklung)
└── RPI_deploy.py                   ← Vollbild-GUI, kein Webserver (Produktion)
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

### 3. `tempolimits.yaml` prüfen

```yaml
path: datasets/sorted_dataset   # relativ zur YAML-Datei, kein Anpassen nötig
train: images/train
val:   images/val

names:
  0: Tempolimit_20
  1: Tempolimit_30
  # ...
```

Der Pfad ist relativ zur Position der `tempolimits.yaml` und funktioniert auf jedem Rechner ohne Änderung, solange die Ordnerstruktur erhalten bleibt.

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

**Debug-Modus** (Entwicklung, Web-UI mit Live-Stream):

```bash
/usr/bin/python3 RPI_debug.py
```

Web-UI: `http://<PI-IP>:8080`

**Deploy-Modus** (Produktion, Vollbild-GUI ohne Webserver):

```bash
/usr/bin/python3 RPI_deploy.py
```

Zeigt nur das erkannte Temposchild als PNG im Vollbild — kein Kamerabild, keine Bounding Boxes, kein HTTP-Server. Startet mit einem 10-Sekunden-Disclaimer.

---

## Pipeline-Details

### Augmentierung (`train_yolo.py`)

Die Offline-Augmentierung simuliert gezielt die Bedingungen bei Autobahnfahrt:

| Effekt | Parameter | Zweck |
|---|---|---|
| Motion Blur | Kernel (13–21), 60 % | 130 km/h Fahrtgeschwindigkeit |
| Affine Shear | Y: ±4°, X: ±3°, Rot: ±4°, 60 % | Rolling-Shutter-Verzerrung (IMX708) |
| Gauß-Rauschen | std_range (0.025–0.048), 60 % | High-ISO-Rauschen (≈ ISO 1600–6400) |
| Farb-Jitter | Hue ±5°, Sat/Val ±20, 60 % | Lichtverhältnisse, Tageszeit |
| Graustufen | 8 % | IMX708 bei sehr schlechtem Licht |
| Helligkeit/Kontrast | 50 % | Tunnel, Gegenlicht |

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
| Screen Capture | Bildschirmaufnahme via `mss`-Bibliothek (schnell: `frombuffer` + BGR-Slice) |

### Oberfläche (v3)

- **Ein Fenster**: Kamerabild und Dashboard werden zu einem einzigen Fenster zusammengefügt (`Camera | Dashboard`)
- **Automotive HUD**: Dunkles Cockpit-Design mit farbkodierten Geschwindigkeitszonen
- **Interaktive Dashboard-Schieber**: Konfidenz, Stabile Frames, Infer-Every-N, Mindest-Boxgröße und Zentriert-Toggle — per Maus klicken & ziehen, keine OpenCV-Trackbars
- **IMG_SIZE automatisch**: wird beim Start aus den Modell-Metadaten gelesen, muss nicht manuell gesetzt werden
- **FPS-Anzeige**: Getrennte Messung für Kamera und Inferenz (Inference läuft asynchron im Hintergrund-Thread)
- **Keyboard-Shortcuts**: `q` / `ESC` = Beenden, `r` = Reset, `s` = Screenshot

### Identische Logik

`PC_application.py` verwendet dieselbe `SpeedStateMachine` und denselben `TemporalDebouncer` wie `RPI_debug.py` und `RPI_deploy.py` — Ergebnisse sind direkt vergleichbar.

---

## Debug-Anwendung (`RPI_debug.py`)

Entwicklungs- und Testversion mit vollem Web-UI. Nachfolger der früheren `RPI_application.py`.

### Modell-Erkennung

Das `.hef`-Modell wird automatisch aus `models/active_hef/` geladen — keine manuelle Pfad-Konfiguration nötig. Liegt genau eine `.hef`-Datei in diesem Ordner, wird sie verwendet. Bei mehreren Dateien wird die erste (alphabetisch) gewählt und eine Warnung ausgegeben.

```
models/
└── active_hef/
    └── 640px.hef    ← wird automatisch erkannt
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
- **KI-Auge** — streamt den exakten Eingabe-Tensor, der an den Chip geht (inkl. Letterboxing)
- **ROI-Crop** — schneidet die unteren 30% ab (Motorhaube ausblenden)
- **JSON-Status** unter `/status`

### Limit-Logik (State Machine)

Die `SpeedStateMachine` verwaltet neben dem aktuellen Limit einen internen Fahrkontext (`innerorts` / `außerorts` / `unbekannt`). Der Kontext wird nur intern verwendet — er ist nicht im Stream sichtbar — und dient ausschließlich dazu, Aufhebebilder korrekt aufzulösen:

- Aufhebeschild nach Ortsschild → **30 km/h**
- Aufhebeschild nach Autobahnschild → **100 km/h**

### Temporal Debouncer

Einzelne verlorene Frames (Hailo-Flackern) werden toleriert. Bei `buffer_size=5` und `required_hits=3` reicht das Muster `[50, 50, None, 50, 50]` für eine Bestätigung.

---

## Deploy-Anwendung (`RPI_deploy.py`)

Produktionsversion für den Fahrzeugeinsatz — kein Webserver, kein Kamerabild, kein Netzwerk-Stream.

### Konzept

Das Skript zeigt ausschließlich das erkannte Temposchild als PNG im Vollbild an. Kein Kamerabild, keine Bounding Boxes, keine Debug-Overlays. Gedacht für den montierten Betrieb an einem Display im Fahrzeug.

### Konfiguration (am Dateianfang)

```python
CONFIDENCE_THRESHOLD = 0.45   # Mindest-Konfidenz
CAMERA_MODE    = "1280x720@60"
ROI_CROP       = False         # untere 30% abschneiden (Fahrzeug-Montage)
INFER_EVERY_N  = 2             # jeden N-ten Frame inferieren
DEBOUNCE_COUNT = 3             # Treffer bis zur Bestätigung
FULLSCREEN     = True          # False = Fenster 800×480
DISCLAIMER_SECONDS = 10        # Dauer des Start-Disclaimers
```

Das Modell wird ebenfalls automatisch aus `models/active_hef/` geladen (identische Logik wie `RPI_debug.py`).

### Start-Disclaimer

Beim Start erscheint für 10 Sekunden ein Hinweistext mit Countdown-Balken. Erst danach beginnt die Echtzeit-Erkennung.

---

## Trainings-Ergebnisse bewerten

### Terminal-Vergleich (`compare_models_advanced.py`)

```bash
python compare_models_advanced.py              # Standard (Score-Sortierung)
python compare_models_advanced.py --export     # + CSV-Export aller Metriken
python compare_models_advanced.py --top 5      # Nur Top-5 anzeigen
python compare_models_advanced.py --no-cards   # Nur Tabelle, keine Detailkarten
python compare_models_advanced.py --sort map50 # Sortierung ändern
```

Durchsucht `runs/` rekursiv und gibt aus:
- **Ranking-Tabelle** mit mAP50-95, mAP50, Precision, Recall, F1, gewichtetem Score und Effizienz
- **Detailkarten** je Lauf (Verluste, ASCII-Balken, Delta zu Platz 1)
- **Sub-Rankings** pro Einzel-Metrik (Top 5)

Der gewichtete **Gesamt-Score** kombiniert alle Metriken:

```
Score = 35% × mAP50-95  +  25% × mAP50  +  15% × Precision  +  15% × Recall  +  10% × F1
```

| mAP50-95 | Bewertung |
|---|---|
| ≥ 0.85 | ★★★ Exzellent — Produktionsreif |
| ≥ 0.70 | ★★☆ Sehr gut |
| ≥ 0.55 | ★★☆ Gut |
| ≥ 0.40 | ★☆☆ Mittel — Unsicher bei 130 km/h |
| < 0.40 | ☆☆☆ Schwach — Mehr Daten oder weniger Augmentierung |

### Grafischer Vergleich (`compare_models_visual.py`)

```bash
python compare_models_visual.py                  # Gesamt + Einzelbilder (Standard)
python compare_models_visual.py --no-split       # Nur Gesamt-Bild
python compare_models_visual.py --top 8          # Nur Top-N Modelle
python compare_models_visual.py --dpi 180        # Höhere Auflösung
python compare_models_visual.py --fmt jpg        # Format: png (Standard) | jpg
```

Speichert alle Bilder in einem datierten Unterordner (`model_reports/comparison_YYYY-MM-DD_HH-MM/`):

| Datei | Inhalt |
|---|---|
| `overview_*.png` | Gesamt-Ansicht: Tabelle + Radar-Chart + Score-Gauge |
| `table_*.png` | Ranking-Tabelle als Einzelbild |
| `radar_*.png` | Radar-Chart (höhere DPI) — alle 5 Metriken im Vergleich |
| `score_*.png` | Score-Gauge mit Berechnungsformel |

---

## Technische Hinweise

**ONNX-Export für den Hailo DFC**
Der Export erfolgt zwingend mit `opset=11`, `half=False` (FP32) und `dynamic=False` (statische Shapes). Direkt nach dem Export prüft `onnx.checker.check_model()` den Graph — ein fehlerhafter Graph führt im DFC zu kryptischen Fehlermeldungen ohne Zeilenangabe.

**BGR/RGB auf dem Pi**
Picamera2 liefert trotz `BGR888`-Konfiguration intern RGB-Daten. `CameraStream` konvertiert jeden Frame sofort nach `capture_array()` zu echtem BGR (`cv2.COLOR_RGB2BGR`). `preprocess()` konvertiert dann BGR→RGB bevor der Tensor an den Hailo-Chip geht. Das `.npy`-Kalibrierungsset ist ebenfalls in RGB gespeichert — konsistent mit dem, was der Chip zur Laufzeit sieht. Diese Pipeline gilt identisch für `RPI_debug.py` und `RPI_deploy.py`.

**Reproduzierbarkeit**
`set_seed(42)` setzt Python-, NumPy-, PyTorch- und CUDA-Zufallsgeneratoren. `torch.backends.cudnn.deterministic = True` macht das Training vollständig reproduzierbar (auf Kosten von ~5–10% Trainingsgeschwindigkeit).

**Namenskollisionen im Datensatz**
`split_dataset.py` benennt Dateien mit dem relativen Pfad als Präfix um (`Unterordner_00001.jpg`), um stille Überschreibungen bei Datensätzen mit identischen Dateinamen (z.B. GTSDB) zu verhindern.

---

## Abhängigkeiten

Vollständige Installation via `setup_venv.bat`. Kernpakete:

> **Albumentations 2.0:** Das Projekt verwendet Albumentations ≥ 2.0.8. Gegenüber 1.4.x wurden drei breaking changes behoben: `GaussNoise(var_limit)` → `std_range`, `ISONoise` (entfernt) → `HueSaturationValue`, `PadIfNeeded(value)` → `fill`.

| Paket | Version | Zweck |
|---|---|---|
| torch | 2.5.1+cu121 | Training (CUDA 12.1) |
| ultralytics | 8.3.x | YOLOv8 / YOLO11 |
| albumentations | >=2.0.8 | Offline-Augmentierung |
| onnx | 1.20.0 | Export + Validierung |
| onnxruntime-gpu | 1.20.1 | ONNX-Inferenz (optional) |
| opencv-python | 4.10.0.84 | Bildverarbeitung |
| numpy | 1.26.4 | Array-Operationen (< 2.0 für ONNX-Kompatibilität) |
| mss | — | Bildschirmaufnahme (`PC_application.py`) |

Auf dem Raspberry Pi: `picamera2`, `hailo` (HailoRT 4.20.0), `opencv-python`, `numpy`.
