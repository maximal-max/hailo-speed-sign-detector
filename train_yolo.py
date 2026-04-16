"""
==============================================================================
YOLO Training & Export (Hailo-8 Optimiert / High-Speed)
==============================================================================

HARDWARE:
  • Host: Raspberry Pi 5
  • KI:   Hailo-8 (26 TOPS) via AI HAT+
  • Cam:  IMX708 (Rolling Shutter & High-ISO berücksichtigt)

SZENARIO:
  • Echtzeit-Erkennung von Verkehrsschildern (Stadt & Autobahn bis 130 km/h).
  • Fokus auf Robustheit gegen Motion-Blur, Rauschen und kleine Objekte auf Distanz.

PIPELINE-ÜBERBLICK:
  1. Augmentiertes Dataset bauen  →  build_augmented_dataset()
  2. Training YAML erzeugen       →  create_training_yaml()
  3. YOLO    Training             →  model.train()
  4. ONNX-Export + Validierung    →  export_model.export() + onnx.checker

  Kalibrierungsset für den Hailo DFC → separates Skript: generate_universal_calib.py
  Dieses Skript liest direkt aus dem Roh-Datensatz und erzeugt JPGs + .npy
  für alle gewünschten Modellgrößen (512 / 640 / 800 px).

WICHTIGE DESIGN-ENTSCHEIDUNGEN:
  • Offline-Augmentierung (Albumentations) statt reine YOLO-Online-Augmentierung,
    damit Motion-Blur und Rolling-Shutter-Effekte exakt steuerbar sind.
  • ONNX-Export: Opset 11, FP32, statische Shapes — Pflicht für den Hailo DFC.
  • ONNX-Validierung direkt nach dem Export fängt Graph-Fehler ab, bevor die
    Datei in den Hailo DFC hochgeladen wird.

VERWENDUNG:
  1. USER_YAML_FILE auf deine tempolimits.yaml zeigen lassen.
  2. DRY_RUN = True zum Testen (1 Epoche, 64 Bilder, kein Augmentierungs-Mult).
  3. DRY_RUN = False für das echte Training.
  4. Ausgabe: ONNX + Kalibrierungsordner → direkt an Hailo DFC übergeben.

==============================================================================
"""

from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path
import os
import shutil
import cv2
import random
import torch
import traceback
import albumentations as A
from tqdm import tqdm
import yaml
import numpy as np


# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Pfad zur zentralen Klassen-/Datensatz-Konfiguration
USER_YAML_FILE = "tempolimits.yaml"

# DRY-RUN-MODUS
# True  → Schneller Funktionstest: 64 Bilder, 1 Epoche, keine Augmentierungs-Vervielfachung.
# False → Echtes Training mit allen Bildern und vollen Epochen.
DRY_RUN = False

# Seed für vollständige Reproduzierbarkeit (Training, Augmentierung, Shuffle)
SEED = 42

# --- Modell & Training ---
NAME       = "640px_YOLO11m"   # Unterordner-Name in RUNS_DIR
MODEL_BASE = "yolo11m.pt"      # Basis-Gewichte (werden automatisch heruntergeladen)
IMG_SIZE   = 640                # Eingabegröße in Pixel (quadratisch); muss mit HEF übereinstimmen
EPOCHS     = 1   if DRY_RUN else 200
PATIENCE   = 50                 # Early-Stopping: Abbruch nach N Epochen ohne Verbesserung
BATCH_SIZE = -1                 # -1 = Ultralytics wählt automatisch anhand des VRAM
WORKERS    = min(os.cpu_count() or 4, 8)  # DataLoader-Threads (max. 8)

# --- Augmentierung ---
# AUG_MULT: Wie oft jedes Trainingsbild zusätzlich augmentiert wird.
# Gesamtgröße Train = Originalbilder × (1 + AUG_MULT)
AUG_MULT = 0 if DRY_RUN else 2

# --- Pfade ---
ROOT     = Path.cwd()
DATA_AUG = ROOT / "datasets" / "augmentation_dataset"      # Augmentiertes Dataset
RUNS_DIR = ROOT / "runs"                                   # YOLO-Trainings-Output


# ==============================================================================
# AUGMENTIERUNGS-PARAMETER
# ==============================================================================

# 1. MOTION BLUR (simuliert Fahrzeugeschwindigkeit bis 130 km/h)
#    Aggressiv (130 km/h): (15, 31) | Mittel: (9, 21) | Sanft: (5, 11)
AUG_BLUR_LIMIT = (13, 21)   # Kernel-Größe (muss ungerade sein → Albumentations rundet auf)
AUG_BLUR_PROB  = 0.6         # Wahrscheinlichkeit pro Bild (0.6 = 60%)

# 2. ROLLING SHUTTER / VIBRATION (simuliert Kamera-Erschütterungen)
#    Aggressiv: (-5, 5) | Mittel: (-3, 3) | Sanft: (-1, 1)
AUG_SHEAR_Y   = (-4, 4)    # Vertikales Kippen in Grad
AUG_SHEAR_X   = (-3, 3)    # Horizontales Verziehen in Grad
AUG_ROTATION  = (-4, 4)    # Leichte Gesamtdrehung in Grad
AUG_GEOM_PROB = 0.6         # Wahrscheinlichkeit pro Bild

# 3. SENSOR-RAUSCHEN (simuliert High-ISO / Nacht / Regen)
#    Albumentations >=2.0 nutzt std_range: Std-Abweichung normiert auf [0.0, 1.0],
#    wobei 1.0 = Pixelmax (255 bei uint8).
#    Umrechnung: std_norm = sqrt(var_pixel) / 255
#      → var=40  → std=6.3  → normiert≈0.025  (≈ IMX708 bei ISO 1600)
#      → var=150 → std=12.2 → normiert≈0.048  (≈ IMX708 bei ISO 6400)
#    Aggressiv: (0.035, 0.056) | Mittel: (0.022, 0.039) | Sanft: (0.012, 0.028)
AUG_NOISE_STD  = (0.025, 0.048)  # Std-Abweichung normiert; entspricht ISO 1600–6400 am IMX708
AUG_NOISE_PROB = 0.6              # Wahrscheinlichkeit pro Bild

# 4. NACHT-/MONOCHROM-SIMULATION (IMX708 in schlechten Lichtverhältnissen)
#    Geringe Wahrscheinlichkeit – nur für Diversität des Trainingssets.
#    num_output_channels=3 zwingend: YOLO erwartet 3-Kanal-Tensoren.
AUG_GRAY_PROB = 0.08              # 8% der Bilder werden in Graustufen konvertiert


# ==============================================================================
# SEEDING
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Setzt alle relevanten Zufallsgeneratoren für vollständige Reproduzierbarkeit.
    Hinweis: deterministic=True im Training kann die Trainingsgeschwindigkeit leicht
    reduzieren, ist für Vergleichbarkeit von Experimenten aber empfehlenswert.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # Nötig bei Multi-GPU-Setup
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark=True ist schneller, aber nicht deterministisch
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================================================================
# METRIKEN-AUSWERTUNG
# ==============================================================================

def print_metrics(results) -> None:
    """
    Liest die Trainingsmetriken aus dem results-Objekt und gibt eine
    strukturierte Bewertung aus. Hauptkriterium ist mAP50-95, da dieser Wert
    die Präzision der Bounding-Box-Lokalisation über mehrere IoU-Schwellen misst.

    Bewertungsschwellen (erfahrungsbasiert für Verkehrsschilder):
      < 0.50  → Modell zu schwach für den Einsatz
      < 0.65  → Akzeptabel für Tests, unsicher bei hoher Geschwindigkeit
      < 0.75  → Solide Produktionsqualität
      ≥ 0.75  → Exzellent, reif für den Einsatz
    """
    if not hasattr(results, 'results_dict'):
        print("⚠ Keine Metriken im results-Objekt gefunden.")
        return

    metrics  = results.results_dict
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    map50    = metrics.get("metrics/mAP50(B)",    0.0)

    print("\n" + "=" * 60)
    print("📊 AUTOMATISCHE MODELL-ANALYSE")
    print("-" * 60)
    print(f"   mAP @ 50-95:  {map50_95:.4f}  (Box-Präzision über alle IoU-Schwellen)")
    print(f"   mAP @ 50:     {map50:.4f}  (Erkennungsrate bei IoU ≥ 0.50)")
    print("-" * 60)

    if map50_95 < 0.50:
        print("   BEWERTUNG: ❌ SCHWACH")
        print("   → Mehr Bilder sammeln oder Augmentierung reduzieren.")
    elif map50_95 < 0.65:
        print("   BEWERTUNG: ⚠  MITTELMÄSSIG")
        print("   → Okay für Labortests, bei 130 km/h nicht verlässlich.")
    elif map50_95 < 0.75:
        print("   BEWERTUNG: ✅ GUT")
        print("   → Solide für den Produktionseinsatz.")
    else:
        print("   BEWERTUNG: ⭐ EXZELLENT")
        print("   → Modell ist reif für den Hailo-Einsatz.")
    print("=" * 60 + "\n")


# ==============================================================================
# AUGMENTIERUNGS-PIPELINES
# ==============================================================================

def pipeline_train() -> A.Compose:
    """
    Augmentierungs-Pipeline für Trainingsbilder.

    Reihenfolge der Transformationen ist bewusst gewählt:
      1. Motion Blur     → simuliert Bewegungsunschärfe durch Fahrtgeschwindigkeit
      2. Geometrie       → simuliert Rolling-Shutter-Verzerrung und Kameravibrationen
      3. Nacht/Mono      → simuliert IMX708 in schlechten Lichtverhältnissen (selten)
      4. Sensor-Rauschen → simuliert High-ISO / schlechte Lichtverhältnisse
      5. Helligkeit      → allgemeine Lichtschwankungen (Tunnel, Gegenlicht)
      6. Letterbox       → auf IMG_SIZE skalieren ohne Seitenverhältnis zu verzerren

    Alle geometrischen Transforms laufen VOR dem Resize, da Albumentations
    die Bounding-Boxes dabei automatisch mitverschiebt.

    BboxParams:
      min_visibility=0.1  → Box wird behalten, solange ≥10% ihrer Fläche sichtbar ist.
                            Bewusst niedrig: Schilder bei 150-200m sind winzig;
                            "partial visibility"-Samples verbessern die Robustheit.
                            0.3 wäre zu aggressiv und würde Randschilder nach Affine
                            systematisch verwerfen.
      check_each_transform=True → Albumentations 2.0: BBox-Validierung nach jedem
                            einzelnen Transform, nicht erst am Ende. Verhindert,
                            dass ungültige Zwischenzustände (z.B. negative Breite
                            nach Affine) stillschweigend propagiert werden.
    """
    return A.Compose([
        # 1. Bewegungsunschärfe: entweder gerichteter Motion-Blur oder isotroper Gauss-Blur
        A.OneOf([
            A.MotionBlur(blur_limit=AUG_BLUR_LIMIT, p=0.6),
            A.GaussianBlur(blur_limit=(5, 9), p=0.2),
        ], p=AUG_BLUR_PROB),

        # 2. Geometrische Verzerrung: Affine-Transform kombiniert Shear, Scale und Rotation
        A.OneOf([
            A.Affine(
                shear={'y': AUG_SHEAR_Y, 'x': AUG_SHEAR_X},
                scale=(0.95, 1.05),
                rotate=AUG_ROTATION,
                p=0.6
            ),
        ], p=AUG_GEOM_PROB),

        # 3. Nacht-/Monochrom-Simulation: IMX708 bei sehr schlechtem Licht
        #    num_output_channels=3: YOLO erwartet 3-Kanal-Tensoren (RGB), kein Grauwert-Tensor.
        #    Position nach Geometrie, vor Rauschen: Noise wird dann realistisch auf dem
        #    bereits entsättigten Bild addiert (so wie ein echter Nacht-Sensor arbeitet).
        A.ToGray(num_output_channels=3, p=AUG_GRAY_PROB),

        # 4. Sensor-Rauschen: Gauß-Rauschen oder Farb-/Helligkeitsjitter
        #    ISONoise wurde in Albumentations 2.0 entfernt → HueSaturationValue
        #    ersetzt den Farbrauschen-Anteil von ISONoise.
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussNoise(std_range=AUG_NOISE_STD, p=0.5),
        ], p=AUG_NOISE_PROB),

        # 5. Helligkeits- und Kontrastanpassung
        A.RandomBrightnessContrast(p=0.5),

        # 6. Letterbox-Resize: Seitenverhältnis bleibt erhalten, Rest wird mit Wert 114 aufgefüllt.
        #    114 = YOLOv11-Standard (ultralytics/utils/ops.py letterbox()).
        #    MUSS identisch zu pipeline_val_clean() und RPI_application.py sein,
        #    da dieser Wert die Basis für das Hailo-Kalibrierungsset bildet.
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=114
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['cls'],
        min_visibility=0.1,        # Box behalten wenn ≥10% sichtbar (wichtig für Fernschilder)
        check_each_transform=True, # v2.0: BBox-Validierung nach jedem Transform-Schritt
    ))


def pipeline_val_clean() -> A.Compose:
    """
    Minimale Pipeline für Validierungsbilder und Kalibrierungsbilder.

    Nur Letterbox-Resize, keine Augmentierung. Diese Pipeline wird an
    drei Stellen identisch verwendet:
      - Validierungsbilder im Augmentierungs-Schritt (build_augmented_dataset)
      - Kalibrierungsbilder für den Hailo DFC → generate_universal_calib.py
      - Inferenz auf dem Pi (preprocess() in RPI_application.py)

    Alle drei müssen pixel-identisch sein, damit die INT8-Quantisierung
    des Hailo-Chips korrekt auf die echte Werteverteilung kalibriert wird.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=114
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['cls'],
        min_visibility=0.1,        # Konsistent mit pipeline_train(); Val-Boxes werden nicht getrimmt
        check_each_transform=True, # v2.0: BBox-Validierung nach jedem Transform-Schritt
    ))


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def load_user_config() -> dict:
    """Lädt die zentrale YAML-Konfiguration (Klassennamen, Datenpfade)."""
    yaml_path = ROOT / USER_YAML_FILE
    if not yaml_path.exists():
        print(f"❌ FEHLER: '{USER_YAML_FILE}' nicht gefunden in {ROOT}")
        exit(1)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_source_path(config: dict, split_key: str):
    """
    Ermittelt den absoluten Pfad zum Bild-Ordner eines Splits (train/val).
    Probiert mehrere Kandidaten durch, um sowohl absolute als auch relative
    Pfade in der YAML zu unterstützen.
    """
    root_path_str  = config.get('path', '')
    split_path_str = config.get(split_key)
    if not split_path_str:
        return None

    candidates = [
        Path(split_path_str),
        Path(root_path_str) / split_path_str,
        ROOT / split_path_str,
        (ROOT / USER_YAML_FILE).parent / split_path_str,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def create_training_yaml(user_config: dict) -> str:
    """
    Erzeugt eine temporäre YAML-Datei für das YOLO-Training.

    Die Original-YAML zeigt auf den Roh-Datensatz. Diese Funktion erstellt
    eine neue YAML, die auf das augmentierte Dataset (DATA_AUG) zeigt.
    Stellt außerdem sicher, dass 'nc' und 'names' korrekt formatiert sind,
    da YOLO beides als Liste erwartet.
    """
    new_config = dict(user_config)   # flache Kopie, Original bleibt unverändert
    new_config['path']  = str(DATA_AUG.absolute())
    new_config['train'] = 'images/train'
    new_config['val']   = 'images/val'
    new_config.pop('test', None)   # Test-Split für Training nicht benötigt

    # Namen normalisieren: YAML-dict ({0: 'X', 1: 'Y'}) → sortierte Liste ['X', 'Y']
    names = new_config.get('names') or new_config.get('labels') or {}
    if isinstance(names, dict):
        names_list = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        names_list = names
    else:
        names_list = list(names) if names else []

    new_config['names'] = names_list
    new_config['nc']    = len(names_list)

    outpath = DATA_AUG / "data_hailo_training.yaml"
    with open(outpath, 'w') as f:
        yaml.dump(new_config, f, sort_keys=False)
    print(f"✔ Trainings-YAML geschrieben: {outpath}  (nc={new_config['nc']})")
    return str(outpath)


def load_yolo_labels(path: Path) -> list:
    """
    Liest YOLO-Label-Datei (cls cx cy w h pro Zeile).
    Gibt leere Liste zurück bei fehlendem oder leerem File.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, 'r') as f:
            return [list(map(float, line.split())) for line in f if line.strip()]
    except ValueError:
        return []


def save_yolo_labels(path: Path, labels: list) -> None:
    """
    Speichert YOLO-Labels mit fixer 6-Stellen-Präzision.

    Fixer Präzisions-String verhindert wissenschaftliche Notation (z.B. 1e-05),
    die von manchen YOLO-Parsern nicht korrekt gelesen wird.
    Alle Koordinaten werden auf [0.0, 1.0] geclampt.
    """
    with open(path, 'w') as f:
        for label in labels:
            cls_id     = int(label[0])
            coords     = [round(max(0.0, min(1.0, float(c))), 6) for c in label[1:]]
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{cls_id} {coords_str}\n")


def apply_and_save(pipeline: A.Compose, img: np.ndarray, labels: list,
                   out_img_path: Path, out_lbl_path: Path,
                   skip_counter: list) -> None:
    """
    Wendet eine Albumentations-Pipeline auf Bild + Labels an und speichert beides.

    skip_counter ist eine einelementige Liste [n], die bei Fehlern inkrementiert
    wird — so kann der Aufrufer zählen, wie viele Bilder übersprungen wurden,
    ohne eine globale Variable zu benötigen.
    """
    bboxes  = [l[1:] for l in labels]
    cls_ids = [l[0]  for l in labels]
    try:
        res        = pipeline(image=img, bboxes=bboxes, cls=cls_ids)
        cv2.imwrite(str(out_img_path), res['image'])
        new_labels = [[res['cls'][i]] + list(res['bboxes'][i])
                      for i in range(len(res['bboxes']))]
        save_yolo_labels(out_lbl_path, new_labels)
    except Exception as e:
        print(f"⚠ Augmentierung übersprungen ({out_img_path.name}): {e}")
        skip_counter[0] += 1


# ==============================================================================
# DATASET-BUILDER
# ==============================================================================

def build_augmented_dataset(user_config: dict) -> None:
    """
    Baut das augmentierte Dataset aus dem Roh-Datensatz.

    Für jeden Split (train/val):
      • Original-Bilder werden mit pipeline_val_clean() (nur Letterbox) kopiert.
      • Trainingsbilder werden zusätzlich AUG_MULT-mal mit pipeline_train()
        augmentiert (Motion-Blur, Shear, Rauschen).
      • Validierungsbilder werden NIE augmentiert — nur Letterbox — damit
        die Validierungsmetriken die echte Erkennungsleistung widerspiegeln.

    Vorhandener DATA_AUG-Ordner wird vollständig gelöscht (Clean Slate).
    """
    if DATA_AUG.exists():
        shutil.rmtree(DATA_AUG)

    train_pipe = pipeline_train()
    val_pipe   = pipeline_val_clean()

    for split in ['train', 'val']:
        (DATA_AUG / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATA_AUG / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starte Dataset-Aufbau (Pfade aus '{USER_YAML_FILE}')...")
    if DRY_RUN:
        print("   ⚠ DRY-RUN AKTIV: Limitiere auf 64 Bilder pro Split, kein Aug-Mult.")

    total_images = 0
    skip_counter = [0]   # Einelementige Liste als mutabler Zähler für apply_and_save

    for split in ['train', 'val']:
        src_img_dir = resolve_source_path(user_config, split)
        if not src_img_dir:
            print(f"   ⚠ Kein Pfad für Split '{split}' gefunden — überspringe.")
            continue

        # Label-Ordner: erwartet parallele Struktur images/train ↔ labels/train
        if src_img_dir.parent.name == "images":
            src_lbl_dir = src_img_dir.parent.parent / "labels" / src_img_dir.name
        else:
            src_lbl_dir = src_img_dir

        if not src_lbl_dir.exists():
            src_lbl_dir = src_img_dir   # Fallback: Labels liegen neben Bildern

        print(f"   → {split.upper()}: {src_img_dir}")

        # Alle gängigen Bildformate einsammeln (case-insensitive durch doppeltes Glob)
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                      "*.JPG", "*.PNG", "*.JPEG"]
        files = sorted(set(f for ext in extensions for f in src_img_dir.glob(ext)))

        if DRY_RUN:
            files = files[:64]

        print(f"      Gefunden: {len(files)} Bilder")
        total_images += len(files)

        for img_path in tqdm(files, desc=f"  {split}"):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            img      = cv2.imread(str(img_path))
            if img is None:
                skip_counter[0] += 1
                continue
            labels = load_yolo_labels(lbl_path)

            dst_img = DATA_AUG / 'images' / split / img_path.name
            dst_lbl = DATA_AUG / 'labels' / split / (img_path.stem + ".txt")

            # Original immer mit der sauberen Val-Pipeline (nur Letterbox)
            apply_and_save(val_pipe, img, labels, dst_img, dst_lbl, skip_counter)

            # Augmentierte Kopien nur für den Train-Split und nicht im Dry-Run
            if split == 'train' and not DRY_RUN:
                for k in range(AUG_MULT):
                    aug_img_path = DATA_AUG / 'images' / split / f"{img_path.stem}_aug{k}.jpg"
                    aug_lbl_path = DATA_AUG / 'labels' / split / f"{img_path.stem}_aug{k}.txt"
                    apply_and_save(train_pipe, img, labels,
                                   aug_img_path, aug_lbl_path, skip_counter)

    if skip_counter[0] > 0:
        ratio = skip_counter[0] / max(total_images, 1)
        print(f"   ⚠ {skip_counter[0]} Bilder übersprungen ({ratio:.1%} des Datasets).")
        if ratio > 0.05:
            print("   ❌ ABBRUCH: Mehr als 5% der Bilder fehlerhaft — Datensatz prüfen!")
            exit(1)

    if total_images == 0:
        print("❌ KRITISCH: Keine Bilder gefunden! YAML-Pfade überprüfen.")
        exit(1)

    print(f"✔ Dataset fertig: {total_images} Originalbilder verarbeitet.")





# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    freeze_support()

    # --- Schritt 0: Seed setzen ---
    print(f"🌱 Seed: {SEED}")
    set_seed(SEED)

    if DRY_RUN:
        print("\n" + "!" * 60)
        print("  ⚠  DRY-RUN AKTIV — Nur Funktionsprüfung (1 Epoche, 64 Bilder)")
        print("!" * 60)

    # --- Schritt 1: Konfiguration laden ---
    print(f"\n📄 Lade Konfiguration: {USER_YAML_FILE}")
    user_config = load_user_config()

    # --- Schritt 2: Dataset aufbauen ---
    build_augmented_dataset(user_config)

    # --- Schritt 3: Temporäre Trainings-YAML erzeugen ---
    training_yaml_path = create_training_yaml(user_config)

    # --- Schritt 4: Training ---
    print("\n🚀 Starte YOLOv8s Training...")
    model = YOLO(MODEL_BASE)

    try:
        results = model.train(
            data=training_yaml_path,
            imgsz=IMG_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            workers=WORKERS,

            # Online-Augmentierung (ergänzend zur Offline-Augmentierung):
            # Farbe: leichte Variation für unterschiedliche Lichtverhältnisse
            hsv_h=0.01,   # Farbton-Variation (±1%)
            hsv_s=0.15,   # Sättigungs-Variation
            hsv_v=0.15,   # Helligkeits-Variation

            # Geometrie: nur was nicht offline abgedeckt ist
            translate=0.1,   # Leichtes Verschieben des Ausschnitts
            scale=0.25,      # Zoom-Variation (±25%)

            # Offline bereits abgedeckt → hier deaktivieren um Dopplung zu vermeiden
            shear=0.0,        # Rolling Shutter → offline
            perspective=0.0,  # Perspektive → offline
            degrees=0.0,      # Rotation → offline

            # Deaktiviert: würden Schilder-Texte spiegeln oder kombinieren
            flipud=0.0,   # Über-Kopf-Flip
            fliplr=0.0,   # Links-Rechts-Flip (Texte auf Schildern werden gespiegelt!)
            mosaic=0.0,   # 4-Bild-Mosaik (zu viele kleine Schilder auf einmal)
            mixup=0.0,    # Halb-Halb Überblendung (verwirrt bei Zahlen-Klassen)

            device=0 if torch.cuda.is_available() else "cpu",
            project=str(RUNS_DIR),
            name=NAME,
            exist_ok=False,    # Fehler wenn der Name bereits existiert → kein versehentl. Überschreiben
            verbose=True,
            plots=True,
            deterministic=True,   # Reproduzierbarkeit (etwas langsamer als benchmark=True)
        )

        print_metrics(results)

    except Exception as e:
        print(f"❌ Training fehlgeschlagen: {e}")
        traceback.print_exc()
        return

    # --- Schritt 5: Bestes Modell suchen ---
    best_pt = RUNS_DIR / NAME / "weights" / "best.pt"
    if not best_pt.exists() and hasattr(results, 'save_dir'):
        best_pt = Path(results.save_dir) / 'weights' / 'best.pt'

    if not best_pt.exists():
        print("❌ best.pt nicht gefunden — Export übersprungen.")
        return

    # --- Schritt 6: ONNX-Export für Hailo DFC ---
    print(f"\n📦 Exportiere '{best_pt}' als ONNX für Hailo DFC...")
    export_model = YOLO(str(best_pt))

    # Pflicht-Einstellungen für den Hailo DFC:
    #   opset=11   → maximale Kompatibilität mit dem DFC-ONNX-Parser
    #   half=False → FP32: der DFC quantisiert selbst auf INT8
    #   dynamic=False → statische Shapes: der DFC benötigt feste Batch-/Bildgrößen
    onnx_path = export_model.export(
        format="onnx",
        opset=11,
        simplify=True,
        imgsz=IMG_SIZE,
        half=False,
        dynamic=False,
    )

    # --- Schritt 7: ONNX-Graph validieren ---
    # Dieser Check fängt fehlerhafte Operator-Verbindungen, fehlende Shapes und
    # andere Graph-Fehler ab, BEVOR die Datei zum Hailo-Compiler hochgeladen wird.
    # Fehlerhafte Graphen führen im DFC zu kryptischen Fehlermeldungen ohne Zeilenangabe.
    print("\n🔍 Validiere ONNX-Graph...")
    try:
        import onnx as onnx_lib
        onnx_model = onnx_lib.load(str(onnx_path))
        onnx_lib.checker.check_model(onnx_model)
        print("✔ ONNX-Graph ist valide — bereit für den Hailo DFC.")
    except ImportError:
        print("⚠ onnx-Paket nicht installiert — Validierung übersprungen.")
        print("  → pip install onnx")
    except Exception as e:
        print(f"❌ ONNX-Validierung fehlgeschlagen: {e}")
        print("   → Datei NICHT an den Hailo DFC übergeben!")
        return

    # --- Abschluss ---
    print("\n" + "=" * 60)
    print("✅ PIPELINE ABGESCHLOSSEN — BEREIT FÜR DEN HAILO COMPILER")
    print("-" * 60)
    print(f"  ONNX-Modell:  {onnx_path}")
    print("-" * 60)
    print("  Nächste Schritte:")
    print("  1. generate_universal_calib.py ausführen → Kalibrierungsset erstellen.")
    print("  2. ONNX + Kalibrierungsbilder in den Hailo Cloud Compiler laden.")
    print("  3. Kompiliertes .hef auf den Pi 5 kopieren.")
    print("  4. MODEL_SIZE in RPI_application.py auf den passenden Wert setzen.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()