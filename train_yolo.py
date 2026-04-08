"""
==============================================================================
YOLOv8s Training & Export (Hailo-8 Optimized / High-Speed)
==============================================================================

HARDWARE:
  • Host: Raspberry Pi 5
  • AI:   Hailo-8 (26 TOPS) via AI HAT+
  • Cam:  IMX708 (Berücksichtigung von Rolling Shutter & High-ISO)

SZENARIO:
  • Echtzeit-Erkennung von Verkehrsschildern (Stadt & Autobahn bis 130 km/h).
  • Fokus auf Robustheit gegen Motion-Blur, Rauschen und kleine Objekte auf Distanz.

HIGHLIGHTS & TECHNIK:
  1. High-Speed Augmentation: Simuliert gezielt 130 km/h Motion-Blur & Rolling-Shutter-Scherung.
  2. Quantization Hardening: Injiziert Noise in die Kalibrierungsdaten, damit der Hailo-Chip
     bei verrauschten Nacht/Regen-Bildern nicht an Genauigkeit verliert.
  3. Precision Labels: Speichert Bounding-Boxen mit fixer 6-Stellen-Präzision.
  4. Hailo-Ready Export: Erzwungenes ONNX Opset 11, FP32 & Static Shapes (Zwingend für DFC).

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

# ===============================
# 🔧 KONFIGURATION
# ===============================

USER_YAML_FILE = "tempolimits.yaml" 

# DRY RUN MODUS (Zum Testen auf True setzen)
# Wenn True: Nimmt nur 20 Bilder, macht keine Augmentierung und trainiert nur 1 Epoche.
DRY_RUN = True

# Seed für Reproduzierbarkeit
SEED = 42

# Modell & Training
NAME = "s_640px"
MODEL_BASE = "yolov8s.pt"
IMG_SIZE = 640
EPOCHS = 1 if DRY_RUN else 200
PATIENCE = 50
BATCH_SIZE = -1
WORKERS = min(os.cpu_count() or 4, 8)

# Augmentation & Calibration
AUG_MULT = 0 if DRY_RUN else 2  # Im Dry-Run keine Vervielfachung
CALIB_IMAGES = 64 if DRY_RUN else 1024

# Pfade
ROOT = Path.cwd()
DATA_AUG = ROOT / "train_yolo_output" / "dataset_aug"
CALIB_DIR = ROOT / "train_yolo_output" / "calibration_hailo"
RUNS_DIR = ROOT  / "runs"

# ===============================
# AUGMENTATION SETTINGS 
# ===============================

# 1. GESCHWINDIGKEIT (Motion Blur)
# Aggressiv (130kmh): (15, 31) | Mittel: (9, 21) | Sanft: (5, 11)
AUG_BLUR_LIMIT = (13, 21)    # <-- Aktuell: Mittel (Guter Startwert)
AUG_BLUR_PROB  = 0.6        # Wahrscheinlichkeit (0.6 = 60% der Bilder)

# 2. VIBRATION & SENSOR (Rolling Shutter)
# Aggressiv: (-5, 5) | Mittel: (-3, 3) | Sanft: (-1, 1)
AUG_SHEAR_Y    = (-4, 4)    # Vertikales Kippen
AUG_SHEAR_X    = (-3, 3)    # Horizontales Verziehen
AUG_ROTATION   = (-4, 4)    # Leichte Drehung
AUG_GEOM_PROB  = 0.6        # Wahrscheinlichkeit

# 3. LICHT & SENSOR (Noise/ISO)
# Aggressiv: (50, 200) | Mittel: (30, 100) | Sanft: (10, 50)
AUG_NOISE_VAR  = (40, 150)  # Rausch-Stärke
AUG_NOISE_PROB = 0.6        # Wahrscheinlichkeit


# ===============================
# 🌱 SEEDING & HELPER
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # für Multi-GPU
    # Für absolute Reproduzierbarkeit (kann langsamer sein):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_metrics(results):
    """Analysiert das Trainingsergebnis und gibt eine Bewertung aus."""
    if not hasattr(results, 'results_dict'):
        print("⚠ Keine Metriken gefunden.")
        return

    metrics = results.results_dict
    # mAP50-95 ist der wichtigste Wert für die Gesamtqualität
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    map50    = metrics.get("metrics/mAP50(B)", 0.0)
    
    print("\n" + "="*60)
    print("📊 AUTOMATISCHE MODELL-ANALYSE")
    print("-" * 60)
    print(f"   mAP @ 50-95:  {map50_95:.4f}  (Präzision der Boxen)")
    print(f"   mAP @ 50:     {map50:.4f}  (Erkennungsrate)")
    print("-" * 60)
    
    if map50_95 < 0.50:
        print("   BEWERTUNG: ❌ SCHWACH")
        print("   -> Tipp: Mehr Bilder oder weniger aggressive Augmentierung.")
    elif map50_95 < 0.65:
        print("   BEWERTUNG: ⚠ MITTELMÄßIG")
        print("   -> Tipp: Okay für Tests, aber unsicher bei 130 km/h.")
    elif map50_95 < 0.75:
        print("   BEWERTUNG: ✅ GUT")
        print("   -> Tipp: Solide für den Einsatz.")
    else:
        print("   BEWERTUNG: ⭐ EXZELLENT")
        print("   -> Tipp: Das Modell ist reif für die Produktion.")
    print("="*60 + "\n")

# ===============================
# 🛠️ AUGMENTATION PIPELINE 
# ===============================

def pipeline_train():
    return A.Compose([
        # 1. Motion Blur
        A.OneOf([
            A.MotionBlur(blur_limit=AUG_BLUR_LIMIT, p=0.6), 
            A.GaussianBlur(blur_limit=(5, 9), p=0.2),
        ], p=AUG_BLUR_PROB),

        # 2. Rolling Shutter
        A.OneOf([
            A.Affine(
                shear={'y': AUG_SHEAR_Y, 'x': AUG_SHEAR_X}, 
                scale=(0.95, 1.05), 
                rotate=AUG_ROTATION, 
                p=0.6
            ), 
        ], p=AUG_GEOM_PROB),

        # 3. Sensor Noise
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5), 
            A.GaussNoise(var_limit=AUG_NOISE_VAR, p=0.5),
        ], p=AUG_NOISE_PROB),

        # 4. Standard
        A.RandomBrightnessContrast(p=0.5),
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=114)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls']))

def pipeline_val_clean():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=114)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls']))

# ===============================
# 📂 HELFER FUNKTIONEN
# ===============================

def load_user_config():
    yaml_path = ROOT / USER_YAML_FILE
    if not yaml_path.exists():
        print(f"❌ FEHLER: '{USER_YAML_FILE}' nicht gefunden!")
        exit(1)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def resolve_source_path(config, split_key):
    root_path_str = config.get('path', '')
    split_path_str = config.get(split_key)
    if not split_path_str: return None
    
    candidates = [
        Path(split_path_str),
        Path(root_path_str) / split_path_str,
        ROOT / split_path_str,
        (ROOT / USER_YAML_FILE).parent / split_path_str
    ]
    for p in candidates:
        if p.exists() and p.is_dir(): return p
    return None

def create_training_yaml(user_config):
    """
    Erzeugt die temporäre training YAML, stellt sicher dass 'nc' und 'names' korrekt sind.
    """
    new_config = dict(user_config)  # shallow copy
    new_config['path'] = str(DATA_AUG.absolute())
    new_config['train'] = 'images/train'
    new_config['val'] = 'images/val'
    if 'test' in new_config:
        del new_config['test']

    # Ensure names are a list and nc exists
    names = new_config.get('names') or new_config.get('labels') or {}
    if isinstance(names, dict):
        # Sortiere Keys, damit die Reihenfolge deterministisch ist
        names_list = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        names_list = names
    else:
        # fallback
        names_list = list(names) if names else []

    new_config['names'] = names_list
    new_config['nc'] = len(names_list)

    outpath = DATA_AUG / "data_hailo_training.yaml"
    with open(outpath, 'w') as f:
        yaml.dump(new_config, f, sort_keys=False)
    print(f"✔ Temporäre Trainings-YAML geschrieben: {outpath} (nc={new_config['nc']})")
    return str(outpath)

def load_yolo_labels(path):
    if not path.exists() or path.stat().st_size == 0: return []
    try:
        with open(path, 'r') as f:
            return [list(map(float, line.split())) for line in f]
    except ValueError: return []

def save_yolo_labels(path, labels):
    """Speichert Labels mit fixer Präzision (6 Stellen)."""
    with open(path, 'w') as f:
        for l in labels:
            cls_id = int(l[0])
            # Clampen auf 0.0 - 1.0 und Runden
            coords = [round(max(0.0, min(1.0, float(c))), 6) for c in l[1:]]
            # Explizite String-Formatierung verhindert wissenschaftliche Notation (1e-5)
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{cls_id} {coords_str}\n")

def apply_and_save(pipeline, img, labels, out_img_path, out_lbl_path):
    bboxes = [l[1:] for l in labels]
    cls_ids = [l[0] for l in labels]
    try:
        res = pipeline(image=img, bboxes=bboxes, cls=cls_ids)
        cv2.imwrite(str(out_img_path), res['image'])
        new_labels = [[res['cls'][i]] + list(res['bboxes'][i]) for i in range(len(res['bboxes']))]
        save_yolo_labels(out_lbl_path, new_labels)
    except Exception as e:
        print(f"⚠ Skip aug error: {e}")

# ===============================
# 🏗️ DATASET BUILDER
# ===============================

def build_augmented_dataset(user_config):
    if DATA_AUG.exists(): shutil.rmtree(DATA_AUG)
    
    train_pipe = pipeline_train()
    val_pipe = pipeline_val_clean()

    for split in ['train', 'val']:
        (DATA_AUG / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATA_AUG / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starte Augmentierung (Lese Pfade aus {USER_YAML_FILE})...")
    if DRY_RUN: print("⚠ DRY RUN AKTIV: Limitiere Bilder!")

    total_images = 0

    for split in ['train', 'val']:
        src_img_dir = resolve_source_path(user_config, split)
        if not src_img_dir: continue

        if src_img_dir.parent.name == "images":
             src_lbl_dir = src_img_dir.parent.parent / "labels" / src_img_dir.name
        else:
             src_lbl_dir = src_img_dir 

        if not src_lbl_dir.exists(): src_lbl_dir = src_img_dir
        print(f"   -> {split.upper()}: {src_img_dir}")
        
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.PNG", "*.JPEG"]
        files = []
        for ext in extensions:
            files.extend(list(src_img_dir.glob(ext)))
        files = sorted(list(set(files)))
        
        # DRY RUN LIMIT
        if DRY_RUN:
            files = files[:64] 

        print(f"      Gefunden: {len(files)} Bilder")
        total_images += len(files)

        for img_path in tqdm(files, desc=f"Processing {split}"):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None: continue
            labels = load_yolo_labels(lbl_path)
            
            dst_img = DATA_AUG / 'images' / split / img_path.name
            dst_lbl = DATA_AUG / 'labels' / split / lbl_path.name

            # Original
            apply_and_save(val_pipe, img, labels, dst_img, dst_lbl)

            # Augmentation (Nur Train)
            if split == 'train' and not DRY_RUN:
                for k in range(AUG_MULT):
                    aug_name = f"{img_path.stem}_aug{k}.jpg"
                    dst_img_aug = DATA_AUG / 'images' / split / aug_name
                    dst_lbl_aug = DATA_AUG / 'labels' / split / (img_path.stem + f"_aug{k}.txt")
                    apply_and_save(train_pipe, img, labels, dst_img_aug, dst_lbl_aug)
    
    if total_images == 0:
        print("❌ CRITICAL: Keine Bilder gefunden!")
        exit(1)

def build_calibration_set():
    """
    Erstellt Calibration Set.
    Noise Injection VOR dem Resize für realistischere Sensor-Simulation.
    """
    if CALIB_DIR.exists(): shutil.rmtree(CALIB_DIR)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    
    src_dir = DATA_AUG / 'images' / 'train'
    if not src_dir.exists(): return
    
    images = list(src_dir.glob("*.jpg"))
    random.shuffle(images)
    limit = min(len(images), CALIB_IMAGES)
    
    print(f"\n📸 Erstelle 'Hardened' Calibration Set ({limit} Bilder)...")
    
    for i, img_path in enumerate(tqdm(images[:limit], desc="Calib Injection")):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # --- NOISE VOR RESIZE (Natürlicher) ---
        # 1. Leichter Blur (30% Chance)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 2. Leichtes Gauss-Rauschen (20% Chance)
        if random.random() < 0.2:
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # -------------------------------------------------

        # Erst JETZT Resize auf 640x640
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(str(CALIB_DIR / f"calib_{i}.jpg"), img)

# ===============================
# 🏁 MAIN
# ===============================

def main():
    freeze_support()
    
    # 0. Seed setzen
    print(f"🌱 Setze Seed auf: {SEED}")
    set_seed(SEED)

    if DRY_RUN:
        print("\n⚠ ACHTUNG: DRY RUN MODUS IST AKTIV! (Nur kurze Funktionsprüfung)")

    # 1. Konfiguration laden
    print(f"\n📄 Lese Config: {USER_YAML_FILE}")
    user_config = load_user_config()
    
    # 2. Dataset bauen
    build_augmented_dataset(user_config)
    build_calibration_set()
    
    # 3. Training YAML erstellen (jetzt mit der korrekten Funktion!)
    training_yaml_path = create_training_yaml(user_config)

    print("\n🚀 Starte YOLOv8s Training...")
    model = YOLO(MODEL_BASE)

    try:
        results = model.train(
            data=training_yaml_path,
            imgsz=IMG_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            
            # --- Angepasste Augmentationseinstellungen ---
            hsv_h=0.01, hsv_s=0.15, hsv_v=0.15,  # Farbvariation
            translate=0.1,                       # Leichtes Verschieben 
            scale=0.25,                          # Zoom 
            
            # --- Schon durch offline Augmentation abgedeckt ---
            shear=0.0,                           # Rolling Shutter 
            perspective=0.0,                     # Perspektive   
            degrees=0.0,                         # Rotation  
            
            ## --- Deaktivierte Augmentierungen ---
            flipud=0.0,                          # Über-Kopf
            fliplr=0.0,                          # Spiegeln bei Schrift!
            mosaic=0.0,                          # 4-Bild-Mosaik
            mixup=0.0,                           # Halb-Halb Mixup
            
            device=0 if torch.cuda.is_available() else "cpu",
            project=str(RUNS_DIR),
            name=NAME,
            exist_ok=False,
            verbose=True,
            plots=True,
            deterministic=True 
        )
        
        # --- Automatische Auswertung ---
        print_metrics(results)
        
    except Exception as e:
        print(f"❌ Training Fehler: {e}")
        traceback.print_exc()
        return

    best_pt = RUNS_DIR / NAME / "weights" / "best.pt"
    if not best_pt.exists() and hasattr(results, 'save_dir'):
        best_pt = Path(results.save_dir) / 'weights' / 'best.pt'

    if best_pt.exists():
        print(f"\n📦 Exportiere {best_pt} für Hailo...")
        export_model = YOLO(str(best_pt))
        
        # FINALER HAILO EXPORT (Strict settings: FP32 + Static)
        onnx = export_model.export(
            format="onnx", 
            opset=11, 
            simplify=True, 
            imgsz=IMG_SIZE, 
            half=False,      # Speichern in FP32
            dynamic=False    # Static Shapes
        )
        
        print("\n" + "="*60)
        print("✅ FERTIG! BEREIT FÜR HAILO COMPILER.")
        print(f"1. ONNX:   {onnx}")
        print(f"2. Calib:  {CALIB_DIR} (Noise-Injected!)")
        print("="*60)

if __name__ == "__main__":
    main()