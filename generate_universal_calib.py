"""
================================================================
Skript zur Generierung von Kalibrierungsbildern für Hailo
================================================================

"""

import cv2
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# ===============================
# 🔧 EINSTELLUNGEN (Flexibel)
# ===============================

# Deine Config-Datei (für die Bildpfade)
YAML_FILE = "tempolimits.yaml"

# Die gewünschten Auflösungen
RESOLUTIONS = [512, 640, 800]

# Anzahl der Kalibrierungsbilder (Flexibel einstellbar)
# Setze dies auf 1024 für das finale "High-End" Set, oder 100 zum Testen.
NUM_IMAGES = 1024

# Haupt-Ausgabeverzeichnis
OUTPUT_ROOT = Path("hailo_calibration")

# Seed für exakte Reproduzierbarkeit
SEED = 42

# --- AUGMENTATION SETTINGS (1:1 aus deinem Training) ---
# Simuliert 130 km/h Motion Blur & Rolling Shutter
AUG_BLUR_LIMIT = (13, 21)   
AUG_BLUR_PROB  = 0.6
AUG_SHEAR_Y    = (-4, 4)    
AUG_SHEAR_X    = (-3, 3)
AUG_ROTATION   = (-4, 4)
AUG_GEOM_PROB  = 0.6
AUG_NOISE_VAR  = (40, 150)
AUG_NOISE_PROB = 0.6

# ===============================
# 🛠️ HELFER FUNKTIONEN
# ===============================

def load_config():
    if not Path(YAML_FILE).exists():
        print(f"❌ FEHLER: '{YAML_FILE}' nicht gefunden!")
        exit(1)
    with open(YAML_FILE, 'r') as f:
        return yaml.safe_load(f)

def resolve_path(config):
    """Findet den Pfad zu den Trainingsbildern basierend auf der YAML."""
    root_path = Path(config.get('path', ''))
    train_path = config.get('train', '')
    
    candidates = [
        Path(train_path),
        root_path / train_path,
        Path.cwd() / train_path,
        Path.cwd() / root_path / train_path
    ]
    
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
            
    if Path(train_path).exists():
        return Path(train_path)
    return None

def get_augmentation_pipeline(img_size):
    """Pipeline ohne Bounding-Boxes (nur Bildmanipulation)."""
    return A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=AUG_BLUR_LIMIT, p=0.6), 
            A.GaussianBlur(blur_limit=(5, 9), p=0.2),
        ], p=AUG_BLUR_PROB),

        A.OneOf([
            A.Affine(
                shear={'y': AUG_SHEAR_Y, 'x': AUG_SHEAR_X}, 
                scale=(0.95, 1.05), 
                rotate=AUG_ROTATION, 
                p=0.6
            ), 
        ], p=AUG_GEOM_PROB),

        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5), 
            A.GaussNoise(var_limit=AUG_NOISE_VAR, p=0.5),
        ], p=AUG_NOISE_PROB),

        A.RandomBrightnessContrast(p=0.5),
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=114)
    ])

def apply_quantization_hardening(img):
    """Zusätzliches Sensor-Rauschen (Hardening)."""
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    if random.random() < 0.2:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return img

# ===============================
# 🚀 MAIN
# ===============================

def main():
    print(f"🌱 Setze Seed auf {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. BEREINIGUNG: Alten Ordner löschen
    if OUTPUT_ROOT.exists():
        print(f"🧹 Bereinige altes Verzeichnis: {OUTPUT_ROOT}")
        try:
            shutil.rmtree(OUTPUT_ROOT)
        except PermissionError:
            print("⚠ Konnte Ordner nicht löschen (Zugriff verweigert). Bitte manuell prüfen.")
            exit(1)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 2. Bilder finden
    config = load_config()
    source_dir = resolve_path(config)
    
    if not source_dir:
        print("❌ Konnte Trainings-Ordner nicht finden!")
        exit(1)

    print(f"📂 Quelle: {source_dir}")
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_images = []
    for ext in extensions:
        all_images.extend(list(source_dir.glob(ext)))
        all_images.extend(list(source_dir.glob(ext.upper())))
    
    all_images = sorted(list(set(all_images)))
    
    # Auswahl der Bilder
    if len(all_images) < NUM_IMAGES:
        print(f"⚠ Warnung: Nur {len(all_images)} Bilder gefunden. Nutze alle verfügbaren.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, NUM_IMAGES)
        print(f"✅ Auswahl: {len(selected_images)} Bilder aus {len(all_images)}.")

    # 3. Generierung
    print(f"🚀 Starte Prozess für Auflösungen: {RESOLUTIONS}")

    for res in RESOLUTIONS:
        # Ordnerstruktur erstellen
        base_dir = OUTPUT_ROOT / f"calib_{res}px"
        img_dir = base_dir / "images"
        
        base_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n⚙️  Verarbeite {res}x{res} Pixel...")
        
        pipeline = get_augmentation_pipeline(res)
        npy_data_list = [] # Liste für numpy array

        for i, img_path in enumerate(tqdm(selected_images, desc=f"Res {res}")):
            # Laden
            img = cv2.imread(str(img_path))
            if img is None: continue

            # Augmentation + Resize
            augmented = pipeline(image=img)['image']

            # Hardening
            final_img = apply_quantization_hardening(augmented)

            # A) Als JPG speichern (im Unterordner 'images')
            out_name = f"calib_{res}_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / out_name), final_img)

            # B) Zur Liste hinzufügen (für .npy)
            # WICHTIG: OpenCV ist BGR, Hailo erwartet oft RGB.
            # Für die .npy konvertieren wir sicherheitshalber zu RGB.
            # (Die JPGs bleiben BGR, da cv2.imread/imwrite das so handhabt, was auch okay ist).
            final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            npy_data_list.append(final_img_rgb)

        # C) Als .npy speichern
        print(f"💾 Erstelle calib_set.npy für {res}px (das kann kurz dauern)...")
        npy_array = np.array(npy_data_list, dtype=np.uint8)
        
        npy_path = base_dir / "calib_set.npy"
        np.save(str(npy_path), npy_array)
        
        print(f"   -> Gespeichert: {npy_path}")
        print(f"   -> Shape: {npy_array.shape} | Größe: {npy_path.stat().st_size / (1024*1024):.2f} MB")

    print("\n" + "="*60)
    print("✅ FERTIG! Neue Ordnerstruktur:")
    print(f"{OUTPUT_ROOT}/")
    print("  ├── calib_512px/")
    print("  │   ├── images/       (JPG Bilder für Cloud Compiler Upload)")
    print("  │   └── calib_set.npy (Ideal für lokalen Compiler)")
    print("  ├── calib_640px/ ...")
    print("  └── calib_800px/ ...")
    print("="*60)

if __name__ == "__main__":
    main()