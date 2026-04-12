"""
==============================================================================
Hailo Kalibrierungsset-Generator
==============================================================================

ZWECK:
  Erstellt Kalibrierungsbilder für den Hailo DFC (Data Flow Compiler), der
  das trainierte FP32-ONNX-Modell auf INT8 quantisiert. Pro gewünschter
  Modellgröße (z.B. 512, 640, 800 px) entsteht ein separater Ordner mit:
    • images/   → 1024 JPG-Dateien für den Hailo Cloud Compiler (Ordner-Upload)
    • calib_set.npy → dasselbe Set als numpy-Array für den lokalen DFC

WARUM EIN EIGENES SKRIPT?
  Das Training (train_yolo.py) erzeugt das ONNX-Modell. Die Kalibrierung ist
  ein davon getrennter Schritt, weil:
    • Verschiedene Modellgrößen (512/640/800 px) brauchen jeweils eigene Sets.
    • Das Set muss nur einmal erstellt werden — nicht bei jedem Training neu.
    • Der Cloud Compiler und der lokale DFC erwarten unterschiedliche Formate
      (JPG-Ordner vs. .npy), beide werden hier erzeugt.

VORVERARBEITUNG: NUR LETTERBOX — KEIN RAUSCHEN, KEINE AUGMENTIERUNG
  Die Kalibrierungsbilder müssen so aussehen wie das, was die Kamera auf dem
  Pi tatsächlich liefert. Augmentierter Motion-Blur oder Rauschen würden
  die Aktivierungsverteilungen verzerren und die INT8-Quantisierung
  verschlechtern. Deshalb: nur Letterbox-Resize (LongestMaxSize + PadIfNeeded,
  grauer Rand Wert 114) — identisch zur preprocess()-Funktion in
  RPI_application.py.

FARBREIHENFOLGE:
  • JPGs werden mit cv2.imwrite in BGR gespeichert (OpenCV-Standard, korrekt).
  • Das .npy-Array wird in RGB gespeichert, da der Hailo DFC bei .npy-Input
    RGB erwartet (entspricht dem RGB-Tensor, den preprocess() an den Chip gibt).

VERWENDUNG:
  1. YAML_FILE auf deine tempolimits.yaml zeigen lassen.
  2. RESOLUTIONS auf die gewünschten Modellgrößen setzen.
  3. NUM_IMAGES auf 1024 (Standard) oder weniger zum Testen.
  4. Skript ausführen — Ausgabe landet in OUTPUT_ROOT.
  5. Für Cloud Compiler: images/-Ordner hochladen.
     Für lokalen DFC:    calib_set.npy übergeben.

==============================================================================
"""

import cv2
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A


# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Zentrale Konfigurations-Datei (Klassen + Datenpfade)
YAML_FILE = "tempolimits.yaml"

# Modellgrößen für die Kalibrierung.
# Jede Größe entspricht einem trainierten Modell (z.B. 640px.hef).
# Nur die Größen eintragen, für die du auch ein .hef kompilieren willst.
RESOLUTIONS = [512, 640, 800]

# Anzahl der Kalibrierungsbilder pro Auflösung.
# 1024 ist der empfohlene Wert für den Hailo DFC.
# Zum Testen reichen 64–128.
NUM_IMAGES = 1024

# Ausgabe-Wurzelverzeichnis (wird bei jedem Lauf komplett neu erstellt)
OUTPUT_ROOT = Path("hailo_calibration")

# Seed für vollständige Reproduzierbarkeit
SEED = 42


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def load_config() -> dict:
    """Lädt die zentrale YAML-Konfiguration."""
    yaml_path = Path(YAML_FILE)
    if not yaml_path.exists():
        print(f"❌ FEHLER: '{YAML_FILE}' nicht gefunden in {Path.cwd()}")
        exit(1)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_train_path(config: dict) -> Path | None:
    """
    Ermittelt den absoluten Pfad zum train-Bildordner aus der YAML.
    Probiert mehrere Kandidaten durch um sowohl absolute als auch
    relative Pfade zu unterstützen — identisch zu resolve_source_path()
    in train_yolo.py.
    """
    root_str  = config.get('path', '')
    train_str = config.get('train', '')

    if not train_str:
        return None

    candidates = [
        Path(train_str),
        Path(root_str) / train_str,
        Path.cwd() / train_str,
        Path.cwd() / Path(root_str) / train_str,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def build_letterbox_pipeline(img_size: int) -> A.Compose:
    """
    Erstellt eine reine Letterbox-Pipeline für eine gegebene Bildgröße.

    LongestMaxSize skaliert das Bild so, dass die längste Seite img_size
    erreicht ohne das Seitenverhältnis zu verändern.
    PadIfNeeded füllt den Rest mit dem Wert 114 auf (grau) — identisch
    zum grauen Padding in preprocess() in RPI_application.py.

    Keine Augmentierung, kein Rauschen — nur das Resize, das auch auf
    dem Pi passiert.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=114,   # Grauer Rand, identisch zu YOLO-Standard und Pi-Inferenz
        ),
    ])


def collect_images(src_dir: Path) -> list[Path]:
    """
    Sammelt alle Bilder aus dem Quellordner (nicht rekursiv).
    Gibt eine sortierte, deduplizierte Liste zurück.
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                  "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"]
    return sorted(set(f for ext in extensions for f in src_dir.glob(ext)))


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    # --- Seed setzen ---
    print(f"🌱 Seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)

    # --- Konfiguration laden ---
    config  = load_config()
    src_dir = resolve_train_path(config)

    if not src_dir:
        print("❌ Train-Pfad aus YAML nicht auflösbar.")
        print("   Prüfe 'path' und 'train' in deiner tempolimits.yaml.")
        exit(1)

    print(f"📂 Quelle: {src_dir}")

    # --- Bilder einsammeln ---
    all_images = collect_images(src_dir)

    if not all_images:
        print(f"❌ Keine Bilder in '{src_dir}' gefunden.")
        exit(1)

    # Auswahl: NUM_IMAGES zufällig aus dem Gesamtpool wählen.
    # random.sample() ohne Zurücklegen: kein Bild kommt doppelt vor.
    if len(all_images) < NUM_IMAGES:
        print(f"⚠ Nur {len(all_images)} Bilder verfügbar — nutze alle "
              f"(gewünscht: {NUM_IMAGES}).")
        selected = all_images
    else:
        selected = random.sample(all_images, NUM_IMAGES)
        print(f"✔ {len(selected)} Bilder ausgewählt aus {len(all_images)} verfügbaren.")

    # --- Alten Output-Ordner löschen ---
    if OUTPUT_ROOT.exists():
        print(f"🧹 Lösche alten Output-Ordner: {OUTPUT_ROOT}")
        try:
            shutil.rmtree(OUTPUT_ROOT)
        except PermissionError:
            print("❌ Zugriff verweigert — bitte Ordner manuell löschen.")
            exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Pro Auflösung ein separates Set generieren ---
    print(f"\n🚀 Starte Generierung für Auflösungen: {RESOLUTIONS}")

    for res in RESOLUTIONS:
        print(f"\n{'='*60}")
        print(f"  Auflösung: {res}×{res} px")
        print(f"{'='*60}")

        # Ordnerstruktur für diese Auflösung
        base_dir = OUTPUT_ROOT / f"calib_{res}px"
        img_dir  = base_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pipeline    = build_letterbox_pipeline(res)
        npy_list    = []   # Sammlung der RGB-Arrays für .npy
        skipped     = 0

        for i, img_path in enumerate(tqdm(selected, desc=f"  {res}px")):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # Letterbox-Resize auf res×res (Seitenverhältnis bleibt erhalten)
            processed_bgr = pipeline(image=img)['image']

            # A) JPG speichern (BGR — cv2.imwrite-Standard, für Cloud Compiler)
            jpg_name = f"calib_{res}px_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / jpg_name), processed_bgr)

            # B) RGB-Array für .npy sammeln
            #    Der Hailo DFC erwartet bei .npy-Input RGB-Reihenfolge,
            #    weil preprocess() auf dem Pi ebenfalls BGR→RGB konvertiert
            #    bevor der Tensor an den Chip übergeben wird.
            npy_list.append(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))

        if skipped > 0:
            print(f"   ⚠ {skipped} Bilder übersprungen (unlesbar).")

        # C) .npy speichern
        #    Shape: (N, res, res, 3) dtype=uint8
        print(f"💾 Erstelle calib_set.npy für {res}px ...")
        npy_array = np.array(npy_list, dtype=np.uint8)
        npy_path  = base_dir / "calib_set.npy"
        np.save(str(npy_path), npy_array)

        size_mb = npy_path.stat().st_size / (1024 * 1024)
        print(f"   → Shape: {npy_array.shape}  |  Größe: {size_mb:.1f} MB")
        print(f"   → JPGs:  {img_dir}")
        print(f"   → NPY:   {npy_path}")

    # --- Abschluss ---
    print("\n" + "=" * 60)
    print("✅ KALIBRIERUNGSSETS FERTIG")
    print("-" * 60)
    print(f"{OUTPUT_ROOT}/")
    for res in RESOLUTIONS:
        print(f"  ├── calib_{res}px/")
        print(f"  │   ├── images/        ← JPGs für Cloud Compiler (Ordner-Upload)")
        print(f"  │   └── calib_set.npy  ← Array für lokalen DFC")
    print("-" * 60)
    print("  Cloud Compiler:  images/-Ordner der passenden Auflösung hochladen.")
    print("  Lokaler DFC:     calib_set.npy als --calib-path übergeben.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()