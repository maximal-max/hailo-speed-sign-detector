"""
==============================================================================
split_dataset.py  —  Datensatz-Splitter (Stratified)
==============================================================================

Bereitet Rohdaten für das YOLO-Training vor.

SCHRITTE:
  1. Bilder und Labels aus SOURCE_DIR einlesen (rekursiv, alle Bildformate).
  2. Stratifizierter Split nach Klasse — seltene Schilder landen garantiert
     in beiden Splits (train + val).
  3. Hintergrundbilder (leere Labels / Negative Samples) werden proportional
     auf die Splits verteilt.
  4. Ergebnis in TARGET_DIR kopieren (YOLO-Ordnerstruktur: images/ + labels/).

WICHTIG:
  Keine .yaml wird kopiert — das Training liest die zentrale tempolimits.yaml
  aus dem Projekt-Root direkt.

VERWENDUNG:
  python split_dataset.py

==============================================================================
"""

import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# ==============================================================================
# KONFIGURATION
# ==============================================================================

SOURCE_DIR = Path("datasets/stable_tempo_dataset")   # Rohdaten-Eingabe
TARGET_DIR = Path("datasets/sorted_dataset")          # Split-Ausgabe (wird neu erstellt)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
LABEL_EXT  = ".txt"

SPLITS = {
    "train": 0.8,
    "val":   0.2,
    "test":  0.0,    # Test-Split deaktiviert
}

SEED = 42
random.seed(SEED)

# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def get_label_path(img_path):
    """
    Sucht die zugehörige YOLO-Label-Datei zu einem Bild.

    Prüft zwei Strukturen:
      1. Side-by-Side:     bild.jpg → bild.txt (gleicher Ordner)
      2. Parallel-Ordner:  images/bild.jpg → labels/bild.txt
    """
    lbl = img_path.with_suffix(LABEL_EXT)
    if lbl.exists():
        return lbl

    try:
        if img_path.parent.name == "images":
            lbl_parallel = img_path.parent.parent / "labels" / f"{img_path.stem}{LABEL_EXT}"
            if lbl_parallel.exists():
                return lbl_parallel
    except Exception:
        pass

    return None

def cleanup_target():
    if TARGET_DIR.exists():
        print(f"🗑️  Bereinige Zielordner: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    
    for split, ratio in SPLITS.items():
        if ratio > 0:
            (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def gather_data():
    print("\n🔍 Sammle Daten (suche auch in ../labels/)...")
    
    fg_class_map = defaultdict(list)
    bg_files = []
    file_contents = {}

    all_images = []
    for ext in IMAGE_EXTS:
        all_images.extend(SOURCE_DIR.rglob(f"*{ext}"))
    
    print(f"   --> {len(all_images)} Bild-Dateien gefunden.")

    found_labels_count = 0

    for img_path in all_images:
        lbl_path = get_label_path(img_path)
        
        classes = []
        is_bg = True

        if lbl_path and lbl_path.exists():
            found_labels_count += 1
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            classes.append(cid)
                            is_bg = False
                        except ValueError:
                            pass
        
        file_contents[img_path] = classes

        if is_bg:
            bg_files.append(img_path)
        else:
            for cid in set(classes):
                fg_class_map[cid].append(img_path)

    print(f"📸 Status: {found_labels_count} Labels gefunden.")
    print(f"   --> {len(fg_class_map)} Klassen-Gruppen (Bilder mit Inhalt)")
    print(f"   --> {len(bg_files)} Background-Bilder")
    
    return fg_class_map, bg_files, file_contents

# ==============================================================================
# SPLIT-LOGIK
# ==============================================================================

def calculate_splits(fg_class_map, bg_files):
    print("\n⚖️  Berechne balancierten Split ...")
    
    final_sets = {s: set() for s in SPLITS.keys()}

    # 1. Background
    random.shuffle(bg_files)
    n_bg = len(bg_files)
    
    idx_train = int(n_bg * SPLITS["train"])
    idx_val = idx_train + int(n_bg * SPLITS["val"])
    
    if SPLITS["train"] > 0: final_sets["train"].update(bg_files[:idx_train])
    if SPLITS["val"] > 0:   final_sets["val"].update(bg_files[idx_train:idx_val])
    if SPLITS["test"] > 0:  final_sets["test"].update(bg_files[idx_val:])

    # 2. Foreground
    sorted_classes = sorted(fg_class_map.keys(), key=lambda k: len(fg_class_map[k]))

    for cid in sorted_classes:
        imgs = fg_class_map[cid]
        random.shuffle(imgs)
        n_total = len(imgs)
        
        target_test = 0
        if SPLITS["test"] > 0 and n_total > 1:
            target_test = max(1, int(n_total * SPLITS["test"]))
            
        target_val = 0
        if SPLITS["val"] > 0 and n_total > 1:
            target_val = max(1, int(n_total * SPLITS["val"]))
    
        c_val, c_test = 0, 0 
        
        for img in imgs:
            if any(img in final_sets[s] for s in SPLITS):
                continue
            
            if c_test < target_test:
                final_sets["test"].add(img)
                c_test += 1
            elif c_val < target_val:
                final_sets["val"].add(img)
                c_val += 1
            elif SPLITS["train"] > 0:
                final_sets["train"].add(img)

    return final_sets

# ==============================================================================
# DATEIEN KOPIEREN & BERICHT
# ==============================================================================

def copy_files(final_sets, file_contents):
    print("\n📥 Kopiere Dateien ...")
    stats = {s: Counter() for s in SPLITS.keys()}
    bg_count = {s: 0 for s in SPLITS.keys()}

    for split_name, files in final_sets.items():
        if not files: continue
            
        for img_src in tqdm(files, desc=f"Split: {split_name}"):
            lbl_src = get_label_path(img_src)
            
            img_dst = TARGET_DIR / "images" / split_name / img_src.name
            lbl_dst = None

            if lbl_src and lbl_src.exists():
                lbl_dst = TARGET_DIR / "labels" / split_name / lbl_src.name

            shutil.copy2(img_src, img_dst)
            if lbl_dst is not None:
                shutil.copy2(lbl_src, lbl_dst)
            
            classes = file_contents[img_src]
            if not classes:
                bg_count[split_name] += 1
            else:
                for cid in classes:
                    stats[split_name][cid] += 1
                    
    return stats, bg_count

def print_report(stats, bg_count):
    print("\n" + "="*60)
    print(f"{'Klasse':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Gesamt':<8}")
    print("-" * 60)
    
    all_cids = set()
    for s in stats.values():
        all_cids.update(s.keys())
    all_cids = sorted(all_cids)
    
    for cid in all_cids:
        t = stats["train"][cid]
        v = stats["val"][cid]
        te = stats["test"][cid]
        print(f"{cid:<10} | {t:<8} | {v:<8} | {te:<8} | {t+v+te:<8}")

    print("-" * 60)
    print(f"{'Background':<10} | {bg_count['train']:<8} | {bg_count['val']:<8} | {bg_count['test']:<8} | {sum(bg_count.values()):<8}")
    print("="*60)
    
    if SPLITS["test"] == 0:
        print("\nℹ️  HINWEIS: Test-Split ist deaktiviert (0%).")

if __name__ == "__main__":
    cleanup_target()
    fg_map, bg_files, file_contents = gather_data()
    splits = calculate_splits(fg_map, bg_files)
    stats, bg_stats = copy_files(splits, file_contents)
    print_report(stats, bg_stats)