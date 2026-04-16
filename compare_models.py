"""
================================================================
YOLO Advanced Model Comparator - Alle Trainings automatisch finden und vergleichen
================================================================
Features:
1. Findet ALLE Trainings (egal ob runs/detect oder runs/hailo_train).
2. Liest 'args.yaml', um Modell-Typ (n/s/m/l/x) und Aufloesung zu finden.
3. Unterstuetzt YOLOv8, YOLO11 und neuere Generationen.
4. Ranking basierend auf mAP50-95.
5. Zeigt mAP@50 und mAP@50-95 an.

"""

import re
import pandas as pd
import yaml
from pathlib import Path

# Suchpfad (Root fuer alle Trainings)
BASE_DIR = Path("runs")

def get_model_info(run_folder):
    """
    Versucht, Metadaten aus args.yaml zu extrahieren.
    Gibt ein Dictionary mit 'variant' (n/s/m) und 'imgsz' zurueck.
    """
    info = {
        "variant": "?",
        "imgsz": "?"
    }

    args_path = run_folder / "args.yaml"
    if args_path.exists():
        try:
            with open(args_path, 'r') as f:
                args = yaml.safe_load(f)

                # Modell-Variante aus Dateiname lesen (z.B. 'yolov8n.pt', 'yolo11s.pt')
                model_name = str(args.get('model', '')).lower()
                model_base = Path(model_name).stem  # z.B. 'yolo11s' oder 'yolov8n'

                SIZE_LABELS = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'l': 'Large', 'x': 'XL'}

                # Generationskennung + Groesse erkennen, z.B. 'yolo11s' -> gen='11', size='s'
                match = re.search(r'yolo(?:v?(\d+))([nsmblx])', model_base)
                if match:
                    gen, size = match.group(1), match.group(2)
                    label = SIZE_LABELS.get(size, size.upper())
                    info['variant'] = f'v{gen}-{label}'
                else:
                    info['variant'] = model_base or model_name

                info['imgsz'] = args.get('imgsz', '?')
        except Exception:
            pass

    return info

def analyze_run(csv_path):
    """Liest die results.csv und findet die beste Epoche (basierend auf mAP50-95)."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception:
        return None

    map_50_95_col = None
    for c in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP_0.5:0.95"]:
        if c in df.columns:
            map_50_95_col = c
            break

    if not map_50_95_col:
        return None

    try:
        best_idx = df[map_50_95_col].idxmax()
        best_row = df.loc[best_idx]
    except Exception:
        return None

    def get_val(candidates):
        for c in candidates:
            if c in df.columns:
                return best_row[c]
        return 0.0

    run_folder = csv_path.parent
    meta = get_model_info(run_folder)
    rel_path = run_folder.relative_to(BASE_DIR)

    return {
        "Name":     str(rel_path),
        "Variant":  meta['variant'],
        "Size":     meta['imgsz'],
        "Epoch":    int(best_row["epoch"]),
        "mAP50-95": best_row[map_50_95_col],
        "mAP50":    get_val(["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5"]),
        "Path":     str(run_folder)
    }

def scan_and_rank():
    if not BASE_DIR.exists():
        print(f"Ordner '{BASE_DIR}' nicht gefunden.")
        return

    print(f"Suche nach Trainings in '{BASE_DIR}' (rekursiv)...")

    csv_files = list(BASE_DIR.rglob("results.csv"))

    if not csv_files:
        print("Keine Trainingsergebnisse gefunden.")
        return

    all_models = []
    print(f"   -> {len(csv_files)} Trainings gefunden. Analysiere Details...")

    for csv_file in csv_files:
        stats = analyze_run(csv_file)
        if stats:
            all_models.append(stats)

    ranked = sorted(all_models, key=lambda x: x['mAP50-95'], reverse=True)

    # --- Spaltenbreiten dynamisch berechnen ---
    w_rank    = max(len("RANK"),  len(str(len(ranked))))
    w_name    = max(len("NAME"),  max(len(m['Name'])        for m in ranked))
    w_variant = max(len("MODEL"), max(len(m['Variant'])     for m in ranked))
    w_size    = max(len("RES"),   max(len(str(m['Size']))   for m in ranked))
    w_epoch   = max(len("EPOCH"), max(len(str(m['Epoch']))  for m in ranked))
    w_map50   = max(len("mAP50"),    6)   # "0.0000"
    w_map9595 = max(len("mAP50-95"), 6)

    header = (f"{'RANK':<{w_rank}}  "
              f"{'NAME':<{w_name}}  "
              f"{'MODEL':<{w_variant}}  "
              f"{'RES':<{w_size}}  "
              f"{'EPOCH':<{w_epoch}}  "
              f"{'mAP50':<{w_map50}}  "
              f"{'mAP50-95':<{w_map9595}}  "
              f"BEWERTUNG")
    sep_thick = "=" * len(header)
    sep_thin  = "-" * len(header)

    print("\n" + sep_thick)
    print(header)
    print(sep_thick)

    for i, m in enumerate(ranked):
        score = m['mAP50-95']

        if score > 0.75:   rate = "Exzellent"
        elif score > 0.60: rate = "Gut"
        elif score > 0.40: rate = "Mittel"
        else:              rate = "Schwach"

        print(f"{i+1:<{w_rank}}  "
              f"{m['Name']:<{w_name}}  "
              f"{m['Variant']:<{w_variant}}  "
              f"{str(m['Size']):<{w_size}}  "
              f"{m['Epoch']:<{w_epoch}}  "
              f"{m['mAP50']:.4f}  "
              f"{score:.4f}    "
              f"{rate}")

    print(sep_thin)

if __name__ == "__main__":
    scan_and_rank()
