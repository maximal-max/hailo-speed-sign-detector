"""
================================================================
YOLOv8 Advanced Model Comparator - Alle Trainings automatisch finden und vergleichen
================================================================
Features:
1. Findet ALLE Trainings (egal ob runs/detect oder runs/hailo_train).
2. Liest 'args.yaml', um Modell-Typ (n/s/m) und Auflösung zu finden.
3. Ranking basierend auf mAP50-95.
4. Zeigt mAP@50 und mAP@50-95 an.

"""

import pandas as pd
import yaml
from pathlib import Path

# Suchpfad (Root für alle Trainings)
BASE_DIR = Path("runs")

def get_model_info(run_folder):
    """
    Versucht, Metadaten aus args.yaml zu extrahieren.
    Gibt ein Dictionary mit 'variant' (n/s/m) und 'imgsz' zurück.
    """
    info = {
        "variant": "?",
        "imgsz": "?"
    }
    
    # 1. Versuch: args.yaml lesen (schnell)
    args_path = run_folder / "args.yaml"
    if args_path.exists():
        try:
            with open(args_path, 'r') as f:
                args = yaml.safe_load(f)
                
                # Modell-Variante erraten (aus Dateiname 'yolov8n.pt' -> 'n')
                model_name = str(args.get('model', '')).lower()
                if 'v8n' in model_name: info['variant'] = 'Nano (n)'
                elif 'v8s' in model_name: info['variant'] = 'Small (s)'
                elif 'v8m' in model_name: info['variant'] = 'Medium (m)'
                elif 'v8l' in model_name: info['variant'] = 'Large (l)'
                else: info['variant'] = model_name
                
                # Auflösung
                info['imgsz'] = args.get('imgsz', '?')
        except:
            pass
            
    return info

def analyze_run(csv_path):
    """Liest die results.csv und findet die beste Epoche (basierend auf mAP50-95)."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() # Leerzeichen entfernen
    except:
        return None

    # Spalte für mAP50-95 finden
    map_50_95_col = None
    for c in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP_0.5:0.95"]:
        if c in df.columns:
            map_50_95_col = c
            break
    
    if not map_50_95_col: return None

    # Beste Epoche finden (Index mit dem höchsten mAP50-95)
    try:
        best_idx = df[map_50_95_col].idxmax()
        best_row = df.loc[best_idx]
    except:
        return None

    # Metriken holen
    def get_val(candidates):
        """Hilfsfunktion, um den Wert aus möglichen Spaltennamen zu finden."""
        for c in candidates:
            if c in df.columns: return best_row[c]
        return 0.0

    run_folder = csv_path.parent
    meta = get_model_info(run_folder)

    # Name des Ordners relativ zum BASE_DIR
    rel_path = run_folder.relative_to(BASE_DIR)
    
    return {
        "Name": str(rel_path),
        "Variant": meta['variant'],
        "Size": meta['imgsz'],
        "Epoch": int(best_row["epoch"]),
        "mAP50-95": best_row[map_50_95_col],
        "mAP50": get_val(["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5"]),
        "Path": str(run_folder)
    }

def scan_and_rank():
    if not BASE_DIR.exists():
        print(f"❌ Ordner '{BASE_DIR}' nicht gefunden.")
        return

    print(f"🔍 Suche nach Trainings in '{BASE_DIR}' (rekursiv)...")
    
    # Rekursiv alle results.csv finden
    csv_files = list(BASE_DIR.rglob("results.csv"))
    
    if not csv_files:
        print("❌ Keine Trainingsergebnisse gefunden.")
        return

    all_models = []
    print(f"   -> {len(csv_files)} Trainings gefunden. Analysiere Details (kann kurz dauern)...")

    for csv_file in csv_files:
        stats = analyze_run(csv_file)
        if stats:
            all_models.append(stats)

    # Sortieren nach mAP50-95 (absteigend)
    ranked = sorted(all_models, key=lambda x: x['mAP50-95'], reverse=True)

    # --- AUSGABE ---
    print("\n" + "="*110)
    print(f"{'RANK':<4} {'NAME':<30} {'MODEL':<10} {'RES':<8} {'EPOCH':<6} {'mAP50':<8} {'mAP50-95':<10} {'BEWERTUNG'}")
    print("="*110)

    for i, m in enumerate(ranked):
        score = m['mAP50-95']
        
        if score > 0.75: rate = "⭐ Exzellent"
        elif score > 0.60: rate = "✅ Gut"
        elif score > 0.40: rate = "⚠️ Mittel"
        else: rate = "❌ Schwach"

        
        print(f"{i+1:<4} {m['Name']:<30} {m['Variant']:<10} {str(m['Size']):<8} {m['Epoch']:<6} {m['mAP50']:.4f}   {score:.4f}     {rate}")

    print("-" * 110)

if __name__ == "__main__":
    scan_and_rank()