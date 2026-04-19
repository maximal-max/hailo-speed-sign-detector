"""
==============================================================================
compare_models_advanced.py  —  YOLO Trainings-Vergleich (Terminal)
==============================================================================

Durchsucht runs/ rekursiv nach results.csv und erstellt ein vollständiges
Terminal-Ranking aller Trainingsläufe. Ablösung der einfachen compare_models.py.

AUSGABE:
  • Kompakte Ranking-Tabelle (alle Kern-Metriken auf einen Blick)
  • Detail-Karten je Lauf (Verluste, ASCII-Balken, Delta zu Platz 1)
  • Sub-Rankings pro Einzel-Metrik (Top 5)
  • Optional: CSV-Export aller Metriken

VERWENDUNG:
  python compare_models_advanced.py                # Standard (Score-Sortierung)
  python compare_models_advanced.py --export       # + CSV-Export
  python compare_models_advanced.py --top 5        # Nur Top-N anzeigen
  python compare_models_advanced.py --no-cards     # Ohne Detail-Karten
  python compare_models_advanced.py --sort map50   # Sortierung wählen

SORTIEROPTIONEN:
  score | map50-95 | map50 | precision | recall | f1 | efficiency

GRAFISCHE AUSWERTUNG:
  Für PNG/JPG-Export → compare_models_visual.py

ABHÄNGIGKEITEN:
  pip install pandas pyyaml
==============================================================================
"""

import re
import sys
import argparse
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime


# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Wurzelpfad aller Trainingsläufe (enthält results.csv-Dateien)
BASE_DIR = Path("runs")

# Gewichtung des kombinierten Gesamt-Scores (Summe = 1.0)
SCORE_WEIGHTS = {
    "mAP50-95":  0.35,
    "mAP50":     0.25,
    "precision": 0.15,
    "recall":    0.15,
    "f1":        0.10,
}

# Mapping CLI-Argument → interner Spaltenname (für --sort)
SORT_KEYS = {
    "score":      "Score",
    "map50-95":   "mAP50-95",
    "map50":      "mAP50",
    "precision":  "Precision",
    "recall":     "Recall",
    "f1":         "F1",
    "efficiency": "mAP/Epoch",
}


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def bar(value: float, max_val: float = 1.0, width: int = 12,
        fill: str = "█", empty: str = "░") -> str:
    """Gibt einen ASCII-Fortschrittsbalken zurück (Breite in Zeichen)."""
    ratio  = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = round(ratio * width)
    return fill * filled + empty * (width - filled)


def rating_label(score: float) -> str:
    """Qualitätslabel basierend auf mAP50-95."""
    if score >= 0.85: return "★★★ Exzellent"
    if score >= 0.70: return "★★☆ Sehr gut"
    if score >= 0.55: return "★★☆ Gut"
    if score >= 0.40: return "★☆☆ Mittel"
    return "☆☆☆ Schwach"


def delta_str(val: float, ref: float, decimals: int = 4) -> str:
    """Gibt die vorzeichenbehaftete Differenz val - ref als String zurück."""
    diff = val - ref
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.{decimals}f}"


# ==============================================================================
# MODELL-INFO & ANALYSE
# ==============================================================================

def get_model_info(run_folder: Path) -> dict:
    """
    Liest Modell-Variante, Auflösung, Epochen, Batch, Optimizer und
    Dataset-Name aus der args.yaml des Trainingslaufs.
    Gibt Platzhalter '?' zurück wenn die Datei fehlt oder nicht lesbar ist.
    """
    info = {"variant": "?", "imgsz": "?", "epochs": "?",
            "dataset": "?", "batch": "?", "optimizer": "?"}

    args_path = run_folder / "args.yaml"
    if not args_path.exists():
        return info

    try:
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)

        model_name = str(args.get("model", "")).lower()
        model_base = Path(model_name).stem

        SIZE_LABELS = {"n": "Nano", "s": "Small", "m": "Medium",
                       "l": "Large", "b": "Base", "x": "XL"}

        match = re.search(r"yolo(?:v?(\d+))([nsmblx])", model_base)
        if match:
            gen, size = match.group(1), match.group(2)
            info["variant"] = f"v{gen}-{SIZE_LABELS.get(size, size.upper())}"
        else:
            info["variant"] = model_base or model_name

        info["imgsz"]     = args.get("imgsz", "?")
        info["epochs"]    = args.get("epochs", "?")
        info["batch"]     = args.get("batch", "?")
        info["optimizer"] = args.get("optimizer", "?")

        data_path = str(args.get("data", ""))
        info["dataset"] = Path(data_path).stem if data_path else "?"

    except Exception:
        pass

    return info


def analyze_run(csv_path: Path) -> dict | None:
    """
    Liest results.csv eines Trainingslaufs und extrahiert alle Metriken
    der besten Epoche (nach mAP50-95). Gibt None zurück wenn die Datei
    nicht auswertbar ist (fehlende Spalten, Parse-Fehler o.ä.).

    Abgeleitete Metriken:
      F1            = harmonisches Mittel aus Precision und Recall
      Score         = gewichteter Gesamt-Score (siehe SCORE_WEIGHTS)
      mAP/Epoch     = Trainingseffizienz (mAP50-95 / beste Epoche)
      Convergence   = Anteil des Trainings bis zum besten Ergebnis [0..1]
      OverfitScore  = Anstieg des val/box_loss nach dem besten Punkt (≥ 0)
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception:
        return None

    # Spaltenname für mAP50-95 variiert je nach YOLO-Version
    map_50_95_col = next(
        (c for c in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP_0.5:0.95"]
         if c in df.columns), None
    )
    if not map_50_95_col:
        return None

    try:
        best_idx = df[map_50_95_col].idxmax()
        best     = df.loc[best_idx]
        last     = df.iloc[-1]
    except Exception:
        return None

    def get(candidates: list[str], row=None) -> float:
        """Gibt den Wert der ersten gefundenen Spalte zurück, sonst 0.0."""
        row = row if row is not None else best
        for c in candidates:
            if c in df.columns:
                v = row[c]
                return float(v) if pd.notna(v) else 0.0
        return 0.0

    run_folder   = csv_path.parent
    meta         = get_model_info(run_folder)
    total_rows   = len(df)
    best_epoch   = int(best["epoch"])

    map50_95  = get([map_50_95_col])
    map50     = get(["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5"])
    precision = get(["metrics/precision(B)", "metrics/precision"])
    recall    = get(["metrics/recall(B)", "metrics/recall"])
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    box_loss = get(["train/box_loss"])
    cls_loss = get(["train/cls_loss"])
    dfl_loss = get(["train/dfl_loss"])
    val_box  = get(["val/box_loss"])
    val_cls  = get(["val/cls_loss"])
    val_dfl  = get(["val/dfl_loss"])

    val_box_final = get(["val/box_loss"], last)

    convergence   = best_epoch / max(total_rows - 1, 1)
    efficiency    = map50_95 / max(best_epoch + 1, 1)
    overfit_score = max(val_box_final - val_box, 0)

    weighted_score = (
        SCORE_WEIGHTS["mAP50-95"]  * map50_95  +
        SCORE_WEIGHTS["mAP50"]     * map50     +
        SCORE_WEIGHTS["precision"] * precision +
        SCORE_WEIGHTS["recall"]    * recall    +
        SCORE_WEIGHTS["f1"]        * f1
    )

    try:
        trained_at = datetime.fromtimestamp(
            csv_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        trained_at = "?"

    return {
        # Identifikation
        "Name":         str(run_folder.relative_to(BASE_DIR)),
        "Path":         str(run_folder),
        "TrainedAt":    trained_at,
        # Modell-Konfiguration (aus args.yaml)
        "Variant":      meta["variant"],
        "Size":         meta["imgsz"],
        "Batch":        meta["batch"],
        "Optimizer":    meta["optimizer"],
        "Dataset":      meta["dataset"],
        "MaxEpochs":    meta["epochs"],
        # Trainings-Ergebnis
        "BestEpoch":    best_epoch,
        "TrainedRows":  total_rows,
        # Haupt-Metriken (beste Epoche)
        "mAP50-95":     map50_95,
        "mAP50":        map50,
        "Precision":    precision,
        "Recall":       recall,
        "F1":           f1,
        # Verluste (beste Epoche)
        "BoxLoss":      box_loss,
        "ClsLoss":      cls_loss,
        "DflLoss":      dfl_loss,
        "ValBoxLoss":   val_box,
        "ValClsLoss":   val_cls,
        "ValDflLoss":   val_dfl,
        # Abgeleitete Metriken
        "Score":        weighted_score,
        "mAP/Epoch":    efficiency,
        "Convergence":  convergence,
        "OverfitScore": overfit_score,
    }


# ==============================================================================
# RANKING-AUSGABE
# ==============================================================================

def print_summary_table(ranked: list[dict], top_n: int | None = None) -> None:
    """
    Gibt die kompakte Ranking-Tabelle mit allen Kern-Metriken aus.
    Spaltenbreiten werden dynamisch an die längsten Einträge angepasst.
    """
    models = ranked[:top_n] if top_n else ranked

    wn = max(len("NAME"),    max(len(m["Name"])            for m in models))
    wv = max(len("MODEL"),   max(len(m["Variant"])         for m in models))
    ws = max(len("RES"),     max(len(str(m["Size"]))       for m in models))
    we = max(len("BEST-EP"), max(len(str(m["BestEpoch"])) for m in models))

    hdr = (f"{'#':<3}  {'NAME':<{wn}}  {'MODEL':<{wv}}  {'RES':<{ws}}  "
           f"{'BEST-EP':<{we}}  {'mAP50-95':>8}  {'mAP50':>6}  "
           f"{'PREC':>6}  {'REC':>6}  {'F1':>6}  "
           f"{'SCORE':>6}  {'EFF':>7}  BEWERTUNG")
    sep = "=" * len(hdr)

    print("\n" + sep)
    print("  RANKING — ZUSAMMENFASSUNG")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for i, m in enumerate(models):
        print(
            f"{i + 1:<3}  "
            f"{m['Name']:<{wn}}  "
            f"{m['Variant']:<{wv}}  "
            f"{str(m['Size']):<{ws}}  "
            f"{m['BestEpoch']:<{we}}  "
            f"{m['mAP50-95']:>8.4f}  "
            f"{m['mAP50']:>6.4f}  "
            f"{m['Precision']:>6.4f}  "
            f"{m['Recall']:>6.4f}  "
            f"{m['F1']:>6.4f}  "
            f"{m['Score']:>6.4f}  "
            f"{m['mAP/Epoch']:>7.5f}  "
            f"{rating_label(m['mAP50-95'])}"
        )

    print("=" * len(hdr))


def print_detail_cards(ranked: list[dict], top_n: int | None = None) -> None:
    """
    Gibt je eine Detailkarte pro Trainingslauf aus. Zeigt Verluste,
    ASCII-Balken für alle Haupt-Metriken und Deltas zu Platz 1.
    """
    models = ranked[:top_n] if top_n else ranked
    ref    = models[0]   # Platz 1 als Referenz für Delta-Berechnung

    print("\n" + "=" * 70)
    print("  DETAIL-KARTEN")
    print("=" * 70)

    for i, m in enumerate(models):
        conv_pct   = m["Convergence"] * 100
        medal      = ["#1 GOLD", "#2 SILBER", "#3 BRONZE"][i] if i < 3 else f"#{i + 1}"
        best_epoch = m["BestEpoch"]

        print(f"\n┌─ {medal}  {m['Name']}  [{m['TrainedAt']}]")
        print(f"│  Modell:    {m['Variant']}  |  Auflösung: {m['Size']}  |  "
              f"Batch: {m['Batch']}  |  Optimizer: {m['Optimizer']}")
        print(f"│  Dataset:   {m['Dataset']}  |  "
              f"Epochen konfiguriert: {m['MaxEpochs']}  |  Best bei: {best_epoch}")
        print("│")
        print(f"│  HAUPT-METRIKEN          BALKEN           DELTA ZU #1")
        print(f"│  mAP50-95  {m['mAP50-95']:.4f}   {bar(m['mAP50-95'])}   "
              f"{delta_str(m['mAP50-95'], ref['mAP50-95'])}")
        print(f"│  mAP50     {m['mAP50']:.4f}   {bar(m['mAP50'])}   "
              f"{delta_str(m['mAP50'], ref['mAP50'])}")
        print(f"│  Precision {m['Precision']:.4f}   {bar(m['Precision'])}   "
              f"{delta_str(m['Precision'], ref['Precision'])}")
        print(f"│  Recall    {m['Recall']:.4f}   {bar(m['Recall'])}   "
              f"{delta_str(m['Recall'], ref['Recall'])}")
        print(f"│  F1-Score  {m['F1']:.4f}   {bar(m['F1'])}")
        print(f"│  Ges.-Score {m['Score']:.4f}   {bar(m['Score'])}")
        print("│")
        print(f"│  VERLUSTE (beste Epoche)")
        print(f"│  Train  — Box: {m['BoxLoss']:.4f}  Cls: {m['ClsLoss']:.4f}  "
              f"Dfl: {m['DflLoss']:.4f}")
        print(f"│  Val    — Box: {m['ValBoxLoss']:.4f}  Cls: {m['ValClsLoss']:.4f}  "
              f"Dfl: {m['ValDflLoss']:.4f}")
        print("│")
        print(f"│  EFFIZIENZ & KONVERGENZ")
        print(f"│  mAP/Epoche: {m['mAP/Epoch']:.5f}   |  "
              f"Best-Epoche bei {conv_pct:.0f}% des Trainings  |  "
              f"Overfit-Indikator: {m['OverfitScore']:.4f}")
        print("└" + "─" * 68)


def print_metric_rankings(ranked: list[dict]) -> None:
    """Gibt je ein Sub-Ranking (Top 5) pro Schlüssel-Metrik aus."""
    metrics = [
        ("mAP50-95",  "Detektionsqualität (streng)"),
        ("mAP50",     "Detektionsqualität (locker)"),
        ("Precision", "Präzision  (wenig False Positives)"),
        ("Recall",    "Recall     (wenig False Negatives)"),
        ("F1",        "F1-Score   (Balance P/R)"),
        ("mAP/Epoch", "Trainingseffizienz"),
    ]

    print("\n" + "=" * 50)
    print("  SUB-RANKINGS PRO METRIK")
    print("=" * 50)

    for key, label in metrics:
        sub = sorted(ranked, key=lambda x: x[key], reverse=True)
        print(f"\n  {label}")
        print(f"  {'-' * 46}")
        for j, m in enumerate(sub[:5]):
            print(f"  {j + 1}. {m['Name']:<30}  {m[key]:.4f}")


# ==============================================================================
# CSV-EXPORT
# ==============================================================================

def export_csv(ranked: list[dict], path: str = "model_comparison.csv") -> None:
    """Exportiert alle Metriken aller Läufe als CSV-Datei."""
    cols = [
        "Name", "Variant", "Size", "Batch", "Optimizer", "Dataset",
        "MaxEpochs", "BestEpoch", "TrainedAt",
        "mAP50-95", "mAP50", "Precision", "Recall", "F1",
        "Score", "mAP/Epoch", "Convergence", "OverfitScore",
        "BoxLoss", "ClsLoss", "DflLoss",
        "ValBoxLoss", "ValClsLoss", "ValDflLoss",
        "Path",
    ]
    pd.DataFrame(ranked)[cols].to_csv(path, index=False)
    print(f"\n  Export gespeichert: {path}")


# ==============================================================================
# EINSTIEGSPUNKT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO Modell-Vergleich — detailliertes Terminal-Ranking"
    )
    p.add_argument("--export",   action="store_true",
                   help="Alle Metriken als CSV speichern")
    p.add_argument("--top",      type=int, default=None,
                   help="Nur Top-N Modelle anzeigen")
    p.add_argument("--sort",     default="score", choices=list(SORT_KEYS.keys()),
                   help="Sortier-Metrik (Standard: score)")
    p.add_argument("--no-cards", action="store_true",
                   help="Detail-Karten überspringen")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not BASE_DIR.exists():
        print(f"  Ordner '{BASE_DIR}' nicht gefunden.")
        sys.exit(1)

    print(f"\n  Suche nach Trainings in '{BASE_DIR}' (rekursiv)...")
    csv_files = list(BASE_DIR.rglob("results.csv"))

    if not csv_files:
        print("  Keine Trainingsergebnisse gefunden.")
        sys.exit(1)

    print(f"  -> {len(csv_files)} Trainings gefunden. Analysiere Metriken...")

    all_models = [r for f in csv_files if (r := analyze_run(f))]

    if not all_models:
        print("  Keine auswertbaren Ergebnisse gefunden.")
        sys.exit(1)

    sort_col = SORT_KEYS[args.sort]
    ranked   = sorted(all_models, key=lambda x: x[sort_col], reverse=True)

    print(f"  Sortiert nach: {sort_col}  |  Modelle ausgewertet: {len(ranked)}")
    print(f"  Score-Gewichtung: " +
          "  ".join(f"{k}={v:.0%}" for k, v in SCORE_WEIGHTS.items()))

    print_summary_table(ranked, args.top)

    if not args.no_cards:
        print_detail_cards(ranked, args.top)

    print_metric_rankings(ranked)

    if args.export:
        export_csv(ranked)

    winner = ranked[0]
    print(f"\n  SIEGER: {winner['Name']}  "
          f"({winner['Variant']}, {winner['Size']}px)  "
          f"→  mAP50-95={winner['mAP50-95']:.4f}  "
          f"Score={winner['Score']:.4f}\n")


if __name__ == "__main__":
    main()
