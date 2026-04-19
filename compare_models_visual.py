"""
==============================================================================
compare_models_visual.py  —  YOLO Trainings-Vergleich als Grafik
==============================================================================

Liest alle abgeschlossenen Trainingsläufe unter runs/ (rekursiv via
results.csv) und exportiert eine strukturierte Auswertung als Bild(er).

Ausgabe-Struktur (model_reports/comparison_YYYY-MM-DD_HH-MM/):
  overview_*.png   — Gesamt-Ansicht: Tabelle + Radar + Score-Gauge
  table_*.png      — Ranking-Tabelle als Einzelbild
  radar_*.png      — Radar-Chart (höhere DPI) als Einzelbild
  score_*.png      — Score-Gauge als Einzelbild

VERWENDUNG:
  python compare_models_visual.py                  # Gesamt + Einzelbilder (Standard)
  python compare_models_visual.py --no-split       # Nur Gesamt-Bild
  python compare_models_visual.py --top 8          # Nur Top-N Modelle
  python compare_models_visual.py --dpi 180        # Auflösung (Standard: 130)
  python compare_models_visual.py --fmt jpg        # Format: png (Standard) | jpg
  python compare_models_visual.py --light          # Helles Theme statt dunklem

ABHÄNGIGKEITEN:
  pip install pandas matplotlib pyyaml
==============================================================================
"""

import re
import sys
import argparse
import math
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from datetime import datetime

# Kein interaktives GUI-Backend nötig — reines Datei-Rendering
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]       = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False   # Minus-Zeichen korrekt darstellen

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Wurzelpfad aller Trainingsläufe (enthält results.csv-Dateien)
BASE_DIR    = Path("runs")

# Ausgabe-Basisverzeichnis; Unterordner wird automatisch erstellt
REPORTS_DIR = Path("model_reports")

# Gewichtung des kombinierten Gesamt-Scores (Summe muss 1.0 ergeben)
SCORE_WEIGHTS = {
    "mAP50-95":  0.35,
    "mAP50":     0.25,
    "Precision": 0.15,
    "Recall":    0.15,
    "F1":        0.10,
}

# Farben für Rang 1 (Gold), 2 (Silber), 3 (Bronze), alle weiteren
RANK_COLORS   = ["#F4C430", "#B8C0CC", "#CD7F32"]
DEFAULT_COLOR = "#4A90D9"

# ---------------------------------------------------------------------------
# Theme-Definitionen  — alle Farbvariablen zentral je Theme
# ---------------------------------------------------------------------------
_DARK_THEME = {
    "BG_DARK":      "#1A1D23",   # Hintergrund gesamte Figur
    "BG_PANEL":     "#22262E",   # Hintergrund einzelner Panels
    "BG_ROW_A":     "#272B34",   # Tabellen-Zeile gerade
    "BG_ROW_B":     "#1E2128",   # Tabellen-Zeile ungerade
    "FG_TEXT":      "#E8ECF0",   # Primärer Text
    "FG_MUTED":     "#8A909A",   # Sekundärer / gedämpfter Text
    "ACCENT":       "#3D7EBF",   # Header-Hintergrundfarbe der Tabelle
    "GRID_COLOR":   "#333840",   # Raster- und Achsenlinien (Radar)
    "TRACK_COLOR":  "#2C313A",   # Hintergrund-Track im Score-Gauge
    "LEGEND_EDGE":  "#555C66",   # Rahmen der Radar-Legende
    "BADGE_TEXT":   "#111827",   # Text auf farbigen Rang-Badges (immer dunkel)
}

_LIGHT_THEME = {
    "BG_DARK":      "#EFF1F5",   # Hintergrund gesamte Figur
    "BG_PANEL":     "#FFFFFF",   # Hintergrund einzelner Panels
    "BG_ROW_A":     "#F3F4F6",   # Tabellen-Zeile gerade
    "BG_ROW_B":     "#FFFFFF",   # Tabellen-Zeile ungerade
    "FG_TEXT":      "#111827",   # Primärer Text
    "FG_MUTED":     "#6B7280",   # Sekundärer / gedämpfter Text
    "ACCENT":       "#1D4ED8",   # Header-Hintergrundfarbe der Tabelle
    "GRID_COLOR":   "#D1D5DB",   # Raster- und Achsenlinien (Radar)
    "TRACK_COLOR":  "#E5E7EB",   # Hintergrund-Track im Score-Gauge
    "LEGEND_EDGE":  "#C9CDD6",   # Rahmen der Radar-Legende
    "BADGE_TEXT":   "#111827",   # Text auf farbigen Rang-Badges (immer dunkel)
}

# Aktive Farbvariablen — werden durch apply_theme() gesetzt (Standard: dark)
BG_DARK     = _DARK_THEME["BG_DARK"]
BG_PANEL    = _DARK_THEME["BG_PANEL"]
BG_ROW_A    = _DARK_THEME["BG_ROW_A"]
BG_ROW_B    = _DARK_THEME["BG_ROW_B"]
FG_TEXT     = _DARK_THEME["FG_TEXT"]
FG_MUTED    = _DARK_THEME["FG_MUTED"]
ACCENT      = _DARK_THEME["ACCENT"]
GRID_COLOR  = _DARK_THEME["GRID_COLOR"]
TRACK_COLOR = _DARK_THEME["TRACK_COLOR"]
LEGEND_EDGE = _DARK_THEME["LEGEND_EDGE"]
BADGE_TEXT  = _DARK_THEME["BADGE_TEXT"]


def apply_theme(light: bool = False) -> str:
    """
    Setzt alle globalen Farbvariablen auf das gewählte Theme.
    Gibt den Theme-Namen als String zurück ('light' oder 'dark').
    """
    global BG_DARK, BG_PANEL, BG_ROW_A, BG_ROW_B
    global FG_TEXT, FG_MUTED, ACCENT
    global GRID_COLOR, TRACK_COLOR, LEGEND_EDGE, BADGE_TEXT

    t = _LIGHT_THEME if light else _DARK_THEME
    BG_DARK     = t["BG_DARK"]
    BG_PANEL    = t["BG_PANEL"]
    BG_ROW_A    = t["BG_ROW_A"]
    BG_ROW_B    = t["BG_ROW_B"]
    FG_TEXT     = t["FG_TEXT"]
    FG_MUTED    = t["FG_MUTED"]
    ACCENT      = t["ACCENT"]
    GRID_COLOR  = t["GRID_COLOR"]
    TRACK_COLOR = t["TRACK_COLOR"]
    LEGEND_EDGE = t["LEGEND_EDGE"]
    BADGE_TEXT  = t["BADGE_TEXT"]
    return "light" if light else "dark"

# Tabellen-Spalten: (Kopfzeilen-Label, x-Position [0..1], Ausrichtung, Daten-Key)
# key=None markiert die Rang-Badge-Spalte (wird gesondert gezeichnet).
TABLE_COLS = [
    ("#",         0.018, "center", None),
    ("Name",      0.095, "left",   "Name"),
    ("Modell",    0.310, "left",   "Variant"),
    ("Res.",      0.445, "center", "Size"),
    ("Best-Ep",   0.510, "center", "BestEpoch"),
    ("mAP50-95",  0.590, "center", "mAP50-95"),
    ("mAP50",     0.670, "center", "mAP50"),
    ("Prec.",     0.740, "center", "Precision"),
    ("Recall",    0.808, "center", "Recall"),
    ("F1",        0.868, "center", "F1"),
    ("Bewertung", 0.990, "right",  "rating"),
]

# Metriken die im Radar-Chart dargestellt werden (Reihenfolge = Achsenfolge)
RADAR_METRICS = ["mAP50-95", "mAP50", "Precision", "Recall", "F1"]

# ==============================================================================
# DATEN-ANALYSE
# ==============================================================================

def get_model_info(run_folder: Path) -> dict:
    """Liest Modell-Variante und Eingabe-Auflösung aus args.yaml des Laufs."""
    info = {"variant": "?", "imgsz": "?", "epochs": "?"}
    args_path = run_folder / "args.yaml"
    if not args_path.exists():
        return info
    try:
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)
        model_name = str(args.get("model", "")).lower()
        model_base = Path(model_name).stem
        SIZE_LABELS = {"n": "Nano", "s": "Small", "m": "Medium",
                       "l": "Large", "b": "Base",  "x": "XL"}
        m = re.search(r"yolo(?:v?(\d+))([nsmblx])", model_base)
        if m:
            info["variant"] = f"v{m.group(1)}-{SIZE_LABELS.get(m.group(2), m.group(2).upper())}"
        else:
            info["variant"] = model_base or model_name
        info["imgsz"]  = args.get("imgsz", "?")
        info["epochs"] = args.get("epochs", "?")
    except Exception:
        pass
    return info


def analyze_run(csv_path: Path) -> dict | None:
    """
    Liest results.csv eines Trainingslaufs und gibt die Metriken der besten
    Epoche (nach mAP50-95) als Dict zurück. Gibt None zurück wenn die Datei
    nicht auswertbar ist.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception:
        return None

    # Spaltenname für mAP50-95 variiert je nach YOLO-Version
    map_col = next(
        (c for c in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP_0.5:0.95"]
         if c in df.columns), None
    )
    if not map_col:
        return None

    try:
        best = df.loc[df[map_col].idxmax()]
    except Exception:
        return None

    def get(candidates: list[str]) -> float:
        """Gibt den Wert der ersten gefundenen Spalte zurück, sonst 0.0."""
        for c in candidates:
            if c in df.columns:
                v = best[c]
                return float(v) if pd.notna(v) else 0.0
        return 0.0

    run_folder = csv_path.parent
    meta       = get_model_info(run_folder)
    best_epoch = int(best["epoch"])

    map50_95  = get([map_col])
    map50     = get(["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5"])
    precision = get(["metrics/precision(B)", "metrics/precision"])
    recall    = get(["metrics/recall(B)", "metrics/recall"])
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    score = (SCORE_WEIGHTS["mAP50-95"]  * map50_95  +
             SCORE_WEIGHTS["mAP50"]     * map50     +
             SCORE_WEIGHTS["Precision"] * precision +
             SCORE_WEIGHTS["Recall"]    * recall    +
             SCORE_WEIGHTS["F1"]        * f1)

    try:
        trained_at = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        trained_at = "?"

    return {
        "Name":      str(run_folder.relative_to(BASE_DIR)),
        "Variant":   meta["variant"],
        "Size":      str(meta["imgsz"]) + "px",
        "BestEpoch": str(best_epoch),
        "TrainedAt": trained_at,
        "mAP50-95":  map50_95,
        "mAP50":     map50,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "Score":     score,
        "Efficiency": map50_95 / max(best_epoch + 1, 1),
        "rating":    _rating(map50_95),
    }


def _rating(score: float) -> str:
    """Qualitätslabel basierend auf mAP50-95."""
    if score >= 0.85: return "Exzellent"
    if score >= 0.70: return "Sehr gut"
    if score >= 0.55: return "Gut"
    if score >= 0.40: return "Mittel"
    return "Schwach"


def load_ranked(top_n: int | None = None) -> list[dict]:
    """Sucht alle results.csv unter BASE_DIR und gibt nach mAP50-95 sortierte Liste zurück."""
    if not BASE_DIR.exists():
        print(f"Ordner '{BASE_DIR}' nicht gefunden.")
        sys.exit(1)
    files = list(BASE_DIR.rglob("results.csv"))
    if not files:
        print("Keine Trainingsergebnisse gefunden.")
        sys.exit(1)
    models = [r for f in files if (r := analyze_run(f))]
    ranked = sorted(models, key=lambda x: x["mAP50-95"], reverse=True)
    return ranked[:top_n] if top_n else ranked


def bar_color(rank: int) -> str:
    """Gibt die Akzentfarbe für einen bestimmten Rang zurück (0-basiert)."""
    return RANK_COLORS[rank] if rank < len(RANK_COLORS) else DEFAULT_COLOR


# ==============================================================================
# ZEICHNEN — Ranking-Tabelle
# ==============================================================================

def _rank_badge(ax, x: float, y: float, rank: int, row_h: float) -> None:
    """
    Zeichnet einen farbigen Rang-Badge (abgerundetes Rechteck + Rang-Nummer).
    Ersetzt Emoji-Medaillen, da DejaVu Sans keine Emoji-Glyphen enthält.
    """
    color   = bar_color(rank)
    bw, bh  = 0.030, row_h * 0.62
    ax.add_patch(FancyBboxPatch(
        (x - bw / 2, y - bh / 2), bw, bh,
        boxstyle="round,pad=0.003", linewidth=0,
        facecolor=color, zorder=3,
    ))
    ax.text(x, y, str(rank + 1), color=BADGE_TEXT, fontsize=7,
            fontweight="bold", ha="center", va="center", zorder=4)


def draw_ranking_table(ax, ranked: list[dict]) -> None:
    """
    Zeichnet die vollständige Ranking-Tabelle in den übergebenen Axes.
    Koordinatensystem: x in [0, 1], y in [0, 1].
    Header belegt die oberste Zeile, Datenzeilen folgen darunter.
    """
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n     = len(ranked)
    # row_h so gewählt, dass Header + n Datenzeilen exakt [0, 1] füllen
    row_h = 1.0 / (n + 1)

    # --- Header-Zeile (farbig hinterlegt, oberster Bereich) ---
    header_bottom = 1.0 - row_h
    header_mid    = 1.0 - row_h * 0.5
    ax.add_patch(FancyBboxPatch(
        (0, header_bottom), 1.0, row_h,
        boxstyle="square,pad=0", linewidth=0, facecolor=ACCENT, zorder=1,
    ))
    for label, x, ha, _ in TABLE_COLS:
        ax.text(x, header_mid, label, color="white", fontsize=7.5,
                fontweight="bold", ha=ha, va="center", zorder=2)

    # --- Datenzeilen (i=0 direkt unter dem Header) ---
    for i, m in enumerate(ranked):
        # +2 statt +1: Zeile 0 beginnt eine row_h unterhalb des Headers
        y_top = 1.0 - row_h * (i + 2)
        y_mid = y_top + row_h * 0.5
        bg    = BG_ROW_A if i % 2 == 0 else BG_ROW_B

        ax.add_patch(FancyBboxPatch(
            (0, y_top), 1.0, row_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=bg, zorder=1,
        ))

        for label, x, ha, key in TABLE_COLS:
            if key is None:
                _rank_badge(ax, x, y_mid, i, row_h)
                continue

            raw = m[key]
            if isinstance(raw, float):
                text = f"{raw:.4f}"
                # mAP50-95 farblich und fett hervorheben (Primär-Ranking-Metrik)
                bold = key == "mAP50-95"
                fc   = bar_color(i) if bold else FG_TEXT
                fs   = 8.0 if bold else 7.5
            else:
                text = str(raw)
                bold = False
                fc   = FG_TEXT
                fs   = 6.8 if key == "Name" else 7.5

            ax.text(x, y_mid, text, color=fc, fontsize=fs,
                    fontweight="bold" if bold else "normal",
                    ha=ha, va="center", zorder=2, clip_on=True)


# ==============================================================================
# ZEICHNEN — Radar-Chart
# ==============================================================================

def draw_radar(ax, ranked: list[dict], top: int = 5) -> None:
    """
    Radar-Chart (Spinnennetz) für die Top-N Modelle über alle RADAR_METRICS.
    Jedes Modell bekommt eine eigene farbige Linie + gefüllten Bereich.
    """
    models = ranked[:top]
    N      = len(RADAR_METRICS)

    # Winkel gleichmäßig verteilt; letzter Winkel = erster (Kreis schließen)
    angles = [n / N * 2 * math.pi for n in range(N)] + [0]

    # Achsen-Styling
    ax.set_facecolor(BG_PANEL)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.set_theta_offset(math.pi / 2)    # 12-Uhr-Position als Start
    ax.set_theta_direction(-1)           # Uhrzeigersinn
    ax.set_rlabel_position(25)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=6, colors=FG_MUTED)
    ax.grid(color=GRID_COLOR, linewidth=0.6)

    # Linie + Füllung für jedes Modell
    for i, m in enumerate(models):
        vals = [m[k] for k in RADAR_METRICS] + [m[RADAR_METRICS[0]]]
        c    = bar_color(i)
        ax.plot(angles, vals, color=c, linewidth=2.0, zorder=3)
        ax.fill(angles, vals, color=c, alpha=0.13, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, color=FG_TEXT, fontsize=9)
    ax.set_title(f"Radar — Top {len(models)}", color=FG_TEXT,
                 fontsize=10, fontweight="bold", pad=18)

    # Legende mit Modellnamen, deutlich größer für Lesbarkeit
    handles = [
        mpatches.Patch(color=bar_color(i),
                       label=f"#{i + 1}  {models[i]['Name'].split('/')[-1]}")
        for i in range(len(models))
    ]
    ax.legend(
        handles        = handles,
        loc            = "center right",
        bbox_to_anchor = (1.58, 0.5),
        fontsize       = 10,
        title          = "Modelle",
        title_fontsize = 10,
        facecolor      = BG_PANEL,
        edgecolor      = LEGEND_EDGE,
        labelcolor     = FG_TEXT,
        framealpha     = 0.95,
        borderpad      = 0.9,
        labelspacing   = 0.7,
    )


# ==============================================================================
# ZEICHNEN — Score-Gauge
# ==============================================================================

def draw_score_gauge(ax, ranked: list[dict]) -> None:
    """
    Horizontaler Fortschrittsbalken je Modell, normiert auf den besten Score.
    Unterhalb des Titels wird die Berechnungsformel des Scores angezeigt.
    """
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(ranked) - 0.5)
    ax.axis("off")

    # Formel aus SCORE_WEIGHTS dynamisch zusammensetzen
    formula = ("Score = " +
               "  +  ".join(
                   f"{w:.0%}\u00d7{k}"          # ×-Zeichen als Unicode
                   for k, w in SCORE_WEIGHTS.items()
               ))

    ax.set_title("Gesamt-Score (gewichtet)", color=FG_TEXT,
                 fontsize=9, fontweight="bold", pad=22)
    ax.text(0.5, 1.02, formula, transform=ax.transAxes,
            color=FG_MUTED, fontsize=6.5, ha="center", va="bottom",
            clip_on=False)

    # Höchster Score als Referenz für die Balkenlänge
    mx = max(m["Score"] for m in ranked) or 1.0

    for i, m in enumerate(ranked):
        y = len(ranked) - 1 - i                # Platz 1 oben
        w = m["Score"] / mx * 0.68             # Balkenbreite relativ zum Besten

        # Hintergrund-Track (volle Breite)
        ax.add_patch(FancyBboxPatch(
            (0.24, y - 0.28), 0.68, 0.56,
            boxstyle="round,pad=0.01", linewidth=0,
            facecolor=TRACK_COLOR, zorder=1,
        ))
        # Farbiger Fortschrittsbalken
        if w > 0:
            ax.add_patch(FancyBboxPatch(
                (0.24, y - 0.28), w, 0.56,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor=bar_color(i), zorder=2, alpha=0.88,
            ))

        c    = bar_color(i)
        name = m["Name"].split("/")[-1]
        ax.text(0.005, y, f"#{i + 1}", color=c, fontsize=7.5,
                fontweight="bold", va="center")
        ax.text(0.055, y, name, color=FG_TEXT, fontsize=6.8,
                va="center", clip_on=True)
        ax.text(0.938, y, f"{m['Score']:.4f}", color=FG_TEXT,
                fontsize=7, va="center", ha="right")


# ==============================================================================
# HILFSFUNKTIONEN — Speichern & Ordner
# ==============================================================================

def make_output_dir(fmt: str, theme: str) -> tuple[Path, str, str]:
    """
    Erstellt einen datierten Ausgabe-Ordner unter REPORTS_DIR.
    Der Theme-Name ('dark' / 'light') wird im Ordnernamen vermerkt.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out   = REPORTS_DIR / f"comparison_{stamp}_{theme}"
    out.mkdir(parents=True, exist_ok=True)
    return out, stamp, fmt.lower().lstrip(".")


def _save(fig, path: Path, dpi: int) -> None:
    """Speichert eine Figur und schließt sie anschließend."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  [Einzel]   {path}")


# ==============================================================================
# GESAMT-BILD  (Tabelle + Radar + Score-Gauge in einer Figur)
# ==============================================================================

def build_overview(ranked: list[dict], out_dir: Path, stamp: str,
                   fmt: str, dpi: int) -> None:
    """Erstellt und speichert die kombinierte Übersichts-Grafik."""
    n     = len(ranked)
    fig_h = max(13, 8 + n * 0.52)   # Höhe wächst mit Modellanzahl

    fig = plt.figure(figsize=(23, fig_h), facecolor=BG_DARK)

    # Vier vertikale Abschnitte: Titel | Tabelle | Lücke | Charts
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.055, 0.42, 0.06, 0.465],
        hspace=0.03, left=0.02, right=0.98, top=0.97, bottom=0.02,
    )

    # --- Titel-Zeile ---
    ax_t = fig.add_subplot(outer[0])
    ax_t.set_facecolor(BG_DARK)
    ax_t.axis("off")
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    ax_t.text(0.5, 0.62, "YOLO  —  Modell-Vergleich & Ranking",
              color=FG_TEXT, fontsize=17, fontweight="bold",
              ha="center", va="center", transform=ax_t.transAxes)
    ax_t.text(0.5, 0.08,
              f"Erstellt: {now}   |   Modelle: {n}   |   Quelle: {BASE_DIR}/",
              color=FG_MUTED, fontsize=8.5,
              ha="center", va="center", transform=ax_t.transAxes)

    # --- Ranking-Tabelle ---
    ax_table = fig.add_subplot(outer[1])
    draw_ranking_table(ax_table, ranked)

    # --- Trenn-Lücke (unsichtbar, nur Abstand) ---
    ax_gap = fig.add_subplot(outer[2])
    ax_gap.set_facecolor(BG_DARK)
    ax_gap.axis("off")

    # --- Chart-Bereich: Radar links, Score-Gauge rechts ---
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3], wspace=0.40,
    )
    ax_radar = fig.add_subplot(inner[0], polar=True)
    ax_gauge = fig.add_subplot(inner[1])

    draw_radar(ax_radar, ranked, top=min(5, n))
    draw_score_gauge(ax_gauge, ranked)

    path = out_dir / f"overview_{stamp}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  [Gesamt]   {path}")


# ==============================================================================
# EINZELBILDER  (Tabelle, Radar, Score-Gauge separat)
# ==============================================================================

def build_split(ranked: list[dict], out_dir: Path, stamp: str,
                fmt: str, dpi: int) -> None:
    """Speichert jeden Chart-Typ als eigene Datei in höherer Auflösung."""
    n = len(ranked)

    # --- Ranking-Tabelle ---
    fig_h = max(4, 1.5 + n * 0.38)
    fig   = plt.figure(figsize=(22, fig_h), facecolor=BG_DARK)
    ax    = fig.add_subplot(111)
    draw_ranking_table(ax, ranked)
    fig.suptitle("YOLO — Ranking-Tabelle", color=FG_TEXT,
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, out_dir / f"table_{stamp}.{fmt}", dpi)

    # --- Radar (+40 DPI für bessere Lesbarkeit der Beschriftungen) ---
    fig = plt.figure(figsize=(11, 10), facecolor=BG_DARK)
    ax  = fig.add_subplot(111, polar=True)
    draw_radar(ax, ranked, top=min(5, n))
    _save(fig, out_dir / f"radar_{stamp}.{fmt}", dpi + 40)

    # --- Score-Gauge ---
    fig_h   = max(4, n * 0.52 + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)
    draw_score_gauge(ax, ranked)
    _save(fig, out_dir / f"score_{stamp}.{fmt}", dpi)


# ==============================================================================
# EINSTIEGSPUNKT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO Modell-Vergleich — exportiert Ranking als Bild(er)"
    )
    p.add_argument("--top",      type=int, default=None,
                   help="Nur die besten N Modelle einbeziehen")
    p.add_argument("--dpi",      type=int, default=130,
                   help="Basis-Auflösung in DPI (Standard: 130)")
    p.add_argument("--fmt",      default="png", choices=["png", "jpg"],
                   help="Ausgabe-Format (Standard: png)")
    p.add_argument("--no-split", action="store_true",
                   help="Keine Einzelbilder — nur das Gesamt-Bild speichern")
    p.add_argument("--light",    action="store_true",
                   help="Helles Theme verwenden (Standard: dunkles Theme)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    theme  = apply_theme(light=args.light)
    ranked = load_ranked(args.top)
    out_dir, stamp, fmt = make_output_dir(args.fmt, theme)

    print(f"\n  {len(ranked)} Modelle geladen.  Theme: {theme}")
    print(f"  Ausgabe-Ordner: {out_dir}\n")

    build_overview(ranked, out_dir, stamp, fmt, args.dpi)

    if not args.no_split:
        print()
        build_split(ranked, out_dir, stamp, fmt, args.dpi)

    print(f"\n  Fertig. Alle Dateien in: {out_dir}/\n")


if __name__ == "__main__":
    main()
