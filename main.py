"""
FastAPI Service zur Auswertung von Schützenvereins-Scorecards (PP1, NPA etc.)
mit einem lokalen Vision-Modell via Ollama.
"""

import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
from pydantic import BaseModel, Field, field_validator

# Optional: EasyOCR fuer handschriftliche Zahlen
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")
USE_EASYOCR = os.environ.get("USE_EASYOCR", "1") == "1" and EASYOCR_AVAILABLE
MAX_IMAGE_SIZE = (1536, 1536)  # Größer für bessere handschriftliche Erkennung

app = FastAPI(
    title="Scorecard OCR API",
    description="Liest Schützenvereins-Auswertekarten per lokalem Ollama-Vision-Modell aus.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic-Modelle
# ---------------------------------------------------------------------------
class ErgebnisZeile(BaseModel):
    """Eine einzelne Zeile in der Ergebnis-Tabelle."""
    kategorie: str = Field("?", description="Ring-Kategorie, z.B. 'X', '10', '9', 'miss'")
    station_1_links: Optional[str] = Field(None, description="Handschriftlicher Wert linke/spalte (PP1=station1, NPA=linke Scheibe)")
    station_2_rechts: Optional[str] = Field(None, description="Handschriftlicher Wert rechte Spalte (PP1=station2+3, NPA=rechte Scheibe)")
    treffer: Optional[str] = Field(None, description="Wert in Treffer/Spalte falls eingetragen")
    ringe: Optional[str] = Field(None, description="Wert in Ringe/Spalte falls eingetragen")

    @field_validator("*", mode="before")
    @classmethod
    def _fix_fields(cls, v):
        return _coerce_str(v)


class Ergebnisse(BaseModel):
    """Zusammengefasste Ergebnisse der Scorecard."""
    zeilen: List[ErgebnisZeile] = Field(default_factory=list)
    station_1_total: Optional[str] = Field(None, description="Summe Station 1 / linke Scheibe")
    station_2_total: Optional[str] = Field(None, description="Summe Station 2+3 / rechte Scheibe")
    treffer_gesamt: Optional[str] = Field(None, description="Gesamt-Treffer")
    ringe_gesamt: Optional[str] = Field(None, description="Gesamt-Ringe/Score")

    @field_validator("station_1_total", "station_2_total", "treffer_gesamt", "ringe_gesamt", mode="before")
    @classmethod
    def _fix_fields(cls, v):
        return _coerce_str(v)


def _coerce_str(v):
    """Hilfsfunktion: Zahlen -> String, Listen/Dicts -> None, Leerstring -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return str(v)
    return None


class ScorecardAuswertung(BaseModel):
    """Vollständiges Auswertungsergebnis einer Scorecard."""
    schuetze: Optional[str] = Field(None, description="Name des Schützen aus Feld 'Schütze / shooter'")
    disziplin: Optional[str] = Field(None, description="Erkannte Disziplin, z.B. 'PP1(AUT)' oder 'NPA(AUT)'")
    disziplin_checkbox: Optional[str] = Field(None, description="Welche Checkbox ist angekreuzt, z.B. 'PP1' oder 'NPA'")
    stand_details: Optional[str] = Field(None, description="Stand-Info, z.B. 'P2 25m Stand xy'")
    datum_zeit: Optional[str] = Field(None, description="Datum und Uhrzeit, z.B. '21.09.2026 18:00 - 21:00'")
    ergebnisse: Ergebnisse = Field(default_factory=Ergebnisse)
    auswerter: Optional[str] = Field(None, description="Name/Unterschrift des Auswerters")
    modell: str = Field(OLLAMA_MODEL, description="Verwendetes AI-Modell")

    @field_validator("schuetze", "disziplin", "disziplin_checkbox", "stand_details", "datum_zeit", "auswerter", mode="before")
    @classmethod
    def _fix_str_fields(cls, v):
        return _coerce_str(v)


# ---------------------------------------------------------------------------
# Prompt fuer qwen2.5vl:3b
# ---------------------------------------------------------------------------
PROMPT_KOPF = """Du siehst den oberen Teil einer Schuetzenvereins-Scorecard.
Lies die Textfelder und gib NUR JSON zurueck.

Felder:
- schuetze: Name unter "Schuetze / shooter"
- disziplin: z.B. "PP1(AUT)"
- disziplin_checkbox: angekreuzte Disziplin, z.B. "PP1"
- stand_details: z.B. "P2 25m Stand xy"
- datum_zeit: z.B. "21.09.2026 18:00 - 21:00"

Regeln: NUR JSON. Leer = null. Verwende AUSSCHLIESSLICH Werte aus dem Bild.
Schema: {"schuetze":null,"disziplin":null,"disziplin_checkbox":null,"stand_details":null,"datum_zeit":null}
"""

PROMPT_TABELLE = """Du siehst die Ergebnis-Tabelle einer Schuetzenvereins-Scorecard.
Lies die handschriftlichen Zahlen und gib NUR JSON zurueck.

AUFBAU - 5 Spalten von links nach rechts:
1. "station 1": handschriftliche Zahl
2. "station 2 + 3": handschriftliche Zahl
3. Kategorie (gedruckt, grauer Kasten): X, 10, 9, 8, 7, 6, 5, 0, miss
4. "Treffer hits": Zahl
5. "Ringe score": Zahl

AUFGABE:
- Lies fuer JEDE Kategorie-Zeile die handschriftlichen Werte in Spalte 1 und 2.
- Reihenfolge von oben nach unten: X, 10, 9, 8, 7, 6, 5, 0, miss.
- Letzte Zeile "total": station_1_total = Spalte 1, station_2_total = Spalte 2,
  treffer_gesamt = Spalte 4, ringe_gesamt = Spalte 5.

WICHTIG: Die null-Werte im Schema sind PLATZHALTER.
Ersetze sie durch die tatsaechlich gelesenen Zahlen aus dem Bild.
Leere Zelle im Bild = null. Schreibe Zahlen EXAKT wie im Bild (z.B. "12(10)").

Regeln: NUR JSON, keine Erklaerungen. DOPPELTE Anfuehrungszeichen.

Schema:
{"zeilen":[{"kategorie":"X","station_1_links":null,"station_2_rechts":null},{"kategorie":"10","station_1_links":null,"station_2_rechts":null},{"kategorie":"9","station_1_links":null,"station_2_rechts":null},{"kategorie":"8","station_1_links":null,"station_2_rechts":null},{"kategorie":"7","station_1_links":null,"station_2_rechts":null},{"kategorie":"6","station_1_links":null,"station_2_rechts":null},{"kategorie":"5","station_1_links":null,"station_2_rechts":null},{"kategorie":"0","station_1_links":null,"station_2_rechts":null},{"kategorie":"miss","station_1_links":null,"station_2_rechts":null}],"station_1_total":null,"station_2_total":null,"treffer_gesamt":null,"ringe_gesamt":null}
"""

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """Skaliert und schaerft das Bild fuer bessere OCR."""
    image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    # Mehr Schaerfe fuer handschriftliche Zahlen
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    # Weniger Kontrast als vorher, damit blaue Tinte nicht verschwindet
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    # Helligkeit leicht erhoehen
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    return image


def split_for_regions(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Teilt das Bild in Kopf (Text) und Tabelle (Zahlen) ohne Verzerrung."""
    w, h = image.size
    # Kopf: obere 48% (Textbox + Checkboxen komplett)
    kopf = image.crop((0, 0, w, int(h * 0.48)))
    # Tabelle: untere 62%, Seitenverhaeltnis beibehalten
    tabelle = image.crop((0, int(h * 0.38), w, h))
    return kopf, tabelle


KATEGORIEN = ["X", "10", "9", "8", "7", "6", "5", "0", "miss", "total"]


def detect_table_structure(image: Image.Image) -> Optional[dict]:
    """
    EasyOCR-Strukturpass: liefert Zeilen-Y-Positionen und Spalten-X-Zentren
    der Tabelle, verankert an den gedruckten Kategorie-Labels (X..total).
    Die handschriftlichen Werte werden NICHT hier gelesen - das macht das LLM
    pro Zelle (siehe read_table_cells).
    """
    if not EASYOCR_AVAILABLE:
        return None

    import numpy as np

    # Lazy-Initialisierung
    if not hasattr(detect_table_structure, "_reader"):
        detect_table_structure._reader = easyocr.Reader(["en", "de"], gpu=False)
    reader = detect_table_structure._reader

    w, h = image.size
    arr = np.array(image)
    gray_f = np.array(image.convert("L")).astype(float)

    items = []
    for bbox, text, conf in reader.readtext(arr):
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({
            "text": text, "conf": float(conf),
            "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
        })
    if not items:
        return None

    # --- Dynamische Kategorien & Zeilenanzahl bestimmen ---
    miss_its = [it for it in items
                if it["text"].lower() == "miss" and it["conf"] >= 0.35 and it["cy"] > 0.3 * h]
    total_its = [it for it in items
                 if it["text"].lower() == "total" and it["conf"] >= 0.35 and it["cy"] > 0.3 * h]

    if miss_its and total_its:
        miss_it = max(miss_its, key=lambda x: x["conf"])
        total_it = max(total_its, key=lambda x: x["conf"])
        cat_x = (miss_it["cx"] + total_it["cx"]) / 2.0
        zero_its = [
            it for it in items
            if it["text"] == "0" and abs(it["cx"] - cat_x) <= 0.06 * w
            and 0 < miss_it["cy"] - it["cy"] < 0.15 * h and it["conf"] >= 0.35
        ]
        row_h_0 = ((total_it["cy"] - zero_its[0]["cy"]) / 2.0
                   if zero_its else (total_it["cy"] - miss_it["cy"]))

        hdr_keywords = (
            "station", "slation", "linke", "tinke", "iinke",
            "rechte", "techte", "scheibe", "schelbe", "schelbo",
            "treffer", "trerier", "hits", "ringe", "score",
        )
        hdr_tokens = [
            it for it in items
            if it["cy"] < miss_it["cy"] - 2.5 * row_h_0
            and it["cx"] > 0.12 * w
            and any(k in it["text"].lower() for k in hdr_keywords)
        ]
        header_cy = (sum(it["cy"] for it in hdr_tokens) / len(hdr_tokens)
                     if hdr_tokens else 0.18 * h)

        x_its = [
            it for it in items
            if it["text"].upper() == "X" and abs(it["cx"] - cat_x) <= 0.06 * w
            and header_cy + 0.3 * row_h_0 < it["cy"] < miss_it["cy"] - 3.0 * row_h_0
            and it["conf"] >= 0.4
        ]
        if x_its:
            n_data_rows = max(5, min(12, round((total_it["cy"] - x_its[0]["cy"]) / row_h_0)))
            row_h_0 = (total_it["cy"] - x_its[0]["cy"]) / n_data_rows
        elif hdr_tokens:
            n_data_rows = max(5, min(12, round((total_it["cy"] - header_cy) / row_h_0) - 1))
            row_h_0 = (total_it["cy"] - header_cy) / (n_data_rows + 1.15)
        else:
            n_data_rows = 9

        # Zeilenhoehe ueber Druckzeichen-Signal in der Kategoriespalte feinjustieren
        r_in = max(6, int(row_h_0 * 0.26))
        r_out1 = max(r_in + 3, int(row_h_0 * 0.38))
        r_out2 = max(r_out1 + 4, int(row_h_0 * 0.68))
        c_prof = gray_f[:, max(0, int(cat_x - r_in)):min(w, int(cat_x + r_in))].mean(axis=1)
        s_prof = 0.5 * (
            gray_f[:, max(0, int(cat_x - r_out2)):max(1, int(cat_x - r_out1))].mean(axis=1)
            + gray_f[:, min(w - 1, int(cat_x + r_out1)):min(w, int(cat_x + r_out2))].mean(axis=1)
        )
        char_sig = np.maximum(0.0, s_prof - c_prof)
        best_rh = max(
            np.linspace(row_h_0 * 0.94, row_h_0 * 1.06, 61),
            key=lambda rh: sum(
                char_sig[max(0, int(round(total_it["cy"] - k * rh)) - 6):
                         min(h, int(round(total_it["cy"] - k * rh)) + 7)].mean()
                for k in range(2, n_data_rows + 1)
            ),
        )

        # Gedruckte Zahlen-Kategorien zwischen "X" und "0" bestimmen
        hbox_list = [
            [
                max(0, int(cat_x - best_rh * 0.65)),
                min(w, int(cat_x + best_rh * 0.65)),
                max(0, int((total_it["cy"] - k * best_rh) - best_rh * 0.36)),
                min(h, int((total_it["cy"] - k * best_rh) + best_rh * 0.36)),
            ]
            for k in range(n_data_rows, 1, -1)
        ]
        rec_cats = reader.recognize(
            arr, horizontal_list=hbox_list, free_list=[], allowlist="0123456789X"
        )
        offset_votes = []
        for it in items:
            if abs(it["cx"] - cat_x) <= 0.06 * w and it["conf"] >= 0.5 and it["text"].isdigit():
                val = int(it["text"])
                k = round((total_it["cy"] - it["cy"]) / best_rh)
                if (3 <= k <= n_data_rows - 1 and 1 <= val <= 10
                        and abs((total_it["cy"] - it["cy"]) - k * best_rh) <= 0.25 * best_rh):
                    offset_votes.append(val - k)
        for k, (_, txt, conf) in zip(range(n_data_rows, 1, -1), rec_cats):
            if 3 <= k <= n_data_rows - 1 and conf >= 0.75 and txt.isdigit():
                val = int(txt)
                if 1 <= val <= 10:
                    offset_votes.append(val - k)
        offset = (max(set(offset_votes), key=offset_votes.count)
                  if offset_votes else (2 if n_data_rows == 9 else n_data_rows - 8))
        categories = (
            ["X"]
            + [str(offset + k) for k in range(n_data_rows - 1, 2, -1)]
            + ["0", "miss", "total"]
        )
    else:
        categories = list(KATEGORIEN)
        n_data_rows = 9

    if categories != KATEGORIEN:
        cat_lower = [c.lower() for c in categories]
        anchors = {}
        for it in items:
            if it["text"].lower() not in cat_lower:
                continue
            box_h = it["y1"] - it["y0"]
            if (abs(it["cx"] - cat_x) <= 0.06 * w and it["conf"] >= 0.45
                    and box_h <= 0.10 * h and it["cy"] > header_cy):
                idx = cat_lower.index(it["text"].lower())
                if abs((total_it["cy"] - it["cy"]) - (n_data_rows - idx) * best_rh) <= 0.35 * best_rh:
                    anchors.setdefault(idx, []).append(it["cy"])
        anchors = {idx: sum(v) / len(v) for idx, v in anchors.items()}
        for k in range(2, n_data_rows + 1):
            idx = n_data_rows - k
            if idx not in anchors:
                cy0 = total_it["cy"] - k * best_rh
                y_lo = max(0, int(round(cy0 - best_rh * 0.25)))
                y_hi = min(h, int(round(cy0 + best_rh * 0.25)) + 1)
                w_sig = char_sig[y_lo:y_hi]
                if w_sig.sum() > 20.0:
                    ys_idx = np.arange(y_lo, y_hi)
                    anchors[idx] = float(np.sum(ys_idx * w_sig) / np.sum(w_sig))

        idxs = np.array(sorted(anchors.keys()))
        cys = np.array([anchors[i] for i in idxs])
        b, a = np.polyfit(idxs, cys, 1)
        row_cy = {idx: anchors.get(idx, a + b * idx) for idx in range(len(categories))}

        top_anchor = row_cy[0]
        header_lists = {"s1": [], "s2": [], "tr": [], "rg": []}
        has_scheibe_hdr = False
        for it in items:
            if it["cy"] >= top_anchor or it["cx"] < 0.12 * w:
                continue
            t = it["text"].lower()
            if ("station" in t or "slation" in t) and it["cx"] < cat_x:
                if "2" in t or "+" in t:
                    header_lists["s2"].append(it["cx"])
                elif "1" in t or not header_lists["s1"]:
                    header_lists["s1"].append(it["cx"])
            elif it["cx"] < cat_x and ("+" in t or t.replace(" ", "") == "2+3"):
                header_lists["s2"].append(it["cx"])
            elif it["cx"] < cat_x and any(k in t for k in ("linke", "tinke", "iinke")):
                header_lists["s1"].append(it["cx"])
                has_scheibe_hdr = True
            elif it["cx"] < cat_x and any(k in t for k in ("rechte", "techte")):
                header_lists["s2"].append(it["cx"])
                has_scheibe_hdr = True
            elif it["cx"] < cat_x and any(k in t for k in ("scheibe", "schelbe", "schelbo")):
                has_scheibe_hdr = True
                if it["cx"] < cat_x - 0.18 * w:
                    header_lists["s1"].append(it["cx"])
                else:
                    header_lists["s2"].append(it["cx"])
            elif it["cx"] > cat_x and ("treffer" in t or "trerier" in t or "hits" in t):
                header_lists["tr"].append(it["cx"])
            elif it["cx"] > cat_x and ("ringe" in t or "score" in t):
                header_lists["rg"].append(it["cx"])

        header = {k: (sum(v) / len(v) if v else None) for k, v in header_lists.items()}
        if header["tr"] is not None and (has_scheibe_hdr or header["s1"] is None or header["s2"] is None):
            col_w = header["tr"] - cat_x
        elif header["s1"] is not None and header["s2"] is not None:
            col_w = header["s2"] - header["s1"]
        elif header["s2"] is not None:
            col_w = cat_x - header["s2"]
        elif header["s1"] is not None:
            col_w = (cat_x - header["s1"]) / 2.0
        else:
            col_w = w * 0.145
    else:
        cat_lower = [c.lower() for c in categories]
        cat_tokens = [it for it in items if it["text"].lower() in cat_lower]
        if len(cat_tokens) < 2:
            logger.warning("EasyOCR: weniger als 2 Kategorie-Labels erkannt")
            return None
        mt_xs = [it["cx"] for it in items
                 if it["text"].lower() in ("miss", "total") and it["conf"] >= 0.5]
        cat_x = (sum(mt_xs) / len(mt_xs)
                 if mt_xs else sorted(t["cx"] for t in cat_tokens)[len(cat_tokens) // 2])
        anchors = {}
        for it in cat_tokens:
            box_h = it["y1"] - it["y0"]
            if (abs(it["cx"] - cat_x) <= 0.06 * w
                    and it["conf"] >= 0.35
                    and box_h <= 0.10 * h
                    and it["cy"] > 0.15 * h):
                idx = cat_lower.index(it["text"].lower())
                anchors.setdefault(idx, []).append(it["cy"])
        anchors = {idx: sum(v) / len(v) for idx, v in anchors.items()}
        if len(anchors) < 2:
            logger.warning("EasyOCR: zu wenige Zeilen-Anker")
            return None
        anchor_idxs = sorted(anchors.keys())
        best_inliers = []
        best_err = float("inf")
        for i in range(len(anchor_idxs)):
            for j in range(i + 1, len(anchor_idxs)):
                idx1, idx2 = anchor_idxs[i], anchor_idxs[j]
                b_cand = (anchors[idx2] - anchors[idx1]) / (idx2 - idx1)
                if not (10.0 <= b_cand <= h * 0.25):
                    continue
                a_cand = anchors[idx1] - b_cand * idx1
                tol = max(6.0, 0.22 * b_cand)
                inliers = [k for k in anchor_idxs
                           if abs(anchors[k] - (a_cand + b_cand * k)) <= tol]
                err = sum(abs(anchors[k] - (a_cand + b_cand * k)) for k in inliers) / len(inliers)
                if (len(inliers) > len(best_inliers)
                        or (len(inliers) == len(best_inliers) and err < best_err)):
                    best_inliers = inliers
                    best_err = err
        if len(best_inliers) >= 2:
            anchors = {k: anchors[k] for k in best_inliers}

        idxs = np.array(list(anchors.keys()))
        cys = np.array(list(anchors.values()))
        b, a = np.polyfit(idxs, cys, 1)
        row_cy = {idx: anchors.get(idx, a + b * idx) for idx in range(len(categories))}

        top_anchor = min(anchors.values())
        header = {"s1": None, "s2": None, "tr": None, "rg": None}
        for it in items:
            if it["cy"] >= top_anchor:
                continue
            t = it["text"].lower()
            if ("station" in t or "slation" in t) and it["cx"] < cat_x:
                if "2" in t or "+" in t:
                    header["s2"] = it["cx"]
                elif "1" in t or header["s1"] is None:
                    header["s1"] = it["cx"]
            elif it["cx"] < cat_x and ("+" in t or t.replace(" ", "") == "2+3"):
                header["s2"] = it["cx"]
            elif it["cx"] > cat_x and ("treffer" in t or "hits" in t):
                header["tr"] = it["cx"]
            elif it["cx"] > cat_x and ("ringe" in t or "score" in t):
                header["rg"] = it["cx"]

        if header["s1"] is not None and header["s2"] is not None:
            col_w = header["s2"] - header["s1"]
        elif header["tr"] is not None:
            col_w = header["tr"] - cat_x
        else:
            col_w = w * 0.145
    centers = [
        header["s1"] if header["s1"] is not None else cat_x - 2 * col_w,
        header["s2"] if header["s2"] is not None else cat_x - col_w,
        cat_x,
        header["tr"] if header["tr"] is not None else cat_x + col_w,
        header["rg"] if header["rg"] is not None else cat_x + 2 * col_w,
    ]

    # Geometrie-Validierung: Spalten muessen streng aufsteigend und sinnvoll
    # dimensioniert sein, sonst -> Fallback auf LLM-Gesamtbild
    if (col_w < 15 or col_w > w * 0.6 or b < 10 or b > h * 0.4
            or not (centers[0] < centers[1] < centers[2] < centers[3] < centers[4])):
        logger.warning("EasyOCR: degenerierte Tabellen-Geometrie (centers=%s, col_w=%.0f, row_h=%.0f)",
                       [round(c) for c in centers], col_w, b)
        return None

    return {
        "categories": categories,
        "row_cy": row_cy,
        "centers": centers,
        "half_w": col_w / 2 * 0.9,
        "row_h": b,
        "w": w,
        "h": h,
        "items": items,
    }


def _estimate_tilt(image: Image.Image, max_deg: float = 8.0, step: float = 0.25) -> float:
    """
    Schaetzt die Verkippung der Tabelle: rotiert probehalber und maximiert
    die Varianz der horizontalen Dunkel-Pixel-Projektion (Gitterlinien
    ergeben scharfe Peaks wenn sie horizontal ausgerichtet sind).
    """
    import numpy as np

    small = image.convert("L")
    if small.width > 700:
        small = small.resize((700, max(1, int(small.height * 700 / small.width))),
                             Image.Resampling.BILINEAR)
    best_ang, best_std = 0.0, -1.0
    ang = -max_deg
    while ang <= max_deg + 1e-9:
        rot = small.rotate(ang, resample=Image.Resampling.BILINEAR,
                           expand=False, fillcolor=255)
        std = float((np.array(rot) < 140).sum(axis=1).std())
        if std > best_std:
            best_std, best_ang = std, ang
        ang += step
    return best_ang


PROMPT_SPALTE = """Du siehst die Ergebnis-Tabelle einer Schiess-Scorecard.
5 Spalten: "station 1" | "station 2 + 3" | Kategorie(grau) | "Treffer" | "Ringe".
Die Zeilen-Kategorien (grau): {cat_str}.

AUFGABE: Lies NUR die handschriftlichen Zahlen in der Spalte "{col_name}".
Ordne jeden Wert der Kategorie seiner Zeile zu.

Antworte NUR als JSON-Objekt mit allen {n_rows} Kategorien als Schluessel:
{tmpl_json}

"ZAHL" ist ein PLATZHALTER - ersetze ihn durch die Zahl die du wirklich im Bild siehst.
null = leere oder durchgestrichene/unkenntliche Zelle.
Die gedruckten Kategorie-Labels (X,10,9,...) sind KEINE Werte - nicht als Wert ausgeben!
"""

PROMPT_SPALTE_ISOLIERT = """Du siehst die Ergebnis-Tabelle einer Schiess-Scorecard.
Links steht die Wert-Spalte mit handschriftlichen Zahlen, direkt rechts daneben die graue Kategorie-Spalte ({cat_str}).

AUFGABE: Lies NUR die handschriftlichen Zahlen in der ersten Spalte links.
Ordne jeden Wert der Kategorie seiner Zeile zu.

Antworte NUR als JSON-Objekt mit allen {n_rows} Kategorien als Schluessel:
{tmpl_json}

"ZAHL" ist ein PLATZHALTER - ersetze ihn durch die Zahl die du wirklich im Bild siehst.
null = leere oder durchgestrichene/unkenntliche Zelle.
Die gedruckten Kategorie-Labels ({cat_str}) sind KEINE Werte - nicht als Wert ausgeben!
"""

PROMPT_TOTALZEILE = """Du siehst die Ergebnis-Tabelle einer Schiess-Scorecard.
In der letzten Zeile ("total") stehen 4 Werte von links nach rechts:
1. station_1_total: Format z.B. "12(10)"
2. station_2_total: Format z.B. "18(20)"
3. treffer_gesamt: z.B. "30"
4. ringe_gesamt: z.B. "300"
Antworte NUR als JSON:
{"station_1_total":null,"station_2_total":null,"treffer_gesamt":null,"ringe_gesamt":null}
"""


def _ink_ratio(gray, box, center=False) -> float:
    """Dunkler-Pixel-Anteil in einer Zelle.
    center=True misst nur die Zellmitte (innerste 60%x50%) - dort schlaegt
    Ueberlauf-Tinte aus Nachbarzellen nicht zu, echte Handschrift schon.
    """
    x0, y0, x1, y1 = box
    cell = gray[y0:y1, x0:x1]
    ch, cw = cell.shape
    if ch < 5 or cw < 5:
        return 0.0
    if center:
        inner = cell[int(ch * .25):int(ch * .75), int(cw * .20):int(cw * .80)]
        if inner.size == 0:
            return 0.0
        dark = inner < 125
        # Echte Ziffern erstrecken sich vertikal ueber mindestens 45% des
        # Zellzentrums; horizontale Gitterlinien/Bleed treffen nur wenige Zeilen.
        if (dark.sum(axis=1) > 0).mean() < 0.45:
            return 0.0
        return float(dark.mean() * 100)
    else:
        inner = cell[int(ch * .15):int(ch * .85), int(cw * .10):int(cw * .90)]
        if inner.size == 0:
            return 0.0
        return float((inner < 125).mean() * 100)


def _is_slash_cell(gray, box) -> tuple[bool, float]:
    """Erkennt duenne diagonale Durchstreich-Linien (/ oder \\) in leeren Zeilen."""
    import numpy as np

    x0, y0, x1, y1 = box
    cell = gray[y0:y1, x0:x1]
    ch, cw = cell.shape
    if ch < 5 or cw < 5:
        return False, 0.0
    inner = cell[int(ch * .18):int(ch * .82), int(cw * .10):int(cw * .90)] < 135
    ys, xs = np.where(inner)
    if len(xs) < 8 or np.std(xs) == 0 or np.std(ys) == 0:
        return False, 0.0
    corr = float(np.corrcoef(xs, ys)[0, 1])
    row_widths = [xs[ys == y_r].max() - xs[ys == y_r].min() + 1 for y_r in np.unique(ys)]
    max_w = float(np.max(row_widths))
    y_span = (ys.max() - ys.min() + 1) / inner.shape[0]
    is_slash = abs(corr) >= 0.88 and max_w <= max(8.0, cw * 0.08) and y_span >= 0.45
    return is_slash, corr


async def read_table_cells(image: Image.Image, struct: dict) -> dict:
    """
    Liest die Tabelle mit wenigen LLM-Aufrufen:
    - 1 Call pro Wert-Spalte (Liste mit 9 Werten von oben nach unten) -
      eine einzelne Spalte top-to-bottom ist eine Aufgabe, die auch kleine
      VLMs zuverlaessig loesen.
    - 1 Call fuer die Total-Zeile.
    - Ink-Gate pro Zelle: kein Pixel -> kein Wert (killt Halluzinationen
      in leeren Zellen/Spalten).
    """
    import asyncio
    import numpy as np

    centers, half_w, row_h = struct["centers"], struct["half_w"], struct["row_h"]
    w_img, h_img = struct["w"], struct["h"]
    row_cy = struct["row_cy"]
    categories = struct.get("categories") or KATEGORIEN
    cats_data = categories[:-1]
    n_rows = len(cats_data)
    gray = np.array(image.convert("L"))
    INK_THRESHOLD = 3.0  # % dunkle Pixel

    def _cell_box(row_idx, col_idx):
        cy = row_cy[row_idx]
        return (
            max(0, int(centers[col_idx] - half_w)),
            max(0, int(cy - row_h * 0.55)),
            min(w_img, int(centers[col_idx] + half_w)),
            min(h_img, int(cy + row_h * 0.55)),
        )

    # Mehrzeilige diagonale Durchstreich-Striche (/ oder \) erkennen,
    # die ueber >= 2 aufeinanderfolgende Zeilen laufen.
    slash_cells = set()
    for col_idx in (0, 1, 3, 4):
        col_slashes = [_is_slash_cell(gray, _cell_box(r, col_idx)) for r in range(n_rows)]
        for r in range(n_rows):
            sl, corr = col_slashes[r]
            if not sl:
                continue
            prev_ok = r > 0 and col_slashes[r - 1][0] and (col_slashes[r - 1][1] * corr > 0)
            next_ok = r < n_rows - 1 and col_slashes[r + 1][0] and (col_slashes[r + 1][1] * corr > 0)
            if prev_ok or next_ok:
                slash_cells.add((r, col_idx))

    def cell_ink(row_idx, col_idx, center=False):
        if (row_idx, col_idx) in slash_cells:
            return 0.0
        return _ink_ratio(gray, _cell_box(row_idx, col_idx), center=center)

    # Enger Tabellen-Crop ohne Seitenspalte. Fuer Spalten-Calls endet der
    # Crop unterhalb der miss-Zeile - die total-Zeile wuerde sonst in die
    # letzten Eintraege "hineinbluten". Der total-Call nutzt die total-Zeile.
    top = max(0, int(row_cy[0] - row_h * 1.5))
    bot_data = min(h_img, int(row_cy[n_rows - 1] + row_h * 1.2))   # bis miss
    bot_full = min(h_img, int(row_cy[n_rows] + row_h * 1.2))       # inkl. total
    gx0 = max(0, int(centers[0] - half_w))
    gx1 = min(w_img, int(centers[4] + half_w))
    if gx1 - gx0 < 50 or bot_full - top < 50:
        logger.warning("Ungueltige Grid-Geometrie - Fallback auf LLM-Gesamtbild")
        return {"zeilen": []}

    active_by_col = {
        c: [r for r in range(n_rows) if cell_ink(r, c, center=True) >= INK_THRESHOLD]
        for c in (0, 1, 3, 4)
    }

    def _encode_resized(crop_img: Image.Image) -> str:
        gw, gh = crop_img.size
        if gw < 1200:
            crop_img = crop_img.resize(
                (int(gw * 1.6), int(gh * 1.6)), Image.Resampling.LANCZOS
            )
        return encode_image(crop_img)

    def _col_b64(col_idx: int) -> str:
        # Isoliert die abgefragte linke/rechte Wert-Spalte direkt neben der
        # grauen Kategorie-Spalte, damit das VLM bei beidseitig ausgefuellten
        # Tabellen (z.B. NPA linke/rechte Scheibe) nicht die Nachbarspalte liest.
        if col_idx == 1:
            grid = image.crop((max(0, int(centers[1] - half_w)), top, gx1, bot_data))
        elif col_idx == 0 and active_by_col[1]:
            left_p = image.crop((gx0, top, min(w_img, int(centers[0] + half_w)), bot_data))
            right_p = image.crop((max(0, int(centers[2] - half_w * 0.80)), top, gx1, bot_data))
            grid = Image.new("RGB", (left_p.width + right_p.width, left_p.height), (255, 255, 255))
            grid.paste(left_p, (0, 0))
            grid.paste(right_p, (left_p.width, 0))
        else:
            grid = image.crop((gx0, top, gx1, bot_data))
        return _encode_resized(grid)

    full_b64 = _encode_resized(image.crop((gx0, top, gx1, bot_full)))

    cat_str = ", ".join(cats_data)
    tmpl_parts = [f'"{cats_data[0]}":"ZAHL"'] + [f'"{c}":null' for c in cats_data[1:]]
    tmpl_json = "{" + ",".join(tmpl_parts) + "}"

    async def col_list(col_idx: int, name: str):
        isolated = col_idx in (0, 1) and (col_idx == 1 or bool(active_by_col[1]))
        if isolated:
            prompt = PROMPT_SPALTE_ISOLIERT.format(
                cat_str=cat_str,
                n_rows=n_rows,
                tmpl_json=tmpl_json,
            )
        else:
            prompt = PROMPT_SPALTE.format(
                cat_str=cat_str,
                col_name=name,
                n_rows=n_rows,
                tmpl_json=tmpl_json,
            )
        return await query_ollama(_col_b64(col_idx), prompt, num_predict=512)

    # Spalten-Gate: nur Spalten fragen, in denen mindestens eine Zelle Tinte hat
    col_defs = [(0, "station 1"), (1, "station 2 + 3"),
                (3, "Treffer hits"), (4, "Ringe score")]
    jobs = {}
    for col_idx, name in col_defs:
        if active_by_col[col_idx]:
            jobs[col_idx] = col_list(col_idx, name)
    jobs["total"] = query_ollama(full_b64, PROMPT_TOTALZEILE, num_predict=256)

    results = await asyncio.gather(*jobs.values())
    lists = dict(zip(jobs.keys(), results))

    def _valid(v):
        return (isinstance(v, str)
                and v.upper() != "ZAHL"
                and bool(re.fullmatch(r"[\d()xX\-]+", v)))

    col_keys = {0: "station_1_links", 1: "station_2_rechts", 3: "treffer", 4: "ringe"}
    zeilen = [{"kategorie": cat} for cat in cats_data]
    for col_idx, key in col_keys.items():
        active = active_by_col[col_idx]
        d = lists.get(col_idx) or {}
        col_vals = {idx: None for idx in range(n_rows)}
        if active and d:
            # Wenn die oberste Zeile ("X") leer ist (0 not in active), das VLM
            # die gelesenen Werte aber ab dem ersten JSON-Schluessel ("X")
            # eingetragen hat, verschieben wir sie auf die aktiven Zeilen.
            v_first = _coerce_str(d.get(cats_data[0]))
            if 0 not in active and v_first and _valid(v_first):
                for k_idx, r_idx in enumerate(active):
                    if k_idx < n_rows:
                        v = _coerce_str(d.get(cats_data[k_idx]))
                        if v and _valid(v):
                            col_vals[r_idx] = v
            else:
                for r_idx in active:
                    cat = cats_data[r_idx]
                    v = _coerce_str(d.get(cat))
                    if v and _valid(v):
                        col_vals[r_idx] = v
        for idx in range(n_rows):
            zeilen[idx][key] = col_vals[idx]

    def _clean_total(val):
        s = _coerce_str(val)
        if not s or s.upper() == "ZAHL" or not re.search(r"\d", s):
            return None
        return s

    td = lists.get("total") or {}
    totals = {
        "station_1_total": _clean_total(td.get("station_1_total")),
        "station_2_total": _clean_total(td.get("station_2_total")),
        "treffer_gesamt": _clean_total(td.get("treffer_gesamt")),
        "ringe_gesamt": _clean_total(td.get("ringe_gesamt")),
    }

    # EasyOCR-Tokens in der Total-Zeile als Ergaenzung/Validierung nutzen
    total_cy = row_cy[n_rows]
    ocr_totals = {0: None, 1: None, 3: None, 4: None}
    for it in struct.get("items") or []:
        if abs(it["cy"] - total_cy) <= row_h * 0.65 and re.fullmatch(r"[\d()]+", it["text"].strip()):
            for c_idx in (0, 1, 3, 4):
                if abs(it["cx"] - centers[c_idx]) <= half_w * 1.1:
                    ocr_totals[c_idx] = it["text"].strip()
    for c_idx, t_key in ((0, "station_1_total"), (1, "station_2_total"),
                         (3, "treffer_gesamt"), (4, "ringe_gesamt")):
        if not totals[t_key] and ocr_totals[c_idx]:
            totals[t_key] = ocr_totals[c_idx]

    def _num(v):
        mm = re.match(r"\d+", v or "")
        return int(mm.group(0)) if mm else None

    # Wenn beide Stationsspalten aktiv sind (z.B. NPA linke & rechte Scheibe),
    # muessen station_1_total, station_2_total, treffer_gesamt und ringe_gesamt
    # zu den Spaltensummen bzw. der gedruckten Total-Zeile passen.
    max_ring = int(cats_data[1]) if len(cats_data) > 1 and cats_data[1].isdigit() else 10
    s1_nums = [_num(r.get("station_1_links")) for r in zeilen if _num(r.get("station_1_links")) is not None]
    s2_nums = [_num(r.get("station_2_rechts")) for r in zeilen if _num(r.get("station_2_rechts")) is not None]
    if s1_nums and s2_nums:
        s1_sum = sum(s1_nums)
        s2_sum = sum(s2_nums)
        totals["station_1_total"] = str(s1_sum)
        totals["station_2_total"] = str(s2_sum)
        totals["treffer_gesamt"] = str(s1_sum + s2_sum)
        if ocr_totals[4] and _num(ocr_totals[4]) != s1_sum + s2_sum:
            totals["ringe_gesamt"] = ocr_totals[4]
        else:
            totals["ringe_gesamt"] = str((s1_sum + s2_sum) * max_ring)

    # Checksum: die Stationswerte (Anzahl Schuesse je Kategorie) muessen sich
    # zum gedruckten treffer_gesamt addieren. Ist die Summe zu gross, ist ein
    # gelesener Wert ueberzaehlig (z.B. durchgestrichener Eintrag oder
    # Halluzination) - die Zelle, deren Entfernen die Summe exakt macht,
    # wird genullt (nur wenn sie eindeutig bestimmbar ist).
    m = re.match(r"\d+", totals.get("treffer_gesamt") or "")
    if m:
        target = int(m.group(0))
        filled = [(i, k) for i, r in enumerate(zeilen)
                  for k in ("station_1_links", "station_2_rechts")
                  if _num(r.get(k)) is not None]
        total = sum(_num(zeilen[i][k]) for i, k in filled)
        if total > target:
            fixes = [(i, k) for i, k in filled
                     if total - _num(zeilen[i][k]) == target]
            if len(fixes) == 1:
                i, k = fixes[0]
                logger.info("Checksum: %s=%s in Zeile %s verworfen "
                            "(durchgestrichen/halluziniert, Summe stimmt jetzt)",
                            k, zeilen[i][k], zeilen[i]["kategorie"])
                zeilen[i][k] = None
            elif (total - target) % 10 == 0:
                key_to_col = {"station_1_links": 0, "station_2_rechts": 1}

                def _x_span_ratio(r_idx, c_idx):
                    x0, y0, x1, y1 = _cell_box(r_idx, c_idx)
                    cell = gray[y0:y1, x0:x1]
                    ch, cw = cell.shape
                    inner = cell[int(ch * .22):int(ch * .78), int(cw * .12):int(cw * .88)] < 125
                    _, xs = np.where(inner)
                    return (xs.max() - xs.min() + 1) / inner.shape[1] if len(xs) > 5 else 0.0

                narrow_teens = [
                    (i, k) for i, k in filled
                    if 11 <= _num(zeilen[i][k]) <= 19
                    and _x_span_ratio(i, key_to_col[k]) < 0.32
                ]
                if len(narrow_teens) * 10 == total - target:
                    for i, k in narrow_teens:
                        old_v = zeilen[i][k]
                        zeilen[i][k] = str(_num(old_v) - 10)
                        logger.info("Checksum: %s=%s -> %s in Zeile %s korrigiert "
                                    "(einzelne Ziffer, Summe stimmt jetzt)",
                                    k, old_v, zeilen[i][k], zeilen[i]["kategorie"])
            elif fixes:
                logger.warning("Checksum: Summe %d > %d, mehrere moegliche "
                               "Fix-Zellen %s - keine automatische Korrektur",
                               total, target, fixes)
        elif total < target:
            logger.warning("Checksum: Summe %d < treffer_gesamt %d - "
                           "vermutlich fehlende Werte", total, target)

    s1_final = [_num(r.get("station_1_links")) for r in zeilen if _num(r.get("station_1_links")) is not None]
    s2_final = [_num(r.get("station_2_rechts")) for r in zeilen if _num(r.get("station_2_rechts")) is not None]
    tr_final = [_num(r.get("treffer")) for r in zeilen if _num(r.get("treffer")) is not None]
    rg_final = [_num(r.get("ringe")) for r in zeilen if _num(r.get("ringe")) is not None]
    zeilen.append({
        "kategorie": "total",
        "station_1_links": (
            str(sum(s1_final)) if (cell_ink(n_rows, 0, center=True) >= INK_THRESHOLD and s1_final) else None
        ),
        "station_2_rechts": (
            str(sum(s2_final)) if (cell_ink(n_rows, 1, center=True) >= INK_THRESHOLD and s2_final) else None
        ),
        "treffer": (
            str(sum(tr_final) if tr_final else sum(s1_final) + sum(s2_final))
            if (cell_ink(n_rows, 3, center=True) >= INK_THRESHOLD and (tr_final or s1_final or s2_final))
            else None
        ),
        "ringe": (
            str(sum(rg_final)) if (cell_ink(n_rows, 4, center=True) >= INK_THRESHOLD and rg_final) else None
        ),
    })

    return {
        "zeilen": zeilen,
        **totals,
    }


def encode_image(image: Image.Image) -> str:
    """Bild als Base64-String fuer Ollama."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


logger = logging.getLogger(__name__)


def extract_json(raw: str) -> str:
    """
    Extrahiert das JSON-Objekt aus einer Text-Antwort, die evtl. Erklärungstext
    oder Markdown-Codeblöcke drumherum hat. Liefert den ersten Inhalt zwischen
    den aeußersten geschweiften Klammern {} zurück.
    """
    raw = raw.strip()
    # Entferne evtl. Markdown-Codeblock
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def repair_json(text: str) -> str:
    """
    Repariert typische JSON-Fehler, die kleine lokale LLMs produzieren:
    - einfache Anfuehrungszeichen -> doppelte
    - Python True/False -> true/false
    - Python None -> null
    - Trailing commas
    - Unquoted keys (best-effort)
    """
    # 1. Einfache Anfuehrungszeichen bei Strings ersetzen (nur aeussere, nicht innere)
    # Das ist heikel – wir versuchen es nur fuer offensichtliche Faelle
    text = re.sub(r"'([^']*)'(?=\s*[:,}\]])", r'"\1"', text)

    # 2. True/False/None -> true/false/null
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    # 3. Trailing commas vor } oder ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # 4. Unquoted keys (best-effort): key: -> "key":
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

    return text


def parse_json_safe(raw: str) -> dict:
    """Parst JSON mit Reparaturversuch, liefert leeres Dict bei Fehler."""
    cleaned = extract_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = repair_json(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.error("JSON parsing failed. Raw: %s", raw[:500])
            return {}


async def query_ollama(image_b64: str, prompt: str, num_predict: int = 2048) -> dict:
    """Fragt Ollama mit einem Bild ab."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt, "images": [image_b64]},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 4096},
    }
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama nicht erreichbar: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ollama-Fehler: {type(exc).__name__}: {exc}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama Status {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Ungueltige Ollama-Antwort: {exc}")

    raw = data.get("message", {}).get("content", "{}")
    logger.info("Raw model response (first 500 chars): %s", raw[:500])
    result = parse_json_safe(raw)
    if not result:
        logger.warning("JSON-Parse fehlgeschlagen, versuche Retry. Raw: %s", raw[:500])
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                resp = await client.post(url, json=payload)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Ollama-Fehler beim Retry: {exc}")
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "{}")
            result = parse_json_safe(raw)
    return result


# ---------------------------------------------------------------------------
# API-Endpunkte
# ---------------------------------------------------------------------------

@app.post(
    "/process-scorecard",
    response_model=ScorecardAuswertung,
    summary="Scorecard-Bild analysieren",
    description="Liest eine Schuetzenvereins-Auswertekarte per lokalem Ollama-Vision-Modell aus.",
)
async def process_scorecard(file: UploadFile = File(...)) -> ScorecardAuswertung:
    # --- Validierung ---
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Ungueltiger Dateityp: {file.content_type}. Nur JPEG/PNG/WebP erlaubt.",
        )

    # --- Bild laden ---
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild konnte nicht geladen werden: {exc}")

    # --- Bild vorverarbeiten und aufteilen ---
    processed = preprocess_image(image)
    kopf_img, tabelle_img = split_for_regions(processed)

    # --- Kopf mit qwen2.5vl:3b ---
    kopf_data = await query_ollama(encode_image(kopf_img), PROMPT_KOPF)

    # --- Tabelle: EasyOCR liefert Geometrie, LLM liest nur Zellen mit Tinte ---
    tabelle_data = {}
    if USE_EASYOCR:
        tilt = _estimate_tilt(tabelle_img)
        if abs(tilt) >= 0.5:
            logger.info("Tabelle um %.2f Grad entkippt", tilt)
            tabelle_img = tabelle_img.rotate(
                tilt, resample=Image.Resampling.BICUBIC, expand=False,
                fillcolor=(255, 255, 255))
        struct = detect_table_structure(tabelle_img)
        if struct:
            tabelle_data = await read_table_cells(tabelle_img, struct)
    if not tabelle_data or not tabelle_data.get("zeilen"):
        if USE_EASYOCR:
            logger.info("Keine Tabellen-Struktur gefunden - Fallback auf LLM-Gesamtbild")
        tabelle_data = await query_ollama(encode_image(tabelle_img), PROMPT_TABELLE)

    # --- Ergebnisse parsen und zusammenfuehren ---
    zeilen_raw = tabelle_data.get("zeilen") or []
    zeilen = [ErgebnisZeile.model_validate(z) for z in zeilen_raw if isinstance(z, dict)]

    ergebnisse = Ergebnisse(
        zeilen=zeilen,
        station_1_total=tabelle_data.get("station_1_total"),
        station_2_total=tabelle_data.get("station_2_total"),
        treffer_gesamt=tabelle_data.get("treffer_gesamt"),
        ringe_gesamt=tabelle_data.get("ringe_gesamt"),
    )

    return ScorecardAuswertung(
        schuetze=kopf_data.get("schuetze"),
        disziplin=kopf_data.get("disziplin"),
        disziplin_checkbox=kopf_data.get("disziplin_checkbox"),
        stand_details=kopf_data.get("stand_details"),
        datum_zeit=kopf_data.get("datum_zeit"),
        ergebnisse=ergebnisse,
        auswerter=kopf_data.get("auswerter"),
        modell=OLLAMA_MODEL,
    )


@app.get("/health")
async def health_check():
    # Pruefe ob Ollama erreichbar ist
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_erreichbar": ollama_ok,
        "modell": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
    }


# ---------------------------------------------------------------------------
# Lokaler Start (nur für Entwicklung)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
