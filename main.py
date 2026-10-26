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
    cat_lower = [c.lower() for c in KATEGORIEN]

    items = []
    for bbox, text, conf in reader.readtext(np.array(image)):
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({
            "text": text, "conf": conf,
            "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
        })
    if not items:
        return None

    cat_tokens = [it for it in items if it["text"].lower() in cat_lower]
    if len(cat_tokens) < 2:
        logger.warning("EasyOCR: weniger als 2 Kategorie-Labels erkannt")
        return None

    cat_x = sorted(t["cx"] for t in cat_tokens)[len(cat_tokens) // 2]
    anchors = {}
    for it in cat_tokens:
        if abs(it["cx"] - cat_x) <= 0.15 * w:
            idx = cat_lower.index(it["text"].lower())
            anchors.setdefault(idx, []).append(it["cy"])
    anchors = {idx: sum(v) / len(v) for idx, v in anchors.items()}
    if len(anchors) < 2:
        logger.warning("EasyOCR: zu wenige Zeilen-Anker")
        return None

    # Zeilen sind gleichmaessig verteilt -> linearer Fit: cy = a + b*idx
    idxs = np.array(list(anchors.keys()))
    cys = np.array(list(anchors.values()))
    b, a = np.polyfit(idxs, cys, 1)

    # --- Spaltenzentren aus Header-Labels ---
    # Nur Tokens oberhalb der ersten Datenzeile akzeptieren und auf der
    # korrekten Seite der Kategorie-Spalte (verhindert z.B. "scorecard"/
    # "scored by" als "Ringe score"-Treffer)
    top_anchor = min(anchors.values())
    header = {"s1": None, "s2": None, "tr": None, "rg": None}
    for it in items:
        if it["cy"] >= top_anchor:
            continue  # nicht im Header-Bereich
        t = it["text"].lower()
        if ("station" in t or "slation" in t) and it["cx"] < cat_x:
            if "2" in t or "+" in t:
                header["s2"] = it["cx"]
            elif "1" in t or header["s1"] is None:
                header["s1"] = it["cx"]
        elif it["cx"] < cat_x and ("+" in t or t.replace(" ", "") == "2+3"):
            # Header-Token "2 + 3" steht separat unter "station"
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
        "row_cy": {idx: anchors.get(idx, a + b * idx) for idx in range(len(KATEGORIEN))},
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
Die Zeilen-Kategorien (grau): X, 10, 9, 8, 7, 6, 5, 0, miss.

AUFGABE: Lies NUR die handschriftlichen Zahlen in der Spalte "{col_name}".
Ordne jeden Wert der Kategorie seiner Zeile zu.

Antworte NUR als JSON-Objekt mit allen 9 Kategorien als Schluessel:
{{"X":"ZAHL","10":null,"9":null,"8":null,"7":null,"6":null,"5":null,"0":null,"miss":null}}

"ZAHL" ist ein PLATZHALTER - ersetze ihn durch die Zahl die du wirklich im Bild siehst.
null = leere oder durchgestrichene/unkenntliche Zelle.
Die gedruckten Kategorie-Labels (X,10,9,...) sind KEINE Werte - nicht als Wert ausgeben!
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
    center=True misst nur die Zellmitte (innerste 40%x50%) - dort schlaegt
    Ueberlauf-Tinte aus Nachbarzellen nicht zu, echte Handschrift schon.
    """
    x0, y0, x1, y1 = box
    cell = gray[y0:y1, x0:x1]
    ch, cw = cell.shape
    if ch < 5 or cw < 5:
        return 0.0
    if center:
        inner = cell[int(ch * .25):int(ch * .75), int(cw * .30):int(cw * .70)]
    else:
        inner = cell[int(ch * .15):int(ch * .85), int(cw * .10):int(cw * .90)]
    if inner.size == 0:
        return 0.0
    return float((inner < 110).mean() * 100)


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
    gray = np.array(image.convert("L"))
    INK_THRESHOLD = 3.0  # % dunkle Pixel

    def cell_ink(row_idx, col_idx, center=False):
        cy = row_cy[row_idx]
        box = (
            max(0, int(centers[col_idx] - half_w)),
            max(0, int(cy - row_h * 0.55)),
            min(w_img, int(centers[col_idx] + half_w)),
            min(h_img, int(cy + row_h * 0.55)),
        )
        return _ink_ratio(gray, box, center=center)

    def cell_has_ink(row_idx, col_idx):
        # Zellmitte: Bleed aus Nachbarzellen faellt dort raus
        return cell_ink(row_idx, col_idx, center=True) >= 2.0

    # Enger Tabellen-Crop ohne Seitenspalte. Fuer Spalten-Calls endet der
    # Crop unterhalb der miss-Zeile - die total-Zeile wuerde sonst in die
    # letzten Eintraege "hineinbluten". Der total-Call nutzt das volle Grid.
    top = max(0, int(row_cy[0] - row_h * 1.5))
    bot_data = min(h_img, int(row_cy[8] + row_h * 1.2))   # bis miss
    bot_full = min(h_img, int(row_cy[9] + row_h * 1.2))   # inkl. total
    gx0 = max(0, int(centers[0] - half_w))
    gx1 = min(w_img, int(centers[4] + half_w))
    if gx1 - gx0 < 50 or bot_full - top < 50:
        logger.warning("Ungueltige Grid-Geometrie - Fallback auf LLM-Gesamtbild")
        return {"zeilen": []}

    def _grid_b64(bot):
        grid = image.crop((gx0, top, gx1, bot))
        gw, gh = grid.size
        if gw < 1200:
            grid = grid.resize((int(gw * 1.6), int(gh * 1.6)), Image.Resampling.LANCZOS)
        return encode_image(grid)

    data_b64 = _grid_b64(bot_data)
    full_b64 = _grid_b64(bot_full) if bot_full != bot_data else data_b64

    async def col_list(name):
        # Rueckgabe: Dict Kategorie -> Wert (oder {})
        return await query_ollama(
            data_b64, PROMPT_SPALTE.format(col_name=name), num_predict=512)

    # Spalten-Gate: nur Spalten fragen, in denen mindestens eine Zelle Tinte hat
    col_defs = [(0, "station 1"), (1, "station 2 + 3"),
                (3, "Treffer hits"), (4, "Ringe score")]
    jobs = {}
    for col_idx, name in col_defs:
        if any(cell_has_ink(i, col_idx) for i in range(9)):
            jobs[col_idx] = col_list(name)
    jobs["total"] = query_ollama(full_b64, PROMPT_TOTALZEILE, num_predict=256)

    results = await asyncio.gather(*jobs.values())
    lists = dict(zip(jobs.keys(), results))

    def _valid(v):
        return isinstance(v, str) and bool(re.fullmatch(r"[\d()xX\-]+", v))

    col_keys = {0: "station_1_links", 1: "station_2_rechts", 3: "treffer", 4: "ringe"}
    zeilen = []
    for idx, cat in enumerate(KATEGORIEN[:9]):
        row = {"kategorie": cat}
        for col_idx, key in col_keys.items():
            val = None
            d = lists.get(col_idx) or {}
            if cat in d:
                v = _coerce_str(d.get(cat))
                ink = cell_ink(idx, col_idx, center=True)
                if v and _valid(v) and ink >= INK_THRESHOLD:
                    val = v
            row[key] = val
        zeilen.append(row)

    td = lists.get("total") or {}
    totals = {
        "station_1_total": _coerce_str(td.get("station_1_total")),
        "station_2_total": _coerce_str(td.get("station_2_total")),
        "treffer_gesamt": _coerce_str(td.get("treffer_gesamt")),
        "ringe_gesamt": _coerce_str(td.get("ringe_gesamt")),
    }

    # Checksum: die Stationswerte (Anzahl Schuesse je Kategorie) muessen sich
    # zum gedruckten treffer_gesamt addieren. Ist die Summe zu gross, ist ein
    # gelesener Wert ueberzaehlig (z.B. durchgestrichener Eintrag oder
    # Halluzination) - die Zelle, deren Entfernen die Summe exakt macht,
    # wird genullt (nur wenn sie eindeutig bestimmbar ist).
    m = re.match(r"\d+", totals.get("treffer_gesamt") or "")
    if m:
        target = int(m.group(0))

        def _num(v):
            mm = re.match(r"\d+", v or "")
            return int(mm.group(0)) if mm else None

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
            elif fixes:
                logger.warning("Checksum: Summe %d > %d, mehrere moegliche "
                               "Fix-Zellen %s - keine automatische Korrektur",
                               total, target, fixes)
        elif total < target:
            logger.warning("Checksum: Summe %d < treffer_gesamt %d - "
                           "vermutlich fehlende Werte", total, target)

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
