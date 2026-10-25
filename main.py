"""
FastAPI Service zur Auswertung von Schützenvereins-Scorecards (PP1, NPA etc.)
mit einem lokalen Vision-Modell via Ollama.
"""

import base64
import json
import os
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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")
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
    kategorie: str = Field(..., description="Ring-Kategorie, z.B. 'X', '10', '9', 'miss'")
    station_1_links: Optional[str] = Field(None, description="Handschriftlicher Wert linke/spalte (PP1=station1, NPA=linke Scheibe)")
    station_2_rechts: Optional[str] = Field(None, description="Handschriftlicher Wert rechte Spalte (PP1=station2+3, NPA=rechte Scheibe)")
    treffer: Optional[str] = Field(None, description="Wert in Treffer/Spalte falls eingetragen")
    ringe: Optional[str] = Field(None, description="Wert in Ringe/Spalte falls eingetragen")


class Ergebnisse(BaseModel):
    """Zusammengefasste Ergebnisse der Scorecard."""
    zeilen: List[ErgebnisZeile] = Field(default_factory=list)
    station_1_total: Optional[str] = Field(None, description="Summe Station 1 / linke Scheibe")
    station_2_total: Optional[str] = Field(None, description="Summe Station 2+3 / rechte Scheibe")
    treffer_gesamt: Optional[str] = Field(None, description="Gesamt-Treffer")
    ringe_gesamt: Optional[str] = Field(None, description="Gesamt-Ringe/Score")


def _coerce_str(v):
    """Hilfsfunktion: Wandelt Listen/Dicts in None um, ansonsten String."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
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
# Prompt fuer llama3.2-vision
# ---------------------------------------------------------------------------
PROMPT_KOPF = """Du siehst den oberen Teil einer Schuetzenvereins-Scorecard.
Lies die Textfelder und gib NUR JSON zurueck.

Felder:
- schuetze: Name unter "Schuetze / shooter"
- disziplin: z.B. "PP1(AUT)"
- disziplin_checkbox: angekreuzte Disziplin, z.B. "PP1"
- stand_details: z.B. "P2 25m Stand xy"
- datum_zeit: z.B. "21.09.2026 18:00 - 21:00"

Regeln: NUR JSON. Leer = null.
Schema: {"schuetze":null,"disziplin":null,"disziplin_checkbox":null,"stand_details":null,"datum_zeit":null}
"""

PROMPT_TABELLE = """Du siehst eine vergroesserte Tabelle aus einer Schuetzenvereins-Scorecard.
Lies die handschriftlichen Zahlen und gib NUR JSON zurueck.

DIE TABELLE:
- Links stehen die Kategorien: X, 10, 9, 8, 7, 6, 5, 0, miss (von oben nach unten)
- Daneben sind Spalten mit handschriftlichen Zahlen
- Du musst fuer JEDE Kategorie die Zahl in der ERSTEN Spalte (links) lesen

WICHTIG: Gib die Zeilen IN DER RICHTIGEN REIHENFOLGE zurueck.
Reihenfolge von oben nach unten: X, 10, 9, 8, 7, 6, 5, 0, miss

ZUSÄTZLICH unten in der Tabelle:
- total_zeile_station_1: Die Zahl in der "total" Zeile, erste Spalte (z.B. "30")
- station_1_total: z.B. "12(10)"
- station_2_total: z.B. "18(20)"
- treffer_gesamt: z.B. "30"
- ringe_gesamt: z.B. "300"

Regeln:
- NUR JSON, keine Erklaerungen.
- DOPPELTE Anfuehrungszeichen.
- Leer oder nicht erkennbar = null.
- Schreibe die Zahlen EXAKT so wie sie im Bild stehen.

Schema:
{"zeilen":[{"kategorie":"X","station_1_links":null,"station_2_rechts":null},{"kategorie":"10","station_1_links":null,"station_2_rechts":null},{"kategorie":"9","station_1_links":null,"station_2_rechts":null},{"kategorie":"8","station_1_links":null,"station_2_rechts":null},{"kategorie":"7","station_1_links":null,"station_2_rechts":null},{"kategorie":"6","station_1_links":null,"station_2_rechts":null},{"kategorie":"5","station_1_links":null,"station_2_rechts":null},{"kategorie":"0","station_1_links":null,"station_2_rechts":null},{"kategorie":"miss","station_1_links":null,"station_2_rechts":null}],"total_zeile_station_1":null,"station_1_total":null,"station_2_total":null,"treffer_gesamt":null,"ringe_gesamt":null}
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
    """Teilt das Bild in Kopf (Text) und Tabelle (Zahlen)."""
    w, h = image.size
    # Kopf: obere 45%
    kopf = image.crop((0, 0, w, int(h * 0.45)))
    # Tabelle: untere 55%, skaliert auf volle Hoehe fuer bessere Erkennung
    tabelle = image.crop((0, int(h * 0.35), w, h))
    # Tabelle auf volle Groesse skalieren
    tabelle = tabelle.resize((w, h), Image.Resampling.LANCZOS)
    return kopf, tabelle


def extract_table_with_easyocr(image: Image.Image) -> dict:
    """Liest die Tabelle mit EasyOCR aus (Zeilen-basiert)."""
    if not EASYOCR_AVAILABLE:
        return {}

    import numpy as np

    # Lazy-Initialisierung
    if not hasattr(extract_table_with_easyocr, "_reader"):
        extract_table_with_easyocr._reader = easyocr.Reader(["en", "de"], gpu=False)

    reader = extract_table_with_easyocr._reader
    w, h = image.size

    # Tabelle in 9 Zeilen aufteilen (X, 10, 9, 8, 7, 6, 5, 0, miss)
    kategorien = ["X", "10", "9", "8", "7", "6", "5", "0", "miss"]
    zeilen_hoehe = h // 9
    zeilen = []

    for i, kat in enumerate(kategorien):
        y1 = i * zeilen_hoehe
        y2 = (i + 1) * zeilen_hoehe if i < 8 else h
        row_img = image.crop((0, y1, w, y2))

        # OCR auf der Zeile ausfuehren
        results = reader.readtext(np.array(row_img))

        # Nur Zahlen extrahieren (hoechste Konfidenz zuerst)
        zahlen = []
        for (_, text, conf) in results:
            text = text.strip()
            if text.isdigit():
                zahlen.append((text, conf))

        # Sortieren nach Konfidenz und ersten 2 nehmen
        zahlen.sort(key=lambda x: x[1], reverse=True)
        station_1 = zahlen[0][0] if zahlen else None

        zeilen.append({
            "kategorie": kat,
            "station_1_links": station_1,
            "station_2_rechts": None,
        })

    # Summenzeile (letzter Streifen)
    summen_img = image.crop((0, h - zeilen_hoehe, w, h))
    summen_results = reader.readtext(np.array(summen_img))
    summen_zahlen = [text for (_, text, _) in summen_results if text.strip().isdigit() or "(" in text]

    # Heuristik fuer Summen
    station_1_total = None
    station_2_total = None
    treffer_gesamt = None
    ringe_gesamt = None

    for (_, text, _) in summen_results:
        text = text.strip()
        if "(" in text and ")" in text:
            if station_1_total is None:
                station_1_total = text
            elif station_2_total is None:
                station_2_total = text

    # Reine Zahlen fuer Treffer und Ringe
    pure_numbers = [text for (_, text, _) in summen_results if text.strip().isdigit()]
    if len(pure_numbers) >= 2:
        treffer_gesamt = pure_numbers[-2]
        ringe_gesamt = pure_numbers[-1]

    return {
        "zeilen": zeilen,
        "station_1_total": station_1_total,
        "station_2_total": station_2_total,
        "treffer_gesamt": treffer_gesamt,
        "ringe_gesamt": ringe_gesamt,
    }


def encode_image(image: Image.Image) -> str:
    """Bild als Base64-String fuer Ollama."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


import logging

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


import re


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


async def query_ollama(image_b64: str, prompt: str) -> dict:
    """Fragt Ollama mit einem Bild ab."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Lies dieses Bild.", "images": [image_b64]},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 2048},
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
    return parse_json_safe(raw)


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

    # --- Kopf mit llama3.2-vision ---
    kopf_data = await query_ollama(encode_image(kopf_img), PROMPT_KOPF)

    # --- Tabelle mit EasyOCR (falls verfuegbar) oder llama3.2-vision ---
    if EASYOCR_AVAILABLE:
        tabelle_data = extract_table_with_easyocr(tabelle_img)
    else:
        tabelle_data = await query_ollama(encode_image(tabelle_img), PROMPT_TABELLE)

    # --- Ergebnisse parsen und zusammenfuehren ---
    zeilen_raw = tabelle_data.get("zeilen") or []
    zeilen = [ErgebnisZeile(**z) for z in zeilen_raw if isinstance(z, dict)]

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
