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
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava")
MAX_IMAGE_SIZE = (1024, 1024)  # Lokale Modelle sind langsamer -> kleineres Bild

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


# ---------------------------------------------------------------------------
# Prompt (kurz und praezise fuer lokale kleine Modelle)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Lies die Schuetzenvereins-Scorecard aus dem Bild aus.

Extrahiere:
- schuetze: Name aus "Schuetze / shooter"
- disziplin: z.B. PP1(AUT) oder NPA(AUT)
- disziplin_checkbox: angekreuzte Disziplin, z.B. PP1 oder NPA
- stand_details: z.B. P2 25m Stand xy
- datum_zeit: z.B. 21.09.2026 18:00 - 21:00
- auswerter: Name des Auswerters (unten auf der Karte), NICHT der Schuetze

Ergebnisse Tabelle:
Lies fuer JEDE Zeile die handschriftlichen Zahlen aus!
PP1-Kategorien: X, 10, 9, 8, 7, 6, 5, 0, miss
NPA-Kategorien: X, 5, 4, 3, 2, 0, miss
Pro Kategorie ein JSON-Objekt in "zeilen".

Wichtig:
- Nur valides JSON zurueckgeben. Keine Erklaerungstexte.
- Alle Strings in DOPPELTEN Anfuehrungszeichen.
- Leere oder nicht erkennbare Felder = null.
- Trage die tatsaechlich gelesenen Zahlen ein, keine Platzhalter.

Beispiel fuer eine PP1-Karte:
{
  "schuetze": "Geuze Michael SG Lustenau",
  "disziplin": "PP1(AUT)",
  "disziplin_checkbox": "PP1",
  "stand_details": "P2 25m Stand xy",
  "datum_zeit": "21.09.2026 18:00 - 21:00",
  "ergebnisse": {
    "zeilen": [
      {"kategorie": "X", "station_1_links": "2", "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "10", "station_1_links": "13", "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "9", "station_1_links": "12", "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "8", "station_1_links": "2", "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "7", "station_1_links": "1", "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "6", "station_1_links": null, "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "5", "station_1_links": null, "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "0", "station_1_links": null, "station_2_rechts": null, "treffer": null, "ringe": null},
      {"kategorie": "miss", "station_1_links": null, "station_2_rechts": null, "treffer": null, "ringe": null}
    ],
    "station_1_total": "12(10)",
    "station_2_total": "18(20)",
    "treffer_gesamt": "30",
    "ringe_gesamt": "300"
  },
  "auswerter": null
}
"""

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def resize_and_encode_image(image: Image.Image) -> str:
    """
    Skaliert das Bild herunter (lokale Modelle sind langsamer, daher kleiner)
    und liefert einen reinen Base64-String (ohne Data-URL-Prefix),
    wie Ollama es erwartet.
    """
    image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
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


def parse_vision_response(raw_json: str) -> ScorecardAuswertung:
    """Parst die JSON-Antwort des Vision-Modells in das Pydantic-Modell."""
    cleaned = extract_json(raw_json)
    logger.info("Cleaned JSON length: %s", len(cleaned))

    # Versuche zuerst direkt zu parsen
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Versuche Reparatur
        logger.warning("Direct JSON parsing failed, attempting repair...")
        repaired = repair_json(cleaned)
        try:
            data = json.loads(repaired)
            logger.info("JSON repair successful")
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON. Raw content (first 2000 chars): %s", raw_json[:2000])
            raise HTTPException(status_code=500, detail=f"Ungueltiges JSON vom Vision-Modell: {exc}")

    # Stelle sicher, dass ergebnisse ein dict ist
    ergebnisse_raw = data.get("ergebnisse") or {}
    zeilen_raw = ergebnisse_raw.get("zeilen") or []

    zeilen = [ErgebnisZeile(**z) for z in zeilen_raw]

    ergebnisse = Ergebnisse(
        zeilen=zeilen,
        station_1_total=ergebnisse_raw.get("station_1_total"),
        station_2_total=ergebnisse_raw.get("station_2_total"),
        treffer_gesamt=ergebnisse_raw.get("treffer_gesamt"),
        ringe_gesamt=ergebnisse_raw.get("ringe_gesamt"),
    )

    return ScorecardAuswertung(
        schuetze=data.get("schuetze"),
        disziplin=data.get("disziplin"),
        disziplin_checkbox=data.get("disziplin_checkbox"),
        stand_details=data.get("stand_details"),
        datum_zeit=data.get("datum_zeit"),
        ergebnisse=ergebnisse,
        auswerter=data.get("auswerter"),
        modell=OLLAMA_MODEL,
    )


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

    # --- Base64-Encoding (nur raw base64 fuer Ollama) ---
    image_b64 = resize_and_encode_image(image)

    # --- Ollama API (/api/chat) ---
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Lies diese Scorecard aus und gib das Ergebnis als JSON zurueck.",
                "images": [image_b64],
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,   # moeglichst deterministisch
            "num_predict": 2048,  # Max. Tokens
        },
    }

    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Verbindung zu Ollama fehlgeschlagen ({OLLAMA_HOST}). Ist Ollama gestartet? Details: {exc}",
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fehler bei Ollama-Anfrage: {exc}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama antwortete mit Status {resp.status_code}: {resp.text}",
        )

    try:
        ollama_data = resp.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Ungueltige JSON-Antwort von Ollama: {exc}")

    raw_json = ollama_data.get("message", {}).get("content", "{}")
    if not raw_json:
        raw_json = "{}"

    return parse_vision_response(raw_json)


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
