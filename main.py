"""
FastAPI Service zur Auswertung von Schützenvereins-Scorecards (PP1, NPA etc.)
mit OpenAI GPT-4o Vision API.
"""

import base64
import json
import os
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Umgebungsvariable 'OPENAI_API_KEY' ist nicht gesetzt. "
        "Bitte setze sie vor dem Start des Servers."
    )

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_IMAGE_SIZE = (2048, 2048)  # GPT-4o Vision Limit ist hoch, aber wir skalieren runter

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(
    title="Scorecard OCR API",
    description="Liest Schützenvereins-Auswertekarten per GPT-4o Vision aus.",
    version="1.0.0",
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
    modell: str = Field(MODEL, description="Verwendetes AI-Modell")


# ---------------------------------------------------------------------------
# System-Prompt (detailliert, damit GPT-4o die Struktur versteht)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du bist ein OCR-Experte für Schützenvereins-Auswertekarten (Scorecards).
Analysiere das hochgeladene Bild sorgfältig und extrahiere alle handschriftlichen und gedruckten Informationen.

## Karten-Struktur

Oben rechts befindet sich eine Box mit:
- "Schütze / shooter:" gefolgt vom Namen und Verein (z.B. "Geuze Michael SG Lustenau")
- Disziplin: z.B. "PP1(AUT)" oder "NPA(AUT)"
- Stand-Details: z.B. "P2 25m Stand xy"
- Datum/Zeit: z.B. "21.09.2026 18:00 - 21:00"

Darunter befinden sich Checkboxen für verschiedene Disziplinen:
- PP1-Karte: PP1, PP1 Optic, PP1 Rifle, PP1-Uhr, PP3, PP4, SM, SM-O
- NPA-Karte: NPA, NPA Open, NPA Rifle.22, NPA Rifle .223, NPA Rifle PCC, Uhr
Es ist immer genau eine Checkbox mit einem Kreuz (X) markiert.

## Ergebnis-Tabelle

Die Tabelle hat folgende Spalten (von links nach rechts):
1. PP1: "station 1" (handschriftliche Treffer pro Ring-Kategorie)
2. PP1: "station 2 + 3" (handschriftliche Treffer pro Ring-Kategorie)
3. Ringe-Kategorien: X, 10, 9, 8, 7, 6, 5, 0, miss, total (gedruckte Labels)
4. "Treffer / hits" (oft leer, manchmal handschriftlich)
5. "Ringe / score" (oft leer, manchmal handschriftlich)

Bei NPA-Karten heißen die ersten beiden Spalten:
1. "linke Scheibe"
2. "rechte Scheibe"
Und die Ringe-Kategorien sind: X, 5, 4, 3, 2, 0, miss, total

Ganz unten in der Tabelle stehen die Summen:
- PP1: z.B. "12(10)" für Station 1, "18(20)" für Station 2+3, Gesamttreffer "30", Gesamt-Ringe "300"
- NPA: z.B. "12" für linke Scheibe, "12" für rechte Scheibe, Gesamttreffer "24", Gesamt-Ringe "120"

Unten auf der Karte steht nochmals "Schütze / shooter:" mit einer Unterschrift. Wenn du den Namen der unterschreibenden Person erkennen kannst, trage ihn als "auswerter" ein.

## Wichtige Hinweise
- Lies handschriftliche Zahlen so genau wie möglich. Wenn du unsicher bist, gib den wahrscheinlichsten Wert an.
- Wenn ein Feld leer ist, verwende null.
- Bei Streichungen oder Korrekturen (z.B. durchgestrichene Zahl + neue Zahl) nimm die korrigierte/gültige Zahl.
- Das Feld "auswerter" ist der Name des Auswerters (falls lesbar), NICHT der Schütze.
- Gib das Ergebnis AUSSCHLIESSLICH als JSON im folgenden Schema zurück. Keine Erklärungstexte, nur JSON.

## JSON-Schema

{
  "schuetze": "string oder null",
  "disziplin": "string oder null (z.B. 'PP1(AUT)')",
  "disziplin_checkbox": "string oder null (z.B. 'PP1' oder 'NPA')",
  "stand_details": "string oder null",
  "datum_zeit": "string oder null",
  "ergebnisse": {
    "zeilen": [
      {
        "kategorie": "string (z.B. 'X', '10', '9', '8', '7', '6', '5', '0', 'miss')",
        "station_1_links": "string oder null",
        "station_2_rechts": "string oder null",
        "treffer": "string oder null",
        "ringe": "string oder null"
      }
    ],
    "station_1_total": "string oder null",
    "station_2_total": "string oder null",
    "treffer_gesamt": "string oder null",
    "ringe_gesamt": "string oder null"
  },
  "auswerter": "string oder null"
}
"""

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def resize_and_encode_image(image: Image.Image) -> str:
    """
    Skaliert das Bild herunter (damit es schneller verarbeitet wird) und
    encodiert es als Base64-Data-URL.
    """
    # GPT-4o kann bis zu 2048x2048 gut verarbeiten, wir behalten Seitenverhältnis bei
    image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    # Konvertiere zu RGB falls nötig (JPEG unterstützt kein RGBA)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def parse_vision_response(raw_json: str) -> ScorecardAuswertung:
    """Parst die JSON-Antwort des Vision-Modells in das Pydantic-Modell."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Ungültiges JSON vom Vision-Modell: {exc}")

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
        modell=MODEL,
    )


# ---------------------------------------------------------------------------
# API-Endpunkte
# ---------------------------------------------------------------------------

@app.post(
    "/process-scorecard",
    response_model=ScorecardAuswertung,
    summary="Scorecard-Bild analysieren",
    description="Liest eine Schützenvereins-Auswertekarte per GPT-4o Vision API aus.",
)
async def process_scorecard(file: UploadFile = File(...)) -> ScorecardAuswertung:
    # --- Validierung ---
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Ungültiger Dateityp: {file.content_type}. Nur JPEG/PNG/WebP erlaubt.",
        )

    # --- Bild laden ---
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild konnte nicht geladen werden: {exc}")

    # --- Base64-Encoding ---
    data_url = resize_and_encode_image(image)

    # --- GPT-4o Vision API ---
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Lies diese Scorecard aus und gib das Ergebnis als JSON zurück.",
                        },
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=2048,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fehler bei OpenAI API: {exc}")

    raw_json = response.choices[0].message.content or "{}"
    return parse_vision_response(raw_json)


@app.get("/health")
async def health_check():
    return {"status": "ok", "modell": MODEL}


# ---------------------------------------------------------------------------
# Lokaler Start (nur für Entwicklung)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
