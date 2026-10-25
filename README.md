# Scorecard OCR API

FastAPI-Service zur automatischen Auswertung von Schützenvereins-Scorecards (PP1, NPA, etc.) mittels OpenAI GPT-4o Vision.

## Schnellstart

1. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

2. **API-Key setzen**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY = "sk-..."

   # Windows CMD
   set OPENAI_API_KEY=sk-...
   ```
   Optional: Lege eine `.env`-Datei an oder nutze das Hosting-System deiner Wahl.

3. **Server starten**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Testen**
   Öffne die automatisch generierte API-Doku: [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpunkte

- `POST /process-scorecard` — Liefert ein strukturiertes JSON aus dem hochgeladenen Scorecard-Bild.
- `GET /health` — Health-Check.

## n8n-Workflow (HTTP Request Node)

1. **HTTP Request Node** anlegen.
2. **Method**: `POST`
3. **URL**: `http://<dein-server>:8000/process-scorecard`
4. **Authentication**: Keine (oder Basic-Auth/Header falls du sie hinzufügst).
5. **Body**: `Form-Data`
   - **Key**: `file` (Typ: `File`)
   - **Value**: Das Bild (JPEG/PNG).
6. **Response** ist das JSON-Objekt, das du direkt in weiteren Nodes weiterverarbeiten kannst.

Beispiel-Response:
```json
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
      ...
    ],
    "station_1_total": "12(10)",
    "station_2_total": "18(20)",
    "treffer_gesamt": "30",
    "ringe_gesamt": "300"
  },
  "auswerter": null,
  "modell": "gpt-4o"
}
```

## Konfiguration via Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `OPENAI_API_KEY` | — | **Pflicht.** Dein OpenAI API-Key. |
| `OPENAI_MODEL` | `gpt-4o` | Alternativ `gpt-4o-mini` für niedrigere Kosten. |
