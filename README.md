# Scorecard OCR API

FastAPI-Service zur automatischen Auswertung von Schützenvereins-Scorecards (PP1, NPA, etc.) mittels **lokalem** Ollama-Vision-Modell. Keine externe API nötig, keine Kosten pro Anfrage.

## Voraussetzungen

1. **Ollama installieren**
   Lade Ollama für Windows/Mac/Linux herunter: https://ollama.com/download

2. **Vision-Modell herunterladen**
   Öffne ein Terminal und ziehe ein Modell:

   ```bash
   # Empfohlen (beste Erkennung fuer handschriftliche Scorecards)
   ollama pull llama3.2-vision

   # Alternative: schneller, aber weniger genau
   ollama pull llava

   # Alternative: sehr schnell und klein, aber ungenau
   ollama pull moondream
   ```

3. **Ollama starten**

   ```bash
   # WICHTIG: ollama serve nimmt KEIN Modell-Argument!
   ollama serve
   ```

   Lass dieses Terminal offen. Das Modell wird beim ersten API-Call automatisch geladen.

## Schnellstart

1. **Abhängigkeiten installieren**

   ```bash
   pip install -r requirements.txt
   ```

2. **Server starten**

   Führe im Projektordner aus (Git Bash / PowerShell / CMD):

   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Falls `python` nicht gefunden wird, probiere `py` statt `python`.

3. **Testen**
   - API-Doku: http://localhost:8000/docs
   - Health-Check (zeigt Ollama-Status & Modell): http://localhost:8000/health

## Anderes Modell nutzen

Der Standard in `main.py` ist `llama3.2-vision`. Falls du ein anderes Modell verwenden möchtest, setze vor dem Server-Start die Umgebungsvariable:

**Windows PowerShell:**

```powershell
$env:OLLAMA_MODEL = "llava"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows CMD:**

```cmd
set OLLAMA_MODEL=llava
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Linux/macOS (Bash):**

```bash
export OLLAMA_MODEL=llava
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpunkte

- `POST /process-scorecard` — Liefert ein strukturiertes JSON aus dem hochgeladenen Scorecard-Bild.
- `GET /health` — Prüft Erreichbarkeit von Ollama und zeigt konfiguriertes Modell an.

## n8n-Workflow (HTTP Request Node)

1. **HTTP Request Node** anlegen.
2. **Method**: `POST`
3. **URL**: `http://<dein-server>:8000/process-scorecard`
4. **Authentication**: Keine.
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
  "modell": "llama3.2-vision"
}
```

## Konfiguration via Umgebungsvariablen

| Variable       | Standard                 | Beschreibung                     |
| -------------- | ------------------------ | -------------------------------- |
| `OLLAMA_HOST`  | `http://localhost:11434` | URL der Ollama-Instanz.          |
| `OLLAMA_MODEL` | `llama3.2-vision`        | Name des lokalen Vision-Modells. |
