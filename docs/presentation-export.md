# Notebook-Präsentationen exportieren

Der Exporter erzeugt aus **bereits ausgeführten und gespeicherten** Jupyter Notebooks
präsentationsfähige HTML-Snapshots. Er führt keine Zelle aus, trainiert keine Modelle und
verändert weder das Quell-Notebook noch dessen Outputs. Die Veröffentlichung besteht aus
einem kleinen HTML-Dokument pro Notebook sowie gemeinsam genutzten, lokalen Assets unter
`reports/presentation/`.

## Installation

Installiere die Exportabhängigkeiten mit dem vorhandenen uv-Workflow:

```bash
uv sync --extra presentation
```

Python 3.11 oder neuer wird über das Projekt verwaltet. Für das Versionieren großer
Präsentations-Assets muss außerdem Git LFS installiert und einmalig aktiviert sein:

```bash
git lfs install
git lfs pull
```

## Export

Führe das gewünschte Notebook auf dem leistungsfähigen Rechner vollständig aus, speichere
es und kontrolliere seine Ergebnisse. Exportiere erst danach. Der Exporter verwendet
ausschließlich den gespeicherten Zustand.

Alle entdeckten Notebooks exportieren:

```bash
uv run python scripts/export_notebooks.py --all
```

Ein einzelnes Notebook exportieren:

```bash
uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb
```

Die Ausgabe liegt standardmäßig unter `reports/presentation/`. Ein anderes Ziel ist mit
`--output-dir PFAD` möglich. Mit `--open` wird bei genau einem Snapshot dessen
Notebook-HTML geöffnet; entscheidend ist dabei die Anzahl tatsächlich gerenderter
Snapshots. Wurden mehrere Snapshots gerendert, öffnet der Standardbrowser
stattdessen die Indexseite. Werden alle ausgewählten Notebooks übersprungen oder schlägt
die Publikation fehl, wird kein Browser geöffnet.

Die Discovery durchsucht `notebooks/` rekursiv nach `.ipynb`-Dateien, sortiert sie
deterministisch und ignoriert die generierten Jupytext-`.py`-Spiegel. Gleichnamige
Notebooks in verschiedenen Unterordnern behalten ihre relative Verzeichnisstruktur unter
`reports/presentation/notebooks/`.

Nach einem Sammelexport enthält der Ordner typischerweise:

```text
reports/presentation/
├── index.html
├── manifest.json
├── notebooks/
│   └── 02_U_Phase.html
└── assets/
    ├── notebooks/
    ├── ui/
    └── vendor/
```

Vorhandene Zieldateien werden atomar ersetzt. Alte Exporte nicht mehr vorhandener
Notebooks werden nicht automatisch gelöscht; die Indexseite kennzeichnet sie als
verwaist.

## Validation

Jedes Notebook wird vor dem Rendern mit `nbformat` gelesen und analysiert. Im normalen
Modus werden auch Snapshots mit Befunden exportiert. Die Abschlusszeile pro Notebook zeigt
Status, Dateigröße sowie die Anzahl von Info-, Warning- und Error-Befunden. Ein Befund ist
damit nicht automatisch ein fehlgeschlagener Export.

Wichtige Codes und ihre Bedeutung:

| Code | Bedeutung | Strict-Blocker |
|---|---|---|
| `UNEXECUTED_CELL` | Nicht leere Codezelle wurde nie ausgeführt | ja |
| `EXECUTED_NO_OUTPUT` | Ausgeführte Setup-/Schreibzelle ohne sichtbaren Output | nein, nur Info |
| `ERROR_OUTPUT` | Gespeicherter Traceback oder Error-Output | ja |
| `DUPLICATE_EXECUTION_COUNT` | Ausführungsnummer kommt mehrfach vor | ja |
| `NON_MONOTONIC_EXECUTION` | Ausführungsnummern fallen in Notebook-Reihenfolge | ja |
| `MISSING_LOCAL_ASSET` | Referenzierte lokale Datei fehlt | ja |
| `UNSAFE_LOCAL_ASSET` | Lokale Referenz verlässt das Repository | ja |
| `EXTERNAL_RUNTIME_RESOURCE` | HTML benötigt eine externe oder nicht lokale URI | ja |
| `EXTERNAL_MAP_TILES` | Plotly-Karte benötigt externe Kartenkacheln | ja |
| `WIDGET_STATE_MISSING` | Widget besitzt keinen passenden eingebetteten Zustand | ja |
| `UNSUPPORTED_MIME` | MIME-Bundle hat keine unterstützte Darstellung/Fallback | ja |
| `LARGE_OUTPUT` | Einzelner gespeicherter Output ist größer als 5 MiB | nein |
| `VERY_LARGE_NOTEBOOK_OUTPUT` | Gespeicherte Outputs überschreiten zusammen 100 MiB | nein |
| `PLACEHOLDER_NOTEBOOK` | Notebook wurde als kleiner TODO-Platzhalter erkannt | nein |
| `WIP_NOTEBOOK` | Notebook wurde als Work in Progress erkannt | ja |

Strict prüft die als Blocker markierten Befunde und rendert das betroffene Notebook nicht:

```bash
uv run python scripts/export_notebooks.py --all --strict
```

Bei mehreren Notebooks läuft die Analyse nach einem Einzelfehler weiter. Exit-Code `0`
bedeutet, dass alle angeforderten Exporte erfolgreich waren; `1` steht für einen
Strict-Blocker, Render-/Publikationsfehler oder einen nicht frischen Check; `2` für einen
Aufruf-, Pfad- oder Repositoryfehler. Große Outputs und legitime outputlose Setup-Zellen
bleiben auch in Strict nicht blockierend.

## nbstripout

Die sichere Reihenfolge vor einem Commit ist:

1. Notebook vollständig ausführen und speichern.
2. Ergebnisse und gespeicherte Outputs im Notebook kontrollieren.
3. HTML exportieren und die Terminalbefunde prüfen.
4. `reports/presentation/` sowie beabsichtigte Notebook-Änderungen gezielt stagen.
5. Commit ausführen; verändert `nbstripout` das Notebook, genau diese Dateien erneut
   stagen und den Commit wiederholen.

Beispiel:

```bash
uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb
git add notebooks/02_U_Phase.ipynb reports/presentation/
git commit -m "docs: publish U-phase presentation"
```

`nbstripout` darf erst **nach** dem Export auf den zu versionierenden Notebook-Zustand wirken:
Die HTML-Datei bewahrt die vorher gespeicherten Outputs, während das versionierte Notebook
klein bleibt. Das Manifest speichert sowohl `snapshot_sha256` für den vollständigen
Snapshot als auch `source_sha256`. `source_sha256` lässt Outputs, Ausführungsnummern und
Widget-Zustand bewusst aus; deshalb macht das reine Entfernen von Outputs durch
`nbstripout` einen Export nicht sofort fachlich veraltet. Quelltext- oder
Markdown-Änderungen tun dies dagegen.

## Offline-Nutzung

Kopiere oder synchronisiere immer den **vollständigen Ordner**
`reports/presentation/`, nicht nur eine einzelne HTML-Datei. Öffne anschließend
`index.html` oder eine Datei unter `notebooks/` direkt im Browser. Auf dem Zielrechner
sind weder Python noch Jupyter, Projektdaten oder trainierte Modelle erforderlich.

CSS, JavaScript, MathJax, Bilder und Plotly-Payloads liegen relativ zur HTML-Datei in
`assets/`. Werden Unterordner ausgelassen oder umbenannt, fehlen Grafiken oder Bedienung.
Ein lokaler HTTP-Server ist für den vorgesehenen `file://`-Betrieb nicht erforderlich.

## PDF

Nutze in der Präsentation die Schaltfläche „Drucken“ und anschließend die PDF-Funktion
des Browsers. Die Schaltfläche lädt Plotly-Grafiken vor dem Druck; auch der
Browser-Druck-Event öffnet Code- und Outputbereiche. Navigation und Bedienelemente werden
in der Druckansicht ausgeblendet, Tabellenbegrenzungen aufgehoben und Grafiken auf die
Seitenbreite begrenzt. Kontrolliere Seitenumbrüche und Papierformat in der Druckvorschau.

## Plotly

Plotly-Daten und die einmalig mitgelieferte Plotly-Laufzeit werden als lokale,
inhaltsadressierte Dateien exportiert. Beim Scrollen lädt JavaScript die Laufzeit und den
jeweiligen Payload erst kurz bevor die Grafik sichtbar wird. Diese Lazy-Loading-Struktur
verwendet relative lokale `<script>`-Dateien und funktioniert daher auch unter `file://`,
solange der vollständige Präsentationsordner erhalten bleibt. Identische Payloads werden
nur einmal gespeichert.

Eine Plotly-Karte mit OpenStreetMap-, Mapbox- oder vergleichbaren Kartenstilen braucht
trotzdem externe Kartenkacheln. Der Exporter meldet dafür `EXTERNAL_MAP_TILES`; Strict
blockiert den Export. Im normalen Modus kann die HTML-Datei erzeugt werden, aber der
Kartenhintergrund bleibt ohne Netzverbindung unvollständig. Ergänze für eine garantiert
offlinefähige Präsentation einen statischen PNG-/SVG-Kartenexport.

## Widgets und Karten

Interaktive Jupyter-Widgets werden derzeit nicht unterstützt, auch wenn der passende
Widget-Zustand im Notebook eingebettet ist: Widget-Manager-Laufzeit und Widget-Zustand
werden nicht veröffentlicht und stehen daher nicht als ausführbare Präsentations-Assets
zur Verfügung.
Fehlt der Zustand, meldet die Validierung zusätzlich `WIDGET_STATE_MISSING`. Für jedes
wichtige Widget-Ergebnis ist daher ein statischer Fallback als PNG, SVG, HTML-Tabelle oder
Textoutput erforderlich.

Aktive HTML-Ausgaben, iframes und damit typische Folium-Ausgaben werden aus
Sicherheitsgründen in einem restriktiven Sandbox-iframe dargestellt. Skripte sind in
dieser Sandbox deaktiviert; externe Tiles funktionieren offline ebenfalls nicht. Für
Folium-Karten ist deshalb ein statischer PNG- oder SVG-Fallback erforderlich. Passive
HTML-Tabellen und Textoutputs werden dagegen direkt, scrollbar und druckbar gerendert.

## Git LFS

Die Repository-Regeln verwalten große Dateien unter
`reports/presentation/assets/notebooks/**` und
`reports/presentation/assets/vendor/**` mit Git LFS. HTML, `index.html`, `manifest.json`
und die kleinen UI-Assets bleiben normale Git-Dateien. Das 5-MiB-Limit des
Pre-Commit-Hooks wird nicht umgangen.

Prüfe vor dem Commit, ob LFS aktiv ist und die großen Assets als LFS-Objekte erkannt
werden:

```bash
git lfs install
git add reports/presentation/
git lfs ls-files
git diff --cached --stat
```

Nach einem Clone oder Pull muss `git lfs pull` die echten Asset-Inhalte bereitstellen;
sonst liegen nur kleine Pointer-Dateien vor. Für eine reine USB-/Ordnerkopie kopierst du
die materialisierten Dateien aus dem Working Tree, nicht Git-Objekte oder LFS-Pointer.

## Freshness

Prüfe vorhandene Snapshots ohne Rendern oder Notebook-Ausführung:

```bash
uv run python scripts/export_notebooks.py --check
```

Der Check vergleicht `source_sha256` aus `manifest.json` mit den aktuellen
Notebook-Inhalten und prüft, ob die exportierte HTML-Datei existiert. Mögliche Zustände
sind `fresh`, `stale`, `missing-export`, `orphaned` und `invalid-source`. Nur wenn alle
Einträge `fresh` sind, endet der Check mit Exit-Code `0`. Ein Dirty-Working-Tree-Hinweis
im HTML macht zusätzlich sichtbar, dass der Snapshot uncommittete Änderungen enthalten
kann; der angezeigte Commit-Hash beschreibt dann nur den letzten Git-Stand.

Fehlt das Manifest, endet der Check kontrolliert mit Exit-Code `1`. Enthält ein valides
Manifest keine Einträge, endet er ebenfalls mit Exit-Code `1`; ein leerer Vergleich gilt
nicht als frischer Export. Ein syntaktisch oder strukturell ungültiges Manifest bleibt
davon getrennt und wird als Manifestfehler gemeldet.

Das Manifest und die eingebetteten Metadaten dokumentieren außerdem Exportzeitpunkt,
Quellpfad, Git-Commit, Branch, Zellzahlen, Befunde und veröffentlichte Assets. Der Exporter
löscht veraltete oder verwaiste Dateien niemals automatisch.

## Platzhalter und WIP

Kleine Notebooks mit expliziten TODO-/Platzhaltermarkern werden transparent als
`placeholder` klassifiziert und bei `--all` standardmäßig übersprungen. Sie erscheinen
nicht als fertige Phase. Falls ein Platzhalter bewusst als Vorschau benötigt wird:

```bash
uv run python scripts/export_notebooks.py --all --include-placeholders
```

Ein WIP-Notebook kann im normalen Modus mit klarer Kennzeichnung exportiert werden, ist
aber ein Strict-Blocker. Für laufende Trainings- oder unvollständige Modellphasen sollte
kein veröffentlichter Snapshot als abgeschlossen dargestellt werden.

## Zukünftige Notebooks

Neue `.ipynb`-Dateien unter `notebooks/` werden bei `--all` ohne Codeänderung entdeckt.
Verwende einen stabilen, aussagekräftigen Dateinamen und eine Markdown-Hauptüberschrift
als Titel. Speichere lokale Bilder innerhalb des Repositories und referenziere sie relativ
zum Notebook. Jupytext-`.py`-Spiegel sind niemals Exportquellen und dürfen nicht direkt
bearbeitet werden.

## Fehlerbehebung

- **`MISSING_LOCAL_ASSET`:** Pfad und Groß-/Kleinschreibung prüfen, Datei innerhalb des
  Repositories ablegen, relativ referenzieren und erneut exportieren.
- **`UNSAFE_LOCAL_ASSET`:** Keine absoluten Pfade oder `..`-Fluchten verwenden. Benötigte
  Datei in einen passenden Projektordner kopieren.
- **Externe Ressource:** CDN-, iframe- oder Tile-Abhängigkeit durch lokales beziehungsweise
  statisches Asset ersetzen. Der Normalmodus exportiert mit einem sichtbaren Befund;
  Strict blockiert.
- **Plotly bleibt leer:** Prüfen, ob `assets/vendor/` und `assets/notebooks/` mitkopiert und
  nach einem Git-Clone durch `git lfs pull` materialisiert wurden.
- **HTML ist veraltet:** `--check` ausführen, Notebook neu speichern und exportieren.
- **Ungültiges Notebook:** Datei in Jupyter öffnen, als gültiges Notebook speichern und
  den Export erneut starten. Andere angeforderte Notebooks werden trotzdem analysiert.
- **Browser öffnet nicht automatisch:** `--open` weglassen und
  `reports/presentation/index.html` manuell öffnen.

## GitHub Pages

Der lokale Export funktioniert vollständig unabhängig von GitHub Pages. Veröffentlicht
werden ausschließlich bereits committete Dateien aus `reports/presentation/`; der
Pages-Workflow führt keine Notebooks und kein Modelltraining aus.

In den Repository-Einstellungen unter **Settings → Pages** muss die Source auf
**GitHub Actions** stehen. Wähle **keine der vorgeschlagenen Workflow-Vorlagen** („Static
HTML“ oder „GitHub Pages Jekyll“), weil das Repository bereits den gezielten Workflow
`.github/workflows/pages.yml` enthält. Ein Branch-Ordner wie
`/reports/presentation` ist bei der Actions-Quelle nicht auszuwählen.

Nach einem Merge auf `main` startet der Workflow nur bei Änderungen an
`reports/presentation/**` oder an der Workflow-Datei. Alternativ lässt er sich über den
Actions-Tab manuell per `workflow_dispatch` starten. Der Checkout lädt LFS-Inhalte,
validiert `index.html` und `manifest.json` und veröffentlicht anschließend genau den
Präsentationsordner. Die einmalige Pages-Umgebung und die öffentliche URL werden nach dem
ersten erfolgreichen Lauf in GitHub angezeigt.
