# Sprechzettel zur Projektpräsentation

**Unfallatlas Deutschland: Schweregrad-Klassifikation von Verkehrsunfällen (QUA³CK)**

Vollständige deutsche Präsentationsnotizen. Jede Folie hat einen Abschnitt mit dem
Kernsatz, den Sprechnotizen und den Begriffen, die du erklären können musst.

Am Ende: ein Glossar aller Fachbegriffe und ein Katalog wahrscheinlicher
Prüfungsfragen mit Antworten.

---

## Aufbau der Foliensätze

Die Präsentation liegt als zwei Gamma-Decks vor, weil eine Generierung auf
20 Karten begrenzt ist. Die Reihenfolge ist durchgehend, du kannst Teil 2 im
Gamma-Editor einfach hinter Teil 1 einfügen.

| Deck | Folien | Inhalt |
|---|---|---|
| Teil 1: Frage und Daten | 1 bis 18 | Titel bis A³, 19 Konfigurationen |
| Teil 2: Modell, Vergleich und Transfer | 19 bis 37 | A³ Oversampling bis Dank |

Beide Decks sind im Format 16:9 und im Theme "Petrol" angelegt, passend zum
Stahlblau der App und des Handouts. Export als PDF oder PowerPoint ist damit
ohne Zuschnitt möglich.

Die Sprechernotizen stehen bewusst **nur in diesem Dokument** und nicht auf den
Folien. Drucke es aus oder lege es auf den zweiten Bildschirm.

---

## Wie du dieses Dokument benutzt

- **Kernsatz**: Der eine Satz, den das Publikum von dieser Folie mitnehmen soll.
  Wenn du nur einen Satz sagst, dann diesen.
- **Sprechnotizen**: Der ausformulierte Gedankengang. Nicht ablesen, sondern
  als Gedächtnisstütze nutzen.
- **Begriffe erklärbar machen**: Fachbegriffe, nach denen auf dieser Folie
  gefragt werden kann, mit Kurzdefinition.
- **Falls nachgefragt wird**: Vorbereitete Antworten auf die wahrscheinlichste
  Rückfrage.

**Faustregel für die Zeit:** 36 Folien in 30 bis 40 Minuten sind etwa
50 bis 65 Sekunden pro Folie. Die drei Folien, für die du dir bewusst mehr Zeit
nehmen solltest, sind markiert mit **[SCHLÜSSELFOLIE]**.

---

## Zeitplan

| Block | Folien | Zielzeit |
|---|---|---|
| Einstieg (Titel, Agenda, QUA³CK, Motivation) | 1-4 | 4 Min |
| Q-Phase | 5-8 | 5 Min |
| U-Phase | 9-16 | 9 Min |
| A³-Phase | 17-24 | 9 Min |
| C-Phase | 25-31 | 8 Min |
| K-Phase | 32-34 | 4 Min |
| Fazit, Quellen, Dank | 35-37 | 2 Min |

Wenn du in Zeitnot kommst: Folien 11 (Datenqualität), 24 (Threshold-Negativbefund)
und 31 (Inference Contract) lassen sich am ehesten kürzen. **Niemals kürzen:**
Folie 20 (arithmetische Decke), Folie 21 (Reframing), Folie 27 (Champion bleibt).

---

## Sprungpunkte für die Live-Demo

Wenn du kurz aus der Präsentation herausspringen willst, sind das die
sinnvollsten Momente:

| Nach Folie | Wohin | Was du zeigst | Dauer |
|---|---|---|---|
| 14 (EDA-Befunde) | Live-Report, U-Phase-Notebook | Cramérs-V-Heatmap, interaktiv | 30 Sek |
| 22 (Kandidaten) | Streamlit-App, Model Comparison | Pareto-Front live | 45 Sek |
| 27 (Champion bleibt) | Streamlit-App, Model Comparison | Der zweisprachige Erklärtext im Aufklapper | 30 Sek |
| 32 (App-Übersicht) | Streamlit-App, Risk Predictor | Ein Szenario klicken, Vorhersage zeigen | 60 Sek |
| 33 (Risikokarte) | Streamlit-App, Overview | Risikobänder ein- und ausschalten | 45 Sek |

**Wichtig:** Vor der Präsentation alle drei Tabs offen und eingeloggt haben.
Die Streamlit-App braucht beim Kaltstart bis zu 30 Sekunden, um aufzuwachen.
Einmal vorher aufrufen, damit sie warm ist.

---

# Teil 1: Frage und Daten

## Folie 1: Titel

**Kernsatz:** Ich habe die Schwere von Verkehrsunfällen in Deutschland aus
öffentlich verfügbaren Daten vorhergesagt und dabei ein methodisch sauberes
Vorgehen wichtiger genommen als eine hohe Zahl.

**Sprechnotizen:**
Kurz halten. Name, Titel, Datengrundlage in einem Satz. Nicht die ganze Folie
vorlesen. Direkt ankündigen, dass es drei Live-Artefakte gibt (Repository,
Live-Report, App) und dass du an passenden Stellen kurz hineinspringst.

**Begriffe erklärbar machen:**
- **QUA³CK**: Prozessmodell für ML-Projekte. Question, Understanding,
  Algorithm/Adapt/Adjust, Conclude & Compare, Knowledge Transfer. Wird auf
  Folie 3 ausführlich erklärt.
- **Unfallatlas**: Offener Datensatz der Statistischen Ämter des Bundes und der
  Länder. Jeder polizeilich erfasste Unfall mit Personenschaden seit 2016,
  ein Datensatz pro Unfall.

---

## Folie 2: Agenda

**Kernsatz:** Die Präsentation folgt exakt dem QUA³CK-Prozess, weil das Projekt
so entstanden ist.

**Sprechnotizen:**
Kurz durchgehen. Ein Hinweis, der Neugier erzeugt: "Die A³-Phase ist die
interessanteste, weil dort das ursprüngliche Ziel gescheitert ist und ich es
begründet ändern musste." Damit ist der Spannungsbogen gesetzt.

---

## Folie 3: Das QUA³CK-Prozessmodell

**Kernsatz:** QUA³CK zwingt dazu, die Erfolgskriterien festzulegen, bevor man
die Daten kennt, und macht damit ein Scheitern überhaupt erst erkennbar.

**Sprechnotizen:**
Jeden Buchstaben einmal aussprechen. Wichtig ist der letzte Satz auf der Folie:
Das Modell ist iterativ. Spätere Phasen dürfen frühere Entscheidungen
widerlegen. Betone, dass genau das hier passiert ist, und dass das kein Fehler
im Prozess ist, sondern der Prozess selbst.

**Begriffe erklärbar machen:**
- **A³**: Steht für drei Tätigkeiten in einer Phase. *Algorithm*: Welche
  Modellfamilie? *Adapt*: Wie passe ich die Daten an, vor allem an das
  Ungleichgewicht der Klassen? *Adjust*: Wie stelle ich die Hyperparameter ein?
- **Iterativ**: Man geht nicht einmal linear durch, sondern springt zurück,
  wenn eine spätere Phase eine frühere Annahme widerlegt.

**Falls nachgefragt wird, "Warum QUA³CK und nicht CRISP-DM?"**
CRISP-DM ist der Industriestandard und sehr ähnlich aufgebaut. QUA³CK war
die Vorgabe des Kurses. Inhaltlich unterscheiden sie sich vor allem darin, dass
QUA³CK die drei A-Tätigkeiten explizit trennt, während CRISP-DM sie unter
"Modeling" zusammenfasst. Der wesentliche gemeinsame Kern ist, dass die
Geschäftsfrage vor der Modellierung steht.

---

## Folie 4: Motivation

**Kernsatz:** Kommunen entscheiden über Verkehrssicherheit heute meist auf Basis
von Unfallzahlen, nicht auf Basis von Bedingungsprofilen, und genau diese Lücke
adressiert das Projekt.

**Sprechnotizen:**
Mit der Zahl einsteigen: rund 270.000 Unfälle mit Personenschaden pro Jahr.
Dann die Verteilung nennen, weil sie die ganze Präsentation prägt: 1, 18, 81
Prozent. Dann der Nutzen: Ein Modell, das aus Bedingungen auf Schwere schließt,
kann priorisieren statt nur zählen.

Wichtig ist der letzte Block, die bewussten Ausschlüsse. Sag deutlich, dass
individuelle Fahrer-Risikoscores und Versicherungstarifierung ausgeschlossen
wurden, und zwar bewusst und vorab. Das zeigt ethische Reflexion und kommt
in Bewertungen gut an.

**Begriffe erklärbar machen:**
- **Vision Zero**: Ursprünglich schwedisches, inzwischen EU-weites
  verkehrspolitisches Ziel, bis 2050 keine Verkehrstoten und
  Schwerverletzten mehr zu haben. Der politische Rahmen, in dem solche
  Analysen konsumiert werden.
- **BASt**: Bundesanstalt für Straßenwesen. Gibt jährlich den Bericht
  "Unfallentwicklung auf deutschen Straßen" heraus.
- **DVR**: Deutscher Verkehrssicherheitsrat.

---

## Folie 5: Die Forschungsfrage

**Kernsatz:** Die Frage war von Anfang an gestuft formuliert, mit einem
vorab definierten Ausweichpfad, deshalb ist die spätere Zieländerung Teil
des Designs und keine nachträgliche Rechtfertigung.

**Sprechnotizen:**
Die Frage einmal langsam vorlesen, sie ist lang. Dann den entscheidenden Punkt
machen: Sie startet mit dem Drei-Klassen-Ziel und bleibt explizit offen für eine
"evidenzgetriebene operative Umformulierung".

Sag diesen Satz bewusst: **"Das ist kein nachträglicher Rettungsanker. Das stand
in Phase Q im Notebook, bevor das erste Modell trainiert wurde."** Wenn du in der
A³-Phase ankommst, verweise zurück auf diese Folie.

Der zweite Block ist auch wichtig: Vorhergesagt wird aus Bedingungen, die zum
Zeitpunkt der Protokollerstellung vorliegen. Das grenzt sauber ab, was das Modell
kann und was nicht.

**Falls nachgefragt wird, "Sagt das Modell vorher, ob ein Unfall passiert?"**
Nein. Das Modell setzt voraus, dass ein Unfall passiert ist, und schätzt, wie
schwer er ausgeht. Unfallvorhersage wäre eine völlig andere Aufgabe und
bräuchte Expositionsdaten, also Verkehrsmengen, die dieser Datensatz nicht hat.

---

## Folie 6: Hypothesen

**Kernsatz:** Drei überprüfbare Hypothesen zu Zeit und Wetter, zu Ortskontext und
zur gestuften Machbarkeit.

**Sprechnotizen:**
H1 und H2 sind inhaltliche Hypothesen, H3 ist eine methodische. Bei H2 unbedingt
die kursiv gesetzte Einschränkung mitsprechen: Geschwindigkeit,
Infrastrukturqualität und Rettungszeit sind plausible Mechanismen, werden aber
nicht gemessen. Es wird kein Kausalanspruch erhoben.

Das ist der Punkt, an dem du zeigst, dass du Korrelation und Kausalität
unterscheidest, ohne dass jemand fragen muss.

Ergebnis vorwegnehmen, in einem Halbsatz: H1 wurde nur schwach bestätigt, die
Wetter-Assoziationen sind sehr klein. H2 wurde in der schwachen Form bestätigt,
Ortsmerkmale tragen messbar bei. H3 ist der eigentliche Befund der Arbeit.

**Begriffe erklärbar machen:**
- **Hypothese**: Eine vorab formulierte, überprüfbare Aussage. Wichtig ist, dass
  sie falsifizierbar ist, also durch die Daten widerlegt werden kann.
- **Baseline**: Ein bewusst einfaches Vergleichsmodell. Wenn das komplexe Modell
  die Baseline nicht schlägt, hat der Aufwand keinen Nutzen.

---

## Folie 7: Zielmetriken und Akzeptanzkriterien

**Kernsatz:** Macro-F1 als Primärmetrik plus ein zweites Gate auf dem Recall der
seltensten Klasse, damit ein Modell nicht dadurch gut aussieht, dass es die
kritischste Klasse ignoriert.

**Sprechnotizen:**
Hier musst du sicher sein. Erkläre Macro-F1 von unten auf:

1. **Precision** einer Klasse: Von allen Fällen, die das Modell als diese Klasse
   vorhergesagt hat, wie viele waren es wirklich?
2. **Recall** einer Klasse: Von allen Fällen, die wirklich diese Klasse sind, wie
   viele hat das Modell gefunden?
3. **F1**: Das harmonische Mittel aus beiden. Harmonisch, weil es bestraft, wenn
   einer der beiden Werte sehr klein ist.
4. **Macro-F1**: Der einfache Durchschnitt der F1-Werte über alle Klassen. Jede
   Klasse zählt gleich viel, egal wie selten sie ist.

Dann das Schlüsselbeispiel: Ein Modell, das immer "Leichtverletzt" sagt, hat
81 Prozent Accuracy. Das klingt gut und ist wertlos. Macro-F1 liegt bei etwa 0,30,
Recall auf die tödliche Klasse bei exakt null. Deshalb Macro-F1 und nicht Accuracy.

Und dann das zweite Gate: Selbst Macro-F1 könnte man theoretisch über die beiden
häufigen Klassen erreichen. Das Recall-Gate auf Klasse 1 verhindert das.

**Der Satz zur Auswahldisziplin ist wichtig für später:** Modellwahl und
Schwellenwert werden ausschließlich auf Validierung 2023 festgelegt. Test 2024
wird genau einmal geöffnet. Merke dir das, du brauchst es auf Folie 27 wieder.

**Falls nachgefragt wird, "Warum harmonisches Mittel bei F1?"**
Weil das arithmetische Mittel von Precision 1,0 und Recall 0,0 immer noch 0,5
wäre, obwohl das Modell nutzlos ist. Das harmonische Mittel ergibt in diesem
Fall 0. Es bestraft Ungleichgewicht zwischen den beiden Größen.

---

## Folie 8: Rahmenbedingungen und Ethik

**Kernsatz:** Interpretierbarkeit war eine harte Vorgabe, keine Kür, und die
Ergebnisse informieren Infrastrukturentscheidungen, nicht Urteile über Personen.

**Sprechnotizen:**
Zügig durchgehen, das ist eine Absicherungsfolie. Die zwei Punkte, bei denen du
verweilen solltest:

Erstens die Interpretierbarkeitsvorgabe. Sie erklärt, warum kein neuronales Netz
im Kandidatenfeld ist. Ein Verkehrsplaner muss nachvollziehen können, warum ein
Ort als Risiko markiert wurde.

Zweitens die ethische Rahmung. Sag ausdrücklich: Merkmalswichtigkeiten sind
Assoziationen, keine Ursachen, und dieser Hinweis steht auch in der App selbst.

**Begriffe erklärbar machen:**
- **Batch-Inferenz**: Vorhersagen werden gesammelt für viele Fälle auf einmal
  berechnet, nicht in Echtzeit einzeln. Das senkt die Anforderungen deutlich.
- **DSGVO-Konformität quellseitig**: Der Datensatz enthält von vornherein keine
  personenbezogenen Daten. Es gibt keine Namen, kein Alter, kein Geschlecht.
  Damit stellt sich die Frage der Anonymisierung gar nicht erst.

---

## Folie 9: Der Datensatz im Steckbrief

**Kernsatz:** 2,09 Millionen Unfälle über neun Jahre, 21 Spalten, öffentlich
lizenziert, vollständig ohne Personenbezug.

**Sprechnotizen:**
Die Tabelle nicht vorlesen. Die drei Zahlen nennen, die zählen: 2.092.401 Zeilen,
21 Spalten, 2016 bis 2024.

Dann die Merkmalsgruppen in einem Durchgang: Zeit, Unfallcharakter, Umgebung,
Beteiligung, Raum. Das gibt dem Publikum die mentale Struktur.

Der COVID-Hinweis ist ein guter, kurzer Beleg dafür, dass du die Daten wirklich
angeschaut hast: 2020 bricht ein, 2024 ist mit 268.519 wieder auf dem Niveau von
2019 mit 268.370.

**Begriffe erklärbar machen:**
- **UKATGEORIE**: Die Zielvariable. Achtung, im Quelldatensatz steht tatsächlich
  "UKATGEORIE" mit Tippfehler statt "UKATEGORIE". Das Projekt übernimmt die
  Schreibweise der Quelle bewusst unverändert. Falls das jemandem auffällt, ist
  das eine gute Antwort.
- **UART vs. UTYP1**: UART ist die *Unfallart*, also die Kollisionsform, zum
  Beispiel Auffahrunfall oder Zusammenstoß mit einem Fußgänger. UTYP1 ist der
  *Unfalltyp*, also die auslösende Konfliktsituation, zum Beispiel Abbiegeunfall
  oder Fahrunfall. Beide werden nach dem Unfall vom Protokollierenden vergeben.
- **Parquet**: Ein spaltenorientiertes Dateiformat. Es speichert Spalte für
  Spalte statt Zeile für Zeile, was Abfragen auf wenige Spalten sehr schnell
  macht und stark komprimiert.
- **Git LFS**: Large File Storage. Eine Git-Erweiterung, die große Binärdateien
  außerhalb der Versionsgeschichte speichert und im Repository nur einen Zeiger
  ablegt.
- **DuckDB**: Eine eingebettete analytische Datenbank, konzeptionell wie SQLite,
  aber spaltenorientiert und für Analysequeries optimiert. Kann direkt SQL auf
  Parquet-Dateien ausführen, ohne sie vorher zu importieren.

---

## Folie 10: Zielvariable und Imbalance

**Kernsatz:** Ein Klassenverhältnis von 1 zu 18 zu 81 macht Accuracy wertlos und
führt ohne Gegenmaßnahme zum Mehrheitsklassen-Kollaps.

**Sprechnotizen:**
Die Verteilung nennen und sofort die Konsequenz: Ein Modell, das nichts lernt,
bekommt 81 Prozent Accuracy geschenkt.

Die Stabilität über die Jahre ist ein wichtiges methodisches Detail: Die
Klassenanteile schwanken über neun Jahrgänge um weniger als einen Prozentpunkt.
Das ist die Rechtfertigung dafür, dass ein chronologischer Split überhaupt
zulässig ist. Gäbe es starke Drift, wäre Training auf alten Jahren wertlos.

Die Dunkelziffer unbedingt ansprechen. Sie ist der ehrlichste Punkt der ganzen
Datenphase: Die Verteilung, die wir sehen, ist auch eine Verteilung des
Meldeverhaltens, nicht nur der Realität. Leichte Unfälle werden seltener
gemeldet. Das lässt sich aus dem Datensatz heraus nicht korrigieren.

**Begriffe erklärbar machen:**
- **Klassenungleichgewicht (Class Imbalance)**: Die Klassen sind sehr
  unterschiedlich häufig. Das Problem ist nicht die Ungleichheit an sich,
  sondern dass Standard-Lernverfahren die Gesamtfehlerrate minimieren und
  deshalb die seltene Klasse ignorieren.
- **Mehrheitsklassen-Kollaps**: Der Zustand, in dem ein Modell gelernt hat,
  immer die häufigste Klasse vorherzusagen, weil das die Fehlerrate minimiert.
- **Dunkelziffer**: Die unbekannte Zahl nie gemeldeter Unfälle.

---

## Folie 11: Datenqualität und Plausibilisierung

**Kernsatz:** Fünf Klassen von Prüfungen wurden systematisch durchgeführt, und
imputiert wurde bewusst erst später in der Pipeline, nicht hier.

**Sprechnotizen:**
Diese Folie kann bei Zeitnot gekürzt werden. Wenn du sie ausführlich machst, ist
der interessanteste Punkt die letzte Box, die bewusste Entscheidung:

**Warum wird in Phase U nicht imputiert?** Weil eine Imputation, die auf dem
gesamten Datensatz berechnet wird, Informationen aus Validierung und Test in das
Training zieht. Wenn ich den Median einer Spalte über alle Daten berechne und
damit fehlende Werte fülle, dann steckt in jedem gefüllten Trainingswert
Information aus dem Testjahr 2024. Das ist Leakage. Deshalb wandert jede
Imputation in die sklearn-Pipeline, wo sie pro Fold nur auf Trainingsdaten
gefittet wird.

Die 8 Prozent Geokodierungsausfall sind ein guter Ehrlichkeitspunkt: Der
Datensatz ist bereits vorgefiltert, bevor wir ihn bekommen, und diese Filterung
ist nicht zufällig.

**Begriffe erklärbar machen:**
- **Sentinel-Wert**: Ein spezieller Zahlenwert, der "kein Messwert" bedeutet,
  zum Beispiel minus 999. Wird er nicht erkannt, rechnet das Modell damit wie
  mit einer echten Temperatur von minus 999 Grad.
- **Imputation**: Das Auffüllen fehlender Werte, zum Beispiel mit dem Median.
- **Selektionsverzerrung (Selection Bias)**: Die beobachtete Stichprobe ist
  nicht zufällig aus der Grundgesamtheit gezogen. Hier: Nur geokodierbare
  Unfälle sind enthalten, und Geokodierbarkeit hängt vermutlich mit Ortstyp
  zusammen.

---

## Folie 12: DWD-Wetteranreicherung

**Kernsatz:** Jeder Unfall wurde über einen k-d-Baum mit der nächsten
Wetterstation verknüpft, mit 99 Prozent räumlicher Abdeckung, aber erheblichen
Lücken bei Wind und Sicht.

**Sprechnotizen:**
Erkläre den k-d-Baum, das ist ein guter Punkt, um Verständnis für effizientes
Rechnen zu zeigen: Naiv müsste man für jeden der 2,09 Millionen Unfälle die
Distanz zu allen rund 400 Stationen berechnen, also fast eine Milliarde
Distanzberechnungen. Ein k-d-Baum ist eine Baumstruktur, die den Raum rekursiv
aufteilt, sodass man beim Suchen ganze Teilbäume ausschließen kann. Die Abfrage
läuft dann in unter 30 Sekunden.

Dann das Join-Detail, das oft übersehen wird und hier ehrlich benannt ist: Der
Unfallatlas hat kein Tagesdatum, nur Jahr, Monat, Stunde. Also wird über alle
Tage eines Monat-Stunde-Buckets gemittelt. Wenn ein Unfall am 3. Juli um 14 Uhr
passierte, bekommt er das Mittel aller 14-Uhr-Messungen im Juli. Das ist
Rauschen, aber es ist kein Leakage, weil die Zukunft nicht einfließt.

**Der Ehrlichkeitspunkt:** Windgeschwindigkeit fehlt zu 50,8 Prozent, Sichtweite
zu 54,4 Prozent. Das ist über die Hälfte. Sag das aktiv, bevor jemand fragt, und
verweise darauf, dass es dokumentiert und an A³ als Risiko übergeben wurde.

**Begriffe erklärbar machen:**
- **k-d-Baum (k-dimensional tree)**: Datenstruktur zur schnellen
  Nächste-Nachbarn-Suche in mehrdimensionalen Räumen.
- **CDC**: Climate Data Center, das Open-Data-Portal des DWD.
- **GeoNutzV**: Die Verordnung, unter der DWD-Daten frei nutzbar sind.
- **Proxy-Variable**: Ein Merkmal, das für etwas anderes steht, das man nicht
  direkt messen kann. `dwd_station_dist_km` ist ein Proxy für ländliche Lage:
  Wo keine Wetterstation in der Nähe ist, ist meist auch wenig Besiedlung.
  Im Notebook wurde geprüft, dass die Todesrate bei Unfällen fernab von
  Stationen über 20 Prozent höher liegt. Deshalb wurde die Spalte bewusst
  behalten, obwohl sie eigentlich ein Datenqualitäts-Artefakt ist.

---

## Folie 13: OSM-Straßenkontext über H3

**Kernsatz:** Statt für 2,09 Millionen Punkte einzeln die nächste Straße zu
suchen, wurde das Straßennetz einmal auf ein Hexagon-Gitter aggregiert.

**Sprechnotizen:**
Erst das Problem: Ein Nächste-Straße-Lookup für 2,09 Millionen Punkte gegen das
gesamte deutsche Straßennetz ist auf einem Laptop nicht praktikabel.

Dann H3 erklären: Uber hat ein System entwickelt, das die Erdoberfläche in
Sechsecke zerlegt, hierarchisch in mehreren Auflösungsstufen. Auf Stufe 8 ist
eine Zelle etwa 0,7 Quadratkilometer groß.

Warum Sechsecke und nicht Quadrate? Bei Sechsecken haben alle sechs Nachbarn
denselben Mittelpunktsabstand. Bei Quadraten sind die diagonalen Nachbarn weiter
weg als die orthogonalen. Für Nachbarschaftsanalysen ist das sauberer.

Der Trick: Das Straßennetz wird einmal pro Bundesland geladen und auf Zellen
aggregiert. Danach ist der Join pro Unfall nur noch ein Nachschlagen der
Zellen-ID, also praktisch kostenlos.

**Die dokumentierte Näherung offen ansprechen:** OSM bildet das heutige Netz ab,
die Unfälle reichen bis 2016 zurück. Ein Tempolimit kann sich geändert haben.
Das ist eine akzeptierte Approximation. Wichtig: Es ist kein Leakage, weil
OSM-Daten nicht vom Unfallausgang abhängen.

**Begriffe erklärbar machen:**
- **H3**: Hexagonales hierarchisches Geo-Indexsystem von Uber.
- **OSM**: OpenStreetMap, ein offenes, von Freiwilligen gepflegtes Kartenprojekt.
  Lizenz: ODbL, Open Database License.
- **Expositionsproxy**: `osm_road_density` steht indirekt für Verkehrsaufkommen.
  Wo mehr Straße ist, fährt in der Regel mehr Verkehr.

---

## Folie 14: Zentrale EDA-Befunde

**Kernsatz:** Kein einziges Merkmal erreicht auch nur Cramérs V von 0,15 zur
Zielvariable, das schwache Signal ist der zentrale Befund der Datenphase.

**Sprechnotizen:**
Diese Folie ist die Vorbereitung auf das spätere Scheitern. Mach sie deshalb
sorgfältig.

Cramérs V erklären: Ein Maß für den Zusammenhang zwischen zwei kategorialen
Variablen. Es basiert auf dem Chi-Quadrat-Test, normiert auf einen Bereich von
0 bis 1. Null bedeutet kein Zusammenhang, eins bedeutet, die eine Variable
bestimmt die andere vollständig.

Dann die Einordnung, die wichtig ist: **0,13 ist schwach.** Als grobe
Orientierung gilt bei diesen Freiheitsgraden etwa: unter 0,1 vernachlässigbar,
0,1 bis 0,3 schwach, 0,3 bis 0,5 mittel, darüber stark. Wir sind also am unteren
Rand von "schwach", und das ist der beste Wert im ganzen Datensatz.

Die zeitlichen Muster sind der anschaulichste Teil: Häufigkeit und Schwere
erzählen unterschiedliche Geschichten. Die meisten Unfälle passieren im
Nachmittagsverkehr. Die schwersten passieren nachts. Wer nur Unfallzahlen zählt,
sieht den Nachmittag. Wer Schwere betrachtet, sieht die Nacht.

Der Negativbefund am Ende ist bewusst drin: Monatlicher Niederschlag und
monatliche Todesrate bewegen sich nicht gemeinsam. Ein Nullergebnis, das
berichtet wird, ist ein Qualitätsmerkmal.

**Möglicher Sprungpunkt:** Hier kannst du kurz in den Live-Report springen und
die interaktive Cramérs-V-Heatmap zeigen.

**Begriffe erklärbar machen:**
- **Cramérs V**: Assoziationsmaß für kategoriale Variablen, abgeleitet aus
  Chi-Quadrat, normiert auf 0 bis 1.
- **EDA**: Exploratory Data Analysis, explorative Datenanalyse. Das systematische
  Anschauen der Daten, bevor modelliert wird.
- **Chi-Quadrat-Test**: Testet, ob die beobachtete gemeinsame Verteilung zweier
  kategorialer Variablen von der abweicht, die bei Unabhängigkeit zu erwarten
  wäre.

**Falls nachgefragt wird, "Warum nicht Korrelation?"**
Der Pearson-Korrelationskoeffizient setzt metrische Variablen und einen linearen
Zusammenhang voraus. Unfallart ist eine Nominalvariable ohne Ordnung, da wäre
Korrelation nicht definiert. Cramérs V ist das passende Gegenstück für
kategoriale Variablen.

---

## Folie 15: Leakage-Prüfung und chronologischer Split

**Kernsatz:** Leakage wurde aktiv gemessen, nicht angenommen, und der Split ist
chronologisch, weil ein Zufallssplit bei Zeitreihendaten die Realität
verfälschen würde.

**Sprechnotizen:**
Zuerst Leakage definieren, weil das der Begriff ist, nach dem am
wahrscheinlichsten gefragt wird: Informationen, die zum Vorhersagezeitpunkt nicht
verfügbar wären, gelangen ins Training. Ergebnis sind Leistungswerte, die im
echten Einsatz nicht halten.

Dann die konkrete Sorge hier: UART und UTYP1 werden nach dem Unfall vom
Protokollierenden vergeben. Könnte in diesen Kategorien die Schwere schon
implizit stecken? Wenn ja, wäre es Zirkelschluss.

Deshalb die Sonde: bedingte Entropiereduktion. Erkläre sie so:
Entropie ist ein Maß für Unsicherheit. H(Y) ist die Unsicherheit über die
Schwere, wenn man nichts weiß. H(Y|X) ist die verbleibende Unsicherheit, wenn
man das Merkmal X kennt. Die Reduktion sagt, wie viel Prozent der Unsicherheit
das Merkmal wegnimmt. Bei nahezu 100 Prozent würde das Merkmal die Antwort
praktisch enthalten.

Ergebnis: UART nimmt 3,4 Prozent weg, UTYP1 2,2 Prozent. Weit unter der
50-Prozent-Auslöseschwelle. Kein Merkmal musste raus.

Dann der Split. Die Zahlen nennen und den Grund: Bei einem Zufallssplit könnte
das Modell auf 2024er-Daten trainieren und auf 2019er-Daten getestet werden. Das
entspricht nicht dem Einsatz, wo man immer in die Zukunft vorhersagt.

Die Verifikation erwähnen: Keine OBJECTID kommt in mehr als einem Split vor.
Das wurde geprüft, nicht angenommen.

**Begriffe erklärbar machen:**
- **Data Leakage**: Siehe oben.
- **Entropie**: Informationstheoretisches Maß für Unsicherheit, in Bit.
- **Bedingte Entropie H(Y|X)**: Die verbleibende Unsicherheit über Y, wenn X
  bekannt ist.
- **OBJECTID**: Eindeutiger Zeilenschlüssel je Unfall.

---

## Folie 16: Die Machbarkeitswarnung [WICHTIG]

**Kernsatz:** Die Datenphase endete mit einer dokumentierten Warnung, dass das
ursprüngliche Ziel möglicherweise nicht erreichbar ist, und zwar bevor das
erste Modell trainiert wurde.

**Sprechnotizen:**
Diese Folie ist der Wendepunkt der Präsentation. Nimm dir Zeit.

Drei Gründe nacheinander:

1. Die tödliche Klasse ist etwa 1 Prozent der Daten.
2. Die stärkste Assoziation ist Cramérs V von 0,13, also schwach.
3. Und der eigentliche Grund: Die physikalisch entscheidenden Determinanten
   fehlen komplett.

Beim dritten Punkt bewusst langsam werden. Zähle auf: Aufprallgeschwindigkeit,
Alter der Insassen, Gurtnutzung, Fahrzeugmasse, Anstoßgeometrie.

Dann der Kernsatz, den du wörtlich sagen solltest:

> "Ob ein Unfall tödlich endet, entscheidet sich vor allem an der kinetischen
> Energie und daran, wie verletzlich der Mensch im Fahrzeug ist. Genau diese
> Größen enthält der öffentliche Datensatz nicht. Was er enthält, ist der
> Kontext drumherum."

Und dann der Schlusssatz der Folie, der die Prüfer überzeugt:

> "Diese Warnung wurde dokumentiert, bevor das erste Modell trainiert wurde.
> Das ist der Unterschied zwischen einer belegten Vorhersage und einer
> nachträglichen Ausrede."

---

## Folie 17: Aufbau der Modellierung

**Kernsatz:** Alles läuft in einer sklearn-Pipeline, damit Vorverarbeitung
niemals über die Split-Grenzen leckt.

**Sprechnotizen:**
Pipeline erklären, das ist der methodisch wichtigste Punkt der Folie: Eine
Pipeline verkettet Vorverarbeitungsschritte und Modell zu einem einzigen Objekt
mit gemeinsamem `fit` und `predict`. Der Effekt: Wenn die Kreuzvalidierung das
Objekt pro Fold neu fittet, werden auch Skalierung, Encoding und Imputation nur
auf dem Trainingsteil dieses Folds gefittet.

Zyklische Kodierung ist anschaulich und lohnt 20 Sekunden: Stunde 23 und
Stunde 0 liegen eine Stunde auseinander, als Zahlen aber 23 auseinander. Kodiert
man die Stunde als Sinus-Kosinus-Paar auf einem Kreis, ist der Abstand wieder
korrekt.

Target-Encoding kurz: Statt 87 Kreis-Spalten per One-Hot zu erzeugen, wird jeder
Kreis durch den mittleren Zielwert seiner Trainingszeilen ersetzt, mit Glättung
gegen Überanpassung bei seltenen Kreisen.

GroupKFold: Gruppiert nach Unfalljahr, damit kein Modell im selben Jahr trainiert
und validiert.

Der Audit-Modus am Ende ist ein starker Punkt: Das Notebook bricht ab, wenn eine
persistierte Metrik nicht auf 1e-9 genau reproduziert wird. Das heißt, die Zahlen
in dieser Präsentation sind nicht abgetippt, sondern werden bei jedem Lauf
gegengeprüft.

**Begriffe erklärbar machen:**
- **sklearn-Pipeline**: Verkettung von Transformationen plus Schätzer in einem
  Objekt.
- **One-Hot-Encoding**: Eine Kategorie mit k Ausprägungen wird zu k
  Null-Eins-Spalten.
- **Target-Encoding**: Ersetzt eine Kategorie durch den mittleren Zielwert ihrer
  Trainingszeilen. Braucht Glättung, sonst überangepasst bei seltenen Kategorien.
- **Kardinalität**: Anzahl verschiedener Ausprägungen einer Kategorie.
- **GroupKFold**: Kreuzvalidierung, die garantiert, dass alle Zeilen einer
  Gruppe im selben Fold landen.
- **Seed**: Startwert des Zufallszahlengenerators. Fixiert man ihn, sind
  Zufallsentscheidungen reproduzierbar.

---

## Folie 18: 19 Konfigurationen gegen das Drei-Klassen-Ziel

**Kernsatz:** Keine der 19 getesteten Konfigurationen erreicht den zulässigen
Bereich; die beste Macro-F1 liegt bei 0,424 gegenüber der geforderten 0,55.

**Sprechnotizen:**
Die Tabelle nicht vorlesen. Stattdessen das Muster zeigen, das sie enthält:

Es gibt zwei Gruppen. Die eine hat brauchbare Macro-F1, aber miserablen Recall
auf die tödliche Klasse, oben Random Forest balanced mit 0,424 und 0,212.
Die andere hat brauchbaren Recall, aber schlechte Macro-F1, XGBoost und LightGBM
balanced mit rund 0,60 Recall, aber nur 0,37 Macro-F1.

Sag dann den entscheidenden Satz: **"Das Gate verlangt beides gleichzeitig. Und
genau die Kombination existiert nicht. Es gibt keinen Punkt im zulässigen
Quadranten."**

Der letzte Satz der Folie ist wichtig, weil er einen Einwand vorwegnimmt: Auch
eine nachträgliche Verschiebung der Entscheidungsgrenze über einen
zweidimensionalen Offset-Sweep erreicht das Gate nicht. Es wurde also nicht nur
"nicht genug getunt".

**Begriffe erklärbar machen:**
- **balanced (Klassengewichtung)**: Fehler auf seltenen Klassen werden im
  Trainingsverlust stärker gewichtet, typischerweise umgekehrt proportional zur
  Klassenhäufigkeit.
- **Threshold Moving**: Nachträgliches Verschieben der Entscheidungsschwelle,
  ohne das Modell neu zu trainieren.
- **Offset-Sweep**: Systematisches Durchprobieren additiver Verschiebungen auf
  den Logits der Minderheitsklassen, um zu prüfen, ob irgendein Arbeitspunkt
  das Gate erreicht.

---

# Teil 2: Modell, Vergleich und Transfer

## Folie 19: Warum Oversampling das Problem verschlimmert

**Kernsatz:** SMOTE und ADASYN lassen den Recall auf die tödliche Klasse auf ein
bis drei Prozent kollabieren; nur Klassengewichtung wirkt überhaupt.

**Sprechnotizen:**
SMOTE erklären: Synthetic Minority Over-sampling Technique. Statt seltene
Beispiele einfach zu duplizieren, sucht SMOTE zu einem Minderheitsbeispiel seine
nächsten Nachbarn derselben Klasse und erzeugt neue Punkte auf den
Verbindungslinien dazwischen. ADASYN ist eine adaptive Variante, die mehr
Beispiele dort erzeugt, wo die Klassifikation schwer ist.

Dann das Ergebnis und die Erklärung, warum es hier scheitert:

> "SMOTE interpoliert zwischen echten Beispielen. Das funktioniert, wenn die
> Klassen im Merkmalsraum getrennt liegen. Hier überlappen sie fast vollständig,
> weil das Signal so schwach ist. Ein interpolierter Punkt zwischen zwei
> tödlichen Unfällen landet dann mitten in einer Wolke aus Leichtverletzten.
> Man erzeugt kein Signal, man erzeugt Rauschen."

Die Ordinal-Zerlegung kurz: Frank und Hall nutzen aus, dass die Klassen geordnet
sind, 1 ist schwerer als 2 ist schwerer als 3. Statt drei Klassen direkt zu
lernen, lernt man zwei binäre Fragen: "schwerer als 1?" und "schwerer als 2?".
Elegant, aber hier mit Recall 0,003 die schlechteste Strategie.

Schluss: Nur Klassengewichtung hebt den Recall über 0,50. Das ist ein Befund,
der der Intuition widerspricht, denn SMOTE gilt als Standardwerkzeug.

**Begriffe erklärbar machen:**
- **SMOTE**: Siehe oben. Wichtig: Darf nur auf Trainingsdaten angewendet werden,
  niemals auf Validierung oder Test, sonst Leakage.
- **ADASYN**: Adaptive Synthetic Sampling.
- **Frank-Hall-Zerlegung**: Ordinale Klassifikation über K minus 1 binäre
  Schwellenklassifikatoren.

---

## Folie 20: Die arithmetische Decke [SCHLÜSSELFOLIE]

**Kernsatz:** Unabhängig vom Modell lässt sich ausrechnen, dass das Gate eine
etwa 90-fache Odds-Anhebung erfordert, die ein Signal von Cramérs V 0,13 nicht
liefern kann.

**Sprechnotizen:**
Das ist die stärkste Folie der Arbeit. Sie zeigt, dass das Scheitern nicht am
Können liegt, sondern strukturell ist. Geh die vier Schritte langsam durch.

**Schritt 1:** Die F1 auf der Mehrheitsklasse liegt bei etwa 0,72. Macro-F1 ist
der Durchschnitt über drei Klassen. Damit dieser Durchschnitt 0,55 erreicht,
müssen die anderen beiden zusammen im Mittel etwa 0,46 beitragen.

**Schritt 2:** Klasse 1 macht 0,94 Prozent aus. Um F1 von 0,46 zu erreichen, wenn
gleichzeitig Recall mindestens 0,50 sein muss, braucht man Precision von etwa
0,42. Also: Von allen als tödlich vorhergesagten Unfällen müssten 42 Prozent
wirklich tödlich sein.

**Schritt 3:** Jetzt die Odds-Rechnung. Odds sind Wahrscheinlichkeit geteilt
durch Gegenwahrscheinlichkeit. Basisrate 0,94 Prozent ergibt Odds von etwa
0,0095. Precision von 0,42 ergibt Odds von etwa 0,72. Der Faktor dazwischen ist
rund 90.

Sag dann diesen Satz:

> "Das Modell müsste eine Teilgruppe finden, in der ein Unfall rund 90-mal
> wahrscheinlicher tödlich endet als im Durchschnitt. Nicht 90 Prozent mehr.
> 90-mal."

**Schritt 4:** Und die stärkste Assoziation im Datensatz liegt bei Cramérs V von
0,13. Ein so schwaches Signal kann eine solche Konzentration nicht erzeugen.

Abschluss: Empirische Front und arithmetische Anforderung zeigen unabhängig
voneinander auf dasselbe Ergebnis.

**Begriffe erklärbar machen:**
- **Odds**: Chancenverhältnis, p geteilt durch 1 minus p. Bei p gleich 0,5 sind
  die Odds 1. Bei p gleich 0,0094 sind sie etwa 0,0095.
- **Precision**: Von den als positiv vorhergesagten Fällen der Anteil, der
  wirklich positiv ist.
- **Basisrate**: Die Häufigkeit der Klasse in der Grundgesamtheit.

**Falls nachgefragt wird, "Ist die 90 exakt?"**
Nein, es ist eine Größenordnungsrechnung. Je nach Rundung landet man zwischen
etwa 76 und 90. Der Punkt ist nicht die zweite Nachkommastelle, sondern die
Größenordnung: Es braucht eine Konzentration um fast zwei Zehnerpotenzen, und
davon sind wir mit den verfügbaren Merkmalen weit entfernt.

---

## Folie 21: Das Reframing auf binäres KSI [SCHLÜSSELFOLIE]

**Kernsatz:** Die Zielvariable wurde auf KSI gegen Leichtverletzt umgestellt,
vorab definiert, sicherheitsrelevant und mit unverändertem Split.

**Sprechnotizen:**
Zuerst die Entscheidung: Das Drei-Klassen-Ziel bleibt als dokumentierter
Hintergrund, ist aber nicht länger primär. Begründung in einem Satz: Weiteres
Tuning würde dieselbe Suche in schwachem Signal wiederholen, ohne die verfügbare
Information zu verändern.

Dann KSI erklären: Killed or Seriously Injured, also getötet oder
schwerverletzt. Das ist ein etablierter Begriff in der Verkehrssicherheitsarbeit,
weil er genau die Grenze zieht, die planerisch relevant ist.

Dann die drei Rechtfertigungen, die du sicher können musst:

1. **Vorab definiert.** Verweis zurück auf Folie 5 und die H3-Hypothese. Der
   Fallback stand im Q-Phase-Notebook, bevor modelliert wurde.
2. **Sicherheitsrelevant.** Der Unterschied zwischen schwer und leicht verletzt
   ist der Unterschied, an dem sich Maßnahmen entscheiden. Es wird also nicht
   das Wichtige weggekürzt, sondern eine Grenze gezogen, die inhaltlich trägt.
3. **Statistisch tragfähig.** Aus 1 Prozent positiver Klasse werden etwa
   19 Prozent. Damit gibt es genug Fälle für belastbare Schätzungen.

Und der Satz, der methodische Sauberkeit zeigt: **Der zeitliche Split und die
Leakage-Grenze bleiben unverändert. Es wird nichts nachträglich angepasst außer
der Zielvariable selbst.**

**Falls nachgefragt wird, "Ist das nicht einfach das Problem leichter machen?"**
Ja, die Aufgabe wird leichter. Das wird auch offen so gesagt. Der Unterschied
zu unsauberem Vorgehen ist dreifach: Erstens war der Pfad vorab definiert.
Zweitens wird die ursprüngliche Zielverfehlung vollständig berichtet und nicht
gelöscht. Drittens bleibt die neue Aufgabe inhaltlich relevant. Unsauber wäre
es, das Ziel so lange zu ändern, bis irgendeine Zahl gut aussieht, und die
Vorgeschichte zu verschweigen.

**Falls nachgefragt wird, "Warum nicht Getötet gegen Rest?"**
Das wäre die härtere Trennung, hätte aber genau das Seltenheitsproblem behalten,
das 1-Prozent-Problem. KSI ist die in der Verkehrssicherheitsliteratur übliche
Zusammenfassung und löst das Häufigkeitsproblem, ohne die planerische Relevanz
zu verlieren.

---

## Folie 22: Zehn Kandidaten im binären Setting

**Kernsatz:** Zehn Modellfamilien von der Zufallsbaseline bis zu drei
Boosting-Verfahren, alle auf identischer Merkmalsbasis und nur auf Validierung
verglichen.

**Sprechnotizen:**
Die Baselines zuerst erklären, weil sie zeigen, dass das Modell wirklich etwas
lernt: Zufallsraten kommt auf 0,439, Mehrheitsklasse auf 0,453. Das ist die
Latte. Alles darüber ist echter Lerngewinn.

Dann die Familien in je einem Satz:
- **Random Forest**: Viele Entscheidungsbäume, jeder auf einer Zufallsstichprobe
  der Daten und einer Zufallsauswahl der Merkmale. Am Ende wird gemittelt. Die
  Idee: Einzelne Bäume überanpassen, aber ihre Fehler sind unkorreliert und
  mitteln sich weg.
- **Gradient Boosting**: Bäume werden nacheinander gebaut, jeder neue korrigiert
  die Fehler der bisherigen Summe. XGBoost, LightGBM und CatBoost sind drei
  optimierte Implementierungen davon.
- **SVM**: Support Vector Machine, sucht die trennende Hyperebene mit maximalem
  Abstand zu den nächsten Punkten.

Dann der ehrliche Rechenhinweis: Der RBF-Kernel-SVM skaliert kubisch mit der
Zeilenzahl. Bei 1,55 Millionen Trainingszeilen ist das nicht machbar, deshalb
wurde auf 8.000 Zeilen gesampelt. Sag das aktiv, es zeigt Bewusstsein für
Rechenkomplexität.

**Möglicher Sprungpunkt:** Hier passt ein kurzer Sprung in die App, Seite Model
Comparison, um die Pareto-Front live zu zeigen.

**Begriffe erklärbar machen:**
- **Ensemble**: Mehrere Modelle, deren Vorhersagen kombiniert werden.
- **Bagging vs. Boosting**: Bagging (Random Forest) trainiert viele Modelle
  parallel auf Zufallsstichproben und mittelt. Boosting trainiert sequenziell,
  jedes Modell korrigiert die Fehler der Vorgänger.
- **Blattweises Wachstum (LightGBM)**: Der Baum wächst dort weiter, wo der
  Gewinn am größten ist, statt Ebene für Ebene. Schneller, aber
  überanpassungsanfälliger.
- **Kernel-Trick (SVM)**: Erlaubt nichtlineare Trennung, indem implizit in einen
  höherdimensionalen Raum abgebildet wird, ohne diesen explizit zu berechnen.
- **RBF**: Radial Basis Function, der gebräuchlichste nichtlineare Kernel.

---

## Folie 23: Champion-Auswahl mit Gate-Logik

**Kernsatz:** Random Forest gewinnt, weil er als Kandidat mit der höchsten
Macro-F1 unter denen, die das Recall-Gate erfüllen, ausgewählt wird, das Gate
ist Vorbedingung, nicht Zielfunktion.

**Sprechnotizen:**
Das ist eine Folie, auf der ein Prüfer leicht einhaken kann, also sei präzise.

Die Regel: Nimm alle Kandidaten, die Recall(KSI) von mindestens 0,50 erfüllen.
Unter diesen nimm den mit der höchsten Macro-F1.

Der scheinbare Widerspruch, den du selbst ansprechen solltest: Random Forest hat
mit 0,540 den *niedrigsten* KSI-Recall aller Baum- und SVM-Kandidaten. Trotzdem
gewinnt er. Warum? Weil er das Gate erfüllt, und unter allen, die es erfüllen,
die höchste Macro-F1 hat.

Sag den Merksatz: **"Das Gate ist eine Vorbedingung, keine Zielfunktion. Es
sortiert aus, es optimiert nicht."**

Dann Optuna: Ein Framework für Hyperparameter-Optimierung, das nicht stumpf ein
Gitter durchprobiert, sondern modellbasiert sucht. Der Tree-structured Parzen
Estimator baut aus den bisherigen Versuchen ein Wahrscheinlichkeitsmodell darüber,
welche Parameterbereiche gute Ergebnisse liefern, und probiert dort weiter.
20 Trials ergaben 180 Bäume, Tiefe 23, mindestens 8 Beobachtungen pro Blatt. Das
hob die Macro-F1 von 0,6011 auf 0,6083.

**Begriffe erklärbar machen:**
- **Hyperparameter**: Einstellungen, die vor dem Training festgelegt werden, im
  Gegensatz zu Parametern, die das Modell aus den Daten lernt.
- **n_estimators**: Anzahl der Bäume im Wald.
- **max_depth**: Maximale Tiefe eines Baums. Begrenzt die Komplexität und wirkt
  gegen Überanpassung.
- **min_samples_leaf**: Mindestanzahl an Beobachtungen in einem Blatt. Verhindert,
  dass Blätter auf einzelne Ausreißer gefittet werden.
- **TPE**: Tree-structured Parzen Estimator, das Suchverfahren von Optuna.

---

## Folie 24: Schwellenwert und Negativbefund

**Kernsatz:** Der Schwellenwert wurde systematisch über 81 Kandidaten optimiert,
und ein zweiter Tuning-Versuch wurde nach einer klaren Regel verworfen statt
schöngerechnet.

**Sprechnotizen:**
Erst der Schwellenwert. Der Punkt, den viele nicht auf dem Schirm haben: Ein
Klassifikator gibt eine Wahrscheinlichkeit aus. Erst der Schwellenwert macht
daraus eine Entscheidung. Der Default 0,5 ist eine Konvention, keine
Optimierung. Hier wurden 81 Werte geprüft und der beste unter der Nebenbedingung
Recall größer gleich 0,50 gewählt: 0,49860.

Interessantes Detail, das du erwähnen kannst: Der optimale Wert liegt sehr nah an
0,5. Das ist kein Zufall, sondern eine Folge der Klassengewichtung, die die
Wahrscheinlichkeiten schon vorher grob ausbalanciert hat.

Dann der Negativbefund. Erkläre die Promotionsregel: Der neue Versuch wird nur
übernommen, wenn er auf *beiden* Achsen nicht schlechter ist. Der
Multi-Objective-Versuch hatte besseren Recall, 0,5187 gegen 0,5053, aber
schlechtere Macro-F1, 0,6057 gegen 0,6083. Also verworfen.

Der Satz, der zeigt, warum das gut ist:

> "Ich hätte auch die Regel nachträglich ändern und den besseren Recall
> berichten können. Die Regel stand aber vorher fest, und der Versuch ist als
> gescheiterte Evidenz dokumentiert statt gelöscht."

**Begriffe erklärbar machen:**
- **Entscheidungsschwelle**: Der Wahrscheinlichkeitswert, ab dem als positiv
  klassifiziert wird.
- **Multi-Objective-Optimierung**: Optimierung mehrerer Zielgrößen gleichzeitig.
  Ergebnis ist keine einzelne beste Lösung, sondern eine Pareto-Menge.
- **Pareto-Front**: Die Menge der Lösungen, bei denen man eine Zielgröße nur
  verbessern kann, indem man eine andere verschlechtert.

---

## Folie 25: Das Testergebnis 2024

**Kernsatz:** Macro-F1 0,604 und Recall(KSI) 0,515 auf einem nie zuvor benutzten
Jahr, beide Gates erfüllt, minimaler Abstand zur Validierung.

**Sprechnotizen:**
Zuerst betonen, wie der Test verwendet wurde: genau einmal geöffnet, für genau
ein Modell, mit dem vorher festgelegten Schwellenwert.

Dann die Zahlen. Der wichtigste Vergleich ist Validierung gegen Test:
0,6083 gegen 0,6039 bei Macro-F1. Beim Recall sogar leicht besser, 0,5053 gegen
0,5151. Sag ausdrücklich, warum das gut ist: Wenn ein Modell auf der Validierung
deutlich besser wäre als auf dem Test, hätte man den Validierungssatz
überangepasst. Hier ist der Abstand minimal.

Dann die Konfusionsmatrix in Worte fassen: Von rund 44.200 echten KSI-Unfällen
wurden 22.767 erkannt und 21.431 übersehen. Von rund 224.300 Leichtunfällen
wurden 172.434 korrekt erkannt und 51.887 fälschlich als KSI markiert.

Und die ehrliche Einordnung, die du nicht auslassen solltest:

> "Rund 21.400 KSI-Unfälle werden übersehen. Das Modell ist ein
> Priorisierungswerkzeug, kein Sicherheitssystem. In einem echten Einsatz müsste
> die Kostenfunktion des Anwenders den Schwellenwert bestimmen, nicht Macro-F1.
> Wenn ein übersehener schwerer Unfall zehnmal teurer ist als ein Fehlalarm,
> gehört der Schwellenwert deutlich nach unten."

**Begriffe erklärbar machen:**
- **Konfusionsmatrix**: Kreuztabelle aus tatsächlicher und vorhergesagter Klasse.
- **False Negative**: Ein KSI-Unfall, der als leicht eingestuft wurde. Hier der
  teure Fehler.
- **False Positive**: Ein Leichtunfall, der als KSI eingestuft wurde. Hier der
  billigere Fehler, kostet nur Aufmerksamkeit.

---

## Folie 26: Die Entscheidungsmatrix

**Kernsatz:** Über vier gewichtete Kriterien führt XGBoost, nicht das
ausgelieferte Modell, und dieses unbequeme Ergebnis wird berichtet statt
versteckt.

**Sprechnotizen:**
Erst die Methodik: Sechs Kriterien waren nominal vorgesehen, jedes mit einem
Gewicht. Jedes Kriterium wird min-max-normalisiert, also auf 0 bis 1 skaliert,
und Kostenkriterien wie Latenz werden invertiert, weil dort niedriger besser ist.

Dann der Punkt, der methodische Sorgfalt zeigt: Zwei Kriterien,
Interpretierbarkeit und Trainingskosten, wurden nie gemessen. Statt sie
subjektiv zu schätzen und damit das Ergebnis zu färben, wurden sie automatisch
ausgeschlossen und die restlichen Gewichte renormalisiert.

Dann das Ergebnis, und zwar offensiv: XGBoost führt mit 0,561, Random Forest ist
Letzter mit 0,500. XGBoost hat mehr Recall und ist dreieinhalbmal schneller.

Kündige an, dass die nächste Folie erklärt, warum trotzdem nicht gewechselt wird.
Das erzeugt Spannung und verhindert, dass jemand vorschnell fragt.

**Begriffe erklärbar machen:**
- **Min-Max-Normalisierung**: Skalierung auf 0 bis 1 über
  (Wert minus Minimum) geteilt durch (Maximum minus Minimum).
- **Kostenkriterium**: Ein Kriterium, bei dem kleiner besser ist, hier die
  Latenz. Wird invertiert, damit größer immer besser bedeutet.
- **Latenz ms/1k**: Millisekunden für 1.000 Vorhersagen.

---

## Folie 27: Warum der Champion Champion bleibt [SCHLÜSSELFOLIE]

**Kernsatz:** Den Champion jetzt zu wechseln würde den Testsatz in den
Auswahlprozess ziehen und damit genau die Unabhängigkeit zerstören, die ihn
wertvoll macht.

**Sprechnotizen:**
Das ist die wichtigste Folie der Präsentation. Wenn du nur eine Sache perfekt
erklärst, dann diese. Nimm dir 90 Sekunden.

Bau das Argument in vier Schritten auf:

**Schritt 1, Die Ausgangslage.** Die Modellauswahl fiel in Phase A³ auf Basis
der Validierungsdaten 2023. Zu diesem Zeitpunkt war Test 2024 unberührt. Er war
reserviert für genau einen Zweck: eine bereits eingefrorene Entscheidung zu
bestätigen. Das ist geschehen.

**Schritt 2, Was ein Wechsel bedeuten würde.** Wollte ich jetzt XGBoost
ausliefern, müsste ich für ihn eine Testzahl berichten. Also müsste ich Test 2024
auch für XGBoost auswerten.

**Schritt 3, Warum das den Test entwertet.** In dem Moment ist der Testsatz Teil
der Auswahl geworden. Ich habe dann zwei Modelle auf dem Test gemessen und das
bessere genommen. Die berichtete Zahl ist dann das Maximum aus zwei Versuchen und
nicht mehr eine unverzerrte Schätzung der Generalisierungsleistung. Und
rückwirkend wäre auch die für Random Forest berichtete Zahl nicht mehr
unabhängig.

**Schritt 4, Der Name des Problems.** Das ist Data Leakage auf der Ebene der
Modellauswahl. Es ist subtiler als Leakage auf Merkmalsebene, weil kein einziger
Datenpunkt ins Training wandert. Trotzdem sickert Information aus dem Test in
die Entscheidung. In der Praxis ist das die häufigere Form.

Dann die Konsequenz: Random Forest bleibt für diesen Zyklus. XGBoost ist
dokumentiert als Kandidat für einen zukünftigen, vorab registrierten Vergleich.

Und der Merksatz zum Schluss, langsam:

> **"Die Validierung entscheidet. Der Test bestätigt nur."**

**Möglicher Sprungpunkt:** Hier lohnt der Sprung in die App, Seite Model
Comparison, wo genau diese Erklärung zweisprachig im Aufklapper steht. Das zeigt,
dass die Überlegung nicht nur in der Präsentation, sondern im Produkt steht.

**Falls nachgefragt wird, "Warum nicht einfach einen neuen Testsatz nehmen?"**
Es gibt keinen. 2024 ist das letzte verfügbare Jahr. Sobald 2025 veröffentlicht
ist, wäre genau das der saubere Weg: vorab festlegen, dass XGBoost gegen Random
Forest auf 2025 verglichen wird, und dann einmal messen.

**Falls nachgefragt wird, "Ist das nicht übertrieben streng?"**
Es ist der Standard, der reproduzierbare Forschung von Zahlenpolitur trennt. Der
ganze Wert eines Holdout-Sets liegt darin, dass es genau einmal benutzt wird.
Benutzt man es mehrfach zur Auswahl, ist es effektiv ein zweiter
Validierungssatz, und man hat keine unabhängige Schätzung mehr.

---

## Folie 28: Permutation Importance

**Kernsatz:** Unfallart und Unfalltyp dominieren, gefolgt von der Beteiligung
ungeschützter Verkehrsteilnehmer, gemessen modellagnostisch, damit alle vier
Finalisten vergleichbar sind.

**Sprechnotizen:**
Das Verfahren zuerst, es ist intuitiv: Nimm eine Merkmalsspalte und mische ihre
Werte über alle Zeilen zufällig durch. Damit ist der Zusammenhang zum Ziel
zerstört, aber die Verteilung der Spalte bleibt unverändert. Dann miss, wie stark
die Modellleistung einbricht. Großer Einbruch bedeutet, das Modell war auf dieses
Merkmal angewiesen.

Der Vorteil gegenüber der eingebauten Feature Importance von Bäumen: Sie ist
modellagnostisch. Man kann Random Forest, XGBoost, LightGBM und CatBoost auf
derselben Stichprobe und mit derselben Metrik vergleichen.

Dann die Rangliste. Unfallart und Unfalltyp vorn, das ist plausibel: Ein
Frontalzusammenstoß ist strukturell gefährlicher als ein Auffahrunfall im
Stau. Dann Motorrad- und Fahrradbeteiligung, also ungeschützte
Verkehrsteilnehmer. Das bestätigt H2 in der schwachen Form.

Der SHAP-Punkt am Ende ist wichtig, weil SHAP im Kurs vorkam und jemand fragen
könnte, warum es nicht verwendet wurde: SHAP erklärt, wie ein *einzelnes* Modell
seine transformierten Merkmale nutzt. Für einen Vergleich *zwischen* Modellen ist
es das falsche Werkzeug, weil jedes Modell einen anderen transformierten
Merkmalsraum hat. Permutation Importance arbeitet auf den Rohspalten und ist
deshalb vergleichbar.

**Ein ehrlicher Zusatz, falls Zeit ist:** Die Rangordnungen der vier Modelle
stimmen nur mäßig überein. Die mittlere Spearman-Korrelation zum Champion liegt
bei 0,37, die Überschneidung der Top-10 bei etwa 40 Prozent. Das heißt: Die
Modelle sind sich einig, *dass* Unfallart wichtig ist, aber uneinig über die
zweite Reihe. Das ist ein Hinweis, die Rangliste nicht überzuinterpretieren.

**Begriffe erklärbar machen:**
- **Permutation Importance**: Siehe oben.
- **Modellagnostisch**: Funktioniert unabhängig von der inneren Struktur des
  Modells, nur über Ein- und Ausgabe.
- **SHAP**: SHapley Additive exPlanations. Verteilt die Abweichung einer
  Vorhersage vom Mittelwert additiv auf die Merkmale, basierend auf dem
  Shapley-Wert aus der kooperativen Spieltheorie.
- **Spearman-Korrelation**: Korrelation der Ränge statt der Werte. Passend, wenn
  man Rangordnungen vergleicht.
- **Jaccard-Index**: Größe der Schnittmenge geteilt durch Größe der
  Vereinigungsmenge. Hier: Wie stark überlappen zwei Top-10-Listen?

---

## Folie 29: Robustheit und Modell-Uneinigkeit

**Kernsatz:** Random Forest ist mit 3,77 Prozent gekippten Vorhersagen das
stabilste Modell bei fehlenden Merkmalen, und die Boosting-Modelle sind sich
untereinander deutlich einiger als mit ihm.

**Sprechnotizen:**
Die Robustheitssonde erklären: Fünf Merkmale, die im echten Betrieb realistisch
ausfallen können, werden einzeln komplett auf NaN gesetzt. Also: Was passiert,
wenn die OSM-Anreicherung ausfällt oder die Wetterstation keine Daten liefert?
Gemessen wird, wie viele Vorhersagen dadurch ihre Klasse wechseln.

Ergebnis: Random Forest 3,77 Prozent, die Boosting-Modelle zwischen 7 und
9 Prozent. Der Vergleichswert zum SVM ist eindrücklich: über 52 Prozent gekippte
Vorhersagen bei einem einzigen fehlenden Merkmal. Das ist ein starkes praktisches
Argument für Baumensembles im Betrieb.

Dann die paarweise Uneinigkeit. Sie misst, wie oft zwei Modelle
unterschiedlich entscheiden. Random Forest gegen die Boosting-Modelle: rund
13 Prozent. Die Boosting-Modelle untereinander: nur 5 Prozent.

Die Interpretation ist interessant: Die drei Boosting-Modelle sind faktisch
Varianten derselben Idee und kommen zu sehr ähnlichen Entscheidungen. Random
Forest ist der methodische Ausreißer. Er optimiert auf ausgewogene Klassenleistung
und drängt nicht so stark auf KSI-Recall.

Praktische Schlussfolgerung, die du ziehen kannst: Für ein Ensemble wäre die
Kombination aus Random Forest und einem Boosting-Modell interessanter als aus
zwei Boosting-Modellen, weil unkorrelierte Fehler sich besser ausgleichen.

**Begriffe erklärbar machen:**
- **Robustheit**: Stabilität der Vorhersagen unter Störungen der Eingabe.
- **NaN**: Not a Number, die Standardkodierung für fehlende Werte.
- **Paarweise Uneinigkeit (Pairwise Disagreement)**: Anteil der Fälle mit
  unterschiedlicher harter Klassenzuweisung.

---

## Folie 30: Limitationen

**Kernsatz:** Die Leistungsgrenze liegt in den Daten, nicht im Modell, und alle
wesentlichen Einschränkungen sind dokumentiert.

**Sprechnotizen:**
Diese Folie zügig, aber vollständig. Sie ist eine Stärke, keine Schwäche.
Prüfer bewerten offen benannte Limitationen positiv.

Der wichtigste Punkt ist der zweite Block, die fehlenden Merkmale. Verweise
zurück auf Folie 16 und 20: Das ist die Ursache der Leistungsdecke, und sie ist
nicht durch bessere Modellierung behebbar.

Der letzte Punkt ist konzeptionell interessant: Der Schwellenwert ist eine
Wertentscheidung, keine technische. Er kodiert implizit, wie teuer ein übersehener
schwerer Unfall im Vergleich zu einem Fehlalarm ist. Diese Abwägung gehört
eigentlich dem Anwender, nicht dem Entwickler.

**Falls nachgefragt wird, "Was würde die Leistung am meisten verbessern?"**
Eindeutig zusätzliche Merkmale, keine besseren Algorithmen. Am wertvollsten wären
Aufprallgeschwindigkeit und Insassenalter. Beide sind in
Unfallforschungsdatenbanken vorhanden, aber nicht im offenen Datensatz, weil sie
personenbezogen oder erhebungsaufwendig sind.

---

## Folie 31: Der Inference Contract

**Kernsatz:** Eine maschinenlesbare Vertragsdatei fixiert Schema, Wertebereiche,
Schwellenwert und Artefakt-Prüfsumme, damit die App ohne Notebook-Ausführung
korrekt arbeitet.

**Sprechnotizen:**
Kann bei Zeitnot gekürzt werden. Wenn du sie machst, ist der Kern:

Das Problem: Wer eine Anwendung auf ein Modell aufsetzt, muss wissen, welche
Spalten in welchem Typ und Wertebereich erwartet werden, welcher Schwellenwert
gilt und welche Modelldatei die richtige ist. Normalerweise steht das verstreut
in Notebooks und wird beim Nachbau falsch rekonstruiert.

Die Lösung: eine JSON-Datei, die all das enthält, generiert aus den echten
Trainingsdaten, nicht aus Annahmen. 30 Pflichtspalten mit Typ, Herkunft und
Wertebereich.

Der stärkste Einzelpunkt ist die SHA-256-Prüfsumme: Beim Laden wird die Prüfsumme
jeder Modelldatei neu berechnet und gegen den Eintrag verglichen. Passt sie
nicht, bricht die Analyse ab. Damit ist ausgeschlossen, dass die Metadaten ein
Modell beschreiben, das inzwischen ein anderes ist.

**Begriffe erklärbar machen:**
- **SHA-256**: Kryptografische Hashfunktion. Erzeugt aus beliebigem Inhalt einen
  Fingerabdruck fester Länge. Schon eine minimale Änderung am Inhalt ändert den
  Hash vollständig.
- **Schema**: Die formale Beschreibung, welche Spalten mit welchen Typen
  erwartet werden.

---

## Folie 32: Die Streamlit-Anwendung

**Kernsatz:** Vier Seiten, die jeweils eine konkrete Nutzerfrage beantworten,
lauffähig allein aus eingecheckten Artefakten.

**Sprechnotizen:**
Die vier Seiten kurz benennen, jeweils mit der Frage, die sie beantwortet. Nicht
alle Details vorlesen.

Der wichtigste technische Punkt: Die App braucht keine Notebook-Ausführung. Alle
Artefakte, also Modell, Contract, Kennzahlen und Kartendaten, liegen im
Repository. Ein Prüfer kann klonen und starten.

**Empfohlener Sprungpunkt:** Hier lohnt sich die Live-Demo am meisten. Zeig den
Risk Predictor, klick ein Beispielszenario, zeig die Vorhersage. 60 Sekunden.
Wenn die App nicht startet, hast du diese Folie als Rückfalloption und kannst
einfach weiterreden.

**Begriffe erklärbar machen:**
- **Streamlit**: Python-Framework, das aus einem Skript eine Web-App macht. Bei
  jeder Interaktion läuft das Skript komplett neu, und Caching verhindert, dass
  teure Berechnungen wiederholt werden.
- **Artefakt**: Eine gespeicherte Datei als Ergebnis eines Analyseschritts, hier
  Modelle, Kennzahlen-CSVs und die Contract-JSON.

---

## Folie 33: Die Risikokarte und die Glättung

**Kernsatz:** Statt absoluter Anteile zeigt die Karte relatives Risiko mit
Bayes'scher Glättung, weil sonst 97,5 Prozent der Fläche einfarbig wären.

**Sprechnotizen:**
Erst das Problem, es ist anschaulich: KSI macht bundesweit 18,91 Prozent aus.
Wenn man eine Zelle rot färbt, sobald die Mehrheit der Unfälle KSI ist, bräuchte
sie das 2,64-Fache des Durchschnitts. Das schaffen 123 von 4.857 Zellen. Die
Karte wäre praktisch einfarbig und damit nutzlos.

Die Lösung: Nicht "ist die Mehrheit schwer?", sondern "ist diese Zelle schlechter
als der Durchschnitt, und um wie viel?".

Dann das zweite Problem, das Glättung nötig macht: Eine Zelle mit drei Unfällen,
von denen zwei KSI sind, hätte eine Rate von 67 Prozent, also das 3,5-Fache des
Durchschnitts. Das ist aber reines Rauschen bei drei Beobachtungen.

Shrinkage erklären: Man tut so, als hätte man zusätzlich k Pseudo-Unfälle
beobachtet, die exakt dem Bundesdurchschnitt entsprechen. Bei vielen echten
Beobachtungen fallen die kaum ins Gewicht. Bei wenigen ziehen sie die Schätzung
stark zum Durchschnitt. Das ist im Kern eine Bayes'sche Schätzung mit dem
Bundesdurchschnitt als Prior.

Warum k gleich 20: empirisch bestimmt, nicht geraten. Bei k gleich 10 rutschen
noch 6-Unfall-Zellen ins höchste Band, bei k gleich 50 bleiben nur 52 Zellen
übrig und die Karte wird flach. Bei 20 sind alle fünf Bänder sinnvoll besetzt.

Die Konfidenz-Transparenz ist ein schöner UX-Punkt: Zellen mit wenig Evidenz
werden blasser gezeichnet. Der Nutzer sieht also nicht nur den Schätzwert,
sondern auch, wie sicher er ist.

**Empfohlener Sprungpunkt:** Karte live zeigen und ein Risikoband ein- und
ausschalten.

**Begriffe erklärbar machen:**
- **Shrinkage**: Schätzungen werden zu einem globalen Mittelwert hingezogen,
  besonders stark bei kleinen Stichproben.
- **Prior**: In der Bayes-Statistik die Vorannahme vor Betrachtung der Daten.
  Hier: die bundesweite KSI-Rate.
- **Relatives Risiko**: Verhältnis der Rate in einer Gruppe zur Rate in der
  Referenz. Wert 2 bedeutet doppelt so hohes Risiko.
- **Pseudo-Beobachtung**: Ein fiktiver Datenpunkt, der die Vorannahme in die
  Schätzung einbringt.

**Falls nachgefragt wird, "Ist das nicht Manipulation der Daten?"**
Nein, es ist eine bewusste, dokumentierte Verzerrung zugunsten der Robustheit.
Ohne Glättung wären die auffälligsten Zellen systematisch die mit den wenigsten
Beobachtungen, weil kleine Stichproben die extremsten Raten produzieren. Die
Glättung verhindert genau diesen Artefakt. Sie ist in der Epidemiologie bei
Karten seltener Ereignisse Standard.

---

## Folie 34: Engineering und Qualitätssicherung

**Kernsatz:** Ein Fehler, den die Standard-Tests nicht sehen konnten, führte zu
einem zusätzlichen Browser-Test als Pflicht-Gate.

**Sprechnotizen:**
Die Geschichte ist konkret und bleibt hängen, deshalb kurz erzählen:

Die Kartenbibliothek verändert die Objekte, die man ihr übergibt, während sie
rendert. Wenn man diese Objekte zwischenspeichert, um Zeit zu sparen,
referenziert ein späterer Aufruf eine JavaScript-Variable, die es nicht mehr
gibt. Die Karte bleibt weiß.

Der tückische Teil: Streamlits eigenes Test-Framework führt keinen Browser aus.
Es hat null Fehler gemeldet, während die Karte im Browser kaputt war. Der Test
war grün, das Produkt war defekt.

Konsequenz: ein Headless-Browser-Test, der eine echte Browser-Instanz startet
und auf JavaScript-Fehler prüft. Plus ein Test, der die Invariante absichert,
dass diese Objekte nie zwischengespeichert werden.

Die Lehre in einem Satz: **"Ein grüner Test ist nur so viel wert wie das, was er
tatsächlich ausführt."**

Dann kurz die restliche Qualitätssicherung: über 440 Tests, Linting, Pre-Commit,
CI bei jedem Push, Notebooks im Audit-Modus.

**Begriffe erklärbar machen:**
- **Caching**: Zwischenspeichern teurer Berechnungsergebnisse.
- **Headless Browser**: Ein echter Browser ohne sichtbare Oberfläche, für
  automatisierte Tests.
- **CI (Continuous Integration)**: Automatische Ausführung von Tests und
  Prüfungen bei jeder Änderung im Repository.
- **Linting**: Statische Codeprüfung auf Stil- und Fehlerpotenzial.
- **Pre-Commit-Hook**: Prüfung, die vor jedem Commit automatisch läuft.

---

## Folie 35: Fazit

**Kernsatz:** Das Modell funktioniert, aber der eigentliche Beitrag liegt in der
methodischen Disziplin an drei Stellen, an denen sie etwas gekostet hat.

**Sprechnotizen:**
Ergebnis nennen: 0,604 und 0,515, beide Gates erfüllt.

Dann die drei Punkte, und zwar mit Betonung darauf, dass jeder etwas gekostet
hat:

1. **Der Negativbefund wurde nicht versteckt.** Es wäre einfacher gewesen, die
   Drei-Klassen-Ergebnisse wegzulassen und nur das binäre Modell zu zeigen.
2. **Das Reframing war vorab definiert.** Das hat Arbeit gemacht in Phase Q, als
   noch niemand wusste, dass man es brauchen würde.
3. **Die Test-Trennung wurde eingehalten, obwohl sie wehtat.** Die
   Entscheidungsmatrix bevorzugt XGBoost. Es wäre attraktiver gewesen, das
   bessere Modell zu präsentieren.

Dann der zentrale Erkenntnissatz:

> "Was die Grenze setzt, sind die Daten, nicht das Modell. Kein Algorithmus kann
> Information erzeugen, die nicht in den Daten steckt."

Ausblick kurz, vier Stichpunkte, nicht ausführen.

---

## Folie 36: Quellen

**Sprechnotizen:**
Nicht vorlesen. Ein Satz genügt: Datensätze oben, Methodenquellen unten, alle
im APA-7-Stil.

Falls jemand nach Fachliteratur zur Unfallforschung fragt, sag ehrlich:

> "Ich zitiere hier bewusst nur Quellen, die ich vollständig belegen kann, also
> die Datensätze und die Methodenpapiere zu den eingesetzten Verfahren. Für den
> Vergleich mit publizierten Ergebnissen aus der Unfallforschung gilt ohnehin
> eine Einschränkung, die im Repository dokumentiert ist: Solche Studien
> arbeiten meist mit Personen- und Fahrzeugmerkmalen, die dieser Datensatz nicht
> hat, oder mit einem anderen Zielformat. Ein direkter Zahlenvergleich wäre
> deshalb irreführend."

Das ist eine starke Antwort, weil sie die Grenze der Vergleichbarkeit selbst
benennt.

---

## Folie 37: Vielen Dank

**Sprechnotizen:**
Kurz. Auf die drei Artefakte verweisen und Fragen einladen. Wenn du magst,
lass die App im Hintergrund offen, damit du bei Fragen direkt hineinspringen
kannst.

---

# Glossar aller Fachbegriffe

Alphabetisch. Für den schnellen Blick vor dem Vortrag.

**A³:** Phase des QUA³CK-Modells: Algorithm (Modellwahl), Adapt (Datenanpassung,
vor allem Imbalance), Adjust (Hyperparameter).

**Accuracy:** Anteil korrekter Vorhersagen an allen Vorhersagen. Bei
unbalancierten Klassen irreführend.

**ADASYN:** Adaptive Synthetic Sampling. Variante von SMOTE, die mehr
synthetische Beispiele dort erzeugt, wo die Klassifikation schwierig ist.

**Bagging:** Bootstrap Aggregating. Viele Modelle parallel auf Zufallsstichproben
trainieren und mitteln. Grundlage von Random Forest.

**Baseline:** Bewusst einfaches Vergleichsmodell, zum Beispiel immer die
Mehrheitsklasse vorhersagen.

**Batch-Inferenz:** Vorhersagen werden gebündelt für viele Fälle berechnet, nicht
einzeln in Echtzeit.

**Boosting:** Modelle werden sequenziell trainiert, jedes korrigiert die Fehler
der bisherigen Summe.

**CatBoost:** Gradient-Boosting-Implementierung mit nativer Behandlung
kategorialer Merkmale.

**CDC:** Climate Data Center, das Open-Data-Portal des Deutschen Wetterdienstes.

**Chi-Quadrat-Test:** Prüft, ob die gemeinsame Verteilung zweier kategorialer
Variablen von der bei Unabhängigkeit erwarteten abweicht.

**CI:** Continuous Integration. Automatische Tests bei jeder Codeänderung.

**Cramérs V:** Assoziationsmaß für kategoriale Variablen, 0 bis 1, abgeleitet
aus Chi-Quadrat.

**Data Leakage:** Information, die zum Vorhersagezeitpunkt nicht verfügbar wäre,
gelangt ins Training oder in die Modellauswahl.

**DuckDB:** Eingebettete spaltenorientierte Analysedatenbank, führt SQL direkt
auf Parquet aus.

**Dunkelziffer:** Zahl der nie gemeldeten Ereignisse.

**Entropie:** Maß für Unsicherheit, gemessen in Bit.

**Ensemble:** Kombination mehrerer Modelle.

**F1-Score:** Harmonisches Mittel aus Precision und Recall.

**False Negative:** Positiver Fall, der als negativ vorhergesagt wurde. Hier: ein
übersehener KSI-Unfall.

**False Positive:** Negativer Fall, der als positiv vorhergesagt wurde. Hier: ein
Fehlalarm.

**Frank-Hall-Zerlegung:** Ordinale Klassifikation über K minus 1 binäre
Schwellenklassifikatoren.

**Git LFS:** Large File Storage. Speichert große Dateien außerhalb der
Git-Historie.

**GroupKFold:** Kreuzvalidierung, bei der alle Zeilen einer Gruppe im selben
Fold landen.

**H3:** Hexagonales hierarchisches Geo-Indexsystem von Uber.

**Holdout-Set:** Datenteil, der bis zur finalen Bewertung unberührt bleibt.

**Hyperparameter:** Einstellung, die vor dem Training festgelegt wird, im
Gegensatz zu gelernten Parametern.

**Imputation:** Auffüllen fehlender Werte.

**Jaccard-Index:** Schnittmenge geteilt durch Vereinigungsmenge.

**Kardinalität:** Anzahl verschiedener Ausprägungen einer Kategorie.

**k-d-Baum:** Datenstruktur für schnelle Nächste-Nachbarn-Suche in mehreren
Dimensionen.

**Kernel-Trick:** Nichtlineare Trennung durch implizite Abbildung in einen
höherdimensionalen Raum.

**Klassengewichtung:** Fehler auf seltenen Klassen werden im Trainingsverlust
stärker gewichtet.

**Klassenungleichgewicht:** Stark unterschiedliche Klassenhäufigkeiten.

**Konfusionsmatrix:** Kreuztabelle aus tatsächlicher und vorhergesagter Klasse.

**KSI:** Killed or Seriously Injured. Getötet oder schwerverletzt, zusammengefasst
als eine Klasse.

**Latenz:** Zeit für eine Vorhersage, hier in Millisekunden pro 1.000 Zeilen.

**LightGBM:** Gradient-Boosting-Implementierung mit blattweisem Baumwachstum.

**Linting:** Statische Codeprüfung.

**Macro-F1:** Ungewichteter Durchschnitt der klassenweisen F1-Werte.

**Mehrheitsklassen-Kollaps:** Das Modell sagt immer die häufigste Klasse voraus.

**Min-Max-Normalisierung:** Skalierung auf den Bereich 0 bis 1.

**Modellagnostisch:** Funktioniert unabhängig von der inneren Modellstruktur.

**NaN:** Not a Number, Kodierung für fehlende Werte.

**Odds:** Chancenverhältnis, p geteilt durch 1 minus p.

**One-Hot-Encoding:** Kategorie mit k Ausprägungen wird zu k Null-Eins-Spalten.

**Optuna:** Framework für Hyperparameter-Optimierung mit modellbasierter Suche.

**OSM:** OpenStreetMap, offenes Kartenprojekt, Lizenz ODbL.

**Overfitting:** Das Modell lernt Rauschen der Trainingsdaten und generalisiert
schlecht.

**Parquet:** Spaltenorientiertes, komprimiertes Dateiformat.

**Pareto-Front:** Menge von Lösungen, bei denen keine Zielgröße verbessert werden
kann, ohne eine andere zu verschlechtern.

**Permutation Importance:** Merkmalswichtigkeit über den Leistungseinbruch beim
zufälligen Durchmischen einer Spalte.

**Pipeline:** Verkettung von Vorverarbeitung und Modell in einem Objekt.

**Precision:** Von den als positiv vorhergesagten Fällen der Anteil, der wirklich
positiv ist.

**Prior:** Vorannahme in der Bayes-Statistik, vor Betrachtung der Daten.

**Proxy-Variable:** Merkmal, das stellvertretend für eine nicht direkt messbare
Größe steht.

**QUA³CK:** Prozessmodell: Question, Understanding, Algorithm/Adapt/Adjust,
Conclude & Compare, Knowledge Transfer.

**Random Forest:** Ensemble aus vielen Entscheidungsbäumen auf Zufallsstichproben
und Zufalls-Merkmalsauswahl.

**RBF:** Radial Basis Function, gebräuchlichster nichtlinearer SVM-Kernel.

**Recall:** Von den tatsächlich positiven Fällen der Anteil, den das Modell
gefunden hat.

**Relatives Risiko:** Verhältnis der Rate in einer Gruppe zur Referenzrate.

**Robustheit:** Stabilität der Vorhersagen unter Störungen der Eingabe.

**Seed:** Startwert des Zufallszahlengenerators, macht Zufall reproduzierbar.

**Selektionsverzerrung:** Die Stichprobe ist nicht zufällig aus der
Grundgesamtheit gezogen.

**Sentinel-Wert:** Spezieller Zahlenwert, der "kein Messwert" kodiert, etwa
minus 999.

**SHA-256:** Kryptografische Hashfunktion, erzeugt einen Fingerabdruck fester
Länge.

**SHAP:** SHapley Additive exPlanations. Verteilt eine Vorhersage additiv auf
die Merkmale.

**Shrinkage:** Schätzungen werden zu einem globalen Mittelwert hingezogen,
besonders bei kleinen Stichproben.

**sklearn:** scikit-learn, die Standard-ML-Bibliothek in Python.

**SMOTE:** Synthetic Minority Over-sampling Technique. Erzeugt synthetische
Minderheitsbeispiele durch Interpolation.

**Spearman-Korrelation:** Korrelation der Ränge statt der Werte.

**Streamlit:** Python-Framework, das aus einem Skript eine Web-App macht.

**SVM:** Support Vector Machine. Sucht die trennende Hyperebene mit maximalem
Abstand zu den nächsten Punkten.

**Target-Encoding:** Ersetzt eine Kategorie durch den mittleren Zielwert ihrer
Trainingszeilen, mit Glättung.

**Threshold Moving:** Nachträgliches Verschieben der Entscheidungsschwelle ohne
Neutraining.

**TPE:** Tree-structured Parzen Estimator, Suchverfahren von Optuna.

**UART:** Unfallart, die Kollisionsform.

**UKATGEORIE:** Zielvariable, Unfallkategorie. Schreibweise mit Tippfehler aus
der Quelle übernommen.

**UTYP1:** Unfalltyp, die auslösende Konfliktsituation.

**Vision Zero:** Verkehrspolitisches Ziel: keine Verkehrstoten und
Schwerverletzten bis 2050.

**XGBoost:** Weit verbreitete Gradient-Boosting-Implementierung.

**Zyklische Kodierung:** Periodische Merkmale werden als Sinus-Kosinus-Paar
kodiert, damit die Distanz zwischen Ende und Anfang stimmt.

---

# Wahrscheinliche Prüfungsfragen mit Antworten

**1. Warum Macro-F1 und nicht Accuracy?**
Weil die Klassen extrem unbalanciert sind. Ein Modell, das immer die
Mehrheitsklasse vorhersagt, erreicht 81 Prozent Accuracy, ohne irgendetwas
gelernt zu haben. Macro-F1 mittelt die F1-Werte über alle Klassen gleich
gewichtet, sodass eine ignorierte Klasse den Wert deutlich senkt.

**2. Warum ein chronologischer Split?**
Weil der Einsatzfall die Vorhersage auf zukünftigen Daten ist. Ein Zufallssplit
würde erlauben, auf 2024er-Daten zu trainieren und auf 2019er zu testen, was die
Leistung optimistisch verzerrt. Zusätzlich wurde geprüft, dass die Klassenanteile
über die Jahre stabil sind, sonst wäre Training auf alten Jahren fragwürdig.

**3. Ist das Reframing auf KSI nicht einfach Zielverfehlung kaschiert?**
Nein, aus drei Gründen. Der Fallback war in Phase Q vorab definiert. Die
ursprüngliche Zielverfehlung wird vollständig berichtet, mit allen 19
Konfigurationen. Und die neue Zielvariable ist inhaltlich relevant, weil die
Grenze zwischen schwer und leicht verletzt die planerisch entscheidende ist.
Unsauber wäre es, das Ziel wiederholt zu ändern, bis eine Zahl passt, und die
Vorgeschichte zu verschweigen.

**4. Warum wird nicht XGBoost ausgeliefert, wenn es in der Matrix führt?**
Weil die Modellauswahl bereits auf Validierungsdaten getroffen wurde und der
Testsatz 2024 danach nur zur Bestätigung geöffnet wurde. Ein Wechsel würde
erfordern, XGBoost ebenfalls auf dem Test zu messen, wodurch der Test Teil der
Auswahl würde. Die berichteten Testzahlen wären dann nicht mehr unabhängig. Das
ist Data Leakage auf Ebene der Modellauswahl.

**5. Was ist der wichtigste Grund für die begrenzte Leistung?**
Fehlende Merkmale, nicht der Algorithmus. Aufprallgeschwindigkeit,
Insassenalter, Gurtnutzung und Fahrzeugmasse bestimmen die Unfallschwere
physikalisch, sind aber im öffentlichen Datensatz nicht enthalten. Die stärkste
verfügbare Assoziation liegt bei Cramérs V von 0,13.

**6. Warum hat SMOTE nicht geholfen?**
Weil SMOTE zwischen echten Minderheitsbeispielen interpoliert. Das setzt voraus,
dass die Klassen im Merkmalsraum halbwegs getrennt liegen. Hier überlappen sie
fast vollständig, sodass interpolierte Punkte in Bereichen landen, die
tatsächlich von der Mehrheitsklasse dominiert werden. Der Recall auf die tödliche
Klasse fiel auf ein bis drei Prozent.

**7. Wie stellt ihr sicher, dass kein Leakage vorliegt?**
Auf vier Ebenen. Erstens die bedingte Entropiereduktion als Sonde auf allen
verdächtigen Merkmalen, alle unter 3,4 Prozent gegen eine Schwelle von 50 Prozent.
Zweitens der chronologische Split mit verifiziert überschneidungsfreien
OBJECTIDs. Drittens alle Vorverarbeitung in einer sklearn-Pipeline, damit
Statistiken nur auf Trainingsdaten gefittet werden. Viertens GroupKFold nach
Jahr in der Kreuzvalidierung.

**8. Warum Permutation Importance statt SHAP?**
Weil vier verschiedene Modellfamilien verglichen werden sollten. SHAP arbeitet
auf den transformierten Merkmalen eines konkreten Modells, die sich zwischen
Modellen unterscheiden. Permutation Importance arbeitet auf den Rohspalten und
ist damit modellagnostisch und vergleichbar.

**9. Ist das Modell einsatzbereit?**
Als Priorisierungswerkzeug ja, als Sicherheitssystem nein. Es übersieht rund
21.400 KSI-Unfälle im Testjahr. Für einen echten Einsatz müsste der Schwellenwert
aus der Kostenfunktion des Anwenders abgeleitet werden, nicht aus Macro-F1, und
es bräuchte eine Regelung zur regelmäßigen Neubewertung wegen möglicher Drift.

**10. Was würdet ihr als Nächstes tun?**
Erstens Wahrscheinlichkeitskalibrierung, damit die ausgegebenen Werte als echte
Wahrscheinlichkeiten interpretierbar sind. Zweitens eine kostenbasierte
Schwellenwahl gemeinsam mit einem Anwender. Drittens, sobald das Jahr 2025
verfügbar ist, ein vorab registrierter Vergleich zwischen Random Forest und
XGBoost auf diesem neuen, unberührten Testjahr.

**11. Warum sind Getötete und Schwerverletzte zusammengefasst, aber nicht
Schwer- und Leichtverletzte?**
Weil die Grenze zwischen schwer und leicht die planerisch entscheidende ist. Ein
Schwerverletzter bedeutet mindestens 24 Stunden stationäre Behandlung. Die
Zusammenfassung von Getöteten und Schwerverletzten ist in der
Verkehrssicherheitsarbeit als KSI etabliert und löst zugleich das
Seltenheitsproblem der Todesfälle.

**12. Wie gehst du mit der Dunkelziffer um?**
Gar nicht, und das ist eine bewusste, dokumentierte Grenze. Leichte Unfälle
werden systematisch seltener gemeldet. Die beobachtete Verteilung ist also auch
eine Verteilung des Meldeverhaltens. Aus dem Datensatz heraus ist das nicht
korrigierbar, weil es keine unabhängige Referenz gibt. Es wird deshalb als
Limitation benannt statt still ignoriert.
