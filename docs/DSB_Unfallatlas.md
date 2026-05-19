Datensatzbeschreibung
Unfallatlas | Kartenanwendung der Statistischen Amter des Bundes und der Lander

## Allgemeine Felder

| Spaltenname | Inhalt | Bemerkung |
| --- | --- | --- |
| ID | Laufende Nummer des Unfalls (pro Unfall ein Datensatz). | |
| ULAND | Bundesland. | Codes siehe unten. |
| UREGBEZ | Regierungsbezirk. | Bildet mit der Kennung des Bundeslandes (ULAND) den amtlichen Gemeindeschlussel. |
| UKREIS | Kreis. | Bildet mit der Kennung des Bundeslandes (ULAND) den amtlichen Gemeindeschlussel. |
| UGEMEINDE | Gemeinde. | Kodierung siehe Gemeindeverzeichnis. |
| UJAHR | Unfalljahr. | |
| UMONAT | Unfallmonat. | |
| USTUNDE | Unfallstunde. | |
| UWOCHENTAG | Wochentag. | Codes siehe unten. |

### ULAND (Bundesland)

| Code | Bundesland | Hinweis |
| --- | --- | --- |
| 01 | Schleswig-Holstein | Daten ab 2016 |
| 02 | Hamburg | Daten ab 2016 |
| 03 | Niedersachsen | Daten ab 2017 |
| 04 | Bremen | Daten ab 2016 |
| 05 | Nordrhein-Westfalen | Daten ab 2019 |
| 06 | Hessen | Daten ab 2016 |
| 07 | Rheinland-Pfalz | Daten ab 2017 |
| 08 | Baden-Wurttemberg | Daten ab 2016 |
| 09 | Bayern | Daten ab 2016 |
| 10 | Saarland | Daten ab 2017 |
| 11 | Berlin | Daten ab 2018 |
| 12 | Brandenburg | Daten ab 2017 |
| 13 | Mecklenburg-Vorpommern | Daten ab 2020 |
| 14 | Sachsen | Daten ab 2016 |
| 15 | Sachsen-Anhalt | Daten ab 2017 |
| 16 | Thuringen | Daten ab 2019 |

### UWOCHENTAG (Wochentag)

| Code | Bedeutung |
| --- | --- |
| 1 | Sonntag |
| 2 | Montag |
| 3 | Dienstag |
| 4 | Mittwoch |
| 5 | Donnerstag |
| 6 | Freitag |
| 7 | Samstag |

## Unfallmerkmale

### UKATEGORIE (Unfallkategorien)

Kriterium der Zuordnung ist jeweils die schwerste Unfallfolge.

| Code | Bedeutung |
| --- | --- |
| 1 | Unfall mit Getoteten |
| 2 | Unfall mit Schwerverletzten |
| 3 | Unfall mit Leichtverletzten |

### UART (Unfallart)

| Code | Bedeutung |
| --- | --- |
| 1 | Zusammenstoss mit anfahrendem/anhaltendem/ruhendem Fahrzeug |
| 2 | Zusammenstoss mit vorausfahrendem / wartendem Fahrzeug |
| 3 | Zusammenstoss mit seitlich in gleicher Richtung fahrendem Fahrzeug |
| 4 | Zusammenstoss mit entgegenkommendem Fahrzeug |
| 5 | Zusammenstoss mit einbiegendem / kreuzendem Fahrzeug |
| 6 | Zusammenstoss zwischen Fahrzeug und Fussganger |
| 7 | Aufprall auf Fahrbahnhindernis |
| 8 | Abkommen von Fahrbahn nach rechts |
| 9 | Abkommen von Fahrbahn nach links |
| 0 | Unfall anderer Art |

### UTYP1 (Unfalltyp)

| Code | Bedeutung |
| --- | --- |
| 1 | Fahrunfall |
| 2 | Abbiegeunfall |
| 3 | Einbiegen / Kreuzen-Unfall |
| 4 | Uberschreiten-Unfall |
| 5 | Unfall durch ruhenden Verkehr |
| 6 | Unfall im Langsverkehr |
| 7 | sonstiger Unfall |

### ULICHTVERH (Lichtverhaltnisse)

| Code | Bedeutung |
| --- | --- |
| 0 | Tageslicht |
| 1 | Dammerung |
| 2 | Dunkelheit |

## Beteiligungen

### IstRad (Unfall mit Rad)

Unfall, an dem mindestens ein Fahrrad beteiligt war.

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne Fahrradbeteiligung |
| 1 | Unfall mit Fahrradbeteiligung |

### IstPKW (Unfall mit Pkw)

Unfall, an dem mindestens ein Personenkraftwagen beteiligt war.

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne PKW-Beteiligung |
| 1 | Unfall mit PKW-Beteiligung |

### IstFuss (Unfall mit Fussganger/in)

Unfall, an dem mindestens eine Fussgangerin oder ein Fussganger beteiligt war.

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne Fussgangerbeteiligung |
| 1 | Unfall mit Fussgangerbeteiligung |

### IstKrad (Unfall mit Kraftrad)

Unfall, an dem mindestens ein Kraftrad (z. B. Mofa, Motorrad/-roller) beteiligt war.

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne Kraftradbeteiligung |
| 1 | Unfall mit Kraftradbeteiligung |

### IstGkfz (Unfall mit Guterkaftfahrzeug, GKFZ)

Unfall, an dem mindestens ein Lastkraftwagen mit Normalaufbau und einem Gesamtgewicht
uber 3,5 t, ein Lastkraftwagen mit Tankauflage bzw. Spezialaufbau, eine Sattelzugmaschine
oder eine andere Zugmaschine beteiligt war.

Hinweis: Diese Kategorie ist in den Jahren 2016 und 2017 in "Unfall mit Sonstigen" enthalten.

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne Guterkaftfahrzeugbeteiligung |
| 1 | Unfall mit Guterkaftfahrzeugbeteiligung |

### IstSonstige (Unfall mit Sonstigen)

Unfall, an dem mindestens ein oben nicht genanntes Verkehrsmittel beteiligt war,
wie z. B. ein Bus oder eine Strassenbahn (2016 und 2017 einschliesslich Unfall
mit Guterkaftfahrzeug (GKFZ), ab 2018 ohne Unfall mit GKFZ).

| Code | Bedeutung |
| --- | --- |
| 0 | Unfall ohne Beteiligung eines oben nicht genannten Verkehrsmittels |
| 1 | Unfall mit Beteiligung eines oben nicht genannten Verkehrsmittels |

## Strassenzustand und Geometrie

### USTRZUSTAND / istStrasse (Strassenzustand)

| Code | Bedeutung |
| --- | --- |
| 0 | trocken |
| 1 | nass/feucht/schlupfrig |
| 2 | winterglatt |

### LINREFX / LINREFY

X-Koordinate (LINREFX) und Y-Koordinate (LINREFY) bilden die Koordinate des auf den
Strassenabschnitt liegenden Unfallortes (UTM-Koordinate des Referenzsystems ETRS89,
Zone 32N).

### XGCSWGS84 / YGCSWGS84

X-Koordinate (XGCSWGS84) und Y-Koordinate (YGCSWGS84) bilden die geographische
Koordinate des auf den Strassenabschnitt liegenden Unfallortes (Dezimalgrad,
Referenzsystem WGS84).

## Plausibilisierung

### PLST (Plausibilisierungsstufe)

| Code | Bedeutung |
| --- | --- |
| 1 | Erfolgreiche Plausibilisierung des Unfallortes nach regularem Verfahren |
| 2 | Erfolgreiche Plausibilisierung des Unfallortes nach erweitertem Verfahren fur Unfalle mit Fahrradbeteiligung |

## Quellen und Hinweise

Stand: 10.06.2025

Gemeindeverzeichnis:
https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/_inhalt.html

Erlauterungen zu Unfallmerkmalen:
https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Verkehrsunfaelle/Methoden/_inhalt.html#sprg371798
