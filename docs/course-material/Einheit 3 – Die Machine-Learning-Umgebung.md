---
title: Einheit 3 – Die Machine-Learning-Umgebung
description: Einheit 3
date: 12-05-2026
time: 12:25
reference: Data Analytics und Big Data
index: ""
subindex: ""
status:
  - begin
---

# Einheit 3 – Die Machine-Learning-Umgebung

>- **Reference Link:** [[Data Analytics und Big Data]]

---
>[!summary]
> Diese Einheit legt das begriffliche Fundament, auf dem die A³-Phase des QUA³CK-Modells aufbaut: Was Machine Learning überhaupt *ist*, welche grundsätzlichen Systemarten es gibt, und woran Modelle scheitern – an schlechten Daten oder an schlechten Algorithmen.
>
> Wer die Begriffe aus dieser Einheit nicht sicher beherrscht, wird spätestens bei der Algorithmusauswahl in A³ raten statt entscheiden.

---

### 1. Warum braucht man ein Verständnis der ML-Umgebung?

Bevor man einen Algorithmus auswählt, sollte man wissen, in welchem begrifflichen Rahmen man sich bewegt. Die Landschaft des Machine Learning ist groß: überwachtes und unüberwachtes Lernen, Batch- und Onlinelernen, instanzbasierte und modellbasierte Verfahren – dazu kommen Begriffe wie Overfitting, Underfitting, Validierungsdatensatz und Hyperparameter, die in praktisch jeder späteren Einheit vorausgesetzt werden.

>[!note]
> Diese Einheit enthält bewusst wenig Code. Es geht um Landkarte, nicht um Werkzeugkasten – die Werkzeuge kommen in den folgenden Einheiten.

---

### 2. Einordnung in das QUA³CK-Modell

Diese Einheit liefert die Grundbegriffe, auf denen zwei QUA³CK-Phasen aufbauen.

| Phase  | Bedeutung                                                         | Rolle in dieser Einheit                                                        |
| ------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **Q**  | Question                                                          | nicht direkt betroffen                                                          |
| **U**  | Understanding the Data                                            | das Datenverständnis aus [[Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung]] entscheidet mit, welche ML-Systemart überhaupt sinnvoll ist |
| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | die hier vorgestellten Lernarten sind die Bausteine der späteren Algorithmusauswahl |
| **C**  | Conclude & Compare                                                | nicht direkt betroffen                                                          |
| **K**  | Knowledge Transfer                                                | nicht direkt betroffen                                                          |

>[!note]
> Diese Einheit ist fundamental sowohl für **U** als auch für **A³**: Ohne die Unterscheidung überwacht/unüberwacht/Reinforcement etc. lässt sich in [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]]s A³-Phase gar nicht sinnvoll über "Algorithm Selection" reden.

---

### 3. Lernziele der Einheit

Nach dieser Einheit solltest du:

- ML in eigenen Worten definieren und von klassischer Programmierung abgrenzen können
- Beispielanwendungen für ML nennen können
- überwachtes, unüberwachtes, selbstüberwachtes und Reinforcement Learning unterscheiden können
- Batch- vs. Onlinelernen sowie instanzbasiertes vs. modellbasiertes Lernen erklären können
- die Hauptursachen für schlechte Modelle (schlechte Daten / schlechte Algorithmen) benennen können
- Overfitting und Underfitting erkennen und Gegenmaßnahmen nennen können
- Train/Validierung/Test/Train-Dev-Set korrekt einsetzen können
- das No-Free-Lunch-Theorem erklären können

>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Maschinelles_Lernen_erklärt.mp4*) und ein Podcast (*Machine_Learning_Die_Landkarte_für_Praktiker.m4a*) sowie mehrere Foliensätze (*Was-ist-Maschinelles-Lernen.pdf*, *Machine-Learning-Praxisleitfaden.pdf*, *Die_Landkarte_des_Machine_Learning.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert – bei Bedarf manuell sichten.

---

### 4. Was ist Machine Learning?

Machine Learning lässt sich locker als die Wissenschaft (und Kunst) beschreiben, Computer so zu programmieren, dass sie aus Daten lernen, statt für jeden Einzelfall explizit programmiert zu werden.

Zwei klassische Definitionen prägen den Diskurs bis heute:

- Arthur Samuel (1959): ML ist das Fachgebiet, das Computern die Fähigkeit verleiht zu lernen, ohne explizit programmiert zu werden.
- Tom Mitchell (1997): Ein Programm lernt aus Erfahrung E bezüglich einer Aufgabe T und eines Leistungsmaßes P, wenn seine durch P gemessene Leistung bei T mit der Erfahrung E wächst.

Am Beispiel Spamfilter: Die Aufgabe T ist das Erkennen neuer Spam-Mails, die Erfahrung E sind die als Spam/Ham markierten Trainingsbeispiele, und das Leistungsmaß P könnte der Anteil korrekt klassifizierter Mails sein (Genauigkeit). Der Teil des Systems, der lernt und Vorhersagen erzeugt, heißt Modell.

>[!important]
> Das bloße Speichern von Daten ist noch kein Machine Learning. Ein Computer, der eine komplette Kopie von Wikipedia herunterlädt, weiß danach mehr Zeichen, aber er wird bei keiner konkreten Aufgabe besser – ihm fehlt schlicht die Aufgabe T.

Der zentrale Unterschied zur klassischen Programmierung: Statt Regeln von Hand zu schreiben und bei jeder Ausnahme nachzubessern, leitet das ML-System Muster direkt aus Daten ab. Das Programm bleibt kürzer, wartbarer und passt sich an, wenn sich die zugrunde liegenden Daten ändern.

---

### 5. Wozu Machine Learning?

Klassische, regelbasierte Programmierung liefert bei vielen Aufgaben lange Listen fragiler Regeln, die bei jeder Ausnahme neu geschrieben werden müssen (klassisches Beispiel: Spam-Wörter wie "Für Sie" werden geblockt, also weichen Spammer auf "4U" aus – und die Regeln beginnen wieder von vorn). ML-Systeme lernen solche Muster stattdessen direkt aus Beispieldaten und passen sich automatisch an neue Varianten an.

| Anwendungsbereich | ML-Aufgabe | Typische Technik |
| --- | --- | --- |
| Produktbilder automatisch klassifizieren | Bildklassifikation | Convolutional Neural Networks (CNNs), Transformer |
| Tumoren in Gehirnscans erkennen | semantische Bildsegmentierung | CNNs, Transformer |
| Nachrichtenartikel klassifizieren | Textklassifikation (NLP) | RNNs, CNNs, Transformer |
| Firmenumsatz für das nächste Jahr vorhersagen | Regression | lineare/polynomielle Regression, Random Forest, neuronale Netze |
| Kreditkartenmissbrauch erkennen | Anomalieerkennung | Isolation Forest, gaußsche Mischverteilungen, Autoencoder |
| Kunden nach Kaufverhalten segmentieren | Clustering | k-Means, DBSCAN |
| Hochdimensionale Daten visualisieren | Dimensionsreduktion | PCA, t-SNE |
| Produktempfehlungen generieren | Empfehlungssystem | neuronale Netze auf Kaufhistorien |
| Intelligenten Spiel-Bot bauen | Reinforcement Learning | Policy-Optimierung (z. B. AlphaGo) |

>[!tip]
> Machine Learning glänzt besonders dort, wo klassische Lösungen entweder aus vielen Spezialregeln bestehen müssten, für komplexe Probleme schlicht kein bekannter Algorithmus existiert, oder sich die Umgebung laufend ändert und ein statisches Regelwerk veralten würde.

---

### 6. Arten von ML-Systemen im Überblick

ML-Systeme lassen sich nach mehreren, sich gegenseitig nicht ausschließenden Kriterien einordnen.

| Kriterium | Ausprägungen |
| --- | --- |
| Trainingsüberwachung | überwacht, unüberwacht, selbstüberwacht, teilüberwacht, Reinforcement Learning |
| Inkrementelles Lernen | Batch-Lernen (offline) vs. Onlinelernen (inkrementell) |
| Art der Verallgemeinerung | instanzbasiert (Vergleich mit bekannten Beispielen) vs. modellbasiert (Vorhersagemodell) |

Diese Kriterien lassen sich frei kombinieren: Ein moderner Spamfilter etwa lernt kontinuierlich mit einem neuronalen Netz aus von Nutzer:innen markierten Beispielen – er ist also gleichzeitig modellbasiert, überwacht und Online.

---

### 7. Überwachtes Lernen

Beim überwachten Lernen enthalten die Trainingsdaten bereits die gewünschten Lösungen, genannt Labels. Die beiden verbreitetsten Aufgaben sind:

- **Klassifikation**: Zuordnung zu diskreten Kategorien, z. B. Spam vs. Ham.
- **Regression**: Vorhersage einer numerischen Zielgröße (Target), z. B. der Preis eines Gebrauchtwagens auf Basis von Merkmalen wie Kilometerstand, Alter und Marke (den sogenannten Prädiktoren).

Beispielalgorithmen: lineare und logistische Regression, k-Nearest Neighbors, Support Vector Machines, Entscheidungsbäume und Random Forests, neuronale Netze.

>[!note]
> "Target" und "Label" werden oft synonym verwendet. Üblich ist aber: Target bei Regressionsaufgaben, Label bei Klassifikation. Merkmale heißen je nach Kontext auch Prädiktoren oder Attribute.

---

### 8. Unüberwachtes Lernen

Beim unüberwachten Lernen sind die Trainingsdaten nicht gelabelt – das System muss Struktur ohne Anleitung selbst entdecken.

| Technik | Ziel | Beispiel |
| --- | --- | --- |
| Clustering | ähnliche Datenpunkte gruppieren | Besuchergruppen eines Blogs identifizieren |
| Dimensionsreduktion | Merkmale vereinfachen ohne großen Informationsverlust | Kilometerstand und Alter eines Autos zu "Abnutzung" verdichten |
| Visualisierung | hochdimensionale Daten in 2D/3D darstellbar machen | t-SNE-Visualisierung semantischer Cluster |
| Anomalieerkennung | ungewöhnliche Datenpunkte erkennen | Kreditkartenbetrug, Produktionsfehler |
| Novelty Detection | neuartige, im Training völlig unbekannte Instanzen erkennen | neue Objektklasse in Bildern |
| Assoziationsregeln lernen | interessante Beziehungen zwischen Merkmalen finden | Warenkorbanalyse im Supermarkt |

>[!tip]
> Dimensionsreduktion lohnt sich oft auch als Vorverarbeitungsschritt vor einem überwachten Lernalgorithmus: weniger Merkmale bedeuten meist schnelleres Training und geringeren Speicherbedarf, manchmal sogar bessere Ergebnisse.

---

### 9. Selbstüberwachtes Lernen und Reinforcement Learning

**Selbstüberwachtes Lernen** erzeugt Labels automatisch aus einem eigentlich ungelabelten Datensatz – etwa indem ein Teil eines Bildes verdeckt und das Modell trainiert wird, das Original wiederherzustellen. Das trainierte Modell wird meist per Transfer Learning auf die eigentliche Zielaufgabe angepasst und mit wenigen gelabelten Beispielen feinjustiert. Weil es während des Trainings generierte Labels nutzt, aber auf einem ungelabelten Rohdatensatz aufsetzt, wird es am besten als eigene Kategorie zwischen überwachtem und unüberwachtem Lernen betrachtet.

**Reinforcement Learning** funktioniert grundlegend anders: Ein Agent beobachtet eine Umgebung, wählt Aktionen nach einer Policy aus und erhält dafür Belohnungen oder Strafen. Ziel ist es, über die Zeit die Policy zu finden, die die kumulierten Belohnungen maximiert. AlphaGo, das den Go-Weltmeister schlug, ist das bekannteste Beispiel.

>[!note]
> Auch teilüberwachtes Lernen existiert als Mischform: viele ungelabelte, wenige gelabelte Instanzen – etwa wenn Fotodienste Gesichter clustern (unüberwacht) und man dann nur ein Label pro Person ergänzt.

---

### 10. Batch-Lernen vs. Onlinelernen

Beim **Batch-Lernen** wird das System offline mit dem gesamten verfügbaren Datensatz trainiert und läuft anschließend unverändert im Produktivbetrieb – es lernt nicht weiter. Weil sich die Welt weiterentwickelt, das Modell aber stehen bleibt, verschlechtert sich die Qualität mit der Zeit (Model Rot / Data Drift). Abhilfe schafft nur ein regelmäßiges Neutraining mit aktuellen Daten, was allerdings Zeit und Rechenkapazität kostet und bei sehr großen Datenmengen an Grenzen stößt.

Beim **Onlinelernen** wird das System inkrementell trainiert, Datenpunkt für Datenpunkt oder in kleinen Mini-Batches. Das eignet sich für sich schnell ändernde Umgebungen und für Systeme mit begrenzten Ressourcen. Ein wichtiger Parameter ist die Lernrate: hoch angesetzt vergisst das System alte Muster schnell, niedrig angesetzt reagiert es träger, aber robuster gegenüber Rauschen und Ausreißern.

>[!important]
> Algorithmen zum Onlinelernen eignen sich auch für **Out-of-Core-Lernen**: Datensätze, die nicht komplett in den Hauptspeicher passen, werden stückweise geladen und trainiert. Das läuft trotz des Namens meist offline – "Onlinelearning" bezieht sich hier eher auf inkrementelles Lernen als auf einen Internetbezug.

---

### 11. Instanzbasiertes vs. modellbasiertes Lernen

Beim **instanzbasierten Lernen** merkt sich das System die Trainingsbeispiele im Wesentlichen auswendig und verallgemeinert über ein Ähnlichkeitsmaß auf neue Fälle (z. B. k-Nearest-Neighbors: neue Instanz gehört zur Mehrheitsklasse der ähnlichsten Nachbarn).

Beim **modellbasierten Lernen** wird stattdessen ein Modell mit Parametern aus den Beispieldaten entwickelt und dieses Modell dann für Vorhersagen genutzt. Der typische Workflow:

1. Daten untersuchen
2. Modell auswählen (z. B. lineares Modell)
3. Kostenfunktion definieren, die misst, wie schlecht das Modell zu den Trainingsdaten passt
4. Modell trainieren – der Algorithmus sucht die Parameter, die die Kostenfunktion minimieren
5. Modell für Vorhersagen auf neuen Daten nutzen (Inferenz)

Klassisches Beispiel: Ein lineares Modell "Zufriedenheit = θ₀ + θ₁ × BIP_pro_Kopf" wird auf Länderdaten trainiert, sodass die beiden Modellparameter θ₀ und θ₁ optimal zu den Trainingspunkten passen. Anschließend lässt sich für ein Land ohne bekannten Zufriedenheitswert eine Vorhersage aus dem BIP pro Kopf ableiten.

>[!note]
> "Modell" ist ein überladener Begriff: Er kann eine Modellart (lineare Regression), eine vollständig spezifizierte Architektur oder das fertig trainierte Modell mit konkreten Parameterwerten meinen. Modellauswahl bedeutet, Art und Architektur festzulegen; Training bedeutet, die konkreten Parameterwerte zu finden.

---

### 12. Die größten Herausforderungen: schlechte Daten

Da ein ML-System im Kern aus Modell + Trainingsdaten besteht, liegen Fehlerquellen entweder im Modell oder in den Daten. Bei den Daten sind vier Probleme besonders häufig:

| Problem | Kurzbeschreibung |
| --- | --- |
| Unzureichende Datenmenge | Die meisten Verfahren brauchen tausende bis Millionen Beispiele; bei komplexen Aufgaben wie Bild-/Spracherkennung kann selbst ein einfacher Algorithmus mit genug Daten gut abschneiden ("die unverschämte Effektivität von Daten") |
| Nicht repräsentative Trainingsdaten / Sampling Bias | Trainingsdaten müssen die Fälle abbilden, auf die später verallgemeinert werden soll – sonst entstehen systematische Verzerrungen (klassisches Beispiel: die Literary-Digest-Umfrage von 1936, die Roosevelts Sieg falsch vorhersagte, weil Stichprobe und Rücklauf wohlhabendere Befragte überrepräsentierten) |
| Minderwertige Datenqualität | Fehler, Ausreißer und Rauschen erschweren dem System, echte Muster von Zufall zu unterscheiden; ein Großteil der Data-Science-Arbeit besteht im Bereinigen solcher Daten |
| Irrelevante Merkmale | zu viele nutzlose oder fehlende relevante Merkmale verschlechtern das Ergebnis ("Müll rein, Müll raus"); Gegenmaßnahme ist gezieltes Feature Engineering (Auswahl, Extraktion, Neuerhebung von Merkmalen) |

>[!important]
> Auch eine sehr große Stichprobe kann verzerrt sein, wenn die Erhebungsmethode fehlerhaft ist. Größe allein schützt nicht vor Sampling Bias – das war schon 1936 das Problem, nicht die Stichprobengröße.

---

### 13. Die größten Herausforderungen: schlechte Algorithmen

Neben schlechten Daten kann auch das Modell selbst das Problem sein – typischerweise durch zu hohe oder zu niedrige Komplexität relativ zu den Daten.

>[!important]
> **Overfitting**: Das Modell passt sich zu genau an die Trainingsdaten an (inklusive Rauschen) und verallgemeinert schlecht auf neue Daten. Tritt bevorzugt bei komplexen Modellen, kleinen oder verrauschten Trainingsdatensätzen auf.

| Gegenmaßnahme bei Overfitting | Idee |
| --- | --- |
| Modell vereinfachen | weniger Parameter, weniger Merkmale, engere Restriktionen (Regularisierung) |
| mehr Trainingsdaten sammeln | reduziert relativen Einfluss von Zufallsmustern |
| Rauschen reduzieren | Datenfehler beheben, Ausreißer entfernen |

Die Stärke der Regularisierung wird über einen **Hyperparameter** gesteuert – im Unterschied zum Modellparameter (der beim Training gelernt wird) wird der Hyperparameter vor dem Training festgelegt und bleibt während des Trainings konstant.

>[!important]
> **Underfitting**: das genaue Gegenteil – das Modell ist zu einfach, um die Struktur der Daten zu erfassen, und liefert selbst auf den Trainingsdaten ungenaue Vorhersagen.

| Gegenmaßnahme bei Underfitting | Idee |
| --- | --- |
| mächtigeres Modell wählen | mehr Parameter, mehr Kapazität |
| bessere Merkmale bereitstellen | Feature Engineering |
| Restriktionen lockern | Regularisierungs-Hyperparameter verringern |

---

### 14. Testen und Validieren

Ob ein Modell gut verallgemeinert, lässt sich nur an neuen, ungesehenen Daten prüfen. Statt das direkt im Produktivsystem zu riskieren, teilt man die Daten in **Trainingsdatensatz** und **Testdatensatz** (häufig 80/20, bei sehr großen Datensätzen reicht ein kleinerer Testanteil). Die Differenz zwischen Trainingsfehler und Fehler auf dem Testdatensatz (Verallgemeinerungsfehler) zeigt Overfitting an.

Sollen mehrere Modelle oder Hyperparameter-Werte verglichen werden, reicht ein einzelner Testdatensatz nicht – wird er wiederholt zur Modellauswahl genutzt, "lernt" man indirekt auf ihm mit und überschätzt die tatsächliche Qualität. Die Lösung ist ein zusätzlicher **Validierungsdatensatz** (Hold-out-Validierung): Mehrere Modellkandidaten werden auf dem Trainingsdatensatz trainiert, auf dem Validierungsdatensatz verglichen, das beste Modell wird abschließend auf dem vollständigen Trainingsdatensatz neu trainiert und ein einziges Mal final auf dem Testdatensatz bewertet. Ist der Validierungsdatensatz zu klein, wird die Modellbewertung ungenau; ist er zu groß, fehlt dem Training Material. **Kreuzvalidierung** mit mehreren kleinen Validierungssets und gemitteltem Ergebnis mildert dieses Dilemma, kostet aber proportional mehr Trainingszeit.

Ein Sonderfall ist **Datendiskrepanz**: Trainingsdaten (z. B. Blumenbilder aus dem Web) unterscheiden sich systematisch von den späteren Produktivdaten (z. B. Fotos aus einer Handy-App). Hier hilft ein zusätzliches **Train-Dev-Set**: ein Teil der Trainingsdaten wird zurückgehalten und nach dem Training geprüft. Schneidet das Modell auf dem Train-Dev-Set schlecht ab, overfittet es bezüglich der Trainingsdaten; schneidet es dort gut, aber auf dem eigentlichen Validierungsdatensatz schlecht ab, liegt das Problem an der Datendiskrepanz zwischen den beiden Datenquellen.

>[!tip]
> Merkregel: Trainingsdatensatz zum Lernen, Validierungsdatensatz (bzw. Train-Dev-Set bei Datendiskrepanz) zum Vergleichen von Modellkandidaten, Testdatensatz für die einzige, abschließende Qualitätsschätzung.

---

### 15. Das No-Free-Lunch-Theorem

Jede Modellwahl trifft implizite Annahmen über die Daten – ein lineares Modell etwa unterstellt, dass die Beziehung zwischen Merkmalen im Kern linear ist und Abweichungen bloßes Rauschen sind. David Wolpert zeigte 1996: Ohne jegliche Annahme über die Daten gibt es keinen Grund, ein Modell einem anderen vorzuziehen (No-Free-Lunch-Theorem). Für manche Datensätze ist ein lineares Modell optimal, für andere ein neuronales Netz – kein Modell ist a priori allen anderen überlegen.

>[!note]
> Da man in der Praxis nicht alle denkbaren Modelle testen kann, trifft man wohlüberlegte Annahmen über die Daten und evaluiert nur eine sinnvoll eingeschränkte Auswahl an Modellen – genau das, was in der A³-Phase von QUA³CK als "Algorithm Selection" passiert.

---

### 16. Praktische Übung zur Einheit

| Aufgabe | Inhalt |
| --- | --- |
| 1. Batch vs. Online | Wähle ein eigenes Beispiel für Batch- vs. Onlinelernen und begründe die Wahl. |
| 2. Systemarten klassifizieren | Klassifiziere drei eigene Datensatz-Ideen nach überwacht/unüberwacht/Reinforcement Learning. |
| 3. Over-/Underfitting-Risiko einschätzen | Beschreibe für eines deiner Beispiele, ob eher Overfitting oder Underfitting das größere Risiko wäre, und warum. |
| 4. Validierungsstrategie entwerfen | Skizziere für ein eigenes Projekt, wie du Trainings-, Validierungs- und Testdatensatz aufteilen würdest. |

---

### 17. Zentrale Begriffe

| Begriff | Kurzdefinition |
| --- | --- |
| **Feature** | Eingabevariable, die das Modell für Vorhersagen nutzt |
| **Label** | Zielvariable bei Klassifikationsaufgaben |
| **Trainingsdatensatz** | Daten, mit denen das Modell lernt |
| **Testdatensatz** | unabhängiger Datensatz zur abschließenden Bewertung des Verallgemeinerungsfehlers |
| **Validierungsdatensatz** | zurückgehaltener Teil der Trainingsdaten zum Vergleich von Modellkandidaten und Hyperparametern |
| **Train-Dev-Set** | zusätzlicher Datensatz zur Unterscheidung von Overfitting und Datendiskrepanz |
| **Hyperparameter** | Einstellung des Lernalgorithmus, vor dem Training festgelegt, bleibt während des Trainings konstant |
| **Modellparameter** | Wert, den der Lernalgorithmus selbst aus den Trainingsdaten bestimmt |
| **Overfitting** | Modell passt sich zu stark an Trainingsdaten (inkl. Rauschen) an und verallgemeinert schlecht |
| **Underfitting** | Modell ist zu einfach, um die Struktur der Daten zu erfassen |
| **Sampling Bias** | systematische Verzerrung durch eine fehlerhafte Erhebungsmethode, unabhängig von der Stichprobengröße |
| **No-Free-Lunch-Theorem** | ohne Annahmen über die Daten ist kein Modell einem anderen a priori überlegen |
| **Batch-Lernen** | Training mit dem gesamten Datensatz auf einmal, offline, kein inkrementelles Lernen |
| **Onlinelernen** | inkrementelles Training mit einzelnen Datenpunkten oder Mini-Batches |
| **Out-of-Core-Lernen** | Onlinelernen-Technik für Datensätze, die nicht in den Hauptspeicher passen |
| **instanzbasiertes Lernen** | Verallgemeinerung durch Vergleich neuer Fälle mit gespeicherten Beispielen über ein Ähnlichkeitsmaß |
| **modellbasiertes Lernen** | Verallgemeinerung über ein aus den Daten trainiertes Vorhersagemodell |

---

### 18. Merksätze

>[!quote]
> Ein Modell kann nur so gut sein wie die Aufgabe T, die Erfahrung E und das Leistungsmaß P, die man ihm mitgibt.

>[!quote]
> Müll rein, Müll raus gilt für Datenmenge, Repräsentativität und Merkmalsauswahl gleichermaßen.

>[!quote]
> Overfitting merkt sich das Rauschen, Underfitting übersieht das Muster.

>[!quote]
> Der Testdatensatz darf nur einmal für das finale Urteil herhalten – wer ihn zur Modellauswahl missbraucht, betrügt sich selbst.

>[!quote]
> Kein Modell gewinnt immer: Ohne Annahmen über die Daten sind alle Modelle gleich gut – und gleich nutzlos.

---

### 19. Prüfungs- und Verständnisfragen

1. Wie unterscheidet sich Machine Learning von klassischer, regelbasierter Programmierung?
2. Nenne vier Anwendungsbereiche, für die sich Machine Learning besonders eignet.
3. Was unterscheidet überwachtes von unüberwachtem Lernen?
4. Welche zwei Aufgabentypen sind beim überwachten Lernen am verbreitetsten?
5. Nenne mindestens drei typische Aufgaben des unüberwachten Lernens.
6. Wodurch unterscheidet sich selbstüberwachtes Lernen von "echtem" unüberwachten Lernen?
7. Was ist der Kerngedanke von Reinforcement Learning, und wodurch unterscheidet er sich von überwachtem Lernen?
8. Worin liegt der Unterschied zwischen Batch- und Onlinelernen, und wann eignet sich Out-of-Core-Lernen?
9. Was unterscheidet instanzbasiertes von modellbasiertem Lernen?
10. Was ist der Unterschied zwischen einem Modellparameter und einem Hyperparameter?
11. Welche Ursachen für "schlechte Daten" kennst du, und was ist Sampling Bias konkret?
12. Was ist Overfitting, und welche drei Gegenmaßnahmen gibt es?
13. Wozu dient ein Validierungsdatensatz, und wann braucht man zusätzlich ein Train-Dev-Set?
14. Was besagt das No-Free-Lunch-Theorem, und welche praktische Konsequenz hat es für die Algorithmusauswahl?

---

### 20. Mini-Zusammenfassung

Machine Learning bedeutet, ein System aus Erfahrung (Daten) statt aus explizit programmierten Regeln lernen zu lassen. Systeme lassen sich nach Trainingsüberwachung (überwacht, unüberwacht, selbstüberwacht, Reinforcement Learning), nach Lernrhythmus (Batch vs. Online) und nach Verallgemeinerungsart (instanzbasiert vs. modellbasiert) einteilen. Modelle scheitern entweder an schlechten Daten (zu wenig, nicht repräsentativ, verrauscht, irrelevante Merkmale) oder an einem schlecht gewählten Algorithmus (Overfitting oder Underfitting). Train-, Validierungs- und Testdatensatz – bei Bedarf ergänzt um ein Train-Dev-Set – sorgen dafür, dass man sich nicht nur auf Hoffnung verlässt, sondern den Verallgemeinerungsfehler tatsächlich abschätzt. Das No-Free-Lunch-Theorem erinnert schließlich daran, dass kein Modell ohne Annahmen über die Daten grundsätzlich besser ist als ein anderes.

---

### Aufgabe

>[!important]
>Ordne dein eigenes QUA³CK-Projekt (oder ein geplantes Projekt) einer ML-Systemart entlang aller drei Kriterien zu (Trainingsüberwachung, Batch/Online, instanz-/modellbasiert) und begründe jede Zuordnung in ein bis zwei Sätzen.
>- Erkläre außerdem, ob in deinem Projekt eher Overfitting oder Underfitting das größere Risiko darstellt, und welche konkrete Gegenmaßnahme du einplanen würdest.
>- Skizziere, wie du Trainings-, Validierungs- und Testdatensatz aufteilen würdest (und ob ein Train-Dev-Set nötig wäre).
