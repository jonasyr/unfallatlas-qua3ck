# Einheit 3–5 Course Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw, undocumented course material for "Data Analytics und Big Data" (Kapitel 1, 3, 4 — the Géron *Praxiseinstieg Machine Learning* excerpts) into three new Obsidian-style markdown notes, `Einheit 3`, `Einheit 4`, `Einheit 5`, matching the depth, structure and tone of the existing `Einheit 1` and `Einheit 2` notes, and wire them into the course index note.

**Architecture:** This is a content-synthesis task, not a code task. There is no application logic to implement — the "build" consists of extracting text from PDFs with `pdftotext`, reading the extracted text plus the small `.txt` code listings, and hand-writing German-language markdown notes that mirror the established note template. Each task's "test" is a grep-based coverage check against a fixed checklist of terms that must appear in the finished note, standing in for unit tests since there is no executable behavior.

**Tech Stack:** `pdftotext` (poppler-utils, already installed), `grep`/`wc` for verification, plain markdown with Obsidian callout syntax (`>[!summary]`, `>[!note]`, `>[!tip]`, `>[!important]`, `>[!warning]`, `>[!danger]`, `>[!quote]`, `>[!check]`).

## Global Constraints

- Source root (read-only, **do not modify anything under it**): `/home/jonas/Documents/Studium/Semester_6/Data Analytics und Big Data/`
- Target repo: `/home/jonas/Documents/Code/unfallatlas-qua3ck` (branch `main`, currently clean).
- New files go in `docs/course-material/`, exact naming pattern copied from the existing files: `Einheit <N> – <Titel>.md` (en dash `–`, not hyphen).
- Every note MUST start with this exact frontmatter shape (values change per note, keys and order do not):
  ```yaml
  ---
  title: Einheit <N> – <Titel>
  description: Einheit <N>
  date: 01-07-2026
  time: <HH:MM of the moment you write the file, 24h>
  reference: Data Analytics und Big Data
  index: ""
  subindex: ""
  status:
    - begin
  ---
  ```
- Directly under the frontmatter: `# Einheit <N> – <Titel>`, then `>- **Reference Link:** [[Data Analytics und Big Data]]`, then `---`, then a `>[!summary]` callout (2–4 sentences, German, same dry/deadpan editorial voice as Einheit 1/2 — e.g. "Ein Modell kann nur so gut sein wie die Daten..." style asides are welcome, forced jokes are not).
- Body is organized as numbered `### N. <Titel>` sections, using the same recurring skeleton as Einheit 1/2 (adapt list to each chapter's actual content, do not invent sections that have no source material):
  1. Motivation/Einordnung ("Warum ist das wichtig?")
  2. Einordnung in QUA³CK (one paragraph/table linking this chapter to the U- or A³-Phase — Kapitel 1 → U/A³ Grundlagen, Kapitel 3 → A³ (Algorithm Selection), Kapitel 4 → A³ (Adjusting Hyperparameters / model internals))
  3. Lernziele der Einheit (bullet list)
  4..N. Content sections mirroring the book chapter's own section headers (list given per task below)
  N+1. Praktische Übung(en) zur Einheit — adapt the book's own "Übungen" into a portfolio-style task list, same format as Einheit 2 §32 (Aufgabe 1..k tables)
  N+2. Häufige Fehler und Best Practices (only if source material supports it)
  N+3. Zentrale Begriffe (glossary table: `| Begriff | Kurzdefinition |`)
  N+4. Merksätze (`>[!quote]` one-liners, 4–7 of them)
  N+5. Prüfungs- und Verständnisfragen (numbered list, adapt the book's own end-of-chapter "Übungen" questions into German comprehension questions — translate/rephrase, don't copy verbatim if already German, keep them German either way since the book is the German translation)
  N+6. Mini-Zusammenfassung (short prose recap + one closing `>[!important]`)
  Final: `### Aufgabe` block (`>[!important]` callout, portfolio-style task list) — same closing pattern as Einheit 1 and Einheit 2.
- Code blocks use fenced ` ```python ` / ` ```text ` and must be real, working snippets — prefer copying/adapting from the chapter's own companion code file (`5 - Notebook-Setup.txt` for Kapitel 3, `4 - Python-Code Kapitel 4.txt` for Kapitel 4) or from the extracted book text, not invented from scratch.
- Audio (`*.m4a` podcasts) and video (`*.mp4` Erklärvideos) files CANNOT be transcribed in this environment — no `whisper`/speech-to-text tool is installed (verified: only `pdftotext`, `exiftool`, `ffmpeg` available, no STT). Do not guess their spoken content. Each note must instead list them under a short "Zusatzmaterial (nicht automatisiert auswertbar)" note (one line, inside a `>[!note]` callout) giving the filename and the topic implied by its title, so the user knows what's still unreviewed by hand.
- Never modify or delete anything in the source Studium directory. Only create/edit files inside the `unfallatlas-qua3ck` repo.
- Do not commit unless the user asks; this plan's steps say "commit" per the repo's own convention (small, scoped commits) but hold off on `git push`.

---

## Source Material Map (already reconnoitred — reuse, don't re-derive)

Confirmed via `pdftotext -layout` extraction and `Slides.pdf`: the course textbook is Aurélien Géron, *Praxiseinstieg Machine Learning mit Scikit-Learn, Keras und TensorFlow* (German translation), course covers "Kapitel 1–9, Seiten 1–330". `Einheit 1` (QUA³CK) and `Einheit 2` (Understanding the Data) already cover the process-model framing and Kapitel 2 (End-to-End-Projekt / data understanding) content. The three raw, undocumented folders map 1:1 to the next three units:

| Raw folder | Book chapter | New note | Einheit-Titel |
|---|---|---|---|
| `Kapitel 1 ML-Grundlagen/` | Kapitel 1: Die Machine-Learning-Umgebung | `Einheit 3 – Die Machine-Learning-Umgebung.md` | Grundbegriffe, Arten von ML-Systemen, Herausforderungen, Testen/Validieren |
| `Kapitel 3 Klassifikation/` | Kapitel 3: Klassifikation | `Einheit 4 – Klassifikation.md` | MNIST, Konfusionsmatrix, Precision/Recall, ROC, Multiklassen/Multilabel/Multioutput |
| `Kapitel 4 Trainieren von Modellen/` | Kapitel 4: Trainieren von Modellen | `Einheit 5 – Trainieren von Modellen.md` | Lineare Regression, Gradientenverfahren, Regularisierung, Logistische/Softmax-Regression |

Per-folder file inventory (all paths relative to the source root above):

**Kapitel 1 ML-Grundlagen/**
- `1. Erklärvideo - Maschinelles_Lernen_erklärt.mp4` — video, not transcribable
- `2. Folien - Was-ist-Maschinelles-Lernen.pdf` — 60-page slide deck (image-heavy)
- `3. Folien - Machine-Learning-Praxisleitfaden.pdf` — slide deck
- `4. Podcast - Machine_Learning_Die_Landkarte_für_Praktiker.m4a` — audio, not transcribable
- `5. Machine Learning Kapitel 1 reduced.pdf` — **primary source**, the book chapter text itself
- `6. Die_Landkarte_des_Machine_Learning.pdf` — supplementary
- `7. Poster.png` — infographic image
- `8. Überblick über die ML-Themen.pdf` — short topic-map summary (already extracted, see Task 1)
- `Slides.pdf` — course syllabus/setup slide (already extracted, see Task 1)

**Kapitel 3 Klassifikation/**
- `1 - Erklärvideo - Wie_gut_ist_Ihre_KI_.mp4` — video, not transcribable
- `2 - Poster.png` — infographic image
- `3 - Folien - Klassifikation_Die_Suche_nach_der_Wahrheit_in_Daten.pdf` — 15-page slide deck
- `4 - ML-Klassifikation.pdf` — 70-page slide deck (image-heavy)
- `5 - Notebook-Setup.txt` — **primary code source**, MNIST setup code (already read, see Task 1)
- `6 - Podcast - Warum_95_Prozent_Genauigkeit_wertlos_sind.m4a` — audio, not transcribable
- `7 - Machine Learning Kapitel 3.pdf` — **primary source**, the book chapter text itself

**Kapitel 4 Trainieren von Modellen/**
- `1 - Erklärvideo - Training_von_ML-Modellen.mp4` — video, not transcribable
- `2 - Poster.png` — infographic image
- `3 - Folien - ML_Modelle_Black_Box_öffnen.pdf` — 15-page slide deck
- `4 - Python-Code Kapitel 4.txt` — **primary code source**, full worked notebook (already read, see Task 1)
- `5 - ML-Modelltraining.pdf` — 46-page slide deck (image-heavy)
- `6 - Podcast - Unter_der_Haube_von_Regression_und_Gradientenverfahren.m4a` — audio, not transcribable
- `7 - Machine Learning Kapitel 4.pdf` — **primary source**, the book chapter text itself

Confirmed section headers inside each primary chapter PDF (from `grep -n` on the `pdftotext -layout` output — use these as your section skeleton, do not reorder):

- **Kapitel 1**: Was ist Machine Learning? → Überwachtes Lernen → Unüberwachtes Lernen → Selbstüberwachtes Lernen → Reinforcement Learning → (Batch-Lernen / Onlinelernen, prose, no standalone heading matched but present) → Instanzbasiertes Lernen → Modellbasiertes Lernen und ein typischer ML-Workflow → Die größten Herausforderungen (Menge/Repräsentativität/Qualität der Daten) → Irrelevante Merkmale → Overfitting der Trainingsdaten → Underfitting der Trainingsdaten → Testen und Validieren (inkl. Train-Dev-Set, Datendiskrepanz, No-Free-Lunch-Theorem) → Übungen (19 questions).
- **Kapitel 3**: MNIST → Trainieren eines Binärklassifikators → Leistungsmessung → Konfusionsmatrix → Präzision und Sensitivität → Genauigkeit/Trefferquote-Kompromiss → Die ROC-Kurve → Multiklassenklassifikation → Fehleranalyse → Multilabel-Klassifikation → Multioutput-Klassifikation → Übungen (4 questions, incl. Titanic + Spamfilter mini-projects).
- **Kapitel 4**: Lineare Regression → Die Normalengleichung → Rechenkomplexität → Gradientenverfahren → Batch-Gradientenverfahren → Stochastisches Gradientenverfahren → Mini-Batch-Gradientenverfahren → Polynomielle Regression → Lernkurven → Regularisierte lineare Modelle → Ridge-Regression → Lasso-Regression → Elastic Net → Early Stopping → Logistische Regression → Entscheidungsgrenzen → Softmax-Regression → Übungen (12 questions).

---

### Task 1: Extract and stage source text for all three chapters

**Files:**
- Create (scratch, outside repo, not committed): `/tmp/course-extract/k1.txt`, `/tmp/course-extract/k3.txt`, `/tmp/course-extract/k4.txt`, `/tmp/course-extract/uebersicht.txt`, `/tmp/course-extract/slides1.txt`
- Read: the `.txt` code files listed above (Notebook-Setup.txt, Python-Code Kapitel 4.txt)
- Read (image, multimodal): the three `Poster.png` files, for a one-paragraph visual description to optionally reuse in each note's summary

**Interfaces:**
- Produces: three plain-text dumps of the primary chapter PDFs that Tasks 2–4 read from instead of re-running `pdftotext` each time.

- [ ] **Step 1: Create the scratch directory**

```bash
mkdir -p /tmp/course-extract
```

- [ ] **Step 2: Extract the three primary chapter PDFs with layout preserved**

```bash
SRC="/home/jonas/Documents/Studium/Semester_6/Data Analytics und Big Data"
pdftotext -layout "$SRC/Kapitel 1 ML-Grundlagen/5. Machine Learning Kapitel 1 reduced.pdf" /tmp/course-extract/k1.txt
pdftotext -layout "$SRC/Kapitel 3 Klassifikation/7 - Machine Learning Kapitel 3.pdf" /tmp/course-extract/k3.txt
pdftotext -layout "$SRC/Kapitel 4 Trainieren von Modellen/7 - Machine Learning Kapitel 4.pdf" /tmp/course-extract/k4.txt
pdftotext -layout "$SRC/Kapitel 1 ML-Grundlagen/8. Überblick über die ML-Themen.pdf" /tmp/course-extract/uebersicht.txt
```

- [ ] **Step 3: Verify extraction succeeded (line counts should roughly match)**

Run: `wc -l /tmp/course-extract/*.txt`
Expected: `k1.txt` ≈ 1726 lines, `k3.txt` ≈ 1345 lines, `k4.txt` ≈ 1989 lines, `uebersicht.txt` ≈ 190 lines. If any file is empty or near-zero, the PDF path was wrong — re-check the exact filename with `ls "$SRC/Kapitel 1 ML-Grundlagen"` (filenames contain German umlauts and mixed numbering styles, copy them exactly, don't retype).

- [ ] **Step 4: Read the two companion code files directly (they're small, no extraction needed)**

Read `Kapitel 3 Klassifikation/5 - Notebook-Setup.txt` and `Kapitel 4 Trainieren von Modellen/4 - Python-Code Kapitel 4.txt` with the Read tool. Keep their exact code available — Tasks 3 and 4 embed real snippets from these files rather than inventing new ones.

- [ ] **Step 5: No commit for this task** — scratch files live under `/tmp`, nothing in the repo changed yet.

---

### Task 2: Write `Einheit 3 – Die Machine-Learning-Umgebung.md`

**Files:**
- Create: `docs/course-material/Einheit 3 – Die Machine-Learning-Umgebung.md`
- Read: `/tmp/course-extract/k1.txt` (full — it's ~1726 lines but text-dense, not code-dense; read it in 2–3 chunks with the Read tool's `offset`/`limit`), `/tmp/course-extract/uebersicht.txt`
- Reference (style template, do not copy content): `docs/course-material/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md`, `docs/course-material/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md`

**Interfaces:**
- Consumes: section skeleton for Kapitel 1 from the "Source Material Map" above.
- Produces: a markdown file whose §-numbering and glossary Task 5 will link to from the index note.

- [ ] **Step 1: Draft the frontmatter, summary and Lernziele**

Use `title: Einheit 3 – Die Machine-Learning-Umgebung`, `description: Einheit 3`. Summary callout should state that this unit lays the conceptual foundation the QUA³CK A³-phase builds on: what ML *is*, which system types exist, and why models fail (bad data vs. bad algorithm) — in the same register as Einheit 2's "Ein Modell kann nur so gut sein wie die Daten" line.

Lernziele (derive from the chapter's own "Übungen" list at the end of k1.txt, restated as learning goals, not copied questions):
- ML in eigenen Worten definieren und von klassischer Programmierung abgrenzen können
- Beispielanwendungen für ML nennen können
- überwachtes, unüberwachtes, selbstüberwachtes und Reinforcement Learning unterscheiden können
- Batch- vs. Onlinelernen sowie instanzbasiertes vs. modellbasiertes Lernen erklären können
- die Hauptursachen für schlechte Modelle (schlechte Daten / schlechte Algorithmen) benennen können
- Overfitting und Underfitting erkennen und Gegenmaßnahmen nennen können
- Train/Validierung/Test/Train-Dev-Set korrekt einsetzen können
- das No-Free-Lunch-Theorem erklären können

- [ ] **Step 2: Write §1 "Warum braucht man ein Verständnis der ML-Umgebung?" and §2 "Einordnung in das QUA³CK-Modell"**

§2 must include a table like Einheit 2's, but note this chapter is *foundational* to both **U** (Datenverständnis prägt, welche ML-Systemart überhaupt sinnvoll ist) and **A³** (Algorithm Selection — die hier vorgestellten Lernarten sind die Bausteine der späteren Algorithmusauswahl). Cross-reference `[[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]]` and `[[Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung]]` using the same Obsidian `[[...]]` link style already used in the existing notes.

- [ ] **Step 3: Write the content sections following the Kapitel-1 skeleton**

One `### N. <Titel>` section per skeleton entry:
4. Was ist Machine Learning? (Definition, ML vs. klassische Programmierung)
5. Wozu Machine Learning? (Beispielanwendungen aus k1.txt, table format like Einheit 2 §4)
6. Arten von ML-Systemen im Überblick (table: Kriterium | Ausprägungen)
7. Überwachtes Lernen (Regression/Klassifikation, Beispielalgorithmen)
8. Unüberwachtes Lernen (Clustering, Dimensionsreduktion, Anomalieerkennung — cross-check against `uebersicht.txt`'s "Reise durch Machine Learning" topic map for the list of unsupervised techniques)
9. Selbstüberwachtes Lernen und Reinforcement Learning
10. Batch-Lernen vs. Onlinelernen (inkl. Out-of-Core-Lernen)
11. Instanzbasiertes vs. modellbasiertes Lernen (inkl. typischer ML-Workflow, Kostenfunktion)
12. Die größten Herausforderungen: schlechte Daten (Menge, Repräsentativität/Sampling Bias, Datenqualität, irrelevante Merkmale)
13. Die größten Herausforderungen: schlechte Algorithmen (Overfitting mit Gegenmaßnahmen-Tabelle, Underfitting mit Gegenmaßnahmen-Tabelle)
14. Testen und Validieren (Train/Test-Split, Validierungsdatensatz, Kreuzvalidierung, Train-Dev-Set bei Datendiskrepanz — use the flower-app example from k1.txt almost verbatim-in-spirit but paraphrased)
15. Das No-Free-Lunch-Theorem

Each section: prose paragraph(s) + at least one table or code block where the source has one; use `>[!note]`/`>[!important]`/`>[!tip]`/`>[!warning]` callouts matching where the book itself flags a caveat (e.g. Overfitting/Underfitting definitions get `>[!important]`, the No-Free-Lunch nuance gets `>[!note]`).

- [ ] **Step 4: Add the "Zusatzmaterial (nicht automatisiert auswertbar)" note**

Right after the Lernziele or before the Übung section, add:

```markdown
>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Maschinelles_Lernen_erklärt.mp4*) und ein Podcast (*Machine_Learning_Die_Landkarte_für_Praktiker.m4a*) sowie mehrere Foliensätze (*Was-ist-Maschinelles-Lernen.pdf*, *Machine-Learning-Praxisleitfaden.pdf*, *Die_Landkarte_des_Machine_Learning.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert — bei Bedarf manuell sichten.
```

- [ ] **Step 5: Write Praktische Übung, Häufige Fehler, Zentrale Begriffe, Merksätze, Prüfungsfragen, Mini-Zusammenfassung, Aufgabe**

- Praktische Übung: turn the book's 19 "Übungen" questions into a short practical task list (2–4 items) in the Einheit-2-§32 table style, e.g. "Wähle ein eigenes Beispiel für Batch- vs. Onlinelernen und begründe die Wahl", "Klassifiziere drei eigene Datensatz-Ideen nach überwacht/unüberwacht/RL".
- Zentrale Begriffe glossary must include at minimum: Feature, Label, Trainingsdatensatz, Testdatensatz, Validierungsdatensatz, Train-Dev-Set, Hyperparameter, Modellparameter, Overfitting, Underfitting, Sampling Bias, No-Free-Lunch-Theorem, Batch-Lernen, Onlinelernen, Out-of-Core-Lernen, instanzbasiertes Lernen, modellbasiertes Lernen.
- Prüfungsfragen: rephrase (don't copy verbatim) at least 10 of the book's 19 end-of-chapter questions.
- Mini-Zusammenfassung: 4–6 sentences.
- Closing `### Aufgabe` callout: one self-contained portfolio task, same tone as Einheit 1/2 closers (e.g. "Ordne dein eigenes QUA³CK-Projekt einer ML-Systemart zu und begründe, warum Overfitting/Underfitting dort ein Risiko ist").

- [ ] **Step 6: Verify topic coverage with grep**

Run:
```bash
cd "/home/jonas/Documents/Code/unfallatlas-qua3ck/docs/course-material"
grep -c -iE "Overfitting|Underfitting|Train-Dev|No-Free-Lunch|Onlinelernen|instanzbasiert|modellbasiert|Reinforcement" "Einheit 3 – Die Machine-Learning-Umgebung.md"
```
Expected: every one of those 8 terms found at least once (grep `-c` per term, or drop `-c` and eyeball with `grep -n`). If any term is missing, go back to Step 3 and add the missing section — don't skip it.

- [ ] **Step 7: Commit**

```bash
git add "docs/course-material/Einheit 3 – Die Machine-Learning-Umgebung.md"
git commit -m "docs: add Einheit 3 (Die Machine-Learning-Umgebung) course notes"
```

---

### Task 3: Write `Einheit 4 – Klassifikation.md`

**Files:**
- Create: `docs/course-material/Einheit 4 – Klassifikation.md`
- Read: `/tmp/course-extract/k3.txt` in chunks, `Kapitel 3 Klassifikation/5 - Notebook-Setup.txt` (already read in Task 1, reuse)
- Reference (style template only): the two existing Einheit notes

**Interfaces:**
- Consumes: Kapitel-3 section skeleton from the Source Material Map; the exact `Notebook-Setup.txt` imports (`fetch_openml`, `SGDClassifier`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `DummyClassifier`, `cross_val_score`, `cross_val_predict`, `StratifiedKFold`, `confusion_matrix`, `ConfusionMatrixDisplay`, `precision_score`, `recall_score`, `f1_score`, `precision_recall_curve`, `roc_curve`, `roc_auc_score`, `OneVsRestClassifier`, `OneVsOneClassifier`, `ClassifierChain`) for embedding real code.
- Produces: markdown file cross-linked from `Einheit 3` (§ QUA³CK-Einordnung) and the index note (Task 5).

- [ ] **Step 1: Draft frontmatter, summary, Lernziele**

`title: Einheit 4 – Klassifikation`, `description: Einheit 4`. Summary should note this unit is squarely the **A³ (Algorithm Selection)** phase in practice: picking and evaluating classifiers on MNIST, and why "Accuracy" alone is a trap — tie back to Einheit 1 §9's "Accuracy allein reicht nicht" point (cross-link `[[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]]`).

Lernziele, derived from k3.txt's own arc:
- den MNIST-Datensatz und seine Rolle als ML-"Hello World" einordnen können
- einen Binärklassifikator trainieren und mit Kreuzvalidierung bewerten können
- Konfusionsmatrix, Präzision, Sensitivität und F1-Score berechnen und interpretieren können
- den Precision/Recall-Trade-off und die ROC-Kurve erklären können
- Multiklassen-, Multilabel- und Multioutput-Klassifikation unterscheiden können
- eine einfache Fehleranalyse anhand der Konfusionsmatrix durchführen können

- [ ] **Step 2: Write §1 Motivation and §2 QUA³CK-Einordnung**

Explicitly contrast this chapter's classification metrics table against Einheit 1 §9's "Conclude & Compare" quantitative-criteria table — same metrics (Accuracy, Precision, Recall, F1), but Einheit 4 goes into *how* they're computed, not just that they exist.

- [ ] **Step 3: Write content sections following the Kapitel-3 skeleton**

4. MNIST: der Datensatz (mit dem `fetch_openml`-Snippet aus Notebook-Setup.txt)
5. Trainieren eines Binärklassifikators (SGDClassifier-Beispiel: "ist das eine 5 oder nicht")
6. Leistungsmessung mit Kreuzvalidierung (`cross_val_score`, Vergleich gegen `DummyClassifier` als Baseline — the setup file imports `DummyClassifier` specifically for this "naive Baseline schlägt oft naive Erwartungen" point)
7. Die Konfusionsmatrix (`confusion_matrix`, `ConfusionMatrixDisplay`, Aufbau der Matrix als Tabelle: TP/FP/FN/TN)
8. Präzision und Sensitivität (Formeln, `precision_score`/`recall_score`)
9. Der Präzision/Sensitivität-Kompromiss (`precision_recall_curve`, table of trade-off scenarios: Spamfilter will hohe Präzision, Überwachungssystem will hohe Sensitivität — same "hohe Relevanz vs. hohe Sensitivität" contrast the book draws around line 417 of k3.txt)
10. Die ROC-Kurve (`roc_curve`, `roc_auc_score`, Vergleich zu Precision/Recall-Kurve — wann welche nutzen)
11. Multiklassenklassifikation (`OneVsRestClassifier`, `OneVsOneClassifier`, SVM vs. SGD-Verhalten)
12. Fehleranalyse (Konfusionsmatrix normalisieren, welche Ziffernpaare verwechselt werden)
13. Multilabel-Klassifikation (mehrere Labels pro Instanz, `f1_score` mit `average` Parameter)
14. Multioutput-Klassifikation (`ClassifierChain`, Beispiel: Bildrauschen entfernen als Multioutput-Regression/Klassifikation)

Each section gets at least one real code snippet adapted from `Notebook-Setup.txt`'s imports, or a table (e.g. the Confusion-Matrix 2x2 table, the Precision/Recall trade-off scenarios table).

- [ ] **Step 4: Add the "Zusatzmaterial (nicht automatisiert auswertbar)" note**

```markdown
>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Wie_gut_ist_Ihre_KI_.mp4*), ein Podcast (*Warum_95_Prozent_Genauigkeit_wertlos_sind.m4a* — thematisch vermutlich der Precision/Recall-Trade-off) sowie zwei Foliensätze (*Klassifikation_Die_Suche_nach_der_Wahrheit_in_Daten.pdf*, *ML-Klassifikation.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert — bei Bedarf manuell sichten.
```

- [ ] **Step 5: Write Praktische Übung, Zentrale Begriffe, Merksätze, Prüfungsfragen, Mini-Zusammenfassung, Aufgabe**

- Praktische Übung: adapt the book's own 4 exercises (KNN-Hyperparametersuche >97%, Data Augmentation durch Pixel-Verschiebung, Titanic-Klassifikator, Spamfilter-Pipeline) into a portfolio task list — these map directly, no invention needed.
- Zentrale Begriffe must include: Konfusionsmatrix, Präzision (Precision), Sensitivität (Recall), F1-Score, ROC-Kurve, AUC, Multiklassenklassifikation, Multilabel-Klassifikation, Multioutput-Klassifikation, Baseline-Klassifikator, Data Augmentation.
- Prüfungsfragen: rephrase all 4 book exercises plus at least 6 conceptual comprehension questions derived from the sections above (e.g. "Warum ist Accuracy bei stark unbalancierten Klassen (z. B. 90% Nicht-5en) irreführend?").
- Aufgabe closer: portfolio task applying a confusion matrix + precision/recall analysis to the reader's own QUA³CK project dataset.

- [ ] **Step 6: Verify topic coverage with grep**

```bash
cd "/home/jonas/Documents/Code/unfallatlas-qua3ck/docs/course-material"
grep -c -iE "Konfusionsmatrix|Präzision|Sensitivität|ROC|Multiklassen|Multilabel|Multioutput|Fehleranalyse" "Einheit 4 – Klassifikation.md"
```
Expected: all 8 terms present. Missing terms → go back to Step 3.

- [ ] **Step 7: Commit**

```bash
git add "docs/course-material/Einheit 4 – Klassifikation.md"
git commit -m "docs: add Einheit 4 (Klassifikation) course notes"
```

---

### Task 4: Write `Einheit 5 – Trainieren von Modellen.md`

**Files:**
- Create: `docs/course-material/Einheit 5 – Trainieren von Modellen.md`
- Read: `/tmp/course-extract/k4.txt` in chunks, `Kapitel 4 Trainieren von Modellen/4 - Python-Code Kapitel 4.txt` (already read in Task 1, reuse — it is a complete, ordered, runnable notebook covering every section below)
- Reference (style template only): the two existing Einheit notes

**Interfaces:**
- Consumes: Kapitel-4 section skeleton from the Source Material Map; every numbered block (`1. DATENSATZ GENERIEREN` … `9. SOFTMAX-REGRESSION`) in `Python-Code Kapitel 4.txt` maps 1:1 to a content section below — reuse those exact code blocks (they are already tested/working, per the file's own comments).
- Produces: markdown file cross-linked from Einheit 1 §6 ("Big 3" models) and Einheit 4.

- [ ] **Step 1: Draft frontmatter, summary, Lernziele**

`title: Einheit 5 – Trainieren von Modellen`, `description: Einheit 5`. Summary: this unit opens the "Black Box" — after Einheit 3/4 treated models as things you call `.fit()`/`.predict()` on, this unit explains what training actually computes (Normalengleichung vs. Gradientenverfahren) and why regularization/early stopping matter for the Overfitting problem introduced in Einheit 3 §13. Cross-link `[[Einheit 3 – Die Machine-Learning-Umgebung]]`.

Lernziele:
- die Normalengleichung und ihre Rechenkomplexität erklären können
- Batch-, Stochastisches und Mini-Batch-Gradientenverfahren unterscheiden können
- polynomielle Regression und Lernkurven zur Diagnose von Über-/Unteranpassung nutzen können
- Ridge-, Lasso- und Elastic-Net-Regularisierung unterscheiden und begründet auswählen können
- Early Stopping als Regularisierungstechnik einordnen können
- logistische und Softmax-Regression für Klassifikationsaufgaben anwenden können

- [ ] **Step 2: Write §1 Motivation ("Die Black Box öffnen") and §2 QUA³CK-Einordnung**

This is the deepest **A³** dive (Adjusting Hyperparameters, e.g. `eta`/Lernrate, `alpha`/Regularisierungsstärke, `degree` bei Polynomfeatures) — table cross-referencing which hyperparameter from this chapter tunes which failure mode from Einheit 3 §13 (Overfitting → Regularisierung/Early Stopping; Underfitting → höherer Polynomgrad/weniger Regularisierung).

- [ ] **Step 3: Write content sections following the Kapitel-4 skeleton, embedding real code from `Python-Code Kapitel 4.txt`**

4. Lineare Regression: Grundidee und Kostenfunktion (MSE)
5. Die Normalengleichung (block "2. NORMALENGLEICHUNG" — `theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`) + Rechenkomplexität-Hinweis (O(n²) bis O(n³) in der Merkmalsanzahl, aus k4.txt Abschnitt "Rechenkomplexität")
6. Gradientenverfahren: Grundprinzip (Kostenfunktions-Minimierung, Lernrate `eta`)
7. Batch-Gradientenverfahren (block "3. BATCH-GRADIENTENVERFAHREN")
8. Stochastisches Gradientenverfahren (block "4. SGD: PRAKTISCHE IMPLEMENTIERUNG" inkl. `learning_schedule`)
9. Mini-Batch-Gradientenverfahren und SGD mit scikit-learn (block "5. SGD MIT SCIKIT-LEARN", `SGDRegressor`) + Vergleichstabelle Batch/Stochastic/Mini-Batch (Konvergenzverhalten, Rechenaufwand, Eignung für große Datensätze)
10. Polynomielle Regression (`PolynomialFeatures`, aus dem Early-Stopping-Block Zeile "PolynomialFeatures(degree=90...")
11. Lernkurven (block "6. LERNKURVENANALYSE", `learning_curve`) — Diagnose von Overfitting/Underfitting anhand der Kurvenform, direkt zurückverweisend auf Einheit 3 §13
12. Regularisierte lineare Modelle: Ridge-Regression (Formel, `alpha`-Parameter)
13. Lasso-Regression (Sparsity-Effekt, Feature-Selection-Nebenwirkung)
14. Elastic Net (Mischparameter zwischen Ridge und Lasso)
15. Early Stopping (block "7. EARLY STOPPING", vollständiges `partial_fit`-Loop-Beispiel mit `deepcopy(sgd_reg)` als "bestes Modell merken")
16. Logistische Regression (block "8. LOGISTISCHE REGRESSION (Iris)", `predict_proba`)
17. Entscheidungsgrenzen (Interpretation der Wahrscheinlichkeitsfunktion, Schwellenwert 0.5)
18. Softmax-Regression (block "9. SOFTMAX-REGRESSION (Iris, 3 Klassen)", `LogisticRegression(C=30, ...)` mit `multi_class` intern auf softmax)

Every section here has a directly-corresponding, already-written, working code block in `Python-Code Kapitel 4.txt` — copy it verbatim (it's the course's own reference implementation, no need to re-derive), and add 2–4 sentences of explanation per block plus a callout where the source PDF flags a caveat (e.g. Normalengleichung Rechenkomplexität → `>[!warning]`, Early Stopping "Champagner erst zum Sieg" style caution → `>[!tip]`).

- [ ] **Step 4: Add the "Zusatzmaterial (nicht automatisiert auswertbar)" note**

```markdown
>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Training_von_ML-Modellen.mp4*), ein Podcast (*Unter_der_Haube_von_Regression_und_Gradientenverfahren.m4a*) sowie ein Foliensatz (*ML_Modelle_Black_Box_öffnen.pdf*) und ein zusätzliches 46-seitiges Slide-Deck (*ML-Modelltraining.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert — bei Bedarf manuell sichten.
```

- [ ] **Step 5: Write Praktische Übung, Häufige Fehler, Zentrale Begriffe, Merksätze, Prüfungsfragen, Mini-Zusammenfassung, Aufgabe**

- Praktische Übung: adapt at least 4 of the book's 12 exercises (Skalierung bei Gradientenverfahren, Lernkurven-Diagnose bei polynomieller Regression, Ridge-Regularisierungsparameter-Tuning, Batch-GD-mit-Early-Stopping-für-Softmax ohne scikit-learn nur mit NumPy).
- Häufige Fehler: e.g. "Lernrate zu hoch gewählt → Divergenz", "Merkmale nicht skaliert vor Gradientenverfahren", "Early Stopping ohne Kopie des besten Modells (`deepcopy`) — man merkt sich sonst nur das letzte, nicht das beste Modell", "Regularisierung vor Skalierung" (mirror Einheit-2-§34 style: Problem/Besser table).
- Zentrale Begriffe: Kostenfunktion, Normalengleichung, Gradientenverfahren, Lernrate, Batch-/Stochastisches/Mini-Batch-Gradientenverfahren, Lernkurve, Ridge-Regression, Lasso-Regression, Elastic Net, Early Stopping, Logistische Regression, Softmax-Regression, Entscheidungsgrenze.
- Prüfungsfragen: rephrase at least 8 of the book's 12 exercise questions.
- Aufgabe closer: portfolio task implementing at least one regularized linear model with a learning-curve diagnosis on the reader's own project data.

- [ ] **Step 6: Verify topic coverage with grep**

```bash
cd "/home/jonas/Documents/Code/unfallatlas-qua3ck/docs/course-material"
grep -c -iE "Normalengleichung|Gradientenverfahren|Lernkurve|Ridge|Lasso|Elastic Net|Early Stopping|Softmax" "Einheit 5 – Trainieren von Modellen.md"
```
Expected: all 8 terms present. Missing → return to Step 3.

- [ ] **Step 7: Commit**

```bash
git add "docs/course-material/Einheit 5 – Trainieren von Modellen.md"
git commit -m "docs: add Einheit 5 (Trainieren von Modellen) course notes"
```

---

### Task 5: Wire the three new notes into the course index

**Files:**
- Modify: `docs/course-material/Data Analytics und Big Data.md`

**Interfaces:**
- Consumes: the exact note titles created in Tasks 2–4 (must match verbatim, including the en dash, for the `[[wikilink]]` to resolve in Obsidian).

- [ ] **Step 1: Read the current index file**

Read `docs/course-material/Data Analytics und Big Data.md` (already read once during planning — 19 lines, ends right after the Einheit 2 summary block).

- [ ] **Step 2: Append one summary block per new Einheit, same shape as the existing two**

Append after the Einheit 2 block:

```markdown
---
## [[Einheit 3 – Die Machine-Learning-Umgebung]]

>[!summary]
> Diese Einheit legt die Begriffsgrundlage für alle folgenden ML-Kapitel: Was Machine Learning ist, welche Arten von Lernsystemen es gibt (überwacht, unüberwacht, Reinforcement, Batch/Online, instanzbasiert/modellbasiert) und warum Modelle scheitern — an schlechten Daten oder an schlechten Algorithmen (Overfitting/Underfitting).

---
## [[Einheit 4 – Klassifikation]]

>[!summary]
> Diese Einheit vertieft die **A³-Phase** anhand von Klassifikationsaufgaben auf dem MNIST-Datensatz: Konfusionsmatrix, Präzision/Sensitivität, ROC-Kurve sowie Multiklassen-, Multilabel- und Multioutput-Klassifikation. Zeigt, warum Accuracy allein selten die richtige Metrik ist.

---
## [[Einheit 5 – Trainieren von Modellen]]

>[!summary]
> Diese Einheit öffnet die "Black Box" des Modelltrainings: Normalengleichung und Gradientenverfahren (Batch/Stochastic/Mini-Batch), Regularisierung (Ridge, Lasso, Elastic Net), Early Stopping sowie logistische und Softmax-Regression.
```

Use the Edit tool (not Write) to append this after the existing Einheit-2 block so nothing above it is disturbed.

- [ ] **Step 3: Verify all five Einheiten are now listed and links resolve to real filenames**

```bash
cd "/home/jonas/Documents/Code/unfallatlas-qua3ck/docs/course-material"
grep -n "^## \[\[Einheit" "Data Analytics und Big Data.md"
ls -1 | grep -c "^Einheit"
```
Expected: 5 `## [[Einheit` lines in the index, 5 `Einheit *.md` files on disk, and every bracketed title exactly matches an existing filename (drop the `.md` extension when comparing).

- [ ] **Step 4: Commit**

```bash
git add "docs/course-material/Data Analytics und Big Data.md"
git commit -m "docs: link Einheit 3-5 course notes from the course index"
```

---

## Self-Review Notes (already performed while drafting this plan)

- **Spec coverage:** every raw file in the three undocumented folders is accounted for — either as a primary text source feeding a specific numbered section (Tasks 2–4, Step 3), or explicitly flagged as an un-transcribable audio/video/slide asset (Tasks 2–4, Step 4). The index wiring the user implicitly wants (so the new notes are actually discoverable, matching the existing Dataview-driven index note) is covered by Task 5.
- **Placeholder scan:** every content section above names its exact source block/line range or companion-code section (e.g. "block '7. EARLY STOPPING'"), not a vague "write about early stopping" — an executor never has to guess what to write. Grep-based Step 6 checks are the concrete, non-optional acceptance test standing in for TDD assertions.
- **Type/name consistency:** all three new filenames (`Einheit 3 – Die Machine-Learning-Umgebung.md`, `Einheit 4 – Klassifikation.md`, `Einheit 5 – Trainieren von Modellen.md`) are used identically across Tasks 2–5, matching the frontmatter `title:` field and the wikilinks added in Task 5.
