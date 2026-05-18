---
title: Data Analytics und Big Data
description: ""
date: 05-05-2026
time: 10:04
index: "[[Studium]]"
subindex: ""
category:
  - source
status:
  - begin
---

# Data Analytics und Big Dataw

>- **Index:** [[Studium]]
>- **Document Tags:**

## Notizen

```dataviewjs
const pages = dv.pages('"6 - Notes"')
  .where(p => p.reference == "Data Analytics und Big Data")
  .sort(p => p.file.name);

dv.table(["Notes", "Description", "Date", "Status"], 
  pages.map(p => [
    p.file.link,
    p.description,
    p.date,
    (() => {
      if (p.status) {
        if (p.status.includes("begin") && p.status.includes("finish")) {
          return "<span style='color:purple'>not Initialized</span>";
        } else if (p.status.includes("begin")) {
          return "<span style='color:red'>Begin</span>";
        } else if (p.status.includes("finish")) {
          return "<span style='color:green'>Finish</span>";
        }
      }
      return "<span style='color:purple'>not Initialized</span>";
    })()
  ])
)
```

---

## [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]]

>[!summary]
> Diese Einheit führt das **QUA³CK-Prozessmodell** als strukturierten Rahmen für Machine-Learning- und Data-Science-Projekte ein.
> 
> Der Fokus liegt darauf, ein ML-Projekt nicht nur technisch umzusetzen, sondern von der **Fragestellung** über die **Datenanalyse** und **Modellentwicklung** bis zur **produktiven Anwendung** systematisch zu planen.

---
## [[Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung]]

>[!summary]
> Diese Einheit vertieft die **U-Phase** des QUA³CK-Modells: **Understanding the Data**.
> Im Mittelpunkt steht die Frage, wie Rohdaten vor der eigentlichen Modellierung systematisch untersucht, bereinigt, visualisiert und vorbereitet werden. Genau hier entscheidet sich oft, ob ein Machine-Learning-Projekt später brauchbare Ergebnisse liefert oder nur hübsch scheitert, wie so viele digitale Hoffnungsprojekte der Menschheit.