# KONZEPTE.md — Das Handwerks-Handbuch

**Zweck:** Das CURRICULUM sagt *wann*, dieses Handbuch sagt *was und wie*. Jedes
Konzept der drei Querschnitts-Stränge: Was es ist · warum es zählt · eine
konkrete Übung am 10kGNAD · wo es im Lernpfad lebt · Quelle. So tief, wie es
moderne Kurse (Made With ML, fast.ai, Google ML Crash Course, Ng) lehren.

Notation: 🟢 Einstieg (Pflicht früh) · 🟡 Mittelstufe (während der Tracks) ·
🔴 Vertiefung (wenn der Rest sitzt).

---

## Strang 1 — Daten-Handwerk

### 🟢 EDA — Explorative Datenanalyse
**Was:** Die Daten systematisch anschauen, *bevor* ein Modell sie sieht:
Verteilungen, Längen, Duplikate, Auffälligkeiten, Beispiele lesen.
**Warum:** Fast jede spätere Überraschung ist eine versäumte EDA-Frage.
Unser bestes Beispiel: 97 % der Artikel sind länger als 128 Tokens — hätte
die EDA gezeigt, *bevor* Track B daran stolperte.
**Übung:** Notebook/Script `eda.py`: Klassenverteilung train vs. test;
Token-Längen-Histogramm pro Klasse; Top-20-Wörter pro Klasse; exakte und
Fast-Duplikate (auch über den Split hinweg!); 10 Artikel pro Klasse lesen
und eigene „menschliche Merkmale" notieren.
**Wo:** F1. **Quelle:** Made With ML „Exploratory Data Analysis".

### 🟢 Datenqualität & Cleaning
**Was:** Encoding-Artefakte, HTML-Reste, Boilerplate (z. B. Agentur-Kürzel,
die die Klasse verraten!), leere/abgeschnittene Texte.
**Warum:** „Garbage in, garbage out" — und schlimmer: *Leckende* Artefakte
(ein Rubrik-Kürzel im Text) machen Modelle scheinbar gut und real nutzlos.
**Übung:** Suche im 10kGNAD nach Mustern, die die Kategorie direkt verraten
(z. B. Ortsmarken, Standard-Floskeln pro Rubrik). Diskutiere: drinlassen
oder entfernen — was entspricht dem echten Einsatz?
**Wo:** F1 → A·P1. **Quelle:** Made With ML „Preprocessing".

### 🟡 Labeling & Annotator-Übereinstimmung
**Was:** Wo kommen Labels her? Menschen annotieren nach einem **Guideline-
Dokument**; Qualität misst man mit Inter-Annotator-Agreement (**Cohens
Kappa**). Unser Datensatz kam fertig gelabelt — im echten Projekt ist das
Labeln oft die halbe Arbeit.
**Warum:** Die „Ground Truth" ist selbst fehlerbehaftet; Kappa < 0,8 heißt:
Die Kategorien sind unscharf definiert — kein Modell kann besser sein als
die Einigkeit der Menschen (unsere „praktische Obergrenze ~90 %" ist genau
das).
**Übung:** Labelt zu zweit (oder du + Claude) dieselben 50 Testartikel
blind; berechne Kappa. Wo seid ihr uneins — und sind das dieselben Klassen,
die auch das Modell verwechselt?
**Wo:** F2-Vertiefung / vor E7 (LLM-Labels brauchen dieselbe Prüfung!).
**Quelle:** SLP3 Kap. 4; Artstein & Poesio (Agreement).

### 🟡 Class Imbalance — Techniken
**Was:** `class_weight` (Fehler bei kleinen Klassen teurer machen),
Oversampling/Undersampling, SMOTE (synthetische Beispiele — bei Text
unüblich), Schwellen-Anpassung.
**Warum:** Etat (601 train) vs. Panorama (1.510) — der Recall-Unterschied
kommt direkt daher.
**Übung:** `class_weight="balanced"` vs. Oversampling der Etat-Klasse vs.
nichts — drei getrackte Experimente, Macro-F1 vergleichen.
**Wo:** A·P3. **Quelle:** scikit-learn „imbalanced" docs; imbalanced-learn.

### 🔴 Data Augmentation für Text
**Was:** Trainingsdaten künstlich vermehren: Back-Translation
(DE→EN→DE), Synonym-Ersetzung, zufälliges Löschen/Tauschen (EDA-Paper),
LLM-Paraphrasen.
**Warum:** Wirkt vor allem im Low-Data-Regime — verbindet sich mit der
Few-Shot-Kurve aus B·P3.
**Übung:** 200-Beispiele-Szenario: original vs. +LLM-Paraphrasen (2× Daten)
— wie viel holt Augmentation gegenüber echten Daten auf?
**Wo:** nach B·P3. **Quelle:** Wei & Zou 2019 („EDA"-Paper).

### 🔴 Data-centric AI
**Was:** Die Haltung: Bei fixem Modell die *Daten* verbessern (Label-Fehler
finden, unklare Fälle nachlabeln) schlägt oft Modell-Tuning. Werkzeug:
Confident Learning (cleanlab) findet wahrscheinliche Label-Fehler.
**Übung:** Lass cleanlab über train laufen; lies die 30 „verdächtigsten"
Labels — wie viele sind wirklich falsch? Neu trainieren ohne sie: Effekt?
**Wo:** nach S1 (wenn alle Modelle stehen, ist der Datensatz der nächste
Hebel). **Quelle:** Ng „Data-centric AI"; cleanlab.

---

## Strang 2 — Experimentier-Handwerk

### 🟢 Reproduzierbarkeit
**Was:** Gleicher Code + gleiche Daten + gleiche Seeds = gleiches Ergebnis.
Konkret: `random_state` überall (Split, Modell, Shuffle), Versionsstände
festhalten (Code via git, Datenstand notieren), Umgebung fixieren
(`requirements.txt` mit Versionen).
**Warum:** Ohne Reproduzierbarkeit ist jeder Vergleich zwischen zwei Läufen
bedeutungslos — man vergleicht Rauschen.
**Übung:** Lauf Track A zweimal ohne Seed — wie stark schwankt die Accuracy?
Dann mit Seed: identisch? Genau diese Differenz ist dein „Rausch-Boden".
**Wo:** F3 Teil 1, ab sofort überall. **Quelle:** Made With ML
„Reproducibility/Versioning"; REFORMS-Empfehlungen.

### 🟢 Experiment-Tracking
**Was:** Jeder Lauf wird protokolliert: Datum, Config, Code-Stand, Metriken,
Artefakte. Werkzeuge: **MLflow** (open source, lokal), Weights & Biases
(Cloud). Einstieg bei uns: `log_experiment()` → `experiments.csv`.
**Warum:** Nach 30 Läufen weiß niemand mehr, was Lauf 12 war. Das
Rundenprotokoll des Optimierungszyklus *ist* Experiment-Tracking von Hand —
das Werkzeug automatisiert es.
**Übung:** Rüste `gnad_utils.log_experiment(etappe, config, val, test,
notiz)` nach; migriere die bisherigen Ergebnisse hinein. Upgrade-Übung in
Track C: dieselben Läufe mit MLflow + UI.
**Wo:** F3 Teil 1; gelebt in jedem P3. **Quelle:** Made With ML „Experiment
Tracking" (MLflow).

### 🟢 Config-Management
**Was:** Alle Stellschrauben eines Laufs an einer benannten Stelle
(dict → YAML-Datei, später Hydra), nie als Magic Numbers im Code.
**Warum:** Ein Experiment ist erst dann eines, wenn seine Config vollständig
benennbar ist — sonst ist es nicht trackbar und nicht reproduzierbar.
**Übung:** Refaktoriere Track A: `config = {...}` ganz oben, Rest liest nur.
**Wo:** F3 Teil 1. **Quelle:** Made With ML „Scripting".

### 🟡 Hyperparameter-Suche — die Eskalationsleiter
**Was:** 1) Von Hand entlang der Fehleranalyse (unser Zyklus) → 2) **Grid
Search** (alles Kreuzprodukt; teuer, vollständig) → 3) **Random Search**
(überraschend effizient, Bergstra & Bengio) → 4) **Bayessche Suche/Optuna**
(lernt, wo es sich lohnt; Pruning bricht aussichtslose Läufe ab).
**Warum:** Die Leiter spart Größenordnungen an Rechenzeit — und zu wissen,
*wann* welche Stufe reicht, ist die eigentliche Kompetenz.
**Übung:** A·P3 mit GridSearchCV (Stufe 2); danach dieselbe Suche mit Optuna
(Stufe 4) — gleiche Qualität in wie viel weniger Läufen?
**Wo:** A·P3 (Grid), C·P3 (Optuna — da zählt jeder Lauf). **Quelle:**
scikit-learn Tuning-Guide; Optuna-Doku; Bergstra & Bengio 2012.

### 🟡 Ablation
**Was:** Rückwärts-Experiment: eine Komponente *entfernen* und messen, ob
sie wirklich beitrug. Standard in jedem Paper („Ablation Study").
**Warum:** Nach 5 Optimierungsrunden weißt du sonst nicht, welche der 5
Änderungen noch nötig sind — oft sind 2 davon totes Gewicht.
**Übung:** Nimm dein bestes A·P3-Modell und entferne die Zutaten einzeln
(n-Gramme raus, class_weight raus, …): Tabelle „ohne X → Val-Score".
**Wo:** Abschluss jedes P3. **Quelle:** übliche Paper-Praxis; ABLATOR.

### 🟡 Daten- & Modell-Tests
**Was:** Tests nicht nur für Code: **Daten-Tests** (Schema, Wertebereiche,
keine leeren Texte, keine Split-Duplikate — Great Expectations) und
**Verhaltens-Tests** für Modelle (CheckList: Invarianzen wie „Ortsnamen
tauschen ändert die Rubrik nicht", gerichtete Erwartungen, Minimal-Fälle).
**Warum:** Accuracy ist ein Durchschnitt — Verhaltens-Tests finden
systematische Löcher, die der Durchschnitt versteckt.
**Übung:** Schreibe 5 Invarianz-Tests für dein A-Modell (z. B. „Wien"→
„Graz" tauschen: bleibt die Vorhersage stabil?). Wie viele bestehen?
**Wo:** vor S2 (was in Produktion geht, wird getestet). **Quelle:** Made
With ML „Testing"; Ribeiro et al. 2020 (CheckList).

### 🔴 Versionierung von Daten & Modellen
**Was:** Git versioniert Code; **DVC** (o. ä.) versioniert Datenstände und
Modell-Artefakte; ein Modell-Register hält fest, welches Modell wo läuft.
**Übung:** Lege `results.json`, das beste Modell (`joblib`) und den
Datenstand als versionierte Artefakte ab; simuliere ein Rollback.
**Wo:** vor/in S2. **Quelle:** Made With ML „Versioning"; DVC-Doku.

---

## Strang 3 — Theorie-Werkzeuge

### 🟢 Bias/Variance & Overfitting-Diagnose
**Was:** Zwei Fehlerquellen: **Bias** (Modell zu simpel — lernt selbst train
nicht) und **Varianz** (Modell zu nervös — lernt train auswendig).
Diagnose-Signal: die Lücke train- vs. val-Score.
**Warum:** Bestimmt die *Richtung* jeder Optimierung: Bias → mächtigeres
Modell/bessere Features; Varianz → Regularisierung/mehr Daten. Wer das
verwechselt, optimiert rückwärts.
**Übung:** Baue die Diagnose in dein Rundenprotokoll ein: jede Runde
train- UND val-Score notieren; klassifiziere jede Runde als Bias- oder
Varianz-Problem.
**Wo:** F4, gelebt ab A·P3. **Quelle:** Google ML Crash Course; Ng ML
Yearning Kap. „Bias and Variance".

### 🟢 Learning Curves
**Was:** Score gegen Trainingsmenge (100/500/1.000/alle) für train und val.
Konvergieren beide auf niedrigem Niveau → Bias, mehr Daten helfen nicht.
Große Lücke, val steigt noch → mehr Daten helfen.
**Warum:** Beantwortet die teuerste Frage im ML: „Sollen wir mehr Daten
labeln?" — *bevor* man es tut. Die Few-Shot-Kurve (B·P3) ist eine Learning
Curve über zwei Modellfamilien.
**Übung:** `sklearn.model_selection.learning_curve` für Track A plotten und
in einem Satz beantworten: Lohnen 5.000 weitere gelabelte Artikel?
**Wo:** F4 → B·P3. **Quelle:** scikit-learn „Learning curves".

### 🟡 Loss-Funktionen & Gradientenabstieg (Intuition)
**Was:** Modelle lernen, indem sie eine **Fehlerfunktion** (Cross-Entropy
bei Klassifikation) schrittweise bergab laufen; die **Learning Rate** ist
die Schrittweite.
**Warum:** Erklärt Track C komplett: zu große Schritte = Divergenz, zu
kleine = ewiges Training; Warmup, Scheduler, Early Stopping sind alles
Steuerungen dieses Bergablaufs.
**Übung:** (Papier genügt) Skizziere, warum LR 5e-5 bei BERT oft kippt,
1e-5 kriecht — und verifiziere dann in C·P3 mit deinen drei Läufen.
**Wo:** F4 → C. **Quelle:** SLP3 Kap. 4 (Gradient Descent); HF Course.

### 🟡 Kalibrierung & Unsicherheit
**Was:** Konfidenzen sind roh oft „übermütig". **Reliability-Diagramm**
macht es sichtbar; `CalibratedClassifierCV` repariert es. Dazu: die
„unsicher"-Option als eigene Antwort.
**Warum:** Die S2-Kaskade steht und fällt mit ehrlichen Konfidenzen — ein
unkalibriertes „90 % sicher" schickt falsche Fälle am LLM vorbei.
**Übung:** Reliability-Diagramm für dein A-Modell vor/nach Kalibrierung;
wähle die Kaskaden-Schwelle datenbasiert statt gefühlt.
**Wo:** F4 → A·P3 → S2. **Quelle:** scikit-learn „Probability calibration".

### 🟡 Statistische Signifikanz
**Was:** Ist 87,1 % wirklich besser als 86,4 %? Werkzeuge: **Konfidenz-
intervall** (bei n=1.028: grob ±2 Punkte), **McNemar-Test** (vergleicht zwei
Modelle auf denselben Beispielen — viel schärfer als zwei getrennte
Intervalle), Bootstrap.
**Warum:** Ohne Signifikanz-Denken feiert man Rauschen als Fortschritt —
und wirft echte Verbesserungen weg.
**Übung:** McNemar-Test für dein bestes vs. zweitbestes Modell; wie viele
Testbeispiele entscheiden sie wirklich unterschiedlich?
**Wo:** F4 → S1. **Quelle:** Dietterich 1998; SLP3 Kap. 4 (statistical
significance).

### 🔴 Erklärbarkeit (XAI)
**Was:** *Warum* diese Vorhersage? Linear: Koeffizienten lesen (gratis!).
Modellagnostisch: **LIME/SHAP** (welche Wörter trieben die Entscheidung).
Transformer: Attention-Heatmaps (mit Vorsicht zu genießen).
**Warum:** Pflicht in regulierten Domänen; und das schnellste
Fehleranalyse-Werkzeug — „das Modell schaut auf das falsche Wort" sieht man
sofort.
**Übung:** SHAP für 5 falsch klassifizierte Artikel deines A-Modells — war
es ein verständlicher Irrtum?
**Wo:** nach A·P4, spannend erneut in C. **Quelle:** SHAP/LIME-Doku;
Molnar, „Interpretable ML" (frei).

---

## Wie die Stränge sich mit dem Lernpfad verweben

| Etappe | Daten-Handwerk | Experimentier-Handwerk | Theorie |
|---|---|---|---|
| E0 | **EDA**, Qualität | **Seeds, Tracking, Config** | Bias/Variance-Begriffe |
| E1 | (Regeln = Feature-Intuition) | erste getrackte Läufe | — |
| E2 (A) | Cleaning, **Imbalance** | **Grid Search, Ablation** | **Diagnose train/val, Learning Curve, Kalibrierung** |
| E3–E4 | — | Tracking weiter | Embedding-Geometrie |
| E5 (B) | Augmentation (🔴) | Encoder-Vergleiche sauber tracken | **Few-Shot-/Learning-Kurve** |
| E6 (C) | — | **MLflow-Upgrade, Optuna** | **Loss & LR, Early Stopping** |
| E7 (D) | **Labeling & Kappa** (LLM als Annotator prüfen!) | Prompt-Versionen tracken | Selbstkonsistenz = Unsicherheit |
| E8 | Destillation = Labeling | Ablation der Kaskade | — |
| E9 (S1) | — | alle Läufe aus dem Tracking | **Signifikanz (McNemar)** |
| E10 (S2) | Daten-Tests | **Modell-Tests, Versionierung** | Kalibrierte Schwellen |
| E11 | Data-centric AI, cleanlab | CI/CD, Monitoring | XAI vertieft |

---

**Quellen (Handwerks-Kanon):**
- Made With ML / MLOps Course (madewithml.com) — Design→Data→Model→Test→
  Reproducibility→Production; Vorbild für Strang 1+2
- Google Machine Learning Crash Course — Bias/Variance, Learning Curves
- Andrew Ng, ML Yearning + „Data-centric AI"
- fast.ai „Practical Deep Learning" — Top-down-Didaktik (erst bauen, dann
  Theorie: unser E-Pfad folgt demselben Prinzip)
- scikit-learn User Guide (Tuning, Calibration, Learning Curves)
- Ribeiro et al. 2020 „Beyond Accuracy: CheckList" · Dietterich 1998
  (Signifikanz) · Bergstra & Bengio 2012 (Random Search) · Wei & Zou 2019
  (Text-Augmentation) · Molnar „Interpretable ML"
