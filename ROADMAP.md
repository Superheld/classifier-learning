# Lern-Roadmap: Textklassifikation

Das Curriculum — **datensatzunabhängig und wiederverwendbar**. Alles
Projektspezifische (Datensatz, Messwerte, Status, nächste Schritte) steht in
**`PROJEKT.md`**. Nachschlagewerke: **`MODELL-LANDKARTE.md`** (alle
Modellfamilien), **`KONZEPTE.md`** (Handwerks-Handbuch). Dashboard:
**`lernpfad.html`**.

---

## 1 · Die Architektur

```
┌───────────────────────── SYNTHESE ─────────────────────────┐
│   S1 Gesamtvergleich          S2 Mini-App (Deployment)     │
├──────────────────── STRATEGIE-TRACKS ──────────────────────┤
│  Track A        Track B        Track C        Track D      │
│  Klassisches ML Embeddings     Finetuning     LLM-Prompt   │
│                                                            │
│  jeder Track:  P1 Vorbereiten → P2 Bauen →                 │
│                P3 Optimieren ⟳ → P4 Einordnen              │
├─────────────────────── FUNDAMENT ──────────────────────────┤
│  F1 Daten & EDA   F2 Messwesen   F3 Experimentieren        │
│  F4 Theorie-Werkzeuge                                      │
└────────────────────────────────────────────────────────────┘
          + 3 Querschnitts-Stränge durch alle Ebenen
```

**Die drei Ebenen:**

- **Fundament** — Werkzeug für alle Tracks, kein „Modul unter anderen".
- **Tracks** — die vier lebenden Modellfamilien. Parallel, gleichrangig,
  identisches Innenschema (das Schema selbst ist Lernstoff).
- **Synthese** — sinnvoll, sobald mindestens zwei Tracks durch P3 sind.

**Standard-Begriffe vs. Projekt-Etiketten:**

| Unser Etikett | Fachbegriff (so heißt es draußen) |
|---|---|
| Track A | classical / traditional ML |
| Track B | feature extraction (frozen embeddings + classifier) |
| Track C | fine-tuning (pretrained transformer) |
| Track D | zero-/few-shot prompting |
| P1–P4, „Track", „Familie" | eigene Didaktik-Verpackung — draußen nicht verwenden |

**Die zwei klassischen Karten** stecken in der Architektur:

| Karte | beantwortet | lebt hier |
|---|---|---|
| CRISP-DM (Prozess) | *Wann* wähle ich ein Modell? | F1 = Data Understanding · P1 = Data Prep · P2/P3 = Modeling · F2/S1 = Evaluation · S2 = Deployment |
| Drei Generationen (Lehrbuch) | *Welches* Modell? | die Tracks (A=Gen 1, B=Gen 2, C/D=Gen 3) |

**Die drei Querschnitts-Stränge** (werden im Fundament eingeführt, in jeder
Etappe erneut geübt — Details je Konzept in `KONZEPTE.md`):

| Strang | Konzepte | eingeführt |
|---|---|---|
| 🔍 Daten-Handwerk | EDA · Cleaning · Labeling & Kappa · Imbalance · Augmentation · Data-centric AI | F1 |
| 🧪 Experimentier-Handwerk | Seeds · Tracking · Config · Hyperparameter-Suche · Ablation · Daten-/Modell-Tests | F3 |
| 📐 Theorie-Werkzeuge | Bias/Variance · Learning Curves · Loss & LR · Kalibrierung · Signifikanz · XAI | F4 |

---

## 2 · Fundament

### F1 — Daten & EDA (+ Baseline)

**Frage:** Was steckt in den Daten — bevor irgendein Modell sie sieht?

**EDA-Checkliste** (für jeden neuen Datensatz):

- [ ] Klassenverteilung (train vs. test — stratifiziert?)
- [ ] Textlängen-Verteilung pro Klasse (Histogramm; Median vs. Ausreißer)
- [ ] Häufigste/seltenste Wörter pro Klasse
- [ ] Duplikate & Fast-Duplikate — auch über den Split hinweg (= Leakage!)
- [ ] Encoding-Artefakte, Boilerplate, Muster, die das Label verraten
- [ ] Vokabular-Überlappung zwischen Klassen (Vorahnung der Verwechslungen)
- [ ] Ein Dutzend Beispiele pro Klasse *lesen* — was nutzt ein Mensch als Signal?

**Baseline danach:** Majority-Klasse tippen (`DummyClassifier`, eine Zeile) —
die Latte, die jedes Modell schlagen muss.

| | |
|---|---|
| Konzepte | EDA, Datenqualität, Split, Leakage, unbalancierte Klassen, Majority-/Random-Baseline |
| Werkzeug | pandas + matplotlib |
| Quelle | Made With ML „EDA"; Google Rules of ML, Regel 1 |
| ✓ Checkpoint | Drei Dinge nennen, die nur die EDA zeigen konnte — und welche Modell-Entscheidung jedes beeinflusst. Warum kann 95 % Accuracy nutzlos sein? |

### F2 — Messwesen

**Frage:** Warum lügt Accuracy allein — und womit misst man ehrlich?

| Werkzeug | zeigt |
|---|---|
| Confusion Matrix | *welche* Klasse mit *welcher* verwechselt wird |
| Precision (pro Klasse) | wie zuverlässig ein Tipp ist |
| Recall (pro Klasse) | wie vollständig eine Klasse gefunden wird |
| F1 / Macro vs. Weighted | Kompromiss; Macro schützt kleine Klassen |
| Fehleranalyse (Beispiele lesen) | *warum* — die Grundlage jedes P3 |

**Das Drei-Sets-Prinzip:**

| Set | Zweck | Regel |
|---|---|---|
| train | Modell lernt | — |
| validation | Varianten vergleichen (innere Schleife) | jede Tuning-Entscheidung hier |
| test | ehrliche Endmessung | zu; genau 2 Messungen pro Track (naive P2, finale P3) |

| | |
|---|---|
| Quelle | SLP3 Kap. 4; scikit-learn `classification_report` |
| ✓ Checkpoint | Modell tippt Klasse X nur bei absoluter Sicherheit — was ist hoch, Precision oder Recall, und warum? |

### F3 — Experimentieren: Setup + Optimierungszyklus

**Frage:** Wie verbessert man systematisch statt zu stochern — und wie zählt
jedes Experiment?

**Teil 1 — das Setup** (vor dem ersten Experiment einrichten):

| Baustein | bedeutet | Minimal-Variante |
|---|---|---|
| Reproduzierbarkeit | Seeds überall; Code/Daten/Umgebung fixiert | `random_state`, requirements mit Versionen |
| Experiment-Tracking | jeder Lauf protokolliert (Config, Metriken) | `experiments.csv` via Helper; Upgrade: MLflow |
| Config-Management | alle Parameter an *einer* benannten Stelle | Config-Dict oben im Script |
| Ablation | Zutat *weglassen* → trug sie wirklich bei? | Tabelle „ohne X → Val-Score" |

**Teil 2 — der Zyklus** (läuft in jedem P3; nur die Stellschrauben wechseln):

1. Messlineal fixieren (Metrik + Val-Set) — am Testset wird nicht gedreht
2. Ist-Stand messen (Val)
3. Fehleranalyse — falsch klassifizierte Val-Beispiele *lesen*
4. Hypothese — „verwechselt X mit Y, weil …"
5. Genau **eine** Änderung, die die Hypothese angreift
6. Neu messen; behalten oder verwerfen — beides ins Rundenprotokoll
7. Wiederholen bis Plateau (2–3 Runden ohne Verbesserung)
8. Erst jetzt: einmal aufs Testset → Ergebnis-Datei

**Eskalationsregel:** Track wechseln erst bei Plateau UND verfehltem Ziel —
ein ausgereizter einfacher Track schlägt oft einen naiven großen.

| | |
|---|---|
| Quelle | Ng, *ML Yearning*; Google Rules of ML Phase II; Made With ML „Tracking/Reproducibility" |
| ✓ Checkpoint | Warum EINE Änderung pro Runde? Kollege kann dein Ergebnis nicht reproduzieren — die drei wahrscheinlichsten Ursachen? |

### F4 — Theorie-Werkzeuge

**Frage:** *Warum* verhält sich ein Modell so — nicht nur *dass*?

| Werkzeug | Signal | Konsequenz |
|---|---|---|
| Bias/Variance | Lücke train- vs. val-Score | große Lücke → Regularisierung/mehr Daten; kleine Lücke, niedrig → mächtigeres Modell/Features |
| Learning Curves | Score vs. Trainingsmenge | zeigt, ob mehr Labeln sich lohnt |
| Loss & Gradientenabstieg | Lernen = Fehlerfunktion bergab | erklärt Learning-Rate-Empfindlichkeit (Track C) |
| Kalibrierung | „80 % sicher" stimmt in 80 %? | Reliability-Diagramm; Basis der S2-Kaskade |

| | |
|---|---|
| Quelle | Google ML Crash Course; SLP3 Kap. 4; scikit-learn „Calibration" |
| ✓ Checkpoint | train 99 % / val 87 % — Bias oder Varianz? Welche zwei P3-Schrauben adressieren genau das? |

---

## 3 · Das Track-Schema

| Phase | Frage | Charakter |
|---|---|---|
| **P1 Vorbereiten** | Wie wird Text zu Input für *diese* Strategie? | strategie-spezifische Data Prep |
| **P2 Bauen** | Was leistet ein Basismodell naiv? | einmalig; endet mit erster Testmessung |
| **P3 Optimieren ⟳** | Wie viel steckt wirklich drin? | Stellschrauben-Katalog + F3-Zyklus, n Runden, Protokoll |
| **P4 Einordnen** | Wann gewinnt dieser Track, wann verliert er? | Grenzen, Kostenprofil, Einsatzempfehlung |

Der P3-Katalog ist ein **Menü, keine Checkliste** — welche Schraube dran
ist, entscheidet die Fehleranalyse.

---

## 4 · Track A — Klassisches ML

| Steckbrief | |
|---|---|
| Fachbegriff | classical / traditional ML |
| Wissensquelle | deine gelabelten Daten |
| Modelle | **Logistische Regression** · Naive Bayes · LinearSVC/SVM · (Bäume: RF, XGBoost — nur bei Text+Zusatzfeatures) |
| Features | Counts · TF-IDF · Wort-/Zeichen-n-Gramme |
| Werkzeug | scikit-learn |
| Quellen | SLP3 Kap. 4 + Anhang B; scikit-learn „Working with Text Data" |

**P1 Vorbereiten:** Tokenisierung → Vokabular → TF-IDF-Gewichtung →
Sparse-Vektoren. ⚠️ Vectorizer nur auf train fitten (sonst Leakage).

**P2 Bauen:** NB und LogReg trainieren, erste Testmessung. Lektion:
Feature-Aufbereitung und Modell müssen zusammenpassen (NB mag rohe Counts).
*✓ Warum bekommt ein Füllwort niedriges, ein Fachwort hohes TF-IDF-Gewicht?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Features | n-Gramme (1,2) · Zeichen-n-Gramme · Stoppwörter/Lemma · min_df/max_df/sublinear_tf | Fehler an Wortformen/Komposita hängen |
| Modell | `C` · LinearSVC · `class_weight="balanced"` | Overfitting; kleine Klassen gehen unter |
| Konfidenz | Kalibrierung (`CalibratedClassifierCV`) | Kaskade (S2) braucht ehrliche Konfidenzen |
| Werkzeug | `Pipeline` + GridSearchCV / CV | systematisch statt stochern |

*✓ Warum hebt class_weight den Recall kleiner Klassen, und was opfert es?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| viel gelabelten Daten · wortlastigen Themen · Budget ~0 · Erklärbarkeits-Pflicht | Synonymen/Paraphrasen · wenig Daten · kurzen Texten |

→ **Schwäche, die zu Track B führt:** kennt keine Bedeutung — Synonyme sind
fremde Wörter.

---

## 5 · Track B — Embeddings

| Steckbrief | |
|---|---|
| Fachbegriff | feature extraction (frozen embeddings + classifier) |
| Wissensquelle | Sprachverständnis: vortrainiertes Modell · Kategorien: deine Daten |
| Modelle | Encoder (sentence-transformers: MiniLM/E5/GTE u. a.; Embedding-APIs) × Kopf (**LogReg** · kNN · SVM) · **SetFit** |
| Werkzeug | sentence-transformers, scikit-learn |
| Quellen | SLP3 Kap. 5; sbert.net; SetFit-Doku |

**P1 Vorbereiten:** Text → Bedeutungsvektor. ⚠️ **Kontextfenster**
(`max_seq_length`) prüfen — lange Texte brauchen Chunking (stückeln +
mitteln). Embeddings cachen.

**P2 Bauen:** Kopf auf eingefrorenen Embeddings, Testmessung. Typische
Lektion: naive Nutzung unterschätzt den Ansatz (Kontextfenster!); echte
Stärke im Low-Data-Regime.
*✓ In welchem Szenario schlagen Embeddings Bag-of-Words deutlich — und warum?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Encoder | Modellwahl: Kontextlänge · Dimension · Sprache | Texte abgeschnitten; Domäne speziell |
| Chunking | Mean-/Max-Pooling · Überlappung · Anfang+Ende | lange Texte |
| Kopf | LogReg-`C` · kNN · SVM | billigster Vergleich überhaupt |
| Datenregime | **Few-Shot-Kurve** (Score vs. Trainingsmenge, beide Familien) | wissen, *wo* Gen 2 gewinnt |
| Königsdisziplin | **SetFit** — Encoder kontrastiv nachschärfen | der eingefrorene Encoder limitiert (Brücke zu C) |

*✓ Few-Shot-Kurve zeichnen: ab welcher Datenmenge lohnt welcher Track?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| wenig Labels · kurzen Texten · Semantik/Synonymen · Embedding-Cache-Setups | langen Dokumenten ohne Chunking · rein wortlastigen Aufgaben |

→ **Schwäche, die zu Track C führt:** der Encoder bleibt eingefroren.

---

## 6 · Track C — Finetuning

| Steckbrief | |
|---|---|
| Fachbegriff | fine-tuning (pretrained transformer) |
| Wissensquelle | vortrainiertes Netz, komplett an die Aufgabe angepasst |
| Modelle | BERT-Familie über die 5 Achsen: RoBERTa/DeBERTa (Optimierung) · DistilBERT/ALBERT (Kompression) · XLM-R & sprachspezifische BERTs · Domänen-BERTs · Longformer (lange Texte) · LoRA/PEFT |
| Werkzeug | Hugging Face transformers · **GPU nötig** |
| Quellen | HF NLP Course „Fine-tuning"; `MODELL-LANDKARTE.md` Teil 2 |

**P1 Vorbereiten:** Subword-Tokenisierung (modelleigener Tokenizer),
`max_length` bewusst wählen, Padding/Batches, Klassifikations-Head.

**P2 Bauen:** erster Lauf mit Standardwerten — meist Bestwert-Kandidat, aber
um Größenordnungen teurer als Track A.
*✓ Unterschied „Kopf auf eingefrorenen Features" (B) vs. „Netz auftauen" (C)?
Warum braucht C eine GPU?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Training | Learning Rate + Warmup (empfindlichste Schraube!) · Epochen + **Early Stopping am Val-Set** · Batch Size | divergiert oder overfittet |
| Kapazität | Layer-Freezing · **LoRA/PEFT** | GPU-Budget knapp; kleine Datenmengen |
| Input | `max_length` (128/256/512) | lange Texte; Speicher vs. Kontext |
| Modell | Achsen-Wahl · Klassengewichte im Loss | Sprache/Domäne; kleine Klassen |

*✓ Warum Early Stopping am Val-Set — was passierte am Testset?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| Maximal-Accuracy · viel Labels · stabilen Kategorien · hohem Volumen | wenig Daten · wechselnden Kategorien · Erklärbarkeits-Pflicht · ohne GPU |

→ **Schwäche, die zu Track D führt:** Labels + GPU nötig; jede
Kategorie-Änderung = neu trainieren.

---

## 7 · Track D — LLM per Prompt

| Steckbrief | |
|---|---|
| Fachbegriff | zero-/few-shot prompting |
| Wissensquelle | Sprach- & Weltwissen: LLM · Kategorien: deine *Beschreibung* |
| Modelle | Claude Haiku (billig/schnell) · Claude Sonnet (stärker) · jedes Instruct-LLM |
| Werkzeug | LLM-API + strukturierter JSON-Output |
| Quellen | Anthropic Classification-Guide; HF LLM Course |
| Rolle | Gegenprobe zum Finetuning: War der Aufwand es wert? (Wer streng „Baseline zuerst" lernen will: D vor C — beide Reihenfolgen gehen.) |

**P1 Vorbereiten — „Data Prep ist Prompt-Design":** Kategoriedefinitionen
schreiben (das eigentliche „Training"!), JSON-Output erzwingen,
Antwort-Validierung + Retries, festes Val-Sample definieren.

**P2 Bauen:** Zero-Shot auf einer Stichprobe, messen — ohne ein einziges
Trainingsbeispiel.
*✓ Zwei Situationen, in denen Prompting ein finegetuntes Modell schlägt,
obwohl es „schwächer" misst?*

**P3 Optimieren — Stellschrauben** (Stellrad ist Sprache, Zyklus identisch):

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Definitionen | fehleranalysegetrieben schärfen · Grenzfall-Regeln | systematische Verwechslungen |
| Beispiele | Few-Shot: zufällig vs. gezielte Grenzfälle | Definitionen reichen nicht |
| Modell | klein vs. groß — Kostenkurve mitzeichnen | Qualität vs. Preis |
| Robustheit | Konfidenz + „unsicher"-Option · Selbstkonsistenz (3×, Mehrheit) | Grenzfälle |

*✓ Am eigenen Protokoll zeigen: Prompt-Iteration = Hyperparameter-Tuning,
nur das Stellrad ist anders.*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| null Labels · wechselnden Kategorien · Begründungs-Bedarf · kleinen/mittleren Volumen | Massenvolumen (laufende Kosten) · harter Latenz · Offline-Pflicht |

→ **Schwäche, die zur Synthese führt:** Kosten/Latenz — Kombinationen lösen es.

---

## 8 · Synthese

### S1 — Gesamtvergleich

- Alle Testmessungen in **einer Tabelle**, ergänzt um die P4-Dimensionen:
  Labels nötig · Kosten/1.000 Texte · Setup-Aufwand · Erklärbarkeit ·
  „neue Kategorie hinzufügen"
- **Metrik-Rückblick** (schließt den CRISP-DM-Kreis): War die Leitmetrik
  richtig? Alles zusätzlich mit Macro-F1 — ändert sich die Rangfolge?
- **Messrauschen:** Konfidenzintervall der Testmenge kennen; Unterschiede
  darunter sind Rauschen. Schärfer: McNemar-Test (`KONZEPTE.md`).
- *✓ Welche zwei Zeilen unterscheiden sich NICHT signifikant — und was heißt
  das für die Wahl?*

### S2 — Mini-App: Hybrid-Classifier (Deployment)

- **Kaskade:** billiger Track (A/B, **kalibrierte** Konfidenz) entscheidet
  klare Fälle; unsichere gehen ans LLM (D) mit Begründung.
- Anzeige: Kategorie · Konfidenz · Entscheidungsweg · Begründung.
- Konzepte: Konfidenz-Schwellen · Kosten-Qualitäts-Tradeoff · Modell
  speichern/laden · UI (Streamlit/CLI) · Daten-/Modell-Tests vor Deployment.
- Die App *ist* die Antwort auf S1: kein Gewinner — jeder Track für die
  Fälle, die er am besten/billigsten kann.

---

## 9 · Der Lernpfad — 12 Etappen

**Reihenfolge = Geschichte des Fachs**: Jede Etappe behebt die Schwäche der
vorherigen. Kern baut aufeinander auf; Kür vertieft, ohne dass Späteres
davon abhängt.

| # | Etappe | Familie | Art | Aufwand | Kern-Übung |
|---|---|---|---|---|---|
| E0 | Fundament (F1–F4) | — | Kern | 1–2 Tage | EDA-Checkliste · Val-Split · Experiment-Log |
| E1 | Regeln & Keywords | 1 | Kern (klein) | 2 Std. | Keyword-Classifier von Hand — bis es wehtut |
| E2 | Klassisches ML = Track A | 2 | Kern | 1–2 Tage | P1–P4 + mind. 3 P3-Runden |
| E3 | Word2Vec & fastText | 3 | Kür | ½–1 Tag | Analogien spielen · fastText vs. Track A |
| E4 | Museum: CNN/LSTM | 4 | Kür | 1–2 Tage | BiLSTM nachbauen — spüren, warum Transformer gewann |
| E5 | Embeddings = Track B | 6 | Kern | 1–2 Tage | P1–P4 + Few-Shot-Kurve + SetFit |
| E6 | Finetuning = Track C | 5 | Kern | 1–2 Tage | P1–P4 + LR-Dreier + LoRA |
| E7 | LLM = Track D | 7 | Kern | 1 Tag | P1–P4 + 3 Prompt-Versionen + Kostenrechnung |
| E8 | Kombinationen | Meta | Kern | 1 Tag | Ensemble (Soft-Voting) · Destillation (LLM labelt → klein lernt) |
| E9 | Gesamtvergleich = S1 | — | Kern | ½ Tag | Vergleichstabelle · Macro-F1 · Signifikanz |
| E10 | Mini-App = S2 | — | Kern | 1–2 Tage | Kaskade als Streamlit/CLI |
| E11 | Ausblick | — | Kür | offen | Multi-Label · Topic Modeling · Active Learning · MLOps |

*Nummerierungs-Hinweis: „Familie" (1–7) = Taxonomie aus `MODELL-LANDKARTE.md`;
Gen-Labels = Lehrbuch-Zoomstufe. Track B = Gen 2 = Familie 6 — gleiche Sache.*

**Der rote Faden (Schwäche → nächste Etappe):**
Raten (E0) → Regeln skalieren nicht (E1) → ML kennt keine Bedeutung (E2) →
statische Vektoren kennen keinen Kontext (E3) → Sequenzmodelle sind mühsam
(E4) → eingefrorene Encoder passen sich nicht an (E5) → Finetuning ist teuer
und starr (E6) → Prompting kostet laufend (E7) → Kombinationen (E8) →
Vergleich (E9) → Produkt (E10).

---

## 10 · Leitplanken

1. **Erst die Baseline, dann eskalieren.**
2. **Test-Split bleibt zu** — Varianten vergleicht man am Val-Set.
3. **Metrik vor Modell** — erst definieren, was „gut" heißt.
4. **Fehleranalyse schlägt Modellwechsel.**
5. **Kosten & Änderbarkeit zählen, nicht nur Accuracy.**
6. **Innere vor äußerer Schleife** — erst den Track ausreizen (P3).
7. **Eine Änderung pro Messung.**

---

## 11 · Quellen

| Quelle | wofür | wo |
|---|---|---|
| Jurafsky & Martin, SLP3 | Lehrbuch-Kanon (Kap. 2, 4, 5, 7) | web.stanford.edu/~jurafsky/slp3 |
| Stanford CS224N | der kanonische NLP-Kurs | web.stanford.edu/class/cs224n |
| Hugging Face NLP/LLM Course | Praxis Track C/D | huggingface.co/learn |
| Made With ML (MLOps Course) | Handwerks-Stränge, EDA, Tracking | madewithml.com |
| Ng, ML Yearning | Fehleranalyse, Val/Test-Disziplin (F3) | frei verfügbar |
| Google, Rules of ML | Baseline-first, Iteration | developers.google.com/machine-learning/guides/rules-of-ml |
| Google ML Crash Course | Bias/Variance, Learning Curves (F4) | developers.google.com/machine-learning/crash-course |
| scikit-learn Guides | Track A, Tuning, Kalibrierung | scikit-learn.org |
| sbert.net + SetFit | Track B | sbert.net · huggingface.co/docs/setfit |
| Anthropic Classification-Guide | Track D | docs.claude.com |
| CRISP-DM | Prozessmodell | Suchbegriff „CRISP-DM" |

---

## 12 · Glossar

**Accuracy** – Anteil korrekter Vorhersagen. **Baseline** – trivialer
Vergleichswert. **CRISP-DM** – Standard-Prozessmodell für ML-Projekte.
**Data Leakage** – Testinformation sickert ins Training. **Early Stopping** –
Training beenden, sobald die Val-Leistung nicht mehr steigt. **EDA** –
explorative Datenanalyse. **Embedding** – Text als Bedeutungsvektor.
**Epoche** – ein Durchlauf durch die Trainingsdaten. **F1** – harmonisches
Mittel aus Precision und Recall. **Few-Shot** – Beispiele im Prompt statt
Training. **Finetuning** – vortrainiertes Netz nachtrainieren.
**Grid Search** – systematisches Durchprobieren von Hyperparametern.
**Ground Truth** – die „wahren" Labels. **Hyperparameter** – Einstellungen,
die *du* wählst. **Kalibrierung** – Konfidenzen ehrlich machen.
**Kontextfenster** – wie viele Tokens ein Modell maximal liest.
**Kreuzvalidierung** – k-faches Rotieren des Val-Splits. **LoRA/PEFT** –
Finetuning light. **n-Gramm** – Folge von n Wörtern/Zeichen als Feature.
**Overfitting** – auswendig gelernt statt Muster. **Precision/Recall** –
Zuverlässigkeit eines Tipps / Vollständigkeit des Findens. **SetFit** –
Encoder mit wenigen Beispielen kontrastiv nachschärfen. **Stratifizierung** –
Split mit gleichen Klassenanteilen. **TF-IDF** – häufig im Dokument, selten
im Korpus = wichtig. **Transfer Learning** – fremdes Vorwissen nutzen.
**Validation-Set** – drittes Set für Tuning-Entscheidungen. **Zero-Shot** –
klassifizieren nur aus Kategoriebeschreibungen.
