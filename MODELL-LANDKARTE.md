# Die vollständige Modell-Landkarte der Textklassifikation

**Zweck dieses Dokuments:** Das CURRICULUM lehrt bewusst *Familien* (vier Tracks),
nicht einzelne Modelle. Dieses Dokument zeigt das **komplette Feld** — jede
Familie, jede Variante, jede Kombination — damit sichtbar ist: Es wird nichts
versteckt, und die scheinbar „vielen Möglichkeiten" aus Blogartikeln und
Rankings falten sich fast alle in wenige Familien zusammen.

**Die Kernthese vorweg:** Artikel wie „Best NLP Models 2025" listen 20–30
Modellnamen. Das sind keine 30 Wege — es sind **7 Familien × 5 Varianten-
Achsen**. Wer die Familien und Achsen kennt, kann jedes neue Modell, das
nächstes Jahr erscheint, sofort einsortieren. Genau das ist der Unterschied
zwischen einer Liste und einer Landkarte.

---

## Teil 1 — Die 7 Familien (chronologisch = didaktisch)

Jede Familie beantwortet dieselbe Frage anders: **„Wie wird aus Text eine
Vorhersage?"**

### Familie 1 — Regeln & Heuristiken (vor-ML, ~seit immer)

Handgeschriebene Regeln: Keywords, reguläre Ausdrücke, Wortlisten,
Entscheidungsbäume aus Menschenhand.

| | |
|---|---|
| Vertreter | Keyword-Matching, Regex, Blocklisten |
| Stärken | 100 % erklärbar, keine Daten nötig, deterministisch |
| Schwächen | skaliert nicht, Wartungsalptraum, keine Generalisierung |
| Heute sinnvoll | triviale stabile Fälle; Vorfilter in Kaskaden |
| In der Roadmap | bewusst ausgelassen (didaktisch unergiebig); taucht als Vorfilter-Idee in S2 auf |

### Familie 2 — Klassisches ML auf Hand-Features (~1960er–heute) → **Track A**

Text wird von Hand zu Zahlen (Features), ein statistisches Modell lernt
Muster.

**Feature-Seite** (austauschbar): Bag-of-Words/Counts, **TF-IDF**,
Wort-n-Gramme, Zeichen-n-Gramme, handgebaute Features (Länge, Großschreibung,
Satzzeichen …).

**Modell-Seite** (austauschbar):

| Modell | Charakter | wann die richtige Wahl |
|---|---|---|
| **Naive Bayes** | Wahrscheinlichkeiten pro Wort; extrem schnell | Baseline, Echtzeit, kleine Daten |
| **Logistische Regression** | gewichtete Summe; robust, kalibrierbar | Standard-Startpunkt (unser A-Modell) |
| **SVM / LinearSVC** | optimale Trennebene; stark in hohen Dimensionen | kleine/mittlere Daten, oft 1–2 Punkte über LogReg |
| kNN | Ähnlichkeit zu Nachbarn | selten auf TF-IDF; besser auf Embeddings |
| Random Forest | viele Entscheidungsbäume | Text + numerische Zusatzfeatures gemischt |
| **Gradient Boosting** (XGBoost, LightGBM) | Bäume, die Fehler der Vorgänger korrigieren | tabellarische Features; auf reinem TF-IDF selten besser als LinearSVC |

**Merksatz:** Auf reinen Text-Features gewinnen die *linearen* Modelle
(LogReg, LinearSVC, NB) fast immer gegen die Baum-Modelle — Bäume glänzen
erst, wenn gemischte Features dazukommen. Deshalb behandelt Track A die
linearen; XGBoost & Co. sind eine **Stellschraube** (A·P3, Bereich Modell),
kein eigener Weg.

### Familie 3 — Statische Wort-Embeddings (2013–2017) → *fehlt in der Roadmap? Nein: Vorstufe von Track B*

Jedes **Wort** bekommt einen festen Bedeutungsvektor, gelernt aus riesigen
Korpora. Klassifikation: Wortvektoren mitteln → kleiner Classifier drauf.

| Vertreter | Besonderheit |
|---|---|
| **Word2Vec** (2013) | der Durchbruch: „König − Mann + Frau ≈ Königin" |
| **GloVe** (2014) | globale Kookkurrenz-Statistik |
| **fastText** (2016) | Subword-Einheiten → robust gegen Tippfehler/Komposita; **auch ein eigener, extrem schneller Classifier** (Facebook, Produktionsklassiker für Millionen Texte/Sekunde) |

„Statisch" = **ein** Vektor pro Wort, egal im welchem Satz: „Bank" (Geld) und
„Bank" (Park) sind derselbe Punkt. Genau diese Schwäche führte zu Familie 5/6.

**Einordnung:** Historische Vorstufe von Track B — Track B (Sentence-
Embeddings) ist die kontextuelle Weiterentwicklung und in fast allen Fällen
überlegen. **Ausnahme mit Praxiswert:** fastText als Classifier, wenn
Geschwindigkeit alles ist. → Optionaler Seitenpfad A+ (s. Teil 4).

### Familie 4 — Neuronale Sequenzmodelle (2014–2018) → *das „Museum": bewusst übersprungen*

Erste Deep-Learning-Welle: Netze lesen den Text als Sequenz, lernen Features
selbst (statt TF-IDF von Hand).

| Vertreter | Idee | historische Rolle |
|---|---|---|
| **CNN für Text** (Kim 2014) | Faltungsfilter erkennen lokale Wortmuster | schnell, gut für kurze Texte |
| **RNN / LSTM / GRU / BiLSTM** | Wort für Wort lesen, Gedächtnis über den Satz | lange Abhängigkeiten; jahrelang State of the Art |
| + Attention (2015–17) | „wichtige Wörter stärker gewichten" | die Idee, aus der der Transformer wurde |
| **ULMFiT, ELMo** (2018) | vortrainierte kontextuelle Sprachmodelle | die direkten BERT-Vorläufer |

**Warum die Roadmap sie überspringt:** 2026 sind sie für Textklassifikation
fast nie mehr die richtige Wahl — sie brauchen den Trainingsaufwand von
Track C, erreichen aber nur Qualität zwischen Track A und B. Ihre didaktische
Bedeutung (Features selbst lernen, Sequenz, Attention) steckt implizit in
Track B/C. Wer die Geschichte nachvollziehen will: SLP3 Kap. 6–8. →
Optionaler Seitenpfad „Museum" (Teil 4), lehrreich, aber Kür der Kür.

### Familie 5 — Transformer-Encoder finetunen (2018–heute) → **Track C**

Vortrainiertes Transformer-Modell (liest alle Wörter gleichzeitig,
kontextuell) wird komplett auf die Aufgabe nachtrainiert. **Hier wohnen die
meisten Namen aus den „Best Models"-Listen** — alles Varianten von BERT
entlang von fünf Achsen (Teil 2).

### Familie 6 — Sentence-Embeddings + kleiner Kopf (2019–heute) → **Track B**

Transformer wird **eingefroren** und liefert nur den Bedeutungsvektor ganzer
Texte; darauf ein simpler Classifier. Vertreter: **SBERT /
sentence-transformers**, Universal Sentence Encoder, kommerzielle
Embedding-APIs (OpenAI, Cohere, Voyage). Zwischenstufe: **SetFit** (Encoder
kontrastiv nachschärfen — halb B, halb C).

*(Chronologisch nach Familie 5 entstanden, praktisch die billigere Schwester —
in der Roadmap als Track B vor C einsortiert, weil konzeptionell einfacher.)*

### Familie 7 — LLMs benutzen (2020/2022–heute) → **Track D**

Generatives Großmodell, Aufgabe in natürlicher Sprache. Zwei Hebel am selben
Objekt — und das ist die Antwort auf die Frage „ist Finetuning nicht eine
Optimierung von Prompting?":

| Hebel | was sich ändert | Workflow |
|---|---|---|
| **Prompting** (Zero-/Few-Shot) | nur der **Input**; Gewichte unangetastet | Stellschrauben aus Sprache → Track D |
| **LLM-Finetuning** | die **Gewichte** (bei offenen LLMs via LoRA; bei APIs, wo angeboten) | Stellschrauben aus Mathematik → konzeptionell Track C, nur mit größerem Modell |

Prompting und Finetuning sind also **Geschwister, nicht Stufen** — zwei
grundverschiedene Eingriffsebenen am selben vortrainierten Modell. Man kann
sie kombinieren (finegetuntes LLM prompten), aber keins ist die „Optimierung"
des anderen. Dazu: In-Context Learning, Instruction-Tuning, strukturierte
Outputs.

---

## Teil 2 — Die 5 Varianten-Achsen (warum 30 Namen ≠ 30 Wege)

Jedes konkrete Modell aus den Rankings ist ein Punkt in diesem Raum.
Die Achsen sind **orthogonal** — sie kombinieren sich frei:

| Achse | Frage | Beispiele |
|---|---|---|
| **1 Architektur-Optimierung** | besser vortrainiert? | BERT → **RoBERTa** (besseres Training) → **DeBERTa** (bessere Attention) → **ELECTRA**, XLNet |
| **2 Größe / Kompression** | kleiner & schneller? | **DistilBERT** (97 % Leistung, 60 % Größe), TinyBERT, **ALBERT**, MobileBERT, MiniLM |
| **3 Sprache** | welche Sprache(n)? | Englisch-only → **gbert** (Deutsch), **mBERT**, **XLM-R** (100+ Sprachen) |
| **4 Domäne** | Fachgebiet-Vortraining? | **BioBERT**, FinBERT, LegalBERT, SciBERT, ClinicalBERT |
| **5 Kontextlänge** | wie lange Texte? | 512 Tokens Standard → **Longformer**, BigBird (4.096+) |

**Praktische Konsequenz:** „Welches Modell?" ist in Track B/C/D immer die
Stellschraube **Modellwahl** in P3 — man wählt pro Achse: *optimierte
Architektur? komprimiert? deutsch? Fachdomäne? lange Texte?* Unsere Wahl
`deepset/gbert-base` = BERT-Familie + Achse 3 (deutsch). Ein
„DistilRoBERTa-financial" wäre Achsen 1+2+4. Neue Modellnamen 2027 → gleiche
Achsen, gleiche Entscheidung.

---

## Teil 3 — Jedes Modell aus den zwei Artikeln, einsortiert

Quellen: vedanganalytics.com „Text Classification — Complete Guide",
mljourney.com „Best NLP Models 2025".

| Aus den Artikeln | Familie | In unserer Architektur |
|---|---|---|
| Tokenisierung, Lemmatisierung, Stoppwörter, Normalisierung | (Vorverarbeitung, keine Familie) | A·P1 bzw. Stellschrauben A·P3 |
| TF-IDF, CountVectorizer, n-Gramme, Zeichen-Features | Fam. 2 (Features) | A·P1 + A·P3 |
| Naive Bayes | Fam. 2 | A·P2 ✅ |
| Logistische Regression | Fam. 2 | A·P2 ✅ |
| SVM | Fam. 2 | A·P3 (LinearSVC-Schraube) |
| Random Forest | Fam. 2 | A·P3, nur bei Mischfeatures |
| XGBoost / LightGBM | Fam. 2 | A·P3, nur bei Mischfeatures |
| GridSearchCV, Cross-Validation | (Methode) | F3 + A·P3 |
| Oversampling/Undersampling (unbalancierte Klassen) | (Methode) | A·P3-Alternative zu `class_weight` |
| Word2Vec, GloVe | Fam. 3 | Vorstufe Track B (Museum) |
| fastText | Fam. 3 | Seitenpfad A+ (echter Produktionsklassiker) |
| CNN für Text | Fam. 4 | Museum |
| RNN, LSTM | Fam. 4 | Museum |
| BERT (Base/Large) | Fam. 5 | Track C, Modellwahl |
| RoBERTa | Fam. 5, Achse 1 | Track C, Modellwahl |
| DeBERTa | Fam. 5, Achse 1 | Track C, Modellwahl (Maximal-Accuracy) |
| XLNet, ELECTRA | Fam. 5, Achse 1 | Track C, Modellwahl |
| DistilBERT, TinyBERT, ALBERT, MobileBERT | Fam. 5, Achse 2 | Track C, Modellwahl (Effizienz) |
| mBERT, XLM-R, DistilmBERT | Fam. 5, Achse 3 | Track C, Modellwahl (unser gbert wohnt hier) |
| BioBERT, FinBERT, LegalBERT, SciBERT, ClinicalBERT | Fam. 5, Achse 4 | Track C, Modellwahl |
| Universal Sentence Encoder | Fam. 6 | Track B, Encoder-Wahl |
| Kontextuelle Embeddings (BERT als Feature-Extraktor) | Fam. 6 | Track B — genau unser Ansatz |
| GPT-4 & Co., Few-Shot, Prompt Engineering, In-Context Learning | Fam. 7 | Track D |
| Ensembles: Voting, Stacking, Bagging, Model Averaging | Kombination | Teil 5 / S2-Erweiterung |
| Metriken, Confusion Matrix, Stratified K-Fold | (Messwesen) | F2 ✅ |
| Hyperparameter-Tuning, Early Stopping, Regularisierung | (Methode) | F3 + P3 aller Tracks |
| Deployment, Monitoring, A/B-Tests | (Betrieb) | S2 (+ Ausblick Teil 6) |

**Bilanz:** Von ~30 genannten Techniken sind **27 bereits verortet** — als
Phase, Stellschraube oder Modellwahl. Echte Lücken der Roadmap waren nur:
fastText (→ Seitenpfad A+), CNN/LSTM/Word2Vec (→ Museum, bewusst) und
Ensembles (→ Teil 5).

---

## Teil 4 — Optionale Seitenpfade (für Vollständigkeit, nicht Pflicht)

### Seitenpfad A+ — fastText & Boosting (½ Tag)
fastText auf 10kGNAD trainieren (eine Zeile CLI!), gegen Track A messen:
Accuracy vs. Geschwindigkeit. Dann XGBoost auf TF-IDF — bestätigt sich, dass
linear auf Text gewinnt? **Lernwert:** das Kosten-Extrem kennenlernen;
Millionen-Texte-pro-Sekunde-Perspektive.

### Seitenpfad „Museum" — CNN/LSTM nachbauen (1–2 Tage, nur bei Interesse)
Ein BiLSTM mit Word2Vec-Embeddings (oder ein Text-CNN) auf 10kGNAD.
**Lernwert:** verstehen, *warum* der Transformer gewann — man spürt die
Mühsal (Padding, Sequenzlängen, langsames Training) und sieht das Ergebnis
zwischen Track A und B landen. Reines Geschichtsverständnis; keine
Produktionsempfehlung 2026.

---

## Teil 5 — Kombinationen (die Meta-Familie)

Alle Familien lassen sich kombinieren — so sehen echte Systeme aus:

- **Ensembles** (Voting/Stacking): mehrere Modelle stimmen ab; +0,5–1,5
  Punkte, ×n Kosten. Klassisches Wettbewerbs-Werkzeug (Kaggle), in Produktion
  seltener. Übung: A- und B-Modell per Soft-Voting kombinieren — schlägt es
  beide Einzelmodelle?
- **Kaskaden**: billiges Modell für klare Fälle, teures für unsichere —
  **unsere S2-App**.
- **Destillation**: großes Modell (D) labelt, kleines (A/B) lernt daraus —
  der Standardweg von „LLM-Prototyp" zu „billigem Produktionsmodell".
- **Regeln + ML**: Regex-Vorfilter vor dem Classifier (Familie 1 lebt weiter).

---

## Teil 6 — Was diese Karte bewusst NICHT abdeckt

Der Vollständigkeit halber, damit „alles" ehrlich bleibt — angrenzende
Gebiete, die eigene Lernprojekte wären:

**Andere Aufgabentypen:** Multi-Label (ein Text, mehrere Kategorien),
hierarchische Klassifikation, Extreme Classification (Millionen Labels),
Sequenz-Labeling (NER), Regression auf Text.
**Kein-Label-Szenarien:** Clustering/Topic Modeling (LDA, BERTopic) — wenn
Kategorien erst gefunden werden müssen.
**Multimodal:** Text + Bild/Audio.
**Betrieb im Großen:** Monitoring, Drift-Erkennung, A/B-Tests, aktives
Lernen (die klügsten Beispiele zum Labeln auswählen) — berührt S2, ist aber
ein eigenes Feld (MLOps).

---

## Fazit — Karte vs. Liste

Die Roadmap bleibt bei vier Tracks, und das ist kein Mangel an
Vollständigkeit, sondern ihre Stärke: **Die vier Tracks sind die vier
lebenden Familien** (2, 5, 6, 7). Familie 1 ist Vorfilter-Handwerk, Familie 3
und 4 sind Geschichte mit je einem Überlebenden (fastText) bzw. Erben
(Transformer). Alles andere — die 30 Namen der Rankings — sind Punkte auf
den fünf Varianten-Achsen und damit **Stellschrauben in P3**, keine neuen
Wege.

Wer die Roadmap durchläuft und diese Karte daneben legt, kann jeden
„Best Models 2027"-Artikel in fünf Minuten einsortieren. Das ist der
Unterschied zwischen „alle Modelle gesehen haben" und „das Feld verstehen".

---

**Quellen dieses Dokuments:**
- vedanganalytics.com — „Text Classification with NLP: A Complete Guide"
- mljourney.com — „Best NLP Models for Text Classification in 2025"
- Jurafsky & Martin, SLP3 (Kap. 4–7 für Familien 2–7)
- Joulin et al. 2016 (fastText); Kim 2014 (Text-CNN); Reimers & Gurevych 2019
  (SBERT); Tunstall et al. 2022 (SetFit)
