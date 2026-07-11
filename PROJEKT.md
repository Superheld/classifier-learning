# PROJEKT.md — Stand & Ergebnisse: 10kGNAD

Alles Projektspezifische an einem Ort. Das Curriculum dazu: `ROADMAP.md`.

---

## 1 · Datensatz

| | |
|---|---|
| Name | 10kGNAD — Ten Thousand German News Articles Dataset |
| Quelle | github.com/tblock/10kGNAD (Artikel von „Der Standard") |
| Umfang | 10.273 Artikel · `train.csv` 9.245 · `test.csv` 1.028 |
| Kategorien (9) | Etat · Inland · International · Kultur · Panorama · Sport · Web · Wirtschaft · Wissenschaft |
| Split | stratifiziert (nachgeprüft) |
| Format ⚠️ | Semikolon-getrennt, Quotechar `'` (nicht `"`), kein Header — Einlesen via `gnad_utils.load_gnad()` |
| Ablage | `data/10kGNAD/` |

**Kategorie-Hinweise** (wichtig für Track D-Prompts und eigene Fehleranalyse):
„Etat" ist die **Medien-Rubrik** des Standard (nicht Haushalt/Finanzen!);
„Panorama" ist die Vermischtes-Rubrik und wirkt als „Abfluss" für Grenzfälle.

## 2 · Ordner-Übersetzung

Die Ordnernamen stammen aus einer früheren Zählung:

| Ordner | gehört zu |
|---|---|
| `stufe0_baseline/` | F1 (Baseline + EDA) |
| `stufe1_bag_of_words/` | Track A |
| `stufe2_evaluation/` | F2 |
| `stufe3_embeddings/` | Track B |
| `stufe4_finetuning/` | Track C |
| `stufe5_llm/` | Track D |
| `stufe6_vergleich/` | S1 |
| `stufe7_app/` | S2 |

## 3 · Fortschritt

| Einheit | Status | Ergebnis (test-Acc) |
|---|---|---|
| F1 Baseline | ✅ | **16,3 %** (Random: 11,3 %) |
| F1 EDA-Programm | ⬜ nachholen | — |
| F2 Messwesen | ✅ Basis (Confusion Matrix als PNG) | (Protokoll) |
| F2 Val-Split | ⬜ `load_gnad_val()` einbauen | — |
| F3 Experiment-Setup | ⬜ Seeds prüfen · `log_experiment()` · Config-Dict | — |
| F4 Theorie-Werkzeuge | ⬜ (parallel zu A·P3) | — |
| Track A · P1+P2 | ✅ | **85,2 %** (LogReg) |
| Track A · P3 | ⬜ 0 Runden | Ziel ~87–88 % |
| Track A · P4 | ◐ | — |
| Track B · P1+P2 | ✅ | 82,5 % naiv → **85,0 %** (Chunking) |
| Track B · P3 (inkl. SetFit) | ⬜ 0 Runden | Ziel >85,2 % |
| Track B · P4 | ◐ | — |
| Track C | ⬜ nächster Track | erwartet ~90–91 % |
| Track D | ⬜ | erwartet 85–92 % |
| E1 Regeln · E3 fastText · E4 Museum | ⬜ (Kür/Einschub) | E1-Erwartung ~50–60 % |
| E8 Kombinationen · S1 · S2 | ⬜ | — |

**▶ Nächster Schritt:** E0 abschließen — EDA-Checkliste, Val-Split,
`log_experiment()`. Danach E1 (kleiner Einschub) oder direkt A·P3.

## 4 · Messwerte im Detail

| Modell | test-Acc | Anmerkung |
|---|---|---|
| Random-Baseline | 11,3 % | |
| Majority-Baseline („Panorama") | 16,3 % | die Latte |
| Naive Bayes + rohe Counts | 84,3 % | mit TF-IDF nur 67 %! |
| **LogReg + TF-IDF** | **85,2 %** | Track-A-Referenz |
| Embeddings naiv (128 Tokens) | 82,5 % | Kontextfenster-Falle |
| Embeddings + Chunking | 85,0 % | gleichauf mit TF-IDF |
| BERT-Finetuning (erwartet) | ~90–91 % | vgl. Papers-with-Code-Leaderboard „10kGNAD" |
| LLM Zero-/Few-Shot (erwartet) | 85–92 % | je nach Prompt |

Messrauschen: bei n=1.028 ist das 95-%-KI grob **±2 Punkte** —
85,2 vs. 85,0 ist „gleichauf".

## 5 · Projekt-Lektionen (was dieser Datensatz uns beigebracht hat)

1. **Kontextfenster-Falle (Track B):** Das naive Embedding-Modell las nur
   128 Tokens — 97 % der Artikel wurden abgeschnitten. Mit EDA vorher wäre
   das aufgefallen. → Deshalb ist die EDA-Checkliste jetzt Pflicht in F1.
2. **Feature × Modell müssen passen (Track A):** Naive Bayes mit TF-IDF
   verlor 17 Punkte gegenüber rohen Counts.
3. **Trennschärfe schlägt Größe (F2):** Sport (F1 0,975) fast perfekt; Etat
   am schwächsten (Recall 0,63) — nicht weil klein, sondern weil unscharf.
   Wissenschaft ist klein UND scharf → glänzt. Praktische Obergrenze ~90 %
   (viele „Fehler" sind echte Grenzfälle).
4. **Low-Data-Stärke der Embeddings:** Bei nur 200 Trainingsbeispielen
   +21 Punkte gegenüber TF-IDF.
5. **Leakage in Zeitlupe (ehrliche Fußnote):** Frühe Vergleiche (min_df,
   Modellvergleich) liefen direkt am Testset — der klassische Fehler.
   Rangfolgen blieben robust, aber: Jedes P3 beginnt mit Val-Split und
   Nachprüfung der alten Entscheidungen.

## 6 · Setup

```bash
cd Klassifizierung
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Track D zusätzlich: `export ANTHROPIC_API_KEY=sk-ant-...`
- Optionale Pakete je Etappe: auskommentiert in `requirements.txt`
- Gemeinsame Helfer: `gnad_utils.py` (laden, Val-Split, `save_result`,
  künftig `log_experiment`) · Ergebnisse: `results.json` ·
  Experiment-Log: `experiments.csv` (ab F3-Setup)
