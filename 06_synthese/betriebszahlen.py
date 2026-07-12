"""
06_synthese — Betriebszahlen-Benchmark (Parameter · Latenz · Größe).

Accuracy ist nicht alles. Für die „welches Modell nehmen?"-Frage zählen im Betrieb
oft **Parameter** (Kapazität), **Latenz** (wie schnell pro Anfrage) und **Größe**
(was man ausliefert). Diese Zahlen misst man *fair* auf **einer** Maschine unter
gleichen Bedingungen — nicht verstreut über Trainingsläufe an verschiedenen Tagen.
Darum ein eigener Harness statt „mitschreiben".

Gemessen wird **end-to-end**: Rohtext → Intent (Encoder-Forward-Pass; der lineare Kopf
ist vernachlässigbar). Das ist die ehrliche Deploy-Latenz — nicht nur der Klassifikator-
Schritt (der würde den eingefrorenen Encoder unfair „gratis" aussehen lassen).

Start:  python 06_synthese/betriebszahlen.py
Ergebnis:  06_synthese/betriebszahlen.json  (liest das Dashboard, Tab „Betriebszahlen")
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from data_utils import load_banking77

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_WARMUP = 5      # erste Läufe verwerfen (Lazy-Init, Caches)
N_MEASURE = 30    # so viele Einzel-Anfragen timen, dann Median

# Die zu vermessenden Familien. params_mio = bekannte Kapazität aus dem Model-Card
# (bei den finegetunten wird sie live gezählt). result_key koppelt an results.json
# für die Accuracy-Spalte im Dashboard.
MODELS = [
    {"familie": "MiniLM (frozen)", "result_key": "B_minilm_logreg", "kind": "st",
     "name": "sentence-transformers/all-MiniLM-L6-v2", "params_mio": 22.7, "prefix": ""},
    {"familie": "mpnet (frozen)", "result_key": "B_mpnet_tuned", "kind": "st",
     "name": "sentence-transformers/all-mpnet-base-v2", "params_mio": 109.0, "prefix": ""},
    {"familie": "E5-base (frozen)", "result_key": "B_e5_logreg", "kind": "st",
     "name": "intfloat/e5-base-v2", "params_mio": 109.0, "prefix": "query: "},
    {"familie": "RoBERTa-ft (finetuned)", "result_key": "C_tuned_roberta", "kind": "hf_seq",
     "dir": str(ROOT / "modelle/roberta_ft"), "params_mio": None},
]


def _median_ms(fn, queries):
    """fn einmal je Query aufrufen, Median der Einzel-Latenzen in Millisekunden."""
    for q in queries[:N_WARMUP]:
        fn(q)
    times = []
    for q in queries[:N_MEASURE]:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def bench_st(spec, queries):
    """Sentence-Transformer (frozen Encoder): Latenz = encode() einer Einzel-Anfrage."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["name"], device=DEVICE)
    prefix = spec["prefix"]
    latenz = _median_ms(lambda q: model.encode([prefix + q], show_progress_bar=False), queries)
    params = spec["params_mio"]
    return latenz, params, round(params * 4, 0)  # fp32 ≈ 4 Byte/Param → grobe MB


def bench_hf_seq(spec, queries):
    """Finegetuntes HF-Klassifikationsnetz: tokenisieren + Forward-Pass, kein Grad."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["dir"])
    model = AutoModelForSequenceClassification.from_pretrained(spec["dir"]).to(DEVICE).eval()

    @torch.no_grad()
    def infer(q):
        inp = tok(q, return_tensors="pt", truncation=True).to(DEVICE)
        model(**inp)

    latenz = _median_ms(infer, queries)
    params = round(model.num_parameters() / 1e6, 1)
    size_mb = round(sum(f.stat().st_size for f in Path(spec["dir"]).rglob("*")) / 1e6, 0)
    return latenz, params, size_mb


def main():
    texts, _ = load_banking77("test")
    queries = list(texts[:N_MEASURE])  # echte Anfragen als Last
    print(f"Device: {DEVICE}  ·  {N_MEASURE} Einzel-Anfragen je Modell (Median)\n")

    out = []
    for spec in MODELS:
        try:
            if spec["kind"] == "st":
                latenz, params, size = bench_st(spec, queries)
            else:
                latenz, params, size = bench_hf_seq(spec, queries)
            row = {"familie": spec["familie"], "result_key": spec["result_key"],
                   "params_mio": params, "latenz_ms": latenz, "groesse_mb": size,
                   "device": DEVICE}
            print(f"  ✓ {spec['familie']:24s}  {params:>6} Mio  {latenz:>7} ms  {size:>6} MB")
        except Exception as e:  # ein Modell fehlt/lädt nicht → überspringen, Rest läuft
            row = {"familie": spec["familie"], "result_key": spec["result_key"],
                   "params_mio": spec["params_mio"], "latenz_ms": None,
                   "groesse_mb": None, "fehler": str(e)[:120]}
            print(f"  ✗ {spec['familie']:24s}  übersprungen: {str(e)[:60]}")
        out.append(row)

    dest = ROOT / "06_synthese" / "betriebszahlen.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {dest}")


if __name__ == "__main__":
    main()
