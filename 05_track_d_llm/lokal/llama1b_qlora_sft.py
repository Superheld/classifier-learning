# %% [markdown]
# # Track D · lokal — Llama-3.2-1B, QLoRA-SFT (banking77) · parametrisiert
#
# ## Finetuning-Experimente sauber getrennt + Dashboard-getrackt
#
# Der Bogen bisher, dasselbe Llama-1B: zero-shot ~9 % → few-shot ~32 % →
# QLoRA-SFT (600 iters) ~84 %. Jetzt wollen wir **länger/anders trainieren** und
# Läufe *vergleichen*, ohne dass sich Ergebnisse gegenseitig überschreiben.
#
# **Der Trick: eine `RUN`-Variable.** Sie läuft durch Modellname, Cache-Tags und
# den `results.json`-Eintrag — so ist jeder Lauf ein eigenes oMLX-Modell, ein eigener
# Cache und eine eigene Scoreboard-Zeile. Kein „stale 84 %" mehr, und du kannst
# `600it` vs. `3ep` vs. `dora` direkt nebeneinander sehen.
#
# **Tracking:** `evaluate_and_save` schreibt (wie in jedem Track) `results.json` +
# `predictions/<name>.json` → das Dashboard (`06_synthese/`) liest beides automatisch
# (Scoreboard + Fehler-Deep-Dive). Die Hyperparameter geben wir als Extra-Spalten mit.
#
# **Mechanik:** Notebook läuft im Haupt-venv (Prep + Messung über oMLX); Training/Fuse
# per Subprozess in `.venv-mlx`. Den Trainingslauf fährst du.

# %% [markdown]
# ## Setup

# %%
import subprocess
import sys
from pathlib import Path

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except (NameError, AttributeError):
    pass

root = Path.cwd()
while not (root / "data_utils.py").exists() and root != root.parent:
    root = root.parent
for d in (root, root / "05_track_d_llm"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from data_utils import load_banking77, load_banking77_split
from llm_client import get_client, classify, stratified_subsample

# %% [markdown]
# ## Konfiguration — alle Stellschrauben an *einer* Stelle
#
# `RUN` ist der Ein-Zeilen-Wechsel je Experiment. Alles andere leitet sich daraus ab.

# %%
RUN = "3ep"                 # ← EINDEUTIGER Name je Experiment (z.B. "600it", "3ep", "dora", "r16")
ITERS = 2500                # Trainingsschritte (2500 × Batch 8 / 8340 ≈ 2,4 Epochen)
BATCH = 8
NUM_LAYERS = 8              # LoRA auf den oberen N Layern (Llama-1B hat 16)
LR = 1e-4
FINE_TUNE_TYPE = "lora"     # "lora" | "dora" | "full" — mlx-lm kann alle drei

# abgeleitet (nicht anfassen):
MLX_PY = root / ".venv-mlx" / "bin" / "python"
BASE_MODEL = Path.home() / ".omlx" / "models" / "Llama-3.2-1B-Instruct-4bit"
HERE = root / "05_track_d_llm" / "lokal"
DATA_DIR = HERE / "ft_data"                       # train/valid.jsonl (laufunabhängig)
ADAPTER_DIR = HERE / f"ft_adapters_{RUN}"         # Adapter je Lauf getrennt
FT_MODEL = f"Llama-1B-banking77-{RUN}"            # so heißt es in oMLX
FUSED_DIR = Path.home() / ".omlx" / "models" / FT_MODEL
RESULT_NAME = f"D_llama1b_ft_{RUN}"              # Scoreboard-Schlüssel
print(f"RUN={RUN}  →  Modell '{FT_MODEL}',  Scoreboard '{RESULT_NAME}'")

# %% [markdown]
# ## Schritt 1 — Split mit Holdout
#
# 2 Beispiele je Klasse (154) aus train **rausnehmen** (few-shot-Pool, im Training
# ungesehen). Finetunt wird auf dem Rest; `val` überwacht; `test` bleibt final.

# %%
tr_t, tr_l, val_t, val_l = load_banking77_split()
test_t, test_l = load_banking77("test")
labels = sorted(set(tr_l))

demo_t, demo_l = stratified_subsample(tr_t, tr_l, per_class=2, seed=42)
demo_set = set(zip(demo_t, demo_l))
ft_t = [t for t, l in zip(tr_t, tr_l) if (t, l) not in demo_set]
ft_l = [l for t, l in zip(tr_t, tr_l) if (t, l) not in demo_set]
epochs = round(ITERS * BATCH / len(ft_t), 2)
print(f"Finetuning: {len(ft_t)}   Holdout: {len(demo_t)}   val: {len(val_t)}   test: {len(test_t)}")
print(f"{ITERS} iters × Batch {BATCH} / {len(ft_t)} ≈ {epochs} Epochen")

# %% [markdown]
# ## Schritt 2 — Trainingsdaten als mlx-lm-jsonl
#
# Chat-Format (System → Anfrage → Label); `--mask-prompt` trainiert nur auf dem Label.
# `valid.jsonl` ist der Val-Split, den mlx-lm *automatisch* zum Überwachen nutzt.

# %%
import json

SYSTEM = "Classify the bank customer's message into exactly one intent label. Respond with only the label."

def to_jsonl(texts, lbls, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, lbls):
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": t},
                {"role": "assistant", "content": l},
            ]}, ensure_ascii=False) + "\n")

DATA_DIR.mkdir(exist_ok=True)
to_jsonl(ft_t, ft_l, DATA_DIR / "train.jsonl")
to_jsonl(val_t, val_l, DATA_DIR / "valid.jsonl")
print("train.jsonl:", len(ft_t), "· valid.jsonl:", len(val_t))

# %% [markdown]
# ## Schritt 3 — Trainieren (der Lauf)
#
# mlx-lm druckt **train-Loss** (alle 10 Schritte) *und* **val-Loss** (alle 100, via
# `--steps-per-eval`) — beides im selben Log. **Overfitting-Wächter:** solange der
# **val-Loss fällt**, lernt es Verallgemeinerbares; **steigt er wieder**, ist da der
# Wendepunkt (und der beste Checkpoint der davor in `ft_adapters_<RUN>/`).

# %%
cmd = [
    str(MLX_PY), "-m", "mlx_lm", "lora",
    "--model", str(BASE_MODEL), "--train", "--data", str(DATA_DIR),
    "--fine-tune-type", FINE_TUNE_TYPE, "--mask-prompt",
    "--num-layers", str(NUM_LAYERS), "--batch-size", str(BATCH),
    "--iters", str(ITERS), "--learning-rate", str(LR),
    "--steps-per-eval", "100", "--adapter-path", str(ADAPTER_DIR),
]
print("Starte:", " ".join(cmd), "\n")
subprocess.run(cmd, check=True)

# %% [markdown]
# ## Schritt 4 — Fusen → oMLX-Modell
#
# ⚠️ **Nach dem Fusen oMLX einmal neu starten**, damit der neue Modellname
# `Llama-1B-banking77-<RUN>` in der Liste erscheint (oMLX scannt beim Start).

# %%
cmd_fuse = [
    str(MLX_PY), "-m", "mlx_lm", "fuse",
    "--model", str(BASE_MODEL), "--adapter-path", str(ADAPTER_DIR),
    "--save-path", str(FUSED_DIR),
]
print("Fuse:", " ".join(cmd_fuse), "\n")
subprocess.run(cmd_fuse, check=True)
print("\nfusioniert nach:", FUSED_DIR, "\n→ jetzt oMLX neu starten, dann Schritt 5.")

# %% [markdown]
# ## Schritt 5 — messen auf test + ins Dashboard tracken
#
# Durch dieselbe `classify()`-Pipeline; der `RUN` isoliert Cache und Scoreboard.
# `evaluate_and_save` schreibt `results.json` (+ Hyperparameter als Spalten) und
# `predictions/<RESULT_NAME>.json` → das Dashboard zeigt den Lauf automatisch.

# %%
from sklearn.metrics import accuracy_score
from eval_utils import evaluate_and_save, plot_confusion_matrix, plot_top_confusions, plot_per_class_f1

client = get_client()

def make_messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]

preds_ft, raws_ft = classify(client, FT_MODEL, test_t, make_messages, labels,
                             cache_tag=f"{FT_MODEL}_zeroshot_test", max_tokens=30)

evaluate_and_save(
    RESULT_NAME, test_l, preds_ft,
    model=f"Llama-3.2-1B {FINE_TUNE_TYPE.upper()}-SFT (lokal, oMLX)",
    note=f"generatives SFT (mask-prompt), {FINE_TUNE_TYPE}, {NUM_LAYERS} Layer, {ITERS} iters (~{epochs} Ep.), test 1x",
    fine_tune_type=FINE_TUNE_TYPE, iters=ITERS, epochs=epochs, num_layers=NUM_LAYERS, lr=LR,
)
print("unknown:", preds_ft.count("unknown"), "/", len(preds_ft))

plot_confusion_matrix(test_l, preds_ft, title=f"Llama-1B {FINE_TUNE_TYPE.upper()}-SFT [{RUN}] — Confusion (test)")
plot_top_confusions(test_l, preds_ft, top=15)
plot_per_class_f1(test_l, preds_ft, worst=20)

# %% [markdown]
# ## Schritt 6 — (optional) few-shot am finegetunten Modell
#
# Wir haben gesehen: few-shot *schadet* dem finegetunten Modell (In-Context ⊥
# In-Weights, Format-Bruch). Hier zum Nachprüfen mit dem besser trainierten Modell —
# rein informativ, **nicht** ins Scoreboard (kein eigener Ansatz, ein Diagnose-Test).

# %%
demo_turns = []
for dt, dl in zip(demo_t, demo_l):
    demo_turns += [{"role": "user", "content": dt}, {"role": "assistant", "content": dl}]

def make_messages_few(text):
    return [{"role": "system", "content": SYSTEM}] + demo_turns + [{"role": "user", "content": text}]

preds_few, _ = classify(client, FT_MODEL, test_t, make_messages_few, labels,
                        cache_tag=f"{FT_MODEL}_fewshot_test", max_tokens=30)
print(f"[{RUN}] finetuned + few-shot:  Accuracy={accuracy_score(test_l, preds_few):.3f}  "
      f"(vs. ohne Demos: {accuracy_score(test_l, preds_ft):.3f})  "
      f"unknown={preds_few.count('unknown')}/{len(preds_few)}")

# %% [markdown]
# ## ✓ Checkpoint
#
# Im Dashboard/`results.json` liegen jetzt die Läufe getrennt (`D_llama1b_ft_600it`,
# `D_llama1b_ft_3ep`, …) mit Hyperparameter-Spalten — direkt vergleichbar. Nächste
# Experimente = nur `RUN` + eine Stellschraube ändern (`ITERS`, `FINE_TUNE_TYPE="dora"`,
# `NUM_LAYERS`, `LR`), neu fahren, oMLX neustarten. Den Overfitting-Punkt liest du an
# der val-Loss-Kurve in Schritt 3 ab.
