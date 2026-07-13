# %% [markdown]
# # Track D · Vorbereiten — LLM per Prompt (banking77)
#
# ## P1 „Vorbereiten" — der gemeinsame Boden für Gen 3b
#
# Dieses Notebook ist die **gemeinsame Vorarbeit** für alle Track-D-Modelle
# (analog zu `../03_track_b_embeddings/vorbereiten.py`). Hier steht das *Warum* und
# die *Mechanik* — die einzelnen Modell-Notebooks (`lokal/llama1b.py`, …) bauen
# darauf auf und wiederholen es nicht.
#
# Track A hat Wörter gezählt (TF-IDF), Track B hat Bedeutung als Vektor genommen
# (Embeddings), Track C hat ein BERT nachtrainiert (Finetuning). Alle drei haben
# eines gemeinsam: sie **lernen aus deinen gelabelten Daten**. Track D bricht damit.

# %% [markdown]
# ## Der Generationssprung: Gen 1 → 2 → 3a → **3b**
#
# Im Curriculum (`CURRICULUM.md`) sind die Ansätze als *Familien* geordnet — grob
# nach der Frage „woher kommt das Wissen des Modells?":
#
# | Gen | Familie | Wissensquelle | Track |
# |----|---------|---------------|-------|
# | 1 | Wörter zählen | deine gelabelten Daten | A |
# | 2 | Embeddings (eingefroren) | vortrainierter Encoder **+** deine Labels | B |
# | 3a | Finetuning | vortrainiertes BERT, **an deinen Labels weitergedreht** | C |
# | **3b** | **LLM per Prompt** | **Sprach- & Weltwissen des LLM · Kategorien: deine *Beschreibung*** | **D** |
#
# Der Sprung von 3a zu 3b ist kein „noch größeres Modell" — es ist ein *anderer
# Umgang*. Prompting (3b) und LLM-Finetuning sind **Geschwister, nicht Stufen**:
# beim einen beschreibst du die Aufgabe in Sprache, beim anderen trainierst du sie
# ein. Wir gehen den Prompt-Weg.

# %% [markdown]
# ## Was hier *fundamental* anders ist
#
# In Track A–C gab es immer zwei Phasen: **`fit`** (aus Trainingsdaten lernen) und
# **`predict`** (auf neue Anfragen anwenden). In Track D gibt es **kein `fit`**:
#
# - **Kein Training, kein Gradient, keine Gewichte, die sich bewegen.** Das Modell
#   ist fertig vortrainiert und bleibt, wie es ist.
# - **Die 77 Kategorien werden nicht *gelernt*, sondern *beschrieben*.** Sie landen
#   als Text im Prompt — hier schlicht als ihre Namen (`card_arrival`,
#   `pin_blocked`, …). Deshalb heißt es in der Doku: **„Data Prep ist
#   Prompt-Design".** Die Vorbereitung ist nicht mehr Vektorisieren, sondern:
#   *wie erkläre ich dem Modell die Aufgabe und die Kategorien?*
# - **Das Wissen kommt aus dem Vortraining** des LLM (Sprache, Welt), nicht aus
#   deinen Daten. Deine Daten brauchst du nur noch zum **Messen** (und optional als
#   Beispiele im Prompt, s. u.).
#
# Konsequenz: der „Trainingsteil" von banking77 ist in Track D fast arbeitslos.
# Wir nutzen ihn nur als Quelle für Val-Beispiele (zum Tunen) und ggf. Few-Shot.

# %% [markdown]
# ## Zwei Spielarten: zero-shot und few-shot
#
# - **Zero-shot** („null Beispiele"): Du gibst dem Modell nur die Aufgabe und die
#   Kategorien — *kein* gelöstes Beispiel. „Hier ist eine Anfrage, hier sind die 77
#   erlaubten Labels, gib genau eines zurück." Das ist der einfachste Fall.
# - **Few-shot** („wenige Beispiele"): Du legst dem Prompt ein paar *gelöste*
#   Beispiele bei (Anfrage → richtiges Label), aus denen das Modell das Muster
#   abliest. Das ist **In-Context-Learning**: das Modell „lernt" für die Dauer
#   dieses einen Aufrufs aus dem Kontext — ohne dass sich ein Gewicht ändert.
#
# Wir starten in P2 zero-shot (die Latte), und few-shot ist eine der Stellschrauben
# in P3. Merke: *mehr Beispiele im Prompt = mehr Kontext = langsamer und irgendwann
# teurer* — auch das ist eine Abwägung, kein „mehr ist immer besser".

# %% [markdown]
# ## Setup
#
# Wie in jedem Notebook: `autoreload`-Guard und ein Root-Bootstrap, der von `cwd`
# nach oben läuft, bis `data_utils.py` gefunden ist — dann Root **und** den
# Track-Ordner auf den `sys.path`. So laufen die Zellen, egal von wo Zed den Kernel
# startet.

# %%
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

# %% [markdown]
# ## Die Mechanik: ein lokales LLM über die OpenAI-API
#
# Wir sprechen das Modell **nicht** über eine Python-Bibliothek an, die es lädt
# (wie `sentence-transformers` in Track B), sondern über **HTTP** gegen einen
# lokalen **Server**. Auf diesem Mac läuft **oMLX** — ein MLX-Server (Apples
# ML-Framework für Apple Silicon), der mehrere Modelle bereithält und auf
# `http://127.0.0.1:8000/v1` lauscht.
#
# Das Entscheidende: dieser Server spricht das **OpenAI-Protokoll**
# (`POST /v1/chat/completions`). Das ist zum *De-facto-Standard* geworden — Ollama,
# LM Studio, vLLM, oMLX, alle sprechen es. Darum nehmen wir den **offiziellen
# `openai`-SDK** und biegen ihn nur per `base_url` auf `localhost`. Kein
# Provider-Sonderweg, kein Cloud-Konto, **keine API-Kosten**.
#
# Die ganze Verbindungs-Mechanik (Client bauen, Key aus der Umgebung lesen, cachen,
# Label parsen) steckt in **`llm_client.py`** — dem Track-D-Pendant zu Track Bs
# `embeddings.py`.

# %%
from llm_client import get_client, list_models

client = get_client()          # openai.OpenAI, aber auf localhost gezeigt
print("Server hält bereit:")
for m in list_models(client):  # GET /v1/models
    print("  •", m)

# %% [markdown]
# ## Ein Aufruf von Hand — damit klar ist, was passiert
#
# Ein Chat-Completion ist eine Liste von **Nachrichten** mit Rollen (`system` =
# Anweisung an das Modell, `user` = die eigentliche Eingabe). Das Modell antwortet
# mit einer `assistant`-Nachricht. `temperature=0` heißt „so deterministisch wie
# möglich" — bei Klassifikation wollen wir das wahrscheinlichste Label, keine
# Kreativität.

# %%
import time

MODEL = "Llama-3.2-1B-Instruct-4bit"   # klein & schnell — der Pipeline-Starter

t0 = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a terse assistant. Answer in one word."},
        {"role": "user", "content": "Is 'my card has not arrived' about a card or a loan?"},
    ],
    temperature=0,
    max_tokens=10,
)
print("Antwort:", repr(resp.choices[0].message.content))
print(f"Dauer: {time.time() - t0:.2f}s")
print("Tokens:", resp.usage.prompt_tokens, "→", resp.usage.completion_tokens)

# %% [markdown]
# ## Die neue Kosten-Achse: **Zeit** statt Geld
#
# In Track A–C war eine Messung in *Sekunden* durch (Matrix mal Vektor). In Track D
# ist jede Vorhersage ein eigener Modell-Lauf. Bei **Llama-1B** sind das ~0,2 s pro
# Anfrage — schnell. Aber der Server hat auch **Reasoning-Modelle** (z.B. Nemotron
# 30B): die „denken" erst mehrere hundert Tokens, bevor sie antworten, und brauchen
# dann leicht **~10 s pro Anfrage**.
#
# Rechne das hoch: der volle Testsatz hat **3076** Anfragen.
# - Llama-1B (~0,2 s):  ~10 Minuten
# - Nemotron (~10 s):   **~8 Stunden**
#
# Das Curriculum nennt lokale offene LLMs „schwächer, aber ohne API-Kosten" — hier
# siehst du: die Kosten sind nicht weg, sie sind nur **Zeit** (und Strom/Wärme)
# statt Dollar. Zwei Konsequenzen, die den ganzen Track-D-Workflow prägen:
#
# 1. **Cachen ist Pflicht.** `llm_client.classify(...)` speichert jede Antwort auf
#    Platte (Schlüssel = Hash aus Modell + Prompt + Sampling). Der zweite Lauf mit
#    identischem Prompt ist sofort da — wie Track Bs `.npy`-Cache.
# 2. **Wir iterieren auf einem Subsample**, nicht am vollen Satz (s. u.).

# %% [markdown]
# ## „Ehrlich messen" in Track D — die Disziplin bleibt, der Ablauf passt sich an
#
# Das Kernprinzip des Projekts gilt unverändert:
# - **Getunt wird nur auf val** (`load_banking77_split()`), **nie am Testset.**
# - Das **Testset wird pro Track genau zweimal** angefasst: naive P2-Messung und
#   finale P3-Messung.
#
# Neu ist nur, *wie* wir mit der Zeit umgehen. Weil ein voller Lauf teuer (in Zeit)
# ist, messen wir **beim Iterieren auf einem festen, stratifizierten Subsample**
# von val (`stratified_subsample`, fester Seed → reproduzierbar). Die schnelle
# Rückmeldung erlaubt eine echte Lernschleife. Die **finale, vergleichbare Zahl**
# holen wir dann in **einem** Lauf auf dem vollen Testsatz — dieselben 3076
# Anfragen, auf denen Track A–C gemessen wurden.
#
# Das ist kein Weichspülen der Disziplin: getunt wird weiter nur auf val, und der
# Test wird nur zweimal angefasst. Der Subsample ist lediglich das *Werkzeug*, um
# die Val-Iteration bezahlbar zu machen.

# %%
# So sieht der Iterier-Subsample aus (hier nur zur Ansicht — genutzt wird er in den
# Modell-Notebooks):
from llm_client import stratified_subsample

_, _, val_t, val_l = load_banking77_split()
labels = sorted(set(val_l))
sub_t, sub_l = stratified_subsample(val_t, val_l, per_class=2)  # 2 je Klasse
print(f"Val gesamt: {len(val_t)}   →   Iterier-Subsample: {len(sub_t)} (2×{len(labels)})")
print(f"Voller Testsatz (nur ganz am Ende): {len(load_banking77('test')[0])}")

# %% [markdown]
# ## Die Latte
#
# Woran messen wir Track D? An den Vorgängern (test-Accuracy):
#
# - **Majority-Baseline:** ~1,30 % (immer die häufigste Klasse raten)
# - **Track A** (TF-IDF + LogReg, getunt): ~90 %
# - **Track B** (mpnet-Embeddings + Kopf, getunt): ~94 %
# - **Track C** (RoBERTa finegetunt): ~94 %
#
# Die spannende, offene Frage von Track D: **Wie weit kommt ein Modell, dem man die
# Aufgabe nur *beschreibt* — ohne es je auf banking77 zu trainieren?** Und was
# kostet es (an Zeit), diese Qualität zu bekommen?

# %% [markdown]
# ## Fahrplan
#
# - **`lokal/llama1b.py`** — P2: der erste konkrete Lauf mit dem kleinen, schnellen
#   Llama-1B. Naiv zero-shot, ehrlich gemessen — die Baseline für Track D.
# - Danach P3: den Prompt tunen (Ausgabe erzwingen, few-shot, ggf. guided decoding)
#   und stärkere Modelle als Vergleich — bis zur finalen Messung am vollen Test.
#
# Weiter in `lokal/llama1b.py`.
