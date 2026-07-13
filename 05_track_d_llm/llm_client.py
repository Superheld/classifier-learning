"""
Plumbing-Helfer für Track D — lokales LLM per OpenAI-kompatibler API.

Analog zu `03_track_b_embeddings/embeddings.py`: hier lebt die *Mechanik*
(Client bauen, aufrufen, cachen, Label parsen), damit die Modell-Notebooks sich
auf das konzentrieren, was Track D ausmacht — den **Prompt** und die **Messung**.

Warum ein lokaler Server über die OpenAI-API?
  oMLX (der lokale MLX-Server auf diesem Mac) spricht dasselbe HTTP-Protokoll wie
  OpenAI: `POST /v1/chat/completions`. Das ist zum De-facto-Standard geworden —
  Ollama, LM Studio, vLLM, oMLX, alle sprechen es. Darum nehmen wir den offiziellen
  `openai`-SDK und biegen ihn nur per `base_url` auf `localhost`. Kein Provider-
  Sonderweg, kein Cloud-Konto, keine API-Kosten — die "Kosten" sind hier Zeit.

Der Cache ist der Grund, warum das praktikabel ist: jede Antwort wird auf Platte
gespeichert (Schlüssel = Hash aus Modell + Nachrichten + Sampling). Ein zweiter
Lauf mit identischem Prompt ist sofort da — genau wie Track Bs `.npy`-Cache.
"""

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI

# --- Verbindung -------------------------------------------------------------
# base_url zeigt auf den lokalen oMLX-Server. Der Key ist lokal und unkritisch;
# wir lesen ihn trotzdem aus der Umgebung (OMLX_API_KEY) und fallen nur zur
# Bequemlichkeit auf die oMLX-Settings zurück — so steht nie ein Klartext-Secret
# in einer eingecheckten Datei.
BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")

_CACHE_DIR = Path(__file__).parent / "cache"


def _read_api_key():
    """Key aus OMLX_API_KEY, sonst aus ~/.omlx/settings.json (auth.api_key)."""
    key = os.environ.get("OMLX_API_KEY")
    if key:
        return key
    settings = Path.home() / ".omlx" / "settings.json"
    if settings.exists():
        with open(settings, encoding="utf-8") as f:
            return json.load(f).get("auth", {}).get("api_key", "")
    return ""


def get_client():
    """Baut einen OpenAI-Client, der auf den lokalen Server zeigt.

    Rückgabe ist ein ganz normaler `openai.OpenAI` — alle Methoden (z.B.
    `client.chat.completions.create(...)`) funktionieren wie gegen die echte
    OpenAI-Cloud, nur dass die Anfragen an oMLX auf localhost gehen.
    """
    return OpenAI(base_url=BASE_URL, api_key=_read_api_key())


def list_models(client=None):
    """Welche Modelle hält der Server gerade bereit? (`GET /v1/models`)"""
    client = client or get_client()
    return [m.id for m in client.models.list().data]


# --- Cache ------------------------------------------------------------------
def _cache_path(tag):
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{tag}.json"


def _cache_key(model, messages, temperature, max_tokens, extra):
    """Stabiler Hash über alles, was die Antwort bestimmt.

    Ändert sich irgendetwas am Prompt (auch nur ein Zeichen), am Modell oder am
    Sampling, ändert sich der Schlüssel — dann wird neu gerechnet. Das ist gewollt:
    ein getunter Prompt ist ein *anderer* Lauf und darf nicht alte Antworten sehen.
    """
    blob = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "extra": extra or {}},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- Label-Parsing ----------------------------------------------------------
def parse_label(raw, labels):
    """Rohtext des LLM → gültiges Label (oder "unknown").

    Ein kleines Modell hält sich nicht immer ans Format. Wir sind nachsichtig,
    aber ehrlich: passt nichts, ist die Vorhersage "unknown" und zählt als falsch
    — wir schummeln kein Label herbei.

    Reihenfolge: (1) exakte Übereinstimmung (case-insensitive), (2) ein Label
    kommt als Teilstring vor (längstes gewinnt — spezifischer), sonst (3) "unknown".
    """
    if raw is None:
        return "unknown"
    text = raw.strip().lower()
    lower = {lbl.lower(): lbl for lbl in labels}
    if text in lower:
        return lower[text]
    hits = [lbl for lbl in labels if lbl.lower() in text]
    if hits:
        return max(hits, key=len)
    return "unknown"


# --- Der eine Aufruf --------------------------------------------------------
def _complete(client, model, messages, temperature, max_tokens, extra):
    """Ein einzelner Chat-Completion-Aufruf. Gibt den Rohtext zurück."""
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
        extra_body=extra or None,
    )
    return resp.choices[0].message.content


def classify(client, model, texts, make_messages, labels, cache_tag,
             temperature=0.0, max_tokens=40, extra=None, progress=True):
    """Klassifiziert `texts` — mit Platten-Cache, resumbar.

    Parameter:
      make_messages : Funktion text -> Nachrichtenliste (der Prompt lebt im
                      Notebook, damit er dort sichtbar getunt werden kann).
      labels        : die gültigen Intent-Namen (zum Parsen).
      cache_tag     : Dateiname des Caches, z.B. "llama1b_p2".
      temperature=0 : deterministisch — bei Klassifikation wollen wir keine
                      Kreativität, sondern das wahrscheinlichste Label.

    Rückgabe: (preds, raws) — geparste Labels und die Rohantworten (fürs Debuggen).
    """
    path = _cache_path(cache_tag)
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)

    preds, raws = [], []
    n_new = 0
    for i, text in enumerate(texts):
        messages = make_messages(text)
        key = _cache_key(model, messages, temperature, max_tokens, extra)
        if key in cache:
            raw = cache[key]
        else:
            raw = _complete(client, model, messages, temperature, max_tokens, extra)
            cache[key] = raw
            n_new += 1
            # Inkrementell sichern: ein abgebrochener Lauf verliert nichts.
            if n_new % 20 == 0:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
        raws.append(raw)
        preds.append(parse_label(raw, labels))
        if progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(texts)}  (neu berechnet: {n_new})", flush=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    if progress:
        print(f"fertig: {len(texts)} Anfragen, davon {n_new} neu berechnet "
              f"(Rest aus Cache '{cache_tag}')")
    return preds, raws


# --- Stratifizierter Subsample ---------------------------------------------
def stratified_subsample(texts, labels, per_class, seed=42):
    """Je Klasse `per_class` Beispiele — fester Seed, reproduzierbar.

    Zum Iterieren (P2/P3) messen wir auf so einem Subsample: schnell genug für
    eine Lernschleife. Die *finale* Zahl kommt am Ende auf dem vollen Testsatz.
    Wichtig bleibt die Disziplin: getunt wird nur auf val, nie am Test.
    """
    import random

    rng = random.Random(seed)
    by_class = {}
    for t, l in zip(texts, labels):
        by_class.setdefault(l, []).append(t)
    sub_t, sub_l = [], []
    for lbl in sorted(by_class):
        picks = by_class[lbl]
        rng.shuffle(picks)
        for t in picks[:per_class]:
            sub_t.append(t)
            sub_l.append(lbl)
    return sub_t, sub_l
