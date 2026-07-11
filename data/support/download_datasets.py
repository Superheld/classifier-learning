"""
Support-Datensätze herunterladen und profilieren
=================================================

Lädt mehrere Customer-Support-/Intent-Datensätze vom Hugging Face Hub, speichert
sie lokal unter data/support/<name>/ und druckt ein Kurzprofil: Anzahl Zeilen,
Anzahl Intents, Textlängen (Median/Max), Beispielzeilen.

Robust: Ein Datensatz, der nicht lädt, killt die anderen nicht.
"""

import statistics
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent

# Pro Datensatz: id, optionaler config, Text- und Label-Spalte.
DATASETS = [
    dict(name="pro_familia_de", id="pro-familia-Bundesverband/pro-familia-intent-dataset-reproductive-health-de",
         config=None, text="intent_example_text", label="intent_name",
         note="ECHT, deutsch, Chatbot-Utterances (kurz), paraphrasenreich"),
    dict(name="banking77", id="PolyAI/banking77",
         config=None, text="text", label="label",
         note="ECHT, englisch, Bank-Kundenanfragen (kurz), 77 Intents"),
    dict(name="clinc150", id="clinc_oos",
         config="plus", text="text", label="intent",
         note="ECHT-nah, englisch, 150 Intents + out-of-scope"),
    dict(name="massive_de", id="AmazonScience/massive",
         config="de-DE", text="utt", label="intent",
         note="ECHT-nah, DEUTSCH, Voice-Assistant-Utterances, 60 Intents"),
    dict(name="bitext_en", id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
         config=None, text="instruction", label="intent",
         note="SYNTHETISCH, englisch, mit Tippfehlern/Varianten"),
]


def profile(ds_dict, cfg):
    """Druckt ein Kurzprofil und gibt die genutzte Split-Referenz zurück."""
    # Split zum Profilieren wählen (train, sonst der erste vorhandene).
    split = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
    ds = ds_dict[split]

    texts = [str(t) for t in ds[cfg["text"]]]
    labels = [str(l) for l in ds[cfg["label"]]]
    laengen = [len(t) for t in texts]
    n_intents = len(set(labels))

    print(f"    Splits:   {', '.join(f'{k}={len(v)}' for k, v in ds_dict.items())}")
    print(f"    Intents:  {n_intents}")
    print(f"    Textlänge (Zeichen): median {int(statistics.median(laengen))}, "
          f"max {max(laengen)}, min {min(laengen)}")
    print(f"    Beispiele:")
    for t, l in list(zip(texts, labels))[:4]:
        kurz = t[:90] + ("…" if len(t) > 90 else "")
        print(f"      [{l:<22}] {kurz}")


def main():
    for cfg in DATASETS:
        print(f"\n=== {cfg['name']} — {cfg['note']} ===")
        try:
            kwargs = dict(trust_remote_code=True)
            if cfg["config"]:
                ds_dict = load_dataset(cfg["id"], cfg["config"], **kwargs)
            else:
                ds_dict = load_dataset(cfg["id"], **kwargs)

            out = HERE / cfg["name"]
            ds_dict.save_to_disk(str(out))
            print(f"    [gespeichert] -> data/support/{cfg['name']}/")
            profile(ds_dict, cfg)
        except Exception as e:
            print(f"    !! übersprungen: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
