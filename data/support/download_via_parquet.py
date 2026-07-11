"""
Script-basierte Datensätze über den Parquet-Zweig laden
=======================================================

Banking77, CLINC150 und MASSIVE laden über veraltete Lade-Scripts, die die
aktuelle `datasets`-Version nicht mehr unterstützt. Hugging Face erzeugt aber
für JEDEN Datensatz automatisch eine Parquet-Konvertierung im Branch
`refs/convert/parquet`. Die laden wir direkt — kein Script nötig.
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

HERE = Path(__file__).parent
BRANCH = "refs/convert/parquet"

DATASETS = [
    dict(name="banking77", id="PolyAI/banking77", config="default", text="text", label="label",
         note="ECHT, englisch, Bank-Kundenanfragen, 77 Intents"),
    dict(name="clinc150", id="clinc_oos", config="plus", text="text", label="intent",
         note="ECHT-nah, englisch, 150 Intents + out-of-scope"),
    dict(name="massive_de", id="AmazonScience/massive", config="de-DE", text="utt", label="intent",
         note="ECHT-nah, DEUTSCH, Voice-Assistant, 60 Intents"),
]


def load_config(cfg):
    """Lädt alle Splits eines Configs aus dem Parquet-Zweig als DataFrames."""
    files = list_repo_files(cfg["id"], repo_type="dataset", revision=BRANCH)
    # Nur Parquet-Dateien dieses Configs: "<config>/<split>/....parquet"
    mine = [f for f in files if f.startswith(cfg["config"] + "/") and f.endswith(".parquet")]
    splits = {}
    for f in mine:
        split = f.split("/")[1]
        local = hf_hub_download(cfg["id"], f, repo_type="dataset", revision=BRANCH)
        splits.setdefault(split, []).append(pd.read_parquet(local))
    return {s: pd.concat(dfs, ignore_index=True) for s, dfs in splits.items()}


def main():
    for cfg in DATASETS:
        print(f"\n=== {cfg['name']} — {cfg['note']} ===")
        try:
            splits = load_config(cfg)
            out = HERE / cfg["name"]
            out.mkdir(exist_ok=True)
            for split, df in splits.items():
                df.to_parquet(out / f"{split}.parquet")

            main_df = splits.get("train", next(iter(splits.values())))
            texts = main_df[cfg["text"]].astype(str)
            laengen = texts.str.len()
            n_intents = main_df[cfg["label"]].nunique()

            print(f"    [gespeichert] -> data/support/{cfg['name']}/")
            print(f"    Splits:   {', '.join(f'{s}={len(df)}' for s, df in splits.items())}")
            print(f"    Intents:  {n_intents}")
            print(f"    Textlänge (Zeichen): median {int(laengen.median())}, "
                  f"max {int(laengen.max())}, min {int(laengen.min())}")
            print(f"    Beispiele:")
            for t, l in list(zip(texts, main_df[cfg["label"]]))[:4]:
                kurz = t[:90] + ("…" if len(t) > 90 else "")
                print(f"      [{str(l):<22}] {kurz}")
        except Exception as e:
            print(f"    !! übersprungen: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
