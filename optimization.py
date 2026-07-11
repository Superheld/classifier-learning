"""
Optimierungszyklus (F3) — track-unabhängiger Greedy-Kern.

`greedy_search` fährt den Zyklus: Default auf val messen, dann je Experiment GENAU
EINE Änderung; besser (an val Macro-F1) → behalten, sonst verwerfen. *Wie* eine
Config zu einem Score wird (TF-IDF-Matrix, fertige Embeddings, …) weiß die
`evaluate`-Funktion, die jeder Track selbst mitbringt. So lebt der Loop nur einmal
— im Root, weil er zu keinem Track gehört.
"""

import pandas as pd


def greedy_search(evaluate, experiments):
    """Greedy-Optimierung auf dem Val-Split.

    evaluate:    Funktion config(dict) -> (val_accuracy, val_macro_f1).
    experiments: Liste von (label, delta_dict) — je eine Änderung gegenüber dem
                 bisher Besten.
    Rückgabe:    (best_config, protocol_df).
    """
    best = {}
    acc0, f1_0 = evaluate(best)
    best_f1 = f1_0
    protocol = [{"schritt": "Start (Default)", "val_macroF1": round(f1_0 * 100, 2),
                 "val_acc": round(acc0 * 100, 2), "behalten": "—"}]
    print(f"Start auf val:  Macro-F1 {f1_0 * 100:.2f} %   Acc {acc0 * 100:.2f} %")

    for label, delta in experiments:
        cand = {**best, **(delta or {})}
        acc, f1 = evaluate(cand)
        better = f1 > best_f1
        print(f"{label:<24} val Macro-F1 {f1 * 100:5.2f} %  "
              f"(best {best_f1 * 100:.2f} %)  -> {'BEHALTEN' if better else 'verworfen'}")
        protocol.append({"schritt": label, "val_macroF1": round(f1 * 100, 2),
                         "val_acc": round(acc * 100, 2),
                         "behalten": "ja" if better else "nein"})
        if better:
            best, best_f1 = cand, f1

    return best, pd.DataFrame(protocol)
