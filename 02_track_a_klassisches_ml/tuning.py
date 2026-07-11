"""
Experiment-Werkzeug (F3 „Experiment-Setup") — der Optimierungszyklus, zentral.

`tune()` fährt den greedy Zyklus: erst Default auf val messen, dann je Experiment
GENAU EINE Änderung; besser (an val Macro-F1) → behalten, sonst verwerfen. Das
Testset wird hier NIE angefasst — die finale test-Messung bleibt im Modell-Notebook.

So bleibt in jeder Modell-Datei nur das Modellspezifische sichtbar: welcher
Klassifikator, welche Knöpfe (Experimente). Der Loop selbst ist einmal gebaut.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


def tune(make_clf, experiments, tr_texts, tr_labels, val_texts, val_labels):
    """Greedy-Optimierung auf dem Val-Split.

    make_clf:    Funktion clf_kwargs(dict) -> fertiger Klassifikator.
    experiments: Liste von (label, vec_change|None, clf_change|None) — je eine
                 Änderung gegenüber dem bisher Besten.
    Rückgabe:    (best_vec_kwargs, best_clf_kwargs, protocol_df).
    """

    def run(vec_kwargs, clf_kwargs):
        vec = TfidfVectorizer(**vec_kwargs)
        Xtr = vec.fit_transform(tr_texts)
        Xval = vec.transform(val_texts)
        clf = make_clf(clf_kwargs)
        clf.fit(Xtr, tr_labels)
        pr = clf.predict(Xval)
        return accuracy_score(val_labels, pr), f1_score(val_labels, pr, average="macro")

    best_vec, best_clf = {}, {}
    acc0, f1_0 = run(best_vec, best_clf)
    best_f1 = f1_0
    protocol = [{"schritt": "Start (Default)", "val_macroF1": round(f1_0 * 100, 2),
                 "val_acc": round(acc0 * 100, 2), "behalten": "—"}]
    print(f"Start auf val:  Macro-F1 {f1_0 * 100:.2f} %   Acc {acc0 * 100:.2f} %")

    for label, vec_change, clf_change in experiments:
        cand_vec = {**best_vec, **(vec_change or {})}
        cand_clf = {**best_clf, **(clf_change or {})}
        acc, f1 = run(cand_vec, cand_clf)
        better = f1 > best_f1
        print(f"{label:<24} val Macro-F1 {f1 * 100:5.2f} %  "
              f"(best {best_f1 * 100:.2f} %)  -> {'BEHALTEN' if better else 'verworfen'}")
        protocol.append({"schritt": label, "val_macroF1": round(f1 * 100, 2),
                         "val_acc": round(acc * 100, 2),
                         "behalten": "ja" if better else "nein"})
        if better:
            best_vec, best_clf, best_f1 = cand_vec, cand_clf, f1

    return best_vec, best_clf, pd.DataFrame(protocol)
