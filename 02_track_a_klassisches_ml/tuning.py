"""
TF-IDF-Kopf-Tuning (Track A). Baut je Config eine TF-IDF-Matrix + Klassifikator
und lässt den generischen Greedy-Kern (`optimization.greedy_search`) darauf laufen.
Track A behält so seine vec/clf-Aufteilung nach außen; der Loop selbst lebt im Root
und wird von Track B (Embeddings) genauso genutzt.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from optimization import greedy_search

# Welche Config-Schlüssel gehören zum Vectorizer (Rest geht an den Klassifikator).
_VEC_PARAMS = set(TfidfVectorizer().get_params())


def tune(make_clf, experiments, tr_texts, tr_labels, val_texts, val_labels):
    """Greedy-Optimierung auf dem Val-Split (TF-IDF + Kopf).

    make_clf:    Funktion clf_kwargs(dict) -> fertiger Klassifikator.
    experiments: Liste von (label, vec_change|None, clf_change|None) — je eine
                 Änderung gegenüber dem bisher Besten.
    Rückgabe:    (best_vec_kwargs, best_clf_kwargs, protocol_df) — wie gehabt.
    """

    def evaluate(cfg):
        vec_kw = {k: v for k, v in cfg.items() if k in _VEC_PARAMS}
        clf_kw = {k: v for k, v in cfg.items() if k not in _VEC_PARAMS}
        vec = TfidfVectorizer(**vec_kw)
        Xtr = vec.fit_transform(tr_texts)
        Xval = vec.transform(val_texts)
        clf = make_clf(clf_kw)
        clf.fit(Xtr, tr_labels)
        pr = clf.predict(Xval)
        return accuracy_score(val_labels, pr), f1_score(val_labels, pr, average="macro")

    # (label, vec_change, clf_change) → (label, flaches delta); vec/clf-Keys kollidieren nicht
    flat = [(label, {**(vc or {}), **(cc or {})}) for label, vc, cc in experiments]
    best, proto = greedy_search(evaluate, flat)

    best_vec = {k: v for k, v in best.items() if k in _VEC_PARAMS}
    best_clf = {k: v for k, v in best.items() if k not in _VEC_PARAMS}
    return best_vec, best_clf, proto
