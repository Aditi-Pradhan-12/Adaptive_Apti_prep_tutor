"""
weakness_diagnoser.py

Implements the actual ML component described in the proposal, section 5.2:
"Questions are tagged with concept-level embeddings (e.g., using
Sentence-BERT). Incorrect responses are clustered in embedding space to
identify specific recurring sub-concepts causing errors."

This replaces the earlier placeholder (a simple accuracy groupby by
puzzle_type) with genuine embedding + clustering, giving concept-level
diagnosis instead of a flat per-type score.

First run will download the sentence-transformers model (~80MB, cached
locally afterward) -- needs an internet connection once.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from concept_map import describe

_model = None


def get_embedding_model():
    """Lazy-loaded singleton so the model isn't reloaded on every Streamlit rerun."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def diagnose_weakness(wrong_rule_ids: list[str], n_clusters: int = None):
    """
    wrong_rule_ids: list of rule_id strings for every INCORRECT attempt so far
                     (duplicates expected and meaningful -- frequency matters).

    Returns a list of cluster summaries, sorted by how many mistakes fall in
    each cluster:
        [{"concepts": ["T3", "T6"], "count": 4,
          "description": "product of two numbers minus a third number; ..."}, ...]

    Clustering groups rule_ids that are CONCEPTUALLY similar (via their text
    description's embedding), not just puzzle_type -- e.g. a triangle rule and
    a grid rule that both hinge on "product minus a number" can land in the
    same cluster, which is exactly the kind of cross-cutting diagnosis a flat
    type-level accuracy score can't give you.
    """
    if not wrong_rule_ids:
        return []

    unique_ids = sorted(set(wrong_rule_ids))

    # Not enough distinct concepts to meaningfully cluster -- just report directly.
    if len(unique_ids) == 1:
        rid = unique_ids[0]
        return [{
            "concepts": [rid],
            "count": wrong_rule_ids.count(rid),
            "description": describe(rid),
        }]

    descriptions = [describe(rid) for rid in unique_ids]
    model = get_embedding_model()
    embeddings = model.encode(descriptions)

    k = n_clusters or max(1, min(3, len(unique_ids)))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for rid, label in zip(unique_ids, labels):
        clusters.setdefault(int(label), []).append(rid)

    summaries = []
    for label, rids in clusters.items():
        count = sum(wrong_rule_ids.count(r) for r in rids)
        descs = [describe(r) for r in rids]
        summaries.append({
            "concepts": rids,
            "count": count,
            "description": "; ".join(descs),
        })

    summaries.sort(key=lambda s: -s["count"])
    return summaries
