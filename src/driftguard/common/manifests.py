"""Detect Kubernetes resources that a rendered manifest declares more than once."""

from collections import Counter

import yaml

# ============= Detection =============


def find_duplicate_resources(rendered: str) -> list[tuple[str, str, str, int]]:
    """
    List resources that appear more than once in a multi-document manifest.

    Helm refuses to install such a manifest ("already exists"), so a duplicate is an install
    failure that `helm template` alone does not report.

    Parameters
    ----------
    rendered : str
        Multi-document YAML, for example the output of `helm template`.

    Returns
    -------
    duplicates : list[tuple[str, str, str, int]]
        `(kind, namespace, name, count)` for each resource declared more than once, sorted. The
        namespace is `-` for cluster-scoped resources.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    for document in yaml.safe_load_all(rendered):
        if not document or "kind" not in document:
            continue
        metadata = document.get("metadata") or {}
        counts[(document["kind"], metadata.get("namespace", "-"), metadata.get("name", ""))] += 1
    return sorted((*key, count) for key, count in counts.items() if count > 1)
