"""
Aggregate within-topic co-authorship community nodes for DMP overlay.

For each topic T, reads the pre-computed per-topic community files from
``coauthorship_results/topic_T/`` and computes one centroid node per
community.  The centroid is the mean UMAP (x, y) of papers in topic T
whose authors belong to that community.

Output columns
--------------
topic_id        : int  — BERTopic topic ID
community_id    : int  — community ID local to the topic
x, y           : float — mean UMAP coordinate of covered papers
n_members       : int  — authors in this community (within this topic)
n_papers        : int  — distinct papers covered
modularity      : float — Louvain modularity of the topic's network
top_authors     : str  — comma-separated top-N authors by strength
label           : str  — "Co-authorship community N: Author1, Author2…"

Usage
-----
revu biblio agg-within-topic-community-nodes \\
  --coauthorship-dir data/coauthorship_results \\
  --modeled         data/causal_modeled_25Mar2026.csv \\
  --embeddings      data/embeddings_2d.npy \\
  --output          data/within_topic_community_nodes.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_within_topic_community_nodes(
    coauthorship_dir: str,
    modeled_path: str,
    embeddings_path: str,
    output_path: str,
    top_k_per_topic: int = 5,
    min_community_members: int = 3,
    top_n_authors: int = 5,
) -> pd.DataFrame:
    """
    Build one centroid node per (topic, community) pair.

    Parameters
    ----------
    coauthorship_dir : str
        Root directory produced by ``revu biblio coauthorship-analysis``.
        Expected sub-directories: ``topic_0/``, ``topic_1/``, …
    modeled_path : str
        Modeled CSV with columns ``id`` and ``topic``.
    embeddings_path : str
        2-D embeddings ``.npy`` file; rows must match ``modeled_path``.
    output_path : str
        Where to write the output CSV.
    top_k_per_topic : int
        Retain only the top-K largest communities per topic.
    min_community_members : int
        Skip communities with fewer than this many authors.
    top_n_authors : int
        Number of authors to include in the node label.

    Returns
    -------
    pd.DataFrame with one row per (topic, community) centroid node.
    """
    coauth_root = Path(coauthorship_dir)

    # ------------------------------------------------------------------
    # 1. Build global paper_id → (x, y) lookup
    # ------------------------------------------------------------------
    print("Loading embeddings and topic assignments …")
    coords = np.load(embeddings_path)                        # (N, 2)
    modeled = pd.read_csv(modeled_path, usecols=["id", "topic"])
    assert len(coords) == len(modeled), (
        f"Row count mismatch: embeddings {len(coords)} vs modeled {len(modeled)}"
    )
    modeled = modeled.reset_index(drop=True)
    modeled["x"] = coords[:, 0]
    modeled["y"] = coords[:, 1]
    paper_xy = modeled.set_index("id")[["x", "y"]]          # fast lookup
    print(f"  {len(paper_xy):,} papers loaded")

    # ------------------------------------------------------------------
    # 2. Discover topic directories
    # ------------------------------------------------------------------
    topic_dirs = sorted(
        p for p in coauth_root.iterdir()
        if p.is_dir() and p.name.startswith("topic_")
    )
    print(f"Found {len(topic_dirs)} topic directories")

    # ------------------------------------------------------------------
    # 3. Process each topic
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    for topic_dir in topic_dirs:
        topic_id_str = topic_dir.name.replace("topic_", "")
        try:
            topic_id = int(topic_id_str)
        except ValueError:
            continue
        if topic_id == -1:
            continue

        communities_csv = topic_dir / "author_communities.csv"
        edges_csv       = topic_dir / "author_paper_edges.csv"
        metrics_csv     = topic_dir / "author_network_metrics.csv"
        summary_csv     = topic_dir / "author_community_summary.csv"

        # Skip topics missing required files
        if not (communities_csv.exists() and edges_csv.exists()):
            continue

        # Load per-topic files
        try:
            communities = pd.read_csv(communities_csv)        # author_name, community_id
            edges       = pd.read_csv(edges_csv)              # id, author_name, …
            communities.columns = communities.columns.str.strip()
            edges.columns       = edges.columns.str.strip()
            communities["author_name"] = communities["author_name"].str.strip()
            edges["author_name"]       = edges["author_name"].str.strip()
        except Exception as exc:
            print(f"  [topic {topic_id}] read error: {exc} — skipping")
            continue

        # Optional: strength for ranking top authors
        strength_map: dict[str, float] = {}
        if metrics_csv.exists():
            try:
                metrics = pd.read_csv(metrics_csv)
                metrics.columns = metrics.columns.str.strip()
                metrics["author_name"] = metrics["author_name"].str.strip()
                strength_map = dict(zip(metrics["author_name"], metrics["strength"]))
            except Exception:
                pass

        # Optional: modularity from summary
        modularity_map: dict[int, float] = {}
        global_modularity: float = float("nan")
        if summary_csv.exists():
            try:
                summary = pd.read_csv(summary_csv)
                summary.columns = summary.columns.str.strip()
                global_modularity = float(summary["modularity"].iloc[0]) if len(summary) > 0 else float("nan")
                modularity_map = dict(zip(summary["community_id"], summary["modularity"]))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Join: paper_id → author_name → community_id
        # ------------------------------------------------------------------
        merged = edges[["id", "author_name"]].merge(
            communities[["author_name", "community_id"]],
            on="author_name",
            how="inner",
        )
        if merged.empty:
            continue

        # Attach UMAP coordinates — only papers present in the global lookup
        merged = merged[merged["id"].isin(paper_xy.index)].copy()
        merged["x"] = paper_xy.loc[merged["id"]].values[:, 0]
        merged["y"] = paper_xy.loc[merged["id"]].values[:, 1]

        if merged.empty:
            continue

        # ------------------------------------------------------------------
        # Group by community_id
        # ------------------------------------------------------------------
        for community_id, group in merged.groupby("community_id"):
            n_papers  = group["id"].nunique()
            cx        = group["x"].mean()
            cy        = group["y"].mean()

            # Authors in this community (from communities CSV)
            community_authors = communities.loc[
                communities["community_id"] == community_id, "author_name"
            ].tolist()
            n_members = len(community_authors)

            if n_members < min_community_members:
                continue

            # Rank by strength, fall back to appearance order
            community_authors_sorted = sorted(
                community_authors,
                key=lambda a: strength_map.get(a, 0.0),
                reverse=True,
            )
            top_authors = community_authors_sorted[:top_n_authors]
            top_authors_str = ", ".join(top_authors)
            label = f"Co-authorship community {community_id}: {top_authors_str}"

            all_rows.append({
                "topic_id":     topic_id,
                "community_id": int(community_id),
                "x":            cx,
                "y":            cy,
                "n_members":    n_members,
                "n_papers":     n_papers,
                "modularity":   modularity_map.get(community_id, global_modularity),
                "top_authors":  top_authors_str,
                "label":        label,
            })

    if not all_rows:
        print("No community nodes produced — check input paths.")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # 4. Per-topic filter: keep top-K by n_members
    # ------------------------------------------------------------------
    result = (
        result
        .sort_values(["topic_id", "n_members"], ascending=[True, False])
        .groupby("topic_id", group_keys=False)
        .head(top_k_per_topic)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\n✓ {len(result):,} community nodes written to {out}")
    print(f"  Topics covered : {result['topic_id'].nunique():,}")
    print(f"  Communities/topic (median): {result.groupby('topic_id').size().median():.1f}")

    return result
