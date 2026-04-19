"""
Compute global co-authorship community × topic affinity matrix.

For each (topic, global_community) pair: how many of the topic's authors
belong to that community, and what fraction does that represent?  Also
derives the dominant community per topic and pairwise cross-topic edges
weighted by the dot-product of community-fraction vectors.

Outputs
-------
topic_community_affinity.csv
    topic_id, global_community_id, n_authors, fraction_of_topic,
    fraction_of_community, is_dominant, topic_x, topic_y, topic_label,
    topic_n_papers

topic_topic_edges.csv
    topic_id_1, topic_id_2, shared_community_id, weight

Usage
-----
revu biblio agg-global-community-topic-affinity \\
  --global-communities data/coauthorship_results/author_communities.csv \\
  --global-paper-edges data/coauthorship_results/author_paper_edges.csv \\
  --modeled            data/causal_modeled_25Mar2026.csv \\
  --embeddings         data/embeddings_2d.npy \\
  --output-dir         data/coauthorship_results/global_community_topic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_global_community_topic_affinity(
    global_communities_path: str,
    global_paper_edges_path: str,
    modeled_path: str,
    embeddings_path: str,
    output_affinity_path: str,
    output_edges_path: str,
    topic_info_path: str | None = None,
    top_n_communities: int = 15,
    edge_min_weight: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a community-topic affinity matrix and cross-topic edge list.

    Parameters
    ----------
    global_communities_path : str
        Global author_communities.csv (columns: author_name, community_id).
    global_paper_edges_path : str
        Global author_paper_edges.csv (columns: id, author_name, ...).
    modeled_path : str
        Modeled CSV with at least 'id' and 'topic' columns.
    embeddings_path : str
        2-D UMAP .npy array; rows must match modeled_path.
    output_affinity_path : str
        Destination for topic_community_affinity.csv.
    output_edges_path : str
        Destination for topic_topic_edges.csv.
    topic_info_path : str or None
        Optional topic-info CSV with 'Topic' and 'Name' columns.
    top_n_communities : int
        Retain only the top-N largest global communities by member count;
        all others are collapsed into community_id = -99 ("Other").
    edge_min_weight : float
        Minimum dot-product weight for a cross-topic edge to be kept.

    Returns
    -------
    (affinity_df, edges_df) DataFrames written to disk.
    """
    print("=" * 60)
    print("GLOBAL COMMUNITY × TOPIC AFFINITY")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load global community assignments
    # ------------------------------------------------------------------
    print("\nLoading global author communities …")
    communities = pd.read_csv(global_communities_path)
    communities.columns = communities.columns.str.strip()
    communities["author_name"] = communities["author_name"].str.strip()

    community_sizes = communities["community_id"].value_counts()
    top_ids: set = set(community_sizes.head(top_n_communities).index.tolist())
    communities["community_id_mapped"] = communities["community_id"].apply(
        lambda c: c if c in top_ids else -99
    )
    print(f"  {len(communities):,} authors, {len(community_sizes):,} raw communities")
    print(f"  Retaining top {top_n_communities}; rest → 'Other' (community_id=-99)")

    author_to_community: dict = (
        communities.set_index("author_name")["community_id_mapped"].to_dict()
    )

    # Total distinct authors per (mapped) community — needed for fraction_of_community
    community_total_authors: dict = (
        communities.groupby("community_id_mapped")["author_name"]
        .nunique()
        .to_dict()
    )

    # ------------------------------------------------------------------
    # 2. Load global paper→author edges and map community
    # ------------------------------------------------------------------
    print("Loading global paper-author edges …")
    edges = pd.read_csv(global_paper_edges_path, usecols=["id", "author_name"])
    edges.columns = edges.columns.str.strip()
    edges["author_name"] = edges["author_name"].str.strip()
    edges = edges[edges["author_name"].isin(author_to_community)].copy()
    edges["global_community_id"] = edges["author_name"].map(author_to_community)
    print(f"  {len(edges):,} paper-author edges after community join")

    # ------------------------------------------------------------------
    # 3. Load topic assignments + embeddings
    # ------------------------------------------------------------------
    print("Loading topic assignments and embeddings …")
    modeled = pd.read_csv(modeled_path, usecols=["id", "topic"])
    coords = np.load(embeddings_path)
    modeled = modeled.reset_index(drop=True)
    modeled["x"] = coords[:, 0]
    modeled["y"] = coords[:, 1]
    modeled = modeled[modeled["topic"] != -1].copy()

    topic_centroids = (
        modeled.groupby("topic")
        .agg(
            topic_x=("x", "mean"),
            topic_y=("y", "mean"),
            topic_n_papers=("id", "count"),
        )
        .reset_index()
        .rename(columns={"topic": "topic_id"})
    )

    label_map: dict[int, str] = {}
    if topic_info_path and Path(topic_info_path).exists():
        info = pd.read_csv(topic_info_path, usecols=["Topic", "Name"])
        for _, row in info.iterrows():
            tid = int(row["Topic"])
            name = str(row["Name"])
            if tid != -1 and name != "-1":
                label_map[tid] = name
    topic_centroids["topic_label"] = topic_centroids["topic_id"].apply(
        lambda t: label_map.get(t, f"Topic {t}")
    )

    paper_to_topic: dict = modeled.set_index("id")["topic"].to_dict()

    # ------------------------------------------------------------------
    # 4. Join paper→topic→community; count distinct authors per cell
    # ------------------------------------------------------------------
    print("Computing community × topic affinity …")
    edges["topic_id"] = edges["id"].map(paper_to_topic)
    paper_join = edges.dropna(subset=["topic_id"]).copy()
    paper_join["topic_id"] = paper_join["topic_id"].astype(int)

    # Distinct (topic, community, author) triples — avoid double-counting
    triples = paper_join[["topic_id", "global_community_id", "author_name"]].drop_duplicates()

    affinity = (
        triples.groupby(["topic_id", "global_community_id"])["author_name"]
        .nunique()
        .rename("n_authors")
        .reset_index()
    )

    topic_total_authors: dict = (
        triples.groupby("topic_id")["author_name"].nunique().to_dict()
    )

    affinity["fraction_of_topic"] = affinity.apply(
        lambda r: r["n_authors"] / topic_total_authors.get(r["topic_id"], 1),
        axis=1,
    )
    affinity["fraction_of_community"] = affinity.apply(
        lambda r: r["n_authors"] / community_total_authors.get(r["global_community_id"], 1),
        axis=1,
    )

    # Dominant community per topic (most authors)
    dominant_lookup: dict = (
        affinity.sort_values("n_authors", ascending=False)
        .groupby("topic_id")["global_community_id"]
        .first()
        .to_dict()
    )
    affinity["is_dominant"] = affinity.apply(
        lambda r: r["global_community_id"] == dominant_lookup.get(r["topic_id"], None),
        axis=1,
    )

    # Attach centroid info
    affinity = affinity.merge(topic_centroids, on="topic_id", how="left")
    print(f"  Affinity matrix: {len(affinity):,} (topic, community) pairs")

    # ------------------------------------------------------------------
    # 5. Cross-topic edges via dot product of fraction-of-topic vectors
    # ------------------------------------------------------------------
    print("Computing cross-topic edges …")
    pivot = affinity.pivot_table(
        index="topic_id",
        columns="global_community_id",
        values="fraction_of_topic",
        fill_value=0.0,
    )
    mat = pivot.values.astype(np.float64)        # (n_topics, n_communities)
    topic_ids = pivot.index.tolist()
    community_cols = pivot.columns.tolist()

    dot = mat @ mat.T                            # (n_topics, n_topics)
    rows_idx, cols_idx = np.where(np.triu(dot >= edge_min_weight, k=1))

    edge_rows = []
    for r, c in zip(rows_idx, cols_idx):
        t1, t2 = topic_ids[r], topic_ids[c]
        # Shared community = community where min(f_t1, f_t2) is maximised
        shared_vec = np.minimum(mat[r], mat[c])
        best = int(np.argmax(shared_vec))
        edge_rows.append({
            "topic_id_1": int(t1),
            "topic_id_2": int(t2),
            "shared_community_id": int(community_cols[best]),
            "weight": float(dot[r, c]),
        })

    edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["topic_id_1", "topic_id_2", "shared_community_id", "weight"]
    )
    print(f"  {len(edges_df):,} cross-topic edges (weight >= {edge_min_weight})")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    for path_str, df in [
        (output_affinity_path, affinity),
        (output_edges_path, edges_df),
    ]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    print(f"\n✓ Affinity   → {output_affinity_path} ({len(affinity):,} rows)")
    print(f"✓ Edges      → {output_edges_path} ({len(edges_df):,} rows)")
    return affinity, edges_df
