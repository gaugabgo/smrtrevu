"""
Per-topic author co-authorship map via DataMapPlot.

Each point is one (author, topic) pair, positioned at the mean UMAP
coordinate of that author's papers within that topic.  Points are coloured
by their per-topic community and labelled at the community centroid —
giving a two-tier hierarchy (topic region → community blob → author node)
that sits in the same coordinate space as the BERTopic topic map.

Noise filters applied in order
-------------------------------
1. Topic -1 (BERTopic outlier bucket) → excluded entirely.
2. Per-topic community -1 (Louvain outlier nodes) → excluded.
3. Authors with fewer than --min-papers-per-topic papers in a topic → excluded.
4. Topics absent from the per-topic coauthorship directory (not analysed) → skipped.
   Small topics naturally receive fewer, smaller points; DMP zoom-reveal
   handles their visibility without hard-filtering them.

Requires
--------
- embeddings_2d.npy               — UMAP 2-D coords, one row per paper
- causal_modeled_*.csv            — paper id + topic assignment
- metadata (deduplicated) CSV     — paper id + pipe-separated author names
- {coauthorship_dir}/topic_*/
      author_communities.csv      — author_name, community_id  (per topic)
- topic_info CSV (optional)       — Topic, Name  (custom labels)

Usage
-----
python src/revu/scripts/visu_author_topic_dmp.py \\
    --embeddings   data/embeddings_2d.npy \\
    --modeled      data/causal_modeled_25Mar2026.csv \\
    --metadata     data/causal_metadata_validated.csv \\
    --coauth-dir   data/coauthorship_results \\
    --topic-info   data/causalinference_modeled_28Mar2026_topic_info_customlabel.csv \\
    --min-papers   3 \\
    --output       data/visualizations/author_topic_communities_dmp.html
"""

import argparse
import re
from pathlib import Path

import datamapplot
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — build per-(author, topic) position table
# ---------------------------------------------------------------------------

def build_author_topic_positions(
    embeddings_path: str,
    modeled_path: str,
    metadata_path: str,
    min_papers: int = 3,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (author, topic) pair that passes
    the minimum-papers filter.

    Columns: author_name, topic, x, y, n_papers
    """
    print("Loading embeddings …")
    coords = np.load(embeddings_path)                          # (N, 2)

    print("Loading modeled topics …")
    modeled = pd.read_csv(modeled_path, usecols=["id", "topic"])
    assert len(coords) == len(modeled), (
        f"Row mismatch: embeddings {len(coords)} vs modeled {len(modeled)}"
    )
    modeled = modeled.reset_index(drop=True)
    modeled["x"] = coords[:, 0]
    modeled["y"] = coords[:, 1]

    # Filter topic -1 immediately
    before = len(modeled)
    modeled = modeled[modeled["topic"] != -1]
    print(f"  Removed {before - len(modeled):,} outlier papers (topic -1)")

    print("Loading metadata …")
    meta_cols = ["id", "authorships.raw_author_name"]
    meta = pd.read_csv(metadata_path, usecols=meta_cols)
    meta.columns = ["id", "authors_raw"]
    meta["authors_raw"] = meta["authors_raw"].fillna("")

    print("Merging and exploding authors …")
    df = modeled.merge(meta, on="id", how="left")
    df = df[df["authors_raw"] != ""].copy()
    df["author_name"] = df["authors_raw"].str.split("|")
    df = df.explode("author_name")
    df["author_name"] = df["author_name"].str.strip()
    df = df[df["author_name"] != ""]

    print("Aggregating per (author, topic) …")
    agg = (
        df.groupby(["author_name", "topic"])
        .agg(x=("x", "mean"), y=("y", "mean"), n_papers=("id", "count"))
        .reset_index()
    )

    before = len(agg)
    agg = agg[agg["n_papers"] >= min_papers].copy()
    print(
        f"  Kept {len(agg):,} (author, topic) pairs "
        f"(removed {before - len(agg):,} below min_papers={min_papers})"
    )
    return agg


# ---------------------------------------------------------------------------
# Step 2 — attach per-topic community labels
# ---------------------------------------------------------------------------

def attach_communities(
    author_topic: pd.DataFrame,
    coauth_dir: str,
) -> pd.DataFrame:
    """
    For each topic in author_topic, load the per-topic author_communities.csv
    from {coauth_dir}/topic_{id}/ and join community_id onto the table.

    Authors not present in a topic's community file get community_id = -1
    (treated as noise and later excluded).
    """
    coauth_path = Path(coauth_dir)
    topics_present = sorted(author_topic["topic"].unique())

    comm_frames = []
    missing = []

    for topic_id in topics_present:
        comm_file = coauth_path / f"topic_{topic_id}" / "author_communities.csv"
        if not comm_file.exists():
            missing.append(topic_id)
            continue
        comm = pd.read_csv(comm_file)
        comm.columns = comm.columns.str.strip()
        comm["author_name"] = comm["author_name"].str.strip()
        comm["topic"] = topic_id
        comm_frames.append(comm[["author_name", "topic", "community_id"]])

    if missing:
        print(
            f"  ⚠ Per-topic community files missing for {len(missing)} topic(s) "
            f"— they will be excluded from the map. "
            f"(Topics: {missing[:10]}{'…' if len(missing) > 10 else ''})"
        )

    if not comm_frames:
        raise FileNotFoundError(
            f"No per-topic community files found under {coauth_dir}/topic_*/ — "
            "run cli_coauthorship with --per_topic first."
        )

    all_comms = pd.concat(comm_frames, ignore_index=True)
    merged = author_topic.merge(all_comms, on=["author_name", "topic"], how="left")
    merged["community_id"] = merged["community_id"].fillna(-1).astype(int)

    # Drop community -1 (Louvain outlier nodes within each topic)
    before = len(merged)
    merged = merged[merged["community_id"] != -1].copy()
    print(
        f"  Removed {before - len(merged):,} author-topic rows "
        "with community_id = -1 (within-topic outliers)"
    )

    # Drop rows where topic has no community file (community_id was never set)
    merged = merged[merged["topic"].isin(all_comms["topic"].unique())].copy()

    print(f"  Final: {len(merged):,} (author, topic, community) rows")
    return merged


# ---------------------------------------------------------------------------
# Step 3 — resolve topic names and build DMP label strings
# ---------------------------------------------------------------------------

def build_labels(
    df: pd.DataFrame,
    topic_info_path: str | None,
) -> np.ndarray:
    """
    Build a label array for DataMapPlot.

    Each unique (topic, community_id) pair gets a label of the form:
        "Topic Name > Community N"
    where "Topic Name" comes from the topic_info CSV if available,
    otherwise falls back to "Topic {id}".

    Returns an ndarray of strings aligned with df rows.
    """
    # Load topic name map
    topic_names: dict[int, str] = {}
    if topic_info_path:
        try:
            ti = pd.read_csv(topic_info_path)
            ti.columns = ti.columns.str.strip()
            col_map = {c.lower(): c for c in ti.columns}
            t_col = next((col_map[k] for k in ["topic", "topic_id"] if k in col_map), None)
            n_col = next((col_map[k] for k in ["name", "customlabel", "custom_label", "label"] if k in col_map), None)
            if t_col and n_col:
                # Use to_numeric with coerce to silently drop misparsed rows
                # (long Representative_Docs strings can shift columns in some CSV parsers)
                topic_ids = pd.to_numeric(ti[t_col], errors="coerce")
                valid = topic_ids.notna()
                topic_names = dict(zip(topic_ids[valid].astype(int), ti[n_col][valid].astype(str)))
                dropped = (~valid).sum()
                if dropped:
                    print(f"  ⚠ Skipped {dropped} misparsed rows in topic info file")
                print(f"  Loaded {len(topic_names)} topic names from {topic_info_path}")
            else:
                print(f"  ⚠ Could not find topic/name columns in {topic_info_path}; using topic IDs")
        except Exception as exc:
            print(f"  ⚠ Could not load topic info: {exc}")

    def _topic_label(tid: int) -> str:
        name = topic_names.get(tid, f"Topic {tid}")
        # Truncate very long names
        if len(name) > 40:
            name = name[:37] + "…"
        return name

    labels = np.array([
        _topic_label(row['topic'])
        for _, row in df.iterrows()
    ])
    return labels


# ---------------------------------------------------------------------------
# Step 4 — size encoding
# ---------------------------------------------------------------------------

def build_marker_sizes(
    df: pd.DataFrame,
    min_px: float = 2.0,
    max_px: float = 20.0,
) -> np.ndarray:
    """
    Marker size proportional to sqrt(n_papers) in that topic,
    scaled to [min_px, max_px].
    """
    s = np.sqrt(df["n_papers"].values.astype(float))
    rng = s.max() - s.min()
    if rng < 1e-9:
        return np.full(len(df), (min_px + max_px) / 2)
    return min_px + (max_px - min_px) * (s - s.min()) / rng


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_author_topic_dmp(
    embeddings_path: str,
    modeled_path: str,
    metadata_path: str,
    coauth_dir: str,
    topic_info_path: str | None = None,
    min_papers: int = 3,
    output_file: str = "author_topic_communities_dmp.html",
    title: str = "Author Communities by Topic",
    interactive: bool = True,
) -> None:
    """
    Build the per-topic author community DataMapPlot.

    Parameters
    ----------
    min_papers : int
        Minimum papers an author must have within a topic to be included.
    interactive : bool
        HTML interactive output (True) or static PNG (False).
    """
    # 1. Build positions
    author_topic = build_author_topic_positions(
        embeddings_path=embeddings_path,
        modeled_path=modeled_path,
        metadata_path=metadata_path,
        min_papers=min_papers,
    )

    # 2. Attach communities
    print("\nAttaching per-topic communities …")
    df = attach_communities(author_topic, coauth_dir=coauth_dir)

    # 3. Build labels
    print("\nBuilding labels …")
    labels = build_labels(df, topic_info_path=topic_info_path)

    # 4. Marker sizes
    marker_sizes = build_marker_sizes(df)

    # 5. Hover text
    hover_text = (
        df["author_name"]
        + " | "
        + df["n_papers"].astype(str)
        + " papers in topic"
    ).tolist()

    coords = df[["x", "y"]].values
    n_unique_labels = len(set(labels))
    print(
        f"\nRendering DataMapPlot: {len(coords):,} points, "
        f"{n_unique_labels} (topic, community) groups …"
    )

    # 6. Render
    if interactive:
        fig = datamapplot.create_interactive_plot(
            coords,
            labels,
            hover_text=hover_text,
            marker_size_array=marker_sizes,
            title=title,
            sub_title=(
                f"{len(df['author_name'].unique()):,} authors · "
                f"{len(df['topic'].unique())} topics · "
                f"min {min_papers} papers per topic"
            ),
            point_radius_min_pixels=1,
            point_radius_max_pixels=8,
        )
        out_path = Path(output_file)
        if out_path.suffix.lower() != ".html":
            out_path = out_path.with_suffix(".html")
        fig.save(str(out_path))
        print(f"\n✓ Interactive map saved to {out_path}")

    else:
        fig, ax = datamapplot.create_plot(
            coords,
            labels,
            marker_size_array=marker_sizes,
            title=title,
            sub_title=(
                f"{len(df['author_name'].unique()):,} authors · "
                f"{len(df['topic'].unique())} topics"
            ),
            figsize=(16, 14),
            dpi=150,
        )
        import matplotlib.pyplot as plt
        out_path = Path(output_file)
        if out_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            out_path = out_path.with_suffix(".png")
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Static map saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-topic author community map via DataMapPlot"
    )
    parser.add_argument(
        "--embeddings", default="data/embeddings_2d.npy",
        help="Path to embeddings_2d.npy",
    )
    parser.add_argument(
        "--modeled", default="data/causal_modeled_25Mar2026.csv",
        help="Modeled CSV with 'id' and 'topic' columns",
    )
    parser.add_argument(
        "--metadata", default="data/causal_metadata_validated.csv",
        help="Metadata CSV with 'id' and 'authorships.raw_author_name' columns",
    )
    parser.add_argument(
        "--coauth-dir", default="data/coauthorship_results",
        help="Root coauthorship output directory (contains topic_*/ subdirs)",
    )
    parser.add_argument(
        "--topic-info", default=None,
        help="Optional topic_info CSV with 'Topic' and 'Name'/'CustomLabel' columns",
    )
    parser.add_argument(
        "--min-papers", type=int, default=3,
        help="Min papers an author must have in a topic to be included (default: 3)",
    )
    parser.add_argument(
        "--output", default="author_topic_communities_dmp.html",
        help="Output file (.html for interactive, .png/.pdf for static)",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Produce a static PNG instead of interactive HTML",
    )
    parser.add_argument(
        "--title", default="Author Communities by Topic",
    )
    args = parser.parse_args()

    visualize_author_topic_dmp(
        embeddings_path=args.embeddings,
        modeled_path=args.modeled,
        metadata_path=args.metadata,
        coauth_dir=args.coauth_dir,
        topic_info_path=args.topic_info,
        min_papers=args.min_papers,
        output_file=args.output,
        title=args.title,
        interactive=not args.static,
    )


if __name__ == "__main__":
    main()
