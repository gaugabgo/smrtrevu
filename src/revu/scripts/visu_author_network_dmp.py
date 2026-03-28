"""
Author co-authorship map via DataMapPlot.

Each author is positioned at the mean UMAP coordinate of their papers,
placing co-authors who work on similar topics near each other — in the
same semantic space as the BERTopic topic map.

Requires
--------
- embeddings_2d.npy      — UMAP 2-D coordinates, one row per paper
- causal_modeled_*.csv   — paper id + topic assignment
- metadata_deduplicated.csv — paper id + pipe-separated author names
- author_communities.csv — author_name + community_id
- author_network_metrics.csv — author_name + strength (for sizing)

Usage
-----
python src/revu/scripts/visu_author_network_dmp.py \
  --coauth-dir data/coauthorship_results \
  --topic-info data/causalinference_modeled_28Mar2026_topic_info_customlabel.csv \
  --min-papers 3 \
  --output data/visualizations/author_topic_communities_dmp.html
"""

import argparse
from pathlib import Path

import datamapplot
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_author_positions(
    embeddings_path: str,
    modeled_path: str,
    metadata_path: str,
    min_papers: int = 5,
) -> pd.DataFrame:
    """
    Compute mean UMAP position for every author with >= min_papers.

    Returns a DataFrame with columns:
        author_name, x, y, n_papers, topics (list of topic ints)
    """
    print("  Loading embeddings …")
    coords = np.load(embeddings_path)          # shape (N, 2)

    print("  Loading modeled topics …")
    modeled = pd.read_csv(modeled_path, usecols=["id", "topic"])

    print("  Loading metadata …")
    meta = pd.read_csv(metadata_path, usecols=["id", "authorships.raw_author_name"])
    meta.columns = ["id", "authors_raw"]
    meta["authors_raw"] = meta["authors_raw"].fillna("")

    # Align embeddings with modeled (same row order assumed; verify)
    assert len(coords) == len(modeled), (
        f"Row count mismatch: embeddings {len(coords)} vs modeled {len(modeled)}"
    )
    modeled = modeled.reset_index(drop=True)
    modeled["x"] = coords[:, 0]
    modeled["y"] = coords[:, 1]

    # Merge author names onto modeled via paper id
    print("  Merging author names onto papers …")
    df = modeled.merge(meta, on="id", how="left")

    # Explode pipe-separated authors → one row per (paper, author)
    print("  Exploding author lists …")
    df = df[df["authors_raw"] != ""].copy()
    df["author_name"] = df["authors_raw"].str.split("|")
    df = df.explode("author_name")
    df["author_name"] = df["author_name"].str.strip()
    df = df[df["author_name"] != ""]

    # Aggregate per author
    print("  Aggregating per author …")
    agg = (
        df.groupby("author_name")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            n_papers=("id", "count"),
            topics=("topic", list),
        )
        .reset_index()
    )

    # Filter by minimum paper count
    agg = agg[agg["n_papers"] >= min_papers].copy()
    print(f"  {len(agg):,} authors with >= {min_papers} papers")
    return agg


def build_label_array(
    authors: pd.DataFrame,
    communities: pd.DataFrame,
    community_summary: pd.DataFrame | None,
    top_label_n: int = 200,
    strength_map: dict | None = None,
) -> np.ndarray:
    """
    Build a label array for datamapplot.

    Labels are community names (or IDs) for the top authors by strength/n_papers;
    unlabelled authors receive the datamapplot "noise" label (empty string).
    """
    # Merge community into authors
    authors = authors.merge(
        communities[["author_name", "community_id"]],
        on="author_name",
        how="left",
    )
    authors["community_id"] = authors["community_id"].fillna(-1).astype(int)

    # Determine label for each author: "Community {id}" or a named label
    if community_summary is not None and "label" in community_summary.columns:
        label_map = dict(
            zip(community_summary["community_id"], community_summary["label"])
        )
    else:
        label_map = {}

    def _community_label(cid: int) -> str:
        if cid == -1:
            return ""
        return label_map.get(cid, f"Community {cid}")

    # Rank authors — use strength if available, else n_papers
    if strength_map:
        authors["_rank"] = authors["author_name"].map(strength_map).fillna(0)
    else:
        authors["_rank"] = authors["n_papers"]

    # Only label the top-N by rank; rest get empty string (noise)
    top_set = set(
        authors.nlargest(top_label_n, "_rank")["author_name"].tolist()
    )

    labels = np.array([
        _community_label(row["community_id"]) if row["author_name"] in top_set else ""
        for _, row in authors.iterrows()
    ])
    return labels, authors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_author_dmp(
    embeddings_path: str,
    modeled_path: str,
    metadata_path: str,
    communities_path: str,
    metrics_path: str,
    topic_info_path: str | None = None,
    min_papers: int = 5,
    top_label_n: int = 200,
    output_file: str = "author_network_dmp.html",
    title: str = "Author Co-authorship Map",
    interactive: bool = True,
) -> None:
    """
    Build a DataMapPlot author map.

    Each author is placed at the mean UMAP coordinate of their papers.
    Authors who publish on similar topics cluster together, mirroring
    the layout of the BERTopic topic map.

    Parameters
    ----------
    min_papers : int
        Exclude authors with fewer than this many papers (reduces clutter).
    top_label_n : int
        Number of prominent authors to label (by collaboration strength).
    interactive : bool
        If True, output an interactive HTML file; otherwise a static PNG.
    """
    # ------------------------------------------------------------------
    # 1. Build per-author positions
    # ------------------------------------------------------------------
    author_pos = load_author_positions(
        embeddings_path=embeddings_path,
        modeled_path=modeled_path,
        metadata_path=metadata_path,
        min_papers=min_papers,
    )

    # ------------------------------------------------------------------
    # 2. Load supporting data
    # ------------------------------------------------------------------
    communities = pd.read_csv(communities_path)
    communities.columns = communities.columns.str.strip()
    communities["author_name"] = communities["author_name"].str.strip()

    metrics = pd.read_csv(metrics_path)
    metrics.columns = metrics.columns.str.strip()
    metrics["author_name"] = metrics["author_name"].str.strip()
    strength_map = dict(zip(metrics["author_name"], metrics["strength"]))

    # Optional: community summary with human-readable labels
    community_summary = None
    # (extend here if you have a community label CSV)

    # ------------------------------------------------------------------
    # 3. Build label array
    # ------------------------------------------------------------------
    labels, author_pos = build_label_array(
        authors=author_pos,
        communities=communities,
        community_summary=community_summary,
        top_label_n=top_label_n,
        strength_map=strength_map,
    )

    coords_2d = author_pos[["x", "y"]].values

    # ------------------------------------------------------------------
    # 4. Node size: sqrt of strength, then normalise to [10, 100]
    # ------------------------------------------------------------------
    strength_vals = np.array([
        strength_map.get(a, 1.0) for a in author_pos["author_name"]
    ])
    s = np.sqrt(strength_vals)
    marker_size = 10 + 90 * (s - s.min()) / (s.max() - s.min() + 1e-9)

    # ------------------------------------------------------------------
    # 5. Hover text (interactive mode): show author name + n_papers
    # ------------------------------------------------------------------
    hover_text = (
        author_pos["author_name"]
        + " ("
        + author_pos["n_papers"].astype(str)
        + " papers)"
    ).tolist()

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    print(f"  Rendering DataMapPlot for {len(coords_2d):,} authors …")

    if interactive:
        fig = datamapplot.create_interactive_plot(
            coords_2d,
            labels,
            hover_text=hover_text,
            marker_size_array=marker_size,
            title=title,
            sub_title=f"Positioned by topic similarity · {len(coords_2d):,} authors · min {min_papers} papers",
            point_radius_min_pixels=1,
            point_radius_max_pixels=12,
        )
        out_path = Path(output_file)
        if out_path.suffix.lower() != ".html":
            out_path = out_path.with_suffix(".html")
        fig.save(str(out_path))
        print(f"  ✓ Interactive map saved to {out_path}")

    else:
        fig, ax = datamapplot.create_plot(
            coords_2d,
            labels,
            marker_size_array=marker_size,
            title=title,
            sub_title=f"Positioned by topic similarity · {len(coords_2d):,} authors",
            figsize=(14, 12),
            dpi=150,
        )
        out_path = Path(output_file)
        if out_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            out_path = out_path.with_suffix(".png")
        fig.savefig(str(out_path), bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  ✓ Static map saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Author co-authorship map via DataMapPlot (UMAP-positioned)"
    )
    parser.add_argument(
        "--embeddings",
        default="data/embeddings_2d.npy",
        help="Path to embeddings_2d.npy",
    )
    parser.add_argument(
        "--modeled",
        default="data/causal_modeled_25Mar2026.csv",
        help="Path to modeled CSV (id, topic columns)",
    )
    parser.add_argument(
        "--metadata",
        default="data/metadata_deduplicated.csv",
        help="Path to metadata CSV (id, authorships.raw_author_name columns)",
    )
    parser.add_argument(
        "--communities",
        default="data/coauthorship_results/author_communities.csv",
    )
    parser.add_argument(
        "--metrics",
        default="data/coauthorship_results/author_network_metrics.csv",
    )
    parser.add_argument(
        "--topic-info",
        default=None,
        help="Optional topic_info CSV for named topic labels",
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        default=5,
        help="Exclude authors with fewer than N papers",
    )
    parser.add_argument(
        "--top-label-n",
        type=int,
        default=200,
        help="Number of top authors to label by community",
    )
    parser.add_argument(
        "--output",
        default="author_network_dmp.html",
        help="Output file (.html for interactive, .png for static)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Produce a static PNG instead of an interactive HTML",
    )
    parser.add_argument(
        "--title",
        default="Author Co-authorship Map",
    )
    args = parser.parse_args()

    visualize_author_dmp(
        embeddings_path=args.embeddings,
        modeled_path=args.modeled,
        metadata_path=args.metadata,
        communities_path=args.communities,
        metrics_path=args.metrics,
        topic_info_path=args.topic_info,
        min_papers=args.min_papers,
        top_label_n=args.top_label_n,
        output_file=args.output,
        title=args.title,
        interactive=not args.static,
    )


if __name__ == "__main__":
    main()
