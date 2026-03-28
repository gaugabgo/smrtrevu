"""
Author co-authorship network visualization using NetworkX.

Loads the top-N authors by collaboration strength from GraphML,
lays out the subgraph, and renders nodes sized by strength and
coloured by community.

Usage
-----
python src/revu/scripts/visu_author_network_nx.py \
    --graphml  data/coauthorship_results/author_network.graphml \
    --metrics  data/coauthorship_results/author_network_metrics.csv \
    --communities data/coauthorship_results/author_communities.csv \
    --top-n 150 \
    --output  author_network_nx.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _spring_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Tuned spring layout — reasonable for subgraphs up to ~150 nodes."""
    n = G.number_of_nodes()
    k = 2.5 / np.sqrt(n) if n > 1 else 1.0
    return nx.spring_layout(G, k=k, iterations=120, weight="weight", seed=seed)


def _kamada_kawai_layout(G: nx.Graph) -> dict:
    """Kamada-Kawai — cleaner for small/medium graphs, ignores edge weights."""
    return nx.kamada_kawai_layout(G)


def _graphviz_layout(G: nx.Graph, prog: str = "neato") -> dict:
    """Graphviz layout (requires pygraphviz + graphviz binaries)."""
    from networkx.drawing.nx_agraph import graphviz_layout
    return graphviz_layout(G, prog=prog)


def compute_layout(G: nx.Graph, method: str = "spring") -> dict:
    """
    Compute 2-D node positions.

    Parameters
    ----------
    method : str
        One of 'spring', 'kamada_kawai', 'graphviz_neato', 'graphviz_fdp'.
        Falls back to spring if graphviz is unavailable.
    """
    if method == "kamada_kawai":
        return _kamada_kawai_layout(G)
    if method.startswith("graphviz"):
        prog = method.split("_", 1)[1] if "_" in method else "neato"
        try:
            return _graphviz_layout(G, prog=prog)
        except (ImportError, Exception) as exc:
            print(f"  ⚠ graphviz layout unavailable ({exc}); falling back to spring")
    return _spring_layout(G)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize_author_network(
    graphml_file: str,
    metrics_file: str,
    communities_file: str,
    top_n: int = 75,
    output_file: str = "author_network_nx.png",
    layout: str = "spring",
    max_edge_width: float = 6.0,
    label_top_n: int = 30,
    figsize: tuple = (16, 14),
    seed: int = 42,
) -> None:
    """
    Render a co-authorship subgraph for the top-N authors.

    Parameters
    ----------
    graphml_file : str
        Path to author_network.graphml.
    metrics_file : str
        Path to author_network_metrics.csv (columns: author_name, strength, …).
    communities_file : str
        Path to author_communities.csv (columns: author_name, community_id).
    top_n : int
        Number of highest-strength authors to include.
    output_file : str
        Destination PNG path.
    layout : str
        'spring' | 'kamada_kawai' | 'graphviz_neato' | 'graphviz_fdp'.
    max_edge_width : float
        Edge width is scaled to [0.3, max_edge_width].
    label_top_n : int
        Show author name labels only for the top-N nodes by strength.
    figsize : tuple
        Matplotlib figure size.
    seed : int
        Random seed for reproducible layouts.
    """
    # ------------------------------------------------------------------
    # 1. Load metrics and community assignments
    # ------------------------------------------------------------------
    metrics = pd.read_csv(metrics_file)
    communities = pd.read_csv(communities_file)

    # Normalise column names (strip leading/trailing spaces)
    metrics.columns = metrics.columns.str.strip()
    communities.columns = communities.columns.str.strip()
    metrics["author_name"] = metrics["author_name"].str.strip()
    communities["author_name"] = communities["author_name"].str.strip()

    # Select top-N by strength
    top_authors = (
        metrics.nlargest(top_n, "strength")["author_name"].tolist()
    )
    top_set = set(top_authors)
    print(f"  Selected {len(top_authors)} authors by strength")

    # Build community colour map
    comm_map: dict[str, int] = dict(
        zip(communities["author_name"], communities["community_id"])
    )
    unique_communities = sorted(
        {comm_map.get(a, -1) for a in top_authors}
    )
    # Use a qualitative colormap with enough distinct colours
    cmap = plt.cm.get_cmap("tab20", max(len(unique_communities), 2))
    comm_color = {c: cmap(i) for i, c in enumerate(unique_communities)}

    # ------------------------------------------------------------------
    # 2. Load GraphML and build subgraph
    # ------------------------------------------------------------------
    print(f"  Loading GraphML from {graphml_file} …")
    G_full = nx.read_graphml(graphml_file)
    print(f"  Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

    # Keep only nodes in top_set; edges where both endpoints are in top_set
    G = G_full.subgraph(top_set).copy()
    print(f"  Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        print("  ⚠ No matching nodes found in GraphML. Check author name formats.")
        return

    # ------------------------------------------------------------------
    # 3. Attach visual attributes
    # ------------------------------------------------------------------
    strength_map: dict[str, float] = dict(
        zip(metrics["author_name"], metrics["strength"])
    )

    node_strength = np.array([strength_map.get(n, 1.0) for n in G.nodes()])
    # Scale node size: sqrt compression so outliers don't dominate
    s_norm = np.sqrt(node_strength)
    node_sizes = 100 + 2500 * (s_norm - s_norm.min()) / (s_norm.max() - s_norm.min() + 1e-9)

    node_colors = [
        comm_color.get(comm_map.get(n, -1), (0.7, 0.7, 0.7, 1.0))
        for n in G.nodes()
    ]

    # Edge widths proportional to co-authorship weight
    edge_weights = np.array([
        G[u][v].get("weight", 1.0) for u, v in G.edges()
    ])
    if edge_weights.max() > edge_weights.min():
        ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
    else:
        ew_norm = np.ones_like(edge_weights)
    edge_widths = 0.3 + ew_norm * (max_edge_width - 0.3)

    # ------------------------------------------------------------------
    # 4. Layout
    # ------------------------------------------------------------------
    print(f"  Computing layout: {layout} …")
    pos = compute_layout(G, method=layout)

    # ------------------------------------------------------------------
    # 5. Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#f8f8f8")

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.35,
        edge_color="#888888",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        linewidths=0.5,
        edgecolors="white",
        ax=ax,
    )

    # Label only the top-label_top_n nodes to avoid clutter
    label_authors = set(
        metrics.nlargest(label_top_n, "strength")["author_name"].tolist()
    ) & set(G.nodes())
    label_dict = {n: n for n in label_authors}
    nx.draw_networkx_labels(
        G, pos,
        labels=label_dict,
        font_size=7,
        font_weight="bold",
        ax=ax,
    )

    # ------------------------------------------------------------------
    # 6. Legend for communities
    # ------------------------------------------------------------------
    patches = [
        mpatches.Patch(
            color=comm_color[c],
            label=f"Community {c}" if c != -1 else "Unassigned",
        )
        for c in unique_communities
    ]
    ax.legend(
        handles=patches,
        title="Community",
        loc="lower left",
        fontsize=7,
        title_fontsize=8,
        framealpha=0.8,
    )

    ax.set_title(
        f"Co-authorship Network — Top {top_n} Authors by Collaboration Strength",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize co-authorship network (top-N authors, NetworkX)"
    )
    parser.add_argument(
        "--graphml",
        default="data/coauthorship_results/author_network.graphml",
        help="Path to author_network.graphml",
    )
    parser.add_argument(
        "--metrics",
        default="data/coauthorship_results/author_network_metrics.csv",
        help="Path to author_network_metrics.csv",
    )
    parser.add_argument(
        "--communities",
        default="data/coauthorship_results/author_communities.csv",
        help="Path to author_communities.csv",
    )
    parser.add_argument("--top-n", type=int, default=75, help="Number of top authors to show")
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai", "graphviz_neato", "graphviz_fdp"],
        default="spring",
        help="Graph layout algorithm",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=30,
        help="Show labels for this many top authors only",
    )
    parser.add_argument(
        "--output",
        default="author_network_nx.png",
        help="Output PNG file",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    visualize_author_network(
        graphml_file=args.graphml,
        metrics_file=args.metrics,
        communities_file=args.communities,
        top_n=args.top_n,
        output_file=args.output,
        layout=args.layout,
        label_top_n=args.label_top_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
