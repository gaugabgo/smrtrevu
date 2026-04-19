"""
visu_coauthorship_hub_spoke.py

Two hub-and-spoke co-authorship network visualizations.

Graph 1 — topic-coloured (``visualize_topic_coauthorship_hub_spoke``):
  Hubs    = (topic, community) pairs from per-topic Louvain partitions.
  Spokes  = top-K authors per topic by within-topic collaboration strength.
  Colour  = topic  (all hubs/satellites sharing a topic share a colour).
  Filter  = top-N most prevalent topics by n_works.
  Data    = {coauthorship_dir}/topic_{id}/author_network_metrics.csv
            {coauthorship_dir}/topic_{id}/author_communities.csv
            {coauthorship_dir}/topic_network_summary.csv

Graph 2 — community-coloured (``visualize_global_coauthorship_hub_spoke``):
  Hubs    = global Louvain communities.
  Spokes  = top-N authors by corpus-wide collaboration strength.
  Colour  = community.
  Filter  = top-N authors by global strength; only communities with ≥1 such author shown.
  Data    = {coauthorship_dir}/author_network.graphml
            {coauthorship_dir}/author_network_metrics.csv
            {coauthorship_dir}/author_communities.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Topic labels — fall back gracefully if the main module is unavailable.
# ---------------------------------------------------------------------------
try:
    from revu.scripts.visu_bibliographic import CUSTOM_LABELS as _DEFAULT_LABELS
except ImportError:
    _DEFAULT_LABELS: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_topic_label(topic_id: int, topic_info_df: pd.DataFrame | None = None) -> str:
    """Resolve a human-readable topic label."""
    if topic_info_df is not None:
        row = topic_info_df[topic_info_df["Topic"] == topic_id]
        if not row.empty:
            return str(row.iloc[0]["Name"])
    return _DEFAULT_LABELS.get(topic_id, f"Topic {topic_id}")


def _load_topic_info(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None


def _make_color_map(ids: list[int], cmap_name: str = "tab20") -> dict[int, tuple]:
    """Map a sorted list of integer ids to distinct RGBA colours."""
    ids_sorted = sorted(set(ids))
    n = max(len(ids_sorted), 2)
    cmap = plt.cm.get_cmap(cmap_name, n)
    return {uid: cmap(i) for i, uid in enumerate(ids_sorted)}


def _radial_positions(center: np.ndarray, n: int, radius: float) -> list[np.ndarray]:
    """Return *n* positions evenly spaced on a circle around *center*."""
    if n == 0:
        return []
    return [
        center + radius * np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)])
        for i in range(n)
    ]


def _spring_layout(G: nx.Graph, k_scale: float = 2.5, seed: int = 42) -> dict:
    n = G.number_of_nodes()
    k = k_scale / np.sqrt(n) if n > 1 else 1.0
    return nx.spring_layout(G, k=k, iterations=120, weight="weight", seed=seed)


def _trunc(s: str, n: int = 26) -> str:
    return s[: n - 1] + "\u2026" if len(s) > n else s


# ---------------------------------------------------------------------------
# Shared draw routine
# ---------------------------------------------------------------------------

def _draw_hub_spoke(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    hub_ids: list,
    sat_ids: list,
    hub_colors: list,
    sat_colors: list,
    hub_sizes: np.ndarray,
    sat_sizes: np.ndarray,
    hub_label_pos: dict,
    hub_labels: dict,
    sat_labels: dict,
    title: str,
    legend_handles: list,
    legend_ncol: int = 1,
    edge_alpha: float = 0.25,
) -> None:
    """Render hubs, satellites, spokes, labels and legend onto *ax*."""
    ax.set_facecolor("#f8f8f8")
    ax.get_figure().patch.set_facecolor("#f8f8f8")

    # Spokes
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, edge_color="#888888", width=0.5)

    # Satellite nodes (drawn first so hubs sit on top)
    if sat_ids:
        nx.draw_networkx_nodes(
            G, pos, nodelist=sat_ids,
            node_size=list(sat_sizes), node_color=sat_colors,
            alpha=0.50, ax=ax,
        )

    # Hub nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=hub_ids,
        node_size=list(hub_sizes), node_color=hub_colors,
        alpha=0.92, ax=ax, edgecolors="white", linewidths=1.5,
    )

    # Satellite labels
    if sat_labels:
        nx.draw_networkx_labels(
            G, pos, labels=sat_labels,
            font_size=5, font_color="#555555", ax=ax,
        )

    # Hub labels (offset slightly upward)
    nx.draw_networkx_labels(
        G, hub_label_pos, labels=hub_labels,
        font_size=6.5, font_weight="bold", font_color="#111111", ax=ax,
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=6.5,
            framealpha=0.85,
            ncol=legend_ncol,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Graph 1: topic-coloured hub-spoke
# ---------------------------------------------------------------------------

def visualize_topic_coauthorship_hub_spoke(
    coauthorship_dir: str,
    top_n_topics: int = 20,
    top_k_per_topic: int = 10,
    label_top_authors: int = 3,
    output_file: str = "coauth_hub_topic.png",
    topic_info_file: str | None = None,
    figsize: tuple = (22, 20),
    seed: int = 42,
) -> None:
    """
    Graph 1: hub-spoke co-authorship network coloured by topic.

    Hubs    = (topic, community) pairs from per-topic Louvain partitions.
    Spokes  = top-K authors per topic by within-topic collaboration strength.
    Colour  = topic.
    Filter  = top-N topics by n_works (from topic_network_summary.csv).

    Layout (3-level):
      1. Topic centroids on a large circle.
      2. Community hubs radially around each topic centroid.
      3. Author satellites radially around each community hub.
    """
    coauth_dir = Path(coauthorship_dir)
    topic_info_df = _load_topic_info(topic_info_file)

    # ------------------------------------------------------------------
    # 1. Load topic prevalence ranking
    # ------------------------------------------------------------------
    summary_path = coauth_dir / "topic_network_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"topic_network_summary.csv not found in {coauth_dir}")

    summary = pd.read_csv(summary_path)
    summary.columns = summary.columns.str.strip()
    summary = summary[summary["topic"] >= 0]

    if "author_nodes" in summary.columns:
        summary = summary[summary["author_nodes"] > 0]

    summary = summary.sort_values("n_works", ascending=False).head(top_n_topics)
    selected_topics: list[int] = summary["topic"].astype(int).tolist()
    print(f"  Selected {len(selected_topics)} topics (top-{top_n_topics} by n_works)")

    # ------------------------------------------------------------------
    # 2. Load per-topic author + community data
    # ------------------------------------------------------------------
    # hub_records: one entry per (topic, community) pair that has ≥1 top-K author
    hub_records: list[dict] = []

    for topic_id in selected_topics:
        topic_dir = coauth_dir / f"topic_{topic_id}"
        metrics_path = topic_dir / "author_network_metrics.csv"
        communities_path = topic_dir / "author_communities.csv"

        if not metrics_path.exists() or not communities_path.exists():
            print(f"  ⚠ Topic {topic_id}: missing files — skipping")
            continue

        metrics = pd.read_csv(metrics_path)
        communities = pd.read_csv(communities_path)
        for df in (metrics, communities):
            df.columns = df.columns.str.strip()
        metrics["author_name"] = metrics["author_name"].str.strip()
        communities["author_name"] = communities["author_name"].str.strip()

        top_k = metrics.nlargest(top_k_per_topic, "strength")
        merged = top_k.merge(communities, on="author_name", how="left")
        merged["community_id"] = merged["community_id"].fillna(-1).astype(int)

        for comm_id, grp in merged.groupby("community_id"):
            hub_records.append(
                {
                    "hub_id": f"hub_{topic_id}_{comm_id}",
                    "topic_id": topic_id,
                    "community_id": int(comm_id),
                    "authors": grp[["author_name", "strength"]].to_dict("records"),
                    "n_members": len(grp),
                }
            )

    if not hub_records:
        print("  ⚠ No hub records — check coauthorship_dir structure.")
        return

    n_hubs_total = len(hub_records)
    print(f"  {n_hubs_total} (topic, community) hubs across {len(selected_topics)} topics")

    # ------------------------------------------------------------------
    # 3. Build hub-spoke graph
    # ------------------------------------------------------------------
    G = nx.Graph()
    for rec in hub_records:
        G.add_node(
            rec["hub_id"],
            node_type="hub",
            topic_id=rec["topic_id"],
            community_id=rec["community_id"],
            n_members=rec["n_members"],
        )
        for auth in rec["authors"]:
            sat_id = f"sat_{rec['topic_id']}_{auth['author_name']}"
            G.add_node(
                sat_id,
                node_type="satellite",
                topic_id=rec["topic_id"],
                author_name=auth["author_name"],
                strength=auth["strength"],
            )
            G.add_edge(rec["hub_id"], sat_id)

    # ------------------------------------------------------------------
    # 4. Colours (by topic)
    # ------------------------------------------------------------------
    topic_color_map = _make_color_map(selected_topics)
    hub_ids = [n for n in G.nodes if G.nodes[n]["node_type"] == "hub"]
    sat_ids = [n for n in G.nodes if G.nodes[n]["node_type"] == "satellite"]
    hub_colors = [topic_color_map[G.nodes[n]["topic_id"]] for n in hub_ids]
    sat_colors = [topic_color_map[G.nodes[n]["topic_id"]] for n in sat_ids]

    # ------------------------------------------------------------------
    # 5. Node sizes
    # ------------------------------------------------------------------
    hub_n = np.array([G.nodes[n]["n_members"] for n in hub_ids], dtype=float)
    hub_sizes = 300.0 + 2000.0 * (hub_n / hub_n.max())

    sat_str = np.array([G.nodes[n]["strength"] for n in sat_ids], dtype=float)
    s_norm = np.sqrt(sat_str)
    denom = s_norm.max() - s_norm.min() + 1e-9
    sat_sizes = 25.0 + 350.0 * (s_norm - s_norm.min()) / denom

    # ------------------------------------------------------------------
    # 6. Layout — spring on hub nodes + rank-based satellite placement
    # ------------------------------------------------------------------

    # Index hubs by topic (needed here and in the labels step)
    topic_to_hub_ids: dict[int, list[str]] = {}
    for rec in hub_records:
        topic_to_hub_ids.setdefault(rec["topic_id"], []).append(rec["hub_id"])

    # Build a hub-only graph with strong phantom edges connecting hubs of the
    # same topic so the spring layout clusters them tightly together.
    hub_graph = nx.Graph()
    for rec in hub_records:
        hub_graph.add_node(rec["hub_id"])
    for hub_id_list in topic_to_hub_ids.values():
        for i in range(len(hub_id_list)):
            for j in range(i + 1, len(hub_id_list)):
                hub_graph.add_edge(hub_id_list[i], hub_id_list[j], weight=8.0)

    n_hubs = hub_graph.number_of_nodes()
    k_val = 1.8 / np.sqrt(n_hubs) if n_hubs > 1 else 1.0
    hub_raw_pos = nx.spring_layout(
        hub_graph, k=k_val, iterations=180, weight="weight", seed=seed
    )
    pos: dict = {hub_id: np.array(p) for hub_id, p in hub_raw_pos.items()}

    # Satellite placement: rank-based radius so top author (by strength) sits
    # closest to the hub and the weakest author sits farthest out.
    # This makes edge length encode strength and keeps the layout clean.
    _SAT_INNER = 0.18   # radius for the strongest author in a community
    _SAT_OUTER = 0.38   # radius for the weakest author in a community

    hub_id_to_rec: dict[str, dict] = {r["hub_id"]: r for r in hub_records}
    for hub_id, rec in hub_id_to_rec.items():
        hub_center = pos[hub_id]
        authors = rec["authors"]   # already sorted by strength descending
        n_s = len(authors)
        for rank, auth in enumerate(authors):
            frac = rank / max(n_s - 1, 1)
            radius = _SAT_INNER + frac * (_SAT_OUTER - _SAT_INNER)
            angle = 2 * np.pi * rank / n_s
            sat_id = f"sat_{rec['topic_id']}_{auth['author_name']}"
            pos[sat_id] = hub_center + radius * np.array([np.cos(angle), np.sin(angle)])

    # ------------------------------------------------------------------
    # 7. Labels
    # ------------------------------------------------------------------
    hub_labels: dict = {}
    for rec in hub_records:
        label = _trunc(_get_topic_label(rec["topic_id"], topic_info_df))
        if len(topic_to_hub_ids[rec["topic_id"]]) > 1:
            label += f"\nC{rec['community_id']}"
        hub_labels[rec["hub_id"]] = label

    hub_label_pos = {k: pos[k] + np.array([0.0, 0.06]) for k in hub_labels}

    sat_labels: dict = {}
    for rec in hub_records:
        sorted_authors = sorted(rec["authors"], key=lambda x: x["strength"], reverse=True)
        for auth in sorted_authors[:label_top_authors]:
            sat_id = f"sat_{rec['topic_id']}_{auth['author_name']}"
            sat_labels[sat_id] = auth["author_name"]

    # ------------------------------------------------------------------
    # 8. Legend
    # ------------------------------------------------------------------
    patches = [
        mpatches.Patch(
            color=topic_color_map[tid],
            label=_trunc(_get_topic_label(tid, topic_info_df), 38),
        )
        for tid in selected_topics
    ]

    # ------------------------------------------------------------------
    # 9. Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    _draw_hub_spoke(
        ax=ax, G=G, pos=pos,
        hub_ids=hub_ids, sat_ids=sat_ids,
        hub_colors=hub_colors, sat_colors=sat_colors,
        hub_sizes=hub_sizes, sat_sizes=sat_sizes,
        hub_label_pos=hub_label_pos, hub_labels=hub_labels,
        sat_labels=sat_labels,
        title=f"Co-authorship Communities by Topic \u2014 Top {len(selected_topics)} Topics",
        legend_handles=patches,
        legend_ncol=2 if len(patches) > 10 else 1,
        edge_alpha=0.40,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved to {output_file}")


# ---------------------------------------------------------------------------
# Graph 2: community-coloured global hub-spoke
# ---------------------------------------------------------------------------

def visualize_global_coauthorship_hub_spoke(
    graphml_file: str,
    metrics_file: str,
    communities_file: str,
    top_n: int = 100,
    label_top_n: int | None = None,
    output_file: str = "coauth_hub_global.png",
    figsize: tuple = (18, 16),
    seed: int = 42,
) -> None:
    """
    Graph 2: hub-spoke co-authorship network coloured by global community.

    Hubs    = global Louvain communities (one hub per community with ≥1 top-N author).
    Spokes  = top-N authors by global collaboration strength.
    Colour  = community.
    Filter  = top-N authors by global strength.

    Parameters
    ----------
    label_top_n : int or None
        How many top authors (by global strength) to label by name.
        ``None`` (default) labels every satellite node in the graph.

    Layout (2-level):
      1. Community hubs positioned by spring layout on a community-level graph
         whose edge weights = cross-community co-authorship counts in the top-N
         subgraph.
      2. Author satellites radially around their community hub, using rank-based
         radii so stronger authors sit closer to their hub.
    """
    # ------------------------------------------------------------------
    # 1. Load metrics and communities
    # ------------------------------------------------------------------
    print(f"  Loading metrics from {metrics_file} \u2026")
    metrics = pd.read_csv(metrics_file)
    metrics.columns = metrics.columns.str.strip()
    metrics["author_name"] = metrics["author_name"].str.strip()

    communities = pd.read_csv(communities_file)
    communities.columns = communities.columns.str.strip()
    communities["author_name"] = communities["author_name"].str.strip()

    top_authors_df = metrics.nlargest(top_n, "strength")
    top_author_set: set[str] = set(top_authors_df["author_name"].tolist())
    print(f"  Selected {len(top_author_set)} top authors by global strength")

    comm_map: dict[str, int] = dict(
        zip(communities["author_name"], communities["community_id"])
    )
    author_comm: dict[str, int] = {
        a: comm_map[a] for a in top_author_set if a in comm_map
    }
    hub_communities: list[int] = sorted({c for c in author_comm.values()})
    print(f"  {len(hub_communities)} communities represented among top-{top_n} authors")

    # ------------------------------------------------------------------
    # 2. Load GraphML — robust to leading-space node IDs
    # ------------------------------------------------------------------
    print(f"  Loading GraphML from {graphml_file} \u2026")
    G_full = nx.read_graphml(graphml_file)
    print(f"  Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

    # Build stripped-name → original-name mapping to handle whitespace in node IDs
    stripped_to_orig: dict[str, str] = {n.strip(): n for n in G_full.nodes()}
    top_authors_in_graphml: set[str] = {
        stripped_to_orig[a] for a in top_author_set if a in stripped_to_orig
    }
    G_sub = G_full.subgraph(top_authors_in_graphml).copy()
    print(f"  Top-{top_n} subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # 3. Build community-level graph for spring layout
    # ------------------------------------------------------------------
    comm_graph = nx.Graph()
    for cid in hub_communities:
        comm_graph.add_node(cid)

    # Map original graphml node IDs back to stripped names for community lookup
    orig_to_stripped: dict[str, str] = {v: k for k, v in stripped_to_orig.items()}
    for u, v, data in G_sub.edges(data=True):
        cu = author_comm.get(orig_to_stripped.get(u, u), -1)
        cv = author_comm.get(orig_to_stripped.get(v, v), -1)
        if cu != -1 and cv != -1 and cu != cv:
            w = float(data.get("weight", 1.0))
            if comm_graph.has_edge(cu, cv):
                comm_graph[cu][cv]["weight"] += w
            else:
                comm_graph.add_edge(cu, cv, weight=w)

    # ------------------------------------------------------------------
    # 4. Layout — spring on community hubs, radial for satellites
    # ------------------------------------------------------------------
    print("  Computing hub layout \u2026")
    hub_pos_raw: dict[int, np.ndarray] = {
        cid: np.array(p) for cid, p in _spring_layout(comm_graph, seed=seed).items()
    }

    # Group authors by community, sorted by strength (descending)
    comm_to_authors: dict[int, list[tuple[str, float]]] = {}
    strength_lookup: dict[str, float] = dict(
        zip(metrics["author_name"], metrics["strength"])
    )
    for author, cid in author_comm.items():
        comm_to_authors.setdefault(cid, []).append(
            (author, float(strength_lookup.get(author, 1.0)))
        )
    for cid in comm_to_authors:
        comm_to_authors[cid].sort(key=lambda x: x[1], reverse=True)

    # Rank-based satellite radii: strongest author closest to hub, weakest farthest.
    _SAT_INNER = 0.14
    _SAT_OUTER = 0.36

    pos: dict = {}
    for cid in hub_communities:
        hub_center = hub_pos_raw[cid]
        pos[f"hub_{cid}"] = hub_center
        authors = comm_to_authors.get(cid, [])   # sorted by strength descending
        n_s = len(authors)
        for rank, (author, _) in enumerate(authors):
            frac = rank / max(n_s - 1, 1)
            radius = _SAT_INNER + frac * (_SAT_OUTER - _SAT_INNER)
            angle = 2 * np.pi * rank / n_s
            pos[f"sat_{author}"] = hub_center + radius * np.array(
                [np.cos(angle), np.sin(angle)]
            )

    # ------------------------------------------------------------------
    # 5. Build hub-spoke graph
    # ------------------------------------------------------------------
    G = nx.Graph()
    for cid in hub_communities:
        G.add_node(
            f"hub_{cid}",
            node_type="hub",
            community_id=cid,
            n_members=len(comm_to_authors.get(cid, [])),
        )
        for author, strength in comm_to_authors.get(cid, []):
            sat_id = f"sat_{author}"
            G.add_node(
                sat_id,
                node_type="satellite",
                author_name=author,
                strength=strength,
                community_id=cid,
            )
            G.add_edge(f"hub_{cid}", sat_id)

    hub_ids = [n for n in G.nodes if G.nodes[n]["node_type"] == "hub"]
    sat_ids = [n for n in G.nodes if G.nodes[n]["node_type"] == "satellite"]

    # ------------------------------------------------------------------
    # 6. Colours (by community)
    # ------------------------------------------------------------------
    community_color_map = _make_color_map(hub_communities)
    hub_colors = [community_color_map[G.nodes[n]["community_id"]] for n in hub_ids]
    sat_colors = [community_color_map[G.nodes[n]["community_id"]] for n in sat_ids]

    # ------------------------------------------------------------------
    # 7. Node sizes
    # ------------------------------------------------------------------
    hub_n = np.array([G.nodes[n]["n_members"] for n in hub_ids], dtype=float)
    hub_sizes = 400.0 + 3000.0 * (hub_n / hub_n.max())

    sat_str = np.array([G.nodes[n]["strength"] for n in sat_ids], dtype=float)
    s_norm = np.sqrt(sat_str)
    denom = s_norm.max() - s_norm.min() + 1e-9
    sat_sizes = 30.0 + 400.0 * (s_norm - s_norm.min()) / denom

    # ------------------------------------------------------------------
    # 8. Labels
    # ------------------------------------------------------------------
    hub_label_map: dict = {}
    for cid in hub_communities:
        authors = comm_to_authors.get(cid, [])
        top_author = _trunc(authors[0][0], 20) if authors else ""
        hub_label_map[f"hub_{cid}"] = f"C{cid}\n{top_author}"

    hub_label_pos = {k: pos[k] + np.array([0.0, 0.04]) for k in hub_label_map}

    # Label all satellite nodes by default; restrict to top-label_top_n when set.
    if label_top_n is None:
        sat_label_map: dict = {
            sat_id: G.nodes[sat_id]["author_name"] for sat_id in sat_ids
        }
    else:
        top_sat_names: set[str] = set(
            metrics.nlargest(label_top_n, "strength")["author_name"].tolist()
        )
        sat_label_map: dict = {
            f"sat_{n}": n for n in top_sat_names if f"sat_{n}" in G.nodes
        }

    # ------------------------------------------------------------------
    # 9. Legend
    # ------------------------------------------------------------------
    patches = [
        mpatches.Patch(color=community_color_map[cid], label=f"Community {cid}")
        for cid in hub_communities
    ]

    # ------------------------------------------------------------------
    # 10. Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    _draw_hub_spoke(
        ax=ax, G=G, pos=pos,
        hub_ids=hub_ids, sat_ids=sat_ids,
        hub_colors=hub_colors, sat_colors=sat_colors,
        hub_sizes=hub_sizes, sat_sizes=sat_sizes,
        hub_label_pos=hub_label_pos, hub_labels=hub_label_map,
        sat_labels=sat_label_map,
        title=f"Global Co-authorship Communities \u2014 Top {top_n} Authors",
        legend_handles=patches,
        legend_ncol=2 if len(patches) > 10 else 1,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved to {output_file}")
