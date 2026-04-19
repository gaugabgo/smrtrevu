"""
Topic community network — Plotly standalone panel (Layer 3).

Nodes  = one per topic, positioned at their UMAP centroids.
         Coloured and grouped by dominant global co-authorship community.
         Sized by number of papers (sqrt-scaled).

Edges  = cross-topic connections from topic_topic_edges.csv.
         Edge colour matches the dominant shared community.
         Edge width scales with connection weight.

The figure is saved as a self-contained HTML file using Plotly's CDN build.

Usage
-----
revu biblio visualize-topic-community-network \\
  --affinity data/coauthorship_results/global_community_topic/topic_community_affinity.csv \\
  --edges    data/coauthorship_results/global_community_topic/topic_topic_edges.csv \\
  --output   data/visualizations/topic_community_network.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

# Qualitative palette — up to 36 distinct colours before cycling
_BASE_PALETTE: list[str] = (
    plotly.colors.qualitative.Plotly
    + plotly.colors.qualitative.Alphabet
)
_OTHER_COLOR = "#AAAAAA"


def _build_color_map(community_ids: list[int]) -> dict[int, str]:
    """Assign a hex color to each community id.  -99 ('Other') → gray."""
    color_map: dict[int, str] = {}
    idx = 0
    for cid in sorted(community_ids):
        if cid == -99:
            color_map[cid] = _OTHER_COLOR
        else:
            color_map[cid] = _BASE_PALETTE[idx % len(_BASE_PALETTE)]
            idx += 1
    return color_map


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_topic_community_network(
    affinity_path: str,
    edges_path: str,
    output_path: str,
    top_n_edges: int | None = 300,
    min_edge_weight: float = 0.0,
    node_size_range: tuple[float, float] = (10.0, 45.0),
    edge_width_range: tuple[float, float] = (0.5, 4.0),
    edge_alpha: float = 0.35,
    title: str = "Topic Community Network",
) -> None:
    """
    Build an interactive Plotly topic-community network and save as HTML.

    Parameters
    ----------
    affinity_path : str
        topic_community_affinity.csv from agg-global-community-topic-affinity.
    edges_path : str
        topic_topic_edges.csv from the same script.
    output_path : str
        Destination HTML file.
    top_n_edges : int or None
        Keep only the top-N heaviest edges (avoids visual clutter).
    min_edge_weight : float
        Additional hard floor on edge weight.
    node_size_range : tuple
        (min_px, max_px) marker size for topic nodes (sqrt-scaled by n_papers).
    edge_width_range : tuple
        (min_px, max_px) line width for edges (scaled by weight).
    edge_alpha : float
        Opacity for edge lines (0–1).
    title : str
        Figure title shown at the top.
    """
    print("=" * 60)
    print("TOPIC COMMUNITY NETWORK")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & validate data
    # ------------------------------------------------------------------
    affinity = pd.read_csv(affinity_path)
    edges_df = pd.read_csv(edges_path)

    # One row per topic: dominant-community entry
    dominant = (
        affinity[affinity["is_dominant"]]
        .drop_duplicates("topic_id")
        .reset_index(drop=True)
    )
    print(f"  Topics: {len(dominant):,}")

    # Filter + limit edges
    edges_df = edges_df[edges_df["weight"] >= min_edge_weight].copy()
    if top_n_edges is not None:
        edges_df = edges_df.nlargest(top_n_edges, "weight").reset_index(drop=True)
    print(f"  Edges:  {len(edges_df):,}")

    # Community color map (sorted so colors are stable across runs)
    all_community_ids = sorted(affinity["global_community_id"].unique())
    color_map = _build_color_map(all_community_ids)

    # ------------------------------------------------------------------
    # 2. Node sizes (sqrt-scaled)
    # ------------------------------------------------------------------
    n_papers = dominant["topic_n_papers"].values.astype(float)
    s = np.sqrt(n_papers)
    s_min, s_max = s.min(), s.max()
    rmin, rmax = node_size_range
    node_sizes = rmin + (rmax - rmin) * (s - s_min) / max(s_max - s_min, 1e-9)

    # ------------------------------------------------------------------
    # 3. Edge widths (linear-scaled by weight)
    # ------------------------------------------------------------------
    if len(edges_df) > 0:
        w = edges_df["weight"].values.astype(float)
        w_min, w_max = w.min(), w.max()
        wmin_px, wmax_px = edge_width_range
        edge_widths = wmin_px + (wmax_px - wmin_px) * (w - w_min) / max(w_max - w_min, 1e-9)
    else:
        edge_widths = np.array([])

    # ------------------------------------------------------------------
    # 4. Build edge traces (one per shared community to group by colour)
    # ------------------------------------------------------------------
    topic_xy = dominant.set_index("topic_id")[["topic_x", "topic_y"]].to_dict("index")

    # Group edge segments by shared_community_id
    edge_groups: dict[int, list[tuple]] = {}
    for i, row in edges_df.iterrows():
        t1, t2 = int(row["topic_id_1"]), int(row["topic_id_2"])
        comm = int(row["shared_community_id"])
        if t1 not in topic_xy or t2 not in topic_xy:
            continue
        p1, p2 = topic_xy[t1], topic_xy[t2]
        width = float(edge_widths[i]) if len(edge_widths) > i else 1.0
        edge_groups.setdefault(comm, []).append(
            (p1["topic_x"], p1["topic_y"], p2["topic_x"], p2["topic_y"], width)
        )

    edge_traces: list[go.Scatter] = []
    for comm, segs in edge_groups.items():
        hex_color = color_map.get(comm, _OTHER_COLOR)
        rgba_color = _hex_to_rgba(hex_color, edge_alpha)
        # Use mean width for the trace (Plotly lines don't vary width per segment)
        mean_width = float(np.mean([s[4] for s in segs]))
        xs: list = []
        ys: list = []
        for x0, y0, x1, y1, _ in segs:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        edge_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=rgba_color, width=mean_width),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # ------------------------------------------------------------------
    # 5. Build node traces (one per dominant community for legend)
    # ------------------------------------------------------------------
    node_traces: list[go.Scatter] = []
    dominant = dominant.reset_index(drop=True)

    for cid, grp in dominant.groupby("global_community_id"):
        hex_color = color_map.get(int(cid), _OTHER_COLOR)
        comm_label = (
            f"Community {int(cid)}" if int(cid) != -99 else "Other communities"
        )
        grp_idx = grp.index.tolist()
        sizes = node_sizes[grp_idx].tolist()

        hover_texts = []
        for _, row in grp.iterrows():
            hover_texts.append(
                f"<b>{row['topic_label']}</b><br>"
                f"Topic {int(row['topic_id'])}<br>"
                f"Papers: {int(row['topic_n_papers']):,}<br>"
                f"Dominant community: {int(cid)}"
                + ("" if int(cid) == -99 else "<br>")
                + (
                    f"Authors in community: {int(row['n_authors']):,}<br>"
                    f"Fraction of topic: {row['fraction_of_topic']:.1%}"
                    if int(cid) != -99
                    else ""
                )
            )

        # Truncate labels for display (full label in hover)
        display_labels = grp["topic_label"].apply(
            lambda lbl: lbl[:35] + "…" if len(str(lbl)) > 35 else str(lbl)
        ).tolist()

        node_traces.append(
            go.Scatter(
                x=grp["topic_x"].values.tolist(),
                y=grp["topic_y"].values.tolist(),
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=hex_color,
                    line=dict(color="white", width=1.5),
                    opacity=0.88,
                ),
                text=display_labels,
                textposition="top center",
                textfont=dict(size=8, color="#333333"),
                name=comm_label,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        )

    # ------------------------------------------------------------------
    # 6. Layout and figure
    # ------------------------------------------------------------------
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=17, color="#222222"),
            x=0.5,
            xanchor="center",
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Global community", font=dict(size=12)),
            itemsizing="constant",
            bordercolor="#cccccc",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.92)",
        ),
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="UMAP dimension 1",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="UMAP dimension 2",
        ),
        plot_bgcolor="#f7f7f7",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=70, b=40),
    )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".html":
        out = out.with_suffix(".html")
    fig.write_html(str(out), include_plotlyjs="cdn")

    print(f"\n✓ Saved to {out}")
    print(f"  Node traces  : {len(node_traces):,} (one per community)")
    print(f"  Edge traces  : {len(edge_traces):,} (one per shared community)")
    print(f"  Topics shown : {len(dominant):,}")
