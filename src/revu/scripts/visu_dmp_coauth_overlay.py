"""
Topic DMP with within-topic co-authorship community overlay.

Renders a DataMapPlot interactive topic map and injects a toggleable canvas
overlay showing within-topic co-authorship community bubbles.  The bubbles are
zoom-synced via DataMapPlot's own ``window.datamap.onViewStateChange`` API and
the deck.gl viewport ``project`` method — the same mechanism used by DMP's
built-in annotation widget.

Layers (toggled with a checkbox panel):
  1. Topic map — always visible (standard DMP output)
  2. Within-topic co-authorship communities — overlay bubbles that appear on
     top of each topic cluster, one per detected community, sized by member
     count and colour-coded by rank within the topic.

Usage
-----
revu biblio visualize-dmp-coauth-overlay \\
  --modeled         data/causal_modeled_25Mar2026.csv \\
  --embeddings      data/embeddings_2d.npy \\
  --community-nodes data/within_topic_community_nodes.csv \\
  --topic-info      data/causal_modeled_1Apr2026_topic_info_customlabels.csv \\
  --metadata        data/causal_metadata_validated.csv \\
  --output          data/visualizations/topic_dmp_coauth_overlay.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import datamapplot
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colours by rank (0 = largest community in topic, 4 = 5th largest)
_RANK_COLORS = [
    "#191970",  
    "#2F4F4F",  
    "#556B2F",  
    "#483D8B",  
    "#708090",  
]

_RADIUS_PX = 9  # fixed bubble CSS-pixel radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_topic_labels(modeled_path: str, topic_info_path: str | None) -> list[str]:
    """
    Return one label per paper (in row order of modeled_path).

    Labels come from the ``Name`` column of topic_info_path, falling back to
    ``"Topic {id}"`` strings.  Topic -1 is always ``"Outlier"``.
    """
    modeled = pd.read_csv(modeled_path, usecols=["topic"])
    topics = modeled["topic"].tolist()

    label_map: dict[int, str] = {-1: "Outlier"}
    if topic_info_path and Path(topic_info_path).exists():
        info = pd.read_csv(topic_info_path, usecols=["Topic", "Name"])
        for _, row in info.iterrows():
            tid = int(row["Topic"])
            label_map[tid] = str(row["Name"]) if str(row["Name"]) != "-1" else "Outlier"

    return [
        label_map.get(t, f"Topic {t}") if t != -1 else "Outlier"
        for t in topics
    ]


def _load_hover_text(modeled_path: str, metadata_path: str | None, labels: list[str]) -> list[str]:
    """Build per-paper hover text strings."""
    modeled = pd.read_csv(modeled_path)

    if metadata_path and Path(metadata_path).exists():
        meta = pd.read_csv(metadata_path, dtype=str).fillna("")
        id_col = next((c for c in meta.columns if c.lower() == "id"), None)
        if id_col:
            merged = modeled.merge(meta, left_on="id", right_on=id_col, how="left").fillna("")
        else:
            merged = modeled
    else:
        merged = modeled

    texts = []
    for pos, row in merged.iterrows():
        parts = [f"Topic: {labels[pos]}"]
        for col, prefix in [
            ("title",                        "Title"),
            ("authorships.raw_author_name",  "Authors"),
            ("publication_year",             "Year"),
            ("cited_by_count",               "Citation Count"),
        ]:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                parts.append(f"{prefix}: {val}")
        if 'DOI' in row and row.get('DOI', '') and str(row['DOI']).strip() not in ('', 'nan'):
            parts.append(f"Link: https://doi.org/{str(row['DOI']).strip()}")
        if 'abstract' in row and pd.notna(row.get('abstract')) and row.get('abstract'):
            parts.append(f"Abstract: {row['abstract']}")
        texts.append("\n".join(parts))
    return texts


def _build_community_node_records(
    community_nodes: pd.DataFrame,
    coord_mean: np.ndarray,
    coord_scale: float,
) -> list[dict]:
    """
    Assign colours and pixel radii to community nodes and return a list of
    dicts ready for JSON serialisation.

    ``coord_mean`` and ``coord_scale`` must match the transform DMP applies
    internally before passing coordinates to deck.gl::

        dmp_x = (30.0 / raw_data_scale) * (raw_x - mean_x)
        dmp_y = (30.0 / raw_data_scale) * (raw_y - mean_y)

    so we apply the same here so that ``vp.project([dmp_x, dmp_y])`` lands
    at the correct screen position.
    """
    # Rank within each topic (0 = largest by n_members)
    community_nodes = community_nodes.copy()
    community_nodes["rank"] = (
        community_nodes
        .groupby("topic_id")["n_members"]
        .rank(method="first", ascending=False)
        .astype(int) - 1
    )

    # Apply the same centring + rescaling DMP uses
    factor = 30.0 / coord_scale
    community_nodes["x_dmp"] = factor * (community_nodes["x"] - coord_mean[0])
    community_nodes["y_dmp"] = factor * (community_nodes["y"] - coord_mean[1])

    radius_px = np.full(len(community_nodes), _RADIUS_PX, dtype=float)

    records = []
    for i, row in community_nodes.iterrows():
        rank = int(row["rank"])
        color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
        ordinal = _ORDINALS[min(rank, len(_ORDINALS) - 1)]
        top_authors = str(row.get("top_authors", ""))

        records.append({
            "x":         float(row["x_dmp"]),
            "y":         float(row["y_dmp"]),
            "color":     color,
            "ordinal":   ordinal,
            "radius_px": float(radius_px[community_nodes.index.get_loc(i)]),
            "authors":   top_authors,
            "n_members": int(row["n_members"]),
            "n_papers":  int(row["n_papers"]),
            "topic_id":  int(row["topic_id"]),
            "community_id": int(row["community_id"]),
        })
    return records


# ---------------------------------------------------------------------------
# Injected HTML / JS / CSS
# ---------------------------------------------------------------------------

_ORDINALS = ["1st", "2nd", "3rd", "4th", "5th"]


def _build_toggle_panel_html() -> str:
    legend_rows = "\n".join(
        f'    <span style="color:{color};font-size:16px;">●</span>'
        f' {_ORDINALS[i]} co-authorship community<br>'
        for i, color in enumerate(_RANK_COLORS)
    )
    return f"""
<div id="coauth-panel"
     style="position:fixed;bottom:32px;left:20px;z-index:300;
            background:rgba(255,255,255,0.93);border-radius:10px;
            padding:10px 14px;font-family:sans-serif;font-size:13px;
            box-shadow:0 2px 10px rgba(0,0,0,0.22);min-width:220px;">
  <div style="font-weight:700;margin-bottom:7px;font-size:14px;">Layers</div>
  <label style="cursor:pointer;display:flex;align-items:center;gap:6px;">
    <input type="checkbox" id="coauth-toggle"
           style="width:14px;height:14px;cursor:pointer;"
           onchange="window.coauthOverlay && window.coauthOverlay.toggle(this.checked)">
    Within-topic co-authorship communities
  </label>
  <div id="coauth-legend"
       style="display:none;margin-top:9px;font-size:11px;line-height:1.8;">
{legend_rows}
    <div style="margin-top:4px;color:#555;">Hover for details</div>
  </div>
</div>
<div id="coauth-tooltip"
     style="display:none;position:fixed;background:rgba(20,20,20,0.88);
            color:#fff;padding:9px 13px;border-radius:7px;font-size:12px;
            max-width:320px;z-index:400;pointer-events:none;
            line-height:1.5;font-family:sans-serif;"></div>
""" 
_TOGGLE_CSS = """
#coauth-panel input[type=checkbox] { accent-color: #E63946; }
"""

def _build_overlay_js(node_records: list[dict]) -> str:
    nodes_json = json.dumps(node_records, separators=(",", ":"))

    return f"""
(function () {{
  /* ── Community overlay data ── */
  const NODES = {nodes_json};

  /* ── Canvas element ── */
  const canvas = document.createElement('canvas');
  canvas.id = 'coauth-overlay-canvas';
  canvas.style.cssText =
    'position:fixed;top:0;left:0;width:100%;height:100%;' +
    'z-index:200;pointer-events:none;display:none;';
  document.body.appendChild(canvas);

  function sizeCanvas() {{
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth, h = window.innerHeight;
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width  = w + 'px';
    canvas.style.height = h + 'px';
    /* Scale once so all subsequent draw calls use CSS pixel coords */
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
  sizeCanvas();
  window.addEventListener('resize', () => {{ sizeCanvas(); draw(); }});

  /* ── Viewport helpers (mirrors annotation.js) ── */
  function getViewport() {{
    const dm = window.datamap;
    if (!dm) return null;
    return (
      dm.deckgl.viewManager?.getViewports()[0] ||
      dm.deckgl.getViewports?.()[0] ||
      null
    );
  }}

  function dataToScreen(x, y) {{
    const vp = getViewport();
    if (!vp) return null;
    return vp.project([x, y]);   // → [screenX, screenY]
  }}

  /* ── Draw ── */
  let visible = false;

  function draw() {{
    const ctx = canvas.getContext('2d');
    /* clearRect in CSS-pixel space (transform already applied by sizeCanvas) */
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    if (!visible) return;

    const vp = getViewport();
    if (!vp) return;

    for (const n of NODES) {{
      const pos = vp.project([n.x, n.y]);
      if (!pos) continue;
      const [sx, sy] = pos;   /* CSS pixels — no dpr multiplication */
      const r = n.radius_px;  /* already in CSS pixels */

      /* filled circle with white stroke */
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, 2 * Math.PI);
      ctx.fillStyle = n.color + 'BF';   /* ~75 % opacity */
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      /* abbreviated label above the bubble (only for larger bubbles) */
      if (r > 13) {{
        ctx.fillStyle = '#111111';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(n.short_label, sx, sy - r - 3);
      }}
    }}
  }}

  /* ── Tooltip ── */
  const tooltip = document.getElementById('coauth-tooltip');

  document.addEventListener('mousemove', (e) => {{
    if (!visible) {{ tooltip.style.display = 'none'; return; }}
    const vp = getViewport();
    if (!vp) return;

    let hovered = null;
    for (const n of NODES) {{
      const pos = vp.project([n.x, n.y]);
      if (!pos) continue;
      if (Math.hypot(e.clientX - pos[0], e.clientY - pos[1]) <= n.radius_px) {{
        hovered = n;
        break;
      }}
    }}

    if (hovered) {{
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top  = (e.clientY - 12) + 'px';
      tooltip.innerHTML =
        '<span style="color:' + hovered.color + ';font-size:16px;">●</span> ' +
        '<strong>' + hovered.ordinal + ' co-authorship community</strong><br>' +
        hovered.authors + '<br>' +
        '<span style="color:#aaa;">Members: ' + hovered.n_members + ' &nbsp;|&nbsp; Papers: ' + hovered.n_papers + '</span>';
    }} else {{
      tooltip.style.display = 'none';
    }}
  }});

  /* ── Toggle ── */
  function toggle(on) {{
    visible = (on !== undefined) ? on : !visible;
    canvas.style.display = visible ? 'block' : 'none';
    const legend = document.getElementById('coauth-legend');
    if (legend) legend.style.display = visible ? 'block' : 'none';
    if (visible) draw();
  }}
  window.coauthOverlay = {{ toggle, draw }};

  /* ── Hook into DMP viewstate changes (wait for datamap to be ready) ── */
  function hookDatamap() {{
    if (!window.datamap) {{ setTimeout(hookDatamap, 150); return; }}
    window.datamap.onViewStateChange('coauth-overlay', draw);
    /* initial draw after DMP has settled */
    setTimeout(draw, 800);
  }}
  hookDatamap();
}})();
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_dmp_with_coauth_overlay(
    modeled_path: str,
    embeddings_path: str,
    community_nodes_path: str,
    output_path: str,
    topic_info_path: str | None = None,
    metadata_path: str | None = None,
    title: str = "Topics in the Causal Inference Literature",
    noise_label: str = "Outlier",
    top_k_communities: int | None = None,
) -> None:
    """
    Build an interactive topic DMP with a toggleable within-topic co-authorship
    community overlay.

    Parameters
    ----------
    modeled_path : str
        Modeled CSV (``id``, ``topic`` columns at minimum).
    embeddings_path : str
        2-D UMAP embeddings ``.npy``; rows must match modeled_path.
    community_nodes_path : str
        Output of ``agg-within-topic-community-nodes`` step.
    output_path : str
        Destination HTML file.
    topic_info_path : str or None
        Topic-info CSV with ``Topic`` and ``Name`` columns for label text.
    metadata_path : str or None
        Metadata CSV merged for hover text (title, authors, year, citations).
    title : str
        Plot title shown in DMP.
    noise_label : str
        Label string used by DMP for noise / outlier points.
    top_k_communities : int or None
        If set, keep only the top-K communities per topic in the overlay
        (overrides whatever is in the CSV; useful to reduce visual clutter).
    """
    print("=" * 60)
    print("DMP WITH CO-AUTHORSHIP OVERLAY")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load embeddings + labels + hover text
    # ------------------------------------------------------------------
    print("\nLoading embeddings …")
    coords = np.load(embeddings_path)               # (N, 2)

    print("Building topic labels …")
    labels = _load_topic_labels(modeled_path, topic_info_path)
    assert len(labels) == len(coords), (
        f"Label count {len(labels)} != embedding rows {len(coords)}"
    )

    print("Building hover text …")
    hover_text = _load_hover_text(modeled_path, metadata_path, labels)

    # ------------------------------------------------------------------
    # 2. Load community nodes
    # ------------------------------------------------------------------
    print("Loading community nodes …")
    community_nodes = pd.read_csv(community_nodes_path)
    print(f"  {len(community_nodes):,} community nodes loaded")

    if top_k_communities is not None:
        community_nodes = (
            community_nodes
            .sort_values(["topic_id", "n_members"], ascending=[True, False])
            .groupby("topic_id", group_keys=False)
            .head(top_k_communities)
            .reset_index(drop=True)
        )
        print(f"  {len(community_nodes):,} after top-{top_k_communities} filter")

    # ------------------------------------------------------------------
    # Compute the same coordinate transform DMP applies internally:
    #   dmp_coords = (30.0 / raw_data_scale) * (raw_coords - mean(raw_coords))
    # We need these parameters so community node coordinates land correctly
    # when passed to vp.project() inside the deck.gl viewport.
    # ------------------------------------------------------------------
    from datamapplot.interactive_helpers import compute_percentile_bounds
    raw_bounds = compute_percentile_bounds(coords)
    raw_data_scale = max(raw_bounds[1] - raw_bounds[0], raw_bounds[3] - raw_bounds[2])
    coord_mean = np.mean(coords, axis=0)

    node_records = _build_community_node_records(community_nodes, coord_mean, raw_data_scale)

    # ------------------------------------------------------------------
    # 3. Build injection strings
    # ------------------------------------------------------------------
    overlay_js  = _build_overlay_js(node_records)
    toggle_html = _build_toggle_panel_html()
    toggle_css  = _TOGGLE_CSS

    # ------------------------------------------------------------------
    # 4. Render DMP
    # ------------------------------------------------------------------
    print(f"\nRendering DataMapPlot for {len(coords):,} papers …")
    fig = datamapplot.create_interactive_plot(
        coords,
        np.array(labels),
        hover_text=hover_text,
        title=title,
        sub_title=(
            f"Interactive data map of BERTopic-extracted topics and within-topic co-authorship communities"
        ),
        noise_label=noise_label,
        enable_search=False,
        initial_zoom_fraction=0.9,
        custom_html=toggle_html,
        custom_css=toggle_css,
        custom_js=overlay_js,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".html":
        out = out.with_suffix(".html")

    fig.save(str(out))
    print(f"\n✓ Saved to {out}")
    print(f"  Community nodes injected : {len(node_records):,}")
    print(f"  Topics with overlay      : {community_nodes['topic_id'].nunique():,}")
