
import click
import pandas as pd
import networkx as nx
import numpy as np
import os
import itertools
import pycountry
import plotly.express as px
from itertools import zip_longest
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class CoauthorshipAnalyzer:
    """
    Analyzes co-authorship patterns from scholarly work metadata.
    Builds author and institution networks, detects communities, and maps geographic distribution.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        topic_df: Optional[pd.DataFrame] = None,
        id_column: str = 'id',
        topic_column: str = 'topic',
        author_column: str = 'authorships.raw_author_name',
        affiliation_column: str = 'authorships.raw_affiliation_strings',
        country_column: str = 'authorships.countries',
        year_column: str = 'publication_year',
    ):
        """
        Parameters
        ----------
        metadata_df : pd.DataFrame
            Works metadata. Author names, affiliations, and countries are expected
            as pipe-delimited strings where position i in each column corresponds
            to the same author.
        topic_df : pd.DataFrame, optional
            DataFrame with topic assignments (must share id_column with metadata_df).
            When provided it is merged into metadata_df so that per-topic analysis
            is available.
        id_column : str
            Column with unique work identifier.
        topic_column : str
            Column in topic_df (or metadata_df) containing topic assignments.
        author_column : str
            Pipe-delimited author names.
        affiliation_column : str
            Pipe-delimited affiliation strings (one slot per author; slots may
            contain multiple institutions separated by ";").
        country_column : str
            Pipe-delimited ISO 3166-1 alpha-2 country codes, one per author.
        year_column : str
            Publication year column.
        """
        self.id_col = id_column
        self.topic_col = topic_column
        self.author_col = author_column
        self.affil_col = affiliation_column
        self.country_col = country_column
        self.year_col = year_column

        if topic_df is not None:
            print(f"Merging topic data ({len(topic_df)} rows) with metadata ({len(metadata_df)} rows)...")
            self.df = pd.merge(
                metadata_df,
                topic_df[[id_column, topic_column]].drop_duplicates(subset=[id_column]),
                on=id_column,
                how='inner',
                suffixes=('', '_topic'),
            )
            print(f"✓ Merged dataset: {len(self.df)} works")
        else:
            self.df = metadata_df.copy()

        self._author_paper_table: Optional[pd.DataFrame] = None
        self._coauthorship_edges: Optional[pd.DataFrame] = None

        # Warn if optional columns are absent
        for col in [affiliation_column, country_column]:
            if col not in self.df.columns:
                print(f"Warning: column '{col}' not found — related analyses will be skipped.")

        print(f"CoauthorshipAnalyzer initialized with {len(self.df)} works.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_pipe(value) -> List[Optional[str]]:
        """Split a pipe-delimited value; empty slots become None."""
        if pd.isna(value) or str(value).strip() == '':
            return []
        return [s if s.strip() != '' else None for s in str(value).split('|')]

    # ------------------------------------------------------------------
    # Table A — Author–Paper Edge List
    # ------------------------------------------------------------------

    def build_author_paper_table(self) -> pd.DataFrame:
        """
        Parse pipe-delimited authorship fields into one row per (work, author).

        Returns
        -------
        pd.DataFrame with columns:
            id, author_name, primary_affiliation, country, publication_year
        """
        rows = []
        has_affil = self.affil_col in self.df.columns
        has_country = self.country_col in self.df.columns

        for _, row in self.df.iterrows():
            work_id = row[self.id_col]
            year = row.get(self.year_col, None)

            authors = self._split_pipe(row.get(self.author_col, ''))
            affils = self._split_pipe(row.get(self.affil_col, '')) if has_affil else []
            countries = self._split_pipe(row.get(self.country_col, '')) if has_country else []

            for author, affil, country in zip_longest(authors, affils, countries, fillvalue=None):
                # Empty string slots already converted to None by _split_pipe
                rows.append({
                    'id': work_id,
                    'author_name': author,
                    'primary_affiliation': affil,
                    'country': country,
                    'publication_year': year,
                })

        self._author_paper_table = pd.DataFrame(rows)
        print(f"Author–paper table: {len(self._author_paper_table)} rows "
              f"({self._author_paper_table['author_name'].notna().sum()} with names)")
        return self._author_paper_table

    # ------------------------------------------------------------------
    # Table B — Co-authorship Edge List
    # ------------------------------------------------------------------

    def build_coauthorship_edges(self) -> pd.DataFrame:
        """
        Generate one row per (author_A, author_B, work_id) co-authorship pair.

        Pairs are normalized: author_A = min(a, b), author_B = max(a, b).
        None / blank author slots are excluded before generating pairs.

        Returns
        -------
        pd.DataFrame with columns: author_A, author_B, work_id, year
        """
        if self._author_paper_table is None:
            self.build_author_paper_table()

        apt = self._author_paper_table
        rows = []

        for work_id, group in apt.groupby('id'):
            authors = group['author_name'].dropna().tolist()
            year = group['publication_year'].iloc[0] if len(group) > 0 else None

            for a, b in itertools.combinations(authors, 2):
                author_a, author_b = (min(a, b), max(a, b))
                rows.append({
                    'author_A': author_a,
                    'author_B': author_b,
                    'work_id': work_id,
                    'year': year,
                })

        self._coauthorship_edges = pd.DataFrame(rows)
        print(f"Co-authorship edge list: {len(self._coauthorship_edges)} raw pairs")
        return self._coauthorship_edges

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_giant_component(G: nx.Graph) -> nx.Graph:
        if G.number_of_nodes() == 0:
            return G
        components = list(nx.connected_components(G))
        giant_nodes = max(components, key=len)
        return G.subgraph(giant_nodes).copy()

    @staticmethod
    def _compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
        """Degree, strength, betweenness (sampled if >5000 nodes) per node."""
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        degree = dict(G.degree())
        strength = dict(G.degree(weight='weight'))

        if G.number_of_nodes() > 5000:
            betweenness = nx.betweenness_centrality(
                G, k=min(1000, G.number_of_nodes()), weight='weight'
            )
        else:
            betweenness = nx.betweenness_centrality(G, weight='weight')

        # Map each node to its connected component id
        component_map = {}
        for comp_id, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                component_map[node] = comp_id

        rows = []
        for node in G.nodes():
            rows.append({
                'node': node,
                'degree': degree[node],
                'strength': strength[node],
                'betweenness': betweenness[node],
                'component_id': component_map[node],
            })
        return pd.DataFrame(rows).sort_values('strength', ascending=False)

    # ------------------------------------------------------------------
    # Author Co-authorship Network
    # ------------------------------------------------------------------

    def build_author_network(self) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Build weighted undirected author co-authorship network.

        Edge weight = number of papers shared by the pair.
        Centrality metrics are computed on the giant component only.

        Returns
        -------
        (G_giant, metrics_df)
        """
        if self._coauthorship_edges is None:
            self.build_coauthorship_edges()

        edges = self._coauthorship_edges

        G = nx.Graph()
        for (a, b), group in edges.groupby(['author_A', 'author_B']):
            weight = len(group['work_id'].unique())
            G.add_edge(a, b, weight=weight)

        print(f"Author network — nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

        G_giant = self._extract_giant_component(G)
        print(f"Giant component — nodes: {G_giant.number_of_nodes()}, "
              f"edges: {G_giant.number_of_edges()}")

        metrics = self._compute_centrality_metrics(G_giant)
        metrics = metrics.rename(columns={'node': 'author_name'})
        return G_giant, metrics

    # ------------------------------------------------------------------
    # Institution Co-authorship Network
    # ------------------------------------------------------------------

    def _get_institution_pairs(self, mode: str) -> pd.DataFrame:
        """
        Return deduplicated (inst_A, inst_B, work_id) rows for institution network.

        Parameters
        ----------
        mode : 'primary' | 'expanded'
        """
        if self._author_paper_table is None:
            self.build_author_paper_table()

        if self.affil_col not in self.df.columns:
            print("Affiliation column not available — institution network skipped.")
            return pd.DataFrame(columns=['inst_A', 'inst_B', 'work_id'])

        apt = self._author_paper_table
        pair_set = set()

        for work_id, group in apt.groupby('id'):
            affils = group['primary_affiliation'].dropna().tolist()

            if mode == 'primary':
                institutions = list(dict.fromkeys(a for a in affils if a))
            else:  # expanded
                institutions = []
                for a in affils:
                    if a:
                        institutions.extend(
                            [s.strip() for s in a.split(';') if s.strip()]
                        )
                institutions = list(dict.fromkeys(institutions))

            for inst_a, inst_b in itertools.combinations(institutions, 2):
                if inst_a == inst_b:
                    continue
                key = (min(inst_a, inst_b), max(inst_a, inst_b), work_id)
                pair_set.add(key)

        rows = [{'inst_A': a, 'inst_B': b, 'work_id': w} for a, b, w in pair_set]
        return pd.DataFrame(rows)

    def build_institution_network(self, mode: str = 'primary') -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Build weighted undirected institution co-authorship network.

        Parameters
        ----------
        mode : 'primary' | 'expanded'
            'primary'  — use the full affiliation slot string as node identity.
            'expanded' — split slot on ";" to get individual institution names.

        Returns
        -------
        (G_giant, metrics_df)
        """
        pairs = self._get_institution_pairs(mode)

        if pairs.empty:
            return nx.Graph(), pd.DataFrame()

        G = nx.Graph()
        for (a, b), group in pairs.groupby(['inst_A', 'inst_B']):
            weight = len(group['work_id'].unique())
            G.add_edge(a, b, weight=weight)

        print(f"Institution network [{mode}] — nodes: {G.number_of_nodes()}, "
              f"edges: {G.number_of_edges()}")

        G_giant = self._extract_giant_component(G)
        print(f"Giant component — nodes: {G_giant.number_of_nodes()}, "
              f"edges: {G_giant.number_of_edges()}")

        metrics = self._compute_centrality_metrics(G_giant)
        metrics = metrics.rename(columns={'node': 'institution'})
        return G_giant, metrics

    # ------------------------------------------------------------------
    # Community Detection (Louvain)
    # ------------------------------------------------------------------

    def detect_communities(
        self, G: nx.Graph, label: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect communities using Louvain algorithm on graph G.

        Falls back to greedy modularity communities if python-louvain is unavailable.

        Parameters
        ----------
        G : nx.Graph
            Graph to partition (should be the giant component).
        label : str
            Column name for node labels in the output (e.g. 'author_name').

        Returns
        -------
        (membership_df, summary_df)
            membership_df : (label, community_id)
            summary_df    : (community_id, n_members, modularity)
        """
        if G.number_of_nodes() == 0:
            return pd.DataFrame(), pd.DataFrame()

        if HAS_LOUVAIN:
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            modularity = community_louvain.modularity(partition, G, weight='weight')
        else:
            print("python-louvain not installed; using NetworkX greedy modularity communities.")
            communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
            partition = {}
            for cid, members in enumerate(communities):
                for node in members:
                    partition[node] = cid
            modularity = nx.algorithms.community.modularity(
                G,
                [set(nodes) for nodes in communities],
                weight='weight',
            )

        membership = pd.DataFrame([
            {label: node, 'community_id': cid}
            for node, cid in partition.items()
        ])

        summary_rows = []
        for cid, members in membership.groupby('community_id'):
            summary_rows.append({
                'community_id': cid,
                'n_members': len(members),
                'modularity': modularity,
            })
        summary = pd.DataFrame(summary_rows).sort_values('n_members', ascending=False)

        print(f"Communities [{label}]: {len(summary)} detected, modularity={modularity:.4f}")
        return membership, summary

    # ------------------------------------------------------------------
    # Geographic Analysis
    # ------------------------------------------------------------------

    def geographic_analysis(self) -> Tuple[pd.DataFrame, object]:
        """
        Compute article counts and proportions by country and render a choropleth.

        Returns
        -------
        (country_stats_df, plotly_figure)
        """
        if self._author_paper_table is None:
            self.build_author_paper_table()

        if self.country_col not in self.df.columns:
            print("Country column not available — geographic analysis skipped.")
            return pd.DataFrame(), None

        apt = self._author_paper_table
        # One row per (work, country); deduplicate so multi-author same-country counts once
        country_presence = apt[['id', 'country']].dropna().drop_duplicates()
        article_counts = country_presence.groupby('country')['id'].nunique().reset_index()
        article_counts.columns = ['country', 'article_count']

        total = apt['id'].nunique()
        article_counts['proportion'] = article_counts['article_count'] / total

        # Convert alpha-2 → alpha-3 for Plotly
        def to_alpha3(code):
            try:
                c = pycountry.countries.get(alpha_2=code)
                return c.alpha_3 if c else None
            except Exception:
                return None

        article_counts['alpha3'] = article_counts['country'].apply(to_alpha3)
        article_counts = article_counts.dropna(subset=['alpha3'])

        fig = px.choropleth(
            article_counts,
            locations='alpha3',
            locationmode='ISO-3',
            color='proportion',
            hover_data={'country': True, 'article_count': True, 'proportion': ':.3f'},
            color_continuous_scale='Blues',
            title='Article Geographic Distribution',
        )

        print(f"Geographic analysis: {len(article_counts)} countries")
        return article_counts, fig

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_full_coauthorship_analysis(
        self,
        output_dir: str,
        affiliation_mode: str = 'both',
        analysis_type: str = 'all',
    ) -> Dict:
        """
        Run and save all coauthorship analyses.

        Parameters
        ----------
        output_dir : str
            Directory where all output files are written.
        affiliation_mode : 'primary' | 'expanded' | 'both'
        analysis_type : 'all' | 'tables' | 'author-network' |
                        'institution-network' | 'communities' | 'geographic'
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        run_tables = analysis_type in ('all', 'tables', 'author-network',
                                       'institution-network', 'communities', 'geographic')
        run_author_net = analysis_type in ('all', 'author-network', 'communities')
        run_inst_net = analysis_type in ('all', 'institution-network', 'communities')
        run_communities = analysis_type in ('all', 'communities')
        run_geo = analysis_type in ('all', 'geographic')

        # --- Tables ---
        if run_tables:
            apt = self.build_author_paper_table()
            apt.to_csv(os.path.join(output_dir, 'author_paper_edges.csv'), index=False)
            print(f"✓ Saved author_paper_edges.csv")

            edges = self.build_coauthorship_edges()
            edges.to_csv(os.path.join(output_dir, 'coauthorship_edges.csv'), index=False)
            print(f"✓ Saved coauthorship_edges.csv")

            results['author_paper_table'] = apt
            results['coauthorship_edges'] = edges

        # --- Author Network ---
        if run_author_net:
            G_author, author_metrics = self.build_author_network()
            if G_author.number_of_nodes() > 0:
                nx.write_graphml(G_author, os.path.join(output_dir, 'author_network.graphml'))
                author_metrics.to_csv(
                    os.path.join(output_dir, 'author_network_metrics.csv'), index=False
                )
                print(f"✓ Saved author_network.graphml + author_network_metrics.csv")
            results['author_network'] = G_author
            results['author_metrics'] = author_metrics

        # --- Institution Networks ---
        modes = []
        if run_inst_net:
            if affiliation_mode in ('primary', 'both'):
                modes.append('primary')
            if affiliation_mode in ('expanded', 'both'):
                modes.append('expanded')

        inst_networks = {}
        for mode in modes:
            G_inst, inst_metrics = self.build_institution_network(mode)
            if G_inst.number_of_nodes() > 0:
                nx.write_graphml(
                    G_inst,
                    os.path.join(output_dir, f'institution_network_{mode}.graphml'),
                )
                inst_metrics.to_csv(
                    os.path.join(output_dir, f'institution_network_{mode}_metrics.csv'),
                    index=False,
                )
                print(f"✓ Saved institution_network_{mode}.graphml + metrics CSV")
            inst_networks[mode] = (G_inst, inst_metrics)
        results['institution_networks'] = inst_networks

        # --- Communities ---
        if run_communities:
            if 'author_network' in results and results['author_network'].number_of_nodes() > 0:
                author_comm, author_comm_summary = self.detect_communities(
                    results['author_network'], 'author_name'
                )
                author_comm.to_csv(
                    os.path.join(output_dir, 'author_communities.csv'), index=False
                )
                author_comm_summary.to_csv(
                    os.path.join(output_dir, 'author_community_summary.csv'), index=False
                )
                print(f"✓ Saved author_communities.csv + author_community_summary.csv")
                results['author_communities'] = author_comm

            for mode, (G_inst, _) in inst_networks.items():
                if G_inst.number_of_nodes() > 0:
                    comm, comm_summary = self.detect_communities(G_inst, 'institution')
                    comm.to_csv(
                        os.path.join(output_dir, f'institution_communities_{mode}.csv'),
                        index=False,
                    )
                    comm_summary.to_csv(
                        os.path.join(output_dir, f'institution_community_summary_{mode}.csv'),
                        index=False,
                    )
                    print(f"✓ Saved institution_communities_{mode}.csv + summary CSV")

        # --- Geographic ---
        if run_geo:
            country_stats, fig = self.geographic_analysis()
            if not country_stats.empty:
                country_stats.to_csv(
                    os.path.join(output_dir, 'country_stats.csv'), index=False
                )
                if fig is not None:
                    fig.write_html(os.path.join(output_dir, 'geographic_map.html'))
                print(f"✓ Saved country_stats.csv + geographic_map.html")
                results['country_stats'] = country_stats

        return results


    # ------------------------------------------------------------------
    # Per-Topic Pipeline
    # ------------------------------------------------------------------

    def run_per_topic_analysis(
        self,
        output_dir: str,
        affiliation_mode: str = 'both',
        analysis_type: str = 'all',
        min_works: int = 10,
    ) -> Dict:
        """
        Run full co-authorship analysis independently for each topic.

        Results are written to ``{output_dir}/topic_{topic_id}/``.
        A cross-topic summary CSV is written to ``{output_dir}/topic_network_summary.csv``.

        Parameters
        ----------
        output_dir : str
            Root output directory.
        affiliation_mode : 'primary' | 'expanded' | 'both'
        analysis_type : str
            Same choices as ``run_full_coauthorship_analysis``.
        min_works : int
            Topics with fewer works than this are skipped.

        Returns
        -------
        dict mapping topic_id → per-topic results dict
        """
        if self.topic_col not in self.df.columns:
            raise ValueError(
                f"Topic column '{self.topic_col}' not found. "
                "Pass a topic_df on init or include the column in metadata_df."
            )

        os.makedirs(output_dir, exist_ok=True)
        topics = sorted(self.df[self.topic_col].dropna().unique())
        print(f"\nRunning per-topic analysis for {len(topics)} topics "
              f"(min_works={min_works})...")

        all_results = {}
        summary_rows = []

        for topic in topics:
            if topic == -1:
                print(f"  Skipping topic -1 (outliers)")
                continue
            topic_df = self.df[self.df[self.topic_col] == topic]
            if len(topic_df) < min_works:
                print(f"  Skipping topic {topic} ({len(topic_df)} works < {min_works})")
                continue

            print(f"\n{'='*50}")
            print(f"Topic {topic}  ({len(topic_df)} works)")
            print('='*50)

            topic_dir = os.path.join(output_dir, f'topic_{topic}')

            # Spin up a fresh sub-analyzer with only this topic's rows
            sub = CoauthorshipAnalyzer(
                metadata_df=topic_df.reset_index(drop=True),
                id_column=self.id_col,
                author_column=self.author_col,
                affiliation_column=self.affil_col,
                country_column=self.country_col,
                year_column=self.year_col,
            )

            topic_results = sub.run_full_coauthorship_analysis(
                output_dir=topic_dir,
                affiliation_mode=affiliation_mode,
                analysis_type=analysis_type,
            )
            all_results[topic] = topic_results

            # Collect summary stats
            row = {'topic': topic, 'n_works': len(topic_df)}
            if 'author_network' in topic_results:
                G = topic_results['author_network']
                row['author_nodes'] = G.number_of_nodes()
                row['author_edges'] = G.number_of_edges()
            if 'institution_networks' in topic_results:
                for mode, (G_inst, _) in topic_results['institution_networks'].items():
                    row[f'inst_nodes_{mode}'] = G_inst.number_of_nodes()
                    row[f'inst_edges_{mode}'] = G_inst.number_of_edges()
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'topic_network_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Topic network summary saved to {summary_path}")

        return all_results


# ============================================================================
# CLI COMMAND
# ============================================================================

@click.command()
@click.option('--metadata_csv', required=True,
              help='CSV file with work metadata (must include pipe-delimited authorship columns).')
@click.option('--output_dir', required=True,
              help='Directory to save analysis results.')
@click.option('--topic_csv', default=None,
              help='CSV with topic assignments (id + topic columns). '
                   'Required when using --per_topic.')
@click.option('--topic_column', default='topic',
              help='Column name for topic assignments (default: topic).')
@click.option('--per_topic', is_flag=True, default=False,
              help='Run analysis separately for each topic. Requires --topic_csv.')
@click.option('--min_works_per_topic', default=10, show_default=True,
              help='Skip topics with fewer works than this (only used with --per_topic).')
@click.option('--analysis_type',
              type=click.Choice(
                  ['all', 'tables', 'author-network', 'institution-network',
                   'communities', 'geographic'],
                  case_sensitive=False,
              ),
              default='all',
              help='Which analyses to run (default: all).')
@click.option('--affiliation_mode',
              type=click.Choice(['primary', 'expanded', 'both'], case_sensitive=False),
              default='both',
              help='Institution node granularity (default: both).')
@click.option('--id_column', default='id',
              help='Column with unique work identifiers (default: id).')
@click.option('--author_column', default='authorships.raw_author_name',
              help='Pipe-delimited author name column.')
@click.option('--affiliation_column', default='authorships.raw_affiliation_strings',
              help='Pipe-delimited affiliation column (one slot per author).')
@click.option('--country_column', default='authorships.countries',
              help='Pipe-delimited country code column (ISO alpha-2, one per author).')
@click.option('--year_column', default='publication_year',
              help='Publication year column.')
def coauthorship_analysis(
    metadata_csv,
    output_dir,
    topic_csv,
    topic_column,
    per_topic,
    min_works_per_topic,
    analysis_type,
    affiliation_mode,
    id_column,
    author_column,
    affiliation_column,
    country_column,
    year_column,
):
    """
    Run co-authorship network analysis on scholarly work metadata.

    Corpus-level: builds a single author/institution network across all works.
    Per-topic: runs the full pipeline independently for each BERTopic cluster,
    writing results to {output_dir}/topic_{id}/.

    Examples:
    ---------
    # Corpus-level
    revu biblio coauthorship-analysis \\
        --metadata_csv data/metadata_deduplicated.csv \\
        --output_dir data/coauthorship_results

    # Per-topic (recommended for diverse corpora)
    revu biblio coauthorship-analysis \\
        --metadata_csv data/metadata_deduplicated.csv \\
        --topic_csv data/causal_modeled_25Mar2026.csv \\
        --output_dir data/coauthorship_results \\
        --per_topic \\
        --min_works_per_topic 15
    """
    print("=" * 60)
    print("CO-AUTHORSHIP ANALYSIS")
    print("=" * 60)

    if per_topic and not topic_csv:
        raise click.UsageError("--per_topic requires --topic_csv to be specified.")

    print(f"\nLoading metadata from {metadata_csv}...")
    metadata_df = pd.read_csv(metadata_csv, low_memory=False)
    print(f"✓ Loaded {len(metadata_df)} works")

    topic_df = None
    if topic_csv:
        print(f"Loading topic assignments from {topic_csv}...")
        topic_df = pd.read_csv(topic_csv, low_memory=False)
        print(f"✓ Loaded {len(topic_df)} topic assignments")

    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    analyzer = CoauthorshipAnalyzer(
        metadata_df=metadata_df,
        topic_df=topic_df,
        id_column=id_column,
        topic_column=topic_column,
        author_column=author_column,
        affiliation_column=affiliation_column,
        country_column=country_column,
        year_column=year_column,
    )

    print("=" * 60)

    if per_topic:
        print(f"Mode: per-topic  |  analysis='{analysis_type}'  |  affiliation='{affiliation_mode}'")
        analyzer.run_per_topic_analysis(
            output_dir=output_dir,
            affiliation_mode=affiliation_mode,
            analysis_type=analysis_type,
            min_works=min_works_per_topic,
        )
    else:
        print(f"Mode: corpus-level  |  analysis='{analysis_type}'  |  affiliation='{affiliation_mode}'")
        analyzer.run_full_coauthorship_analysis(
            output_dir=output_dir,
            affiliation_mode=affiliation_mode,
            analysis_type=analysis_type,
        )

    print("\n" + "=" * 60)
    print("✅ CO-AUTHORSHIP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {output_dir}")
