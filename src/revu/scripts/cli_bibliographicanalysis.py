
import click
import pandas as pd
import networkx as nx
import numpy as np
import os
import itertools
import pycountry
import plotly.express as px
import community as community_louvain
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TopicBibliometricAnalyzer:
    """
    Analyzes scholarly works using network analysis with topic modeling results.
    Computes topic centrality, influence, trends, and homophily.
    """
    
    def __init__(
        self,
        topic_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        topic_column: str = 'topic',
        id_column: str = 'id'
    ):
        """
        Initialize analyzer with bibliographic data.

        Parameters:
        -----------
        topic_df : pd.DataFrame
            DataFrame with columns including 'id', 'title', and topic assignments
        metadata_df : pd.DataFrame, optional
            DataFrame with columns including 'id', 'referenced_works', 'publication_year',
            'cited_by_count'. If None, assumes all data is in topic_df
        topic_column : str
            Name of column containing topic assignments from BERTopic
        id_column : str
            Name of column containing unique work identifiers for linking
        """
        self.topic_column = topic_column
        self.id_column = id_column
        self.citation_network = None

        # If metadata_df is provided, merge the two dataframes on id_column
        if metadata_df is not None:
            print(f"Merging topic data ({len(topic_df)} rows) with metadata ({len(metadata_df)} rows)...")
            self.df = pd.merge(
                topic_df,
                metadata_df,
                on=id_column,
                how='inner',
                suffixes=('', '_metadata')
            )
            print(f"✓ Merged dataset contains {len(self.df)} works")
        else:
            self.df = topic_df.copy()
            print(f"Using single data source with {len(self.df)} works")

        # Ensure referenced_works exists and is string type
        if 'referenced_works' not in self.df.columns:
            print("Warning: 'referenced_works' column not found. Citation network analysis will be limited.")
            self.df['referenced_works'] = ''
        else:
            self.df['referenced_works'] = self.df['referenced_works'].fillna('')

        print(f"Topics found: {self.df[topic_column].nunique()}")
    
    def build_citation_network(self) -> nx.DiGraph:
        """
        Build directed citation network where edges go from citing -> cited work.
        
        Returns:
        --------
        nx.DiGraph : Citation network with topic attributes
        """
        G = nx.DiGraph()
        
        # Add all works as nodes with attributes
        for idx, row in self.df.iterrows():
            work_id = row[self.id_column]
            G.add_node(
                work_id,
                topic=row[self.topic_column],
                year=row.get('publication_year', None),
                cited_by_count=row.get('cited_by_count', 0),
                title=row.get('title', '')
            )

        # Add citation edges
        edge_count = 0
        for idx, row in self.df.iterrows():
            citing_id = row[self.id_column]
            
            # Parse referenced works
            if pd.notna(row['referenced_works']) and row['referenced_works']:
                referenced = str(row['referenced_works']).split('|')
                referenced = [ref.strip() for ref in referenced if ref.strip()]
                
                for cited_id in referenced:
                    # Only add edge if cited work is in our corpus
                    if cited_id in G.nodes():
                        G.add_edge(citing_id, cited_id)
                        edge_count += 1
        
        self.citation_network = G
        print(f"\nBuilt citation network:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.6f}")
        
        return G
    
    def compute_topic_centrality(self) -> pd.DataFrame:
        """
        Compute various centrality measures aggregated by topic.
        
        Returns:
        --------
        pd.DataFrame : Topic-level centrality metrics
        """
        if self.citation_network is None:
            self.build_citation_network()
        
        G = self.citation_network
        
        # Compute node-level centralities
        in_degree_cent = nx.in_degree_centrality(G)
        out_degree_cent = nx.out_degree_centrality(G)
        
        # PageRank as a prestige measure
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        except:
            pagerank = {node: 0 for node in G.nodes()}
        
        # Betweenness (sample for large graphs)
        if G.number_of_nodes() > 5000:
            betweenness = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes()))
        else:
            betweenness = nx.betweenness_centrality(G)
        
        # Aggregate by topic
        topic_metrics = defaultdict(lambda: {
            'in_degree': [],
            'out_degree': [],
            'pagerank': [],
            'betweenness': [],
            'count': 0
        })
        
        for node in G.nodes():
            topic = G.nodes[node]['topic']
            topic_metrics[topic]['in_degree'].append(in_degree_cent[node])
            topic_metrics[topic]['out_degree'].append(out_degree_cent[node])
            topic_metrics[topic]['pagerank'].append(pagerank[node])
            topic_metrics[topic]['betweenness'].append(betweenness[node])
            topic_metrics[topic]['count'] += 1
        
        # Compute summary statistics
        results = []
        for topic, metrics in topic_metrics.items():
            results.append({
                'topic': topic,
                'n_works': metrics['count'],
                'mean_in_degree_centrality': np.mean(metrics['in_degree']),
                'mean_out_degree_centrality': np.mean(metrics['out_degree']),
                'mean_pagerank': np.mean(metrics['pagerank']),
                'mean_betweenness': np.mean(metrics['betweenness']),
                'median_in_degree_centrality': np.median(metrics['in_degree']),
                'median_pagerank': np.median(metrics['pagerank'])
            })
        
        df_centrality = pd.DataFrame(results).sort_values('mean_pagerank', ascending=False)
        print("\nTopic Centrality computed successfully")
        
        return df_centrality
    
    def compute_topic_influence(self) -> pd.DataFrame:
        """
        Compute topic influence based on citation patterns and impact.
        
        Returns:
        --------
        pd.DataFrame : Topic influence metrics
        """
        if self.citation_network is None:
            self.build_citation_network()
        
        G = self.citation_network
        
        topic_influence = defaultdict(lambda: {
            'total_citations_received': 0,
            'total_citations_given': 0,
            'avg_citations_per_work': 0,
            'h_index_proxy': [],
            'works': []
        })
        
        # Aggregate citation data by topic
        for node in G.nodes():
            topic = G.nodes[node]['topic']
            in_degree = G.in_degree(node)  # Citations received
            out_degree = G.out_degree(node)  # Citations given
            external_citations = G.nodes[node].get('cited_by_count', 0)
            
            topic_influence[topic]['total_citations_received'] += in_degree
            topic_influence[topic]['total_citations_given'] += out_degree
            topic_influence[topic]['h_index_proxy'].append(external_citations)
            topic_influence[topic]['works'].append(node)
        
        # Compute influence metrics
        results = []
        for topic, data in topic_influence.items():
            n_works = len(data['works'])
            
            # H-index calculation (simplified)
            citations = sorted(data['h_index_proxy'], reverse=True)
            h_index = 0
            for i, c in enumerate(citations, 1):
                if c >= i:
                    h_index = i
                else:
                    break
            
            # Citation balance (influence vs dependence)
            total_out = data['total_citations_given']
            total_in = data['total_citations_received']
            citation_balance = (total_in - total_out) / max(total_in + total_out, 1)
            
            results.append({
                'topic': topic,
                'n_works': n_works,
                'total_internal_citations_received': total_in,
                'total_internal_citations_given': total_out,
                'avg_citations_received': total_in / n_works if n_works > 0 else 0,
                'avg_citations_given': total_out / n_works if n_works > 0 else 0,
                'citation_balance': citation_balance,
                'h_index_proxy': h_index,
                'total_external_citations': sum(data['h_index_proxy'])
            })
        
        df_influence = pd.DataFrame(results).sort_values('total_external_citations', ascending=False)
        print("\nTopic Influence computed successfully")
        
        return df_influence
    
    def compute_topic_trends(self, window_size: int = 3) -> pd.DataFrame:
        """
        Compute temporal trends for topics over publication years.
        
        Parameters:
        -----------
        window_size : int
            Size of rolling window for trend calculation
            
        Returns:
        --------
        pd.DataFrame : Topic trends over time
        """
        # Filter out works without publication year
        df_temporal = self.df[self.df['publication_year'].notna()].copy()
        
        # Count works by topic and year
        topic_year_counts = df_temporal.groupby([self.topic_column, 'publication_year']).size()
        topic_year_counts = topic_year_counts.reset_index(name='count')
        
        # Compute total works per year
        year_totals = df_temporal.groupby('publication_year').size()
        
        # Calculate proportions
        topic_year_counts['total_works_year'] = topic_year_counts['publication_year'].map(year_totals)
        topic_year_counts['proportion'] = topic_year_counts['count'] / topic_year_counts['total_works_year']
        
        # Compute trends for each topic
        trends = []
        for topic in topic_year_counts[self.topic_column].unique():
            topic_data = topic_year_counts[topic_year_counts[self.topic_column] == topic].copy()
            topic_data = topic_data.sort_values('publication_year')
            
            if len(topic_data) >= 2:
                # Linear trend
                years = topic_data['publication_year'].values
                proportions = topic_data['proportion'].values
                
                # Simple linear regression
                year_mean = years.mean()
                prop_mean = proportions.mean()
                
                numerator = np.sum((years - year_mean) * (proportions - prop_mean))
                denominator = np.sum((years - year_mean) ** 2)
                
                slope = numerator / denominator if denominator != 0 else 0
                
                # Rolling average for recent trend
                if len(topic_data) >= window_size:
                    recent_avg = topic_data.tail(window_size)['proportion'].mean()
                    early_avg = topic_data.head(window_size)['proportion'].mean()
                    recent_change = recent_avg - early_avg
                else:
                    recent_change = 0
                
                trends.append({
                    'topic': topic,
                    'first_year': int(years.min()),
                    'last_year': int(years.max()),
                    'n_years': len(years),
                    'total_works': int(topic_data['count'].sum()),
                    'avg_proportion': proportions.mean(),
                    'trend_slope': slope,
                    'recent_change': recent_change,
                    'current_proportion': proportions[-1]
                })
        
        df_trends = pd.DataFrame(trends).sort_values('trend_slope', ascending=False)
        print("\nTopic Trends computed successfully")
        
        return df_trends
    
    def compute_topic_homophily(self) -> pd.DataFrame:
        """
        Compute homophily (assortativity) - tendency for same-topic works to cite each other.
        
        Returns:
        --------
        pd.DataFrame : Topic homophily metrics
        """
        if self.citation_network is None:
            self.build_citation_network()
        
        G = self.citation_network
        
        # Overall topic assortativity
        try:
            global_assortativity = nx.attribute_assortativity_coefficient(G, 'topic')
        except:
            global_assortativity = None
        
        # Topic-specific homophily
        topic_homophily = defaultdict(lambda: {
            'internal_citations': 0,
            'external_citations': 0,
            'total_edges': 0
        })
        
        for u, v in G.edges():
            topic_u = G.nodes[u]['topic']
            topic_v = G.nodes[v]['topic']
            
            topic_homophily[topic_u]['total_edges'] += 1
            
            if topic_u == topic_v:
                topic_homophily[topic_u]['internal_citations'] += 1
            else:
                topic_homophily[topic_u]['external_citations'] += 1
        
        # Compute homophily ratio
        results = []
        for topic, data in topic_homophily.items():
            total = data['total_edges']
            if total > 0:
                homophily_ratio = data['internal_citations'] / total
            else:
                homophily_ratio = 0
            
            # Expected homophily (proportion of nodes with this topic)
            n_topic = len([n for n in G.nodes() if G.nodes[n]['topic'] == topic])
            expected_homophily = n_topic / G.number_of_nodes()
            
            # Excess homophily
            excess_homophily = homophily_ratio - expected_homophily
            
            results.append({
                'topic': topic,
                'internal_citations': data['internal_citations'],
                'external_citations': data['external_citations'],
                'total_outgoing_citations': total,
                'homophily_ratio': homophily_ratio,
                'expected_homophily': expected_homophily,
                'excess_homophily': excess_homophily
            })
        
        df_homophily = pd.DataFrame(results).sort_values('excess_homophily', ascending=False)
        
        print("\nTopic Homophily computed successfully")
        if global_assortativity is not None:
            print(f"Global topic assortativity: {global_assortativity:.4f}")
        
        return df_homophily
    
    def run_full_analysis(self, window_size: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Run all analyses and return results dictionary.
        
        Parameters:
        -----------
        window_size : int
            Window size for trend analysis
            
        Returns:
        --------
        dict : Dictionary containing all analysis results
        """
        print("="*60)
        print("RUNNING FULL BIBLIOMETRIC TOPIC ANALYSIS")
        print("="*60)
        
        results = {
            'centrality': self.compute_topic_centrality(),
            'influence': self.compute_topic_influence(),
            'trends': self.compute_topic_trends(window_size),
            'homophily': self.compute_topic_homophily()
        }
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return results


# ============================================================================
# CLI COMMANDS
# ============================================================================

@click.command()
@click.option('--topic_csv', required=True, help='CSV file with topic assignments and titles')
@click.option('--metadata_csv', default=None, help='CSV file with metadata (referenced_works, publication_year, cited_by_count). If not provided, assumes all data is in topic_csv')
@click.option('--output_dir', required=True, help='Directory to save analysis results')
@click.option('--topic_column', default='topic', help='Name of column containing topic assignments')
@click.option('--id_column', default='id', help='Name of column containing unique work IDs for linking')
@click.option('--window_size', default=3, help='Window size for trend analysis')
@click.option('--analysis_type',
              type=click.Choice(['all', 'centrality', 'influence', 'trends', 'homophily'], case_sensitive=False),
              default='all',
              help='Type of analysis to run')
def run_bibliometric_analysis(topic_csv, metadata_csv, output_dir, topic_column, id_column, window_size, analysis_type):
    """
    Run bibliometric analysis on scholarly works with topic modeling results.

    This command analyzes citation networks, topic centrality, influence, trends, and homophily.
    It supports separate data sources for topic assignments and metadata, linked by ID.

    Example:
    --------
    revu biblio run-analysis \\
        --topic_csv data/topics_modeled.csv \\
        --metadata_csv data/metadata.csv \\
        --output_dir data/biblio_results \\
        --topic_column topic \\
        --id_column id
    """
    print("="*60)
    print("BIBLIOMETRIC ANALYSIS")
    print("="*60)

    # Load data
    print(f"\nLoading topic data from {topic_csv}...")
    topic_df = pd.read_csv(topic_csv, low_memory=False)
    print(f"✓ Loaded {len(topic_df)} works with topics")

    metadata_df = None
    if metadata_csv:
        print(f"\nLoading metadata from {metadata_csv}...")
        metadata_df = pd.read_csv(metadata_csv, low_memory=False)
        print(f"✓ Loaded metadata for {len(metadata_df)} works")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")

    # Initialize analyzer
    print("\n" + "="*60)
    analyzer = TopicBibliometricAnalyzer(
        topic_df=topic_df,
        metadata_df=metadata_df,
        topic_column=topic_column,
        id_column=id_column
    )

    # Run requested analyses
    results = {}

    if analysis_type in ['all', 'centrality']:
        print("\n" + "="*60)
        print("Computing Topic Centrality...")
        print("="*60)
        results['centrality'] = analyzer.compute_topic_centrality()
        centrality_path = os.path.join(output_dir, 'topic_centrality.csv')
        results['centrality'].to_csv(centrality_path, index=False)
        print(f"✓ Centrality results saved to {centrality_path}")

    if analysis_type in ['all', 'influence']:
        print("\n" + "="*60)
        print("Computing Topic Influence...")
        print("="*60)
        results['influence'] = analyzer.compute_topic_influence()
        influence_path = os.path.join(output_dir, 'topic_influence.csv')
        results['influence'].to_csv(influence_path, index=False)
        print(f"✓ Influence results saved to {influence_path}")

    if analysis_type in ['all', 'trends']:
        print("\n" + "="*60)
        print("Computing Topic Trends...")
        print("="*60)
        results['trends'] = analyzer.compute_topic_trends(window_size=window_size)
        trends_path = os.path.join(output_dir, 'topic_trends.csv')
        results['trends'].to_csv(trends_path, index=False)
        print(f"✓ Trends results saved to {trends_path}")

    if analysis_type in ['all', 'homophily']:
        print("\n" + "="*60)
        print("Computing Topic Homophily...")
        print("="*60)
        results['homophily'] = analyzer.compute_topic_homophily()
        homophily_path = os.path.join(output_dir, 'topic_homophily.csv')
        results['homophily'].to_csv(homophily_path, index=False)
        print(f"✓ Homophily results saved to {homophily_path}")

    # Create summary report
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BIBLIOMETRIC ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input Files:\n")
        f.write(f"  - Topic CSV: {topic_csv}\n")
        f.write(f"  - Metadata CSV: {metadata_csv if metadata_csv else 'N/A (all data in topic CSV)'}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  - Topic Column: {topic_column}\n")
        f.write(f"  - ID Column: {id_column}\n")
        f.write(f"  - Window Size: {window_size}\n")
        f.write(f"  - Analysis Type: {analysis_type}\n\n")
        f.write(f"Data Summary:\n")
        f.write(f"  - Total Works: {len(analyzer.df)}\n")
        f.write(f"  - Total Topics: {analyzer.df[topic_column].nunique()}\n\n")
        f.write(f"Output Files:\n")
        if 'centrality' in results:
            f.write(f"  - Topic Centrality: topic_centrality.csv ({len(results['centrality'])} topics)\n")
        if 'influence' in results:
            f.write(f"  - Topic Influence: topic_influence.csv ({len(results['influence'])} topics)\n")
        if 'trends' in results:
            f.write(f"  - Topic Trends: topic_trends.csv ({len(results['trends'])} topics)\n")
        if 'homophily' in results:
            f.write(f"  - Topic Homophily: topic_homophily.csv ({len(results['homophily'])} topics)\n")
        f.write(f"\n")
        f.write("="*60 + "\n")

    print(f"✓ Analysis summary saved to {summary_path}")

    print("\n" + "="*60)
    print("✅ BIBLIOMETRIC ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nGenerated files:")
    if 'centrality' in results:
        print(f"  - topic_centrality.csv")
    if 'influence' in results:
        print(f"  - topic_influence.csv")
    if 'trends' in results:
        print(f"  - topic_trends.csv")
    if 'homophily' in results:
        print(f"  - topic_homophily.csv")
    print(f"  - analysis_summary.txt")

    return results


@click.command()
@click.option('--topic_csv', required=True, help='CSV file with topic assignments')
@click.option('--metadata_csv', default=None, help='CSV file with metadata')
@click.option('--output_path', required=True, help='Path to save citation network (GraphML format)')
@click.option('--topic_column', default='topic', help='Name of column containing topic assignments')
@click.option('--id_column', default='id', help='Name of column containing unique work IDs')
def build_network(topic_csv, metadata_csv, output_path, topic_column, id_column):
    """
    Build and save citation network as GraphML file.

    Example:
    --------
    revu biblio build-network \\
        --topic_csv data/topics.csv \\
        --metadata_csv data/metadata.csv \\
        --output_path data/citation_network.graphml
    """
    print("="*60)
    print("BUILDING CITATION NETWORK")
    print("="*60)

    # Load data
    print(f"\nLoading topic data from {topic_csv}...")
    topic_df = pd.read_csv(topic_csv, low_memory=False)

    metadata_df = None
    if metadata_csv:
        print(f"Loading metadata from {metadata_csv}...")
        metadata_df = pd.read_csv(metadata_csv, low_memory=False)

    # Initialize analyzer and build network
    analyzer = TopicBibliometricAnalyzer(
        topic_df=topic_df,
        metadata_df=metadata_df,
        topic_column=topic_column,
        id_column=id_column
    )

    G = analyzer.build_citation_network()

    # Save network
    print(f"\nSaving network to {output_path}...")
    nx.write_graphml(G, output_path)
    print(f"✓ Citation network saved to {output_path}")

    print("\n" + "="*60)
    print("✅ NETWORK BUILD COMPLETE")
    print("="*60)