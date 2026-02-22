import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class BibliographicVisualizer:
    """Visualizer for bibliographic network analyses"""

    def __init__(self, data_dir='.', topic_info_file=None):
        self.data_dir = Path(data_dir)
        self.fig_size = (12, 8)
        self.topic_names = None

        # Load topic name mapping if provided
        if topic_info_file:
            self.load_topic_names(topic_info_file)

    def load_data(self, filename):
        """Load CSV data"""
        df = pd.read_csv(self.data_dir / filename)

        # Filter out topic -1 (BERTopic's outlier topic) if it exists
        if 'topic' in df.columns and -1 in df['topic'].values:
            original_len = len(df)
            df = df[df['topic'] != -1]
            print(f"  ℹ Filtered out outlier topic -1 ({original_len - len(df)} row removed)")

        return df

    def load_topic_names(self, topic_info_file):
        """
        Load topic names from a topic info CSV file.

        Parameters:
        -----------
        topic_info_file : str
            Path to CSV file with 'Topic' and 'Name' columns
        """
        try:
            # Try multiple path resolution strategies
            topic_info_path = None

            # Strategy 1: Check if it's an absolute path
            if Path(topic_info_file).is_absolute():
                topic_info_path = Path(topic_info_file)
            # Strategy 2: Try relative to data_dir
            elif (self.data_dir / topic_info_file).exists():
                topic_info_path = self.data_dir / topic_info_file
            # Strategy 3: Try relative to current working directory
            elif Path(topic_info_file).exists():
                topic_info_path = Path(topic_info_file)
            else:
                # Try one more time - maybe it's relative to cwd
                topic_info_path = Path.cwd() / topic_info_file

            print(f"Attempting to load topic info from: {topic_info_path}")
            topic_info = pd.read_csv(topic_info_path)

            # Handle different possible column names (case-insensitive)
            topic_col = None
            name_col = None

            # Create a mapping of lowercase column names to actual column names
            col_map = {col.lower(): col for col in topic_info.columns}

            # Look for topic column
            for possible_name in ['topic', 'topic_id', 'topicid']:
                if possible_name in col_map:
                    topic_col = col_map[possible_name]
                    break

            # Look for name column
            for possible_name in ['name', 'topic_name', 'topicname', 'representation']:
                if possible_name in col_map:
                    name_col = col_map[possible_name]
                    break

            if topic_col and name_col:
                self.topic_names = dict(zip(topic_info[topic_col], topic_info[name_col]))
                print(f"✓ Loaded {len(self.topic_names)} topic names from {topic_info_file}")
                print(f"  Using columns: '{topic_col}' (ID) and '{name_col}' (Name)")
            else:
                print(f"⚠ Warning: Could not find 'topic' and 'name' columns in {topic_info_file}")
                print(f"   Available columns: {list(topic_info.columns)}")
                print(f"   Found topic_col: {topic_col}, name_col: {name_col}")
        except FileNotFoundError as e:
            print(f"⚠ Warning: Topic info file not found: {topic_info_file}")
            print(f"   Tried path: {topic_info_path}")
        except Exception as e:
            print(f"⚠ Warning: Error loading topic names: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_topic_label(self, topic_id):
        """
        Get display label for a topic (name if available, otherwise ID).

        Parameters:
        -----------
        topic_id : int or str
            Topic ID

        Returns:
        --------
        str : Topic label for display
        """
        if self.topic_names and topic_id in self.topic_names:
            name = self.topic_names[topic_id]
            # Truncate long names and add topic ID
            if len(str(name)) > 40:
                return f"{topic_id}: {str(name)[:37]}..."
            return f"{topic_id}: {name}"
        return str(topic_id)

    def filter_by_prevalence(self, df, top_n=None, min_works=None):
        """
        Filter topics by prevalence (number of works).

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        top_n : int, optional
            Keep only top N most prevalent topics
        min_works : int, optional
            Keep only topics with at least this many works

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        # Find the works column
        work_col = None
        for possible_col in ['n_works', 'total_works', 'Count', 'count']:
            if possible_col in df.columns:
                work_col = possible_col
                break

        if work_col is None:
            print(f"  ⚠ Warning: No works/count column found for prevalence filtering. Available columns: {list(df.columns)}")
            print(f"  Skipping prevalence filter (cannot determine prevalence without works column)")
            return df

        if top_n is not None:
            df = df.nlargest(top_n, work_col)
            print(f"  Filtered to top {top_n} most prevalent topics (by {work_col})")

        if min_works is not None:
            df = df[df[work_col] >= min_works]
            print(f"  Filtered to topics with {work_col} >= {min_works} ({len(df)} topics)")

        return df

    def filter_by_centrality(self, df, metric='mean_pagerank', top_n=None, bottom_n=None, threshold=None):
        """
        Filter topics by centrality measures.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with centrality metrics
        metric : str
            Centrality metric to filter on (default: 'mean_pagerank')
        top_n : int, optional
            Keep only top N topics by centrality
        bottom_n : int, optional
            Keep only bottom N topics by centrality
        threshold : float, optional
            Keep only topics with centrality above this threshold

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, metric)
            print(f"  Filtered to top {top_n} topics by {metric}")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, metric)
            print(f"  Filtered to bottom {bottom_n} topics by {metric}")

        if threshold is not None:
            df = df[df[metric] >= threshold]
            print(f"  Filtered to topics with {metric} >= {threshold} ({len(df)} topics)")

        return df

    def filter_by_homophily(self, df, top_n=None, bottom_n=None, min_excess=None, max_excess=None):
        """
        Filter topics by homophily (excess_homophily).

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with homophily metrics
        top_n : int, optional
            Keep only top N topics by excess homophily (most homophilic)
        bottom_n : int, optional
            Keep only bottom N topics by excess homophily (least homophilic)
        min_excess : float, optional
            Keep only topics with excess_homophily >= this value
        max_excess : float, optional
            Keep only topics with excess_homophily <= this value

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, 'excess_homophily')
            print(f"  Filtered to top {top_n} most homophilic topics")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, 'excess_homophily')
            print(f"  Filtered to bottom {bottom_n} least homophilic topics")

        if min_excess is not None:
            df = df[df['excess_homophily'] >= min_excess]
            print(f"  Filtered to topics with excess_homophily >= {min_excess} ({len(df)} topics)")

        if max_excess is not None:
            df = df[df['excess_homophily'] <= max_excess]
            print(f"  Filtered to topics with excess_homophily <= {max_excess} ({len(df)} topics)")

        return df

    def filter_by_influence(self, df, metric='total_external_citations', top_n=None, bottom_n=None, threshold=None):
        """
        Filter topics by influence metrics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with influence metrics
        metric : str
            Influence metric to filter on (default: 'total_external_citations')
            Options: 'total_external_citations', 'citation_balance', 'h_index_proxy'
        top_n : int, optional
            Keep only top N topics by influence
        bottom_n : int, optional
            Keep only bottom N topics by influence
        threshold : float, optional
            Keep only topics with influence above this threshold

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, metric)
            print(f"  Filtered to top {top_n} topics by {metric}")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, metric)
            print(f"  Filtered to bottom {bottom_n} topics by {metric}")

        if threshold is not None:
            df = df[df[metric] >= threshold]
            print(f"  Filtered to topics with {metric} >= {threshold} ({len(df)} topics)")

        return df

    def filter_by_trends(self, df, top_n=None, bottom_n=None, min_slope=None, max_slope=None,
                        growing_only=False, declining_only=False):
        """
        Filter topics by temporal trends.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with trend metrics
        top_n : int, optional
            Keep only top N topics by trend slope (fastest growing)
        bottom_n : int, optional
            Keep only bottom N topics by trend slope (fastest declining)
        min_slope : float, optional
            Keep only topics with trend_slope >= this value
        max_slope : float, optional
            Keep only topics with trend_slope <= this value
        growing_only : bool
            Keep only topics with positive trend slope
        declining_only : bool
            Keep only topics with negative trend slope

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, 'trend_slope')
            print(f"  Filtered to top {top_n} fastest growing topics")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, 'trend_slope')
            print(f"  Filtered to bottom {bottom_n} fastest declining topics")

        if min_slope is not None:
            df = df[df['trend_slope'] >= min_slope]
            print(f"  Filtered to topics with trend_slope >= {min_slope} ({len(df)} topics)")

        if max_slope is not None:
            df = df[df['trend_slope'] <= max_slope]
            print(f"  Filtered to topics with trend_slope <= {max_slope} ({len(df)} topics)")

        if growing_only:
            df = df[df['trend_slope'] > 0]
            print(f"  Filtered to growing topics only ({len(df)} topics)")

        if declining_only:
            df = df[df['trend_slope'] < 0]
            print(f"  Filtered to declining topics only ({len(df)} topics)")

        return df
    
    def visualize_homophily(self, df, output_file='homophily_viz.png'):
        """
        Visualize homophily analysis
        Shows citation patterns within vs. across topics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Homophily Analysis: Citation Patterns', fontsize=16, fontweight='bold')

        # Get topic labels
        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        # 1. Homophily ratio comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(topic_labels))

        ax1.bar(x_pos - 0.2, df['homophily_ratio'], 0.4, label='Observed', alpha=0.8)
        ax1.bar(x_pos + 0.2, df['expected_homophily'], 0.4, label='Expected', alpha=0.8)
        ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Homophily Ratio')
        ax1.set_title('Observed vs Expected Homophily')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Excess homophily
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df['excess_homophily']]
        ax2.barh(topic_labels, df['excess_homophily'], color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.set_xlabel('Excess Homophily')
        ax2.set_title('Excess Homophily by Topic')
        ax2.tick_params(axis='y', labelsize=8)
        ax2.grid(axis='x', alpha=0.3)

        # 3. Citation breakdown
        ax3 = axes[1, 0]
        width = 0.35
        ax3.bar(x_pos - width/2, df['internal_citations'], width, label='Internal', alpha=0.8)
        ax3.bar(x_pos + width/2, df['external_citations'], width, label='External', alpha=0.8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Number of Citations')
        ax3.set_title('Internal vs External Citations')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Network diagram
        ax4 = axes[1, 1]
        G = nx.DiGraph()

        for _, row in df.iterrows():
            G.add_node(row['topic'],
                      internal=row['internal_citations'],
                      external=row['external_citations'])

        # Add edges based on homophily (self-loops for internal citations)
        for _, row in df.iterrows():
            if row['internal_citations'] > 0:
                G.add_edge(row['topic'], row['topic'],
                          weight=row['internal_citations'])

        pos = nx.spring_layout(G, k=2, iterations=50)
        node_sizes = [G.nodes[node]['internal'] * 10 for node in G.nodes()]

        # Create label mapping for network
        label_map = {topic_id: self.get_topic_label(topic_id) for topic_id in G.nodes()}

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color='lightblue', alpha=0.8, ax=ax4)
        nx.draw_networkx_labels(G, pos, labels=label_map, font_size=6, ax=ax4)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.3,
                              edge_color='gray', ax=ax4,
                              connectionstyle='arc3,rad=0.1')
        ax4.set_title('Citation Network (node size = internal citations)')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Homophily visualization saved to {output_file}")
    
    def visualize_centrality(self, df, output_file='centrality_viz.png'):
        """
        Visualize centrality measures
        Shows network importance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Centrality Analysis: Network Position', fontsize=16, fontweight='bold')

        # Get topic labels
        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        # 1. In-degree vs Out-degree
        ax1 = axes[0, 0]
        ax1.scatter(df['mean_out_degree_centrality'],
                   df['mean_in_degree_centrality'],
                   s=df['n_works']*10, alpha=0.6)
        for i, topic_label in enumerate(topic_labels):
            ax1.annotate(topic_label,
                        (df['mean_out_degree_centrality'].iloc[i],
                         df['mean_in_degree_centrality'].iloc[i]),
                        fontsize=6, alpha=0.7)
        ax1.set_xlabel('Mean Out-Degree Centrality')
        ax1.set_ylabel('Mean In-Degree Centrality')
        ax1.set_title('In-Degree vs Out-Degree (size = n_works)')
        ax1.grid(alpha=0.3)

        # 2. PageRank distribution
        ax2 = axes[0, 1]
        x_pos = np.arange(len(topic_labels))
        ax2.bar(x_pos, df['mean_pagerank'], alpha=0.7, color='coral')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Mean PageRank')
        ax2.set_title('PageRank by Topic')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Betweenness centrality
        ax3 = axes[1, 0]
        ax3.barh(topic_labels, df['mean_betweenness'], alpha=0.7, color='lightgreen')
        ax3.set_xlabel('Mean Betweenness Centrality')
        ax3.set_title('Betweenness: Bridge Position')
        ax3.tick_params(axis='y', labelsize=8)
        ax3.grid(axis='x', alpha=0.3)

        # 4. Centrality comparison heatmap
        ax4 = axes[1, 1]
        centrality_cols = ['mean_in_degree_centrality', 'mean_out_degree_centrality',
                          'mean_pagerank', 'mean_betweenness']

        # Normalize each column for comparison
        normalized_data = df[centrality_cols].copy()
        for col in centrality_cols:
            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                   (normalized_data[col].max() - normalized_data[col].min())

        im = ax4.imshow(normalized_data.T, aspect='auto', cmap='YlOrRd')
        ax4.set_xticks(np.arange(len(topic_labels)))
        ax4.set_yticks(np.arange(len(centrality_cols)))
        ax4.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(['In-Degree', 'Out-Degree', 'PageRank', 'Betweenness'])
        ax4.set_title('Normalized Centrality Measures')
        plt.colorbar(im, ax=ax4, label='Normalized Value')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Centrality visualization saved to {output_file}")
    
    def visualize_influence(self, df, output_file='influence_viz.png'):
        """
        Visualize influence metrics
        Shows citation impact and balance
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Influence Analysis: Citation Impact', fontsize=16, fontweight='bold')

        # Get topic labels
        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        # 1. Citation balance
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in df['citation_balance']]
        ax1.barh(topic_labels, df['citation_balance'], color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.set_xlabel('Citation Balance (received - given)')
        ax1.set_title('Net Citation Flow')
        ax1.tick_params(axis='y', labelsize=8)
        ax1.grid(axis='x', alpha=0.3)

        # 2. H-index proxy
        ax2 = axes[0, 1]
        x_pos = np.arange(len(topic_labels))
        ax2.bar(x_pos, df['h_index_proxy'], alpha=0.7, color='purple')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('H-Index Proxy')
        ax2.set_title('Research Impact (H-Index Proxy)')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Internal vs External citations received
        ax3 = axes[1, 0]
        ax3.scatter(df['total_internal_citations_received'],
                   df['total_external_citations'],
                   s=df['n_works']*10, alpha=0.6, color='teal')
        for i, topic_label in enumerate(topic_labels):
            ax3.annotate(topic_label,
                        (df['total_internal_citations_received'].iloc[i],
                         df['total_external_citations'].iloc[i]),
                        fontsize=6, alpha=0.7)
        ax3.set_xlabel('Internal Citations Received')
        ax3.set_ylabel('External Citations')
        ax3.set_title('Internal vs External Impact')
        ax3.grid(alpha=0.3)

        # 4. Average citations given vs received
        ax4 = axes[1, 1]
        width = 0.35
        x_pos = np.arange(len(topic_labels))
        ax4.bar(x_pos - width/2, df['avg_citations_received'], width,
               label='Received', alpha=0.8, color='blue')
        ax4.bar(x_pos + width/2, df['avg_citations_given'], width,
               label='Given', alpha=0.8, color='orange')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Average Citations per Work')
        ax4.set_title('Citation Activity')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Influence visualization saved to {output_file}")
    
    def visualize_trends(self, df, output_file='trends_viz.png'):
        """
        Visualize temporal trends
        Shows topic evolution over time
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trends Analysis: Topic Evolution', fontsize=16, fontweight='bold')

        # Get topic labels
        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        # 1. Timeline spans
        ax1 = axes[0, 0]
        for i, row in df.iterrows():
            ax1.barh(i, row['last_year'] - row['first_year'],
                    left=row['first_year'], alpha=0.7)
            ax1.text(row['first_year'], i, str(row['first_year']),
                    va='center', fontsize=7)
            ax1.text(row['last_year'], i, str(row['last_year']),
                    va='center', fontsize=7)
        ax1.set_yticks(range(len(topic_labels)))
        ax1.set_yticklabels(topic_labels, fontsize=8)
        ax1.set_xlabel('Year')
        ax1.set_title('Topic Temporal Spans')
        ax1.grid(axis='x', alpha=0.3)

        # 2. Trend slopes
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df['trend_slope']]
        ax2.barh(topic_labels, df['trend_slope'], color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.set_xlabel('Trend Slope')
        ax2.set_title('Growth Trends (positive = growing)')
        ax2.tick_params(axis='y', labelsize=8)
        ax2.grid(axis='x', alpha=0.3)

        # 3. Current proportion
        ax3 = axes[1, 0]
        ax3.pie(df['current_proportion'], labels=topic_labels, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 7})
        ax3.set_title('Current Topic Distribution')

        # 4. Total works vs recent change
        ax4 = axes[1, 1]
        ax4.scatter(df['total_works'], df['recent_change'],
                   s=df['avg_proportion']*1000, alpha=0.6)
        for i, topic_label in enumerate(topic_labels):
            ax4.annotate(topic_label,
                        (df['total_works'].iloc[i], df['recent_change'].iloc[i]),
                        fontsize=6, alpha=0.7)
        ax4.set_xlabel('Total Works')
        ax4.set_ylabel('Recent Change')
        ax4.set_title('Volume vs Recent Momentum')
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trends visualization saved to {output_file}")
    
    def generate_all_visualizations(self, 
                                   homophily_file='homophily.csv',
                                   centrality_file='centrality.csv',
                                   influence_file='influence.csv',
                                   trends_file='trends.csv'):
        """Generate all visualizations at once"""
        print("Starting bibliographic analysis visualizations...")
        
        try:
            df_homophily = self.load_data(homophily_file)
            self.visualize_homophily(df_homophily)
        except FileNotFoundError:
            print(f"Warning: {homophily_file} not found, skipping homophily visualization")
        
        try:
            df_centrality = self.load_data(centrality_file)
            self.visualize_centrality(df_centrality)
        except FileNotFoundError:
            print(f"Warning: {centrality_file} not found, skipping centrality visualization")
        
        try:
            df_influence = self.load_data(influence_file)
            self.visualize_influence(df_influence)
        except FileNotFoundError:
            print(f"Warning: {influence_file} not found, skipping influence visualization")
        
        try:
            df_trends = self.load_data(trends_file)
            self.visualize_trends(df_trends)
        except FileNotFoundError:
            print(f"Warning: {trends_file} not found, skipping trends visualization")
        
        print("\nAll visualizations complete!")