import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import random

class HierarchyVisualizer:
    def __init__(self, json_path: str):
        """
        Initialize the visualizer with the path to the JSON file
        
        Args:
            json_path: Path to the hierarchy analysis JSON file
        """
        self.json_path = json_path
        self.G = nx.Graph()
        self.node_colors = []
        self.node_sizes = []
        self.labels = {}
        
    def load_data(self) -> Dict:
        """Load the JSON data from file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
            
    def create_graph(self, data: Dict):
        """
        Create a networkx graph from the hierarchy data
        
        Args:
            data: Loaded JSON data
        """
        # Add root node
        self.G.add_node("root")
        self.node_colors.append("#1f77b4")  # Blue for root
        self.node_sizes.append(3000)
        self.labels["root"] = "Root"
        
        # Process clusters
        for cluster_id, cluster_data in data["root"]["clusters"].items():
            cluster_node = f"cluster_{cluster_id}"
            self.G.add_node(cluster_node)
            self.G.add_edge("root", cluster_node)
            
            # Add cluster node properties
            self.node_colors.append("#ff7f0e")  # Orange for clusters
            self.node_sizes.append(2000)
            self.labels[cluster_node] = f"Cluster {cluster_id}"
            
            # Process queries in cluster
            if "queries" in cluster_data:
                for query_idx, (query, query_data) in enumerate(cluster_data["queries"].items()):
                    # Limit number of queries per cluster to prevent overcrowding
                    if query_idx >= 5:
                        break
                        
                    query_node = f"query_{cluster_id}_{query_idx}"
                    self.G.add_node(query_node)
                    self.G.add_edge(cluster_node, query_node)
                    
                    # Add query node properties
                    self.node_colors.append("#2ca02c")  # Green for queries
                    self.node_sizes.append(1000)
                    # Truncate long queries for better visualization
                    truncated_query = query[:30] + "..." if len(query) > 30 else query
                    self.labels[query_node] = truncated_query

    def visualize(self, output_path: str = None):
        """
        Create and save the force-directed graph visualization
        
        Args:
            output_path: Path where to save the visualization
        """
        plt.figure(figsize=(20, 20))
        
        # Create layout
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Draw the graph
        nx.draw(self.G,
                pos=pos,
                node_color=self.node_colors,
                node_size=self.node_sizes,
                labels=self.labels,
                font_size=8,
                font_weight="bold",
                edge_color="#cccccc",
                width=1,
                with_labels=True)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor="#1f77b4", markersize=15, label='Root'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor="#ff7f0e", markersize=15, label='Clusters'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor="#2ca02c", markersize=15, label='Queries')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        # Save or show the visualization
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    # Set up paths
    json_path = "/home/self-rag/scripts/historical_database_analysis/output/mobilecs2/hierarchy/hierarchy_analysis.json"
    output_path = "/home/self-rag/scripts/historical_database_analysis/output/mobilecs2/hierarchy/hierarchy_visualization.png"
    
    # Create visualizer
    visualizer = HierarchyVisualizer(json_path)
    
    # Load and process data
    data = visualizer.load_data()
    if data:
        visualizer.create_graph(data)
        visualizer.visualize(output_path)
    else:
        print("Failed to load data. Please check the JSON file path.")

if __name__ == "__main__":
    main()