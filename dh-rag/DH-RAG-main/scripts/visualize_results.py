import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import traceback  # 添加此行

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    logging.warning("无法设置中文字体，将使用默认字体")

def visualize_clustering(clustering_output_path, clustering_vis_dir):
    """
    可视化 Historical Query Clustering 的结果，生成聚类数量柱状图。
    """
    try:
        os.makedirs(clustering_vis_dir, exist_ok=True)
        output_image_path = os.path.join(clustering_vis_dir, "clustering_results.png")
        
        with open(clustering_output_path, 'r', encoding='utf-8') as f:
            clustering_output = json.load(f)
        
        n_clusters = clustering_output['n_clusters']
        cluster_sizes = [len(queries) for queries in clustering_output['clusters'].values()]
        cluster_ids = list(range(n_clusters))  # 使用数字作为cluster IDs
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=cluster_ids, y=cluster_sizes, color='blue')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Queries')
        plt.title('Historical Query Clustering Results')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        
        logging.info(f"Clustering results saved to {output_image_path}")
    except Exception as e:
        logging.error(f"Error visualizing clustering results: {str(e)}")

def visualize_hierarchical_matching_alternative(hierarchical_output_path, hierarchical_vis_dir):
    """
    使用matplotlib来可视化层次匹配结果
    """
    try:
        os.makedirs(hierarchical_vis_dir, exist_ok=True)
        
        # 修改为一次性读取整个JSON文件
        with open(hierarchical_output_path, 'r', encoding='utf-8') as f:
            hierarchical_output = json.load(f)
        
        # 绘制查询节点
        plt.figure(figsize=(15, 10))
        query = hierarchical_output['query']
        plt.text(0.5, 0.95, f"Query: {query[:100]}", 
                horizontalalignment='center', 
                bbox=dict(facecolor='lightblue', alpha=0.5))
        
        # 为每个集群创建可视化
        n_clusters = len(hierarchical_output['top_clusters'])
        for i, cluster in enumerate(hierarchical_output['top_clusters']):
            # 计算x位置
            x = (i + 1) / (n_clusters + 1)
            
            # 绘制集群节点
            plt.text(x, 0.7, 
                    f"Cluster {cluster['cluster_id']}\nSimilarity: {cluster['cluster_similarity']:.2f}", 
                    horizontalalignment='center',
                    bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            # 连接到根节点
            plt.plot([0.5, x], [0.93, 0.75], 'k-', alpha=0.3)
            
            # 绘制匹配项
            for j, match in enumerate(cluster['matches'][:5]):  # 只显示前5个匹配
                y = 0.5 - j * 0.1
                if y > 0:
                    plt.text(x, y,
                            f"Match: {match['query'][:50]}...\nSim: {match['similarity']:.2f}\n" +
                            f"Combined: {match.get('combined_similarity', 0):.2f}",
                            horizontalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.5))
                    plt.plot([x, x], [0.65, y + 0.05], 'k-', alpha=0.3)
        
        plt.axis('off')
        plt.tight_layout()
        
        # 生成安全的文件名
        safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_path = os.path.join(hierarchical_vis_dir, f"hierarchical_matching_{safe_query}.png")
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Created hierarchical visualization for query: {query[:50]}...")

    except Exception as e:
        logging.error(f"Error in hierarchical matching visualization: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")

def visualize_chain_of_thought(chain_output_path, chain_vis_dir):
    """
    可视化 Chain of Thought Tracking 的结果，生成每个查询的相似度变化折线图。
    """
    try:
        os.makedirs(chain_vis_dir, exist_ok=True)
        
        with open(chain_output_path, 'r', encoding='utf-8') as f:
            chain_outputs = [json.loads(line) for line in f]
        
        for chain in chain_outputs:
            query = chain['query']
            steps = chain.get('chain_of_thought', [])
            if not steps:
                logging.warning(f"Query '{query[:50]}...' has no chain of thought results, skipping.")
                continue
            
            step_numbers = [step['step'] for step in steps]
            similarities = [step['similarity'] for step in steps]
            
            # 绘制折线图，使用渐变色
            plt.figure(figsize=(10, 6))
            norm = Normalize(vmin=min(similarities), vmax=max(similarities))
            sm = ScalarMappable(cmap='viridis', norm=norm)
            
            # 绘制折线图并添加渐变色
            for i in range(1, len(steps)):
                plt.plot(step_numbers[i-1:i+1], similarities[i-1:i+1], color=sm.to_rgba(similarities[i-1]))
            
            plt.xlabel('Step')
            plt.ylabel('Similarity')
            plt.title(f'Chain of Thought for Query: {query[:50]}...')
            plt.xticks(step_numbers)
            plt.ylim(0, 1)
            plt.colorbar(sm, label='Similarity')  # Colorbar for similarity scale
            plt.grid(True)
            plt.tight_layout()

            # 保存文件
            safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            plot_filename = f"chain_of_thought_{safe_query}.png"
            plot_path = os.path.join(chain_vis_dir, plot_filename)
            
            plt.savefig(plot_path)
            plt.close()
            
            logging.info(f"Chain of thought visualization saved to {plot_path}")
    
    except Exception as e:
        logging.error(f"Error visualizing chain of thought results: {str(e)}")

def main():
    # 设置基本路径
    BASE_DIR = "./output"
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # 创建专门的可视化目录
    VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    CLUSTERING_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, "clustering")
    HIERARCHICAL_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, "hierarchical_matching")
    CHAIN_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, "chain_of_thought")
    
    # 确保所有目录存在
    for dir_path in [OUTPUT_DIR, VISUALIZATIONS_DIR, CLUSTERING_VIS_DIR, HIERARCHICAL_VIS_DIR, CHAIN_VIS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 定义输入文件路径
    CLUSTERING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "historical_query_clustering_output.json")
    HIERARCHICAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hierarchical_matching_output.jsonl")
    CHAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "chain_of_thought_results.jsonl")

    # 可视化聚类结果
    if os.path.exists(CLUSTERING_OUTPUT_PATH):
        visualize_clustering(CLUSTERING_OUTPUT_PATH, CLUSTERING_VIS_DIR)
    else:
        logging.error(f"Clustering output file not found: {CLUSTERING_OUTPUT_PATH}")
    
    # 可视化层次匹配结果
    if os.path.exists(HIERARCHICAL_OUTPUT_PATH):
        visualize_hierarchical_matching_alternative(HIERARCHICAL_OUTPUT_PATH, HIERARCHICAL_VIS_DIR)
    else:
        logging.error(f"Hierarchical matching output file not found: {HIERARCHICAL_OUTPUT_PATH}")
    
    # 可视化思维链结果
    if os.path.exists(CHAIN_OUTPUT_PATH):
        visualize_chain_of_thought(CHAIN_OUTPUT_PATH, CHAIN_VIS_DIR)
    else:
        logging.error(f"Chain of thought output file not found: {CHAIN_OUTPUT_PATH}")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()