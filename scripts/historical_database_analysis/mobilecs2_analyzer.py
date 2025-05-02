import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # This is to avoid warnings in Hugging Face Tokenizers
import sys
import json
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import time
import networkx as nx
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# 设置基础日志配置
def setup_root_logging():
    """Setup root logging configuration"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 清除现有的处理器
    root_logger.handlers = []
    root_logger.addHandler(console_handler)

# 在任何其他代码执行之前调用
setup_root_logging()

# 更新基础路径配置
BASE_DIR = "/home/self-rag"
DATASET_NAME = "mobilecs2" # 之后可以改为 "popqa", "triviaqa", "pubhealth", "coqa"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "scripts/historical_database_analysis/output", DATASET_NAME) # 修改输出目录，添加数据集名称子目录

class MobileCS2Analyzer:
    def __init__(self, data_path: str = None, output_dir: str = None, retrieval_config: Dict = None):
        """
        Initialize the analyzer with paths and basic setup
        """
        self.data_path = data_path or os.path.join(DATA_DIR, "valid_data_zfy/valid_conversations.json")
        self.output_dir = output_dir or os.path.join(BASE_DIR, "scripts/historical_database_analysis/output", "mobilecs2")
        
        # 检索配置
        self.retrieval_config = retrieval_config or {
            'top_k': 5,
            'similarity_threshold': 0.5,
            'retrieval_model': 'facebook/contriever-msmarco'
        }
        
        # 初始化编码器模型
        logging.info("Initializing sentence transformer model...")
        self.model = SentenceTransformer('facebook/contriever-msmarco')
        
        # 加载检索数据库
        self.retrieval_database = self.load_retrieval_database()
        
        # 加载或计算段落embeddings
        logging.info("Loading or computing passage embeddings...")
        self.passage_embeddings = self.load_or_compute_embeddings()
        
        # 创建输出目录
        for subdir in ['clustering', 'hierarchy', 'chain_of_thought', 'retrieval_results']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        self.setup_logging()
    def inspect_embeddings_file(self, file_path: str):
        """检查 embeddings 文件的格式"""
        logging.info(f"Inspecting embeddings file: {file_path}")
        raw_data = np.load(file_path, allow_pickle=True)
        
        logging.info(f"Type of loaded data: {type(raw_data)}")
        if isinstance(raw_data, tuple):
            for i, item in enumerate(raw_data):
                logging.info(f"Item {i} type: {type(item)}")
                if isinstance(item, np.ndarray):
                    logging.info(f"Array {i} shape: {item.shape}")
                    logging.info(f"Array {i} dtype: {item.dtype}")
        elif isinstance(raw_data, np.ndarray):
            logging.info(f"Array shape: {raw_data.shape}")
            logging.info(f"Array dtype: {raw_data.dtype}")
            
        return raw_data
    def compute_and_save_embeddings(self) -> np.ndarray:
        """计算并保存段落的embeddings"""
        logging.info("Computing passage embeddings...")
        
        # 计算所有段落的embeddings
        batch_size = 32
        all_embeddings = []
        passages = [p['text'] for p in self.retrieval_database]
        
        logging.info(f"Computing embeddings for {len(passages)} passages")
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings)
            if i % 100 == 0:
                logging.info(f"Processed {i}/{len(passages)} passages")
        
        embeddings_array = np.array(all_embeddings)
        logging.info(f"Created embeddings array with shape {embeddings_array.shape}")
        
        # 保存embeddings到文件
        embedding_path = os.path.join(os.path.dirname(__file__), "../../data/embedding_data/passages_00.npy")
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        
        np.save(embedding_path, embeddings_array)
        logging.info(f"Saved embeddings to {embedding_path}")
        
        return embeddings_array

    def load_or_compute_embeddings(self) -> np.ndarray:
        """加载或计算embeddings"""
        embedding_path = os.path.join(os.path.dirname(__file__), "../../data/embedding_data/passages_00.npy")
        
        if os.path.exists(embedding_path):
            try:
                embeddings = np.load(embedding_path)
                if embeddings.shape[0] == len(self.retrieval_database):  # 验证维度是否匹配
                    logging.info(f"Loaded precomputed embeddings with shape {embeddings.shape}")
                    return embeddings
            except Exception as e:
                logging.error(f"Error loading embeddings: {str(e)}")
        
        # 如果没有找到文件或加载失败，重新计算
        logging.info("Computing new embeddings...")
        return self.compute_and_save_embeddings()

    def retrieve_passages_for_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """为历史query检索相关段落，使用预计算的embeddings"""
        try:
            logging.info(f"Retrieving passages for query: {query[:50]}...")
            
            # 计算查询的嵌入向量
            query_embedding = self.model.encode([query])[0]
            logging.info(f"Query embedding shape: {query_embedding.shape}")
            logging.info(f"Passage embeddings shape: {self.passage_embeddings.shape}")
            
            # 计算相似度
            similarities = cosine_similarity([query_embedding], self.passage_embeddings)[0]
            
            # 获取最相似的段落
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            scored_passages = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > self.retrieval_config['similarity_threshold']:
                    passage = self.retrieval_database[idx]
                    scored_passages.append({
                        'text': passage['text'],
                        'score': float(similarity),
                        'id': passage.get('id', ''),
                        'title': passage.get('title', '')
                    })
            
            logging.info(f"Retrieved {len(scored_passages)} relevant passages")
            return scored_passages
                    
        except Exception as e:
            logging.error(f"Error retrieving passages for query {query}: {str(e)}")
            raise
    def load_retrieval_database(self) -> List[Dict]:
        """加载检索数据库"""
        database_path = os.path.join(DATA_DIR, "transformed_data.jsonl")
        
        if not os.path.exists(database_path):
            logging.error(f"Retrieval database not found at: {database_path}")
            logging.info(f"Current working directory: {os.getcwd()}")
            logging.info(f"Checking if parent directory exists: {os.path.exists(DATA_DIR)}")
            logging.info(f"Files in parent directory: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'Directory not found'}")
            raise FileNotFoundError(f"Database file not found: {database_path}")
        
        database = []
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                for line in f:
                    database.append(json.loads(line))
            logging.info(f"Successfully loaded {len(database)} entries from retrieval database")
        except Exception as e:
            logging.error(f"Error loading database: {str(e)}")
            raise
        
        return database



    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG,  # 修改日志级别为 DEBUG
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'analysis.log')),
                logging.StreamHandler()
            ]
        )

    def load_data(self, test_mode=False) -> Dict:
        """Load and parse the MobileCS2 dataset"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if test_mode:
                # 测试模式下只取前10个对话
                data = dict(list(data.items())[:10])
                logging.info("Running in test mode with reduced dataset")
        logging.info(f"Loaded data with {len(data)} conversations")
        return data

    def analyze_query_distribution(self, data: Dict) -> Dict:
        """
        Analyze the distribution of queries in conversations
        
        Returns:
            Dict containing statistics about query distribution
        """
        stats = {
            'total_conversations': len(data),
            'total_queries': 0,
            'new_queries': 0,
            'historical_queries': 0,
            'queries_per_conversation': []
        }
        
        for conv_id, conv_data in data.items():
            new_queries = sum(1 for msg in conv_data['log'] if msg.get('api_query') == '[QA]')
            total_queries = len(conv_data['log'])
            historical_queries = total_queries - new_queries
            
            stats['total_queries'] += total_queries
            stats['new_queries'] += new_queries
            stats['historical_queries'] += historical_queries
            stats['queries_per_conversation'].append({
                'total': total_queries,
                'new': new_queries,
                'historical': historical_queries
            })
            
        logging.info(f"Analysis complete: {stats['total_queries']} total queries")
        return stats

    def perform_clustering(self, data: Dict, n_clusters: int = 10) -> Dict:
        """Perform clustering on historical queries with quality assessment"""
        logging.info("Starting clustering process...")
        
        # 收集历史查询
        logging.info("Collecting historical queries...")
        historical_queries = []
        for conv_id, conv_data in data.items():
            for msg in conv_data['log']:
                if msg.get('api_query') != '[QA]':
                    historical_queries.append(msg['user'])
        
        logging.info(f"Found {len(historical_queries)} historical queries")
        
        # 调整聚类数
        if len(historical_queries) < n_clusters:
            n_clusters = max(2, len(historical_queries) // 2)
            logging.info(f"Adjusted number of clusters to {n_clusters}")
        
        # 转换为向量表示
        logging.info("Converting queries to embeddings...")
        embeddings = []
        batch_size = 32  # 添加批处理
        for i in range(0, len(historical_queries), batch_size):
            batch = historical_queries[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
            if i % 100 == 0:
                logging.info(f"Processed {i}/{len(historical_queries)} queries")
        embeddings = np.array(embeddings)
        
        # 执行聚类
        logging.info("Performing KMeans clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # 明确设置 n_init
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 计算聚类质量
        logging.info("Calculating clustering quality...")
        silhouette_avg = 0
        if len(set(cluster_labels)) > 1:
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        # 组织结果
        logging.info("Organizing clustering results...")
        clusters = defaultdict(list)
        for query, label in zip(historical_queries, cluster_labels):
            clusters[int(label)].append(query)
        
        cluster_info = {
            'n_clusters': n_clusters,
            'total_historical_queries': len(historical_queries),
            'cluster_sizes': {str(i): len(cluster) for i, cluster in clusters.items()},
            'clusters': {str(k): v for k, v in clusters.items()},
            'silhouette_score': silhouette_avg,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        logging.info("Clustering process completed")
        return cluster_info

    def analyze_hierarchical_structure(self, data: Dict, cluster_info: Dict) -> Dict:
        hierarchy_tree = {
            'root': {
                'clusters': {}  # 第一层：聚类类别
            }
        }
        
        # 初始化聚类节点
        for cluster_id in range(cluster_info['n_clusters']):
            cluster_id_str = str(cluster_id)
            hierarchy_tree['root']['clusters'][cluster_id_str] = {
                'historical_queries': {},  # 第二层：历史query节点
                'metadata': {
                    'count': 0,
                    'description': f'Historical Query Cluster {cluster_id}'
                }
            }
        
        # 处理历史query信息
        for conv_id, conv_data in data.items():
            for msg in conv_data['log']:
                if msg.get('api_query') != '[QA]':
                    query = msg['user']
                    
                    # 为该历史query检索相关段落
                    retrieved_passages = self.retrieve_passages_for_query(query, top_k=5)  # 获取top-k个相关段落
                    
                    # 构建query节点的信息
                    query_node = {
                        'retrieved_passages': retrieved_passages,  # 主要信息：检索段落
                        'metadata': {  # 保留少量有用的元信息
                            'frequency': 1,  # 可以后续累加
                            'success_rate': self.calculate_success_rate(conv_data, msg),  # 可选：计算该查询的成功率
                            'avg_response_time': self.calculate_avg_response_time(conv_data, msg)  # 可选：平均响应时间
                        }
                    }
                    
                    # 找到对应的聚类
                    query_vector = self.model.encode([query])[0]
                    nearest_cluster = self.find_nearest_cluster(query_vector, cluster_info)
                    
                    # 更新树结构
                    cluster_node = hierarchy_tree['root']['clusters'][str(nearest_cluster)]
                    if query in cluster_node['historical_queries']:
                        # 如果查询已存在，更新统计信息
                        cluster_node['historical_queries'][query]['metadata']['frequency'] += 1
                    else:
                        # 新增查询节点
                        cluster_node['historical_queries'][query] = query_node
                        cluster_node['metadata']['count'] += 1
        
        return hierarchy_tree

    def find_nearest_cluster(self, query_vector: np.ndarray, cluster_info: Dict) -> str:
        """
        找到最近的聚类
        
        Args:
            query_vector: 查询向量
            cluster_info: 聚类信息
        
        Returns:
            str: 最近聚类的ID
        """
        min_distance = float('inf')
        nearest_cluster = "0"  # 默认聚类ID
        
        # 使用聚类中心计算距离
        for cluster_id in range(cluster_info['n_clusters']):
            cluster_center = cluster_info['cluster_centers'][cluster_id]
            distance = np.linalg.norm(query_vector - cluster_center)
            
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = str(cluster_id)
        
        return nearest_cluster

    def calculate_success_rate(self, conv_data: Dict, msg: Dict) -> float:
        """
        计算查询成功率
        """
        try:
            # 简单的成功率计算：如果有系统回复就认为是成功
            return 1.0 if 'system' in msg and msg['system'] else 0.0
        except Exception as e:
            logging.error(f"Error calculating success rate: {str(e)}")
            return 0.0

    def calculate_avg_response_time(self, conv_data: Dict, msg: Dict) -> float:
        """
        计算平均响应时间
        """
        try:
            # 由于数据中可能没有实际的时间戳，这里返回一个默认值
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating response time: {str(e)}")
            return 0.0


    def load_cluster_results(self):
        """加载聚类结果"""
        cluster_file = os.path.join(self.output_dir, 'clustering', 'cluster_results.json')
        with open(cluster_file, 'r') as f:
            return json.load(f)


    def find_cluster_for_query(self, query: str, cluster_info: Dict) -> str:
        """
        查找查询所属的聚类
        
        Args:
            query: 需要归类的查询文本
            cluster_info: 聚类信息字典
        
        Returns:
            str: 最匹配的聚类ID
        """
        logging.info(f"Finding cluster for query: {query[:50]}...")  # 只显示前50个字符
        
        # 将查询转换为向量
        query_vector = self.model.encode([query])[0]
        
        # 计算与每个聚类中心的距离
        min_distance = float('inf')
        best_cluster = "0"  # 默认聚类
        
        total_clusters = len(cluster_info['clusters'])
        for i, cluster_id in enumerate(cluster_info['clusters'].keys()):
            if i % 10 == 0:  # 每处理10个聚类打印一次日志
                logging.info(f"Processing cluster {i}/{total_clusters}")
                
            cluster_vectors = []
            
            # 获取该聚类中所有查询的向量
            try:
                for query_item in cluster_info['clusters'][cluster_id]:
                    try:
                        # 根据查询的数据结构来获取查询文本
                        query_text = query_item if isinstance(query_item, str) else query_item[0]
                        cluster_vectors.append(self.model.encode([query_text])[0])
                    except Exception as e:
                        logging.warning(f"Error encoding query in cluster {cluster_id}: {str(e)}")
                        continue
                
                if cluster_vectors:
                    # 计算聚类中心
                    cluster_center = np.mean(cluster_vectors, axis=0)
                    # 计算距离
                    distance = np.linalg.norm(query_vector - cluster_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
                        
            except Exception as e:
                logging.error(f"Error processing cluster {cluster_id}: {str(e)}")
                continue
        
        logging.info(f"Best matching cluster found: {best_cluster}")
        return best_cluster  # 返回字符串形式的cluster_id


    def generate_hierarchy_summary(self, hierarchy: Dict) -> Dict:
        """
        生成层次结构的概要分析
        """
        summary = {
            'cluster_stats': {},
            'total_unique_queries': 0,
            'query_frequency_distribution': {},
            'context_analysis': {}
        }
        
        try:
            # 遍历每个聚类
            for cluster_id, cluster_data in hierarchy['root']['clusters'].items():
                queries = cluster_data.get('queries', {})
                
                # 计算该聚类的统计数据
                query_frequencies = []
                total_responses = 0
                for q_data in queries.values():
                    frequency = q_data['metadata']['frequency']
                    query_frequencies.append(frequency)
                    total_responses += len(q_data.get('responses', []))
                
                summary['cluster_stats'][cluster_id] = {
                    'total_unique_queries': len(queries),
                    'total_responses': total_responses,
                    'avg_frequency': float(np.mean(query_frequencies)) if query_frequencies else 0,
                    'queries_sample': list(queries.keys())[:5]  # 保存前5个查询作为样本
                }
            
            # 计算整体统计
            summary['total_unique_queries'] = sum(
                stats['total_unique_queries'] 
                for stats in summary['cluster_stats'].values()
            )
            
            # 保存摘要
            summary_file = os.path.join(self.output_dir, 'hierarchy', 'hierarchy_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            return summary
            
        except Exception as e:
            logging.error(f"Error generating hierarchy summary: {str(e)}")
            return {}


    def visualize_tree_structure(self, hierarchy: Dict):
        """
        创建并保存树结构的可视化图
        """
        try:
            logging.info("Starting tree structure visualization")
            logging.info(f"Number of clusters in hierarchy: {len(hierarchy['root']['clusters'])}")
            
            G = nx.DiGraph()
            G.add_node("root")
            
            query_count = 0
            # 添加第一层：聚类节点
            for cluster_id, cluster_data in hierarchy['root']['clusters'].items():
                cluster_label = f"Cluster {cluster_id}"
                G.add_edge("root", cluster_label)
                logging.info(f"Processing cluster {cluster_id} with {len(cluster_data.get('queries', {}))} queries")
                
                # 添加第二层：查询节点
                for query, query_data in cluster_data.get('queries', {}).items():
                    query_count += 1
                    query_label = query[:30] + "..." if len(query) > 30 else query
                    G.add_edge(cluster_label, query_label)
                    G.nodes[query_label]['frequency'] = query_data['metadata']['frequency']
            
            logging.info(f"Total queries visualized: {query_count}")
            
            plt.figure(figsize=(20, 10))
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=2000, font_size=8, arrows=True)
            
            plt.savefig(os.path.join(self.output_dir, 'tree_structure.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Tree structure visualization completed and saved")
            
        except Exception as e:
            logging.error(f"Error in tree structure visualization: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())


    def extract_context(self, log: List, current_msg: Dict) -> Dict:
        """提取查询的上下文信息"""
        # 获取当前消息的索引
        current_idx = log.index(current_msg)
        
        context = {
            'previous_queries': [],
            'following_queries': [],
            'temporal_info': {
                'timestamp': time.time(),
                'message_position': current_idx
            },
            'semantic_info': {
                'topic': self.extract_topic(current_msg['user']),
                'is_new_query': current_msg.get('api_query') == '[QA]'
            }
        }
        
        # 获取前面的查询
        if current_idx > 0:
            context['previous_queries'] = [
                msg['user'] for msg in log[max(0, current_idx-3):current_idx]
            ]
        
        # 获取后面的查询
        if current_idx < len(log) - 1:
            context['following_queries'] = [
                msg['user'] for msg in log[current_idx+1:min(len(log), current_idx+4)]
            ]
        
        return context


    def analyze_chain_of_thought(self, data: Dict, hierarchy: Dict) -> Dict:
        """
        基于新的层次结构进行chain of thought分析
        """
        chains = []
        
        for conv_id, conv_data in data.items():
            current_chain = []
            historical_queries = []
            
            for msg in conv_data['log']:
                if msg.get('api_query') == '[QA]':
                    new_query = msg['user']
                    
                    # 1. 找到匹配的聚类和历史查询
                    cluster_matches = self.find_matching_clusters(new_query, hierarchy)
                    
                    chain_entry = {
                        'new_query': new_query,
                        'reasoning_chain': [],
                        'retrieved_info': {
                            'traditional_rag': [],  # 传统RAG检索的段落
                            'historical_passages': []  # 从历史查询获取的段落
                        }
                    }
                    
                    # 2. 从传统RAG获取检索段落
                    traditional_passages = self.retrieve_passages_for_query(
                        new_query, 
                        top_k=self.retrieval_config['top_k']
                    )
                    chain_entry['retrieved_info']['traditional_rag'] = traditional_passages
                    
                    # 3. 从历史查询中获取相关段落
                    for cluster_id, similarity in cluster_matches:
                        cluster = hierarchy['root']['clusters'][str(cluster_id)]
                        matching_queries = self.find_similar_queries_in_cluster(
                            new_query, 
                            cluster['historical_queries']
                        )
                        
                        for query, query_similarity in matching_queries:
                            historical_passages = cluster['historical_queries'][query]['retrieved_passages']
                            
                            # 构建推理步骤
                            reasoning_step = {
                                'query': query,
                                'similarity': query_similarity,
                                'cluster_id': cluster_id,
                                'passages': historical_passages,
                                'reasoning': self.generate_reasoning(
                                    new_query, 
                                    query, 
                                    historical_passages
                                )
                            }
                            chain_entry['reasoning_chain'].append(reasoning_step)
                            chain_entry['retrieved_info']['historical_passages'].extend(historical_passages)
                    
                    current_chain.append(chain_entry)
                else:
                    historical_queries.append(msg)
            
            if current_chain:
                chains.append(current_chain)
        
        return self.generate_chain_analysis(chains)


    def find_matching_clusters(self, query: str, hierarchy: Dict, top_k: int = 3) -> List[Tuple[int, float]]:
        """找到与查询最匹配的聚类"""
        query_embedding = self.model.encode([query])[0]
        cluster_similarities = []
        
        for cluster_id, cluster in hierarchy['root']['clusters'].items():
            # 计算与聚类中心的相似度
            similarity = cosine_similarity(
                [query_embedding], 
                [np.mean([
                    self.model.encode([q])[0] 
                    for q in cluster['historical_queries'].keys()
                ], axis=0)]
            )[0][0]
            
            cluster_similarities.append((int(cluster_id), similarity))
        
        return sorted(cluster_similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def find_similar_queries_in_cluster(
        self, 
        query: str, 
        historical_queries: Dict, 
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """在聚类中找到与查询最相似的历史查询"""
        query_embedding = self.model.encode([query])[0]
        similarities = []
        
        for hist_query in historical_queries:
            hist_embedding = self.model.encode([hist_query])[0]
            similarity = cosine_similarity([query_embedding], [hist_embedding])[0][0]
            similarities.append((hist_query, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def generate_reasoning(
        self, 
        new_query: str, 
        historical_query: str, 
        passages: List[Dict]
    ) -> str:
        """生成推理步骤的描述"""
        return (
            f"Based on historical query '{historical_query}' which is similar to "
            f"current query '{new_query}', we can utilize {len(passages)} relevant "
            f"passages to provide information."
        )



    def find_similar_historical_queries(self, new_query: str, historical_queries: List[Dict], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        找到与new query最相似的历史query
        
        Args:
            new_query: 新查询
            historical_queries: 历史查询列表
            top_k: 返回的最相似查询数量
        
        Returns:
            List[Tuple[str, float]]: 相似查询及其相似度得分
        """
        if not historical_queries:
            return []
            
        # 使用模型计算相似度
        new_query_embedding = self.model.encode(new_query)
        historical_embeddings = self.model.encode([h['query'] for h in historical_queries])
        
        # 计算相似度
        similarities = cosine_similarity(
            [new_query_embedding], 
            historical_embeddings
        )[0]
        
        # 获取top-k相似的查询
        similar_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_queries = [
            (historical_queries[idx]['query'], similarities[idx])
            for idx in similar_indices
            if similarities[idx] > 0.5  # 设置相似度阈值
        ]
        
        return similar_queries

    def get_retrieval_passages(self, similar_queries: List[Tuple[str, float]]) -> Dict[str, List[str]]:
        """
        获取历史query对应的检索段落
        
        Args:
            similar_queries: 相似查询列表(query, similarity score)
        
        Returns:
            Dict[str, List[str]]: 每个查询对应的检索段落
        """
        retrieval_results = {}
        
        # 读取检索结果文件
        retrieval_file = os.path.join(self.output_dir, "retrieval_results.jsonl")
        if os.path.exists(retrieval_file):
            with open(retrieval_file, 'r') as f:
                retrieval_data = [json.loads(line) for line in f]
                
            # 为每个查询获取检索段落
            for query, _ in similar_queries:
                passages = []
                for data in retrieval_data:
                    if data['query'] == query and 'passages' in data:
                        passages = data['passages']
                        break
                retrieval_results[query] = passages
        
        return retrieval_results


    def extract_topic(self, query: str) -> str:
        """Helper method to extract topic from query"""
        # Simple keyword-based topic extraction
        keywords = {
            'data': 'data_plan',
            'bill': 'billing',
            'charge': 'billing',
            'package': 'subscription',
            'problem': 'technical_support'
        }
        
        query_lower = query.lower()
        for keyword, topic in keywords.items():
            if keyword in query_lower:
                return topic
        return 'general'

    def visualize_results(self, stats: Dict, cluster_info: Dict, hierarchy: Dict):
        """
        Create visualizations for analysis results
        """
        # Query type distribution (New vs Historical)
        plt.figure(figsize=(10, 6))
        plt.bar(['Historical Queries', 'New Queries'], 
                [stats['historical_queries'], stats['new_queries']])
        plt.title('Query Type Distribution')
        plt.savefig(os.path.join(self.output_dir, 'query_distribution.png'))
        plt.close()
        
        # Historical query cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_sizes = {f'Cluster {k}': v for k, v in cluster_info['cluster_sizes'].items()}
        plt.bar(cluster_sizes.keys(), cluster_sizes.values())
        plt.title('Historical Query Cluster Sizes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_sizes.png'))
        plt.close()
        
        # Historical query frequency in hierarchy
        plt.figure(figsize=(10, 6))
        cluster_counts = {
            f'Cluster {cluster}': data['metadata']['count'] 
            for cluster, data in hierarchy['root']['clusters'].items()
        }
        plt.bar(cluster_counts.keys(), cluster_counts.values())
        plt.title('Historical Queries per Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'))
        plt.close()
            

def main():
    print("Starting the program...")  # 添加直接打印
    try:
        print("Initializing logging...")  # 添加直接打印
        logging.info("Starting analysis process...")
        
        # 1. 参数配置 (移到最前面)
        parser = argparse.ArgumentParser(description='MobileCS2 Analysis Tool')
        parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, "valid_data_zfy/valid_conversations.json"))
        parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
        parser.add_argument('--n_clusters', type=int, default=10)
        parser.add_argument('--top_k', type=int, default=5)
        parser.add_argument('--similarity_threshold', type=float, default=0.5)
        args = parser.parse_args()
        
        # 2. 检查路径
        logging.info(f"Checking paths...")
        logging.info(f"BASE_DIR: {BASE_DIR}")
        logging.info(f"DATA_DIR: {DATA_DIR}")
        logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
        
        if not os.path.exists(DATA_DIR):
            logging.error(f"Data directory not found: {DATA_DIR}")
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
            
        # 检查关键文件
        required_files = [
            os.path.join(DATA_DIR, "transformed_data.jsonl"),
            os.path.join(DATA_DIR, "valid_data_zfy/valid_conversations.json")
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logging.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
                
        logging.info("All required paths and files exist")

        # 3. 初始化分析器
        retrieval_config = {
            'top_k': args.top_k,
            'similarity_threshold': args.similarity_threshold
        }
        analyzer = MobileCS2Analyzer(
            args.data_path, 
            args.output_dir,
            retrieval_config=retrieval_config
        )

        # 4. 加载数据
        logging.info("Loading conversation data...")
        data = analyzer.load_data()
        logging.info(f"Loaded {len(data)} conversations")
        
        # 5. 执行分析流程
        # 5.1 聚类分析
        logging.info("Starting clustering analysis...")
        cluster_info = analyzer.perform_clustering(data, n_clusters=args.n_clusters)
        logging.info(f"Created {cluster_info['n_clusters']} clusters")

        # 5.2 构建层次结构
        logging.info("Building hierarchical structure...")
        hierarchy = analyzer.analyze_hierarchical_structure(data, cluster_info)
        logging.info("Built hierarchical structure with retrieved passages")

        # 5.3 Chain of Thought分析
        logging.info("Performing chain of thought analysis...")
        chain_analysis = analyzer.analyze_chain_of_thought(data, hierarchy)
        logging.info(f"Completed chain of thought analysis")

        # 5.4 查询分布分析
        logging.info("Analyzing query distribution...")
        query_stats = analyzer.analyze_query_distribution(data)

        # 6. 生成和保存结果
        logging.info("Generating and saving final results...")
        final_results = {
            'query_stats': query_stats,
            'clustering': {
                'n_clusters': cluster_info['n_clusters'],
                'total_queries': cluster_info['total_historical_queries'],
                'cluster_sizes': cluster_info['cluster_sizes']
            },
            'hierarchy': {
                'total_passages_retrieved': sum(
                    len(q['retrieved_passages'])
                    for c in hierarchy['root']['clusters'].values()
                    for q in c['historical_queries'].values()
                ),
                'clusters_summary': {
                    str(cid): {
                        'query_count': len(cluster['historical_queries']),
                        'total_passages': sum(
                            len(q['retrieved_passages'])
                            for q in cluster['historical_queries'].values()
                        )
                    }
                    for cid, cluster in hierarchy['root']['clusters'].items()
                }
            },
            'chain_of_thought': {
                'total_chains': chain_analysis['total_chains'],
                'avg_chain_length': chain_analysis['avg_chain_length'],
                'avg_passages_per_chain': np.mean([
                    len(c['retrieved_info']['traditional_rag']) + 
                    len(c['retrieved_info']['historical_passages'])
                    for chain in chain_analysis['chain_details']
                    for c in chain
                ])
            }
        }

        # 7. 保存结果
        output_file = os.path.join(args.output_dir, 'analysis_summary.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Analysis results saved to {output_file}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()