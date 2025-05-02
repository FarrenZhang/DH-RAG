import json
import os
import sys
import logging
from collections import namedtuple
import argparse
from datetime import datetime
import openai
import time
import src.index
from src.passage_retrieval_zfy import Retriever
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from src.query_history_match import QueryMatcher
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import torch
import hashlib

RETRIEVAL_DATABASE_PATH = "/home/dh-rag/data/mobilecs2_data/topiocqa_jm/retrieval_database.jsonl"
QUERY_CONTEXT_MAPPING_PATH = "/home/dh-rag/data/mobilecs2_data/topiocqa_jm/query_context_mapping.json"

# 加载预训练模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 设置 OpenAI API 配置
openai.api_base = "https://api.holdai.top/v1"
openai.api_key = "sk-XCyY4uo7POh8OuMp97EeE08eFc2c43528076361f5f498eF1"

# 设置输出目录结构
BASE_DIR = "./topiocqa"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = BASE_DIR
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 确保目录存在
for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 生成基于当前时间的日志文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = f'app_{current_time}.log'
LOG_FILE = os.path.join(LOG_DIR, log_file_name)

# 设置日志
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 添加一个控制台处理器，这样日志也会输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 记录程序开始运行的日志
logging.info(f"Program started. Log file: {LOG_FILE}")

# 文件路径
INPUT_DATA_FILE = os.path.join(INPUT_DIR, "test_input.json")
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, "test_output.json")
RETRIEVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "retrieval_results.jsonl")
HISTORY_MATCH_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "history_match_output.jsonl")
HISTORICAL_DATABASE_FILE = os.path.join(OUTPUT_DIR, "historical_database.jsonl")
CLUSTERED_DATABASE_FILE = os.path.join(OUTPUT_DIR, "clustered_database.jsonl")
CHAIN_OF_THOUGHT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "chain_of_thought_results.jsonl")
INTEGRATED_RESULTS_FILE = os.path.join(OUTPUT_DIR, "integrated_results.jsonl")
GPT3_ANSWERS_FILE = os.path.join(OUTPUT_DIR, "gpt3_answers.jsonl")

ConversationData = namedtuple('ConversationData', ['history', 'query'])

#Added by Claude
def preprocess_data(input_file, output_file, mapping_file):
    all_contexts = {}
    query_context_mapping = {}

    with open(input_file, 'r') as f:
        data = json.load(f)
        for conv_id, conversation in data.items():
            for message in conversation['log']:
                if message['api_query'] == "[QA]":
                    query = message['user']
                    query_id = hashlib.md5(query.encode()).hexdigest()
                    query_context_mapping[query_id] = []

                    if 'ctxs' in message:
                        for ctx in message['ctxs']:
                            ctx_id = hashlib.md5(ctx['text'].encode()).hexdigest()
                            
                            if ctx_id not in all_contexts:
                                all_contexts[ctx_id] = ctx
                            
                            query_context_mapping[query_id].append(ctx_id)

    with open(output_file, 'w') as f:
        for ctx_id, ctx in all_contexts.items():
            ctx['id'] = ctx_id
            json.dump(ctx, f)
            f.write('\n')

    with open(mapping_file, 'w') as f:
        json.dump(query_context_mapping, f)

    return query_context_mapping

def load_data(data_path):
    try:
        with open(data_path, "r") as fin:
            if data_path.endswith(".json"):
                data = json.load(fin)
            elif data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in fin]
            else:
                raise ValueError("Unsupported file format. Use .json or .jsonl")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {str(e)}")
        return None

# 新增: 加载历史数据库的函数
def load_historical_database():
    historical_database = []
    if os.path.exists(HISTORICAL_DATABASE_FILE):
        with open(HISTORICAL_DATABASE_FILE, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                historical_database.append((data['query'], data['answer'], data['timestamp']))
    return historical_database
# 修改: 聚类函数
def cluster_historical_database(historical_database):
    queries = [item[0] for item in historical_database]
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(queries)
    
    n_clusters = min(10, len(queries)) # 要想改变聚类的种类多少，可以修改这个值
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    with open(CLUSTERED_DATABASE_FILE, 'w') as f:
        for (query, answer, timestamp, standard_answer), label in zip(historical_database, cluster_labels):
            json.dump({
                'query': query,
                'answer': answer,
                'timestamp': timestamp,
                'standard_answer': standard_answer,
                'cluster': int(label)
            }, f)
            f.write('\n')
    
    return list(zip(historical_database, cluster_labels))

# 同时更新 load_clustered_database 函数
def load_clustered_database():
    clustered_database = []
    if os.path.exists(CLUSTERED_DATABASE_FILE):
        with open(CLUSTERED_DATABASE_FILE, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                clustered_database.append(((data['query'], data['answer'], data['timestamp'], data['standard_answer']), data['cluster']))
    return clustered_database

def cosine_similarity_numpy(A, B):
    return np.dot(A, B.T) / (np.linalg.norm(A) * np.linalg.norm(B))

# 3. 添加层次匹配函数
def hierarchical_matching(query, historical_database):
    """
    执行层次匹配
    
    参数:
    query (str): 用户的查询
    historical_database (list): 历史数据库,每项包含(query, answer, timestamp, standard_answer)
    
    返回:
    list: 最相关的匹配结果列表
    """
    logging.info(f"Starting hierarchical matching for query: {query}")
    logging.info(f"Historical database size: {len(historical_database)}")
    
    # 1. 利用聚类信息
    clustered_database = load_clustered_database()
    logging.info(f"Clustered database size: {len(clustered_database)}")
    
    if not clustered_database:
        logging.warning("Clustered database is empty. Returning empty result.")
        return []
    
    # 2. 使用QueryMatcher中的函数计算相似度
    matcher = QueryMatcher(None, None)  # 初始化QueryMatcher
    
    # 3. 第一层:基于聚类的匹配
    cluster_similarities = []
    unique_clusters = set(cluster for _, cluster in clustered_database)
    logging.info(f"Number of unique clusters: {len(unique_clusters)}")
    
    for cluster in unique_clusters:
        cluster_queries = [item[0][0] for item, c in clustered_database if c == cluster]
        logging.info(f"Cluster {cluster} size: {len(cluster_queries)}")
        
        if not cluster_queries:
            logging.warning(f"No queries found for cluster {cluster}. Skipping.")
            continue
        
        try:
            cluster_similarity = max(matcher.calculate_similarity(query, cq) for cq in cluster_queries)
            cluster_similarities.append((cluster, cluster_similarity))
        except ValueError as e:
            logging.warning(f"Error calculating similarity for cluster {cluster}: {str(e)}")
            continue
    
    logging.info(f"Number of clusters with valid similarities: {len(cluster_similarities)}")
    
    if not cluster_similarities:
        logging.warning("No valid cluster similarities found. Returning empty result.")
        return []
    
    # 选择最相似的前3个聚类
    top_clusters = sorted(cluster_similarities, key=lambda x: x[1], reverse=True)[:3]
    logging.info(f"Top 3 clusters: {top_clusters}")
    
    # 4. 第二层:在选定的聚类中进行精细匹配
    fine_grained_matches = []
    for cluster, cluster_similarity in top_clusters:
        cluster_items = [(item, c) for item, c in clustered_database if c == cluster]
        for (historical_query, answer, timestamp, standard_answer), _ in cluster_items:
            try:
                similarity = matcher.calculate_similarity(query, historical_query)
                fine_grained_matches.append({
                    "query": historical_query,
                    "answer": answer,
                    "timestamp": timestamp,
                    "standard_answer": standard_answer,
                    "similarity": similarity
                })
            except ValueError as e:
                logging.warning(f"Error calculating similarity for query '{historical_query[:50]}...': {str(e)}")
                continue
    
    logging.info(f"Number of fine-grained matches: {len(fine_grained_matches)}")
    
    if not fine_grained_matches:
        logging.warning("No valid fine-grained matches found. Returning empty result.")
        return []
    
    # 5. 对精细匹配结果进行排序
    sorted_matches = sorted(fine_grained_matches, key=lambda x: x["similarity"], reverse=True)
    
    # 6. 使用TF-IDF进行额外的相似度检查
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([query] + [m["query"] for m in sorted_matches])
        tfidf_array = tfidf_matrix.toarray()
        cosine_similarities = cosine_similarity_numpy(tfidf_array[0:1], tfidf_array[1:]).flatten()
    except ValueError as e:
        logging.warning(f"Error in TF-IDF vectorization: {str(e)}")
        return sorted_matches[:5]  # 如果TF-IDF失败,直接返回前5个匹配
    
    # 7. 综合考虑初始相似度和TF-IDF相似度
    for i, cos_sim in enumerate(cosine_similarities):
        sorted_matches[i]["combined_similarity"] = (sorted_matches[i]["similarity"] + cos_sim) / 2
    
    # 8. 最终排序
    final_matches = sorted(sorted_matches, key=lambda x: x["combined_similarity"], reverse=True)
    
    logging.info(f"Returning top {min(5, len(final_matches))} matches")
    return final_matches[:5]  # 返回最相关的前5个匹配

def chain_of_thought_search(query, historical_database, max_steps=5, similarity_threshold=0.4):
    """
    执行扩展的Chain of Thought搜索
    
    参数:
    query (str): 用户的查询
    historical_database (list): 历史数据库,每项包含(query, answer, timestamp, rag_result)
    max_steps (int): 最大搜索步骤数
    similarity_threshold (float): 相似度阈值
    
    返回:
    list: 相关的历史记录列表
    """
    
    logging.info(f"Starting expanded chain of thought search for query: {query}")
    logging.info(f"Historical database size: {len(historical_database)}")
    logging.info(f"Similarity threshold: {similarity_threshold}")
    
    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    
    # 提取所有历史查询、答案和RAG结果，确保它们是字符串
    historical_queries = [str(item[0]) for item in historical_database]
    historical_answers = [str(item[1]) for item in historical_database]
    historical_rag_results = [str(item[3]) if len(item) > 3 else "" for item in historical_database]
    
    all_historical_texts = historical_queries + historical_answers + historical_rag_results
    
    logging.info(f"Number of historical texts: {len(all_historical_texts)}")
    
    try:
        # 拟合向量化器并转换所有文本
        tfidf_matrix = vectorizer.fit_transform(all_historical_texts + [query])
        logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    except Exception as e:
        logging.error(f"Error in TF-IDF vectorization: {str(e)}")
        return []
    
    # 初始化结果列表和已访问的项目集合
    results = []
    visited = set()
    
    def get_similar_items(current_query, tfidf_matrix, vectorizer, threshold):
        """获取与当前查询相似的历史项目"""
        try:
            current_vector = vectorizer.transform([current_query])
            similarities = cosine_similarity(current_vector, tfidf_matrix[:-1])[0]
            similar_indices = np.where(similarities > threshold)[0]
            similar_items = []
            for i in similar_indices:
                if i < len(historical_queries):
                    similar_items.append((historical_database[i], similarities[i], "query"))
                elif i < len(historical_queries) + len(historical_answers):
                    similar_items.append((historical_database[i - len(historical_queries)], similarities[i], "answer"))
                else:
                    similar_items.append((historical_database[i - len(historical_queries) - len(historical_answers)], similarities[i], "rag"))
            logging.info(f"Found {len(similar_items)} similar items with threshold {threshold}")
            return similar_items
        except Exception as e:
            logging.error(f"Error in get_similar_items: {str(e)}")
            return []
    
    def expand_query(current_query, item, item_type):
        """扩展查询,模拟思维链"""
        if item_type == "query":
            expanded = f"{current_query} {str(item[0]).split()[:5]}"
        elif item_type == "answer":
            expanded = f"{current_query} {str(item[1]).split()[:5]}"
        else:  # rag
            expanded = f"{current_query} {str(item[3]).split()[:5] if len(item) > 3 else ''}"
        logging.info(f"Expanded query: {expanded}")
        return expanded
    
    # 主搜索循环
    current_query = query
    for step in range(max_steps):
        logging.info(f"Search step {step + 1}")
        # 获取相似的项目
        similar_items = get_similar_items(current_query, tfidf_matrix, vectorizer, similarity_threshold)
        
        # 如果没有相似的项目,退出循环
        if not similar_items:
            logging.info(f"No similar items found in step {step + 1}. Exiting search.")
            break
        
        # 选择最相似的项目
        best_match, similarity, item_type = max(similar_items, key=lambda x: x[1])
        logging.info(f"Best match: {best_match[0]}, Similarity: {similarity}, Type: {item_type}")
        
        # 如果项目已被访问,退出循环
        if best_match[0] in visited:
            logging.info(f"Item {best_match[0]} already visited. Exiting search.")
            break
        
        # 添加到结果列表并标记为已访问
        results.append((best_match, similarity, item_type))
        visited.add(best_match[0])
        
        # 扩展查询
        current_query = expand_query(current_query, best_match, item_type)
    
    logging.info(f"Expanded chain of thought search completed. Found {len(results)} results.")
    
    # 在函数结束前存储结果
    try:
        with open(CHAIN_OF_THOUGHT_RESULTS_FILE, 'a') as f:
            result = {
                "query": query,
                "results": [{"match": r[0], "similarity": r[1], "type": r[2]} for r in results],
                "timestamp": time.time()
            }
            json.dump(result, f)
            f.write('\n')
        logging.info(f"Chain of thought results for query '{query[:50]}...' saved to {CHAIN_OF_THOUGHT_RESULTS_FILE}")
    except Exception as e:
        logging.error(f"Error saving chain of thought results: {str(e)}")
    
    return results

# 添加日志记录以帮助调试
def dynamic_retrieve(query, historical_database):
    hierarchical_matches = hierarchical_matching(query, historical_database)
    chain_of_thought_matches = chain_of_thought_search(query, historical_database)
    
    logging.info(f"Hierarchical matches type: {type(hierarchical_matches)}")
    logging.info(f"Chain of thought matches type: {type(chain_of_thought_matches)}")
    
    if hierarchical_matches:
        logging.info(f"Sample hierarchical match: {hierarchical_matches[0]}")
    if chain_of_thought_matches:
        logging.info(f"Sample chain of thought match: {chain_of_thought_matches[0]}")
    
    return hierarchical_matches, chain_of_thought_matches

def query_to_vector(query):
    """
    将查询转换为向量表示
    
    参数:
    query (str): 输入的查询文本
    
    返回:
    numpy.ndarray: 查询的向量表示
    """
    # 使用模型将查询转换为向量
    query_vector = model.encode(query, convert_to_tensor=True)
    return query_vector.cpu().numpy()

def result_to_vector(result):
    """
    将检索结果转换为向量表示
    
    参数:
    result (dict): 包含检索结果信息的字典
    
    返回:
    numpy.ndarray: 结果的向量表示
    """
    # 假设结果字典中有一个'content'键,包含主要文本内容
    # 如果没有'content'键,就使用字典中所有值的字符串表示
    if 'content' in result:
        text = result['content']
    else:
        text = ' '.join(str(v) for v in result.values())
    
    # 使用模型将文本转换为向量
    result_vector = model.encode(text, convert_to_tensor=True)
    return result_vector.cpu().numpy()

# 修改integrate_results函数以使用这些新函数
def integrate_results(static_results, hierarchical_matches, chain_of_thought_matches, query):
    """
    整合来自不同来源的检索结果
    
    参数:
    static_results: 传统知识库检索结果
    hierarchical_matches: 层次匹配结果
    chain_of_thought_matches: 思维链匹配结果 (全局匹配)
    query: 原始查询
    
    返回:
    list: 整合后的结果列表
    """
    def safe_extract(match):
        """安全地从匹配结果中提取信息"""
        if isinstance(match, dict):
            return match
        elif isinstance(match, (list, tuple)) and len(match) > 0:
            return match[0] if isinstance(match[0], dict) else {"content": str(match[0])}
        else:
            return {"content": str(match)}
    
    # 处理不同来源的结果
    Rv = [safe_extract(result) for result in static_results]
    Rh = [safe_extract(match) for match in hierarchical_matches]
    Rg = [safe_extract(match) for match in chain_of_thought_matches]
    
    # 合并所有结果
    R = Rv + Rh + Rg
    
    # 将查询和结果转换为向量表示
    query_vec = torch.tensor(query_to_vector(query)).unsqueeze(0)
    result_vecs = torch.tensor([result_to_vector(r) for r in R])
    
    # 计算注意力权重
    attention_weights = torch.nn.functional.softmax(torch.mm(query_vec, result_vecs.t()), dim=1)
    
    # 应用注意力权重
    weighted_results = []
    for i, result in enumerate(R):
        weighted_result = {**result, "weight": float(attention_weights[0][i])}
        weighted_results.append(weighted_result)
    
    # 按权重排序
    sorted_results = sorted(weighted_results, key=lambda x: x["weight"], reverse=True)
    
    # 在函数结束前存储结果
    try:
        with open(INTEGRATED_RESULTS_FILE, 'a') as f:
            result = {
                "query": query,
                "static_results": static_results,
                "hierarchical_matches": hierarchical_matches,
                "chain_of_thought_matches": chain_of_thought_matches,
                "integrated_results": sorted_results,
                "timestamp": time.time()
            }
            json.dump(result, f)
            f.write('\n')
        logging.info(f"Integrated results saved to {INTEGRATED_RESULTS_FILE}")
    except Exception as e:
        logging.error(f"Error saving integrated results: {str(e)}")
    
    return sorted_results

# 更新retrieve函数以使用新的integrate_results
def retrieve(args, paired_query, historical_database):
    retriever = Retriever(args)
    retriever.setup_retriever()
    
    results = []
    for idx, data in enumerate(paired_query):
        query = data.query['user'].strip()
        query_id = hashlib.md5(query.encode()).hexdigest()
        relevant_context_ids = query_context_mapping.get(query_id, [])
        
        # Perform search on all documents
        static_results = retriever.search_document(query, args.n_docs)
        
        # Filter results based on relevant context IDs
        filtered_static_results = [
            result for result in static_results 
            if result['id'] in relevant_context_ids
        ]
        
        hierarchical_matches, chain_of_thought_matches = dynamic_retrieve(query, historical_database)
        
        integrated_results = integrate_results(filtered_static_results, hierarchical_matches, chain_of_thought_matches, query)
        results.append({"query": query, "integrated_results": integrated_results})
    
    # Output and store results
    if args.output_dir:
        try:
            with open(RETRIEVAL_RESULTS_FILE, "w") as fout:
                for result in results:
                    json.dump(result, fout)
                    fout.write("\n")
        except Exception as e:
            logging.error(f"Error writing retrieval results: {str(e)}")
    
    return results


# 新增: 查询匹配函数
def query_matching(query, category):
    # 实现查询级匹配
    # 这里需要根据具体需求实现
    return []

# 新增: 全局匹配函数
def global_matching(query, historical_database):
    # 实现全局匹配
    # 这里需要根据具体需求实现
    return []

def history_match_single(data):
    matcher = QueryMatcher(None, HISTORY_MATCH_OUTPUT_FILE)
    
    topN = 8
    new_query = data.query['user']
    
    current_similarities = []
    current_top_queries = []
    current_results = []

    logging.info(f"Processing query: {new_query}")
    logging.info(f"History length: {len(data.history)}")

    for index, history_item in enumerate(data.history):
        historical_query = history_item["user"]
        similarity = matcher.calculate_similarity(new_query, historical_query)
        current_similarities.append(similarity)
        current_top_queries.append(historical_query)
        current_results.append("")  # 添加一个空字符串作为占位符

    logging.info(f"Number of historical queries processed: {len(current_top_queries)}")

    rag_result = ""  # 用于存储RAG结果
    try:
        with open(RETRIEVAL_RESULTS_FILE, 'r') as file:
            retrieval_data = [json.loads(line) for line in file]
            for item in retrieval_data:
                if item['query'] == new_query:
                    logging.info(f"Found query in retrieval data: {new_query}")
                    if item['integrated_results']:  # 修改: 使用integrated_results
                        rag_result = item['integrated_results'][0]['text']
                    else:
                        rag_result = "No results available"
                    break
    except Exception as e:
        logging.error(f"Error processing retrieval results: {str(e)}")

    sorted_data = sorted(zip(current_similarities, current_top_queries, current_results), reverse=True, key=lambda x: x[0])
    sorted_similarities, sorted_queries, sorted_results = zip(*sorted_data) if sorted_data else ([], [], [])

    logging.info(f"Current query: {new_query}")
    logging.info(f"Standard answer: {data.query['system']}")
    logging.info(f"Number of sorted queries: {len(sorted_queries)}")
    logging.info(f"Top {min(topN, len(sorted_queries))} queries with highest similarity:")
    for i in range(min(topN, len(sorted_queries))):
        logging.info(f"Query {i+1}: {sorted_queries[i]} (Similarity: {sorted_similarities[i]})")
    
    average_similarity = sum(sorted_similarities) / len(sorted_similarities) if sorted_similarities else 0
    logging.info(f"Average similarity: {average_similarity}")

    try:
        with open(HISTORY_MATCH_OUTPUT_FILE, "a") as f:
            if sorted_similarities and sorted_similarities[0] > 0.05:
                query_result = {
                    "query": new_query,
                    "standard_answer": data.query['system'],
                    "similarity": sorted_similarities[0] if sorted_similarities else 0,
                    "result": rag_result,
                    "top_queries": sorted_queries[:topN],
                    "top_similarities": sorted_similarities[:topN],
                    "average_similarity": average_similarity
                }
                json.dump(query_result, f)
                f.write("\n")
    except Exception as e:
        logging.error(f"Error writing history match output: {str(e)}")

def load_valid_data_with_history_and_query(input_file_path):
    try:
        logging.info(f"Attempting to load data from: {input_file_path}")
        
        if not os.path.exists(input_file_path):
            logging.error(f"Input file does not exist: {input_file_path}")
            return None

        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        paired_data = []
        qa_count = 0
        total_conversations = len(data)
        history_lengths = []

        for conversation_id, conversation_data in data.items():
            qa_found = False
            qa_incount = 0
            for index, message in enumerate(conversation_data['log']):
                if message['api_query'] == "[QA]":
                    query = message
                    standard_answer = message['system']
                    history = conversation_data['log'][:index] + conversation_data['log'][index+1:]
                    history_length = len(history)
                    history_lengths.append(history_length)
                    paired_data.append(ConversationData(history, query))
                    qa_count += 1
                    qa_found = True
                    qa_incount += 1
                    logging.info(f"Conversation {conversation_id}: History length = {history_length}")
                    
                if qa_incount == 2:
                    break  
                    
            
            if not qa_found:
                logging.info(f"Conversation {conversation_id} does not contain a [QA] item.")

        logging.info(f"Total conversations processed: {total_conversations}")
        logging.info(f"Total conversations with [QA] items: {qa_count}")
        logging.info(f"Percentage of conversations with [QA] items: {(qa_count/total_conversations)*100:.2f}%")
        
        if history_lengths:
            avg_history_length = sum(history_lengths) / len(history_lengths)
            max_history_length = max(history_lengths)
            min_history_length = min(history_lengths)
            logging.info(f"Average history length: {avg_history_length:.2f}")
            logging.info(f"Maximum history length: {max_history_length}")
            logging.info(f"Minimum history length: {min_history_length}")

        return paired_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            logging.error(f"File content (first 100 characters): {file_content[:100]}")
    except Exception as e:
        logging.error(f"Error loading valid data: {str(e)}")
    
    return None

# 修改initialize_output_files函数
def initialize_output_files():
    """初始化所有输出文件"""
    files_to_initialize = [
        HISTORY_MATCH_OUTPUT_FILE,
        GPT3_ANSWERS_FILE,
        CHAIN_OF_THOUGHT_RESULTS_FILE,
        INTEGRATED_RESULTS_FILE,
        RETRIEVAL_RESULTS_FILE,
        CLUSTERED_DATABASE_FILE,
        HISTORICAL_DATABASE_FILE
    ]
    for file_path in files_to_initialize:
        try:
            with open(file_path, "w") as f:
                f.write("")  # 创建一个空文件
            logging.info(f"Initialized output file: {file_path}")
        except Exception as e:
            logging.error(f"Error initializing output file {file_path}: {str(e)}")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--new_query", type=str, default="none")
    parser.add_argument("--data", type=str, default="none")

    return parser.parse_args()

# 修改: generate_answer 函数（继续）
def generate_answer(args, integrated_results, paired_data):
    logging.info("开始生成GPT-3.5-turbo答案...")
    
    gpt3_answers = []

    for item, data in zip(integrated_results, paired_data):
        query = item['query']
        context = item['integrated_results']
        standard_answer = data.query['system']  # 获取标准答案

        prompt = f"""
        Query: {query}

        Context: {context}

        Based on the above information, please provide a comprehensive answer to the query.
        """
        logging.info(f"Query '{query[:50]}...'的Prompt: {prompt}")

        try:
            time.sleep(1)  # 为了避免OpenAI API的速率限制，等待1秒
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                temperature=0.7,
            )
            gpt3_answer = response.choices[0].message['content'].strip()
            logging.info(f"为查询生成的GPT-3.5-turbo答案: {gpt3_answer[:50]}...")
            
        except openai.error.OpenAIError as e:
            logging.error(f"查询'{query[:50]}...'的OpenAI API错误: {str(e)}")
            gpt3_answer = f"由于API问题无法生成答案: {str(e)}"
        except Exception as e:
            logging.error(f"查询'{query[:50]}...'的GPT-3.5-turbo调用中出现意外错误: {str(e)}")
            gpt3_answer = f"由于意外问题无法生成答案: {str(e)}"

        gpt3_answers.append({
            "query": query,
            "prompt": prompt,
            "gpt3_answer": gpt3_answer,
            "context": context,
            "standard_answer": standard_answer
        })

        # 每处理一个项目就写入文件一次
        try:
            with open(GPT3_ANSWERS_FILE, 'a') as f:
                json.dump(gpt3_answers[-1], f)
                f.write('\n')
            logging.info(f"查询'{query[:50]}...'的答案已写入 {GPT3_ANSWERS_FILE}")
        except Exception as e:
            logging.error(f"将查询'{query[:50]}...'的答案写入文件时出错: {str(e)}")

    logging.info(f"所有GPT-3.5-turbo答案已处理完毕。总计: {len(gpt3_answers)}")
    return gpt3_answers

def perform_bleu_evaluation():
    # 自动下载'punkt'资源
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("正在下载NLTK的'punkt'资源...")
        nltk.download('punkt', quiet=True)
        logging.info("下载完成。")

    gpt3_answers = []
    standard_answers = []
    queries = []
    prompts = []
    
    # 读取gpt3_answers文件
    try:
        with open(GPT3_ANSWERS_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                gpt3_answers.append(data['gpt3_answer'])
                standard_answers.append(data['standard_answer'])  # 使用保存的标准答案
                queries.append(data['query'])
                prompts.append(data['prompt'])
    except Exception as e:
        logging.error(f"读取gpt3_answers文件时出错: {str(e)}")
        return 0.0

    # 对答案进行分词
    gpt3_answers_tokenized = [word_tokenize(answer.lower()) for answer in gpt3_answers]
    standard_answers_tokenized = [word_tokenize(answer.lower()) for answer in standard_answers]

    # 计算BLEU分数
    bleu_scores = []
    for gpt3_answer, standard_answer in zip(gpt3_answers_tokenized, standard_answers_tokenized):
        bleu_score = sentence_bleu([standard_answer], gpt3_answer)
        bleu_scores.append(bleu_score)

    # 计算平均BLEU分数
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # 记录结果
    logging.info(f"评估的答案对数量: {len(bleu_scores)}")
    logging.info(f"平均BLEU分数: {average_bleu:.4f}")

    # 将详细结果写入文件
    results_file = os.path.join(OUTPUT_DIR, "bleu_evaluation_results.jsonl")
    try:
        with open(results_file, 'w') as f:
            for i, (query, prompt, gpt3_answer, standard_answer, bleu_score) in enumerate(zip(queries, prompts, gpt3_answers, standard_answers, bleu_scores)):
                result = {
                    "id": i,
                    "query": query,
                    "prompt": prompt,
                    "gpt3_answer": gpt3_answer,
                    "standard_answer": standard_answer,
                    "bleu_score": bleu_score
                }
                json.dump(result, f)
                f.write('\n')
        logging.info(f"详细的BLEU评估结果已写入 {results_file}")
    except Exception as e:
        logging.error(f"写入BLEU评估结果时出错: {str(e)}")

    return average_bleu

# 修改: 动态更新历史数据库的函数
def update_historical_database(historical_database, new_query, answer, standard_answer):
    new_entry = (new_query, answer, time.time(), standard_answer)
    historical_database.append(new_entry)
    historical_database = filter_and_score(historical_database)
    
    with open(HISTORICAL_DATABASE_FILE, 'a') as f:
        json.dump({
            'query': new_query,
            'answer': answer,
            'timestamp': new_entry[2],
            'standard_answer': standard_answer
        }, f)
        f.write('\n')
    
    return historical_database

# 新增: 过滤和评分函数
def filter_and_score(historical_database):
    # 实现过滤和评分逻辑
    # 这里需要根据具体需求实现
    return historical_database
def update_historical_database(historical_database, new_query, answer, standard_answer):
    if not all([new_query, answer, standard_answer]):
        logging.warning("Missing data in update_historical_database. Skipping this update.")
        return historical_database

    new_entry = (new_query, answer, time.time(), standard_answer)
    historical_database.append(new_entry)
    historical_database = filter_and_score(historical_database)
    
    try:
        with open(HISTORICAL_DATABASE_FILE, 'a') as f:
            json.dump({
                'query': new_query,
                'answer': answer,
                'timestamp': new_entry[2],
                'standard_answer': standard_answer
            }, f)
            f.write('\n')
    except Exception as e:
        logging.error(f"Error writing to historical database file: {str(e)}")

    return historical_database
def time_decay(time_diff):
    """时间衰减函数"""
    return np.exp(-time_diff / (7 * 24 * 3600))  # 使用7天作为衰减周期

def calculate_relevance(query1, query2):
    """计算两个查询之间的相关性"""
    vec1 = model.encode(query1, convert_to_tensor=True)
    vec2 = model.encode(query2, convert_to_tensor=True)
    return float(torch.cosine_similarity(vec1, vec2, dim=0))

def cluster_database(database):
    """对数据库进行聚类"""
    queries = [entry['query'] for entry in database]
    vectors = model.encode(queries)
    
    n_clusters = min(10, len(queries))  # 假设我们想要最多10个聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)
    
    for entry, label in zip(database, cluster_labels):
        entry['cluster'] = int(label)
    
    return database

def save_to_file(database, filename):
    """将数据库保存到文件"""
    with open(filename, 'w') as f:
        for entry in database:
            json.dump(entry, f)
            f.write('\n')

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    BASE_DIR = args.output_dir if args.output_dir else "./topiocqa"
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    OUTPUT_DIR = BASE_DIR
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # Initialize distributed mode (if using SLURM cluster)
    src.slurm.init_distributed_mode(args)
    
    # Preprocess data
    query_context_mapping = preprocess_data(args.data, RETRIEVAL_DATABASE_PATH, QUERY_CONTEXT_MAPPING_PATH)
    
    # Get input file path
    input_file_path = args.data
    
    # Load valid data with history and query
    paired_data = load_valid_data_with_history_and_query(input_file_path)
    
    # Initialize output files
    initialize_output_files()
    
    # Initialize or load historical database
    historical_database = load_historical_database()
    
    if paired_data:
        # Initial dynamic update of historical database
        for data in paired_data:
            if 'system' in data.query:
                historical_database = update_historical_database(
                    historical_database, 
                    data.query['user'],
                    data.query['system'],
                    data.query['system']
                )
        
        # Cluster historical database
        clustered_database = cluster_historical_database(historical_database)

        # Perform retrieval and get integrated results
        integrated_results = retrieve(args, paired_data, historical_database)
        
        # Generate answers using GPT-3
        gpt3_answers = generate_answer(args, integrated_results, paired_data)
        
        # Update historical database with generated answers
        for data, answer in zip(paired_data, gpt3_answers):
            historical_database = update_historical_database(
                historical_database, 
                data.query['user'],
                answer['gpt3_answer'],
                data.query['system']
            )
        
        # Perform BLEU evaluation
        average_bleu = perform_bleu_evaluation()
        logging.info(f"Overall average BLEU score: {average_bleu:.4f}")
    else:
        logging.error("Failed to load valid data. Exiting program.")
        sys.exit(1)

    # Log program completion
    logging.info("Program execution completed.")