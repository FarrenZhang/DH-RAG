import argparse
import numpy as np
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from rank_bm25 import BM25Okapi
from performance_debugger import PerformanceDebugger
from performance_monitor import PerformanceMonitor
from performance_utils import timer_decorator
import torch
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class BM25EfficiencyTest:
    def __init__(self, args):
        self.args = args
        self.config = self._init_config()
        self.debugger = PerformanceDebugger(
            log_interval=self.config.MONITOR_INTERVAL,
            detailed_monitoring=self.config.DETAILED_MONITORING
        )
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # 设置输出目录
        self.output_dir = Path(args.output_dir) / "bm25_efficiency_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化BM25所需资源
        self._init_nltk()
        self.passages = []
        self.bm25 = None
        
        # 设置日志
        self._setup_logging()

    def _init_config(self):
        class Config:
            def __init__(self, args):
                self.OUTPUT_DIR = args.output_dir
                self.MONITOR_INTERVAL = 0.5
                self.ENABLE_GPU_MONITORING = True
                self.DETAILED_MONITORING = True
        return Config(self.args)

    def _setup_logging(self):
        log_file = self.output_dir / f"bm25_efficiency_test_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def _init_nltk(self):
        """初始化NLTK资源"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        """预处理文本"""
        # 分词
        tokens = word_tokenize(text.lower())
        # 移除停用词和标点符号
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens

    def load_passages(self):
        """加载文档集合"""
        try:
            with open(self.args.passages, 'r', encoding='utf-8') as f:
                passages_data = [json.loads(line) for line in f]
                
            # 预处理所有文档
            self.passages = []
            self.tokenized_passages = []
            
            for passage in passages_data:
                self.passages.append(passage)
                tokens = self.preprocess_text(passage.get('text', ''))
                self.tokenized_passages.append(tokens)
                
            # 初始化BM25
            self.bm25 = BM25Okapi(self.tokenized_passages)
            logging.info(f"Loaded and preprocessed {len(self.passages)} passages")
            
        except Exception as e:
            logging.error(f"Error loading passages: {str(e)}")
            raise

    def run_retrieval_test(self, query: str, n_docs: int = 5) -> List[Dict[str, Any]]:
        """执行BM25检索测试"""
        start_time = time.time()
        
        try:
            # 预处理查询
            tokenized_query = self.preprocess_text(query)
            
            # 获取文档得分
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # 获取排名最高的文档索引
            top_n_indices = np.argsort(doc_scores)[-n_docs:][::-1]
            
            # 获取检索结果
            results = []
            for idx in top_n_indices:
                results.append({
                    'text': self.passages[idx].get('text', ''),
                    'score': float(doc_scores[idx]),
                    'id': self.passages[idx].get('id', str(idx))
                })
            
        except Exception as e:
            logging.error(f"Retrieval error: {str(e)}")
            results = []
            
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # 记录性能指标
        self.performance_monitor.record_retrieval_time(retrieval_time)
        
        return results

    def run_efficiency_test(self, test_queries: List[str]):
        """运行完整的效率测试"""
        self.debugger.start_monitoring()
        self.performance_monitor.start_monitoring()
        
        try:
            # 加载文档集合
            with self.debugger.monitor_operation("load_passages"):
                self.load_passages()
            
            total_times = []
            batch_sizes = self.args.batch_sizes if hasattr(self.args, 'batch_sizes') else [1]
            
            for batch_size in batch_sizes:
                logging.info(f"\nTesting batch size: {batch_size}")
                
                # 将查询分成批次
                for i in range(0, len(test_queries), batch_size):
                    batch = test_queries[i:i + batch_size]
                    batch_start_time = time.time()
                    
                    for query in batch:
                        # 执行检索
                        with self.debugger.monitor_operation("retrieval"):
                            results = self.run_retrieval_test(query, self.args.n_docs)
                            
                        logging.info(f"Query: {query[:50]}...")
                        logging.info(f"Retrieved {len(results)} documents")
                    
                    batch_time = time.time() - batch_start_time
                    total_times.append(batch_time)
                    
                    logging.info(f"Batch processing time: {batch_time:.4f}s")
                    
            # 处理和保存性能指标
            self.performance_monitor.process_metrics()
            
        finally:
            self.debugger.stop_monitoring()
            self.performance_monitor.stop_monitoring()

def main():
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--passages", type=str, required=True,
                        help="Path to the passages file (.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--n_docs", type=int, default=5,
                        help="Number of documents to retrieve")
    
    # 效率测试特定参数
    parser.add_argument("--queries_json_file", type=str, required=True,
                        help="Path to the JSON file containing queries")
    parser.add_argument("--num_test_queries", type=int, default=216,
                        help="Number of queries to test")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--monitor_interval", type=float, default=0.5,
                        help="Interval for performance monitoring")
    parser.add_argument("--detailed_monitoring", action="store_true",
                        help="Enable detailed performance monitoring")
    
    args = parser.parse_args()
    
    # 读取测试查询
    with open(args.queries_json_file, 'r') as f:
        queries_data = json.load(f)
        test_queries = []
        for item in queries_data[:args.num_test_queries]:
            if isinstance(item, dict):
                query = item.get('instruction', '') or item.get('query', '') or item.get('text', '')
            else:
                query = str(item)
            test_queries.append(query)
    
    # 运行效率测试
    efficiency_test = BM25EfficiencyTest(args)
    efficiency_test.run_efficiency_test(test_queries)

if __name__ == "__main__":
    main()