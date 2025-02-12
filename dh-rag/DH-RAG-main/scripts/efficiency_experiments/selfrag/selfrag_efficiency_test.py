import argparse
import numpy as np
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import backoff
import openai
from performance_debugger import PerformanceDebugger
from performance_monitor import PerformanceMonitor
from performance_utils import timer_decorator

class SelfRAGEfficiencyTest:
    def __init__(self, args):
        self.args = args
        self.config = self._init_config()
        self.debugger = PerformanceDebugger(
            log_interval=self.config.MONITOR_INTERVAL,
            detailed_monitoring=self.config.DETAILED_MONITORING
        )
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # 设置输出目录
        self.output_dir = Path(args.output_dir) / "efficiency_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        log_file = self.output_dir / f"efficiency_test_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def run_retrieval_test(self, query, n_docs=5):
        """测试检索性能"""
        start_time = time.time()
        
        try:
            # 使用SelfRAG的检索模块
            from passage_retrieval import Retriever
            retriever = Retriever(self.args)
            retriever.setup_retriever()
            
            results = retriever.search_document(query, n_docs)
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            self.performance_monitor.record_retrieval_time(retrieval_time)
            return results
        except Exception as e:
            logging.error(f"Error in retrieval: {str(e)}")
            return []

    def run_generation_test(self, prompt, model_name="facebook/opt-1.3b"):
        """测试生成性能"""
        start_time = time.time()
        
        try:
            # 初始化模型和tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 设置模型为评估模式
            model.eval()
            
            # 对输入进行编码
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logging.error(f"Error in generation: {str(e)}")
            response = f"Generation error: {str(e)}"
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        self.performance_monitor.record_generation_time(generation_time)
        return response

    def run_efficiency_test(self, test_queries):
        """运行完整的效率测试"""
        self.debugger.start_monitoring()
        self.performance_monitor.start_monitoring()
        
        try:
            total_times = []
            for query in test_queries:
                start_time = time.time()
                
                # 检索阶段
                with self.debugger.monitor_operation("retrieval"):
                    retrieved_docs = self.run_retrieval_test(query)
                
                # 生成阶段
                with self.debugger.monitor_operation("generation"):
                    prompt = self._create_prompt(query, retrieved_docs)
                    response = self.run_generation_test(prompt)
                
                end_time = time.time()
                total_time = end_time - start_time
                total_times.append(total_time)
                
                self.performance_monitor.record_total_time(total_time)
                
                logging.info(f"Query: {query}")
                logging.info(f"Total processing time: {total_time:.4f}s")
            
            # 处理和保存性能指标
            self.performance_monitor.process_metrics()
            
        finally:
            self.debugger.stop_monitoring()
            self.performance_monitor.stop_monitoring()

    def _create_prompt(self, query, retrieved_docs):
        """创建提示词"""
        prompt = f"Query: {query}\n\nContext: "
        for doc in retrieved_docs:
            prompt += f"\n[{doc.get('title', 'Document')}]: {doc.get('text', '')}"
        prompt += "\n\nPlease provide a comprehensive answer based on the above context."
        return prompt

def main():
    parser = argparse.ArgumentParser()
    
    # 添加与原始代码相同的参数
    parser.add_argument("--model_name_or_path", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--passages", type=str, required=True)
    parser.add_argument("--passages_embeddings", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_docs", type=int, default=5)
    
    # 效率测试特定参数
    parser.add_argument("--queries_json_file", type=str, required=True, 
                        help="Path to the JSON file containing queries")
    parser.add_argument("--num_test_queries", type=int, default=10, 
                        help="Number of queries to test")
    
    args = parser.parse_args()
    
    # 读取JSON文件中的查询
    with open(args.queries_json_file, 'r') as f:
        data = json.load(f)
        test_queries = []
        for item in data[:args.num_test_queries]:  # 限制测试查询数量
            if 'instruction' in item:
                test_queries.append(item['instruction'])
    
    # 运行效率测试
    efficiency_test = SelfRAGEfficiencyTest(args)
    efficiency_test.run_efficiency_test(test_queries)

if __name__ == "__main__":
    main()