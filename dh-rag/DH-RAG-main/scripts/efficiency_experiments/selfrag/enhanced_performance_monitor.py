import argparse
import logging
import os
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from enhanced_performance_monitor import EnhancedPerformanceMonitor

class SelfRAGEfficiencyTest:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.performance_monitor = EnhancedPerformanceMonitor(
            config=self.config,
            output_dir=self.output_dir
        )
        
    def setup_environment(self):
        """设置实验环境"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            filename=self.output_dir / "experiment.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 缓存设置
        self.retrieval_cache = {}
        
    def load_model(self):
        """加载模型，支持量化"""
        try:
            if self.args.quantization:
                # 8-bit量化
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            raise
            
    def run_retrieval_test(self, query_batch, n_docs=5):
        """批量检索测试"""
        start_time = time.time()
        
        results = []
        cache_hits = 0
        
        try:
            for query in query_batch:
                # 检查缓存
                if query in self.retrieval_cache:
                    results.append(self.retrieval_cache[query])
                    cache_hits += 1
                    continue
                    
                # 执行检索
                retrieved_docs = self.retriever.search_document(query, n_docs)
                results.append(retrieved_docs)
                
                # 更新缓存
                self.retrieval_cache[query] = retrieved_docs
                
        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            return []
            
        end_time = time.time()
        latency = end_time - start_time
        
        # 记录性能指标
        self.performance_monitor.record_retrieval(
            latency=latency,
            batch_size=len(query_batch)
        )
        
        logging.info(f"Retrieval batch size: {len(query_batch)}, "
                    f"latency: {latency:.3f}s, "
                    f"cache hits: {cache_hits}")
                    
        return results
        
    def run_generation_test(self, prompts, max_new_tokens=100):
        """批量生成测试"""
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.args.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            responses = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return []
            
        end_time = time.time()
        latency = end_time - start_time
        
        # 记录性能指标
        self.performance_monitor.record_generation(
            latency=latency,
            batch_size=len(prompts)
        )
        
        logging.info(f"Generation batch size: {len(prompts)}, "
                    f"latency: {latency:.3f}s")
                    
        return responses

    def run_experiment(self, test_queries, batch_sizes=[1, 4, 8, 16]):
        """运行完整实验"""
        self.performance_monitor.start_monitoring()
        
        try:
            # 测试不同batch size
            for batch_size in batch_sizes:
                logging.info(f"Testing batch size: {batch_size}")
                
                # 将查询分成批次
                for i in range(0, len(test_queries), batch_size):
                    batch = test_queries[i:i + batch_size]
                    
                    # 检索阶段
                    retrieved_docs = self.run_retrieval_test(
                        batch,
                        n_docs=self.args.n_docs
                    )
                    
                    # 生成阶段
                    prompts = self.create_prompts(batch, retrieved_docs)
                    responses = self.run_generation_test(
                        prompts,
                        max_new_tokens=self.args.max_new_tokens
                    )
                    
            # 保存结果
            metrics = self.performance_monitor.save_results()
            
            logging.info("Experiment completed successfully")
            return metrics
            
        except Exception as e:
            logging.error(f"Experiment failed: {e}")
            raise
        finally:
            # 清理资源
            self.cleanup()

    def create_prompts(self, queries, retrieved_docs):
        """创建提示词"""
        prompts = []
        for query, docs in zip(queries, retrieved_docs):
            prompt = f"Query: {query}\n\nContext: "
            # 添加检索文档作为上下文
            for doc in docs:
                prompt += f"\n[Document]: {doc.get('text', '')}"
            prompt += "\n\nBased on the above context, please provide a comprehensive answer."
            prompts.append(prompt)
        return prompts

    def cleanup(self):
        """清理资源"""
        # 清除缓存
        self.retrieval_cache.clear()
        torch.cuda.empty_cache()

    def save_experiment_config(self):
        """保存实验配置"""
        config = {
            "model_name": self.args.model_name,
            "n_docs": self.args.n_docs,
            "max_new_tokens": self.args.max_new_tokens,
            "batch_sizes": self.args.batch_sizes,
            "quantization": self.args.quantization,
            "cache_enabled": self.args.use_cache,
            "device": str(self.device),
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        }
        
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
def main():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--passages", type=str, required=True)
    parser.add_argument("--passages_embeddings", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_docs", type=int, default=5)

    # 效率测试特定参数
    parser.add_argument("--queries_json_file", type=str, required=True, 
                        help="Path to the JSON file containing queries")
    parser.add_argument("--num_test_queries", type=int, default=216, 
                        help="Number of queries to test")
    # 新增参数
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--monitor_interval", type=float, default=0.5,
                        help="Interval for performance monitoring")
    parser.add_argument("--detailed_monitoring", action="store_true",
                        help="Enable detailed performance monitoring")

    # 实验参数
    parser.add_argument("--n_docs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--num_test_queries", type=int, default=100)
    
    # 优化参数
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    
    # 读取测试查询
    with open(args.queries_file, "r") as f:
        queries = json.load(f)
        test_queries = queries[:args.num_test_queries]
    
    # 运行实验
    experiment = SelfRAGEfficiencyTest(args)
    experiment.load_model()
    experiment.save_experiment_config()
    
    metrics = experiment.run_experiment(test_queries)
    
    # 输出关键指标
    print("\nExperiment Results:")
    print(f"Average Retrieval Latency: {metrics['retrieval_latency']['mean']:.3f}s")
    print(f"Average Generation Latency: {metrics['generation_latency']['mean']:.3f}s")
    print(f"Overall Throughput: {metrics['throughput']:.2f} queries/second")

if __name__ == "__main__":
    main()