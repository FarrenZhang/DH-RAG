import argparse
import logging
import os
from datetime import datetime
import time
import sys
from typing import List, Dict

from scripts.config.config import Config
from data_loader import DataLoader
from retriever import Retriever
from answer_generator import AnswerGenerator
from evaluator import Evaluator
from historical_database import HistoricalDatabase

def setup_logging(config):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'app_{current_time}.log'
    log_file = os.path.join(config.LOG_DIR, log_file_name)
    
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Program started. Log file: {log_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG system for question answering")
    
    # Add all the arguments
    parser.add_argument("--model_name_or_path", type=str, help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--passages", type=str, help="Path to the passages file")
    parser.add_argument("--passages_embeddings", type=str, help="Path to the passages embeddings")
    parser.add_argument("--data", type=str, help="Path to the input data file")
    parser.add_argument("--output_dir", type=str, help="Directory where the output will be saved")
    parser.add_argument("--new_query", type=str, help="New query to process")
    parser.add_argument("--n_docs", type=int, default=5, help="Number of documents to retrieve")
    
    # Add other arguments that were in your original script
    parser.add_argument("--query", type=str, default=None, help=".json file containing question and answers, similar format to reader data")
    parser.add_argument("--validation_workers", type=int, default=32, help="Number of parallel processes to validate results")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--no_fp16", action="store_true", help="Inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+", help="Language(s) of the data")
    parser.add_argument("--dataset", type=str, default="none", help="Dataset name")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Normalize text")

    return parser.parse_args()

def save_results(config: Config, integrated_results: List[Dict], gpt3_answers: List[Dict], average_bleu: float):
    """
    保存程序的执行结果
    
    :param config: 配置对象
    :param integrated_results: 集成的检索结果
    :param gpt3_answers: GPT-3生成的答案
    :param average_bleu: 平均BLEU分数
    """
    import json
    import os

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 保存集成结果
    with open(os.path.join(config.OUTPUT_DIR, 'integrated_results.json'), 'w', encoding='utf-8') as f:
        json.dump(integrated_results, f, ensure_ascii=False, indent=2)

    # 保存GPT-3答案
    with open(os.path.join(config.OUTPUT_DIR, 'gpt3_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(gpt3_answers, f, ensure_ascii=False, indent=2)

    # 保存评估结果
    with open(os.path.join(config.OUTPUT_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'average_bleu': average_bleu}, f, ensure_ascii=False, indent=2)

    logging.info(f"结果已保存到目录: {config.OUTPUT_DIR}")

def main():
    # 初始化配置
    args = parse_arguments()
    config = Config(args)
    logging.info(f"使用的配置:\n{config}")
    print(f"使用的配置:\n{config}")
    # 设置日志
    setup_logging(config)
    
    # 记录程序开始时间
    start_time = time.time()

    try:
        # 初始化数据加载器并加载数据
        data_loader = DataLoader(config)
        paired_data = data_loader.load_data()
        
        if not paired_data:
            raise ValueError("无法加载有效数据。")

        # 初始化检索器和历史数据库
        retriever = Retriever(config)
        historical_database = HistoricalDatabase()
        
        # 执行检索
        integrated_results = retriever.retrieve(paired_data, historical_database)
        
        # 初始化答案生成器并生成答案
        answer_generator = AnswerGenerator(config)
        gpt3_answers = answer_generator.generate_answers(integrated_results)
        
        # 更新历史数据库
        for data, answer in zip(paired_data, gpt3_answers):
            historical_database.update(data.query['user'], answer['gpt3_answer'])
        
        # 执行评估
        evaluator = Evaluator(config)
        average_bleu = evaluator.perform_bleu_evaluation()
        logging.info(f"总体平均BLEU分数: {average_bleu:.4f}")

        # 保存结果
        save_results(config, integrated_results, gpt3_answers, average_bleu)

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {str(e)}")
        sys.exit(1)

    finally:
        # 计算并记录总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"程序总运行时间: {total_time:.2f} 秒")

    logging.info("程序正常结束。")

if __name__ == "__main__":
    main()