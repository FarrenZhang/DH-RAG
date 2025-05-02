import os

class Config:
    def __init__(self, args):
        # 设置基础目录
        self.BASE_DIR = "./output"
        self.INPUT_DIR = os.path.join(self.BASE_DIR, "input")
        self.OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(self.BASE_DIR, "output")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        
        # 设置输入数据文件
        self.INPUT_DATA_FILE = args.data if args.data else os.path.join(self.INPUT_DIR, "test_input.json")
        
        # 设置其他输出文件路径
        self.OUTPUT_DATA_FILE = os.path.join(self.OUTPUT_DIR, "test_output.json")
        self.RETRIEVAL_RESULTS_FILE = os.path.join(self.OUTPUT_DIR, "retrieval_results.jsonl")
        self.HISTORY_MATCH_OUTPUT_FILE = os.path.join(self.OUTPUT_DIR, "history_match_output.jsonl")
        self.GPT3_ANSWERS_FILE = os.path.join(self.OUTPUT_DIR, "gpt3_answers.jsonl")
        
        # 设置API相关配置
        self.OPENAI_API_BASE = "https://api.holdai.top/v1"
        self.OPENAI_API_KEY = "sk-XCyY4uo7POh8OuMp97EeE08eFc2c43528076361f5f498eF1"
        
        # 从args中获取其他配置参数
        self.MODEL_NAME_OR_PATH = args.model_name_or_path
        self.PASSAGES = args.passages
        self.PASSAGES_EMBEDDINGS = args.passages_embeddings
        self.n_docs = args.n_docs
        
        # 可以继续添加其他从args中获取的参数
        # self.SOME_OTHER_PARAM = args.some_other_param
        
        # 确保必要的目录存在
        for directory in [self.INPUT_DIR, self.OUTPUT_DIR, self.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)
        
    def __str__(self):
        """返回配置的字符串表示，方便日志记录"""
        return f"""
        配置信息:
        - 输入数据文件: {self.INPUT_DATA_FILE}
        - 输出目录: {self.OUTPUT_DIR}
        - 模型路径: {self.MODEL_NAME_OR_PATH}
        - 段落文件: {self.PASSAGES}
        - 段落嵌入: {self.PASSAGES_EMBEDDINGS}
        - 文档数量: {self.n_docs}
        """

# 使用示例（在main函数中）:
# args = parse_arguments()
# config = Config(args)
# print(config)  # 这将打印出所有的配置信息