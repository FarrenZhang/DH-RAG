# DH-RAG

DH-RAG是一个基于检索增强生成(RAG)的对话系统基线。

## 系统流程

```mermaid
graph TD
    A[开始] --> B[解析命令行参数]
    B --> C[初始化分布式模式]
    C --> D[加载数据]
    D --> E{数据加载成功?}
    E -->|是| F[初始化输出文件]
    E -->|否| Z[记录错误并退出]
    F --> G[初始化历史数据库]
    G --> N[对历史数据库进行聚类]
    N --> H[执行检索操作]
    H --> I[生成GPT-3答案]
    I --> J[更新历史数据库]
    J --> K[执行BLEU评估]
    K --> L[记录评估结果]
    L --> M[结束]

    subgraph 检索操作
        H --> H1[静态检索]
        H --> H2[动态检索]
        H2 --> H21[层次匹配]
        H2 --> H22[Chain of Thought搜索]
        H1 --> H3[整合结果]
        H21 --> H3
        H22 --> H3
    end

    subgraph 更新历史数据库
        J --> J1[添加新数据]
        J1 --> J2[过滤和评分]
    end

    subgraph BLEU评估
        K --> K1[读取GPT-3答案]
        K1 --> K2[计算BLEU分数]
        K2 --> K3[计算平均分数]
    end
```

## 运行环境

1. 启用docker环境:
   ```
   docker run --gpus all -it -v /home/feiyuan/glusterfs/DH-RAG:/home/dh-rag feiyuan_self_rag_0506 /bin/bash
   ```

2. 启用conda环境:
   ```
   cs
   ```

3. 进入执行目录:
   ```
   cd /home/dh-rag/scripts
   ```

## 运行指令

运行baseline脚本:

```
python run_mobilecs2_baseline_zfy_0811.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages ../data/mobilecs2_data/transformed_data.jsonl \
    --passages_embeddings "../data/mobilecs2_data/embedding_data/passages_00" \
    --data ../data/mobilecs2_data/valid_data_zfy/valid_conversations.json  \
    --output_dir ./output_data/mobilecs2_data/ \
    --n_docs 5
```

## 文件说明

- `INPUT_DATA_FILE`: 输入数据文件,包含原始查询和对话历史。
- `OUTPUT_DATA_FILE`: 输出数据文件,用于存储最终处理结果。
- `RETRIEVAL_RESULTS_FILE`: 存储检索系统的结果。
- `HISTORY_MATCH_OUTPUT_FILE`: 存储历史匹配的结果。
- `HISTORICAL_DATABASE_FILE`: 历史数据库文件,存储处理过的查询和答案。
- `CLUSTERED_DATABASE_FILE`: 存储聚类处理后的历史数据库。
- `CHAIN_OF_THOUGHT_RESULTS_FILE`: 存储"思维链"搜索的结果。
- `INTEGRATED_RESULTS_FILE`: 存储整合后的检索结果。
- `GPT3_ANSWERS_FILE`: 存储GPT-3.5-turbo生成的答案。

## 未来模块拆分计划

- `main.py`: 主脚本,程序入口点。
- `config.py`: 配置类,管理所有配置参数。
- `data_loader.py`: 数据加载和处理。
- `retriever.py`: 信息检索。
- `answer_generator.py`: GPT-3.5-turbo答案生成。
- `evaluator.py`: BLEU评分评估。
- `historical_database.py`: 历史数据管理和更新。
- `utils.py`: 通用工具函数。