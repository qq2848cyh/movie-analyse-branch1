"""项目配置文件 — 集中管理所有数据路径和配置参数"""

import os

# ==================== 基础路径配置 ====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== 数据库路径 ====================

BIGDATA_DB_PATH = os.path.join(DATA_DIR, "bigdata_movies.db")
NEW_MOVIES_DB_PATH = os.path.join(DATA_DIR, "new_movies.db")
NEW_MOVIES_CSV = os.path.join(DATA_DIR, "new_data", "douban_all_movies.csv")

# ==================== 缓存子目录路径 ====================

CACHE_RECOMMEND_DIR = os.path.join(CACHE_DIR, "recommend")
CACHE_SENTIMENT_DIR = os.path.join(CACHE_DIR, "sentiment")
CACHE_NETWORK_DIR = os.path.join(CACHE_DIR, "network")
CACHE_CHARTS_DIR = os.path.join(CACHE_DIR, "charts")

# 确保缓存子目录存在
for _dir in [CACHE_RECOMMEND_DIR, CACHE_SENTIMENT_DIR, CACHE_NETWORK_DIR, CACHE_CHARTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ==================== 模型缓存路径 ====================

# TF-IDF 推荐模型
CACHE_TFIDF = os.path.join(CACHE_RECOMMEND_DIR, "tfidf_model.pkl")

# SVD 协同过滤模型
CACHE_SVD = os.path.join(CACHE_RECOMMEND_DIR, "svd_model.pkl")
CACHE_SVD_NPZ = os.path.join(CACHE_RECOMMEND_DIR, "svd_item_vectors.npz")

# BGE-Large-Zh 深度语义模型
CACHE_SBERT = os.path.join(CACHE_RECOMMEND_DIR, "sbert_model.pkl")
CACHE_SBERT_EMB = os.path.join(CACHE_RECOMMEND_DIR, "sbert_embeddings.npy")

# 推荐评估报告
EVAL_REPORT_PATH = os.path.join(CACHE_RECOMMEND_DIR, "recommend_eval.json")

# ==================== 情感分析缓存路径 ====================

# Word2Vec 词向量
WORD2VEC_PATH = os.path.join(CACHE_SENTIMENT_DIR, "word2vec_200d.bin")
WORD2IDX_PATH = os.path.join(CACHE_SENTIMENT_DIR, "word2idx.json")
EMBED_MATRIX_PATH = os.path.join(CACHE_SENTIMENT_DIR, "embedding_matrix.npy")
PUNCT_INDICES_PATH = os.path.join(CACHE_SENTIMENT_DIR, "punct_indices.json")

# BiLSTM-Attention 模型
DL_MODEL_BINARY_PATH = os.path.join(CACHE_SENTIMENT_DIR, "dl_model_binary.pt")
DL_MODEL_FIVE_PATH = os.path.join(CACHE_SENTIMENT_DIR, "dl_model_five.pt")

# 训练曲线
SENTIMENT_CURVES_DIR = os.path.join(CACHE_SENTIMENT_DIR, "curves")
os.makedirs(SENTIMENT_CURVES_DIR, exist_ok=True)

# ==================== 网络分析缓存路径 ====================

NETWORK_STATS_PATH = os.path.join(CACHE_NETWORK_DIR, "network_stats.pkl")

# ==================== 图表缓存路径 ====================

CHARTS_NEW_CACHE = os.path.join(CACHE_CHARTS_DIR, "new_charts.pkl")
CHARTS_BIGDATA_CACHE = os.path.join(CACHE_CHARTS_DIR, "bigdata_charts.pkl")

# ==================== 模型参数配置 ====================

# Word2Vec 参数
VOCAB_SIZE = 15000
EMBED_DIM = 200
HIDDEN_DIM = 128
MAX_SEQ_LEN = 100
BATCH_SIZE = 64

# 推荐系统参数
RECOMMEND_EVAL_SAMPLES = 200
RECOMMEND_MIN_VOTES = 10

# 情感分析参数
SENTIMENT_TRAIN_SAMPLES = 15000
SENTIMENT_CV_FOLDS = 5

# ==================== 数据库相关 ====================

# 大数据集 CSV 目录
BIGDATA_CSV_DIR = os.path.join(DATA_DIR, "bigdata")

# 全文索引表名
FTS_TABLE_SUFFIX = "_fts"
