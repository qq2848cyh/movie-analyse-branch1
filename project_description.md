# 电影数据分析平台 — 详细项目说明

---

## 一、项目整体架构

```
movie-analyse-branch1/
├── Project/                        # Flask 后端核心
│   ├── app.py                      # 应用入口
│   ├── config.py                   # 全局配置中心
│   ├── routes/                     # 路由控制层 (Blueprint)
│   │   ├── __init__.py
│   │   ├── main.py                 # 首页 / 导航
│   │   ├── movies.py               # 精选电影(2000+)模块
│   │   ├── bigdata.py              # 大数据集(2.7万+)模块
│   │   └── analysis.py             # 智能分析(情感/推荐/网络)模块
│   ├── utils/                      # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── base_manager.py         # 数据库管理基类
│   │   ├── db_manager.py           # 大数据集SQLite管理(SQL聚合+FTS5全文索引)
│   │   ├── new_movie_manager.py    # 精选电影SQLite管理(2000+部)
│   │   ├── sentiment_analyzer.py   # 传统机器学习情感分析
│   │   ├── sentiment_dl.py         # 深度学习情感分析(BiLSTM-Attention)
│   │   ├── bigdata_recommend_engine.py  # 三模型推荐引擎
│   │   ├── network_analyzer.py     # 协作网络分析
│   │   ├── echarts_visualizer.py   # ECharts图表数据生成
│   │   ├── cache_manager.py        # 统一缓存管理
│   │   └── stopwords.py            # 中文停用词表
│   └── templates/                  # Jinja2 前端模板 (14个页面)
│       ├── welcome.html            # 欢迎页
│       ├── index.html / index_bigdata.html          # 系统首页
│       ├── movie.html / movie_bigdata.html          # 电影列表页
│       ├── score.html / score_bigdata.html          # 评分分析页
│       ├── cloud.html / cloud_bigdata.html          # 词云展示页
│       ├── analysis_network.html                    # 网络分析页
│       ├── analysis_sentiment.html                  # 情感分析页
│       ├── analysis_recommend.html                  # 推荐系统页
│       └── aboutMe.html                             # 关于页面
├── static/                         # 前端静态资源 (Bootstrap4 + jQuery + ECharts)
│   └── assets/
│       ├── css/style.css
│       ├── js/main.js, echarts.min.js
│       └── vendor/ (Bootstrap, jQuery, AOS, boxicons, venobox, isotope, waypoints, counterup 等)
├── data/                           # ⚠️ 数据目录 (被.gitignore排除)
│   ├── new_data/
│   │   └── douban_all_movies.csv   # 精选2000+电影CSV
│   ├── bigdata/                    # 大数据集CSV目录
│   │   └── *.csv                   # 豆瓣评论/电影原始CSV文件
│   ├── bigdata_movies.db           # 大数据集SQLite数据库
│   ├── new_movies.db               # 精选电影SQLite数据库
│   ├── cache/                      # 模型/图表缓存
│   │   ├── recommend/              # 推荐模型缓存 (TF-IDF/SVD/BGE-Large)
│   │   ├── sentiment/              # 情感分析缓存 (Word2Vec/BiLSTM/训练曲线)
│   │   ├── network/                # 网络分析缓存
│   │   └── charts/                 # 图表数据缓存
│   └── hf_cache/                   # HuggingFace模型离线缓存
├── thesis_requirement/             # 论文要求文档
│   ├── 本科毕业设计（论文）要求与撰写规范（理工类）-本部潇湘.doc
│   └── exp.txt                     # 摘要撰写要求说明
├── backup/                         # ⚠️ 优化前代码备份 (被.gitignore排除)
├── requirements.txt                # Python依赖
└── .gitignore
```

### 技术栈总览

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask (Blueprint 路由拆分) |
| 数据库 | SQLite (WAL 模式 + FTS5 全文索引) |
| 中文分词 | jieba |
| 传统机器学习 | scikit-learn (TF-IDF, NaiveBayes, LogisticRegression, TruncatedSVD) |
| 梯度提升 | LightGBM |
| 深度学习 | PyTorch — **BiLSTM-Attention**（情感分析）+ **BGE-Large-Zh v1.5**（推荐系统，326M参数，1024维语义嵌入） |
| 词向量 | Gensim Word2Vec, BGE-Large-Zh (1024维) |
| 图分析 | NetworkX + python-louvain (Louvain 社区发现) |
| 前端可视化 | ECharts, Bootstrap 4, jQuery, AOS 动画 |
| 缓存方案 | pickle 序列化 + 磁盘持久化 |
| GPU 加速 | **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)**，深度学习模型均启用 GPU 加速，BGE-Large-Zh 额外启用 FP16 半精度 |

---

## 二、数据层设计

### 2.1 数据源

项目共有**两个独立的数据库**，通过各自的 Manager 管理：

| 数据库 | Manager | 规模 | 来源 |
|--------|---------|------|------|
| `bigdata_movies.db` | `DBManager` | 约 2.7 万部电影 + 海量评论 | `data/bigdata/*.csv` |
| `new_movies.db` | `NewMovieManager` | 约 2000+ 部精选电影 | `data/new_data/douban_all_movies.csv` |

### 2.2 DBManager 设计要点

- **配置驱动建表**：通过 `schema_mapping.json` 定义 CSV 列到数据库列的映射关系
- **FTS5 全文索引**：对电影名称和剧情描述建立 FTS5 全文索引，支持中文模糊搜索
- **分块导入**：使用 `pandas.read_csv(chunksize=50000)` 分块读取，避免内存溢出
- **关联表**：自动从 CSV 中解析「电影-演员」、「电影-导演」多对多关系，建独立的 `movie_actors`、`movie_directors` 关联表
- **词云缓存**：预计算词云数据并存入 `wordcloud_cache` 表，避免重复分词计算

### 2.3 缓存策略

整个系统采用**懒加载 + pickle 持久化**的缓存策略：

- **模型训练后自动缓存**：所有模型训练完成后将结果 pickle 写入磁盘
- **首次请求触发预热**：后台线程（`threading.Thread(daemon=True)`）在首次 API 请求时启动
- **缓存命中直接返回**：后续请求直接从磁盘加载，无需重新训练
- **强制刷新**：API 支持 `?refresh=1` 参数强制重新计算

---

## 三、核心功能模块详解

---

### 3.1 情感分析系统

#### 3.1.1 传统机器学习方案 (`sentiment_analyzer.py`)

**理论框架：**

采用**监督学习**的方式进行文本情感分类，构建了完整的 Pipeline：

```
原始评论 → jieba分词 → 去停用词 → TF-IDF向量化 → 特征选择(χ²) → 分类器 → 情感标签
```

**关键技术细节：**

| 环节 | 实现 |
|------|------|
| **分词** | jieba 精确模式分词，保留长度 ≥ 2 的纯中文词 |
| **停用词** | 自定义中文停用词表 (`stopwords.py`) |
| **向量化** | `TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)` — 权重 = 1+log(tf)，使用双字短语捕捉搭配信息 |
| **特征选择** | **χ² (Chi-Square) 检验** — 计算每个词与情感标签的相关性，保留 Top 2000 最有区分力的特征词 |
| **训练集** | 15000 条随机抽样评论，80%/20% 分层划分（`stratify=y` 保证正负比例一致） |
| **交叉验证** | `StratifiedKFold(n_splits=5)` 五折分层交叉验证 |

**使用的三种分类器：**

| 分类器 | 理论基础 | 参数 |
|--------|----------|------|
| **MultinomialNB** | 朴素贝叶斯 — 基于特征条件独立假设，使用多项式分布建模 TF-IDF 特征 | `alpha=0.5` (拉普拉斯平滑) |
| **LogisticRegression** | 逻辑回归 — 线性分类边界 + Sigmoid 概率映射，适合高维稀疏文本特征 | `C=1.0, max_iter=1000` |
| **LightGBM** | 梯度提升决策树 — 基于 Histogram 的 GBDT 实现，Leaf-wise 生长策略，训练速度极快 | `n_estimators=800, max_depth=10, lr=0.03` |

**评估指标：** Accuracy、Precision、Recall、F1-Score，并在五折 CV 上报告均值 ± 标准差。

**关键词提取原理：**

通过计算正/负类样本 TF-IDF 均值向量的差值（`pos_vec - neg_vec`），差值最大的词即为**最具正向区分力的关键词**（如"精彩""感人"），差值最小的词为**最具负向区分力的关键词**（如"无聊""烂片"）。同时对每个 1-5 分评分级别也做同样的分析，构建评分类别关键词表。

**时间趋势分析：**

使用 **Mann-Whitney U 检验**（非参数检验，不假设数据正态分布）判断 2012 年前后评分是否存在显著性变化（`p < 0.05` 为显著）。

**超参调优：**

通过 `GridSearchCV` 对 LightGBM 的 `learning_rate` 在 `[0.02, 0.03, 0.05]` 范围内进行 2 折网格搜索。

---

#### 3.1.2 深度学习方案 (`sentiment_dl.py`)

**理论框架 — BiLSTM + Self-Attention：**

```
原始评论 → jieba分词 → 词表映射(WORD2IDX) → Padding/Truncate →
Embedding Layer (Word2Vec预训练权重) → BiLSTM →
Self-Attention加权池化 + Mean Pooling残差连接 → LayerNorm →
Dropout → FC(64) + ReLU → FC(num_classes)
```

**模型架构详解：**

| 层级 | 配置 | 理论基础 |
|------|------|----------|
| **词表构建** | 从 10 万条随机评论中统计词频，取 Top 14998 词 + `<PAD>` + `<UNK>` = 15000 词 | 高频词覆盖，低频词归入 `<UNK>` |
| **Word2Vec 预训练** | Gensim Skip-gram (sg=1), vector_size=200, window=5, min_count=3 | **Skip-gram** 利用中心词预测上下文，对低频词效果更好，适合中文分词后的稀疏词表 |
| **Embedding** | 15000 × 200，用 Word2Vec 权重初始化，训练中允许微调 | **迁移学习** — 将大规模无监督预训练知识注入下游分类任务 |
| **BiLSTM** | 1 层双向 LSTM，hidden_dim=128 → 拼接后维度 256 | **双向建模** — 同时捕捉上下文的前向和后向语义依赖关系 |
| **Self-Attention** | `Linear(256→64) → Linear(64→1)` 计算每个词的注意力分数 | 让模型自动学习对**情感判断最重要的词**，抑制无关词和标点符号 |
| **标点遮蔽** | 在 Attention 计算前，将标点位置的 score 设为 `-1e9` | 避免标点符号干扰注意力权重分配 |
| **残差连接** | `LayerNorm(Attention_Pool + Mean_Pool)` | 借鉴 Transformer 的残差设计，融合注意力池化和均值池化，防止梯度消失 |
| **正则化** | Dropout=0.5, Adam weight_decay=1e-4 | 双重正则化防止过拟合 |
| **学习率调度** | `ReduceLROnPlateau(mode='min', factor=0.5, patience=3)` | 验证 F1 不提升时自动减半学习率 |
| **Early Stopping** | patience=5，监控验证集 F1-Score | 防止过拟合，自动选取验证集最优模型权重 |

**训练配置：**

| 参数 | 值 |
|------|-----|
| 最大序列长度 (MAX_SEQ_LEN) | 100 |
| Batch Size | 64 |
| 优化器 | Adam (lr=0.001, weight_decay=1e-4) |
| 最大 Epoch | 30（实际由 Early Stopping 提前终止） |
| 训练样本 | 15000 条，80/20 分层划分 |
| GPU 加速 | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |

**两个分类任务：**

- **二分类：** 情感正/负（rating ≥ 4 为正类）
- **五分类：** 直接预测 1-5 星评分

**Attention 可视化 (`get_attention_visualization`)：**

加载训练好的二分类模型，对输入评论的每个词计算 Attention 权重（Softmax 归一化），前端可用热力图直观展示哪些词对情感判断最关键。例如输入「这部电影太精彩了」，模型可能对「精彩」分配最高的注意力权重。

**训练曲线记录：**

每个 Epoch 记录验证集 F1/Accuracy，保存为 JSON 格式（`dl_2cls_history.json`、`dl_5cls_history.json`），前端可绘制 Loss/F1 收敛曲线。

---

### 3.2 推荐系统 (`bigdata_recommend_engine.py`)

**设计理念：三模型对比实验架构**，使用统一的评估框架进行公平对比。

#### 3.2.1 统一数据融合

```
new_movies.db (2000+部)  +  bigdata_movies.db (2.7万部)
                    ↓
        _load_unified_movies() 合并去重
                    ↓
        过滤 votes ≥ RECOMMEND_MIN_VOTES(=10)
                    ↓
             统一电影库 (约2.7万+部)
```

通过 `movie_id` 去重，`new_movies.db` 的数据优先保留（`source: "new_movies"`），确保精选数据不丢失。

---

#### 3.2.2 模型 A：TF-IDF 基于内容的推荐（Baseline）

**理论基础：TF-IDF + 余弦相似度**

TF-IDF 衡量词语在文档集合中的重要性：
- TF（词频）= 词在当前文档中的出现频率
- IDF（逆文档频率）= log(总文档数 / 包含该词的文档数)
- TF-IDF = TF × IDF，高权重词 = 在当前文档频繁出现 + 在整体语料中罕见

将每部电影表示为 TF-IDF 向量，通过**余弦距离**找最近邻电影作为推荐结果。

**特征工程（文本拼接策略）：**

```python
doc = genres×2 + directors×2 + actors×2 + tags×2 + summary(截断300字) + languages
```

对 genres/directors/actors/tags 重复两次以**加大结构化特征的权重**，确保类型和主创信息在推荐中占主导地位。剧情摘要截断至 300 字防止长文本稀释结构化信息。

**模型参数：**

| 参数 | 值 | 理论依据 |
|------|-----|----------|
| 最大特征数 (max_features) | 8000 | 在覆盖率和计算效率间取得平衡 |
| N-gram | (1, 2) | 同时捕捉单字和双字短语如「科幻」「动作片」 |
| min_df | 3 | 过滤仅出现 1-2 次的噪声词 |
| sublinear_tf | True | 使用 1+log(tf) 抑制高频词权重膨胀 |

**搜索：** `NearestNeighbors(n_neighbors=50, metric='cosine')`

---

#### 3.2.3 模型 B：TruncatedSVD 协同过滤推荐

**理论基础：矩阵分解 (Matrix Factorization)**

协同过滤的核心思想：利用「用户-电影」的交互矩阵发现潜在隐含因子（Latent Factors），将高维稀疏的用户评分矩阵分解为低维稠密的隐向量空间。

**Pipeline：**

```
用户评分矩阵 (稀疏CSR) → TruncatedSVD(n_components=128, n_iter=10)
                                    ↓
                         电影隐向量矩阵 (Item Embeddings, n_items × 128)
                                    ↓
                          NearestNeighbors(KNN, metric='cosine')
                                    ↓
                          基于电影隐向量的协同过滤推荐
```

**数据预处理（关键步骤）：**

| 步骤 | 操作 | 目的 |
|------|------|------|
| **用户过滤** | 只保留评分次数 ≥ 5 的用户 | 过滤不活跃用户，减少评分矩阵噪声 |
| **电影过滤** | 只保留被评分次数 ≥ 10 的电影 | 确保每部电影有足够多的协同交互信号 |
| **稀疏矩阵** | `scipy.sparse.csr_matrix` (Compressed Sparse Row) | 高效存储大规模稀疏评分矩阵，仅存储非零元素 |

**与经典协同过滤的对比：**

| 方案 | 区别 |
|------|------|
| **经典 SVD** | 需要先填充缺失值（通常用均值），计算量随矩阵规模平方增长 |
| **TruncatedSVD（本方案）** | 直接在稀疏矩阵上做截断 SVD 分解（隐语义模型 LSA），不填充缺失值，利用稀疏矩阵运算大幅降低计算量 |
| **FunkSVD / SVD++** | 使用 SGD 随机梯度下降优化，显式建模用户/物品偏置项 |
| **NCF (Neural CF)** | 用神经网络替代矩阵分解的线性内积交互 |

**参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| n_components | 128 | 隐向量维度，在表达能力与计算效率间平衡 |
| n_iter | 10 | Arnoldi 迭代次数，足够收敛 |
| random_state | 42 | 固定随机种子保证结果可复现 |

**解释方差 (Explained Variance Ratio)：** 衡量 128 个潜在因子总共解释了原始评分矩阵多少比例的信息量。

**局限性说明：**

- SVD 只能对训练集（有交互数据）中存在的电影进行推荐
- 如果查询的电影不在 SVD 训练集中，返回「该电影不在 SVD 训练集中（互动不足）」提示
- 这是**纯协同过滤**的固有限制，不利用电影内容信息

---

#### 3.2.4 模型 C：BGE-Large-Zh 深度语义推荐

**理论基础：Sentence-BERT 语义嵌入 + 余弦最近邻检索**

BGE-Large-Zh（**B**AAI **G**eneral **E**mbedding，北京智源人工智能研究院发布）是一个基于 BERT 架构的中文句子嵌入模型，专为检索和语义相似度任务优化。

**模型特性：**

| 属性 | 值 |
|------|-----|
| 模型名 | `BAAI/bge-large-zh-v1.5` |
| 基础架构 | BERT-Large (Transformer Encoder) |
| 参数量 | 326M (3.26 亿) |
| 嵌入维度 | 1024 |
| 精度模式 | **FP16 半精度**（显存需求从 ~1.3GB 降至 ~650MB） |
| 最大序列长度 | 512 tokens |
| 归一化 | L2 归一化 (余弦相似度等价于向量内积) |
| GPU 加速 | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |

**文本构造策略（将结构化电影信息转化为自然语言描述）：**

```
电影名称。类型：{genres}。导演：{directors}。评分：{rating}分，{votes}人评价。剧情：{summary}。标签：{tags}
```

将所有电影的结构化文本通过 BGE-Large-Zh 编码为 **1024 维 L2 归一化向量**。归一化后向量之间的余弦相似度等价于内积，大幅加速检索。

**批量编码配置：**

| 模式 | Batch Size | 说明 |
|------|-----------|------|
| GPU + FP16 | 256 | 半精度 + 大 batch 吞吐最高 |
| CPU / FP32 | 128 | 未启用 GPU 时的回退方案 |

**检索：** `NearestNeighbors(n_neighbors=50, metric='cosine')`

**为什么 BGE-Large-Zh 优于 TF-IDF：**

| 维度 | TF-IDF | BGE-Large-Zh |
|------|--------|--------------|
| 语义理解 | 词袋模型，无词序信息 | 深层 Transformer 编码，捕捉完整上下文语义 |
| 同义词 | 「好看」「精彩」「很棒」视为不同词 | 编码后向量相近，语义等价 |
| 跨领域 | 依赖关键词匹配 | 基于预训练的通用中文语义理解 |
| 向量维度 | 8000（稀疏） | 1024（稠密） |

---

#### 3.2.5 统一评估框架

三组模型使用**完全相同的 200 部测试集**（`RandomState(42)` 固定随机种子），确保可公平对比：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **precision@10** | Top-10 推荐中与查询电影类型或导演匹配的比例 | `genre_relevant / 10` |
| **hit_rate@10** | Top-10 中至少命中一部类型匹配电影的概率 | `命中数 / 测试集大小` |
| **tag_actor_recall@10** | Top-10 中标签或演员重合的比例 | `(标签交集 + 演员交集) / 10` |
| **rating_quality** | 加权质量分，推荐高分热门电影 | `log(1+votes) × rating / 10` |
| **coverage** | 推荐结果覆盖全库的比例 | `推荐涉及的不同电影数 / 总电影数` |
| **diversity** | 推荐结果的差异性 | `1 - (Top-10 内部类型平均重合度)` |

**最终排序策略：**

```python
score = similarity × log(1 + votes) × rating / 10
```

综合**相似度、热度（投票数对数变换）、评分**三个维度进行最终排序，避免仅推荐高分冷门电影或热门低分电影。

**综合评估报告：**

`generate_eval_report()` 自动对比三模型，输出最佳精度模型、最高命中率模型、最快训练模型等信息，保存为 JSON 格式。

---

### 3.3 协作网络分析 (`network_analyzer.py`)

**理论基础：图论 + 社交网络分析 (Social Network Analysis, SNA)**

#### 3.3.1 图的构建

**演员协作网络：**

```
节点 = 演员 (person_id 为唯一标识，name 为显示名称)
边   = 两个演员共同出演同一部电影（权重 = 合作次数）
```

- **过滤条件：** 演员至少出演 ≥ 3 部电影（过滤龙套角色，减少噪声节点）
- **截断策略：** 每部电影最多取前 10 位演员（防止如《建国大业》等全明星电影产生无意义的完全图）
- **边构建：** 使用 `itertools.combinations` 枚举电影内所有演员对，累加合作权重

**导演-演员协作网络：**

```
节点 = 导演 + 演员（节点带 type 属性区分，便于前端双色渲染）
边   = 导演与演员在电影中的合作关系
```

#### 3.3.2 网络分析指标体系

| 分析维度 | 方法 | 理论基础 |
|----------|------|----------|
| **全局统计** | 节点数 / 边数 / 密度 / 平均度 | 网络基本拓扑特征描述 |
| **小世界特性检验** | 对比真实网络与同规模随机网络的**聚类系数**和**平均最短路径** | Watts-Strogatz 小世界网络理论：若 `C_real ≫ C_random` 且 `L_real ≈ L_random`，则网络具有小世界特性 |
| **度分布与幂律拟合** | 对数坐标线性回归 `log(P(k)) = -α·log(k) + C` + MLE 极大似然估计 | **无标度网络 (Scale-Free)** 理论：若度分布服从 `P(k) ∝ k^(-α)`，则少数枢纽节点连接大量节点，网络对随机故障鲁棒、对蓄意攻击脆弱 |
| **KS 检验** | `scipy.stats.kstest` 对度分布做幂律分布的 Kolmogorov-Smirnov 拟合优度检验 | 统计检验幂律分布假设的显著性 |
| **K-Core 分解** | `nx.core_number(G)` | 逐步剥离度数 < k 的节点，揭示网络的**核心-边缘层次结构**，最高 k-core 为网络的「核心骨架」 |
| **中心性分析** | Degree Centrality + Betweenness Centrality | 度数中心性衡量节点的局部重要度（合作广度）；介数中心性衡量节点在网络中的全局桥梁作用（信息流动的必经之路） |
| **社区发现** | **Louvain 算法** (`community_louvain.best_partition`) | 基于**模块度 (Modularity) 优化**的层次聚类 — 通过迭代局部优化和网络聚合两阶段，自动发现演员的自然社群（如「香港电影圈」「内地喜剧圈」等） |
| **力导向图** | 取 Top 200 中心性最高节点，按 Louvain 社区着色 | 前端 ECharts 力导向布局渲染，直观展示演员社群结构 |
| **度关联性** | `degree_assortativity_coefficient` | 衡量高度数节点是否倾向连接高度数节点（同配网络，如社交网络）还是低度数节点（异配网络，如技术网络） |

**Louvain 社区发现算法原理：**

1. **阶段一（局部优化）：** 将每个节点初始化为独立社区，迭代遍历所有节点，计算将其移动到每个邻居社区后的模块度增益 ΔQ，选择增益最大的移动
2. **阶段二（网络聚合）：** 将阶段一产生的社区聚合成「超节点」，构建新的加权网络
3. 重复上述两阶段，直到模块度不再增加（局部最优）

---

### 3.4 数据可视化 (`echarts_visualizer.py`)

为前端 ECharts 生成完整的 `option` 配置字典，覆盖五种图表类型：

| 图表 | ECharts 类型 | 展示内容 | 交互特性 |
|------|-------------|----------|----------|
| **评分分布** | 柱状图 (bar) | 评分区间分布，渐变色柱体 | 工具栏：保存图片 / 数据缩放 / 还原 |
| **年份分布** | 折线图 (line) + dataZoom | 各年份电影数量趋势，半透明面积填充 | 底部滑块 + 内部滚轮缩放 |
| **国家分布** | 玫瑰图 (pie, radius) | Top 10 国家 + 「其他」，自动清洗中英混杂 | 悬停显示百分比 |
| **类型分布** | 雷达图 (radar) | Top 8 电影类型，粉色半透明填充 | 直观对比各类型数量 |
| **导演排名** | 横向柱状图 (bar) | Top 15 导演作品数量 | 工具栏：数据视图 / 保存图片 / 还原 |

**国家名称标准化：**

内置 `COUNTRY_MAP`（英文→中文映射）和 `FONT_MAP`（繁体→简体映射），将中国香港/澳门/台湾统一归入「中国」类别，处理「USA→美国」「Danmark→丹麦」等异构数据。

---

### 3.5 词云生成 (`db_manager.py`)

通过 `get_wordcloud_data()` 方法：
- jieba 分词 + 中文停用词过滤
- 统计词频，按频率降序排列
- 缓存到 `wordcloud_cache` 数据库表，支持 `?refresh=1` 参数强制重建
- 前端使用 ECharts 扩展或第三方词云库渲染

---

## 四、路由与 API 架构

### Blueprint 注册

```python
main_bp        →  /, /index, /home, /aboutMe
movies_bp      →  /movie, /score, /word, /api/charts/*
bigdata_bp     →  /bigdata/index, /bigdata/movie, /bigdata/score, /bigdata/word,
                  /api/bigdata/charts/*, /api/bigdata/wordcloud/*,
                  /api/bigdata/import, /api/bigdata/stats
analysis_bp    →  /analysis/network, /analysis/sentiment, /analysis/recommend,
                  /api/analysis/network/stats,
                  /api/analysis/sentiment/all, /dl, /predict, /dl/attention, /dl/curves,
                  /api/analysis/recommend/train, /by_movie, /eval,
                  /api/movie/detail, /api/analysis/export
```

### 关键设计模式

- **懒加载 + 线程安全预热：** 首次 API 请求时通过 `_ensure_warmup()` 启动后台 Daemon 线程，使用 `threading.Lock` 确保只启动一次
- **渐进式数据返回：** 预热未完成时返回 `{"status": "warming", "message": "..."}` 状态码，前端可轮询等待
- **多模型切换：** 推荐 API 通过 `?model=tfidf|svd|sbert` 参数在三组模型间切换
- **缓存透明化：** 路由层对缓存的存在与否透明，自动判断是加载缓存还是触发后台计算

---

## 五、GPU 加速说明

**硬件环境：** NVIDIA GeForce RTX 3050 Laptop GPU，4GB VRAM

**深度学习模型 GPU 使用情况：**

| 模型 | GPU 加速 | 精度 | 说明 |
|------|----------|------|------|
| **BiLSTM-Attention** (情感分析) | ✅ GPU | FP32 | Embedding + LSTM + Attention 均在 GPU 上训练和推理；模型轻量（约 9M 参数），4GB 显存完全充足 |
| **BGE-Large-Zh** (推荐系统) | ✅ GPU | **FP16 半精度** | 326M 参数的大模型，FP16 下显存需求从 ~1.3GB 降至 ~650MB，同时 batch_size 可从 128 提升至 256 |
| **传统 ML 模型** (LightGBM 等) | ❌ CPU | — | 基于 CPU 的 GBDT 梯度提升，不依赖 GPU |

**BGE-Large-Zh 的 FP16 自适应逻辑：**

```python
if torch.cuda.is_available():
    model.half()           # 切换为 FP16 半精度
    use_fp16 = True
    batch_size = 256       # FP16 下可用更大 batch
else:
    batch_size = 128       # CPU 回退方案
```

---

## 六、数据目录说明（`.gitignore` 排除项）

以下目录和文件类型被 `.gitignore` 排除在版本控制之外：

| 路径/类型 | 说明 |
|-----------|------|
| `data/` | **核心数据目录** — 包含 CSV 源文件、SQLite 数据库、模型缓存、词向量 (`.bin`/`.npy`/`.pt`/`.pkl`) |
| `backup/` | 优化重构前的旧版项目代码备份 |
| `*.csv`, `*.json`, `*.xlsx` | 数据文件 |
| `*.pkl`, `*.log`, `*.pyc` | 缓存/日志/编译文件 |
| `.venv/`, `.idea/`, `.vscode/` | IDE 和虚拟环境配置 |

---

## 七、论文相关文件

| 文件 | 内容 |
|------|------|
| `thesis_requirement/本科毕业设计（论文）要求与撰写规范（理工类）-本部潇湘.doc` | 学校毕业论文格式与撰写规范文档 |
| `thesis_requirement/exp.txt` | 摘要撰写要求：需包含研究目的和重要性、主要工作内容、基本结论和研究成果、结论意义；摘要中应说明采用的**技术、数据库、系统功能模块**；正文中不要出现大量源代码（可放关键代码） |
| `backup/` | 优化前的项目代码备份（不在版本控制中） |
