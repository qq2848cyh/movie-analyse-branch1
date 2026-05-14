# 数据库分析与表结构说明

---

## 一、总体概览

项目使用**两部独立的 SQLite 数据库**，分别服务于不同的业务场景：

| 数据库 | 路径 | 数据来源 | 用途 | 管理类 |
|--------|------|----------|------|--------|
| **大数据分析库** | `data/bigdata_movies.db` | `data/bigdata/` 下 3 个 CSV（movies.csv、person.csv、comments.csv） | 情感分析、推荐系统、协作网络分析、大数据集浏览 | `DBManager` |
| **精选展示库** | `data/new_movies.db` | `data/new_data/douban_all_movies.csv` | 精选电影展示页、ECharts 可视化、词云 | `NewMovieManager` |

> **两者完全独立**，不存在跨库外键。各自的数据导入、缓存、查询互不干扰。

---

## 二、大数据分析库 `bigdata_movies.db`

### 2.1 表总览

| 表名 | 类型 | 数据量 | 说明 |
|------|------|--------|------|
| `movies` | 原始表 | **140,502** 行 | 全量豆瓣电影元数据 |
| `movies_valid` | 派生表（物化视图） | **25,968** 行 | 有效电影（评分>0 且 评分人数>0） |
| `person` | 原始表 | **72,959** 行 | 演职人员档案（演员/导演/编剧等） |
| `comments` | 原始表 | **4,428,475** 行 | 全量用户评论 |
| `comments_valid` | 派生表（物化视图） | **2,694,154** 行 | 有效电影的评论 |
| `comments_fts` | FTS5 虚拟表 | — | 评论文本全文索引 |
| `movie_actors` | 关联表 | **213,434** 行 | 电影-演员多对多映射 |
| `movie_directors` | 关联表 | **36,318** 行 | 电影-导演多对多映射 |
| `wordcloud_cache` | 缓存表 | 1 行 | 词云计算缓存 |

### 2.2 各表详细说明

---

#### `movies` — 电影元数据（原始表）

**数据来源**：`data/bigdata/movies.csv`

| 字段 | 类型 | 说明 | 索引 |
|------|------|------|------|
| `movie_id` | INTEGER | **主键**，豆瓣电影唯一ID | — |
| `name` | TEXT | 电影名称（中文） | ✅ `idx_movies_name` |
| `actors` | TEXT | 主演列表（用 `\|` 分隔，"姓名:ID"格式） | — |
| `cover` | TEXT | 封面图URL | — |
| `directors` | TEXT | 导演列表（用 `\|` 分隔） | — |
| `douban_score` | REAL | 豆瓣评分（0~10） | ✅ `idx_movies_douban_score` |
| `douban_votes` | REAL | 评分人数 | — |
| `genres` | TEXT | 类型标签（`/` 分隔，如"剧情/爱情"） | — |
| `imdb_id` | TEXT | IMDb 编号 | — |
| `languages` | TEXT | 语言（`/` 分隔） | — |
| `mins` | REAL | 片长（分钟） | — |
| `official_site` | TEXT | 官方网站URL | — |
| `regions` | TEXT | 制片国家/地区（`/` 分隔） | — |
| `release_date` | TEXT | 上映日期 | — |
| `storyline` | TEXT | 剧情简介 | — |
| `tags` | TEXT | 用户标签 | — |
| `year` | INTEGER | 上映年份（清洗后，范围 1907~2019） | ✅ `idx_movies_year` |

> **注意**：原始 CSV 中 `ACTORS` 字段为 `姓名:ID|姓名:ID|...` 格式，导入时拆分为 `movie_actors` 关联表；`DIRECTOR_IDS` 同理导入到 `movie_directors`。原始 CSV 中还包含 `ALIAS`（别名）、`SLUG`（短链接）等字段，导入时被跳过。

---

#### `movies_valid` — 有效电影（派生表）

由原始 `movies` 表通过以下 SQL 过滤生成：

```sql
CREATE TABLE movies_valid AS
SELECT * FROM movies
WHERE douban_score > 0 AND douban_votes > 0
```

**数据意义**：剔除评分或评分人数为 0/null 的无效记录，保证分析数据质量。后续所有**情感分析、推荐系统、图表统计**均基于此表。

**索引**：
- `idx_valid_score` — 评分降序排序
- `idx_valid_votes` — 评分人数降序
- `idx_valid_year` — 按年份筛选
- `idx_valid_name` — 电影名搜索

**关键统计**：

| 指标 | 值 |
|------|-----|
| 有效电影数 | **25,968** 部 |
| 平均豆瓣评分 | **6.59** |
| 年份范围 | **1907 ~ 2019** |
| 累计评分人次 | **207,144,033** |

---

#### `person` — 演职人员档案

**数据来源**：`data/bigdata/person.csv`

| 字段 | 类型 | 说明 |
|------|------|------|
| `person_id` | INTEGER | **主键**，人员唯一ID |
| `name` | TEXT | 姓名（索引） |
| `sex` | TEXT | 性别 |
| `name_en` | TEXT | 英文名 |
| `name_zh` | TEXT | 中文名（可能包含别名） |
| `birth` | TEXT | 出生日期 |
| `birthplace` | TEXT | 出生地 |
| `constellatory` | TEXT | 星座 |
| `profession` | TEXT | 职业（如"演员"、"导演 / 编剧"等，`/` 分隔多重身份） |
| `biography` | TEXT | 个人简介 |

**数据规模**：**72,959** 人

**职业分布（Top 5）**：

| 职业 | 人数 |
|------|------|
| 演员 | 38,933 |
| 演员 / 配音 | 5,659 |
| 导演 / 编剧 | 2,210 |
| 演员 / 编剧 | 1,863 |
| 导演 | 1,300 |

**索引**：`idx_person_name`

---

#### `comments` — 用户评论（原始表）

**数据来源**：`data/bigdata/comments.csv`

| 字段 | 类型 | 说明 |
|------|------|------|
| `comment_id` | INTEGER | **主键**，评论唯一ID |
| `user_md5` | TEXT | 用户标识（MD5匿名化） |
| `movie_id` | INTEGER | 关联电影ID → `movies.movie_id` |
| `content` | TEXT | 评论文本内容 |
| `votes` | INTEGER | 该评论获赞数 |
| `comment_time` | TEXT | 评论时间 |
| `rating` | INTEGER | 用户评分（1~5星） |

**数据规模**：**4,428,475** 条

**索引**：
- `idx_comments_movie_id` — 按电影查询评论
- `idx_comments_rating` — 按评分筛选

---

#### `comments_valid` — 有效评论（派生表）

```sql
CREATE TABLE comments_valid AS
SELECT c.* FROM comments c
INNER JOIN movies_valid mv ON c.movie_id = mv.movie_id
```

只保留属于有效电影的评论，过滤掉无效电影关联的评论数据。

**索引**：
- `idx_cvalid_mid` — 按 movie_id 快速查询
- `idx_cvalid_cid` — 按 comment_id 快速查询

**关键统计**：

| 指标 | 值 |
|------|-----|
| 有效评论数 | **2,694,154** 条 |
| 平均星级 | **3.34** |
| 评论累计获赞 | **15,838,042** |

---

#### `comments_fts` — FTS5 全文索引

```sql
CREATE VIRTUAL TABLE comments_fts
USING fts5(content, content='comments', content_rowid='comment_id')
```

基于 SQLite FTS5 引擎的全文搜索索引，支持对评论内容 `content` 字段的快速文本检索。作为外部内容表（`content='comments'`），索引数据直接映射到 `comments` 表，修改原始表后需要 `REBUILD` 重建索引。

---

#### `movie_actors` — 电影-演员关联表

| 字段 | 类型 | 约束 |
|------|------|------|
| `movie_id` | INTEGER | **联合主键** → `movies.movie_id` |
| `person_id` | INTEGER | **联合主键** → `person.person_id` |
| `actor_name` | TEXT | 演员姓名（冗余存储，便于展示） |

**数据规模**：**213,434** 对关联

**来源**：从 `movies.csv` 的 `ACTOR_IDS` 字段（格式 `姓名:ID|姓名:ID|...`）解析生成。

**用途**：协作网络分析的**演员协作图构建**——同一部电影中的演员两两连边，构成演员合作网络。

**索引**：`idx_movie_actors_person_id`（按人员查询其参演电影）

---

#### `movie_directors` — 电影-导演关联表

| 字段 | 类型 | 约束 |
|------|------|------|
| `movie_id` | INTEGER | **联合主键** → `movies.movie_id` |
| `person_id` | INTEGER | **联合主键** → `person.person_id` |
| `director_name` | TEXT | 导演姓名（冗余存储） |

**数据规模**：**36,318** 对关联

**来源**：从 `movies.csv` 的 `DIRECTOR_IDS` 字段（格式 `姓名:ID|姓名:ID|...`）解析生成。

---

#### `wordcloud_cache` — 词云缓存

| 字段 | 类型 | 说明 |
|------|------|------|
| `cache_key` | TEXT | **主键**，缓存键（当前固定为 `'all'`） |
| `data` | TEXT | JSON 格式的词云数据 |

---

### 2.3 表间关系（E-R 图）

```
                        ┌─────────────────┐
                        │     person       │
                        │  72,959 人       │
                        │  ─────────────   │
                        │  person_id (PK)  │
                        └───┬──────┬───────┘
                            │      │
              ┌─────────────┘      └──────────────┐
              │ N:N                               │ N:N
     ┌────────┴──────────┐            ┌───────────┴───────────┐
     │  movie_actors     │            │  movie_directors      │
     │  213,434 对       │            │  36,318 对            │
     │  ───────────────  │            │  ─────────────────    │
     │  movie_id  (PK)   │            │  movie_id    (PK)     │
     │  person_id (PK)   │            │  person_id   (PK)     │
     └────────┬──────────┘            └───────────┬───────────┘
              │ N:N                               │ N:N
              └──────────┐        ┌───────────────┘
                    ┌────┴────────┴─────┐
                    │     movies         │
                    │  140,502 部        │
                    │  ────────────      │
                    │  movie_id (PK)     │
                    │  ↓ 过滤            │
                    │  movies_valid      │
                    │  25,968 部         │
                    └────────┬───────────┘
                             │ 1:N
                    ┌────────┴───────────┐
                    │    comments         │
                    │  4,428,475 条       │
                    │  ──────────────     │
                    │  comment_id (PK)    │
                    │  movie_id    (FK)   │
                    │  ↓ 过滤             │
                    │  comments_valid     │
                    │  2,694,154 条       │
                    └────────┬───────────┘
                             │ FTS5索引
                    ┌────────┴───────────┐
                    │  comments_fts      │
                    │  (全文搜索)        │
                    └────────────────────┘
```

**关联键说明**：

| 关系 | 类型 | 关联方式 |
|------|------|----------|
| `movies` → `comments` | **1:N** | `movies.movie_id = comments.movie_id` |
| `movies` → `movie_actors` → `person` | **N:N** | 通过 `movie_actors` 中间表 |
| `movies` → `movie_directors` → `person` | **N:N** | 通过 `movie_directors` 中间表 |
| `movies` → `movies_valid` | **派生** | 过滤 `douban_score > 0 AND douban_votes > 0` |
| `comments` → `comments_valid` | **派生** | 仅保留属于 `movies_valid` 的评论 |

---

### 2.4 数据流与业务用途

```
┌─────────────────────────┐
│  data/bigdata/*.csv     │  ← 原始数据源（3个CSV文件）
└──────┬──────────────────┘
       │ DBManager.import_all()
       ▼
┌─────────────────────────┐
│  movies / person /      │  ← 原始表（全量导入）
│  comments               │
└──────┬──────────────────┘
       │ ensure_valid_table()
       ▼
┌─────────────────────────┐
│  movies_valid /         │  ← 过滤后的有效数据
│  comments_valid         │
└──────┬──────────────────┘
       │
       ├──→ SentimentAnalyzer   → 情感分析（ML + DL）
       ├──→ NetworkAnalyzer     → 协作网络分析（演员/导演）
       ├──→ BigdataRecommender  → 三模型推荐系统
       ├──→ EChartsVisualizer   → 图表数据生成
       └──→ DBManager.get_movies_page() → 大数据集浏览
```

---

## 三、精选展示库 `new_movies.db`

### 3.1 表总览

| 表名 | 类型 | 数据量 | 说明 |
|------|------|--------|------|
| `movies` | 原始表 | **2,065** 行 | 精选高质量电影（高评分 + 高评分人数） |
| `wordcloud_cache` | 缓存表 | 1 行 | 词云计算缓存 |

### 3.2 `movies` 表详细说明

**数据来源**：`data/new_data/douban_all_movies.csv`

| 字段 | 类型 | 说明 | 索引 |
|------|------|------|------|
| `movie_id` | INTEGER | **主键**，豆瓣电影唯一ID | — |
| `title` | TEXT | 电影标题 | — |
| `rating` | REAL | 豆瓣评分 | ✅ `idx_new_rating` (DESC) |
| `total_ratings` | INTEGER | 评分总人数 | — |
| `directors` | TEXT | 导演（逗号分隔） | — |
| `actors` | TEXT | 主演（逗号分隔） | — |
| `screenwriters` | TEXT | 编剧 | — |
| `release_date` | TEXT | 上映日期 | — |
| `year` | INTEGER | 上映年份（清洗后，1931~2026） | ✅ `idx_new_year` |
| `genres` | TEXT | 类型标签 | — |
| `countries` | TEXT | 制片国家/地区 | — |
| `languages` | TEXT | 语言 | — |
| `runtime` | TEXT | 片长 | — |
| `summary` | TEXT | 剧情简介 | — |
| `link` | TEXT | 豆瓣详情页链接 | — |
| `poster` | TEXT | 海报图片URL | — |
| `tags` | TEXT | 用户标签 | — |

**关键统计**：

| 指标 | 值 |
|------|-----|
| 电影数 | **2,065** 部 |
| 平均评分 | **7.46** |
| 最高评分 | **9.7** |
| 年份范围 | **1931 ~ 2026** |
| 累计评分人次 | **594,127,484** |

> **数据意义**：相比大数据集（平均分 6.59），精选集平均分高达 7.46，体现了**高分精品**的筛选策略。评分总人次近 6 亿，说明入选电影均为高热度、高关注度的作品。

### 3.3 业务用途

- **精选电影展示页**：分页浏览、关键词搜索（片名/导演/演员/类型/国家/简介/年份）
- **ECharts 可视化**：评分分布柱状图、年份折线图、国家玫瑰图、类型雷达图、导演排行
- **词云生成**：从片名和简介中提取高频词

---

## 四、两库对比与定位

| 维度 | `bigdata_movies.db` | `new_movies.db` |
|------|---------------------|-----------------|
| **定位** | 大数据分析引擎 | 精选展示前台 |
| **电影数** | 140,502 raw / 25,968 valid | 2,065 |
| **评论数据** | ✅ 269 万条 | ❌ 无 |
| **人员档案** | ✅ 7.3 万 | ❌ 无 |
| **全文索引** | ✅ FTS5 | ❌ 无 |
| **多对多关系** | ✅ actor/director 关联表 | ❌ 扁平化 |
| **平均评分** | 6.59 | 7.46 |
| **评分人次** | 2.07 亿 | 5.94 亿 |
| **使用模块** | 情感分析、推荐系统、网络分析 | 电影浏览、图表展示 |
| **路由** | `bigdata.py` + `analysis.py` | `movies.py` |
| **管理类** | `DBManager` | `NewMovieManager` |

### 核心设计理念

1. **数据质量过滤**：`movies_valid` 剔除无效数据（评分 ≤ 0 或评分人数 = 0），保证 2.6 万部有效电影的分析基础
2. **分层服务**：原始表做数据存储，`_valid` 表做业务查询，FTS5 虚拟表做全文搜索
3. **冗余关联**：`movie_actors` 和 `movie_directors` 中冗余存储姓名字段，避免展示时的 JOIN 开销
4. **精选 vs 大库**：new_movies.db 精选高分高热电影用于前台展示，bigdata_movies.db 承载全部数据用于后台分析——**前台追求体验速度，后台追求数据深度**

---

## 五、索引策略总结

### `bigdata_movies.db`

| 索引名 | 表 | 列 | 场景 |
|--------|-----|-----|------|
| `idx_movies_year` | movies | year | 按年份区间查询 |
| `idx_movies_douban_score` | movies | douban_score | 按评分排序 |
| `idx_movies_name` | movies | name | 按片名搜索 |
| `idx_comments_movie_id` | comments | movie_id | 查询某电影的全部评论 |
| `idx_comments_rating` | comments | rating | 按评分筛选评论 |
| `idx_person_name` | person | name | 按人名查询 |
| `idx_movie_actors_person_id` | movie_actors | person_id | 查询某演员的电影 |
| `idx_movie_directors_person_id` | movie_directors | person_id | 查询某导演的电影 |
| `idx_valid_score` | movies_valid | douban_score DESC | 评分排序 |
| `idx_valid_votes` | movies_valid | douban_votes DESC | 热度排序 |
| `idx_valid_year` | movies_valid | year | 年份筛选 |
| `idx_valid_name` | movies_valid | name | 名称搜索 |
| `idx_cvalid_mid` | comments_valid | movie_id | 评论按电影查询 |
| `idx_cvalid_cid` | comments_valid | comment_id | 评论按ID查询 |
| `comments_fts` | comments_fts (FTS5) | content | 评论文本全文搜索 |

### `new_movies.db`

| 索引名 | 表 | 列 | 场景 |
|--------|-----|-----|------|
| `idx_new_rating` | movies | rating DESC | 评分降序展示 |
| `idx_new_year` | movies | year | 按年份筛选 |
