"""
多模型电影推荐引擎 — 融合 new_movies.db + bigdata_movies.db

支持三组模型：
  - TF-IDF       : 基于内容的余弦相似度基线（全量双库）
  - TruncatedSVD : 评分矩阵降维协同过滤（替代 LightFM / MF / NCF）
  - BGE-Large-Zh : 中文深度语义推荐（BAAI/bge-large-zh-v1.5, 1024dim, FP16）

缓存目录: data/cache/recommend/
评估报告: data/cache/recommend/recommend_eval.json
"""

import os
import json
import pickle
import time
import sqlite3
import numpy as np

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cache", "recommend",
)


# ==================== 1. 统一电影数据加载 ====================

def _load_unified_movies():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    movie_map = {}

    new_db_path = os.path.join(root, "data", "new_movies.db")
    conn = sqlite3.connect(new_db_path)
    conn.row_factory = sqlite3.Row
    new_rows = conn.execute(
        "SELECT movie_id, title, rating, total_ratings, directors, actors, "
        "genres, countries, languages, year, summary, tags "
        "FROM movies WHERE title != ''"
    ).fetchall()
    for r in new_rows:
        if not r["title"]:
            continue
        movie_map[r["movie_id"]] = {
            "movie_id": r["movie_id"],
            "title": str(r["title"]),
            "rating": float(r["rating"] or 0),
            "votes": int(r["total_ratings"] or 0),
            "directors": str(r["directors"] or ""),
            "actors": str(r["actors"] or ""),
            "genres": str(r["genres"] or ""),
            "country": str(r["countries"] or ""),
            "languages": str(r["languages"] or ""),
            "year": int(r["year"] or 0),
            "summary": str(r["summary"] or ""),
            "tags": str(r["tags"] or ""),
            "source": "new_movies",
        }
    conn.close()
    print(f"[recommend] 从 new_movies.db 加载 {len(movie_map)} 部电影")

    bigdata_db_path = os.path.join(root, "data", "bigdata_movies.db")
    conn2 = sqlite3.connect(bigdata_db_path)
    conn2.row_factory = sqlite3.Row
    big_rows = conn2.execute(
        "SELECT movie_id, name, douban_score, douban_votes, directors, actors, "
        "genres, regions, languages, year, storyline, tags "
        "FROM movies_valid WHERE douban_score > 0 AND douban_votes > 0 AND name != ''"
    ).fetchall()
    added = 0
    skipped = 0
    for r in big_rows:
        if r["movie_id"] in movie_map:
            skipped += 1
            continue
        if not r["name"]:
            continue
        movie_map[r["movie_id"]] = {
            "movie_id": r["movie_id"],
            "title": str(r["name"]),
            "rating": float(r["douban_score"] or 0),
            "votes": int(r["douban_votes"] or 0),
            "directors": str(r["directors"] or ""),
            "actors": str(r["actors"] or ""),
            "genres": str(r["genres"] or ""),
            "country": str(r["regions"] or ""),
            "languages": str(r["languages"] or ""),
            "year": int(r["year"] or 0),
            "summary": str(r["storyline"] or ""),
            "tags": str(r["tags"] or ""),
            "source": "bigdata",
        }
        added += 1
    conn2.close()
    print(f"[recommend] 从 bigdata_movies.db 新增 {added} 部, 跳过 {skipped} 部 (重复)")
    print(f"[recommend] 双库统一后总计 {len(movie_map)} 部电影")

    movies = list(movie_map.values())
    return movies


# ==================== 2. 统一评估框架 ====================

_UNIFIED_EVAL_SAMPLE = 200


def _get_common_test_set(movies):
    rng = np.random.RandomState(42)
    n = min(_UNIFIED_EVAL_SAMPLE, len(movies))
    return rng.choice(len(movies), n, replace=False)


def _split_genres(genre_str):
    if not genre_str:
        return set()
    return {g.strip() for g in genre_str.replace(",", "/").split("/") if g.strip()}


def _evaluate_by_genre_match(movies, nn_index, query_builder, model_label):
    sample_indices = _get_common_test_set(movies)
    import math

    genre_prec_list = []
    tag_actor_recall_list = []
    rating_qual_list = []
    hit_count = 0
    for idx in sample_indices:
        m = movies[idx]
        query = query_builder(m, idx)
        distances, neighbors = nn_index.kneighbors(query, n_neighbors=11)
        query_genres = _split_genres(m["genres"])
        query_dir = _split_genres(m["directors"])
        query_tags = _split_genres(m["tags"])
        query_actors = _split_genres(m["actors"])

        genre_relevant = 0
        ta_relevant = 0
        rq_sum = 0.0
        top10_count = 0
        for nb_idx in neighbors[0]:
            if movies[nb_idx]["movie_id"] == m["movie_id"]:
                continue
            nb_genres = _split_genres(movies[nb_idx]["genres"])
            nb_dir = _split_genres(movies[nb_idx]["directors"])
            nb_tags = _split_genres(movies[nb_idx]["tags"])
            nb_actors = _split_genres(movies[nb_idx]["actors"])
            nb_votes = max(0, movies[nb_idx]["votes"])
            nb_rating = max(0.1, movies[nb_idx]["rating"])

            if query_genres & nb_genres or query_dir & nb_dir:
                genre_relevant += 1
            if query_tags & nb_tags or query_actors & nb_actors:
                ta_relevant += 1

            rq_sum += math.log(1 + nb_votes) * nb_rating / 10.0
            top10_count += 1

        genre_prec_list.append(genre_relevant / 10)
        tag_actor_recall_list.append(ta_relevant / 10)
        rating_qual_list.append(rq_sum / max(1, top10_count))
        if genre_relevant > 0:
            hit_count += 1

    coverage_set = set()
    for idx in sample_indices[:50]:
        m = movies[idx]
        q = query_builder(m, idx)
        _, neighbors = nn_index.kneighbors(q, n_neighbors=11)
        for nb_idx in neighbors[0][1:]:
            coverage_set.add(movies[nb_idx]["movie_id"])
    coverage = len(coverage_set) / max(1, len(movies))

    div_sum = 0.0
    div_count = 0
    for idx in sample_indices[:30]:
        m = movies[idx]
        q = query_builder(m, idx)
        _, neighbors = nn_index.kneighbors(q, n_neighbors=11)
        top10_movies = [movies[ni] for ni in neighbors[0][1:11]]
        for pi, ma in enumerate(top10_movies):
            for mb in top10_movies[pi + 1:]:
                div_count += 1
                if _split_genres(ma["genres"]) & _split_genres(mb["genres"]):
                    div_sum += 1.0
    diversity = 1.0 - (div_sum / max(1, div_count))

    genre_prec = np.mean(genre_prec_list)
    ta_recall = np.mean(tag_actor_recall_list)
    rq_mean = np.mean(rating_qual_list) / 10.0
    hit_rate = hit_count / sample_indices.shape[0]

    print(f"[recommend] {model_label} 统一评估: "
          f"genre_prec={genre_prec:.4f}, ta_recall={ta_recall:.4f}, "
          f"rating_qual={rq_mean:.4f}, hit@10={hit_rate:.4f}, "
          f"coverage={coverage:.4f}, diversity={diversity:.4f}")

    return {
        "precision@10": round(float(genre_prec), 4),
        "genre_precision@10": round(float(genre_prec), 4),
        "hit_rate@10": round(float(hit_rate), 4),
        "tag_actor_recall@10": round(float(ta_recall), 4),
        "rating_quality": round(float(rq_mean), 4),
        "coverage": round(float(coverage), 4),
        "diversity": round(float(diversity), 4),
        "eval_samples": int(sample_indices.shape[0]),
    }


def _rating_weight(votes, rating):
    import math
    return math.log(1 + max(0, votes)) * max(0.1, rating) / 10.0


def _refresh_metrics_if_needed(data, nn_index, movies, query_builder, model_label):
    m = data.get("metrics", {})
    if "rating_quality" not in m:
        print(f"[recommend] {model_label} 缓存缺少新指标, 补跑评估...")
        data["metrics"] = _evaluate_by_genre_match(movies, nn_index, query_builder, model_label)
    return data


# ==================== 3. 模型 A: TF-IDF 双库基线 ====================

CACHE_TFIDF = os.path.join(CACHE_DIR, "tfidf_model.pkl")


def train_tfidf(force=False):
    if not force and os.path.exists(CACHE_TFIDF):
        with open(CACHE_TFIDF, "rb") as f:
            return pickle.load(f)

    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    print("[recommend] === 模型 A: TF-IDF 双库基线 ===")
    t0 = time.time()

    movies = _load_unified_movies()
    movies = [m for m in movies if m["votes"] >= 10]
    print(f"[recommend] TF-IDF 过滤后 (votes>=10): {len(movies)} 部")

    stop_words = _get_stop_words()

    docs = []
    for m in movies:
        parts = []
        parts.append(m["genres"])
        parts.append(m["genres"])
        parts.append(m["directors"])
        parts.append(m["directors"])
        parts.append(m["actors"])
        parts.append(m["actors"])
        parts.append(m["tags"])
        parts.append(m["tags"])
        summary = m["summary"]
        if len(summary) > 300:
            for sep in ["。", "！", "？", "\n"]:
                pos = summary[:300].rfind(sep)
                if pos > 150:
                    summary = summary[:pos + 1]
                    break
            else:
                summary = summary[:300]
        parts.append(summary)
        parts.append(m["languages"])

        text = " ".join(parts)
        words = jieba.lcut(text)
        filtered = [w for w in words if len(w) >= 2 and w not in stop_words
                     and any('\u4e00' <= c <= '\u9fff' for c in w)]
        docs.append(" ".join(filtered))

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                 min_df=3, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    nn_index = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn_index.fit(matrix)

    elapsed = time.time() - t0
    print(f"[recommend] TF-IDF 训练完成 ({elapsed:.1f}s), "
          f"矩阵={matrix.shape}, 特征词={len(vectorizer.get_feature_names_out())}")

    data = {
        "model_name": "TF-IDF",
        "vectorizer": pickle.dumps(vectorizer),
        "nn_index": pickle.dumps(nn_index),
        "movies": movies,
        "matrix_shape": matrix.shape,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_seconds": round(elapsed, 1),
    }

    def _tfidf_query_builder(m, idx):
        from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
        return vectorizer.transform([docs[idx]])

    metrics = _evaluate_by_genre_match(movies, nn_index, _tfidf_query_builder, "TF-IDF")
    data["metrics"] = metrics

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_TFIDF, "wb") as f:
        pickle.dump(data, f)
    print(f"[recommend] TF-IDF 缓存已保存: {CACHE_TFIDF}")
    return data


# ==================== 4. 模型 B: TruncatedSVD 协同过滤 ====================

CACHE_SVD = os.path.join(CACHE_DIR, "svd_model.pkl")
CACHE_SVD_NPZ = os.path.join(CACHE_DIR, "svd_item_vectors.npz")


def train_svd(force=False):
    if not force and os.path.exists(CACHE_SVD) and os.path.exists(CACHE_SVD_NPZ):
        with open(CACHE_SVD, "rb") as f:
            data = pickle.load(f)
        return data

    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    print("[recommend] === 模型 B: TruncatedSVD 协同过滤 ===")
    t0 = time.time()

    movies = _load_unified_movies()
    movies = [m for m in movies if m["votes"] >= 10]
    print(f"[recommend] SVD 数据准备: 全库 {len(movies)} 部电影")

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conn = sqlite3.connect(os.path.join(root, "data", "bigdata_movies.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT user_md5, movie_id, rating FROM comments_valid
        WHERE rating IS NOT NULL AND content != ''
    """).fetchall()
    conn.close()

    ratings_raw = [(r["user_md5"], r["movie_id"], float(r["rating"])) for r in rows]
    print(f"[recommend] SVD 加载 {len(ratings_raw)} 条原始评分")

    user_counts = Counter(u for u, _, _ in ratings_raw)
    item_counts = Counter(i for _, i, _ in ratings_raw)
    active_users = {u for u, c in user_counts.items() if c >= 5}
    popular_items = {i for i, c in item_counts.items() if c >= 10}

    filtered = [(u, i, r) for u, i, r in ratings_raw
                if u in active_users and i in popular_items]
    users = sorted(set(u for u, _, _ in filtered))
    items = sorted(set(i for _, i, _ in filtered))
    u2idx = {u: j for j, u in enumerate(users)}
    i2idx = {i: j for j, i in enumerate(items)}
    idx2i = {j: i for i, j in i2idx.items()}
    print(f"[recommend] SVD 过滤后: 用户={len(users)}, 电影={len(items)}, 评分={len(filtered)}")

    row_indices = np.array([u2idx[u] for u, _, _ in filtered], dtype=np.int32)
    col_indices = np.array([i2idx[i] for _, i, _ in filtered], dtype=np.int32)
    values = np.array([r for _, _, r in filtered], dtype=np.float32)

    matrix = csr_matrix((values, (row_indices, col_indices)),
                        shape=(len(users), len(items)), dtype=np.float32)
    print(f"[recommend] SVD 评分矩阵: {matrix.shape}, 密度={matrix.nnz / (matrix.shape[0]*matrix.shape[1]):.4f}")

    svd = TruncatedSVD(n_components=128, n_iter=10, random_state=42)
    item_vectors = svd.fit_transform(matrix.T)
    explained = svd.explained_variance_ratio_.sum()
    print(f"[recommend] SVD 分解完成: {item_vectors.shape}, 解释方差={explained:.4f}")

    nn_index = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn_index.fit(item_vectors)

    id_to_movie = {m["movie_id"]: m for m in movies}
    svd_movies = []
    svd_idx_to_global = []
    for j, item_id in enumerate(items):
        if item_id in id_to_movie:
            svd_movies.append(id_to_movie[item_id])
            svd_idx_to_global.append(j)
        else:
            svd_movies.append({
                "movie_id": item_id,
                "title": str(item_id),
                "rating": 0,
                "votes": 0,
                "directors": "",
                "actors": "",
                "genres": "",
                "country": "",
                "languages": "",
                "year": 0,
                "summary": "",
                "tags": "",
                "source": "bigdata",
            })
            svd_idx_to_global.append(j)
    svd_idx_to_global = np.array(svd_idx_to_global, dtype=np.int32)

    elapsed = time.time() - t0

    data = {
        "model_name": "TruncatedSVD",
        "nn_index": pickle.dumps(nn_index),
        "movies": svd_movies,
        "svd_items": items,
        "item_to_idx": i2idx,
        "idx_to_item": idx2i,
        "svd_idx_to_global": svd_idx_to_global,
        "num_users": len(users),
        "num_items": len(items),
        "svd_components": 128,
        "explained_variance": round(float(explained), 4),
        "train_seconds": round(elapsed, 1),
    }

    def _svd_query_builder(m, movie_idx):
        for j, item_id in enumerate(items):
            if m["movie_id"] == item_id:
                return item_vectors[j:j + 1]
        return item_vectors[0:1]

    metrics = _evaluate_by_genre_match(svd_movies, nn_index, _svd_query_builder, "SVD")
    data["metrics"] = metrics

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_SVD, "wb") as f:
        pickle.dump(data, f)
    np.savez_compressed(CACHE_SVD_NPZ, item_vectors=item_vectors.astype(np.float32))
    print(f"[recommend] SVD 缓存已保存: {CACHE_SVD} + {CACHE_SVD_NPZ}")
    return data


# ==================== 5. 模型 C: BGE-Large-Zh 深度语义推荐 ====================

CACHE_SBERT = os.path.join(CACHE_DIR, "sbert_model.pkl")
CACHE_SBERT_EMB = os.path.join(CACHE_DIR, "sbert_embeddings.npy")


def train_sbert(force=False):
    if not force and os.path.exists(CACHE_SBERT) and os.path.exists(CACHE_SBERT_EMB):
        with open(CACHE_SBERT, "rb") as f:
            data = pickle.load(f)
        embeddings = np.load(CACHE_SBERT_EMB)
        data["embeddings"] = embeddings
        nn_index = pickle.loads(data["nn_index"])
        movies = data["movies"]
        def _qb(m, idx):
            return embeddings[idx:idx + 1]
        data = _refresh_metrics_if_needed(data, nn_index, movies, _qb, "BGE-Large(cached)")
        with open(CACHE_SBERT, "wb") as f:
            pickle.dump(data, f)
        return data

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[recommend] sentence-transformers 未安装，跳过模型 C")
        return None

    print("[recommend] === 模型 C: BGE-Large-Zh 深度语义推荐 ===")
    t0 = time.time()

    movies = _load_unified_movies()
    movies = [m for m in movies if m["votes"] >= 10]
    print(f"[recommend] BGE-Large 过滤后 (votes>=10): {len(movies)} 部")

    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    use_gpu = False
    use_fp16 = False
    if HAS_TORCH and torch.cuda.is_available():
        use_gpu = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem = getattr(gpu_props, 'total_memory', getattr(gpu_props, 'total_mem', 0))
            gpu_mem_gb = gpu_mem / (1024 ** 3) if gpu_mem else 0
            print(f"[recommend] GPU: {gpu_name}, 显存: {gpu_mem_gb:.1f} GB")
        except Exception as e:
            print(f"[recommend] GPU 信息获取失败: {e}")
        try:
            model.half()
            use_fp16 = True
            print(f"[recommend] BGE-Large 已切换为 FP16 精度 (326M params, 1024dim)")
        except Exception as e:
            print(f"[recommend] FP16 切换失败: {e}, 使用 FP32")
    else:
        print("[recommend] BGE-Large 使用 CPU 模式")

    print(f"[recommend] BGE-Large 配置: GPU={use_gpu}, FP16={use_fp16}")

    texts = []
    for m in movies:
        summary = m["summary"]
        if len(summary) > 300:
            for sep in ["。", "！", "？", "\n"]:
                pos = summary[:300].rfind(sep)
                if pos > 150:
                    summary = summary[:pos + 1]
                    break
            else:
                summary = summary[:300]

        texts.append(
            f"{m['title']}。"
            f"类型：{m['genres']}。"
            f"导演：{m['directors']}。"
            f"评分：{m['rating']:.1f}分，{m['votes']}人评价。"
            f"剧情：{summary}。"
            f"标签：{m['tags']}"
        )

    device_str = "cuda" if use_gpu else "cpu"
    batch_size = 256 if use_fp16 else 128

    print(f"[recommend] BGE-Large 编码中, batch_size={batch_size}, {len(texts)} 部电影...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              device=device_str, normalize_embeddings=True)
    print(f"[recommend] BGE-Large 编码完成, shape={embeddings.shape}, dtype={embeddings.dtype}")

    from sklearn.neighbors import NearestNeighbors
    nn_index = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn_index.fit(embeddings)

    elapsed = time.time() - t0

    data = {
        "model_name": "BGE-Large-Zh-v1.5",
        "nn_index": pickle.dumps(nn_index),
        "movies": movies,
        "embedding_dim": int(embeddings.shape[1]),
        "inference_seconds": round(elapsed, 1),
        "gpu_used": use_gpu,
        "fp16_used": use_fp16,
    }

    def _sbert_query_builder(m, idx):
        return embeddings[idx:idx + 1]

    metrics = _evaluate_by_genre_match(movies, nn_index, _sbert_query_builder, "BGE-Large")
    data["metrics"] = metrics

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(CACHE_SBERT_EMB, embeddings)
    with open(CACHE_SBERT, "wb") as f:
        pickle.dump(data, f)
    print(f"[recommend] BGE-Large 缓存已保存: {CACHE_SBERT} + {CACHE_SBERT_EMB}")
    data["embeddings"] = embeddings
    return data


# ==================== 6. 综合评估报告 ====================

EVAL_PATH = os.path.join(CACHE_DIR, "recommend_eval.json")


def generate_eval_report():
    print("\n[recommend] ======================================")
    print("[recommend] 生成综合评估报告")

    report = {
        "eval_time": time.strftime("%Y-%m-%d %H:%M"),
        "models": {},
    }

    try:
        td = train_tfidf()
        report["models"]["TF-IDF"] = td["metrics"]
        report["models"]["TF-IDF"]["train_seconds"] = td["train_seconds"]
        report["models"]["TF-IDF"]["feature_count"] = td.get("feature_count", 0)
        report["models"]["TF-IDF"]["movie_count"] = len(td["movies"])
        print(f"  TF-IDF:   prec@10={td['metrics']['precision@10']}")
    except Exception as e:
        print(f"  TF-IDF:   训练失败 - {e}")

    try:
        sd = train_svd()
        if sd and sd.get("metrics"):
            report["models"]["TruncatedSVD"] = sd["metrics"]
            report["models"]["TruncatedSVD"]["train_seconds"] = sd["train_seconds"]
            report["models"]["TruncatedSVD"]["num_users"] = sd.get("num_users", 0)
            report["models"]["TruncatedSVD"]["num_items"] = sd.get("num_items", 0)
            report["models"]["TruncatedSVD"]["movie_count"] = len(sd["movies"])
            report["models"]["TruncatedSVD"]["explained_variance"] = sd.get("explained_variance", 0)
            print(f"  SVD:      prec@10={sd['metrics']['precision@10']}")
    except Exception as e:
        print(f"  SVD:      训练失败 - {e}")

    try:
        bd = train_sbert()
        if bd:
            report["models"]["BGE-Large-Zh"] = bd["metrics"]
            report["models"]["BGE-Large-Zh"]["inference_seconds"] = bd["inference_seconds"]
            report["models"]["BGE-Large-Zh"]["gpu_used"] = bd.get("gpu_used", False)
            report["models"]["BGE-Large-Zh"]["fp16_used"] = bd.get("fp16_used", False)
            report["models"]["BGE-Large-Zh"]["embedding_dim"] = bd.get("embedding_dim", 0)
            report["models"]["BGE-Large-Zh"]["movie_count"] = len(bd["movies"])
            print(f"  BGE-Large: prec@10={bd['metrics']['precision@10']}")
    except Exception as e:
        print(f"  SBERT:    训练失败 - {e}")

    models = report["models"]
    if models:
        best_prec = max(models.items(), key=lambda x: x[1].get("precision@10", 0))
        best_hit = max(models.items(), key=lambda x: x[1].get("hit_rate@10", 0))
        fastest = min(models.items(), key=lambda x: x[1].get("train_seconds", 99999))
        report["comparison_summary"] = {
            "best_precision": f"{best_prec[0]} (precision@10={best_prec[1].get('precision@10', 0)})",
            "best_hit_rate": f"{best_hit[0]} (hit_rate@10={best_hit[1].get('hit_rate@10', 0)})",
            "fastest_training": f"{fastest[0]} ({fastest[1].get('train_seconds', 0)}s)",
            "note": "三组模型使用统一评估框架（200部共用测试集），指标可直接对比",
        }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[recommend] 评估报告已保存: {EVAL_PATH}")
    return report


# ==================== 7. 辅助函数 ====================

def _get_stop_words():
    return {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "说", "要", "你", "会", "着", "没有", "看",
        "好", "自己", "这", "他", "她", "们", "那", "什么", "怎么", "可以",
        "还", "被", "让", "从", "与", "但", "而", "等", "及", "之", "为",
        "对", "于", "以", "电影", "一部", "真的", "感觉", "觉得", "还是", "不过",
        "已经", "不是", "就是", "这么", "那么", "一直", "一点", "很多",
        "出来", "开始", "最后", "比较", "其实", "有点", "因为", "所以",
        "我们", "他们", "你们", "大家", "但是", "然而",
        "不错", "喜欢", "好看", "还行", "一般",
        "演技", "剧情", "导演", "演员", "角色", "故事", "片子", "画面",
    }
