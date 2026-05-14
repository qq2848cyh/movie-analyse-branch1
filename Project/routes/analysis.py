"""智能分析路由模块"""

import os
import threading
import pickle
import json
import numpy as np

from flask import Blueprint, render_template, jsonify, request

import config as cfg
from utils.network_analyzer import NetworkAnalyzer
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.sentiment_dl import train_bilstm_attn, get_attention_visualization

analysis_bp = Blueprint("analysis", __name__)

CACHE_DIR = cfg.CACHE_DIR
CACHE_NETWORK = cfg.NETWORK_STATS_PATH
CACHE_SENTIMENT = os.path.join(cfg.CACHE_SENTIMENT_DIR, "sentiment_results.pkl")
CACHE_SENTIMENT_DL = os.path.join(cfg.CACHE_SENTIMENT_DIR, "sentiment_dl_results.pkl")

for _cache_path in (CACHE_NETWORK, CACHE_SENTIMENT, CACHE_SENTIMENT_DL):
    os.makedirs(os.path.dirname(_cache_path), exist_ok=True)

_network_cache = None
_sentiment_cache = None
_sentiment_dl_cache = None
_warmup_started = False
_warmup_lock = threading.Lock()


def _sanitize_for_json(obj):
    """递归转换 numpy 类型为 Python 原生类型，确保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load_pickle_cache(path, name):
    """从磁盘加载 pickle 缓存"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[cache] 从磁盘加载 {name} ✓", flush=True)
        return data
    except Exception as e:
        print(f"[cache] {name} 加载失败: {e}", flush=True)
        return None


def _save_pickle_cache(path, data, name):
    """保存 pickle 缓存到磁盘"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache] {name} 已持久化", flush=True)
    except Exception as e:
        print(f"[cache] {name} 保存失败: {e}", flush=True)


def _warmup_thread():
    """后台预热线程 — 网络分析 / 情感分析 / 推荐系统"""
    global _network_cache, _sentiment_cache, _sentiment_dl_cache

    _network_cache = _load_pickle_cache(CACHE_NETWORK, "网络分析")
    _sentiment_cache = _load_pickle_cache(CACHE_SENTIMENT, "情感分析")
    _sentiment_dl_cache = _load_pickle_cache(CACHE_SENTIMENT_DL, "情感分析DL")

    from utils.db_manager import DBManager
    db = DBManager(cfg.BIGDATA_DB_PATH)
    need_network = _network_cache is None
    need_sentiment = _sentiment_cache is None
    need_sentiment_dl = _sentiment_dl_cache is None

    try:
        if need_network or need_sentiment or need_sentiment_dl:
            print("[warmup] 开始后台计算...", flush=True)

        if need_network:
            print("[warmup] 构建协作网络...", flush=True)
            na = NetworkAnalyzer(db)
            na.build()
            _network_cache = na.get_all_stats()
            _save_pickle_cache(CACHE_NETWORK, _network_cache, "网络分析")
            print("[warmup] 协作网络分析完成 ✓", flush=True)

        if need_sentiment:
            print("[warmup] 训练情感分析模型...", flush=True)
            sa = SentimentAnalyzer(db)
            _sentiment_cache = sa.get_all_results(n_samples=2000000)
            _save_pickle_cache(CACHE_SENTIMENT, _sentiment_cache, "情感分析")
            print("[warmup] 情感分析完成 ✓", flush=True)

        if need_sentiment_dl:
            print("[warmup] 训练 BiLSTM-Attention 模型（约需 5-6 小时）...", flush=True)
            _sentiment_dl_cache = train_bilstm_attn(db, n_samples=2000000)
            _save_pickle_cache(CACHE_SENTIMENT_DL, _sentiment_dl_cache, "情感分析DL")
            print("[warmup] BiLSTM-Attention 训练完成 ✓", flush=True)

        if not need_network and not need_sentiment and not need_sentiment_dl:
            print("[warmup] 全部从缓存加载，无需计算 ✓", flush=True)
        else:
            print("[warmup] 全部模块预热完成!", flush=True)

        print("[warmup] 预热推荐系统模型...", flush=True)
        try:
            from utils.bigdata_recommend_engine import train_tfidf, train_svd, train_sbert
            train_tfidf()
            print("[warmup] TF-IDF 推荐模型就绪 ✓", flush=True)
            train_svd()
            print("[warmup] SVD 协同过滤模型就绪 ✓", flush=True)
            train_sbert()
            print("[warmup] BGE-Large-Zh 深度语义模型就绪 ✓", flush=True)
            from utils.bigdata_recommend_engine import generate_eval_report
            generate_eval_report()
            print("[warmup] 推荐系统评估报告已生成 ✓", flush=True)
            print("[warmup] 推荐系统全部模型预热完成!", flush=True)
        except Exception as e:
            print(f"[warmup] 推荐系统预热失败: {e}", flush=True)

    except Exception as e:
        import traceback
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[warmup] 预热失败: {err_msg}", flush=True)
        if _network_cache is None:
            _network_cache = {"status": "error", "message": str(e)}
        if _sentiment_cache is None:
            _sentiment_cache = {"status": "error", "message": str(e)}
        if _sentiment_dl_cache is None:
            _sentiment_dl_cache = {"status": "error", "message": str(e)}


def _ensure_warmup():
    """确保预热仅启动一次（线程安全）"""
    global _warmup_started
    if not _warmup_started:
        with _warmup_lock:
            if not _warmup_started:
                _warmup_started = True
                threading.Thread(target=_warmup_thread, daemon=True).start()


# ==================== 智能分析页面路由 ====================

@analysis_bp.route("/analysis/network")
def analysis_network():
    _ensure_warmup()
    return render_template("analysis_network.html")


@analysis_bp.route("/analysis/sentiment")
def analysis_sentiment():
    return render_template("analysis_sentiment.html")


@analysis_bp.route("/analysis/recommend")
def analysis_recommend():
    return render_template("analysis_recommend.html")


# ==================== 网络分析 API ====================

@analysis_bp.route("/api/analysis/network/stats")
def api_network_stats():
    if _network_cache is not None:
        return jsonify(_sanitize_for_json(_network_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "网络分析数据正在预热中，请稍后刷新"})


# ==================== 情感分析 API ====================

@analysis_bp.route("/api/analysis/sentiment/all")
def api_sentiment_all():
    if _sentiment_cache is not None:
        return jsonify(_sanitize_for_json(_sentiment_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "情感分析数据正在预热中，请稍后刷新"})


@analysis_bp.route("/api/analysis/sentiment/dl")
def api_sentiment_dl():
    if _sentiment_dl_cache is not None:
        return jsonify(_sanitize_for_json(_sentiment_dl_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "深度学习模型正在训练中，请稍后刷新"})


@analysis_bp.route("/api/analysis/sentiment/predict")
def api_sentiment_predict():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入评论内容"})

    try:
        from utils.db_manager import DBManager
        db = DBManager(cfg.BIGDATA_DB_PATH)
        analyzer = SentimentAnalyzer(db)
        analyzer.train(n_samples=15000)
        pred = analyzer.predict(text)
        return jsonify({"text": text, "prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)})


@analysis_bp.route("/api/analysis/sentiment/dl/attention")
def api_sentiment_dl_attention():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入评论内容"})
    if _sentiment_dl_cache is None or not isinstance(_sentiment_dl_cache, dict):
        return jsonify({"error": "深度学习模型尚未就绪，请等待训练完成"})
    if _sentiment_dl_cache.get("status") == "error":
        return jsonify({"error": "深度学习模型训练失败"})
    try:
        result = get_attention_visualization(text)
        return jsonify(_sanitize_for_json(result))
    except Exception as e:
        return jsonify({"error": f"注意力分析失败: {e}"})


@analysis_bp.route("/api/analysis/sentiment/dl/curves")
def api_sentiment_dl_curves():
    curves_dir = cfg.SENTIMENT_CURVES_DIR
    result = {}
    for fn in ["dl_2cls_history.json", "dl_5cls_history.json", "dl_train_metrics.json", "ml_train_metrics.json"]:
        fp = os.path.join(curves_dir, fn)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                result[fn.replace(".json", "")] = json.load(f)
    return jsonify(result)


# ==================== 推荐系统 API ====================

@analysis_bp.route("/api/analysis/recommend/train")
def api_recommend_train():
    model = request.args.get("model", "tfidf", type=str).lower()
    try:
        from utils.bigdata_recommend_engine import train_tfidf, train_svd, train_sbert

        if model == "svd":
            train_svd(force=True)
            return jsonify({"status": "success", "model": "svd"})
        elif model == "sbert":
            train_sbert(force=True)
            return jsonify({"status": "success", "model": "sbert"})
        else:
            train_tfidf(force=True)
            return jsonify({"status": "success", "model": "tfidf"})
    except Exception as e:
        return jsonify({"error": str(e)})


@analysis_bp.route("/api/analysis/recommend/by_movie")
def api_recommend_by_movie():
    movie = request.args.get("movie", "我不是药神")
    model = request.args.get("model", "tfidf")

    if model == "tfidf":
        return _recommend_tfidf(movie)
    if model == "svd":
        return _recommend_svd(movie)
    if model == "sbert":
        return _recommend_sbert(movie)

    return _recommend_tfidf(movie)


def _recommend_tfidf(movie_name):
    from utils.bigdata_recommend_engine import train_tfidf
    from utils.stopwords import get_recommend_stop_words
    import jieba

    data = train_tfidf()
    vectorizer = pickle.loads(data["vectorizer"])
    nn_index = pickle.loads(data["nn_index"])
    movies = data["movies"]

    query_lower = movie_name.lower()
    best_idx = -1
    best_score = 0
    for i, m in enumerate(movies):
        name_lower = m["title"].lower()
        common = sum(1 for c in query_lower if c in name_lower)
        score = common / max(len(query_lower), 1)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score < 0.3:
        best_idx = -1
        for i, m in enumerate(movies):
            if query_lower in m["title"].lower() or m["title"].lower() in query_lower:
                best_idx = i
                break
    if best_idx < 0:
        return jsonify({"query": movie_name, "results": [], "model": "tfidf",
                        "message": f"未找到包含「{movie_name}」的电影"})

    src = movies[best_idx]
    stop_words = get_recommend_stop_words()
    query_text = f"{src['genres']} {src['directors']} {src['actors']} {src['summary'][:200]} {src['tags']}"
    words = jieba.lcut(query_text)
    query_vec = vectorizer.transform([" ".join(w for w in words if len(w) >= 2 and w not in stop_words)])
    distances, neighbors = nn_index.kneighbors(query_vec, n_neighbors=11)

    results = []
    seen = {best_idx}
    for _, nb_idx in enumerate(neighbors[0]):
        if nb_idx in seen:
            continue
        seen.add(nb_idx)
        dist = float(distances[0][list(neighbors[0]).index(nb_idx)])
        m = movies[nb_idx]
        results.append({
            "name": m["title"],
            "similarity": round(float(1.0 / (1.0 + dist)), 3),
            "reason": m["genres"][:30],
            "rating": float(m["rating"]),
            "votes": int(m["votes"]),
        })

    results.sort(key=lambda x: x["similarity"] * np.log(1 + max(0, x["votes"])) * max(0.1, x["rating"]) / 10.0,
                 reverse=True)
    return jsonify({"query": movie_name, "results": results[:10],
                    "model": "tfidf", "movie_count": len(movies)})


def _recommend_svd(movie_name):
    from utils.bigdata_recommend_engine import train_svd, train_tfidf

    data = train_svd()
    tfidf_data = train_tfidf()
    tfidf_movies = tfidf_data["movies"]

    nn_index = pickle.loads(data["nn_index"])
    svd_movies = data["movies"]
    i2idx = data["item_to_idx"]
    item_vectors = np.load(cfg.CACHE_SVD_NPZ)["item_vectors"]

    query_lower = movie_name.lower()
    id_to_movie = {m["movie_id"]: m for m in tfidf_movies}
    best_mid = None
    for m in tfidf_movies:
        if query_lower in m["title"].lower() or m["title"].lower() in query_lower:
            best_mid = m["movie_id"]
            break
    if best_mid is None:
        return jsonify({"query": movie_name, "results": [], "model": "svd"})
    if best_mid not in i2idx:
        title_lower = movie_name.lower()
        for real_id, idx in i2idx.items():
            if real_id in id_to_movie and title_lower in id_to_movie[real_id]["title"].lower():
                best_mid = real_id
                break
    if best_mid is None or best_mid not in i2idx:
        return jsonify({"query": movie_name, "results": [], "model": "svd",
                        "reason": "该电影不在 SVD 训练集中（互动不足）"})

    item_idx = i2idx[best_mid]
    query_vec = item_vectors[item_idx:item_idx + 1]
    distances, neighbors = nn_index.kneighbors(query_vec, n_neighbors=11)

    svd_id_to_movie = {m["movie_id"]: m for m in svd_movies}
    results = []
    seen = {item_idx}
    for nb_idx in neighbors[0]:
        if nb_idx in seen or nb_idx == item_idx:
            continue
        seen.add(nb_idx)
        real_id = data["idx_to_item"].get(nb_idx)
        if real_id and real_id in svd_id_to_movie:
            dist = float(distances[0][list(neighbors[0]).index(nb_idx)])
            m = svd_id_to_movie[real_id]
            results.append({
                "name": m["title"],
                "similarity": round(float(1.0 / (1.0 + dist)), 3),
                "reason": m["genres"][:30],
                "rating": float(m["rating"]),
                "votes": int(m["votes"]),
            })

    results.sort(key=lambda x: x["similarity"] * np.log(1 + max(0, x["votes"])) * max(0.1, x["rating"]) / 10.0, reverse=True)
    return jsonify({"query": movie_name, "results": results[:10],
                    "model": "svd", "movie_count": len(svd_movies)})


def _recommend_sbert(movie_name):
    from utils.bigdata_recommend_engine import train_sbert

    data = train_sbert()
    if data is None:
        return jsonify({"error": "BGE-Large 模型不可用，请安装 sentence-transformers"})

    nn_index = pickle.loads(data["nn_index"])
    movies = data["movies"]
    embeddings = data["embeddings"]

    query_lower = movie_name.lower()
    best_idx = -1
    for i, m in enumerate(movies):
        if query_lower in m["title"].lower() or m["title"].lower() in query_lower:
            best_idx = i
            break
    if best_idx < 0:
        return jsonify({"query": movie_name, "results": [], "model": "sbert",
                        "message": f"未找到包含「{movie_name}」的电影"})

    query_emb = embeddings[best_idx:best_idx + 1]
    distances, neighbors = nn_index.kneighbors(query_emb, n_neighbors=11)

    results = []
    seen = {best_idx}
    for nb_idx in neighbors[0]:
        if nb_idx in seen:
            continue
        seen.add(nb_idx)
        dist = float(distances[0][list(neighbors[0]).index(nb_idx)])
        m = movies[nb_idx]
        results.append({
            "name": m["title"],
            "similarity": round(float(1.0 / (1.0 + dist)), 3),
            "reason": m["genres"][:30],
            "rating": float(m["rating"]),
            "votes": int(m["votes"]),
        })

    results.sort(key=lambda x: x["similarity"] * np.log(1 + max(0, x["votes"])) * max(0.1, x["rating"]) / 10.0,
                 reverse=True)
    return jsonify({"query": movie_name, "results": results[:10],
                    "model": "sbert", "movie_count": len(movies)})


@analysis_bp.route("/api/analysis/recommend/eval")
def api_recommend_eval():
    eval_path = cfg.EVAL_REPORT_PATH
    if not os.path.exists(eval_path):
        try:
            from utils.bigdata_recommend_engine import generate_eval_report
            generate_eval_report()
        except Exception as e:
            return jsonify({"error": f"评估报告生成失败: {e}"})
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "评估报告不存在，请先训练模型"})


@analysis_bp.route("/api/movie/detail")
def api_movie_detail():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "请提供电影名称"})
    try:
        from utils.bigdata_recommend_engine import _load_unified_movies
        movies = _load_unified_movies()
        name_lower = name.lower()
        best = None
        for m in movies:
            if name_lower in m["title"].lower() or m["title"].lower() in name_lower:
                best = m
                break
        if not best:
            return jsonify({"error": f"未找到电影「{name}」"})
        return jsonify({
            "title": best["title"],
            "year": best["year"],
            "rating": best["rating"],
            "votes": best["votes"],
            "directors": best["directors"],
            "actors": best["actors"],
            "genres": best["genres"],
            "country": best["country"],
            "languages": best["languages"],
            "summary": best["summary"],
            "tags": best["tags"],
            "movie_id": best["movie_id"],
            "source": best["source"],
        })
    except Exception as e:
        return jsonify({"error": f"查询失败: {e}"})


@analysis_bp.route("/api/analysis/export")
def api_export_results():
    _ensure_warmup()
    export = {}

    if _network_cache and "status" not in _network_cache:
        from utils.db_manager import DBManager
        na = NetworkAnalyzer(DBManager(cfg.BIGDATA_DB_PATH))
        na.build()
        export.update(na.export_results())

    if _sentiment_cache and "status" not in _sentiment_cache:
        from utils.db_manager import DBManager
        sa = SentimentAnalyzer(DBManager(cfg.BIGDATA_DB_PATH))
        export.update(sa.export_results())

    try:
        from utils.bigdata_recommend_engine import train_tfidf
        td = train_tfidf()
        export.update({
            "recommendation_system": {
                "电影库规模": td["matrix_shape"][0],
                "特征维度": td.get("feature_count", 0),
                "strategy": "TF-IDF 双数据库融合内容推荐 · 27,304 部电影",
            }
        })
    except Exception:
        pass

    return jsonify({"status": "ok", "summary": {
        k: list(v.keys()) for k, v in export.items()
    }})
