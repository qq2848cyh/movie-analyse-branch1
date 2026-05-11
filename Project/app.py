"""
电影数据分析平台 — Flask 主应用

路由模块划分:
  - 首页 & 导航
  - 精选电影（movie / score / wordcloud）
  - 精选电影图表 API
  - 智能分析（network / sentiment / recommend）
  - 大数据集（movie / score / wordcloud）
  - 大数据集图表 API
  - 大数据集词云 API
"""

import os
os.environ["HF_HUB_CACHE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "hf_cache"
)

import json
import threading
import pickle
import numpy as np
from flask import Flask, render_template, jsonify, request

from config import BIGDATA_DB_PATH, NEW_MOVIES_DB_PATH, NEW_MOVIES_CSV
from utils.db_manager import DBManager
from utils.new_movie_manager import NewMovieManager
from utils.echarts_visualizer import EChartsVisualizer
from utils.network_analyzer import NetworkAnalyzer
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.sentiment_dl import train_bilstm_attn, get_attention_visualization

# ==================== 路径初始化 ====================

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

app = Flask(
    __name__,
    static_folder=os.path.join(root_dir, "static"),
    template_folder=os.path.join(current_dir, "templates"),
)


# ==================== 工具函数 ====================

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


def _paginate(page, total, per_page=20):
    """计算分页参数"""
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start_index = (page - 1) * per_page + 1
    end_index = min(start_index + per_page - 1, total)
    return page, total_pages, start_index, end_index


def _chart_api(db_getter, chart_method):
    """通用图表 API 工厂 — 消除重复的模式代码"""
    db = db_getter()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    method = getattr(visualizer, chart_method)
    return jsonify(method(df))


# ==================== 数据库懒加载 ====================

_bigdata_db = None
_new_movies_db = None


def get_bigdata_db():
    """获取大数据集数据库单例"""
    global _bigdata_db
    if _bigdata_db is None:
        _bigdata_db = DBManager(os.path.join(root_dir, BIGDATA_DB_PATH))
    return _bigdata_db


def get_new_movies_db():
    """获取精选电影数据库单例（首次调用时自动从 CSV 导入）"""
    global _new_movies_db
    if _new_movies_db is None:
        db_path = os.path.join(root_dir, NEW_MOVIES_DB_PATH)
        csv_path = os.path.join(root_dir, NEW_MOVIES_CSV)
        _new_movies_db = NewMovieManager(db_path)
        if _new_movies_db.get_total_count() == 0 and os.path.exists(csv_path):
            count = _new_movies_db.import_from_csv(csv_path)
            print(f"[new_movies] 自动导入 {count} 条记录", flush=True)
    return _new_movies_db


# ==================== 首页 & 导航 ====================

@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return welcome()


@app.route("/aboutMe")
def about_me():
    return render_template("aboutMe.html")


# ==================== 精选电影模块 ====================

@app.route("/movie")
def movie():
    db = get_new_movies_db()
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, search=search)
    page, total_pages, start, end = _paginate(page, total)

    return render_template("movie.html",
        movies=movies, page=page, total_pages=total_pages,
        total_movies=total, start_index=start, end_index=end, search=search)


@app.route("/score")
def score():
    db = get_new_movies_db()
    return render_template("score.html", movies_count=db.get_total_count())


@app.route("/word")
def word():
    db = get_new_movies_db()
    return render_template("cloud.html", movies_count=db.get_total_count())


# ==================== 精选电影图表 API ====================

CACHE_NEW_CHARTS = os.path.join(root_dir, "data", "cache", "new_charts.pkl")


def _load_charts_cache(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_charts_cache(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


@app.route("/api/charts/rating")
def api_rating_chart():
    return _chart_api(get_new_movies_db, "prepare_rating_distribution")


@app.route("/api/charts/year")
def api_year_chart():
    return _chart_api(get_new_movies_db, "prepare_year_distribution")


@app.route("/api/charts/country")
def api_country_chart():
    return _chart_api(get_new_movies_db, "prepare_country_distribution")


@app.route("/api/charts/genre")
def api_genre_chart():
    return _chart_api(get_new_movies_db, "prepare_genre_distribution")


@app.route("/api/charts/director")
def api_director_chart():
    return _chart_api(get_new_movies_db, "prepare_director_ranking")


@app.route("/api/charts/all")
def api_all_charts():
    refresh = request.args.get("refresh", "0", type=str)
    if refresh != "1":
        cached = _load_charts_cache(CACHE_NEW_CHARTS)
        if cached is not None:
            return jsonify(cached)

    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    data = visualizer.generate_all_charts_data(df)
    _save_charts_cache(CACHE_NEW_CHARTS, data)
    return jsonify(data)


@app.route("/api/wordcloud/data")
def api_wordcloud_data():
    db = get_new_movies_db()
    refresh = request.args.get("refresh", "0", type=str)
    result = db.get_wordcloud_data(force_refresh=(refresh == "1"))
    return jsonify(result)


# ==================== 智能分析模块 ====================

CACHE_DIR = os.path.join(root_dir, "data", "cache")
CACHE_NETWORK = os.path.join(CACHE_DIR, "network", "network_stats.pkl")
CACHE_SENTIMENT = os.path.join(CACHE_DIR, "sentiment", "sentiment_results.pkl")
CACHE_SENTIMENT_DL = os.path.join(CACHE_DIR, "sentiment", "sentiment_dl_results.pkl")

for _cache_path in (CACHE_NETWORK, CACHE_SENTIMENT, CACHE_SENTIMENT_DL):
    os.makedirs(os.path.dirname(_cache_path), exist_ok=True)

_network_cache = None
_sentiment_cache = None
_sentiment_dl_cache = None
_warmup_started = False
_warmup_lock = threading.Lock()


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

    db = get_bigdata_db()
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


# --- 智能分析页面路由 ---

@app.route("/analysis/network")
def analysis_network():
    _ensure_warmup()
    return render_template("analysis_network.html")


@app.route("/analysis/sentiment")
def analysis_sentiment():
    return render_template("analysis_sentiment.html")


@app.route("/analysis/recommend")
def analysis_recommend():
    return render_template("analysis_recommend.html")


# --- 智能分析 API 路由 ---

@app.route("/api/analysis/network/stats")
def api_network_stats():
    if _network_cache is not None:
        return jsonify(_sanitize_for_json(_network_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "网络分析数据正在预热中，请稍后刷新"})


@app.route("/api/analysis/sentiment/all")
def api_sentiment_all():
    if _sentiment_cache is not None:
        return jsonify(_sanitize_for_json(_sentiment_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "情感分析数据正在预热中，请稍后刷新"})


@app.route("/api/analysis/sentiment/dl")
def api_sentiment_dl():
    if _sentiment_dl_cache is not None:
        return jsonify(_sanitize_for_json(_sentiment_dl_cache))
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "深度学习模型正在训练中，请稍后刷新"})


@app.route("/api/analysis/sentiment/dl/attention")
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


@app.route("/api/analysis/sentiment/dl/curves")
def api_sentiment_dl_curves():
    curves_dir = os.path.join(CACHE_DIR, "sentiment", "curves")
    result = {}
    for fn in ["dl_2cls_history.json", "dl_5cls_history.json", "dl_train_metrics.json", "ml_train_metrics.json"]:
        fp = os.path.join(curves_dir, fn)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                result[fn.replace(".json", "")] = json.load(f)
    return jsonify(result)


@app.route("/api/analysis/recommend/by_movie")
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
    from utils.bigdata_recommend_engine import train_tfidf, _get_stop_words
    import jieba
    import numpy as np
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
    stop_words = _get_stop_words()
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
    import numpy as np

    data = train_svd()
    tfidf_data = train_tfidf()
    tfidf_movies = tfidf_data["movies"]

    nn_index = pickle.loads(data["nn_index"])
    svd_movies = data["movies"]
    items = data["svd_items"]
    i2idx = data["item_to_idx"]
    item_vectors = np.load(os.path.join(root_dir, "data", "cache", "recommend", "svd_item_vectors.npz"))["item_vectors"]

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
            w = np.log(1 + max(0, m["votes"])) * max(0.1, m["rating"]) / 10.0
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
    import numpy as np
    from utils.bigdata_recommend_engine import train_sbert
    data = train_sbert()
    nn_index = pickle.loads(data["nn_index"])
    movies = data["movies"]
    embeddings = np.load(os.path.join(root_dir, "data", "cache", "recommend", "sbert_embeddings.npy"))

    query_lower = movie_name.lower()
    best_idx = -1
    for i, m in enumerate(movies):
        if query_lower in m["title"].lower() or m["title"].lower() in query_lower:
            best_idx = i
            break
    if best_idx < 0:
        return jsonify({"query": movie_name, "results": [], "model": "sbert",
                        "message": f"未找到包含「{movie_name}」的电影"})

    query_emb = embeddings[best_idx:best_idx+1]
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



@app.route("/api/analysis/recommend/eval")
def api_recommend_eval():
    eval_path = os.path.join(root_dir, "data", "cache", "recommend", "recommend_eval.json")
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


@app.route("/api/movie/detail")
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


@app.route("/api/analysis/export")
def api_export_results():
    _ensure_warmup()
    export = {}

    if _network_cache and "status" not in _network_cache:
        na = NetworkAnalyzer(get_bigdata_db())
        na.build()
        export.update(na.export_results())

    if _sentiment_cache and "status" not in _sentiment_cache:
        sa = SentimentAnalyzer(get_bigdata_db())
        export.update(sa.export_results())

    from utils.bigdata_recommend_engine import train_tfidf
    td = train_tfidf()
    export.update({
        "recommendation_system": {
            "电影库规模": td["matrix_shape"][0],
            "特征维度": td.get("feature_count", 0),
            "strategy": "TF-IDF 双数据库融合内容推荐 · 27,304 部电影",
        }
    })

    return jsonify({"status": "ok", "summary": {
        k: list(v.keys()) for k, v in export.items()
    }})


# ==================== 大数据集模块 ====================

@app.route("/bigdata/index")
def bigdata_index():
    return render_template("index_bigdata.html")


@app.route("/bigdata/movie")
def bigdata_movie():
    db = get_bigdata_db()
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, search=search)
    page, total_pages, start, end = _paginate(page, total)

    return render_template("movie_bigdata.html",
        movies=movies, page=page, total_pages=total_pages,
        total_movies=total, start_index=start, end_index=end, search=search)


@app.route("/bigdata/score")
def bigdata_score():
    db = get_bigdata_db()
    return render_template("score_bigdata.html", movies_count=db.get_total_count())


@app.route("/bigdata/word")
def bigdata_word():
    db = get_bigdata_db()
    return render_template("cloud_bigdata.html", movies_count=db.get_total_count())


# ==================== 大数据集图表 API ====================

CACHE_BIGDATA_CHARTS = os.path.join(root_dir, "data", "cache", "bigdata_charts.pkl")


@app.route("/api/bigdata/charts/rating")
def api_bigdata_rating_chart():
    return _chart_api(get_bigdata_db, "prepare_rating_distribution")


@app.route("/api/bigdata/charts/year")
def api_bigdata_year_chart():
    return _chart_api(get_bigdata_db, "prepare_year_distribution")


@app.route("/api/bigdata/charts/country")
def api_bigdata_country_chart():
    return _chart_api(get_bigdata_db, "prepare_country_distribution")


@app.route("/api/bigdata/charts/genre")
def api_bigdata_genre_chart():
    return _chart_api(get_bigdata_db, "prepare_genre_distribution")


@app.route("/api/bigdata/charts/director")
def api_bigdata_director_chart():
    return _chart_api(get_bigdata_db, "prepare_director_ranking")


@app.route("/api/bigdata/charts/all")
def api_bigdata_all_charts():
    refresh = request.args.get("refresh", "0", type=str)
    if refresh != "1":
        cached = _load_charts_cache(CACHE_BIGDATA_CHARTS)
        if cached is not None:
            return jsonify(cached)

    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    data = visualizer.generate_all_charts_data(df)
    _save_charts_cache(CACHE_BIGDATA_CHARTS, data)
    return jsonify(data)


@app.route("/api/bigdata/wordcloud/data")
def api_bigdata_wordcloud_data():
    db = get_bigdata_db()
    refresh = request.args.get("refresh", "0", type=str)
    result = db.get_wordcloud_data(force_refresh=(refresh == "1"))
    return jsonify(result)


# ==================== 启动入口 ====================

if __name__ == "__main__":
    app.run(debug=True)
