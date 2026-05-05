import sys
import os
import threading
import pickle
from flask import Flask, render_template, jsonify, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BIGDATA_DB_PATH, NEW_MOVIES_DB_PATH, NEW_MOVIES_CSV
from utils.echarts_visualizer import EChartsVisualizer
from utils.db_manager import DBManager
from utils.new_movie_manager import NewMovieManager
from utils.network_analyzer import NetworkAnalyzer
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.recommend_engine import RecommendEngine

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

app = Flask(
    __name__,
    static_folder=os.path.join(root_dir, "static"),
    template_folder=os.path.join(current_dir, "templates"),
)

_bigdata_db = None
_new_movies_db = None


def get_bigdata_db():
    global _bigdata_db
    if _bigdata_db is None:
        _bigdata_db = DBManager(os.path.join(root_dir, BIGDATA_DB_PATH))
    return _bigdata_db


def get_new_movies_db():
    global _new_movies_db
    if _new_movies_db is None:
        db_path = os.path.join(root_dir, NEW_MOVIES_DB_PATH)
        csv_path = os.path.join(root_dir, NEW_MOVIES_CSV)
        _new_movies_db = NewMovieManager(db_path)
        if _new_movies_db.get_total_count() == 0 and os.path.exists(csv_path):
            count = _new_movies_db.import_from_csv(csv_path)
            print(f"[new_movies] 自动导入 {count} 条记录", flush=True)
    return _new_movies_db


@app.route("/")
def welcome():
    """新首页：功能选择界面"""
    return render_template("welcome.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return welcome()


@app.route("/movie")
def movie():
    db = get_new_movies_db()
    per_page = 20
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, per_page=per_page, search=search)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start_index = (page - 1) * per_page + 1
    end_index = min(start_index + per_page - 1, total)

    return render_template(
        "movie.html",
        movies=movies,
        page=page,
        total_pages=total_pages,
        total_movies=total,
        start_index=start_index,
        end_index=end_index,
        search=search,
    )


@app.route("/score")
def score():
    db = get_new_movies_db()
    total = db.get_total_count()
    return render_template("score.html", movies_count=total)


@app.route("/api/charts/rating")
def api_rating_chart():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_rating_distribution(df)
    return jsonify(chart_data)


@app.route("/api/charts/year")
def api_year_chart():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_year_distribution(df)
    return jsonify(chart_data)


@app.route("/api/charts/country")
def api_country_chart():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_country_distribution(df)
    return jsonify(chart_data)


@app.route("/api/charts/genre")
def api_genre_chart():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_genre_distribution(df)
    return jsonify(chart_data)


@app.route("/api/charts/director")
def api_director_chart():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_director_ranking(df)
    return jsonify(chart_data)


@app.route("/api/charts/all")
def api_all_charts():
    db = get_new_movies_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    all_charts = visualizer.generate_all_charts_data(df)
    return jsonify(all_charts)


@app.route("/word")
def word():
    db = get_new_movies_db()
    total = db.get_total_count()
    return render_template("cloud.html", movies_count=total)


@app.route("/api/wordcloud/data")
def api_wordcloud_data():
    import jieba

    db = get_new_movies_db()
    titles, summaries = db.get_wordcloud_texts()

    if not titles and not summaries:
        return jsonify({"error": "暂无数据"})

    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "说", "要", "你", "会", "着", "没有", "看",
        "好", "自己", "这", "他", "她", "们", "那", "什么", "怎么", "可以",
        "还", "被", "让", "从", "与", "但", "而", "等", "及", "之", "为",
        "对", "于", "以", "电影", "一部", "真的", "感觉", "觉得", "还是", "不过",
        "已经", "不是", "就是", "这么", "那么", "一直", "一点", "很多",
        "出来", "开始", "最后", "比较", "其实", "有点", "因为", "所以",
        "我们", "他们", "你们", "自己", "大家", "但是", "然而",
    }

    def count_chinese_words(texts, top_n=50):
        freq = {}
        for text in texts:
            words = jieba.lcut(str(text))
            for w in words:
                w = w.strip("，。！？；：""''（）【】《》、…—·,.!?;:()[]{}<>\"'|/\\@#$%^&*+=~` \t")
                if len(w) >= 2 and any('\u4e00' <= c <= '\u9fff' for c in w):
                    if w not in stop_words:
                        freq[w] = freq.get(w, 0) + 1
        return [{"name": w, "value": c}
                for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    return jsonify({
        "titles": count_chinese_words(titles),
        "summaries": count_chinese_words(summaries),
    })


# ==================== 智能分析模块路由 ====================

_network_cache = None
_sentiment_cache = None
_recommend_engine = None
_warmup_started = False
_warmup_lock = threading.Lock()

CACHE_DIR = os.path.join(root_dir, "data", "cache")
CACHE_NETWORK = os.path.join(CACHE_DIR, "network_stats.pkl")
CACHE_SENTIMENT = os.path.join(CACHE_DIR, "sentiment_results.pkl")


def _load_cache(path, name):
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


def _save_cache(path, data, name):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache] {name} 已持久化", flush=True)
    except Exception as e:
        print(f"[cache] {name} 保存失败: {e}", flush=True)


def _warmup_thread():
    global _network_cache, _sentiment_cache, _recommend_engine

    _network_cache = _load_cache(CACHE_NETWORK, "网络分析")
    _sentiment_cache = _load_cache(CACHE_SENTIMENT, "情感分析")

    db = get_bigdata_db()
    need_network = _network_cache is None
    need_sentiment = _sentiment_cache is None

    try:
        if need_network or need_sentiment or _recommend_engine is None:
            print("[warmup] 开始后台计算...", flush=True)

        if need_network:
            print("[warmup] 构建协作网络...", flush=True)
            na = NetworkAnalyzer(db)
            na.build()
            _network_cache = na.get_all_stats()
            _save_cache(CACHE_NETWORK, _network_cache, "网络分析")
            print("[warmup] 协作网络分析完成 ✓", flush=True)

        if need_sentiment:
            print("[warmup] 训练情感分析模型...", flush=True)
            sa = SentimentAnalyzer(db)
            _sentiment_cache = sa.get_all_results(n_samples=10000)
            _save_cache(CACHE_SENTIMENT, _sentiment_cache, "情感分析")
            print("[warmup] 情感分析完成 ✓", flush=True)

        if _recommend_engine is None:
            print("[warmup] 初始化推荐引擎...", flush=True)
            re_obj = RecommendEngine(db, content_db_manager=get_new_movies_db())
            re_obj.train_content()
            _recommend_engine = re_obj
            print("[warmup] 推荐引擎就绪 ✓", flush=True)

        if not need_network and not need_sentiment and _recommend_engine is not None:
            print("[warmup] 全部从缓存加载，无需计算 ✓", flush=True)
        else:
            print("[warmup] 全部模块预热完成!", flush=True)
    except Exception as e:
        import traceback
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[warmup] 预热失败: {err_msg}", flush=True)
        if _network_cache is None:
            _network_cache = {"status": "error", "message": str(e)}
        if _sentiment_cache is None:
            _sentiment_cache = {"status": "error", "message": str(e)}


def _ensure_warmup():
    global _warmup_started
    if not _warmup_started:
        with _warmup_lock:
            if not _warmup_started:
                _warmup_started = True
                t = threading.Thread(target=_warmup_thread)
                t.start()


@app.route("/analysis/network")
def analysis_network():
    _ensure_warmup()
    return render_template("analysis_network.html")


@app.route("/api/analysis/network/stats")
def api_network_stats():
    if _network_cache is not None:
        return jsonify(_network_cache)
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "网络分析数据正在预热中，请稍后刷新"})


@app.route("/api/analysis/sentiment/all")
def api_sentiment_all():
    if _sentiment_cache is not None:
        return jsonify(_sentiment_cache)
    _ensure_warmup()
    return jsonify({"status": "warming", "message": "情感分析数据正在预热中，请稍后刷新"})


@app.route("/api/analysis/recommend/by_movie")
def api_recommend_by_movie():
    _ensure_warmup()
    movie = request.args.get("movie", "我不是药神")
    if _recommend_engine is None:
        return jsonify({"status": "warming", "message": "推荐引擎正在预热中"})
    results = _recommend_engine.recommend_by_movie(movie)
    return jsonify({"query": movie, "results": results})


@app.route("/api/analysis/recommend/train")
def api_recommend_train():
    re_obj = get_recommend_engine()
    metrics = re_obj.train_cf()
    re_obj.train_content()
    return jsonify({"status": "ok", "cf_metrics": metrics})


@app.route("/analysis/sentiment")
def analysis_sentiment():
    return render_template("analysis_sentiment.html")


@app.route("/analysis/recommend")
def analysis_recommend():
    return render_template("analysis_recommend.html")


@app.route("/team")
def team():
    return render_template("team.html")


@app.route("/aboutMe")
def aboutMe():
    return render_template("aboutMe.html")


# ==================== 大数据电影集路由 ====================

@app.route("/bigdata/index")
def bigdata_index():
    return render_template("index_bigdata.html")


@app.route("/bigdata/movie")
def bigdata_movie():
    db = get_bigdata_db()
    per_page = 20
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, per_page=per_page, search=search)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start_index = (page - 1) * per_page + 1
    end_index = min(start_index + per_page - 1, total)

    return render_template(
        "movie_bigdata.html",
        movies=movies,
        page=page,
        total_pages=total_pages,
        total_movies=total,
        start_index=start_index,
        end_index=end_index,
        search=search,
    )


@app.route("/bigdata/score")
def bigdata_score():
    db = get_bigdata_db()
    total = db.get_total_count()
    return render_template("score_bigdata.html", movies_count=total)


@app.route("/bigdata/word")
def bigdata_word():
    db = get_bigdata_db()
    total = db.get_total_count()
    return render_template("cloud_bigdata.html", movies_count=total)


# ==================== 大数据电影集 API 路由 ====================

@app.route("/api/bigdata/charts/rating")
def api_bigdata_rating_chart():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_rating_distribution(df)
    return jsonify(chart_data)


@app.route("/api/bigdata/charts/year")
def api_bigdata_year_chart():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_year_distribution(df)
    return jsonify(chart_data)


@app.route("/api/bigdata/charts/country")
def api_bigdata_country_chart():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_country_distribution(df)
    return jsonify(chart_data)


@app.route("/api/bigdata/charts/genre")
def api_bigdata_genre_chart():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_genre_distribution(df)
    return jsonify(chart_data)


@app.route("/api/bigdata/charts/director")
def api_bigdata_director_chart():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    chart_data = visualizer.prepare_director_ranking(df)
    return jsonify(chart_data)


@app.route("/api/bigdata/charts/all")
def api_bigdata_all_charts():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    all_charts = visualizer.generate_all_charts_data(df)
    return jsonify(all_charts)


@app.route("/api/bigdata/wordcloud/data")
def api_bigdata_wordcloud_data():
    db = get_bigdata_db()
    result = db.get_wordcloud_data()
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
