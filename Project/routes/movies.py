"""精选电影路由模块"""

from flask import Blueprint, render_template, jsonify, request
import pickle

import config as cfg
from utils.db_manager import DBManager
from utils.new_movie_manager import NewMovieManager
from utils.echarts_visualizer import EChartsVisualizer

movies_bp = Blueprint("movies", __name__)

# 全局缓存
CACHE_NEW_CHARTS = cfg.CHARTS_NEW_CACHE

# 数据库实例
_new_movies_db = None


def get_new_movies_db():
    """获取精选电影数据库单例"""
    global _new_movies_db
    if _new_movies_db is None:
        _new_movies_db = NewMovieManager(cfg.NEW_MOVIES_DB_PATH)
        if _new_movies_db.get_total_count() == 0 and os.path.exists(cfg.NEW_MOVIES_CSV):
            count = _new_movies_db.import_from_csv(cfg.NEW_MOVIES_CSV)
            print(f"[new_movies] 自动导入 {count} 条记录", flush=True)
    return _new_movies_db


def _sanitize_for_json(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    import numpy as np
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


def _load_charts_cache(path):
    """加载图表缓存"""
    import os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_charts_cache(path, data):
    """保存图表缓存"""
    import os
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _chart_api(db_getter, chart_method):
    """通用图表 API 工厂"""
    db = db_getter()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    method = getattr(visualizer, chart_method)
    result = method(df)
    return jsonify(_sanitize_for_json(result))


import os


# ==================== 精选电影页面路由 ====================

@movies_bp.route("/movie")
def movie():
    db = get_new_movies_db()
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, search=search)
    page, total_pages, start, end = _paginate(page, total)

    return render_template("movie.html",
                          movies=movies, page=page, total_pages=total_pages,
                          total_movies=total, start_index=start, end_index=end, search=search)


@movies_bp.route("/score")
def score():
    db = get_new_movies_db()
    return render_template("score.html", movies_count=db.get_total_count())


@movies_bp.route("/word")
def word():
    db = get_new_movies_db()
    return render_template("cloud.html", movies_count=db.get_total_count())


# ==================== 精选电影图表 API ====================

@movies_bp.route("/api/charts/rating")
def api_rating_chart():
    return _chart_api(get_new_movies_db, "prepare_rating_distribution")


@movies_bp.route("/api/charts/year")
def api_year_chart():
    return _chart_api(get_new_movies_db, "prepare_year_distribution")


@movies_bp.route("/api/charts/country")
def api_country_chart():
    return _chart_api(get_new_movies_db, "prepare_country_distribution")


@movies_bp.route("/api/charts/genre")
def api_genre_chart():
    return _chart_api(get_new_movies_db, "prepare_genre_distribution")


@movies_bp.route("/api/charts/director")
def api_director_chart():
    return _chart_api(get_new_movies_db, "prepare_director_ranking")


@movies_bp.route("/api/charts/all")
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


@movies_bp.route("/api/wordcloud/data")
def api_wordcloud_data():
    db = get_new_movies_db()
    refresh = request.args.get("refresh", "0", type=str)
    result = db.get_wordcloud_data(force_refresh=(refresh == "1"))
    return jsonify(result)
