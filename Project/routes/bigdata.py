"""大数据集路由模块"""

from flask import Blueprint, render_template, jsonify, request
import pickle

import config as cfg
from utils.db_manager import DBManager
from utils.echarts_visualizer import EChartsVisualizer

bigdata_bp = Blueprint("bigdata", __name__)

# 缓存路径
CACHE_BIGDATA_CHARTS = cfg.CHARTS_BIGDATA_CACHE

# 数据库实例
_bigdata_db = None


def get_bigdata_db():
    """获取大数据集数据库单例"""
    global _bigdata_db
    if _bigdata_db is None:
        _bigdata_db = DBManager(cfg.BIGDATA_DB_PATH)
    return _bigdata_db


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


import os


# ==================== 大数据集页面路由 ====================

@bigdata_bp.route("/bigdata/index")
def bigdata_index():
    return render_template("index_bigdata.html")


@bigdata_bp.route("/bigdata/movie")
def bigdata_movie():
    db = get_bigdata_db()
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)

    movies, total = db.get_movies_page(page=page, search=search)
    total_pages = max(1, (total + 20 - 1) // 20)
    page = max(1, min(page, total_pages))
    start_index = (page - 1) * 20 + 1
    end_index = min(start_index + 20 - 1, total)

    return render_template("movie_bigdata.html",
                          movies=movies, page=page, total_pages=total_pages,
                          total_movies=total, start_index=start_index, end_index=end_index, search=search)


@bigdata_bp.route("/bigdata/score")
def bigdata_score():
    db = get_bigdata_db()
    return render_template("score_bigdata.html", movies_count=db.get_total_count())


@bigdata_bp.route("/bigdata/word")
def bigdata_word():
    db = get_bigdata_db()
    return render_template("cloud_bigdata.html", movies_count=db.get_total_count())


# ==================== 大数据集图表 API ====================

@bigdata_bp.route("/api/bigdata/charts/all")
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


@bigdata_bp.route("/api/bigdata/charts/rating")
def api_bigdata_rating():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    result = visualizer.prepare_rating_distribution(df)
    return jsonify(_sanitize_for_json(result))


@bigdata_bp.route("/api/bigdata/charts/year")
def api_bigdata_year():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    result = visualizer.prepare_year_distribution(df)
    return jsonify(_sanitize_for_json(result))


@bigdata_bp.route("/api/bigdata/charts/country")
def api_bigdata_country():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    result = visualizer.prepare_country_distribution(df)
    return jsonify(_sanitize_for_json(result))


@bigdata_bp.route("/api/bigdata/charts/genre")
def api_bigdata_genre():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    result = visualizer.prepare_genre_distribution(df)
    return jsonify(_sanitize_for_json(result))


@bigdata_bp.route("/api/bigdata/charts/director")
def api_bigdata_director():
    db = get_bigdata_db()
    df = db.get_analysis_dataframe()
    if df.empty:
        return jsonify({"error": "暂无数据"})
    visualizer = EChartsVisualizer()
    result = visualizer.prepare_director_ranking(df)
    return jsonify(_sanitize_for_json(result))


# ==================== 大数据集词云 API ====================

@bigdata_bp.route("/api/bigdata/wordcloud/data")
def api_bigdata_wordcloud_data():
    db = get_bigdata_db()
    refresh = request.args.get("refresh", "0", type=str)
    result = db.get_wordcloud_data(force_refresh=(refresh == "1"))
    return jsonify(result)


@bigdata_bp.route("/api/bigdata/wordcloud/comments")
def api_bigdata_wordcloud_comments():
    db = get_bigdata_db()
    refresh = request.args.get("refresh", "0", type=str)
    result = db.compute_wordcloud_data(force_refresh=(refresh == "1"))
    return jsonify(result)


# ==================== 数据库管理 API ====================

@bigdata_bp.route("/api/bigdata/import")
def api_bigdata_import():
    force = request.args.get("force", "0", type=str)
    if force != "1":
        return jsonify({"error": "需要传入 force=1 确认导入"})

    try:
        db = get_bigdata_db()
        db.ensure_valid()
        return jsonify({"status": "success", "message": "导入完成"})
    except Exception as e:
        return jsonify({"error": str(e)})


@bigdata_bp.route("/api/bigdata/stats")
def api_bigdata_stats():
    db = get_bigdata_db()
    try:
        stats = db.get_database_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})
