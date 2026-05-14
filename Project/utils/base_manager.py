"""基础数据库管理基类 — 抽取 DBManager 和 NewMovieManager 的共同逻辑"""

import sqlite3
import os
import json
import pandas as pd
from typing import Optional, Tuple, List, Dict


class BaseMovieManager:
    """电影数据库管理基类"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_dir()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_total_count(self) -> int:
        """获取电影总数"""
        c = self._get_conn()
        count = c.execute(f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0]
        c.close()
        return count

    def get_movies_page(self, page: int = 1, per_page: int = 20, search: str = "") -> Tuple[List, int]:
        """分页获取电影列表"""
        c = self._get_conn()
        offset = (page - 1) * per_page

        if search:
            query = f"""
                SELECT * FROM {self.TABLE}
                WHERE title LIKE ? OR directors LIKE ? OR actors LIKE ? OR genres LIKE ?
                ORDER BY rating DESC, total_ratings DESC
                LIMIT ? OFFSET ?
            """
            pattern = f"%{search}%"
            rows = c.execute(query, (pattern, pattern, pattern, pattern, per_page, offset)).fetchall()
            total = c.execute(
                f"""
                SELECT COUNT(*) FROM {self.TABLE}
                WHERE title LIKE ? OR directors LIKE ? OR actors LIKE ? OR genres LIKE ?
                """,
                (pattern, pattern, pattern, pattern)
            ).fetchone()[0]
        else:
            rows = c.execute(
                f"SELECT * FROM {self.TABLE} ORDER BY rating DESC, total_ratings DESC LIMIT ? OFFSET ?",
                (per_page, offset)
            ).fetchall()
            total = self.get_total_count()

        c.close()
        return [dict(row) for row in rows], total

    def get_analysis_dataframe(self) -> pd.DataFrame:
        """获取分析用 DataFrame"""
        c = self._get_conn()
        df = pd.read_sql(f"SELECT * FROM {self.TABLE}", c)
        c.close()
        return df

    def get_wordcloud_texts(self) -> Tuple[List[str], List[str]]:
        """获取词云文本数据"""
        c = self._get_conn()
        rows = c.execute(f"SELECT title, summary FROM {self.TABLE} WHERE title != ''").fetchall()
        c.close()
        titles = [str(r["title"]) for r in rows]
        summaries = [str(r["summary"]) for r in rows if r["summary"]]
        return titles, summaries

    def _load_wc_cache(self) -> Optional[Dict]:
        """加载词云缓存"""
        cache_path = self._get_wc_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                import pickle
                return pickle.load(f)
        except Exception:
            return None

    def _save_wc_cache(self, data: Dict):
        """保存词云缓存"""
        cache_path = self._get_wc_cache_path()
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                import pickle
                pickle.dump(data, f)
        except Exception:
            pass

    def _get_wc_cache_path(self) -> str:
        """获取词云缓存路径（子类需实现）"""
        raise NotImplementedError("子类必须实现 _get_wc_cache_path() 方法")
