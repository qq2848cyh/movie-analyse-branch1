import sqlite3
import os
import json
import pandas as pd
from typing import Optional, Tuple, List, Dict

from .stopwords import get_stop_words
from .base_manager import BaseMovieManager


class NewMovieManager(BaseMovieManager):
    """2000+ 电影数据集 SQLite 管理类（替代原 Top250 CSV）"""

    TABLE = "movies"

    COLUMNS = [
        ("movie_id",       "INTEGER PRIMARY KEY"),
        ("title",          "TEXT"),
        ("rating",         "REAL"),
        ("total_ratings",  "INTEGER"),
        ("directors",      "TEXT"),
        ("actors",         "TEXT"),
        ("screenwriters",  "TEXT"),
        ("release_date",   "TEXT"),
        ("year",           "INTEGER"),
        ("genres",         "TEXT"),
        ("countries",      "TEXT"),
        ("languages",      "TEXT"),
        ("runtime",        "TEXT"),
        ("summary",        "TEXT"),
        ("link",           "TEXT"),
        ("poster",         "TEXT"),
        ("tags",           "TEXT"),
    ]

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        cols = ", ".join(f"{n} {t}" for n, t in self.COLUMNS)
        with self._get_conn() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {self.TABLE} ({cols})")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_new_rating ON {self.TABLE}(rating DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_new_year ON {self.TABLE}(year)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS wordcloud_cache ("
                "cache_key TEXT PRIMARY KEY, data TEXT)"
            )

    def import_from_csv(self, csv_path: str) -> int:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        total = 0

        with self._get_conn() as conn:
            conn.execute(f"DELETE FROM {self.TABLE}")

            for _, row in df.iterrows():
                year = None
                rd = str(row.get("release_date", "")).strip()
                if rd and rd != "nan":
                    year_str = rd[:4]
                    if year_str.isdigit():
                        year = int(year_str)

                values = (
                    int(row["movie_id"]) if row.get("movie_id") and str(row["movie_id"]) != "nan" else None,
                    str(row.get("title", "")) if str(row.get("title", "")) != "nan" else None,
                    float(row["rating"]) if row.get("rating") and str(row["rating"]) != "nan" else None,
                    int(float(row["total_ratings"])) if row.get("total_ratings") and str(row["total_ratings"]) != "nan" else None,
                    str(row.get("directors", "")) if str(row.get("directors", "")) != "nan" else None,
                    str(row.get("actors", "")) if str(row.get("actors", "")) != "nan" else None,
                    str(row.get("screenwriters", "")) if str(row.get("screenwriters", "")) != "nan" else None,
                    rd if rd else None,
                    year,
                    str(row.get("genres", "")) if str(row.get("genres", "")) != "nan" else None,
                    str(row.get("countries", "")) if str(row.get("countries", "")) != "nan" else None,
                    str(row.get("languages", "")) if str(row.get("languages", "")) != "nan" else None,
                    str(row.get("runtime", "")) if str(row.get("runtime", "")) != "nan" else None,
                    str(row.get("summary", "")) if str(row.get("summary", "")) != "nan" else None,
                    str(row.get("link", "")) if str(row.get("link", "")) != "nan" else None,
                    str(row.get("poster", "")) if str(row.get("poster", "")) != "nan" else None,
                    str(row.get("tags", "")) if str(row.get("tags", "")) != "nan" else None,
                )
                conn.execute(
                    f"INSERT INTO {self.TABLE} (movie_id,title,rating,total_ratings,"
                    "directors,actors,screenwriters,release_date,year,genres,countries,"
                    "languages,runtime,summary,link,poster,tags) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values
                )
                total += 1

        return total

    def get_movies_page(
        self, page: int = 1, per_page: int = 20, search: str = ""
    ) -> Tuple[List[Dict], int]:
        params = []
        where = ""
        if search.strip():
            kw = f"%{search.strip()}%"
            where = ("WHERE title LIKE ? OR directors LIKE ? OR actors LIKE ? "
                     "OR genres LIKE ? OR countries LIKE ? OR summary LIKE ? "
                     "OR year LIKE ?")
            params = [kw] * 7

        with self._get_conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM {self.TABLE} {where}", params
            ).fetchone()[0]

        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        offset = (page - 1) * per_page

        sql = f"""
            SELECT movie_id, title, rating, total_ratings, directors, actors,
                   release_date, year, genres, countries, summary
            FROM {self.TABLE}
            {where}
            ORDER BY rating DESC, total_ratings DESC
            LIMIT ? OFFSET ?
        """
        conn = self._get_conn()
        rows = conn.execute(sql, params + [per_page, offset]).fetchall()

        start_rank = offset + 1
        results = []
        for i, row in enumerate(rows):
            results.append({
                "rank":           start_rank + i,
                "title":          row["title"],
                "nums-rating":    row["rating"],
                "comment_nums":   row["total_ratings"],
                "director":       row["directors"],
                "actors":         row["actors"],
                "comment":        row["summary"],
                "year":           row["year"],
                "country":        row["countries"],
                "classification": row["genres"],
            })
        conn.close()
        return results, total

    def get_analysis_dataframe(self) -> pd.DataFrame:
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT rating AS \"nums-rating\", year, countries AS country, "
                f"genres AS classification, directors AS director "
                f"FROM {self.TABLE}"
            ).fetchall()

        data = []
        for row in rows:
            record = {}
            for key in row.keys():
                record[key] = row[key]
            data.append(record)

        if data:
            return pd.DataFrame(data)
        return pd.DataFrame(
            columns=["nums-rating", "year", "country", "classification", "director"]
        )

    def _load_wc_cache(self) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT data FROM wordcloud_cache WHERE cache_key = 'all'"
            ).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def _save_wc_cache(self, data: Dict):
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM wordcloud_cache WHERE cache_key = 'all'"
            )
            conn.execute(
                "INSERT INTO wordcloud_cache (cache_key, data) VALUES ('all', ?)",
                (json.dumps(data, ensure_ascii=False),)
            )

    def get_wordcloud_data(self, force_refresh: bool = False) -> Dict:
        if not force_refresh:
            cached = self._load_wc_cache()
            if cached:
                return cached

        import jieba

        titles, summaries = self.get_wordcloud_texts()

        stop_words = get_stop_words()

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

        result = {
            "titles": count_chinese_words(titles),
            "summaries": count_chinese_words(summaries),
        }
        self._save_wc_cache(result)
        return result
