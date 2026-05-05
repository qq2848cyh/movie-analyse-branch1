import sqlite3
import os
import json
import csv
import re
import pandas as pd
from typing import Optional, Tuple, List, Dict


class DBManager:
    """
    SQLite 数据库管理类
    支持：配置驱动建表、分块CSV导入、FTS5全文索引、SQL聚合查询、分页搜索
    """

    def __init__(self, db_path: str, schema_path: str = None):
        self.db_path = db_path
        if schema_path is None:
            schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "schema_mapping.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.csv_dir = self._get_csv_dir()
        self._ensure_all_tables()

    def _get_csv_dir(self):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(os.path.dirname(project_dir), "data", "bigdata")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        return conn

    def _ensure_all_tables(self):
        with self._get_conn() as conn:
            for table_name, table_config in self.schema["tables"].items():
                cols = []
                pk = table_config.get("primary_key", None)
                for db_col, col_conf in table_config["columns"].items():
                    col_def = f"{db_col} {col_conf['type']}"
                    cols.append(col_def)

                if pk:
                    cols.append(f"PRIMARY KEY ({', '.join(pk)})")

                sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})"
                conn.execute(sql)

                if "fts5" in table_config:
                    fts_columns = ", ".join(table_config["fts5"])
                    pk_col = list(table_config["columns"].keys())[0]
                    fts_sql = (
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_fts "
                        f"USING fts5({fts_columns}, content='{table_name}', "
                        f"content_rowid='{pk_col}')"
                    )
                    conn.execute(fts_sql)

        with self._get_conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS wordcloud_cache ("
                "cache_key TEXT PRIMARY KEY, data TEXT)"
            )

    def _ensure_valid_table(self):
        with self._get_conn() as conn:
            conn.execute("DROP TABLE IF EXISTS movies_valid")
            conn.execute("""
                CREATE TABLE movies_valid AS
                SELECT * FROM movies
                WHERE douban_score > 0 AND douban_votes > 0
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_valid_score ON movies_valid(douban_score DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_valid_votes ON movies_valid(douban_votes DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_valid_year ON movies_valid(year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_valid_name ON movies_valid(name)"
            )
            cnt = conn.execute("SELECT COUNT(*) FROM movies_valid").fetchone()[0]
            print(f"  [movies_valid] 构建完成，有效电影 {cnt} 部", flush=True)

            conn.execute("DROP TABLE IF EXISTS comments_valid")
            conn.execute("""
                CREATE TABLE comments_valid AS
                SELECT c.* FROM comments c
                INNER JOIN movies_valid mv ON c.movie_id = mv.movie_id
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cvalid_mid ON comments_valid(movie_id)"
            )
            ccnt = conn.execute("SELECT COUNT(*) FROM comments_valid").fetchone()[0]
            print(f"  [comments_valid] 构建完成，有效评论 {ccnt} 条", flush=True)

            conn.execute("DELETE FROM wordcloud_cache")

    def _create_indexes(self):
        with self._get_conn() as conn:
            for table_name, table_config in self.schema["tables"].items():
                for idx_col in table_config.get("indexes", []):
                    idx_name = f"idx_{table_name}_{idx_col}"
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({idx_col})"
                    )

    def _clean_year(self, val):
        if val is None or val == "":
            return None
        val = str(val).strip()
        if val in ("2049", "0", "0.0"):
            return None
        val = val.rstrip(".0")
        try:
            y = int(float(val))
            if 1800 <= y <= 2030:
                return y
            return None
        except (ValueError, TypeError):
            return None

    def _clean_number(self, val):
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def import_all(self) -> Dict[str, int]:
        """导入所有表，返回各表导入行数"""
        results = {}
        for table_name in self.schema["import_order"]:
            count = self._import_table(table_name)
            results[table_name] = count
            print(f"  [{table_name}] 导入 {count} 行", flush=True)

        self._import_link_tables()
        self._create_indexes()
        self._rebuild_fts()
        return results

    def _import_table(self, table_name: str) -> int:
        table_config = self.schema["tables"][table_name]
        csv_file = table_config.get("csv_file")
        if csv_file is None:
            return 0

        csv_path = os.path.join(self.csv_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  警告: {csv_path} 不存在，跳过")
            return 0

        skip_cols = set(table_config.get("skip_columns", []))
        column_map = {}
        db_columns = []
        for db_col, col_conf in table_config["columns"].items():
            csv_col = col_conf["csv_column"]
            if csv_col in skip_cols:
                continue
            column_map[csv_col] = db_col
            db_columns.append(db_col)

        placeholders = ", ".join("?" for _ in db_columns)
        columns_str = ", ".join(db_columns)
        sql = f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})"

        total = 0
        chunksize = 50000

        with self._get_conn() as conn:
            reader = pd.read_csv(
                csv_path, encoding="utf-8-sig", chunksize=chunksize,
                dtype=str, keep_default_na=False, na_values=[]
            )

            for chunk in reader:
                batch = []
                for _, row in chunk.iterrows():
                    values = []
                    for csv_col in column_map:
                        val = row.get(csv_col, "")
                        db_col = column_map[csv_col]

                        if table_name == "movies":
                            if db_col == "year":
                                val = self._clean_year(val)
                            elif db_col == "douban_score":
                                val = self._clean_number(val)
                            elif db_col == "douban_votes":
                                val = self._clean_number(val)

                        if pd.isna(val) or val == "":
                            val = None

                        values.append(val)
                    batch.append(tuple(values))

                conn.executemany(sql, batch)
                total += len(batch)
                print(f"  [{table_name}] 已导入 {total} 行...", flush=True)

        return total

    def _import_link_tables(self):
        movies_config = self.schema["tables"]["movies"]
        csv_path = os.path.join(self.csv_dir, movies_config["csv_file"])

        if not os.path.exists(csv_path):
            return

        with self._get_conn() as conn:
            conn.execute("DELETE FROM movie_actors")
            conn.execute("DELETE FROM movie_directors")

        actor_batch = []
        director_batch = []
        chunksize = 50000

        reader = pd.read_csv(
            csv_path, encoding="utf-8-sig", chunksize=chunksize,
            dtype=str, keep_default_na=False, na_values=[]
        )

        for chunk in reader:
            for _, row in chunk.iterrows():
                movie_id = self._clean_number(row.get("MOVIE_ID", ""))
                if movie_id is None:
                    continue

                actor_ids_str = str(row.get("ACTOR_IDS", ""))
                if actor_ids_str and actor_ids_str != "nan":
                    for pair in actor_ids_str.split("|"):
                        pair = pair.strip()
                        if ":" in pair:
                            name, pid = pair.rsplit(":", 1)
                            name = name.strip()
                            pid = self._clean_number(pid.strip())
                            if pid and name:
                                actor_batch.append((int(movie_id), int(pid), name))

                director_ids_str = str(row.get("DIRECTOR_IDS", ""))
                if director_ids_str and director_ids_str != "nan":
                    for pair in director_ids_str.split("|"):
                        pair = pair.strip()
                        if ":" in pair:
                            name, pid = pair.rsplit(":", 1)
                            name = name.strip()
                            pid = self._clean_number(pid.strip())
                            if pid and name:
                                director_batch.append((int(movie_id), int(pid), name))

        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movie_actors (movie_id, person_id, actor_name) VALUES (?, ?, ?)",
                actor_batch)
            conn.executemany(
                "INSERT OR IGNORE INTO movie_directors (movie_id, person_id, director_name) VALUES (?, ?, ?)",
                director_batch)

        print(f"  [movie_actors] 导入 {len(actor_batch)} 行", flush=True)
        print(f"  [movie_directors] 导入 {len(director_batch)} 行", flush=True)

    def _rebuild_fts(self):
        with self._get_conn() as conn:
            conn.execute("INSERT INTO comments_fts(comments_fts) VALUES('rebuild')")
        print("  [comments_fts] FTS5 全文索引重建完成")

    def get_total_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM movies_valid").fetchone()
            return row[0] if row else 0

    def get_movies_page(
        self, page: int = 1, per_page: int = 20, search: Optional[str] = None
    ) -> Tuple[List[Dict], int]:
        params = []
        where_parts = []

        if search and search.strip():
            keyword = f"%{search.strip()}%"
            for col in self.schema["search_columns"]:
                where_parts.append(f"{col} LIKE ?")
                params.append(keyword)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " OR ".join(where_parts)

        with self._get_conn() as conn:
            count_sql = f"SELECT COUNT(*) FROM movies_valid {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        offset = (page - 1) * per_page

        data_sql = f"""
            SELECT movie_id, name, actors, directors, douban_score, douban_votes,
                   genres, regions, year, storyline, imdb_id, languages
            FROM movies_valid
            {where_clause}
            ORDER BY douban_score DESC, douban_votes DESC
            LIMIT ? OFFSET ?
        """

        with self._get_conn() as conn:
            rows = conn.execute(data_sql, params + [per_page, offset]).fetchall()

        start_rank = offset + 1
        results = []
        for i, row in enumerate(rows):
            results.append({
                "rank": start_rank + i,
                "title": row["name"],
                "director": row["directors"],
                "actors": row["actors"],
                "nums-rating": row["douban_score"],
                "comment_nums": row["douban_votes"],
                "comment": row["storyline"],
                "year": row["year"],
                "country": row["regions"],
                "classification": row["genres"],
            })

        return results, total

    def _get_echarts_column_map(self):
        col_map = {}
        aliases = []
        for echarts_col, mapping in self.schema["echarts_mapping"].items():
            sql_alias = mapping.get("sql_alias", mapping["column"])
            col_map[echarts_col] = sql_alias
            if sql_alias != mapping["column"]:
                aliases.append(f"{mapping['column']} AS {sql_alias}")
        return col_map, aliases

    def get_analysis_dataframe(self) -> pd.DataFrame:
        col_map, aliases = self._get_echarts_column_map()

        select_parts = []
        for echarts_col in ["nums-rating", "year", "country", "classification", "director"]:
            mapping = self.schema["echarts_mapping"][echarts_col]
            src_col = mapping["column"]
            sql_alias = mapping.get("sql_alias", src_col)
            if sql_alias != src_col:
                select_parts.append(f"{src_col} AS {sql_alias}")
            else:
                select_parts.append(src_col)

        sql = f"SELECT {', '.join(select_parts)} FROM movies_valid"

        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()

        data = []
        reverse_map = {v: k for k, v in col_map.items()}
        for row in rows:
            record = {}
            for col_name in row.keys():
                echarts_col = reverse_map.get(col_name, col_name)
                record[echarts_col] = row[col_name]
            data.append(record)

        if data:
            return pd.DataFrame(data)
        return pd.DataFrame(columns=list(col_map.keys()))

    def get_wordcloud_data(self, force_refresh: bool = False) -> Dict:
        if not force_refresh:
            cached = self._load_wc_cache()
            if cached:
                return cached

        result = self._compute_wordcloud_data()
        self._save_wc_cache(result)
        return result

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

    def _compute_wordcloud_data(self) -> Dict:
        result = {}

        with self._get_conn() as conn:
            titles = conn.execute(
                "SELECT name FROM movies_valid WHERE name != ''"
            ).fetchall()
            title_text = " ".join(r["name"] for r in titles if r["name"])
            result["titles"] = self._count_words(title_text, top_n=80)

            genres = conn.execute(
                "SELECT genres FROM movies_valid WHERE genres != ''"
            ).fetchall()
            result["genres"] = self._count_genres(genres, top_n=30)

            max_id = conn.execute("SELECT MAX(comment_id) FROM comments_valid").fetchone()[0] or 1
            sample_count = 20000
            sample_ids = set()
            import random as _random
            while len(sample_ids) < sample_count:
                sid = _random.randint(1, max_id)
                sample_ids.add(sid)

            placeholders = ",".join("?" for _ in sample_ids)
            comment_rows = conn.execute(
                f"SELECT content FROM comments_valid WHERE content != '' AND comment_id IN ({placeholders})",
                list(sample_ids)
            ).fetchall()

            comment_text = " ".join(r["content"] for r in comment_rows if r["content"])
            result["comments"] = self._count_words(comment_text, top_n=80, chinese_only=True)

            total_movies = conn.execute(
                "SELECT COUNT(*) FROM movies_valid"
            ).fetchone()[0]
            total_comments = conn.execute("SELECT COUNT(*) FROM comments_valid").fetchone()[0]
            result["stats"] = {
                "total_movies": total_movies,
                "total_comments": total_comments,
            }

        return result

    def _count_words(self, text: str, top_n: int = 80, chinese_only: bool = False) -> List[Dict]:
        freq = {}
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
            "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
            "什么", "怎么", "如何", "为什么", "因为", "所以", "但是", "然而",
            "可以", "这个", "那个", "还", "被", "把", "让", "从", "与", "或",
            "等", "及", "之", "为", "对", "于", "以", "而", "且", "但",
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "its", "and", "or", "not", "no", "but", "if", "so",
            "as", "we", "he", "she", "they", "this", "that", "these",
            "those", "my", "your", "his", "her", "our", "their", "all",
            "than", "then", "just", "about", "into", "over", "also",
            "very", "too", "only", "more", "some", "any", "each", "every",
            "both", "few", "most", "other", "such", "now", "up", "out",
            "when", "who", "how", "what", "which", "where", "there",
            "here", "one", "two", "like", "get", "got", "make", "made",
            "through", "after", "before", "between", "during", "while",
            "电影", "一部", "真的", "感觉", "觉得", "还是", "不过", "已经",
            "不是", "就是", "或者", "这么", "那么", "一直", "一点", "很多",
            "出来", "开始", "最后", "比较", "其实", "有点", "东西", "地方",
            "因为", "所以", "我们", "他们", "你们", "它们", "自己", "大家",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "20", "30", "100", "200",
        }
        for word in text.replace("\n", " ").split():
            word = word.strip("，。！？；：""''（）【】《》、…—·,.!?;:()[]{}<>\"'|/\\@#$%^&*+=~`")
            if len(word) >= 2 and word not in stop_words:
                word_lower = word.lower()
                if word_lower in stop_words:
                    continue
                if chinese_only and not any('\u4e00' <= c <= '\u9fff' for c in word):
                    continue
                freq[word] = freq.get(word, 0) + 1

        return [
            {"name": w, "value": c}
            for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ]

    def _count_genres(self, rows, top_n: int = 30) -> List[Dict]:
        freq = {}
        for row in rows:
            genres_str = row["genres"]
            if not genres_str:
                continue
            for g in genres_str.replace("/", " ").split():
                g = g.strip()
                if g:
                    freq[g] = freq.get(g, 0) + 1
        return [
            {"name": w, "value": c}
            for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ]

    def get_statistics(self) -> Dict:
        with self._get_conn() as conn:
            total_movies = conn.execute("SELECT COUNT(*) FROM movies_valid").fetchone()[0]
            rated = total_movies
            avg_score = conn.execute(
                "SELECT ROUND(AVG(douban_score), 2) FROM movies_valid"
            ).fetchone()[0] or 0
            min_year = conn.execute(
                "SELECT MIN(year) FROM movies_valid WHERE year IS NOT NULL"
            ).fetchone()[0]
            max_year = conn.execute(
                "SELECT MAX(year) FROM movies_valid WHERE year IS NOT NULL"
            ).fetchone()[0]
            total_comments = conn.execute("SELECT COUNT(*) FROM comments_valid").fetchone()[0]

        return {
            "total_movies": total_movies,
            "rated_movies": rated,
            "avg_score": avg_score,
            "min_year": min_year,
            "max_year": max_year,
            "total_comments": total_comments,
        }

    def clear_data(self):
        with self._get_conn() as conn:
            for table_name in ["movie_actors", "movie_directors",
                               "comments_valid", "movies_valid",
                               "comments", "person", "movies"]:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute("DROP TABLE IF EXISTS comments_fts")
            conn.execute("DROP TABLE IF EXISTS wordcloud_cache")
        self._ensure_all_tables()
