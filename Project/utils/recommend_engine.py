import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


class RecommendEngine:
    """电影推荐引擎 — SVD 协同过滤 + TF-IDF 内容推荐"""

    def __init__(self, db_manager, content_db_manager=None):
        self.db = db_manager
        self.content_db = content_db_manager or db_manager
        self.user_vecs = {}
        self.movie_vecs = {}
        self.global_bias = 0
        self.content_matrix = None
        self.content_vectorizer = None
        self.movie_ids = []
        self.movie_names = {}
        self.trained = False

    def train_cf(self, n_factors: int = 30, n_epochs: int = 12, n_samples: int = 150000):
        import random

        c = self.db._get_conn()

        rows = c.execute("""
            SELECT user_md5, movie_id, rating FROM comments_valid
            WHERE user_md5 != '' AND rating IS NOT NULL
            ORDER BY RANDOM() LIMIT ?
        """, (n_samples,)).fetchall()

        user_ids = {}
        movie_ids = {}
        ratings = []

        for r in rows:
            u = user_ids.setdefault(r["user_md5"], len(user_ids))
            m = movie_ids.setdefault(r["movie_id"], len(movie_ids))
            ratings.append((u, m, float(r["rating"])))

        n_users = len(user_ids)
        n_movies = len(movie_ids)

        self.global_bias = np.mean([r[2] for r in ratings])
        lr = 0.005
        reg = 0.02

        self.user_vecs = {i: np.random.normal(0, 0.1, n_factors) for i in range(n_users)}
        self.movie_vecs = {i: np.random.normal(0, 0.1, n_factors) for i in range(n_movies)}
        self.user_bias = np.zeros(n_users)
        self.movie_bias = np.zeros(n_movies)

        print(f"[cf] training SVD on {len(ratings)} ratings, {n_users} users x {n_movies} movies...", flush=True)
        for epoch in range(n_epochs):
            random.shuffle(ratings)
            total_err = 0
            for u, m, r in ratings:
                pred = (
                    self.global_bias
                    + self.user_bias[u]
                    + self.movie_bias[m]
                    + np.dot(self.user_vecs[u], self.movie_vecs[m])
                )
                err = r - pred
                total_err += err * err

                self.user_bias[u] += lr * (err - reg * self.user_bias[u])
                self.movie_bias[m] += lr * (err - reg * self.movie_bias[m])
                grad_u = self.movie_vecs[m] * err - reg * self.user_vecs[u]
                grad_m = self.user_vecs[u] * err - reg * self.movie_vecs[m]
                self.user_vecs[u] += lr * grad_u
                self.movie_vecs[u] = self.user_vecs.get(u, np.zeros(n_factors))
                self.movie_vecs[m] += lr * grad_m

            rmse = np.sqrt(total_err / len(ratings))
            if (epoch + 1) % 3 == 0:
                print(f"  epoch {epoch+1}/{n_epochs}, RMSE={rmse:.4f}", flush=True)

        self._movie_id_map = {v: k for k, v in movie_ids.items()}
        self._user_id_map = {v: k for k, v in user_ids.items()}
        self.trained = True

        return {"rmse": round(rmse, 4), "users": n_users, "movies": n_movies, "ratings": len(ratings)}

    def train_content(self):
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer

        c = self.content_db._get_conn()
        rows = c.execute("""
            SELECT movie_id, title, genres, directors, actors, summary, tags
            FROM movies
            WHERE title != ''
        """).fetchall()

        docs = []
        self.movie_ids = []
        self.movie_names = {}
        for r in rows:
            text = (str(r["genres"] or "") + " " + str(r["directors"] or "") + " " +
                    str(r["actors"] or "") + " " + str(r["summary"] or "")[:200] + " " +
                    str(r["tags"] or ""))
            words = jieba.lcut(text)
            docs.append(" ".join([w for w in words if len(w) >= 2]))
            self.movie_ids.append(r["movie_id"])
            self.movie_names[r["movie_id"]] = r["title"]

        self.content_vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True)
        self.content_matrix = self.content_vectorizer.fit_transform(docs)

        return {"movies": len(self.movie_ids)}

    def recommend_by_movie(self, movie_name: str, top_n: int = 10) -> List[Dict]:
        if self.content_matrix is None:
            self.train_content()

        from sklearn.metrics.pairwise import cosine_similarity

        idx = None
        query_lower = movie_name.lower()
        for i, mid in enumerate(self.movie_ids):
            name_lower = self.movie_names[mid].lower()
            if query_lower in name_lower or name_lower in query_lower:
                idx = i
                break

        if idx is None:
            for i, mid in enumerate(self.movie_ids):
                name_str = self.movie_names[mid]
                common = sum(1 for c in movie_name if c in name_str)
                if common >= len(movie_name) * 0.6:
                    idx = i
                    break

        if idx is None:
            return []

        sim = cosine_similarity(self.content_matrix[idx:idx+1], self.content_matrix)[0]
        top = np.argsort(-sim)[1:top_n+1]

        return [
            {
                "name": self.movie_names[self.movie_ids[i]],
                "similarity": round(float(sim[i]), 3),
            }
            for i in top if sim[i] > 0
        ]

    def recommend_by_cf(self, user_md5: str, top_n: int = 10) -> List[Dict]:
        if not self.trained:
            self.train_cf()

        if user_md5 not in self._user_id_map:
            return []

        u = self._user_id_map[user_md5]
        u_vec = self.user_vecs[u]
        scores = {}
        for m_idx, m_vec in self.movie_vecs.items():
            pred = self.global_bias + self.user_bias[u] + self.movie_bias[m_idx] + np.dot(u_vec, m_vec)
            scores[self._movie_id_map[m_idx]] = pred

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        result = []
        for mid, score in top:
            name = self.movie_names.get(mid) if self.movie_names else str(mid)
            if not self.movie_names:
                c = self.db._get_conn()
                r = c.execute("SELECT name FROM movies_valid WHERE movie_id = ?", (mid,)).fetchone()
                name = r["name"] if r else str(mid)
            result.append({"name": name, "predicted_rating": round(float(score), 1)})

        return result

    def get_all_results(self) -> Dict:
        return {
            "cf_metrics": self._cf_metrics if hasattr(self, '_cf_metrics') else {},
            "by_movie_sample": self.recommend_by_movie("我不是药神"),
            "content_ready": self.content_matrix is not None,
            "cf_ready": self.trained,
        }
