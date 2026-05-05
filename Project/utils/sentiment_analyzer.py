import numpy as np
import json
from typing import Dict, List, Optional
from collections import Counter

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb


class SentimentAnalyzer:
    """评论情感分析器 — 多模型对比 + 关键词分析 + 时间趋势"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.trained = False
        self.models = {}
        self.vectorizer = None
        self.metrics = {}

    def _load_samples(self, n_samples: int = 60000):
        c = self.db._get_conn()
        rows = c.execute(
            "SELECT content, rating, comment_time FROM comments_valid "
            "WHERE content != '' AND rating IS NOT NULL "
            "ORDER BY RANDOM() LIMIT ?",
            (n_samples,),
        ).fetchall()
        return rows

    def train(self, n_samples: int = 60000):

        samples = self._load_samples(n_samples)
        texts = [s["content"] for s in samples]
        ratings = [s["rating"] for s in samples]
        labels = [1 if r >= 4 else 0 for r in ratings]

        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "说", "要", "你", "会", "着", "没有", "看",
            "好", "自己", "这", "他", "她", "们", "那", "什么", "怎么", "可以",
            "还", "被", "让", "从", "与", "但", "而", "等", "及", "之", "为",
            "对", "于", "以", "因为", "所以", "我们", "他们", "但是", "然而",
        }

        def tokenize(text):
            words = jieba.lcut(str(text))
            return [
                w for w in words
                if len(w) >= 2 and w not in stop_words
                and any("\u4e00" <= c <= "\u9fff" for c in w)
            ]

        print("[sentiment] tokenizing...", flush=True)
        tokenized = [" ".join(tokenize(t)) for t in texts]

        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(tokenized)
        y = np.array(labels)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            "NaiveBayes": MultinomialNB(alpha=0.5),
            "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.05,
                random_state=42, verbose=-1, force_row_wise=True,
            ),
        }

        print("[sentiment] training...", flush=True)
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            model.fit(X, y)
            preds = model.predict(X)
            self.models[name] = model
            self.metrics[name] = {
                "accuracy": round(accuracy_score(y, preds), 4),
                "cv_mean": round(scores.mean(), 4),
                "cv_std": round(scores.std(), 4),
            }
            print(f"  {name}: acc={self.metrics[name]['accuracy']}, cv={self.metrics[name]['cv_mean']}+/-{self.metrics[name]['cv_std']}", flush=True)

        self.trained = True

        top_features = {}
        for name, model in models.items():
            if hasattr(model, "coef_"):
                coef = model.coef_[0]
                idxs = np.argsort(-np.abs(coef))[:30]
                top_features[name] = [
                    {"word": self.vectorizer.get_feature_names_out()[i], "weight": round(float(coef[i]), 4)}
                    for i in idxs
                ]
            elif hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                idxs = np.argsort(-imp)[:30]
                top_features[name] = [
                    {"word": self.vectorizer.get_feature_names_out()[i], "weight": round(float(imp[i]), 4)}
                    for i in idxs
                ]

        pos_idx = [i for i, l in enumerate(labels) if l == 1]
        neg_idx = [i for i, l in enumerate(labels) if l == 0]
        pos_vec = X[pos_idx].mean(axis=0).A1
        neg_vec = X[neg_idx].mean(axis=0).A1
        diff = pos_vec - neg_vec
        top_diff = np.argsort(-diff)[:30]
        bottom_diff = np.argsort(diff)[:30]

        key_words = {
            "positive": [
                {"word": self.vectorizer.get_feature_names_out()[i], "weight": round(float(diff[i]), 4)}
                for i in top_diff
            ],
            "negative": [
                {"word": self.vectorizer.get_feature_names_out()[i], "weight": round(float(diff[i]), 4)}
                for i in bottom_diff
            ],
        }

        return {
            "metrics": self.metrics,
            "top_features": top_features,
            "key_words": key_words,
        }

    def get_trend_data(self) -> List[Dict]:
        c = self.db._get_conn()
        rows = c.execute("""
            SELECT
                SUBSTR(comment_time, 1, 4) as year,
                AVG(rating) as avg_rating,
                COUNT(*) as count,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_ratio
            FROM comments_valid
            WHERE comment_time != '' AND rating IS NOT NULL
            GROUP BY year
            HAVING count > 50
            ORDER BY year
        """).fetchall()
        return [
            {
                "year": r["year"],
                "avg_rating": round(r["avg_rating"], 2),
                "count": r["count"],
                "positive_ratio": round(r["positive_ratio"], 1),
            }
            for r in rows
        ]

    def get_rating_distribution(self) -> Dict:
        c = self.db._get_conn()
        rows = c.execute("""
            SELECT rating, COUNT(*) as count
            FROM comments_valid WHERE rating IS NOT NULL
            GROUP BY rating ORDER BY rating
        """).fetchall()
        return {
            "labels": [str(r["rating"]) for r in rows],
            "counts": [r["count"] for r in rows],
        }

    def get_all_results(self, n_samples: int = 60000) -> Dict:
        if not self.trained:
            train_result = self.train(n_samples)
        else:
            train_result = {
                "metrics": self.metrics,
            }

        return {
            "classification": train_result,
            "trend": self.get_trend_data(),
            "rating_distribution": self.get_rating_distribution(),
        }
