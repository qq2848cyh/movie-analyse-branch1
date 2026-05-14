import os
import time
import numpy as np
import json
from typing import Dict, List, Optional
from collections import Counter

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
)
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import lightgbm as lgb

from .stopwords import get_ml_stop_words

try:
    from ..config import SENTIMENT_CURVES_DIR
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SENTIMENT_CURVES_DIR


class SentimentAnalyzer:
    def __init__(self, db_manager):
        self.db = db_manager
        self.trained = False
        self.models = {}
        self.best_model = None
        self.vectorizer = None
        self.metrics = {}
        self.test_results = {}
        self.chi2_features = None

    def load_samples(self, n_samples: int = 60000):
        c = self.db._get_conn()
        rows = c.execute(
            "SELECT content, rating, comment_time FROM comments_valid "
            "WHERE content != '' AND rating IS NOT NULL "
            "ORDER BY CAST(rowid * 10007 % 100003 AS INTEGER) LIMIT ?",
            (n_samples,),
        ).fetchall()
        return rows

    _load_samples = load_samples

    def train(self, n_samples: int = 15000):

        samples = self._load_samples(n_samples)
        texts = [s["content"] for s in samples]
        total = len(texts)
        ratings = [s["rating"] for s in samples]
        labels_binary = [1 if r >= 4 else 0 for r in ratings]
        labels_five = [int(r) - 1 for r in ratings]

        stop_words = get_ml_stop_words()

        def tokenize(text):
            words = jieba.lcut(str(text))
            return [
                w for w in words
                if len(w) >= 2 and w not in stop_words
                and any("\u4e00" <= c <= "\u9fff" for c in w)
            ]

        print(f"[sentiment] tokenizing ({total} samples)...", flush=True)
        t_tok = time.time()
        tokenized = [" ".join(tokenize(t)) for t in texts]
        print(f"[sentiment] tokenizing done ({int(time.time()-t_tok)}s)", flush=True)

        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(tokenized)

        from sklearn.feature_selection import chi2
        chi2_scores, _ = chi2(X, labels_binary)
        top_idx = np.argsort(-chi2_scores)[:2000]
        self.chi2_features = [
            {"word": self.vectorizer.get_feature_names_out()[i], "chi2": round(float(chi2_scores[i]), 2)}
            for i in top_idx[:50]
        ]

        y = np.array(labels_binary)
        y5 = np.array(labels_five)

        X_train, X_test, y_train, y_test, y5_train, y5_test = train_test_split(
            X, y, y5, test_size=0.2, random_state=42, stratify=y
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            "NaiveBayes": MultinomialNB(alpha=0.5),
            "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=800, max_depth=10, learning_rate=0.03,
                min_child_samples=50,
                random_state=42, verbose=-1, force_row_wise=True,
            ),
        }

        print("[sentiment] training binary classification (3 models)...", flush=True)
        t_bin = time.time()
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
            model.fit(X_train, y_train)
            preds_test = model.predict(X_test)
            self.models[name] = model
            self.test_results[name] = {
                "accuracy": round(float(accuracy_score(y_test, preds_test)), 4),
                "precision": round(float(precision_score(y_test, preds_test)), 4),
                "recall": round(float(recall_score(y_test, preds_test)), 4),
                "f1": round(float(f1_score(y_test, preds_test)), 4),
                "cv_mean": round(float(cv_scores.mean()), 4),
                "cv_std": round(float(cv_scores.std()), 4),
            }
        print(f"  LightGBM: acc={self.test_results['LightGBM']['accuracy']:.4f} f1={self.test_results['LightGBM']['f1']:.4f} "
              f"({int(time.time()-t_bin)}s)", flush=True)

        preds_train = self.models["LightGBM"].predict(X_train)
        train_metrics = {
            "accuracy": round(float(accuracy_score(y_train, preds_train)), 4),
            "precision": round(float(precision_score(y_train, preds_train)), 4),
            "recall": round(float(recall_score(y_train, preds_train)), 4),
            "f1": round(float(f1_score(y_train, preds_train)), 4),
        }
        print(f"  LightGBM_train: acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1']:.4f}", flush=True)
        self.ml_train_metrics = {"LightGBM": {"train": train_metrics, "test": self.test_results["LightGBM"]}}

        print("[sentiment] tuning LightGBM...", flush=True)
        try:
            param_grid = {
                "learning_rate": [0.02, 0.03, 0.05],
            }
            grid = GridSearchCV(
                lgb.LGBMClassifier(n_estimators=600, max_depth=10, min_child_samples=50,
                                   random_state=42, verbose=-1, force_row_wise=True),
                param_grid, cv=2, scoring="accuracy", n_jobs=1, verbose=0
            )
            t_gs = time.time()
            grid.fit(X_train, y_train)
            best_lgb = grid.best_estimator_
            best_lgb_preds = best_lgb.predict(X_test)
            self.test_results["LightGBM_tuned"] = {
                "accuracy": round(float(accuracy_score(y_test, best_lgb_preds)), 4),
                "precision": round(float(precision_score(y_test, best_lgb_preds)), 4),
                "recall": round(float(recall_score(y_test, best_lgb_preds)), 4),
                "f1": round(float(f1_score(y_test, best_lgb_preds)), 4),
                "best_params": str(grid.best_params_),
            }
            self.models["LightGBM_tuned"] = best_lgb
            print(f"  LightGBM_tuned: acc={self.test_results['LightGBM_tuned']['accuracy']} "
                  f"params={grid.best_params_} ({int(time.time()-t_gs)}s)", flush=True)
            tuned_train_preds = best_lgb.predict(X_train)
            tuned_train = {
                "accuracy": round(float(accuracy_score(y_train, tuned_train_preds)), 4),
                "f1": round(float(f1_score(y_train, tuned_train_preds)), 4),
            }
            print(f"  LightGBM_tuned_train: acc={tuned_train['accuracy']:.4f} f1={tuned_train['f1']:.4f}", flush=True)
            self.ml_train_metrics["LightGBM_tuned"] = {"train": tuned_train, "test": self.test_results["LightGBM_tuned"]}
        except Exception as e:
            print(f"  LightGBM tuning skipped: {e}", flush=True)

        print("[sentiment] training 5-class classification...", flush=True)
        t_5cls = time.time()
        five_models = {
            "LogisticRegression_5cls": LogisticRegression(C=1.0, max_iter=1500, random_state=42),
            "LightGBM_5cls": lgb.LGBMClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.03,
                min_child_samples=50,
                random_state=42, verbose=-1, force_row_wise=True,
                num_class=5,
            ),
        }
        for name, model in five_models.items():
            model.fit(X_train, y5_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y5_test, preds)
            cm = confusion_matrix(y5_test, preds, labels=[0, 1, 2, 3, 4])
            self.models[name] = model
            self.test_results[name] = {
                "accuracy": round(float(acc), 4),
                "confusion_matrix": cm.tolist(),
                "per_class_precision": [round(float(v), 4) for v in precision_score(y5_test, preds, average=None, labels=[0,1,2,3,4])],
                "per_class_recall": [round(float(v), 4) for v in recall_score(y5_test, preds, average=None, labels=[0,1,2,3,4])],
                "per_class_f1": [round(float(v), 4) for v in f1_score(y5_test, preds, average=None, labels=[0,1,2,3,4])],
            }
            self.best_model = model
        print(f"  LightGBM_5cls: acc={self.test_results['LightGBM_5cls']['accuracy']:.4f} "
              f"({int(time.time()-t_5cls)}s)", flush=True)
        preds5_train = self.models["LightGBM_5cls"].predict(X_train)
        train5_acc = round(float(accuracy_score(y5_train, preds5_train)), 4)
        print(f"  LightGBM_5cls_train: acc={train5_acc:.4f}", flush=True)
        self.ml_train_metrics["LightGBM_5cls"] = {"train": {"accuracy": train5_acc}, "test": {"accuracy": self.test_results["LightGBM_5cls"]["accuracy"]}}

        self.trained = True

        pos_idx = [i for i, l in enumerate(y) if l == 1]
        neg_idx = [i for i, l in enumerate(y) if l == 0]
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

        rating_keywords = {}
        for r_val in range(1, 6):
            ridx = [i for i, r in enumerate(ratings) if r == r_val]
            if len(ridx) > 10:
                rvec = X[ridx].mean(axis=0).A1
                allvec = X.mean(axis=0).A1
                rdiff = rvec - allvec
                top_r = np.argsort(-rdiff)[:20]
                rating_keywords[f"rating_{r_val}"] = [
                    {"word": self.vectorizer.get_feature_names_out()[i], "weight": round(float(rdiff[i]), 4)}
                    for i in top_r
                ]

        return {
            "test_results": self.test_results,
            "key_words": key_words,
            "rating_keywords": rating_keywords,
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

        result = []
        for r in rows:
            result.append({
                "year": r["year"],
                "avg_rating": round(r["avg_rating"], 2),
                "count": r["count"],
                "positive_ratio": round(r["positive_ratio"], 1),
            })

        from scipy import stats
        first_half = [x["avg_rating"] for x in result if x["year"] < "2012"]
        second_half = [x["avg_rating"] for x in result if x["year"] >= "2012"]
        if len(first_half) >= 3 and len(second_half) >= 3:
            u_stat, p_val = stats.mannwhitneyu(first_half, second_half, alternative="two-sided")
        else:
            u_stat, p_val = 0, 1

        return {
            "points": result,
            "mann_whitney_u": round(float(u_stat), 2),
            "p_value": round(float(p_val), 4),
            "significant": p_val < 0.05,
        }

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

    def get_all_results(self, n_samples: int = 15000) -> Dict:
        if not self.trained:
            train_result = self.train(n_samples)
        else:
            train_result = {"test_results": self.test_results}

        trend_data = self.get_trend_data()

        curves_dir = SENTIMENT_CURVES_DIR
        os.makedirs(curves_dir, exist_ok=True)
        if hasattr(self, "ml_train_metrics") and self.ml_train_metrics:
            with open(os.path.join(curves_dir, "ml_train_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(self.ml_train_metrics, f, ensure_ascii=False, indent=2)

        return {
            "classification": train_result,
            "trend": trend_data,
            "rating_distribution": self.get_rating_distribution(),
        }

    def export_results(self) -> Dict:
        if not self.trained:
            self.train(15000)

        return {
            "sentiment_analysis": {
                "binary_classification": self.test_results,
                "chi2_top_features": self.chi2_features[:20] if self.chi2_features else [],
                "five_class_best": self.test_results.get("LightGBM_5cls", {}),
                "time_trend": {
                    "mann_whitney_u": self.get_trend_data().get("mann_whitney_u", 0),
                    "p_value": self.get_trend_data().get("p_value", 1),
                    "significant_change": self.get_trend_data().get("significant", False),
                },
            }
        }
