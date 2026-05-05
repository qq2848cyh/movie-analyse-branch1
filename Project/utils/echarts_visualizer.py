"""
ECharts 交互式可视化模块

为前端提供可直接使用的 ECharts option 字典，涵盖评分、年份、
国家、类型、导演五个维度的图表数据。
"""

import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Set


class EChartsVisualizer:
    """ECharts 图表数据生成器"""

    COUNTRY_MAP = {
        "USA": "美国", "UK": "英国",
        "Denmark": "丹麦", "Danmark": "丹麦",
        "Switzerland": "瑞士", "Sweden": "瑞典", "Finland": "芬兰",
        "Canada": "加拿大", "India": "印度", "Indian": "印度", "india": "印度",
        "Brazil": "巴西", "Argentina": "阿根廷", "Mexico": "墨西哥",
        "Australia": "澳大利亚", "Russia": "俄罗斯", "Russian": "俄罗斯", "Ru": "俄罗斯",
        "France": "法国", "Germany": "德国",
        "Belgium": "比利时", "Poland": "波兰", "Portugal": "葡萄牙",
        "Ireland": "爱尔兰", "Netherlands": "荷兰", "Spain": "西班牙", "SPain": "西班牙",
        "Iceland": "冰島", "Czech Republic": "捷克",
        "Bahamas": "巴哈马", "Morocco": "摩洛哥", "Malta": "马耳他",
        "Luxembourg": "卢森堡", "Slovakia": "斯洛伐克", "Slovenia": "斯洛文尼亚",
        "Croatia": "克罗地亚", "Bulgaria": "保加利亚", "Estonia": "爱沙尼亚",
        "Latvia": "拉脱维亚", "Lithuania": "立陶宛", "Cyprus": "塞浦路斯",
        "Albania": "阿尔巴尼亚", "Georgia": "格鲁吉亚", "Algeria": "阿尔及利亚",
        "Tunisia": "突尼斯", "Zambia": "赞比亚", "Zimbabwe": "津巴布韦", "Guinea": "几内亚",
        "Senegal": "塞内加尔", "Cambodia": "柬埔寨", "Sri Lanka": "斯里兰卡",
        "Palestine": "巴勒斯坦", "Panama": "巴拿马", "Honduras": "洪都拉斯",
        "Jamaica": "牙买加", "Cuba": "古巴", "Dominican Republic": "多米尼加",
        "Puerto Rico": "波多黎各", "Colombia": "哥伦比亚", "Chile": "智利",
        "Paraguay": "巴拉圭", "Aruba": "阿鲁巴",
        "Afghanistan": "阿富汗",
        "Czechoslovakia": "捷克斯洛伐克", "East Germany": "东德",
        "Bosnia and Herzegovina": "波黑",
        "Federal Republic of Yugoslavia": "南斯拉夫",
        "United Arab Emirates": "阿联酋",
    }

    FONT_MAP = {
        "丹麥": "丹麦", "法國": "法国", "冰島": "冰岛",
        "馬來西亞": "马来西亚",
        "加拿大Canada": "加拿大",
    }

    CHINESE_REGIONS = {"中国大陆", "中国香港", "中国澳门", "中国台湾"}

    NON_COUNTRY = {"八一电影制片厂", "Gallifrey", "NBC"}

    @classmethod
    def _normalize_countries(cls, country_series) -> Set[str]:
        """将 regions/countries 列清洗为标准化国家集合"""
        result = set()
        for val in country_series.dropna():
            for raw_part in str(val).split("/"):
                p = raw_part.strip()
                if not p:
                    continue
                p = re.sub(r"[（(][^)）]*[）)]", "", p).strip()
                if not p:
                    continue
                for sub in p.split():
                    sub = sub.strip()
                    if not sub or sub in cls.NON_COUNTRY:
                        continue
                    sub = cls.COUNTRY_MAP.get(sub, sub)
                    sub = cls.FONT_MAP.get(sub, sub)
                    if sub in cls.NON_COUNTRY or len(sub) < 2:
                        continue
                    if sub in cls.CHINESE_REGIONS:
                        sub = "中国"
                    result.add(sub)
        return result

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def _count_movies_with_country(cls, country_series, target: str) -> int:
        """统计包含指定国家的电影数（处理多国合作场景）"""
        cnt = 0
        chinese_regions = cls.CHINESE_REGIONS
        for val in country_series.dropna():
            found = False
            for raw_part in str(val).split("/"):
                p = raw_part.strip()
                if not p:
                    continue
                p = re.sub(r"[（(][^)）]*[）)]", "", p).strip()
                if not p:
                    continue
                for sub in p.split():
                    sub = sub.strip()
                    sub = cls.COUNTRY_MAP.get(sub, sub)
                    sub = cls.FONT_MAP.get(sub, sub)
                    if target == "中国" and sub in chinese_regions:
                        found = True
                        break
                    if sub == target:
                        found = True
                        break
                if found:
                    break
            if found:
                cnt += 1
        return cnt

    def prepare_rating_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评分分布柱状图（整数区间：0-1, 1-2, ..., 9-10）"""
        ratings = pd.to_numeric(df["nums-rating"], errors="coerce").dropna()
        min_r = int(np.floor(ratings.min()))
        max_r = int(np.ceil(ratings.max()))
        bins = np.arange(min_r, max_r + 1, 1)
        hist, _ = np.histogram(ratings, bins=bins)
        bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(hist))]

        return {
            "title": {"text": "电影评分分布", "left": "center"},
            "tooltip": {"trigger": "axis", "formatter": "{b}分<br/>数量: {c}部"},
            "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}},
            "xAxis": {
                "type": "category",
                "data": bin_labels,
                "axisLabel": {"rotate": 30},
            },
            "yAxis": {"type": "value", "name": "电影数量"},
            "series": [{
                "name": "评分分布",
                "type": "bar",
                "data": [int(x) for x in hist.tolist()],
                "itemStyle": {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#5470c6"},
                            {"offset": 1, "color": "#91cc75"},
                        ],
                    },
                },
                "emphasis": {"focus": "series"},
            }],
        }

    def prepare_year_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """年份分布折线图（含 dataZoom 滑块，覆盖所有年份）"""
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        year_counts = years.value_counts().sort_index()

        return {
            "title": {"text": "电影上映年份分布", "left": "center"},
            "tooltip": {"trigger": "axis", "formatter": "{b}年<br/>数量: {c}部"},
            "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {"yAxisIndex": "none"}, "restore": {}}},
            "dataZoom": [
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 10},
                {"type": "inside"},
            ],
            "xAxis": {
                "type": "category",
                "data": year_counts.index.astype(str).tolist(),
                "boundaryGap": False,
            },
            "yAxis": {"type": "value", "name": "电影数量"},
            "series": [{
                "name": "年份分布",
                "type": "line",
                "data": [int(x) for x in year_counts.values.tolist()],
                "smooth": False,
                "symbol": "none",
                "lineStyle": {"width": 2},
                "itemStyle": {"color": "#ee6666"},
                "areaStyle": {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(238, 102, 102, 0.5)"},
                            {"offset": 1, "color": "rgba(238, 102, 102, 0.05)"},
                        ],
                    },
                },
            }],
        }

    def prepare_country_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """制片国家/地区分布玫瑰图（自动清洗中英混杂）"""
        normalized = self._normalize_countries(df["country"])
        counts = pd.Series({
            c: self._count_movies_with_country(df["country"], c)
            for c in normalized
        }).sort_values(ascending=False)
        top = counts.head(10)
        other = counts[10:].sum()

        data = [{"value": int(c), "name": n} for n, c in top.items()]
        if other > 0:
            data.append({"value": int(other), "name": "其他"})
        return {
            "title": {"text": "制片国家/地区分布", "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{a}<br/>{b}: {c} ({d}%)"},
            "legend": {"orient": "vertical", "left": "left", "top": "middle"},
            "series": [{
                "name": "国家分布",
                "type": "pie",
                "radius": ["40%", "70%"],
                "center": ["60%", "50%"],
                "data": data,
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}},
                "label": {"show": False},
                "labelLine": {"show": False},
            }],
        }

    def prepare_genre_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """电影类型分布雷达图"""
        genres = []
        for val in df["classification"].dropna():
            genres.extend(g.strip() for g in str(val).split("/"))

        genre_counts = pd.Series(genres).value_counts().head(8)
        genre_data = [int(x) for x in genre_counts.values.tolist()]
        max_val = int(genre_counts.max())

        return {
            "title": {"text": "电影类型分布", "left": "center"},
            "tooltip": {},
            "radar": {
                "indicator": [{"name": g, "max": max_val} for g in genre_counts.index],
            },
            "series": [{
                "name": "类型分布",
                "type": "radar",
                "data": [{
                    "value": genre_data,
                    "name": "电影数量",
                    "areaStyle": {"color": "rgba(255, 182, 193, 0.6)"},
                    "lineStyle": {"color": "#ffb6c1", "width": 2},
                }],
            }],
        }

    def prepare_director_ranking(self, df: pd.DataFrame) -> Dict[str, Any]:
        """导演作品数量排名横向柱状图"""
        all_directors = []
        for val in df["director"].dropna():
            all_directors.extend(d.strip() for d in str(val).split("/"))

        director_counts = pd.Series(all_directors).value_counts().head(15)
        total = len(set(all_directors))

        return {
            "title": [
                {"text": "导演作品数量排名", "left": "center", "textStyle": {"fontSize": 16}},
                {"text": f"显示前15位导演（共{total}位导演，{director_counts.sum()}部电影）",
                 "left": "center", "top": "8%", "textStyle": {"fontSize": 12, "color": "#666"}},
            ],
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "toolbox": {"feature": {"dataView": {"readOnly": False}, "saveAsImage": {}, "restore": {}}},
            "grid": {"left": "15%", "right": "5%", "bottom": "15%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "value", "name": "作品数量",
                "nameLocation": "middle", "nameGap": 30,
                "nameTextStyle": {"fontSize": 12, "padding": [10, 0, 0, 0]},
            },
            "yAxis": {
                "type": "category",
                "data": director_counts.index.tolist(),
                "axisLabel": {"interval": 0, "fontSize": 11, "width": 120, "overflow": "truncate", "ellipsis": "..."},
            },
            "series": [{
                "name": "作品数量",
                "type": "bar",
                "data": [int(x) for x in director_counts.values.tolist()],
                "itemStyle": {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": "#fac858"},
                            {"offset": 1, "color": "#ee6666"},
                        ],
                    },
                },
                "label": {"show": True, "position": "right", "formatter": "{c}部", "fontSize": 11},
            }],
        }

    def generate_all_charts_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """一次性生成全部五种图表数据及统计摘要"""
        ratings = pd.to_numeric(df["nums-rating"], errors="coerce").dropna()
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        avg_rating = round(float(ratings.mean()), 2) if len(ratings) > 0 else 0
        earliest_year = int(years.min()) if len(years) > 0 else 0
        latest_year = int(years.max()) if len(years) > 0 else 0

        country_count = len(self._normalize_countries(df["country"]))

        return {
            "stats": {
                "avg_rating": avg_rating,
                "earliest_year": earliest_year,
                "latest_year": latest_year,
                "rated_count": int(len(ratings)),
                "country_count": country_count,
            },
            "rating_distribution": self.prepare_rating_distribution(df),
            "year_distribution": self.prepare_year_distribution(df),
            "country_distribution": self.prepare_country_distribution(df),
            "genre_distribution": self.prepare_genre_distribution(df),
            "director_ranking": self.prepare_director_ranking(df),
        }
