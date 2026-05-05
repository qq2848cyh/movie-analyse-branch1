"""
交互式ECharts可视化模块，用于生成前端可交互的图表数据

功能包括：
- 评分分布柱状图（可缩放、悬停查看详情）
- 年份分布折线图（可缩放、数据点交互）
- 国家分布饼图（可点击筛选、悬停显示百分比）
- 类型分布雷达图（多维度对比）
- 导演排名柱状图（横向滚动、排序）
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any


class EChartsVisualizer:
    """
    ECharts交互式可视化类
    """

    def __init__(self, logger: logging.Logger = None):
        """初始化可视化器"""
        self.logger = logger if logger else logging.getLogger(__name__)

    def prepare_rating_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备评分分布数据（柱状图）
        
        Returns:
            ECharts配置字典
        """
        ratings = pd.to_numeric(df["nums-rating"], errors="coerce").dropna()
        rating_counts = ratings.value_counts().sort_index()
        
        # 创建数据区间
        bins = np.arange(8.0, 9.8, 0.1)
        hist, _ = np.histogram(ratings, bins=bins)
        
        # 确保数据类型为Python原生类型
        hist_data = [int(x) for x in hist.tolist()]
        
        option = {
            "title": {"text": "电影评分分布", "left": "center"},
            "tooltip": {
                "trigger": "axis",
                "formatter": "评分: {b}<br/>数量: {c}部"
            },
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataZoom": {},
                    "restore": {}
                }
            },
            "xAxis": {
                "type": "category",
                "data": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(hist))],
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {"type": "value", "name": "电影数量"},
            "series": [{
                "name": "评分分布",
                "type": "bar",
                "data": hist_data,
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#5470c6"},
                            {"offset": 1, "color": "#91cc75"}
                        ]
                    }
                },
                "emphasis": {"focus": "series"}
            }]
        }
        return option

    def prepare_year_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备年份分布数据（折线图）
        
        Returns:
            ECharts配置字典
        """
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        year_counts = years.value_counts().sort_index()
        
        # 确保数据类型为Python原生类型
        year_data = [int(x) for x in year_counts.values.tolist()]
        
        option = {
            "title": {"text": "电影上映年份分布", "left": "center"},
            "tooltip": {
                "trigger": "axis",
                "formatter": "年份: {b}<br/>数量: {c}部"
            },
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataZoom": {
                        "yAxisIndex": "none"
                    },
                    "restore": {}
                }
            },
            "xAxis": {
                "type": "category",
                "data": year_counts.index.astype(str).tolist(),
                "boundaryGap": False
            },
            "yAxis": {"type": "value", "name": "电影数量"},
            "series": [{
                "name": "年份分布",
                "type": "line",
                "data": year_data,
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "lineStyle": {"width": 3},
                "itemStyle": {"color": "#ee6666"},
                "areaStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(238, 102, 102, 0.6)"},
                            {"offset": 1, "color": "rgba(238, 102, 102, 0.1)"}
                        ]
                    }
                }
            }]
        }
        return option

    def prepare_country_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备国家分布数据（饼图）
        
        Returns:
            ECharts配置字典
        """
        # 处理国家数据（可能有多个国家）
        countries = []
        for country_str in df["country"].dropna():
            if "/" in country_str:
                countries.extend([c.strip() for c in country_str.split("/")])
            else:
                countries.append(country_str.strip())
        
        # 将中国各地区统一为中国
        normalized_countries = []
        for country in countries:
            if country in ["中国大陆", "中国香港", "中国澳门", "中国台湾"]:
                normalized_countries.append("中国")
            else:
                normalized_countries.append(country)
        
        country_counts = pd.Series(normalized_countries).value_counts()
        
        # 只显示前10个国家，其他归为"其他"
        top_countries = country_counts.head(10)
        other_count = country_counts[10:].sum() if len(country_counts) > 10 else 0
        
        data = []
        for country, count in top_countries.items():
            data.append({"value": int(count), "name": country})
        
        if other_count > 0:
            data.append({"value": int(other_count), "name": "其他"})

        option = {
            "title": {"text": "制片国家/地区分布", "left": "center"},
            "tooltip": {
                "trigger": "item",
                "formatter": "{a} <br/>{b}: {c} ({d}%)"
            },
            "legend": {
                "orient": "vertical",
                "left": "left",
                "top": "middle"
            },
            "series": [{
                "name": "国家分布",
                "type": "pie",
                "radius": ["40%", "70%"],
                "center": ["60%", "50%"],
                "data": data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                },
                "label": {"show": False},
                "labelLine": {"show": False}
            }]
        }
        return option

    def prepare_genre_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备类型分布数据（雷达图）
        
        Returns:
            ECharts配置字典
        """
        # 处理电影类型数据
        genres = []
        for genre_str in df["classification"].dropna():
            if "/" in genre_str:
                genres.extend([g.strip() for g in genre_str.split("/")])
            else:
                genres.append(genre_str.strip())
        
        genre_counts = pd.Series(genres).value_counts().head(8)
        
        # 确保数据类型为Python原生类型
        genre_data = [int(x) for x in genre_counts.values.tolist()]
        
        option = {
            "title": {"text": "电影类型分布", "left": "center"},
            "tooltip": {},
            "radar": {
                "indicator": [{"name": genre, "max": int(genre_counts.max())} 
                             for genre in genre_counts.index]
            },
            "series": [{
                "name": "类型分布",
                "type": "radar",
                "data": [{
                    "value": genre_data,
                    "name": "电影数量",
                    "areaStyle": {"color": "rgba(255, 182, 193, 0.6)"},
                    "lineStyle": {"color": "#ffb6c1", "width": 2}
                }]
            }]
        }
        return option

    def prepare_director_ranking(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备导演排名数据（横向柱状图）
        
        Returns:
            ECharts配置字典
        """
        # 处理导演数据：分割多个导演，统计每个导演的作品数量
        all_directors = []
        for director_str in df["director"].dropna():
            # 处理多个导演的情况（用"/"分隔）
            if "/" in director_str:
                directors = [d.strip() for d in director_str.split("/")]
                all_directors.extend(directors)
            else:
                all_directors.append(director_str.strip())
        
        # 统计导演作品数量，取前15位
        director_counts = pd.Series(all_directors).value_counts().head(15)
        
        # 确保数据类型为Python原生类型
        director_data = [int(x) for x in director_counts.values.tolist()]
        
        # 计算统计信息
        total_directors = len(set(all_directors))
        top_directors_movie_count = director_counts.sum()
        
        option = {
            "title": [
                {
                    "text": "导演作品数量排名", 
                    "left": "center", 
                    "textStyle": {"fontSize": 16}
                },
                {
                    "text": f"显示前15位导演（共{total_directors}位导演，{top_directors_movie_count}部电影）",
                    "left": "center",
                    "top": "8%",
                    "textStyle": {
                        "fontSize": 12,
                        "color": "#666"
                    }
                }
            ],
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"}
            },
            "toolbox": {
                "feature": {
                    "dataView": {"readOnly": False},
                    "saveAsImage": {},
                    "restore": {}
                }
            },
            "grid": {
                "left": "15%", 
                "right": "5%", 
                "bottom": "15%",  # 增加底部空间以容纳横坐标名称
                "top": "15%",
                "containLabel": True
            },
            "xAxis": {
                "type": "value", 
                "name": "作品数量",
                "nameLocation": "middle",
                "nameGap": 30,  # 增加名称与坐标轴的距离
                "nameTextStyle": {
                    "fontSize": 12,
                    "padding": [10, 0, 0, 0]  # 增加上边距
                }
            },
            "yAxis": {
                "type": "category",
                "data": director_counts.index.tolist(),
                "axisLabel": {
                    "interval": 0,
                    "fontSize": 11,
                    "width": 120,
                    "overflow": "truncate",
                    "ellipsis": "..."
                }
            },
            "series": [{
                "name": "作品数量",
                "type": "bar",
                "data": director_data,
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 1, "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": "#fac858"},
                            {"offset": 1, "color": "#ee6666"}
                        ]
                    }
                },
                "label": {
                    "show": True,
                    "position": "right",
                    "formatter": "{c}部",
                    "fontSize": 11
                }
            }]
        }
        return option

    def generate_all_charts_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成所有图表的数据
        
        Returns:
            包含所有图表配置的字典
        """
        return {
            "rating_distribution": self.prepare_rating_distribution(df),
            "year_distribution": self.prepare_year_distribution(df),
            "country_distribution": self.prepare_country_distribution(df),
            "genre_distribution": self.prepare_genre_distribution(df),
            "director_ranking": self.prepare_director_ranking(df)
        }