"""路由模块 — 统一导入所有子路由"""

from .main import main_bp
from .movies import movies_bp
from .bigdata import bigdata_bp
from .analysis import analysis_bp

__all__ = ["main_bp", "movies_bp", "bigdata_bp", "analysis_bp"]
