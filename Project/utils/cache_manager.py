"""统一缓存管理工具类 — 集中处理各类缓存操作"""

import os
import pickle
import json

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config import CACHE_DIR
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CACHE_DIR


class CacheManager:
    """通用缓存管理器"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_path(self, cache_name: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, cache_name)

    def load(self, cache_name: str, default=None):
        """加载缓存数据"""
        path = self.get_path(cache_name)
        if not os.path.exists(path):
            return default

        ext = os.path.splitext(cache_name)[1].lower()
        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            return default

    def save(self, cache_name: str, data):
        """保存缓存数据"""
        path = self.get_path(cache_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ext = os.path.splitext(cache_name)[1].lower()
        try:
            if ext == ".json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            return True
        except Exception:
            return False

    def exists(self, cache_name: str) -> bool:
        """检查缓存是否存在"""
        return os.path.exists(self.get_path(cache_name))

    def remove(self, cache_name: str) -> bool:
        """删除缓存"""
        path = self.get_path(cache_name)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except Exception:
                return False
        return False

    def clear(self):
        """清空所有缓存"""
        for filename in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# 全局缓存管理器实例
main_cache = CacheManager(CACHE_DIR)
