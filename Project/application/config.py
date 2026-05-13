"""项目配置文件 - 集中管理数据路径与环境配置"""

import os
from typing import Optional

class Config:
    BASE_DATA_DIR = "data"
    BIGDATA_DB_PATH = os.path.join(BASE_DATA_DIR, "bigdata_movies.db")
    NEW_MOVIES_DB_PATH = os.path.join(BASE_DATA_DIR, "new_movies.db")
    NEW_MOVIES_CSV = os.path.join(BASE_DATA_DIR, "new_data", "douban_all_movies.csv")
    DEBUG = False


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


def get_config(env_name: Optional[str] = None):
    env = (env_name or os.getenv("APP_ENV", "development")).lower()
    if env in {"prod", "production"}:
        return ProductionConfig
    return DevelopmentConfig


BASE_DATA_DIR = Config.BASE_DATA_DIR
BIGDATA_DB_PATH = Config.BIGDATA_DB_PATH
NEW_MOVIES_DB_PATH = Config.NEW_MOVIES_DB_PATH
NEW_MOVIES_CSV = Config.NEW_MOVIES_CSV

os.makedirs(BASE_DATA_DIR, exist_ok=True)
