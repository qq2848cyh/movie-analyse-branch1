"""项目配置文件 - 集中管理数据路径"""

import os

BASE_DATA_DIR = "data"
BIGDATA_DB_PATH = os.path.join(BASE_DATA_DIR, "bigdata_movies.db")
NEW_MOVIES_DB_PATH = os.path.join(BASE_DATA_DIR, "new_movies.db")
NEW_MOVIES_CSV = os.path.join(BASE_DATA_DIR, "new_data", "douban_all_movies.csv")

os.makedirs(BASE_DATA_DIR, exist_ok=True)
