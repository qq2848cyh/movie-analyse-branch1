import os

# 数据保存文件路径
BASE_DATA_DIR = "data"
CSV_PATH = os.path.join(BASE_DATA_DIR, "douban_top250_movies.csv")
BIGDATA_DB_PATH = os.path.join(BASE_DATA_DIR, "bigdata_movies.db")
NEW_MOVIES_DB_PATH = os.path.join(BASE_DATA_DIR, "new_movies.db")
NEW_MOVIES_CSV = os.path.join(BASE_DATA_DIR, "new_data", "douban_all_movies.csv")

# 图片保存目录
BASE_STATIC_DIR = "static/visualization"
IMAGE_SAVE_DIR = os.path.join(BASE_STATIC_DIR, "images")

# 确保目录存在
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)
