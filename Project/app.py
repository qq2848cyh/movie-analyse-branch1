"""
电影数据分析平台 — Flask 主应用

路由已拆分到 routes/ 目录下的各个模块中
"""

import os
import sys

# 设置 HuggingFace 缓存路径（本地缓存包含 hub/ 子目录，需用 HF_HOME）
import config as cfg
os.environ["HF_HOME"] = os.path.join(cfg.DATA_DIR, "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"

from flask import Flask

# 导入路由模块
from routes import main_bp, movies_bp, bigdata_bp, analysis_bp

# ==================== 应用初始化 ====================

app = Flask(
    __name__,
    static_folder=os.path.join(cfg.BASE_DIR, "static"),
    template_folder=os.path.join(cfg.BASE_DIR, "Project", "templates"),
)

# 注册路由蓝图
app.register_blueprint(main_bp)
app.register_blueprint(movies_bp)
app.register_blueprint(bigdata_bp)
app.register_blueprint(analysis_bp)

# ==================== 启动服务 ====================

if __name__ == "__main__":
    print(f"[Flask] 启动电影数据分析平台...")
    print(f"[Flask] 项目目录: {cfg.BASE_DIR}")
    print(f"[Flask] 数据目录: {cfg.DATA_DIR}")
    print(f"[Flask] 缓存目录: {cfg.CACHE_DIR}")
    print()
    
    # 预加载数据库连接（懒加载机制会在首次请求时自动触发）
    print("[Flask] 预加载模块...")
    
    # 启动服务器
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False,
    )
