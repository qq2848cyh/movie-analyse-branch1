from typing import Optional

from application.config import get_config
from application.routes.main import app as _app


def create_app(env_name: Optional[str] = None):
    config_cls = get_config(env_name)
    _app.config.from_object(config_cls)
    return _app


__all__ = ["create_app"]
