from application.utils import db_manager as _module

globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})
__all__ = [name for name in dir(_module) if not name.startswith("__")]
