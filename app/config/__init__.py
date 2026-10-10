"""App-level configuration package.

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
"""

from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
