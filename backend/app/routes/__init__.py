"""
Initialize routes package
"""
from .auth import router as auth_router
from .prediction import router as prediction_router
from .chat import router as chat_router

__all__ = [
    "auth_router",
    "prediction_router",
    "chat_router"
]
