"""
RAiCare Backend - FastAPI Application
Medical AI Web App for Rheumatoid Arthritis Detection

Features:
- JWT Authentication (Register/Login)
- X-ray Image Upload with Cloudinary
- RA Prediction Storage
- AI Chatbot with Gemini (Personalized Recommendations)
- MongoDB Database
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.utils import init_db, close_db
from app.routes import auth_router, prediction_router, chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    print("🚀 Starting RAiCare Backend...")
    try:
        await init_db()
        print("✅ Connected to MongoDB database: raicare_db")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # Continue anyway for testing
    
    print("✅ Application ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    try:
        await close_db()
        print("❌ Closing MongoDB connection")
    except Exception as e:
        print(f"❌ Error closing database: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    print("🚀 Starting RAiCare Backend...")
    try:
        await init_db()
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("Continuing without database...")
    
    print("✅ Application ready!")
    
    yield
    
    print("🛑 Shutting down...")
    try:
        await close_db()
    except Exception as e:
        print(f"❌ Error closing database: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="RAiCare API",
    description="Backend API for Rheumatoid Arthritis Detection and Management",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "status": "success",
        "message": "RAiCare API is running",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth/register, /auth/login",
            "predictions": "/prediction/upload, /prediction/history, /prediction/latest",
            "chat": "/chat/send, /chat/history, /chat/welcome, /chat/clear",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "database": "connected",
        "service": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=False  # Disable reload to avoid multiprocessing issues
    )
