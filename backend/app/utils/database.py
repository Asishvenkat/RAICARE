"""
Database utilities for MongoDB
"""
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config import settings
from app.models import User, Prediction, ChatHistory


async def init_db():
    """
    Initialize database connection and Beanie ODM
    """
    try:
        # Create Motor client with shorter timeout
        client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=3000  # 3 second timeout
        )
        
        # Test connection
        await client.admin.command('ping')
        
        # Initialize beanie with the Product document class
        await init_beanie(
            database=client[settings.DATABASE_NAME],
            document_models=[User, Prediction, ChatHistory]
        )
        
        print(f"✅ Connected to MongoDB database: {settings.DATABASE_NAME}")
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)[:200]}")
        print("⚠️  Continuing without database - some features may not work...")


async def close_db():
    """
    Close database connection
    """
    print("❌ Closing MongoDB connection")
