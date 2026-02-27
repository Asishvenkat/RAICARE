"""
MongoDB Models for RAiCare
"""
from beanie import Document
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# ============ USER MODEL ============
class User(Document):
    username: str = Field(..., unique=True, index=True)
    email: EmailStr = Field(..., unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "users"
        
    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com"
            }
        }


# ============ PREDICTION MODEL ============
class Prediction(Document):
    user_id: str = Field(..., index=True)
    image_url: str
    result_percentage: float = Field(..., ge=0, le=100)
    severity_level: SeverityLevel
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "predictions"
        
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "image_url": "https://cloudinary.com/image.jpg",
                "result_percentage": 75.5,
                "severity_level": "moderate"
            }
        }


# ============ CHAT HISTORY MODEL ============
class ChatHistory(Document):
    user_id: str = Field(..., index=True)
    message: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "chat_history"
        
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "message": "What foods should I avoid?",
                "response": "Based on your moderate RA condition..."
            }
        }


# ============ PYDANTIC SCHEMAS ============

# User Schemas
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# Prediction Schemas
class PredictionCreate(BaseModel):
    result_percentage: float = Field(..., ge=0, le=100)
    severity_level: SeverityLevel


class PredictionResponse(BaseModel):
    id: str
    user_id: str
    image_url: str
    result_percentage: float
    severity_level: SeverityLevel
    timestamp: datetime


# Chat Schemas
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    id: str
    user_id: str
    message: str
    response: str
    timestamp: datetime


# API Response Schemas
class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None
