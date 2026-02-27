"""
Chat Routes - AI Chatbot for Personalized RA Recommendations
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models import User, ChatHistory, ChatMessage, ChatResponse, APIResponse, Prediction
from app.utils import get_current_user
from app.services.chatbot_service import get_chatbot_response, generate_welcome_message

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("/send", response_model=APIResponse)
async def send_chat_message(
    chat_data: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """
    Send message to AI chatbot and get personalized response
    
    - **message**: User's message/question
    
    Response is personalized based on user's latest RA prediction and severity level
    """
    # Get user's latest prediction to personalize response
    latest_prediction = await Prediction.find(
        Prediction.user_id == str(current_user.id)
    ).sort([("timestamp", -1)]).first_or_none()
    
    if not latest_prediction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No RA prediction found. Please upload an X-ray and get a prediction first."
        )
    
    try:
        # Get AI response based on severity level
        ai_response = await get_chatbot_response(
            user_message=chat_data.message,
            severity_level=latest_prediction.severity_level,
            result_percentage=latest_prediction.result_percentage
        )
        
        # Save chat history
        chat_entry = ChatHistory(
            user_id=str(current_user.id),
            message=chat_data.message,
            response=ai_response
        )
        
        await chat_entry.insert()
        
        chat_response = ChatResponse(
            id=str(chat_entry.id),
            user_id=chat_entry.user_id,
            message=chat_entry.message,
            response=chat_entry.response,
            timestamp=chat_entry.timestamp
        )
        
        return APIResponse(
            status="success",
            message="Response generated",
            data={
                "chat": chat_response.dict(),
                "context": {
                    "severity_level": latest_prediction.severity_level.value,
                    "result_percentage": latest_prediction.result_percentage
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.get("/history", response_model=APIResponse)
async def get_chat_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """
    Get user's chat history
    
    - **limit**: Maximum number of messages to return (default: 20)
    """
    # Fetch chat history for current user, sorted by timestamp (newest first)
    chats = await ChatHistory.find(
        ChatHistory.user_id == str(current_user.id)
    ).sort(-ChatHistory.timestamp).limit(limit).to_list()
    
    chat_list = [
        ChatResponse(
            id=str(chat.id),
            user_id=chat.user_id,
            message=chat.message,
            response=chat.response,
            timestamp=chat.timestamp
        ).dict()
        for chat in chats
    ]
    
    return APIResponse(
        status="success",
        message=f"Retrieved {len(chat_list)} chat message(s)",
        data={
            "chats": chat_list,
            "total": len(chat_list)
        }
    )


@router.get("/welcome", response_model=APIResponse)
async def get_welcome_message(current_user: User = Depends(get_current_user)):
    """
    Get personalized welcome message based on user's latest prediction
    """
    # Get user's latest prediction
    latest_prediction = await Prediction.find(
        Prediction.user_id == str(current_user.id)
    ).sort([("timestamp", -1)]).first_or_none()
    
    if not latest_prediction:
        return APIResponse(
            status="success",
            message="Welcome message",
            data={
                "message": "👋 Welcome to RAiCare! Please upload an X-ray to get started with personalized recommendations."
            }
        )
    
    welcome_msg = await generate_welcome_message(
        severity_level=latest_prediction.severity_level,
        result_percentage=latest_prediction.result_percentage
    )
    
    return APIResponse(
        status="success",
        message="Welcome message generated",
        data={
            "message": welcome_msg,
            "context": {
                "severity_level": latest_prediction.severity_level.value,
                "result_percentage": latest_prediction.result_percentage
            }
        }
    )


@router.delete("/clear", response_model=APIResponse)
async def clear_chat_history(current_user: User = Depends(get_current_user)):
    """
    Clear user's entire chat history
    """
    # Delete all chat history for current user
    result = await ChatHistory.find(
        ChatHistory.user_id == str(current_user.id)
    ).delete()
    
    return APIResponse(
        status="success",
        message=f"Chat history cleared ({result.deleted_count} messages deleted)",
        data={"deleted_count": result.deleted_count}
    )
