"""
Prediction Routes - Image Upload and Prediction History
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import List
from app.models import User, Prediction, PredictionCreate, PredictionResponse, APIResponse, SeverityLevel
from app.utils import get_current_user
from app.services.cloudinary_service import upload_image_to_cloudinary
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/prediction", tags=["Predictions"])


@router.post("/upload", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def upload_prediction(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload X-ray image and run RA prediction using AI model
    
    - **file**: X-ray image file (JPG, JPEG, PNG)
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, and PNG images are allowed"
        )
    
    try:
        # Run AI prediction on the image
        prediction_result = prediction_service.predict_image(file)
        
        # Reset file pointer for Cloudinary upload
        await file.seek(0)
        
        # Upload image to Cloudinary
        image_url = await upload_image_to_cloudinary(file)
        
        # Map severity level to enum
        severity_mapping = {
            "none": SeverityLevel.NONE,
            "mild": SeverityLevel.MILD,
            "moderate": SeverityLevel.MODERATE,
            "severe": SeverityLevel.SEVERE
        }
        
        # Save prediction to database
        new_prediction = Prediction(
            user_id=str(current_user.id),
            image_url=image_url,
            result_percentage=float(prediction_result["result_percentage"]),
            severity_level=severity_mapping[prediction_result["severity_level"]]
        )
        
        print(f"DEBUG ROUTE - Saving prediction: {prediction_result['result_percentage']} ({type(prediction_result['result_percentage'])})")
        
        await new_prediction.insert()
        
        print(f"DEBUG ROUTE - Saved to DB: {new_prediction.result_percentage} ({type(new_prediction.result_percentage)})")
        
        prediction_response = PredictionResponse(
            id=str(new_prediction.id),
            user_id=new_prediction.user_id,
            image_url=new_prediction.image_url,
            result_percentage=new_prediction.result_percentage,
            severity_level=new_prediction.severity_level,
            timestamp=new_prediction.timestamp
        )
        
        return APIResponse(
            status="success",
            message="Prediction completed and saved successfully",
            data={
                "prediction": {
                    "id": str(new_prediction.id),
                    "user_id": new_prediction.user_id,
                    "image_url": new_prediction.image_url,
                    "result_percentage": float(new_prediction.result_percentage),
                    "severity_level": new_prediction.severity_level.value,
                    "timestamp": new_prediction.timestamp.isoformat()
                },
                "ai_result": prediction_result
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process prediction: {str(e)}"
        )


@router.get("/history", response_model=APIResponse)
async def get_prediction_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """
    Get user's prediction history
    
    - **limit**: Maximum number of records to return (default: 10)
    """
    # Fetch predictions for current user, sorted by timestamp (newest first)
    predictions = await Prediction.find(
        Prediction.user_id == str(current_user.id)
    ).sort(-Prediction.timestamp).limit(limit).to_list()
    
    prediction_list = [
        PredictionResponse(
            id=str(pred.id),
            user_id=pred.user_id,
            image_url=pred.image_url,
            result_percentage=pred.result_percentage,
            severity_level=pred.severity_level,
            timestamp=pred.timestamp
        ).dict()
        for pred in predictions
    ]
    
    return APIResponse(
        status="success",
        message=f"Retrieved {len(prediction_list)} prediction(s)",
        data={
            "predictions": prediction_list,
            "total": len(prediction_list)
        }
    )


@router.get("/latest", response_model=APIResponse)
async def get_latest_prediction(current_user: User = Depends(get_current_user)):
    """
    Get user's most recent prediction
    """
    # Fetch most recent prediction
    latest_prediction = await Prediction.find(
        Prediction.user_id == str(current_user.id)
    ).sort([("timestamp", -1)]).first_or_none()
    
    if not latest_prediction:
        return APIResponse(
            status="success",
            message="No predictions found",
            data={"prediction": None}
        )
    
    prediction_response = PredictionResponse(
        id=str(latest_prediction.id),
        user_id=latest_prediction.user_id,
        image_url=latest_prediction.image_url,
        result_percentage=latest_prediction.result_percentage,
        severity_level=latest_prediction.severity_level,
        timestamp=latest_prediction.timestamp
    )
    
    return APIResponse(
        status="success",
        message="Latest prediction retrieved",
        data={"prediction": prediction_response.dict()}
    )
