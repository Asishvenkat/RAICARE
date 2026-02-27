"""
Cloudinary Service - Image Upload Handler
"""
import cloudinary
import cloudinary.uploader
from fastapi import UploadFile
from app.config import settings

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)


async def upload_image_to_cloudinary(file: UploadFile) -> str:
    """
    Upload image to Cloudinary and return the URL
    
    Args:
        file: UploadFile object containing the image
        
    Returns:
        str: Public URL of uploaded image
    """
    try:
        # Read file content
        contents = await file.read()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            contents,
            folder="raicare_xrays",
            resource_type="image",
            allowed_formats=["jpg", "jpeg", "png"]
        )
        
        # Return secure URL
        return upload_result.get("secure_url")
        
    except Exception as e:
        raise Exception(f"Failed to upload image to Cloudinary: {str(e)}")
