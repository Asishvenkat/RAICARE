"""
Authentication Routes - User Registration and Login
"""
from fastapi import APIRouter, HTTPException, status
from app.models import User, UserRegister, UserLogin, Token, UserResponse, APIResponse
from app.utils import get_password_hash, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """
    Register a new user
    
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 6 characters)
    """
    # Check if username already exists
    existing_user = await User.find_one(User.username == user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await User.find_one(User.email == user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    await new_user.insert()
    
    return APIResponse(
        status="success",
        message="User registered successfully",
        data={
            "user": {
                "id": str(new_user.id),
                "username": new_user.username,
                "email": new_user.email,
                "created_at": new_user.created_at.isoformat()
            }
        }
    )


@router.post("/login", response_model=APIResponse)
async def login_user(user_credentials: UserLogin):
    """
    Login user and get JWT token
    
    - **email**: User's email
    - **password**: User's password
    """
    # Find user by email
    user = await User.find_one(User.email == user_credentials.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    user_response = UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        created_at=user.created_at
    )
    
    token_data = Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )
    
    return APIResponse(
        status="success",
        message="Login successful",
        data=token_data.dict()
    )
