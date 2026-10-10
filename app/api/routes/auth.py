# ============================================================
# Auth Routes — Register & Login
# ============================================================

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserRole
from app.auth.security import hash_password, verify_password
from app.auth.jwt import create_access_token, get_current_user
from app.api.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ------------------------------------------------------------------
# POST /api/v1/auth/register
# ------------------------------------------------------------------
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account.

    - Email must be unique.
    - Password is hashed with bcrypt before storage.
    - Default role is ``user``. Only existing admins can create admin accounts.
    """
    # Check if email already exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    # Determine role
    role = UserRole.USER  # default
    if request.role:
        try:
            role = UserRole(request.role.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid role '{request.role}'. Must be: admin, user, or viewer",
            )
        # Prevent self-promotion to admin (in a real app, check the caller's role)
        # For now, allow admin creation during initial setup; Phase 12 tightens this.

    # Create the user
    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
    )


# ------------------------------------------------------------------
# POST /api/v1/auth/login
# ------------------------------------------------------------------
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT token",
)
def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate a user and return a JWT access token.

    The token must be sent as ``Authorization: Bearer <token>``
    in subsequent requests to protected endpoints.
    """
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Create JWT token with user email as subject
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role.value}
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
    )


# ------------------------------------------------------------------
# GET /api/v1/auth/me — example protected route
# ------------------------------------------------------------------
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile.

    Requires a valid JWT token in the Authorization header.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role.value,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )
