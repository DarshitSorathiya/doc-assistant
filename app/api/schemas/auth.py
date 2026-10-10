# ============================================================
# Auth Pydantic Schemas — request/response models
# ============================================================

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    """Request body for POST /api/v1/auth/register."""
    email: str = Field(
        ...,
        min_length=5,
        max_length=255,
        description="User email address",
        examples=["user@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 characters)",
        examples=["SecurePass123!"],
    )
    role: Optional[str] = Field(
        default=None,
        description="Role: admin, user, or viewer. Defaults to 'user'. "
                    "Only admins can set role to 'admin'.",
    )


class UserResponse(BaseModel):
    """Public representation of a user (never includes password)."""
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Login
# ------------------------------------------------------------------
class UserLoginRequest(BaseModel):
    """Request body for POST /api/v1/auth/login."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """JWT token response after successful login."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
