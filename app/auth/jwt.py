# ============================================================
# JWT Token Management
# ============================================================
# Creates and verifies JWT access tokens.
# Also provides FastAPI dependencies:
#   - get_current_user: extracts the authenticated user from the token
#   - require_role(*roles): enforces RBAC on specific endpoints
# ============================================================

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database import get_db
from app.models import User, UserRole

# ------------------------------------------------------------------
# OAuth2 scheme — tells FastAPI to look for "Authorization: Bearer ..."
# ------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ------------------------------------------------------------------
# Token creation
# ------------------------------------------------------------------
def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        data: Payload dict — must include ``sub`` (user email).
        expires_delta: Optional custom expiry. Defaults to settings value.

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


# ------------------------------------------------------------------
# Token verification & user extraction
# ------------------------------------------------------------------
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: decode the JWT and return the authenticated User.

    Raises:
        HTTPException 401 if the token is invalid, expired, or the user
        doesn't exist / is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


get_current_active_user = get_current_user


# ------------------------------------------------------------------
# RBAC dependency
# ------------------------------------------------------------------
def require_role(*allowed_roles: UserRole):
    """Return a FastAPI dependency that enforces role-based access.

    Usage:

        @router.get("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
        def admin_endpoint():
            ...

    Or inject the user directly:

        @router.get("/admin-data")
        def admin_data(user: User = Depends(require_role(UserRole.ADMIN))):
            ...
    """

    def _role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role.value}' is not authorized. "
                       f"Required: {[r.value for r in allowed_roles]}",
            )
        return current_user

    return _role_checker
