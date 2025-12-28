# Auth module for Firebase Authentication
# Provides middleware and utilities for user authentication

import os
from functools import wraps
from typing import Optional, Callable
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Firebase Admin SDK
import firebase_admin
from firebase_admin import auth, credentials

# Initialize Firebase Admin SDK
_firebase_app = None

def get_firebase_app():
    """Get or initialize the Firebase Admin app."""
    global _firebase_app
    if _firebase_app is None:
        # In Cloud Run, this uses Application Default Credentials
        # Locally, set GOOGLE_APPLICATION_CREDENTIALS to a service account key
        try:
            _firebase_app = firebase_admin.get_app()
        except ValueError:
            # App not initialized yet
            cred = credentials.ApplicationDefault()
            _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app


# Security scheme for extracting Bearer tokens
security = HTTPBearer(auto_error=False)


class UserInfo:
    """Represents an authenticated user."""
    def __init__(self, uid: str, email: Optional[str] = None,
                 display_name: Optional[str] = None, is_admin: bool = False):
        self.uid = uid
        self.email = email
        self.display_name = display_name
        self.is_admin = is_admin

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "email": self.email,
            "display_name": self.display_name,
            "is_admin": self.is_admin
        }


# Admin email addresses (can also be stored in Firestore for dynamic updates)
ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "").split(",")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[UserInfo]:
    """
    Dependency to get the current authenticated user.
    Returns None if no valid token is provided.
    """
    if credentials is None:
        return None

    try:
        get_firebase_app()
        # Verify the ID token
        decoded_token = auth.verify_id_token(credentials.credentials)

        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        display_name = decoded_token.get("name")

        # Check if user is admin
        is_admin = email in ADMIN_EMAILS if email else False

        return UserInfo(
            uid=uid,
            email=email,
            display_name=display_name,
            is_admin=is_admin
        )
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInfo:
    """
    Dependency that requires authentication.
    Raises 401 if not authenticated.
    """
    try:
        if credentials is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        user = await get_current_user(credentials)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return user
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Auth error: {str(e)}")


async def require_admin(
    user: UserInfo = Depends(require_auth)
) -> UserInfo:
    """
    Dependency that requires admin privileges.
    Raises 403 if not an admin.
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def get_user_from_request(request: Request) -> Optional[UserInfo]:
    """
    Utility to get user info attached to a request.
    Use this in routes where auth is optional.
    """
    return getattr(request.state, "user", None)
