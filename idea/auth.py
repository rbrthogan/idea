# Auth module for Firebase Authentication
# Provides middleware and utilities for user authentication

import logging
import os
from typing import Optional
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Firebase Admin SDK
import firebase_admin
from firebase_admin import auth, credentials

# Initialize Firebase Admin SDK
_firebase_app = None
logger = logging.getLogger(__name__)

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
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
LOCAL_REQUEST_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in TRUE_ENV_VALUES


def _local_dev_auth_enabled() -> bool:
    return _env_flag("IDEA_DEV_AUTH")


def _local_host_name(value: Optional[str]) -> str:
    if not value:
        return ""
    host = value.split(",", 1)[0].strip().lower()
    if host.startswith("[") and "]" in host:
        return host[1:host.index("]")]
    return host.split(":", 1)[0]


def _is_local_request(request: Optional[Request]) -> bool:
    if request is None:
        return False

    client_host = getattr(getattr(request, "client", None), "host", None)
    if client_host:
        return _local_host_name(client_host) in LOCAL_REQUEST_HOSTS

    return _local_host_name(request.headers.get("host")) in LOCAL_REQUEST_HOSTS


def _dev_auth_allowed(request: Optional[Request]) -> bool:
    return _local_dev_auth_enabled() and _is_local_request(request)


def get_dev_auth_token() -> str:
    return os.getenv("IDEA_DEV_AUTH_TOKEN", "local-dev-token")


def get_dev_user_info() -> UserInfo:
    email = os.getenv("IDEA_DEV_AUTH_EMAIL", "local-dev@example.local")
    return UserInfo(
        uid=os.getenv("IDEA_DEV_AUTH_UID", "local-dev-user"),
        email=email,
        display_name=os.getenv("IDEA_DEV_AUTH_NAME", "Local Dev"),
        is_admin=_env_flag("IDEA_DEV_AUTH_ADMIN", "1") or email in ADMIN_EMAILS,
    )


def get_dev_auth_client_config(request: Optional[Request] = None) -> dict:
    if not _dev_auth_allowed(request):
        return {"enabled": False}

    user = get_dev_user_info()
    return {
        "enabled": True,
        "token": get_dev_auth_token(),
        "uid": user.uid,
        "email": user.email,
        "displayName": user.display_name,
        "isAdmin": user.is_admin,
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> Optional[UserInfo]:
    """
    Dependency to get the current authenticated user.
    Returns None if no valid token is provided.
    """
    if _dev_auth_allowed(request):
        if credentials is None or credentials.credentials == get_dev_auth_token():
            return get_dev_user_info()

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
    except Exception:
        logger.warning("Token verification failed", exc_info=True)
        return None


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> UserInfo:
    """
    Dependency that requires authentication.
    Raises 401 if not authenticated.
    """
    try:
        user = await get_current_user(credentials, request)
        if user is None:
            if credentials is None:
                raise HTTPException(status_code=401, detail="Authentication required")
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
