"""Resolve model credentials without exposing server keys to production users."""
import os
from typing import Optional

from starlette.requests import Request

from idea import database as db
from idea.auth import UserInfo, get_dev_auth_client_config


async def user_provider_credential(user: UserInfo, provider: str, request: Optional[Request] = None):
    stored = await db.get_user_api_key(user.uid, provider)
    if stored:
        return stored, "stored"
    config = get_dev_auth_client_config(request)
    if config.get("enabled") and config.get("uid") == user.uid:
        variable = {"google_generative_ai": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
        key = os.getenv(variable, "").strip() if variable else ""
        if key:
            return key, "environment"
    return None, "missing"


async def resolve_user_provider_key(user: UserInfo, provider: str, request: Optional[Request] = None):
    return (await user_provider_credential(user, provider, request))[0]
