# Admin module for monitoring user activity
# Protected routes for administrators only

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from idea.auth import require_admin, UserInfo
from idea import database as db

router = APIRouter(prefix="/admin", tags=["admin"])

templates = Jinja2Templates(directory="idea/static/html")


@router.get("/")
@router.get("/")
async def admin_dashboard(request: Request):
    """Serve the admin dashboard page."""
    return templates.TemplateResponse("admin.html", {
        "request": request
    })


@router.get("/api/users")
async def list_users(user: UserInfo = Depends(require_admin)):
    """Get summary of all users."""
    try:
        users = await db.get_all_users_summary()
        return JSONResponse({
            "status": "success",
            "users": users,
            "total": len(users)
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/api/stats")
async def get_stats(user: UserInfo = Depends(require_admin)):
    """Get overall platform statistics."""
    try:
        users = await db.get_all_users_summary()

        total_evolutions = sum(u.get("evolution_count", 0) for u in users)
        total_templates = sum(u.get("template_count", 0) for u in users)
        users_with_api_key = sum(1 for u in users if u.get("has_api_key"))

        return JSONResponse({
            "status": "success",
            "stats": {
                "total_users": len(users),
                "users_with_api_key": users_with_api_key,
                "total_evolutions": total_evolutions,
                "total_templates": total_templates
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/api/user/{user_id}/evolutions")
async def get_user_evolutions(user_id: str, admin: UserInfo = Depends(require_admin)):
    """Get evolutions for a specific user (admin view)."""
    try:
        evolutions = await db.list_evolutions(user_id, limit=100)
        return JSONResponse({
            "status": "success",
            "evolutions": evolutions,
            "user_id": user_id
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
