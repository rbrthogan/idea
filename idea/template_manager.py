"""
Template Manager API endpoints for CRUD operations on YAML prompt templates
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import yaml
import os
from datetime import datetime

from idea.prompts.loader import list_available_templates, validate_template
from idea.prompts.validation import validate_template_file, TemplateValidator

# Create router for template management
router = APIRouter(prefix="/api/templates", tags=["templates"])

# Get templates directory
TEMPLATES_DIR = Path(__file__).parent / "prompts" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


class TemplateCreateRequest(BaseModel):
    """Request model for creating a new template"""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    author: str = Field(default="User", description="Template author")
    item_type: str = Field(..., description="Type of items generated")
    special_requirements: Optional[str] = Field(None, description="Special requirements for this template type")
    context_prompt: str = Field(..., description="Context generation prompt")
    idea_prompt: str = Field(..., description="Idea generation prompt")
    new_idea_prompt: str = Field(..., description="New idea generation prompt")
    format_prompt: str = Field(..., description="Format prompt")
    critique_prompt: str = Field(..., description="Critique prompt")
    refine_prompt: str = Field(..., description="Refine prompt")
    breed_prompt: str = Field(..., description="Breed prompt")
    comparison_criteria: List[str] = Field(..., description="Comparison criteria")


class TemplateUpdateRequest(BaseModel):
    """Request model for updating a template"""
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    item_type: Optional[str] = None
    special_requirements: Optional[str] = None
    context_prompt: Optional[str] = None
    idea_prompt: Optional[str] = None
    new_idea_prompt: Optional[str] = None
    format_prompt: Optional[str] = None
    critique_prompt: Optional[str] = None
    refine_prompt: Optional[str] = None
    breed_prompt: Optional[str] = None
    comparison_criteria: Optional[List[str]] = None


def get_template_starter() -> Dict[str, Any]:
    """Get a starter template for new templates"""
    return {
        "name": "Custom Template",
        "description": "A custom prompt template for generating ideas",
        "version": "1.0.0",
        "author": "User",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "metadata": {
            "item_type": "custom ideas"
        },
        "special_requirements": "Add any special constraints, formatting rules, or requirements here.\n"
                              "These will be inserted into your prompts wherever you use {requirements}.\n"
                              "For example: 'Must be exactly 100 words' or 'Should be implementable as a browser game'.",
        "prompts": {
            "context": "Generate a list of 50 concepts relevant to the domain. These concepts should include:\n"
                      "techniques, methods, objects, themes, styles, or approaches.\n"
                      "Return the list as: CONCEPTS:<concept1>, <concept2>, <concept3>...",
            "idea": "Using the above context for inspiration, generate a creative and innovative idea.\n"
                   "Keep it detailed enough to be useful but concise enough to be clear.\n"
                   "\n{requirements}",
            "new_idea": "You are given the preceding list of ideas.\n"
                       "Considering these ideas, propose a new idea that could be completely new\n"
                       "or could combine elements from the existing ideas.\n"
                       "Please avoid minor refinements and create something that is a significant departure.\n"
                       "\n{requirements}",
            "format": "Take the following idea and rewrite it in a clear, structured format.\n"
                     "Ensure it has a compelling title and well-organized content: {input_text}",
            "critique": "You are an expert evaluator reviewing the following idea:\n"
                       "{idea}\n\n"
                       "Provide critical feedback, pointing out both strengths and weaknesses.\n"
                       "If the idea lacks clarity, detail, or originality, suggest specific improvements.\n"
                       "No additional text, just the critique.",
            "refine": "Current Idea:\n{idea}\n\nCritique: {critique}\n\n"
                     "Please review both the idea and critique, then create your own improved version.\n"
                     "This could be a refinement of the original or a fresh take on it.\n"
                     "No additional text, just the refined idea.\n"
                     "\n{requirements}",
            "breed": "{ideas}\n\n"
                    "You are presented with the above ideas and asked to create a new one.\n"
                    "This can be a combination of existing ideas or something completely new that they inspired.\n"
                    "Focus on originality and bringing something new to the table.\n"
                    "Think outside the box and be creative.\n"
                    "\n{requirements}"
        },
        "comparison_criteria": [
            "originality and creativity",
            "clarity and coherence",
            "practical value",
            "potential impact"
        ]
    }


@router.get("/")
async def list_templates():
    """List all available templates"""
    try:
        templates = list_available_templates()
        return JSONResponse({
            "status": "success",
            "templates": templates
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/starter")
async def get_starter_template():
    """Get a starter template for creating new templates"""
    try:
        starter = get_template_starter()
        return JSONResponse({
            "status": "success",
            "template": starter
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/{template_id}")
async def get_template(template_id: str):
    """Get a specific template by ID"""
    try:
        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        if not template_path.exists():
            raise HTTPException(status_code=404, detail="Template not found")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)

        # Get validation info
        is_valid, warnings = validate_template(template_id)

        return JSONResponse({
            "status": "success",
            "template": template_data,
            "validation": {
                "is_valid": is_valid,
                "warnings": warnings
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.post("/")
async def create_template(request: TemplateCreateRequest):
    """Create a new template"""
    try:
        # Generate template ID from name
        template_id = request.name.lower().replace(" ", "_").replace("-", "_")
        template_id = "".join(c for c in template_id if c.isalnum() or c == "_")

        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        # Check if template already exists
        if template_path.exists():
            return JSONResponse({
                "status": "error",
                "message": f"Template '{template_id}' already exists"
            }, status_code=400)

        # Build template data
        template_data = {
            "name": request.name,
            "description": request.description,
            "version": "1.0.0",
            "author": request.author,
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "metadata": {
                "item_type": request.item_type
            },
            "prompts": {
                "context": request.context_prompt,
                "idea": request.idea_prompt,
                "new_idea": request.new_idea_prompt,
                "format": request.format_prompt,
                "critique": request.critique_prompt,
                "refine": request.refine_prompt,
                "breed": request.breed_prompt
            },
            "comparison_criteria": request.comparison_criteria
        }

        # Add optional requirements
        if request.special_requirements:
            template_data["special_requirements"] = request.special_requirements

        # Validate template before saving
        try:
            TemplateValidator.validate_dict(template_data)
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": f"Template validation failed: {e}"
            }, status_code=400)

        # Save template
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, sort_keys=False, indent=2)

        return JSONResponse({
            "status": "success",
            "message": f"Template '{template_id}' created successfully",
            "template_id": template_id
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.put("/{template_id}")
async def update_template(template_id: str, request: TemplateUpdateRequest):
    """Update an existing template"""
    try:
        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        if not template_path.exists():
            raise HTTPException(status_code=404, detail="Template not found")

        # Load existing template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)

        # Update fields that were provided
        if request.name is not None:
            template_data["name"] = request.name
        if request.description is not None:
            template_data["description"] = request.description
        if request.author is not None:
            template_data["author"] = request.author
        if request.item_type is not None:
            template_data["metadata"]["item_type"] = request.item_type

        # Update prompts
        if request.context_prompt is not None:
            template_data["prompts"]["context"] = request.context_prompt
        if request.idea_prompt is not None:
            template_data["prompts"]["idea"] = request.idea_prompt
        if request.new_idea_prompt is not None:
            template_data["prompts"]["new_idea"] = request.new_idea_prompt
        if request.format_prompt is not None:
            template_data["prompts"]["format"] = request.format_prompt
        if request.critique_prompt is not None:
            template_data["prompts"]["critique"] = request.critique_prompt
        if request.refine_prompt is not None:
            template_data["prompts"]["refine"] = request.refine_prompt
        if request.breed_prompt is not None:
            template_data["prompts"]["breed"] = request.breed_prompt

        # Update criteria
        if request.comparison_criteria is not None:
            template_data["comparison_criteria"] = request.comparison_criteria

        # Update optional requirements
        if request.special_requirements is not None:
            if request.special_requirements:
                template_data["special_requirements"] = request.special_requirements
            elif "special_requirements" in template_data:
                del template_data["special_requirements"]

        # Validate updated template
        try:
            TemplateValidator.validate_dict(template_data)
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": f"Template validation failed: {e}"
            }, status_code=400)

        # Save updated template
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, sort_keys=False, indent=2)

        return JSONResponse({
            "status": "success",
            "message": f"Template '{template_id}' updated successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    """Delete a template"""
    try:
        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        if not template_path.exists():
            raise HTTPException(status_code=404, detail="Template not found")

        # Don't allow deletion of core templates
        core_templates = {"drabble", "airesearch", "game_design"}
        if template_id in core_templates:
            return JSONResponse({
                "status": "error",
                "message": f"Cannot delete core template '{template_id}'"
            }, status_code=400)

        # Delete the template file
        template_path.unlink()

        return JSONResponse({
            "status": "success",
            "message": f"Template '{template_id}' deleted successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.post("/{template_id}/validate")
async def validate_template_endpoint(template_id: str):
    """Validate a specific template"""
    try:
        is_valid, warnings = validate_template(template_id)

        return JSONResponse({
            "status": "success",
            "validation": {
                "is_valid": is_valid,
                "warnings": warnings
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)