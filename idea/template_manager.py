"""
Template Manager API endpoints for CRUD operations on YAML prompt templates
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import yaml
from datetime import datetime

from idea.prompts.loader import list_available_templates, validate_template
from idea.prompts.validation import TemplateValidator
from idea.llm import LLMWrapper
from idea.auth import require_auth, UserInfo
from idea import database as db

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
    specific_prompt: str = Field(..., description="Specific prompt generation from context")
    idea_prompt: str = Field(..., description="Idea generation prompt")
    format_prompt: str = Field(..., description="Format prompt")
    critique_prompt: str = Field(..., description="Critique prompt")
    refine_prompt: str = Field(..., description="Refine prompt")
    breed_prompt: str = Field(..., description="Breed prompt")
    genotype_encode_prompt: str = Field(..., description="Genotype encoding prompt")
    comparison_criteria: List[str] = Field(..., description="Comparison criteria")


class TemplateUpdateRequest(BaseModel):
    """Request model for updating a template"""
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    item_type: Optional[str] = None
    special_requirements: Optional[str] = None
    context_prompt: Optional[str] = None
    specific_prompt: Optional[str] = None
    idea_prompt: Optional[str] = None
    format_prompt: Optional[str] = None
    critique_prompt: Optional[str] = None
    refine_prompt: Optional[str] = None
    breed_prompt: Optional[str] = None
    genotype_encode_prompt: Optional[str] = None
    comparison_criteria: Optional[List[str]] = None


class TemplateGenerateRequest(BaseModel):
    """Request model for generating a draft template"""
    idea_type_suggestion: str = Field(..., description="Brief description of the idea type to generate a template for")


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
            "specific_prompt": "From the following concepts, randomly select 2 or 3 that would create an interesting combination:\n\n"
                              "{context_pool}\n\n"
                              "Create a brief, focused prompt that combines these concepts in an interesting way.\n"
                              "Your concise prompt:",
            "idea": "Using the above context for inspiration, generate a creative and innovative idea.\n"
                   "Keep it detailed enough to be useful but concise enough to be clear.\n"
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
                    "\n{requirements}",
            "genotype_encode": "You are an expert at analyzing ideas and extracting their fundamental building blocks.\n"
                              "Your task is to convert a full idea into its basic genetic elements (genotype).\n\n"
                              "Extract the essential components that define this idea - strip away specific details and examples.\n"
                              "Think of it as the 'DNA' of the idea.\n\n"
                              "Format the genotype as a condensed list of basic elements, separated by semicolons.\n\n"
                              "Idea to encode:\n{idea_content}\n\nGenotype:"
        },
        "comparison_criteria": [
            "originality and creativity",
            "clarity and coherence",
            "practical value",
            "potential impact"
        ]
    }


@router.get("/")
async def list_templates(user: UserInfo = Depends(require_auth)):
    """List all available templates (system + user templates)"""
    try:
        # Get system templates from bundled YAML files
        system_templates = list_available_templates()

        # Mark system templates
        for template_id, template_info in system_templates.items():
            if 'error' not in template_info:
                template_info['is_system'] = True

        # Get user's custom templates from Firestore
        user_templates = await db.list_user_templates(user.uid)

        # Add user templates to the result (with is_system=False)
        for ut in user_templates:
            template_id = ut.get('id')
            # Don't override system templates
            if template_id not in system_templates:
                system_templates[template_id] = {
                    **ut,
                    'is_system': False
                }

        return JSONResponse({
            "status": "success",
            "templates": system_templates
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
async def get_template(template_id: str, user: UserInfo = Depends(require_auth)):
    """Get a specific template by ID (checks system templates then user templates)"""
    try:
        # First check system templates (bundled YAML files)
        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            template_data['is_system'] = True

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

        # Check user's templates in Firestore
        user_template = await db.get_user_template(user.uid, template_id)
        if user_template:
            user_template['is_system'] = False
            return JSONResponse({
                "status": "success",
                "template": user_template,
                "validation": {"is_valid": True, "warnings": []}
            })

        raise HTTPException(status_code=404, detail="Template not found")

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.post("/")
async def create_template(request: TemplateCreateRequest, user: UserInfo = Depends(require_auth)):
    """Create a new template (saved to Firestore for the user)"""
    try:
        # Generate template ID from name
        template_id = request.name.lower().replace(" ", "_").replace("-", "_")
        template_id = "".join(c for c in template_id if c.isalnum() or c == "_")

        # Check if user already has a template with this ID
        existing = await db.get_user_template(user.uid, template_id)
        if existing:
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
                "specific_prompt": request.specific_prompt,
                "idea": request.idea_prompt,
                "format": request.format_prompt,
                "critique": request.critique_prompt,
                "refine": request.refine_prompt,
                "breed": request.breed_prompt,
                "genotype_encode": request.genotype_encode_prompt
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

        # Save template to Firestore
        await db.save_user_template(user.uid, template_id, template_data)

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
        if request.specific_prompt is not None:
            template_data["prompts"]["specific_prompt"] = request.specific_prompt
        if request.idea_prompt is not None:
            template_data["prompts"]["idea"] = request.idea_prompt
        if request.format_prompt is not None:
            template_data["prompts"]["format"] = request.format_prompt
        if request.critique_prompt is not None:
            template_data["prompts"]["critique"] = request.critique_prompt
        if request.refine_prompt is not None:
            template_data["prompts"]["refine"] = request.refine_prompt
        if request.breed_prompt is not None:
            template_data["prompts"]["breed"] = request.breed_prompt
        if request.genotype_encode_prompt is not None:
            template_data["prompts"]["genotype_encode"] = request.genotype_encode_prompt
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


def get_core_templates() -> set:
    """
    Dynamically determine core templates based on their metadata or location.
    Core templates are those that come with the system by default.
    """
    core_templates = set()
    templates = list_available_templates()

    for template_id, template_info in templates.items():
        # Consider templates as core if they have "Original Idea App" as author
        # or if they're in the standard templates that ship with the system
        if (template_info.get('author') == 'Original Idea App' or
            template_id in {'drabble', 'airesearch', 'game_design'}):
            core_templates.add(template_id)

    return core_templates


@router.delete("/{template_id}")
async def delete_template(template_id: str, user: UserInfo = Depends(require_auth)):
    """Delete a user template (system templates cannot be deleted)"""
    try:
        # Don't allow deletion of core/system templates
        core_templates = get_core_templates()
        template_path = TEMPLATES_DIR / f"{template_id}.yaml"

        if template_id in core_templates or template_path.exists():
            return JSONResponse({
                "status": "error",
                "message": f"Cannot delete system template '{template_id}'"
            }, status_code=400)

        # Delete the user's template from Firestore
        deleted = await db.delete_user_template(user.uid, template_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Template not found")

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


@router.post("/generate")
async def generate_template(request: TemplateGenerateRequest, user: UserInfo = Depends(require_auth)):
    """Generate a draft template using Gemini 3 Pro based on user's idea type suggestion"""
    try:
        # Get user's API key
        api_key = await db.get_user_api_key(user.uid)
        if not api_key:
            return JSONResponse({
                "status": "error",
                "message": "API Key not configured. Please set it in Settings."
            }, status_code=400)

        # Get the existing 3 core templates as examples
        example_templates = {}
        for template_id in ['airesearch', 'drabble', 'game_design']:
            template_path = TEMPLATES_DIR / f"{template_id}.yaml"
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    example_templates[template_id] = yaml.safe_load(f)

        if len(example_templates) < 3:
            return JSONResponse({
                "status": "error",
                "message": "Unable to load all required example templates"
            }, status_code=500)

        # Create the few-shot prompt using existing templates
        prompt = create_template_generation_prompt(request.idea_type_suggestion, example_templates)

        # Use Gemini 2.5 Pro to generate the template
        llm = LLMWrapper(
            provider="google_generative_ai",
            model_name="gemini-3-pro-preview",
            temperature=0.7,  # Balanced creativity and consistency
            top_p=0.9,
            api_key=api_key
        )

        response = llm.generate_text(prompt)

        # Parse the response into a template structure
        generated_template = parse_generated_template(response, request.idea_type_suggestion)

        return JSONResponse({
            "status": "success",
            "template": generated_template,
            "message": "Draft template generated successfully. Please review and edit before saving."
        })

    except Exception as e:
        print(f"Error generating template: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to generate template: {str(e)}"
        }, status_code=500)


def create_template_generation_prompt(idea_type_suggestion: str, example_templates: Dict[str, Any]) -> str:
    """Create a few-shot prompt using existing templates as examples"""

    # Format the examples
    examples_text = ""
    for template_id, template_data in example_templates.items():
        examples_text += f"\n--- EXAMPLE {template_id.upper()} TEMPLATE ---\n"
        examples_text += f"Name: {template_data['name']}\n"
        examples_text += f"Description: {template_data['description']}\n"
        examples_text += f"Item Type: {template_data['metadata']['item_type']}\n"

        if 'special_requirements' in template_data:
            examples_text += f"Special Requirements: {template_data['special_requirements']}\n"

        examples_text += f"Context Prompt: {template_data['prompts']['context']}\n"
        examples_text += f"Specific Prompt: {template_data['prompts']['specific_prompt']}\n"
        examples_text += f"Idea Prompt: {template_data['prompts']['idea']}\n"
        examples_text += f"Format Prompt: {template_data['prompts']['format']}\n"
        examples_text += f"Critique Prompt: {template_data['prompts']['critique']}\n"
        examples_text += f"Refine Prompt: {template_data['prompts']['refine']}\n"
        examples_text += f"Breed Prompt: {template_data['prompts']['breed']}\n"
        examples_text += f"Genotype Encode Prompt: {template_data['prompts']['genotype_encode']}\n"
        examples_text += f"Comparison Criteria: {', '.join(template_data['comparison_criteria'])}\n"

    # Create the main prompt
    prompt = f"""You are an expert at creating prompt templates for AI idea generation systems. I will show you three existing templates as examples, and then ask you to create a new template based on a user's idea type suggestion.

Here are the existing template examples:
{examples_text}

Now, please create a new template for the following idea type:
"{idea_type_suggestion}"

Guidelines:
1. Follow the same structure as the examples above
2. Create prompts that are specific to the requested idea type
3. Make the context prompt generate 50 relevant concepts for that domain
4. Make the specific_prompt create focused prompts from the context pool using {{context_pool}} placeholder
5. Ensure the idea prompt incorporates context and includes {{requirements}} placeholder
6. Make the format prompt include {{input_text}} placeholder
7. Make the critique prompt include {{idea}} placeholder
8. Make the refine prompt include {{idea}} and {{critique}} placeholders
9. Make the breed prompt include {{ideas}} placeholder
10. Make the genotype_encode prompt extract fundamental elements using {{idea_content}} placeholder
11. Choose appropriate comparison criteria for evaluating ideas in this domain
12. If the idea type has specific constraints (like word limits, format requirements, etc.), include them in special_requirements

Please provide your response in this exact format:

Name: [Template name]
Description: [Template description]
Item Type: [Type of items generated]
Special Requirements: [Any special constraints or requirements, or "None" if not applicable]
Context Prompt: [Context generation prompt]
Specific Prompt: [Specific prompt creation - must include {{context_pool}}]
Idea Prompt: [Idea generation prompt - must include {{requirements}}]
Format Prompt: [Format prompt - must include {{input_text}}]
Critique Prompt: [Critique prompt - must include {{idea}}]
Refine Prompt: [Refine prompt - must include {{idea}} and {{critique}}]
Breed Prompt: [Breed prompt - must include {{ideas}}]
Genotype Encode Prompt: [Genotype encoding prompt - must include {{idea_content}}]
Comparison Criteria: [Comma-separated list of criteria]

Be creative and adapt the prompts to be highly relevant to the specific idea type requested."""

    return prompt


def parse_generated_template(response: str, idea_type_suggestion: str) -> Dict[str, Any]:
    """Parse the LLM response into a template structure"""

    template = {
        "name": "Generated Template",
        "description": f"AI-generated template for {idea_type_suggestion}",
        "version": "1.0.0",
        "author": "AI Assistant",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "metadata": {
            "item_type": idea_type_suggestion
        },
        "prompts": {},
        "comparison_criteria": []
    }

    try:
        lines = response.strip().split('\n')
        current_field = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for field headers
            if line.startswith('Name:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'name'
                current_content = [line[5:].strip()]
            elif line.startswith('Description:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'description'
                current_content = [line[12:].strip()]
            elif line.startswith('Item Type:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'item_type'
                current_content = [line[10:].strip()]
            elif line.startswith('Special Requirements:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'special_requirements'
                current_content = [line[21:].strip()]
            elif line.startswith('Context Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'context_prompt'
                current_content = [line[15:].strip()]
            elif line.startswith('Specific Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'specific_prompt'
                current_content = [line[16:].strip()]
            elif line.startswith('Idea Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'idea_prompt'
                current_content = [line[12:].strip()]
            elif line.startswith('Format Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'format_prompt'
                current_content = [line[14:].strip()]
            elif line.startswith('Critique Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'critique_prompt'
                current_content = [line[16:].strip()]
            elif line.startswith('Refine Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'refine_prompt'
                current_content = [line[14:].strip()]
            elif line.startswith('Breed Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'breed_prompt'
                current_content = [line[13:].strip()]
            elif line.startswith('Genotype Encode Prompt:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'genotype_encode_prompt'
                current_content = [line[23:].strip()]
            elif line.startswith('Comparison Criteria:'):
                if current_field:
                    _save_field(template, current_field, current_content)
                current_field = 'comparison_criteria'
                current_content = [line[20:].strip()]
            else:
                # Continuation of current field
                if current_field and line:
                    current_content.append(line)

        # Save the last field
        if current_field:
            _save_field(template, current_field, current_content)

    except Exception as e:
        print(f"Error parsing generated template: {e}")
        # Return a basic template with error indication
        template["name"] = "Generated Template (Parse Error)"
        template["description"] = f"AI-generated template for {idea_type_suggestion} - Please review and edit"

    return template


def _save_field(template: Dict[str, Any], field: str, content: List[str]):
    """Helper function to save parsed field content to template"""
    content_str = '\n'.join(content).strip()

    if field == 'name':
        template['name'] = content_str
    elif field == 'description':
        template['description'] = content_str
    elif field == 'item_type':
        template['metadata']['item_type'] = content_str
    elif field == 'special_requirements':
        if content_str.lower() not in ['none', 'n/a', 'not applicable', '']:
            template['special_requirements'] = content_str
    elif field == 'context_prompt':
        template['prompts']['context'] = content_str
    elif field == 'specific_prompt':
        template['prompts']['specific_prompt'] = content_str
    elif field == 'idea_prompt':
        template['prompts']['idea'] = content_str
    elif field == 'format_prompt':
        template['prompts']['format'] = content_str
    elif field == 'critique_prompt':
        template['prompts']['critique'] = content_str
    elif field == 'refine_prompt':
        template['prompts']['refine'] = content_str
    elif field == 'breed_prompt':
        template['prompts']['breed'] = content_str
    elif field == 'genotype_encode_prompt':
        template['prompts']['genotype_encode'] = content_str
    elif field == 'comparison_criteria':
        # Split by comma and clean up
        criteria = [c.strip() for c in content_str.split(',') if c.strip()]
        template['comparison_criteria'] = criteria