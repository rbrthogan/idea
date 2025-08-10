# YAML Prompt Templates

This directory contains YAML-based prompt templates for the Idea Evolution system. These templates provide a user-friendly way to create and share new prompt configurations without requiring Python programming knowledge.

## Template Structure

Each YAML template follows this structure:

```yaml
name: "Template Name"
description: "What this template generates"
version: "1.0.0"  # Semantic versioning
author: "Your Name"
created_date: "2024-01-15"  # YYYY-MM-DD format

metadata:
  item_type: "Type of items generated (e.g., 'stories', 'game designs')"

# Special requirements for this template type (optional)
special_requirements: |
  Special constraints, formatting rules, or requirements for this template.
  These will be automatically inserted into prompts using {requirements} placeholder.
  For example: word count limits, complexity constraints, formatting rules, etc.

prompts:
  context: |
    Prompt for generating contextual concepts/inspiration

  idea: |
    Prompt for generating initial ideas
    Can reference {requirements} for template-specific constraints

  # (No separate new_idea key; new ideas during breeding are produced via
  # context -> specific_prompt -> idea pipeline)

  format: |
    Prompt for formatting raw ideas
    Must include {input_text} placeholder

  critique: |
    Prompt for critiquing ideas
    Must include {idea} placeholder

  refine: |
    Prompt for refining ideas based on critique
    Must include {idea} and {critique} placeholders
    Can reference {requirements} for template-specific constraints

  breed: |
    Prompt for breeding/combining ideas
    Must include {ideas} placeholder
    Can reference {requirements} for template-specific constraints

comparison_criteria:
  - "First criterion (most important)"
  - "Second criterion"
  - "Additional criteria..."
```

## Available Templates

- **drabble.yaml**: Generate 100-word short stories
- **airesearch.yaml**: Generate GPU poor AI research proposals
- **game_design.yaml**: Generate browser game design concepts

## Creating New Templates

1. Use the Template Manager UI at `/templates` for an easy visual interface
2. Or copy an existing template as a starting point
3. Modify the metadata and prompts for your domain
4. Use the `special_requirements` field for template-specific constraints
5. Test your template using the validation tools
6. Save with a descriptive filename (e.g., `business_ideas.yaml`)

## Validation

Templates are automatically validated when loaded. The system checks for:

- Required fields and structure
- Proper placeholder usage in prompts
- Valid semantic versioning
- Proper date formatting

## Template Placeholders (required at authoring time)

### Required Placeholders
- `{input_text}` in format prompts
- `{idea}` in critique prompts
- `{idea}` and `{critique}` in refine prompts
- `{ideas}` in breed prompts

### Optional Interpolation
- `{requirements}` in idea/refine/breed prompts is replaced with the `special_requirements` block if present

## Special Requirements

The `special_requirements` field is used to specify template-specific constraints. Examples:

- **Story templates**: "Must be exactly 100 words with a complete narrative arc"
- **Game templates**: "Should be implementable as a simple browser game"
- **Business templates**: "Focus on B2B SaaS solutions with clear revenue models"
- **Research templates**: "Should include methodology, expected outcomes, and feasibility assessment"

Use the `{requirements}` placeholder in your prompts to automatically include these constraints.

## Using Templates

Templates are automatically loaded by the system when available. The loader will:

1. Look for YAML templates first
2. Fall back to Python modules if no YAML template exists
3. Validate templates on loading
4. Report any warnings or errors

## Contributing Templates

To contribute a new template:

1. Create your YAML template following the structure above
2. Use the Template Manager UI for easy creation and testing
3. Test it thoroughly with the validation system
4. Submit a pull request with your template
5. Include examples of the types of ideas it generates

## Backward Compatibility

The YAML system is fully backward compatible with existing Python prompt modules. The system will automatically use YAML templates when available, but fall back to Python modules for existing functionality.