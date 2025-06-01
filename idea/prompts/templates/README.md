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

# Optional template-specific requirements
format_requirements: |
  Special formatting requirements that can be interpolated into prompts
  using {format_requirements} placeholder

design_requirements: |
  Special design requirements that can be interpolated into prompts
  using {design_requirements} placeholder

prompts:
  context: |
    Prompt for generating contextual concepts/inspiration

  idea: |
    Prompt for generating initial ideas
    Can reference {format_requirements} or {design_requirements}

  new_idea: |
    Prompt for generating new ideas from existing ones

  format: |
    Prompt for formatting raw ideas
    Must include {input_text} placeholder

  critique: |
    Prompt for critiquing ideas
    Must include {idea} placeholder

  refine: |
    Prompt for refining ideas based on critique
    Must include {idea} and {critique} placeholders

  breed: |
    Prompt for breeding/combining ideas
    Must include {ideas} placeholder

comparison_criteria:
  - "First criterion (most important)"
  - "Second criterion"
  - "Additional criteria..."
```

## Available Templates

- **drabble.yaml**: Generate 100-word short stories
- **airesearch.yaml**: Generate AI research proposals
- **game_design.yaml**: Generate browser game design concepts

## Creating New Templates

1. Copy an existing template as a starting point
2. Modify the metadata and prompts for your domain
3. Test your template using the validation tools
4. Save with a descriptive filename (e.g., `business_ideas.yaml`)

## Validation

Templates are automatically validated when loaded. The system checks for:

- Required fields and structure
- Proper placeholder usage in prompts
- Valid semantic versioning
- Proper date formatting

## Template Placeholders

### Required Placeholders
- `{input_text}` in format prompts
- `{idea}` in critique prompts
- `{idea}` and `{critique}` in refine prompts
- `{ideas}` in breed prompts

### Optional Placeholders
- `{format_requirements}` - Interpolates format_requirements field
- `{design_requirements}` - Interpolates design_requirements field

## Using Templates

Templates are automatically loaded by the system when available. The loader will:

1. Look for YAML templates first
2. Fall back to Python modules if no YAML template exists
3. Validate templates on loading
4. Report any warnings or errors

## Contributing Templates

To contribute a new template:

1. Create your YAML template following the structure above
2. Test it thoroughly with the validation system
3. Submit a pull request with your template
4. Include examples of the types of ideas it generates

## Backward Compatibility

The YAML system is fully backward compatible with existing Python prompt modules. The system will automatically use YAML templates when available, but fall back to Python modules for existing functionality.