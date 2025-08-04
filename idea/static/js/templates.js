/**
 * Enhanced Template Management JavaScript with Unified UX
 */

let currentTemplateId = null;
let editMode = false;
let allTemplates = {};
let currentFilter = 'all';
let currentSearch = '';

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    showLoadingState();
    loadTemplates();

    // Check for pending template from main interface
    checkForPendingTemplate();
});

/**
 * Show loading state
 */
function showLoadingState() {
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('templatesGrid').style.display = 'none';
    document.getElementById('emptyState').style.display = 'none';
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    document.getElementById('loadingState').style.display = 'none';
}

/**
 * Determine if a template is a core template based on its metadata
 */
function isCoreTemplate(templateId, templateInfo) {
    // A template is considered core if:
    // 1. It's authored by "Original Idea App"
    // 2. It's one of the original templates that shipped with the system
    return templateInfo.author === 'Original Idea App' ||
           ['drabble', 'airesearch', 'game_design'].includes(templateId);
}

/**
 * Load and display all templates with enhanced UX
 */
async function loadTemplates() {
    try {
        const response = await fetch('/api/templates/');
        const data = await response.json();

        if (data.status === 'success') {
            allTemplates = data.templates;
            displayTemplates(allTemplates);
            updateStatistics();
            hideLoadingState();

            // Add slight delay for smooth animation
            setTimeout(() => {
                document.getElementById('templatesGrid').style.display = 'flex';
            }, 100);
        } else {
            showAlert('Error loading templates: ' + data.message, 'danger');
            hideLoadingState();
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading templates:', error);
        showAlert('Error loading templates: ' + error.message, 'danger');
        hideLoadingState();
        showEmptyState();
    }
}

/**
 * Display templates with enhanced styling and animations
 */
function displayTemplates(templates) {
    const filteredTemplates = filterAndSearchTemplates(templates);
    const grid = document.getElementById('templatesGrid');
    grid.innerHTML = '';

    if (Object.keys(filteredTemplates).length === 0) {
        showEmptyState();
        return;
    }

    hideLoadingState();
    hideEmptyState();

    let cardIndex = 0;
    const totalTemplates = Object.keys(templates).length;
    let coreCount = 0;
    let customCount = 0;

    for (const [templateId, templateInfo] of Object.entries(filteredTemplates)) {
        if (templateInfo.error) continue;

        const isCore = isCoreTemplate(templateId, templateInfo);

        if (isCore) {
            coreCount++;
        } else {
            customCount++;
        }

        const card = document.createElement('div');
        card.className = 'col-md-6 col-lg-4 mb-4';
        card.style.animationDelay = `${cardIndex * 0.1}s`;

        const cardClass = isCore ? 'core-template' : 'custom-template';
        const badgeClass = isCore ? 'bg-success' : 'bg-primary';

        card.innerHTML = `
            <div class="card template-card h-100 ${cardClass}">
                <div class="card-body d-flex flex-column">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <h5 class="card-title mb-0">${templateInfo.name || templateId}</h5>
                        <div>
                            <span class="badge ${badgeClass} template-badge">
                                ${isCore ? 'Core' : 'Custom'}
                            </span>
                        </div>
                    </div>

                    <p class="card-text text-muted flex-grow-1">${templateInfo.description || 'No description available'}</p>

                    <div class="template-meta mb-3">
                        <div><i class="fas fa-user me-1"></i>${templateInfo.author || 'Unknown'}</div>
                        ${templateInfo.version ? `<div><i class="fas fa-tag me-1"></i>v${templateInfo.version}</div>` : ''}
                        <div><i class="fas fa-lightbulb me-1"></i>${templateInfo.item_type || 'General'}</div>
                    </div>

                    <div class="template-actions mt-auto">
                        <div class="d-flex gap-2">
                            <button class="btn btn-outline-primary btn-sm flex-fill" onclick="viewTemplate('${templateId}')">
                                <i class="fas fa-eye me-1"></i>View
                            </button>
                            ${!isCore ? `
                                <button class="btn btn-outline-warning btn-sm" onclick="editTemplateBtn('${templateId}')" title="Edit">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button class="btn btn-outline-danger btn-sm" onclick="confirmDeleteTemplate('${templateId}')" title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;

        grid.appendChild(card);
        cardIndex++;
    }

    // Update stats
    document.getElementById('totalCount').textContent = totalTemplates;
    document.getElementById('coreCount').textContent = coreCount;
    document.getElementById('customCount').textContent = customCount;
}

/**
 * Filter and search templates
 */
function filterAndSearchTemplates(templates) {
    let filtered = {};

    for (const [templateId, templateInfo] of Object.entries(templates)) {
        if (templateInfo.error) continue;

        const isCore = isCoreTemplate(templateId, templateInfo);

        // Apply filter
        if (currentFilter === 'core' && !isCore) continue;
        if (currentFilter === 'custom' && isCore) continue;

        // Apply search
        if (currentSearch) {
            const searchTerm = currentSearch.toLowerCase();
            const name = (templateInfo.name || templateId).toLowerCase();
            const description = (templateInfo.description || '').toLowerCase();
            const author = (templateInfo.author || '').toLowerCase();

            if (!name.includes(searchTerm) &&
                !description.includes(searchTerm) &&
                !author.includes(searchTerm)) {
                continue;
            }
        }

        filtered[templateId] = templateInfo;
    }

    return filtered;
}

/**
 * Show empty state
 */
function showEmptyState() {
    document.getElementById('templatesGrid').style.display = 'none';
    document.getElementById('emptyState').style.display = 'block';
}

/**
 * Hide empty state
 */
function hideEmptyState() {
    document.getElementById('emptyState').style.display = 'none';
}

/**
 * Update statistics
 */
function updateStatistics() {
    let totalCount = 0;
    let coreCount = 0;
    let customCount = 0;

    for (const [templateId, templateInfo] of Object.entries(allTemplates)) {
        if (templateInfo.error) continue;

        totalCount++;
        if (isCoreTemplate(templateId, templateInfo)) {
            coreCount++;
        } else {
            customCount++;
        }
    }

    document.getElementById('totalCount').textContent = totalCount;
    document.getElementById('coreCount').textContent = coreCount;
    document.getElementById('customCount').textContent = customCount;
}

/**
 * Filter templates
 */
function filterTemplates() {
    currentFilter = document.getElementById('templateFilter').value;
    displayTemplates(allTemplates);
}

/**
 * Search templates
 */
function searchTemplates() {
    currentSearch = document.getElementById('templateSearch').value;
    displayTemplates(allTemplates);
}

/**
 * View template details with enhanced modal
 */
async function viewTemplate(templateId) {
    try {
        showButtonLoading('view', templateId);

        const response = await fetch(`/api/templates/${templateId}`);
        const data = await response.json();

        if (data.status === 'success') {
            currentTemplateId = templateId;
            displayTemplateDetails(data.template, data.validation);

            // Show/hide action buttons based on template type
            const coreTemplates = ['drabble', 'airesearch', 'game_design'];
            const isCore = coreTemplates.includes(templateId);
            document.getElementById('editTemplateBtn').style.display = isCore ? 'none' : 'inline-block';
            document.getElementById('deleteTemplateBtn').style.display = isCore ? 'none' : 'inline-block';

            const modal = new bootstrap.Modal(document.getElementById('templateModal'));
            modal.show();
        } else {
            showAlert('Error loading template: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error loading template:', error);
        showAlert('Error loading template: ' + error.message, 'danger');
    } finally {
        hideButtonLoading('view', templateId);
    }
}

/**
 * Display enhanced template details
 */
function displayTemplateDetails(template, validation) {
    const container = document.getElementById('templateDetails');

    let validationHtml = '';
    if (validation && (!validation.is_valid || validation.warnings.length > 0)) {
        const alertClass = validation.is_valid ? 'alert-warning' : 'alert-danger';
        const iconClass = validation.is_valid ? 'fa-exclamation-triangle' : 'fa-exclamation-circle';

        validationHtml = `
            <div class="alert ${alertClass} mb-4">
                <h6><i class="fas ${iconClass} me-2"></i>Validation Issues</h6>
                ${validation.warnings.map(warning => `<div class="ms-3">• ${warning}</div>`).join('')}
            </div>
        `;
    }

    container.innerHTML = `
        ${validationHtml}

        <div class="row mb-4">
            <div class="col-md-8">
                <h3 class="mb-2">${template.name}</h3>
                <p class="text-muted lead">${template.description}</p>
            </div>
            <div class="col-md-4">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6 class="card-title">Template Info</h6>
                        <div class="small">
                            <div class="mb-1"><strong>Version:</strong> ${template.version}</div>
                            <div class="mb-1"><strong>Author:</strong> ${template.author}</div>
                            <div class="mb-1"><strong>Created:</strong> ${template.created_date}</div>
                            <div><strong>Item Type:</strong> ${template.metadata.item_type}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        ${template.special_requirements ? `
            <div class="prompt-section p-3 mb-4">
                <h6 class="mb-3"><i class="fas fa-exclamation-circle me-2"></i>Special Requirements</h6>
                <pre class="mb-0" style="white-space: pre-wrap; font-size: 0.9rem;">${template.special_requirements}</pre>
                <div class="form-text mt-2">
                    <strong>Usage:</strong> These requirements are automatically inserted into prompts using the <code>{requirements}</code> placeholder.
                </div>
            </div>
        ` : ''}

        <h5 class="mt-5 mb-4"><i class="fas fa-brain me-2"></i>Prompts Configuration</h5>

        <div class="accordion" id="promptDetailsAccordion">
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#contextDetail">
                        <i class="fas fa-lightbulb me-2"></i>Context Prompt
                    </button>
                </h2>
                <div id="contextDetail" class="accordion-collapse collapse show">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.context}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#ideaDetail">
                        <i class="fas fa-brain me-2"></i>Idea Prompt
                    </button>
                </h2>
                <div id="ideaDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.idea}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#formatDetail">
                        <i class="fas fa-align-left me-2"></i>Format Prompt
                    </button>
                </h2>
                <div id="formatDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.format}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#critiqueDetail">
                        <i class="fas fa-search me-2"></i>Critique Prompt
                    </button>
                </h2>
                <div id="critiqueDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.critique}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#refineDetail">
                        <i class="fas fa-hammer me-2"></i>Refine Prompt
                    </button>
                </h2>
                <div id="refineDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.refine}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#breedDetail">
                        <i class="fas fa-dna me-2"></i>Breed Prompt
                    </button>
                </h2>
                <div id="breedDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.breed}</pre>
                    </div>
                </div>
            </div>

            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#genotypeEncodeDetail">
                        <i class="fas fa-code me-2"></i>Genotype Encode Prompt
                    </button>
                </h2>
                <div id="genotypeEncodeDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.genotype_encode}</pre>
                    </div>
                </div>
            </div>
        </div>

        <h5 class="mt-5 mb-3"><i class="fas fa-balance-scale me-2"></i>Comparison Criteria</h5>
        <div class="mb-3">
            ${template.comparison_criteria.map(criterion =>
                `<span class="criteria-item">${criterion}</span>`
            ).join('')}
        </div>
    `;
}

/**
 * Show create template modal with enhanced UX
 */
async function showCreateModal() {
    editMode = false;
    currentTemplateId = null;
    document.getElementById('editModalTitle').innerHTML = '<i class="fas fa-plus me-2"></i>Create New Template';

    // Show the generate section for new templates
    document.getElementById('generateSection').style.display = 'block';

    try {
        const response = await fetch('/api/templates/starter');
        const data = await response.json();

        if (data.status === 'success') {
            populateForm(data.template);
        } else {
            clearForm();
        }
    } catch (error) {
        console.error('Error loading starter template:', error);
        clearForm();
    }

    const modal = new bootstrap.Modal(document.getElementById('editModal'));
    modal.show();
}

/**
 * Edit template with enhanced feedback
 */
function editTemplate() {
    editTemplateBtn(currentTemplateId);
    bootstrap.Modal.getInstance(document.getElementById('templateModal')).hide();
}

/**
 * Show edit template modal
 */
async function editTemplateBtn(templateId) {
    editMode = true;
    currentTemplateId = templateId;
    document.getElementById('editModalTitle').innerHTML = '<i class="fas fa-edit me-2"></i>Edit Template';

    // Hide the generate section for editing existing templates
    document.getElementById('generateSection').style.display = 'none';

    try {
        const response = await fetch(`/api/templates/${templateId}`);
        const data = await response.json();

        if (data.status === 'success') {
            populateForm(data.template);
            const modal = new bootstrap.Modal(document.getElementById('editModal'));
            modal.show();
        } else {
            showAlert('Error loading template for editing: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error loading template for editing:', error);
        showAlert('Error loading template for editing: ' + error.message, 'danger');
    }
}

/**
 * Populate form with enhanced validation
 */
function populateForm(template) {
    document.getElementById('templateName').value = template.name || '';
    document.getElementById('templateDescription').value = template.description || '';
    document.getElementById('templateAuthor').value = template.author || 'User';
    document.getElementById('templateItemType').value = template.metadata?.item_type || '';

    // Handle special requirements
    document.getElementById('specialRequirements').value = template.special_requirements || '';

    document.getElementById('contextPrompt').value = template.prompts?.context || '';
    document.getElementById('specificPrompt').value = template.prompts?.specific_prompt || '';
    document.getElementById('ideaPrompt').value = template.prompts?.idea || '';
    document.getElementById('formatPrompt').value = template.prompts?.format || '';
    document.getElementById('critiquePrompt').value = template.prompts?.critique || '';
    document.getElementById('refinePrompt').value = template.prompts?.refine || '';
    document.getElementById('breedPrompt').value = template.prompts?.breed || '';
    document.getElementById('genotypeEncodePrompt').value = template.prompts?.genotype_encode || '';

    // Populate criteria
    const criteriaContainer = document.getElementById('criteriaContainer');
    criteriaContainer.innerHTML = '';

    const criteria = template.comparison_criteria || [];
    criteria.forEach((criterion, index) => {
        addCriterion(criterion);
    });

    if (criteria.length === 0) {
        addCriterion();
    }
}

/**
 * Clear form
 */
function clearForm() {
    document.getElementById('templateForm').reset();
    document.getElementById('criteriaContainer').innerHTML = '';

    // Clear generation fields
    document.getElementById('ideaTypeSuggestion').value = '';
    document.getElementById('generateStatus').style.display = 'none';

    // Reset generation section appearance
    const generateSection = document.getElementById('generateSection');
    generateSection.style.opacity = '1';
    generateSection.style.transform = 'scale(1)';

    addCriterion();
}

/**
 * Add criterion input with enhanced styling
 */
function addCriterion(value = '') {
    const container = document.getElementById('criteriaContainer');
    const index = container.children.length;

    const criterionDiv = document.createElement('div');
    criterionDiv.className = 'input-group mb-2';
    criterionDiv.innerHTML = `
        <input type="text" class="form-control" name="criterion" value="${value}"
               placeholder="e.g., originality and creativity" required>
        <button type="button" class="btn btn-outline-danger" onclick="removeCriterion(this)"
                title="Remove criterion">
            <i class="fas fa-times"></i>
        </button>
    `;

    container.appendChild(criterionDiv);
}

/**
 * Remove criterion input
 */
function removeCriterion(button) {
    const container = document.getElementById('criteriaContainer');
    if (container.children.length > 1) {
        button.parentElement.remove();
    }
}

/**
 * Save template with enhanced feedback
 */
async function saveTemplate() {
    const saveButton = document.querySelector('#editModal .modal-footer .btn-primary');
    const originalText = saveButton.innerHTML;

    try {
        // Show loading state
        saveButton.disabled = true;
        saveButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';

        const formData = {
            name: document.getElementById('templateName').value,
            description: document.getElementById('templateDescription').value,
            author: document.getElementById('templateAuthor').value,
            item_type: document.getElementById('templateItemType').value,
            special_requirements: document.getElementById('specialRequirements').value,
            context_prompt: document.getElementById('contextPrompt').value,
            specific_prompt: document.getElementById('specificPrompt').value,
            idea_prompt: document.getElementById('ideaPrompt').value,
            format_prompt: document.getElementById('formatPrompt').value,
            critique_prompt: document.getElementById('critiquePrompt').value,
            refine_prompt: document.getElementById('refinePrompt').value,
            breed_prompt: document.getElementById('breedPrompt').value,
            genotype_encode_prompt: document.getElementById('genotypeEncodePrompt').value,
            comparison_criteria: Array.from(document.querySelectorAll('input[name="criterion"]'))
                .map(input => input.value.trim())
                .filter(value => value.length > 0)
        };

        const url = editMode ? `/api/templates/${currentTemplateId}` : '/api/templates/';
        const method = editMode ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.status === 'success') {
            showAlert(data.message, 'success');
            bootstrap.Modal.getInstance(document.getElementById('editModal')).hide();

            // Reload templates with smooth transition
            showLoadingState();
            setTimeout(() => {
                loadTemplates();
            }, 300);
        } else {
            showAlert('Error saving template: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error saving template:', error);
        showAlert('Error saving template: ' + error.message, 'danger');
    } finally {
        saveButton.disabled = false;
        saveButton.innerHTML = originalText;
    }
}

/**
 * Confirm delete with enhanced modal
 */
function confirmDeleteTemplate(templateId = null) {
    if (templateId) {
        currentTemplateId = templateId;
    }

    fetch(`/api/templates/${currentTemplateId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('deleteTemplateName').textContent = data.template.name;
                const modal = new bootstrap.Modal(document.getElementById('deleteModal'));
                modal.show();
            }
        })
        .catch(error => {
            console.error('Error loading template for deletion:', error);
        });
}

/**
 * Delete template with enhanced feedback
 */
async function deleteTemplate() {
    const deleteButton = document.querySelector('#deleteModal .modal-footer .btn-danger');
    const originalText = deleteButton.innerHTML;

    try {
        deleteButton.disabled = true;
        deleteButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Deleting...';

        const response = await fetch(`/api/templates/${currentTemplateId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.status === 'success') {
            showAlert(data.message, 'success');
            bootstrap.Modal.getInstance(document.getElementById('deleteModal')).hide();

            // Close details modal if open
            const detailsModal = bootstrap.Modal.getInstance(document.getElementById('templateModal'));
            if (detailsModal) {
                detailsModal.hide();
            }

            // Reload templates with smooth transition
            showLoadingState();
            setTimeout(() => {
                loadTemplates();
            }, 300);
        } else {
            showAlert('Error deleting template: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error deleting template:', error);
        showAlert('Error deleting template: ' + error.message, 'danger');
    } finally {
        deleteButton.disabled = false;
        deleteButton.innerHTML = originalText;
    }
}

/**
 * Show button loading state
 */
function showButtonLoading(action, templateId) {
    const buttons = document.querySelectorAll(`[onclick*="${templateId}"]`);
    buttons.forEach(button => {
        if (button.innerHTML.includes('fa-eye') && action === 'view') {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
        }
    });
}

/**
 * Hide button loading state
 */
function hideButtonLoading(action, templateId) {
    const buttons = document.querySelectorAll(`[onclick*="${templateId}"]`);
    buttons.forEach(button => {
        if (button.disabled && button.innerHTML.includes('spinner')) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-eye me-1"></i>View';
        }
    });
}

/**
 * Enhanced alert system with animations
 */
function showAlert(message, type) {
    const alertContainer = document.getElementById('alertContainer');

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.style.opacity = '0';
    alertDiv.style.transform = 'translateY(-20px)';
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    alertContainer.appendChild(alertDiv);

    // Trigger animation
    requestAnimationFrame(() => {
        alertDiv.style.transition = 'all 0.3s ease';
        alertDiv.style.opacity = '1';
        alertDiv.style.transform = 'translateY(0)';
    });

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv && alertDiv.parentNode) {
            alertDiv.style.opacity = '0';
            alertDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                if (alertDiv && alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 300);
        }
    }, 5000);
}

/**
 * Generate template using AI (Gemini 2.5 Pro)
 */
async function generateTemplate() {
    const ideaTypeSuggestion = document.getElementById('ideaTypeSuggestion').value.trim();

    if (!ideaTypeSuggestion) {
        showAlert('Please describe your idea type before generating a template.', 'warning');
        document.getElementById('ideaTypeSuggestion').focus();
        return;
    }

    const generateBtn = document.getElementById('generateTemplateBtn');
    const statusDiv = document.getElementById('generateStatus');
    const statusText = document.getElementById('generateStatusText');

    // Show loading state
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating...';
    statusDiv.style.display = 'block';
    statusDiv.className = 'alert alert-info mt-3';
    statusText.textContent = 'Generating template with Gemini 2.5 Pro... This may take a moment.';

    try {
        const response = await fetch('/api/templates/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                idea_type_suggestion: ideaTypeSuggestion
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Success - populate form with generated template
            populateForm(data.template);

            // Update status
            statusDiv.className = 'alert alert-success mt-3';
            statusText.textContent = data.message || 'Template generated successfully! Review and edit as needed.';

            // Collapse the generation section to focus on editing
            setTimeout(() => {
                const generateSection = document.getElementById('generateSection');
                generateSection.style.opacity = '0.5';
                generateSection.style.transform = 'scale(0.98)';
            }, 1000);

            showAlert('Template generated successfully! Review and customize before saving.', 'success');
        } else {
            // Error
            statusDiv.className = 'alert alert-danger mt-3';
            statusText.textContent = data.message || 'Failed to generate template';
            showAlert('Error generating template: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error generating template:', error);
        statusDiv.className = 'alert alert-danger mt-3';
        statusText.textContent = 'Network error occurred while generating template';
        showAlert('Error generating template: ' + error.message, 'danger');
    } finally {
        // Reset button
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic me-1"></i>Generate Template';
    }
}

/**
 * Clear generation and reset form
 */
function clearGeneration() {
    // Clear the generation input
    document.getElementById('ideaTypeSuggestion').value = '';

    // Hide status
    document.getElementById('generateStatus').style.display = 'none';

    // Reset generation section appearance
    const generateSection = document.getElementById('generateSection');
    generateSection.style.opacity = '1';
    generateSection.style.transform = 'scale(1)';

    // Clear the form
    clearForm();

    showAlert('Form cleared. Ready for new template generation.', 'info');
}

/**
 * Check if there's a pending template to edit from the main interface
 */
function checkForPendingTemplate() {
    const pendingTemplate = sessionStorage.getItem('pendingTemplate');
    const urlParams = new URLSearchParams(window.location.search);

    if (pendingTemplate && urlParams.get('pending') === 'true') {
        try {
            const template = JSON.parse(pendingTemplate);

            // Clear the session storage
            sessionStorage.removeItem('pendingTemplate');

            // Show create modal with the pending template
            setTimeout(() => {
                showCreateModalWithTemplate(template);
            }, 500);

        } catch (error) {
            console.error('Error parsing pending template:', error);
        }
    }
}

/**
 * Show create modal pre-populated with a template
 */
function showCreateModalWithTemplate(template) {
    editMode = false;
    currentTemplateId = null;
    document.getElementById('editModalTitle').innerHTML = '<i class="fas fa-edit me-2"></i>Review & Edit Generated Template';

    // Show the generate section but collapsed
    const generateSection = document.getElementById('generateSection');
    if (generateSection) {
        generateSection.style.display = 'none';
    }

    // Populate with the generated template
    populateForm(template);

    // Add a notice about the generated template
    const modal = document.getElementById('editModal');
    const existingNotice = modal.querySelector('.generated-template-notice');
    if (!existingNotice) {
        const notice = document.createElement('div');
        notice.className = 'alert alert-info generated-template-notice';
        notice.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            <strong>Generated Template:</strong> This template was created by AI. Review and customize as needed before saving.
        `;

        const modalBody = modal.querySelector('.modal-body');
        modalBody.insertBefore(notice, modalBody.firstChild);
    }

    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();

    showAlert('Generated template loaded for review. Customize as needed and save when ready.', 'success');
}
