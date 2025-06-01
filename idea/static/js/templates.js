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
    const grid = document.getElementById('templatesGrid');
    grid.innerHTML = '';

    const filteredTemplates = filterAndSearchTemplates(templates);

    if (Object.keys(filteredTemplates).length === 0) {
        showEmptyState();
        return;
    }

    document.getElementById('emptyState').style.display = 'none';

    const coreTemplates = ['drabble', 'airesearch', 'game_design'];
    let cardIndex = 0;

    for (const [templateId, templateInfo] of Object.entries(filteredTemplates)) {
        if (templateInfo.error) {
            continue; // Skip errored templates
        }

        const isCore = coreTemplates.includes(templateId);
        const cardClass = isCore ? 'core-template' : 'custom-template';
        const badgeClass = isCore ? 'bg-success' : 'bg-secondary';

        const card = document.createElement('div');
        card.className = 'col-lg-4 col-md-6 mb-4';
        card.style.animationDelay = `${cardIndex * 0.1}s`;

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
}

/**
 * Filter and search templates
 */
function filterAndSearchTemplates(templates) {
    let filtered = {};
    const coreTemplates = ['drabble', 'airesearch', 'game_design'];

    for (const [templateId, templateInfo] of Object.entries(templates)) {
        if (templateInfo.error) continue;

        const isCore = coreTemplates.includes(templateId);

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
 * Update statistics
 */
function updateStatistics() {
    const coreTemplates = ['drabble', 'airesearch', 'game_design'];
    let totalCount = 0;
    let coreCount = 0;
    let customCount = 0;

    for (const [templateId, templateInfo] of Object.entries(allTemplates)) {
        if (templateInfo.error) continue;

        totalCount++;
        if (coreTemplates.includes(templateId)) {
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
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#newIdeaDetail">
                        <i class="fas fa-plus-circle me-2"></i>New Idea Prompt
                    </button>
                </h2>
                <div id="newIdeaDetail" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <pre style="white-space: pre-wrap; font-size: 0.9rem;">${template.prompts.new_idea}</pre>
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
    document.getElementById('ideaPrompt').value = template.prompts?.idea || '';
    document.getElementById('newIdeaPrompt').value = template.prompts?.new_idea || '';
    document.getElementById('formatPrompt').value = template.prompts?.format || '';
    document.getElementById('critiquePrompt').value = template.prompts?.critique || '';
    document.getElementById('refinePrompt').value = template.prompts?.refine || '';
    document.getElementById('breedPrompt').value = template.prompts?.breed || '';

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
            idea_prompt: document.getElementById('ideaPrompt').value,
            new_idea_prompt: document.getElementById('newIdeaPrompt').value,
            format_prompt: document.getElementById('formatPrompt').value,
            critique_prompt: document.getElementById('critiquePrompt').value,
            refine_prompt: document.getElementById('refinePrompt').value,
            breed_prompt: document.getElementById('breedPrompt').value,
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