// static/js/viewer.js

// static/js/viewer.js

// CSS styles for lineage modal and buttons are now in viewer.css


let isEvolutionRunning = false;
let currentContextIndex = 0;
let contexts = [];
let specificPrompts = [];
let breedingPrompts = [];  // Store breeding prompts for each generation
let tournamentHistory = [];  // Store Swiss tournament details
let currentEvolutionId = null;
let currentEvolutionName = null;
let currentEvolutionData = null;
let generations = [];
let currentModalIdea = null;
let lastFullHistoryFetch = 0;

// Evolution timing tracking
let evolutionStartTime = null;
let lastActivityTime = null;
let activityLog = [];
const MAX_ACTIVITY_LOG_ITEMS = 5;
let elapsedTimeInterval = null;
let tournamentCountTouched = false;
let lastRenderedHistoryVersion = -1;
let lastStoredHistoryVersion = -1;
let persistStateTimeout = null;
let openTournamentPanelGeneration = null;
const LOCAL_STORAGE_PERSIST_DELAY_MS = 1200;
let lastDiversityRenderSignature = '';
let diversityPlotSized = false;

// Track previous status message to detect changes for activity log
let previousStatusMessage = '';
let previousProgress = 0;
let lastProgressBarValue = -1;
let lastContextSignature = '';
let lastTournamentHistorySignature = '';
let lastTokenCountsSignature = '';
let lastOracleUpdateKey = '';
let lastEliteUpdateKey = '';
let lastDiagnosticsSummaryKey = '';
let latestDiagnosticsTokenCounts = null;
let evolutionViewMode = 'tree';
let lastEvolutionTreeSignature = '';
let treeResizeListenerAttached = false;
let currentEvolutionTree = null;
let selectedTreeNodeId = null;

/**
 * Load available templates and populate the idea type dropdown
 */
let allTemplates = []; // Store templates globally
const LAST_TEMPLATE_STORAGE_KEY = 'lastSelectedTemplateId';

function safeSignature(value) {
    try {
        return JSON.stringify(value ?? null);
    } catch (error) {
        return String(value ?? '');
    }
}

function prettifyDiagnosticKey(key) {
    return String(key || '').replace(/_/g, ' ');
}

function getDiagnosticDescription(key) {
    const descriptions = {
        generation_errors: 'LLM generation request failed and was retried or bypassed.',
        blocked_or_empty_responses: 'Model response was blocked/empty and fallback handling was used.',
        provider_client_unavailable: 'Provider client was unavailable for at least one request.',
        context_missing_concepts_marker: 'Expected CONCEPTS marker was missing; fallback parsing was used.',
        context_empty_after_parse: 'Context response could not be parsed into concepts.',
        formatter_schema_parse_failures: 'Structured formatter JSON parse failed and fallback parsing was used.',
        formatter_json_clean_recovery_failures: 'JSON clean-up fallback also failed during formatting.',
        formatter_fallback_used: 'Formatter fallback path was used instead of strict schema output.',
        formatter_json_clean_recovery: 'Formatter recovered using JSON clean-up fallback.',
        formatter_title_content_recovery: 'Formatter recovered via Title/Content extraction fallback.',
        compare_ideas_errors: 'Pairwise comparison call failed for at least one matchup.',
        oracle_parse_fallbacks: 'Oracle returned unstructured output and fallback parsing was used.',
        oracle_parse_errors: 'Oracle parsing raised an error and fallback handling was used.',
        seed_generation_failures: 'Seed generation attempt failed and was retried.',
        seed_context_novelty_regenerations: 'Seed context was regenerated to improve novelty.',
        seed_context_novelty_guardrail_exhausted: 'Novelty retries were exhausted for one or more seeds.',
        survivor_refinement_failures: 'Stochastic survivor refinement failed and the original survivor was kept.',
        invalid_survivor_refinements: 'Stochastic survivor refinement returned unusable content and was discarded.',
    };
    return descriptions[key] || 'No description available for this diagnostic key.';
}

function updateSurvivorRefinementControls() {
    const refineAllEl = document.getElementById('refineAllIdeas');
    const rateEl = document.getElementById('survivorRefinementRate');
    const rateValueEl = document.getElementById('survivorRefinementRateValue');

    if (rateEl && rateValueEl) {
        rateValueEl.textContent = parseFloat(rateEl.value || '0').toFixed(2);
    }

    if (!refineAllEl || !rateEl) {
        return;
    }

    rateEl.disabled = refineAllEl.checked;
}

function formatDiagnosticTimestamp(value) {
    if (!value) return '';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleString();
}

function getDiagnosticsSummary(tokenCounts) {
    const totals = tokenCounts?.diagnostics?.totals;
    if (!totals || typeof totals !== 'object') {
        return null;
    }

    const entries = Object.entries(totals)
        .filter(([, value]) => Number.isFinite(value) && value > 0)
        .sort((a, b) => b[1] - a[1]);

    const totalCount = entries.reduce((sum, [, value]) => sum + value, 0);
    const topEntries = entries
        .slice(0, 3)
        .map(([key, value]) => `${prettifyDiagnosticKey(key)}: ${value}`);

    return {
        totalCount,
        topEntries,
        signature: safeSignature(entries),
    };
}

function getStoredTemplateId() {
    try {
        return localStorage.getItem(LAST_TEMPLATE_STORAGE_KEY) || '';
    } catch (error) {
        return '';
    }
}

function storeTemplateId(templateId) {
    try {
        localStorage.setItem(LAST_TEMPLATE_STORAGE_KEY, templateId);
    } catch (error) {
        // Ignore localStorage failures (private mode / quota issues).
    }
}

function resetTemplateDisplay(message = 'No template selected') {
    const display = document.getElementById('ideaTypeDisplay');
    if (!display) return;
    display.textContent = message;
    display.classList.add('text-muted', 'fst-italic');
}

/**
 * Load available templates and populate the sidebar list
 */
async function loadTemplateTypes(options = {}) {
    const {
        preserveSelection = true,
        suppressDefaultSelection = false,
    } = options;

    try {
        const response = await fetch('/api/template-types');
        const data = await response.json();

        if (data.status === 'success') {
            allTemplates = data.templates;
            renderTemplateList(allTemplates);

            const currentType = document.getElementById('ideaType').value;
            const hasCurrentType = allTemplates.some(t => t.id === currentType);
            const storedType = getStoredTemplateId();
            const hasStoredType = allTemplates.some(t => t.id === storedType);

            if (preserveSelection && hasCurrentType) {
                selectTemplate(currentType, false);
            } else if (hasStoredType) {
                selectTemplate(storedType, false);
            } else if (!suppressDefaultSelection && allTemplates.length > 0) {
                selectTemplate(allTemplates[0].id, false);
            } else if (!hasCurrentType) {
                document.getElementById('ideaType').value = '';
                resetTemplateDisplay();
            }
        } else {
            console.error('Error loading template types:', data.message);
            renderTemplateList([]);
            resetTemplateDisplay('Unable to load templates');
        }
    } catch (error) {
        console.error('Error loading template types:', error);
        renderTemplateList([]);
        resetTemplateDisplay('Unable to load templates');
    }
}

function renderTemplateList(templates) {
    const container = document.getElementById('templatesList');
    if (!container) return;

    container.innerHTML = '';

    if (templates.length === 0) {
        container.innerHTML = '<div class="text-muted text-center p-3">No templates found</div>';
        return;
    }

    const currentType = document.getElementById('ideaType').value;

    templates.forEach(template => {
        const div = document.createElement('div');
        div.className = `template-list-item ${template.id === currentType ? 'active' : ''}`;
        div.onclick = (e) => {
            // Prevent triggering if clicking action buttons
            if (e.target.closest('.btn')) return;
            selectTemplate(template.id);
        };

        div.innerHTML = `
            <h6>${escapeHtml(template.name)} ${template.is_system ? '<span class="badge bg-light text-dark border ms-1">System</span>' : '<span class="badge bg-primary ms-1">Custom</span>'}</h6>
            <p>${escapeHtml(template.description || 'No description')}</p>
            <div class="template-actions">
                <button class="btn btn-xs btn-outline-secondary" onclick="editTemplateMainPage('${template.id}')">
                    <i class="fas fa-edit"></i> ${template.is_system ? 'Customize' : 'Edit'}
                </button>
            </div>
        `;

        container.appendChild(div);
    });
}

function selectTemplate(id, slideBack = true) {
    const template = allTemplates.find(t => t.id === id);
    if (!template) return false;

    // Update hidden input
    document.getElementById('ideaType').value = id;
    storeTemplateId(id);

    // Update selector text
    const display = document.getElementById('ideaTypeDisplay');
    if (display) {
        display.textContent = template.name;
        display.classList.remove('text-muted', 'fst-italic');
    }

    // Re-render list to update active state
    renderTemplateList(allTemplates);
    if (typeof window.onTemplateSelectionChanged === 'function') {
        window.onTemplateSelectionChanged();
    }

    if (slideBack) {
        showSidebarView('main');
    }

    return true;
}



function filterTemplates(query) {
    if (!query) {
        renderTemplateList(allTemplates);
        return;
    }

    const lower = query.toLowerCase();
    const filtered = allTemplates.filter(t =>
        t.name.toLowerCase().includes(lower) ||
        (t.description && t.description.toLowerCase().includes(lower))
    );
    renderTemplateList(filtered);
}

// Expose to window
window.selectTemplate = selectTemplate;
window.editTemplate = editTemplateMainPage;
window.filterTemplates = filterTemplates;
window.refreshTemplateCatalog = async function (options = {}) {
    await loadTemplateTypes(options);
};

window.addEventListener('idea-auth-ready', () => {
    loadTemplateTypes({ preserveSelection: true, suppressDefaultSelection: false });
});

if (window.__refreshTemplatesOnViewerReady) {
    window.__refreshTemplatesOnViewerReady = false;
    loadTemplateTypes({ preserveSelection: true, suppressDefaultSelection: false });
}

// Initialize
// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    // Load initial data
    await loadTemplateTypes();
    await loadHistory();

    // Check if we have a current evolution running or saved
    const restored = await restoreCurrentEvolution();

    // If not restored from running evolution, check for resumable checkpoints
    if (!restored) {
        await checkForResumableCheckpoints();
    }

    // Start polling for progress
    // pollProgress(); // Managed by restoreCurrentEvolution / startEvolution
    // setInterval(pollProgress, 1000);

    // Helper for safe event listeners
    const addListener = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) {
            // Remove existing listener to prevent duplicates
            el.removeEventListener(event, handler);
            el.addEventListener(event, handler);
        }
    };

    const updateTournamentCountFromPopSize = () => {
        const popSizeEl = document.getElementById('popSize');
        const countEl = document.getElementById('tournamentCount');
        const countValueEl = document.getElementById('tournamentCountValue');
        if (!popSizeEl || !countEl) return;

        const popSize = parseInt(popSizeEl.value || '0');
        if (!popSize || Number.isNaN(popSize)) return;

        if (!tournamentCountTouched) {
            countEl.value = '1';
        }

        if (countValueEl) {
            countValueEl.textContent = parseFloat(countEl.value).toFixed(2);
        }
    };

    addListener('tournamentCount', 'input', () => {
        tournamentCountTouched = true;
        const countEl = document.getElementById('tournamentCount');
        const countValueEl = document.getElementById('tournamentCountValue');
        if (countEl && countValueEl) {
            countValueEl.textContent = parseFloat(countEl.value).toFixed(2);
        }
    });
    addListener('popSize', 'input', updateTournamentCountFromPopSize);
    updateTournamentCountFromPopSize();
    addListener('refineAllIdeas', 'change', updateSurvivorRefinementControls);
    addListener('survivorRefinementRate', 'input', updateSurvivorRefinementControls);
    updateSurvivorRefinementControls();
    addListener('islandCount', 'input', () => {
        const el = document.getElementById('islandCount');
        const valueEl = document.getElementById('islandCountValue');
        if (el && valueEl) valueEl.textContent = parseInt(el.value || '1', 10);
    });
    addListener('migrationRate', 'input', () => {
        const el = document.getElementById('migrationRate');
        const valueEl = document.getElementById('migrationRateValue');
        if (el && valueEl) valueEl.textContent = parseFloat(el.value || '0').toFixed(2);
    });
    addListener('migrationNoveltyProtectionQuantile', 'input', () => {
        const el = document.getElementById('migrationNoveltyProtectionQuantile');
        const valueEl = document.getElementById('migrationNoveltyProtectionQuantileValue');
        if (el && valueEl) valueEl.textContent = parseFloat(el.value || '0.8').toFixed(2);
    });
    document.querySelectorAll('.evolution-view-toggle-btn').forEach((button) => {
        button.addEventListener('click', () => setEvolutionViewMode(button.dataset.viewMode));
    });
    setupModelChangeListener();

    // --- Evolution Controls ---

    addListener('startButton', 'click', async function () {
        console.log("Starting evolution...");
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');

        const popSize = parseInt(document.getElementById('popSize').value);
        const generations = parseInt(document.getElementById('generations').value);
        const ideaType = document.getElementById('ideaType').value;
        const modelType = document.getElementById('modelType').value;

        if (!ideaType) {
            alert('Please select a template before starting evolution.');
            if (typeof window.openTemplateLibrary === 'function') {
                window.openTemplateLibrary({ scroll: true, focusSearch: true });
            } else {
                showSidebarView('templates');
            }
            return;
        }

        const creativeTemp = parseFloat(document.getElementById('creativeTemp').value);
        const topP = parseFloat(document.getElementById('topP').value);

        // Tournament count (1.0 = full Swiss tournament)
        const tournamentCount = parseFloat(document.getElementById('tournamentCount').value);

        // Get mutation rate
        const mutationRate = parseFloat(document.getElementById('mutationRate').value);
        const seedContextPoolSizeInput = document.getElementById('seedContextPoolSize');
        const seedContextPoolSize = seedContextPoolSizeInput ? parseInt(seedContextPoolSizeInput.value) : null;
        const replacementRate = parseFloat(document.getElementById('replacementRate').value);
        const fitnessAlpha = parseFloat(document.getElementById('fitnessAlpha').value);
        const ageDecayRate = parseFloat(document.getElementById('ageDecayRate').value);
        const refineAllIdeas = document.getElementById('refineAllIdeas')?.checked ?? false;
        const survivorRefinementRate = parseFloat(
            document.getElementById('survivorRefinementRate')?.value ?? '0.1'
        );
        const islandCount = parseInt(document.getElementById('islandCount')?.value ?? '2', 10);
        const migrationRate = parseFloat(document.getElementById('migrationRate')?.value ?? '0.05');
        const migrationNoveltyProtectionQuantile = parseFloat(
            document.getElementById('migrationNoveltyProtectionQuantile')?.value ?? '0.8'
        );

        // Get model-aware thinking settings.
        const thinkingSettings = getThinkingSettings();

        // Get max budget
        const maxBudgetInput = document.getElementById('maxBudget');
        const maxBudget = maxBudgetInput && maxBudgetInput.value ? parseFloat(maxBudgetInput.value) : null;

        // Get evolution name (optional)
        const evolutionNameInput = document.getElementById('evolutionNameInput');
        const evolutionName = evolutionNameInput?.value?.trim() || null;

        const requestBody = {
            popSize,
            generations,
            ideaType,
            modelType,
            creativeTemp,
            topP,
            tournamentCount,
            mutationRate,
            seedContextPoolSize: Number.isInteger(seedContextPoolSize) ? seedContextPoolSize : null,
            replacementRate,
            fitnessAlpha,
            ageDecayRate,
            refineAllIdeas,
            survivorRefinementRate,
            islandCount,
            migrationRate,
            migrationNoveltyProtectionQuantile,
            thinkingBudget: thinkingSettings.thinkingBudget,
            thinkingLevel: thinkingSettings.thinkingLevel,
            maxBudget,
            evolutionName
        };

        console.log("Request body JSON:", JSON.stringify(requestBody));

        // Hide name input section when starting
        const nameSection = document.getElementById('evolutionNameSection');
        if (nameSection) nameSection.style.display = 'none';

        // Reset UI state
        resetUIState();

        // Hook for state management
        if (window.onEvolutionStart) {
            window.onEvolutionStart();
        }

        // Update button states
        if (startButton) {
            startButton.disabled = true;
            startButton.textContent = 'Running...';
        }
        if (stopButton) {
            stopButton.disabled = false;
            stopButton.style.display = 'block';
        }

        try {
            const response = await fetch('/api/start-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();

                // Store evolution identity
                currentEvolutionId = data.evolution_id;
                currentEvolutionName = data.evolution_name;

                // Reset contexts and index
                contexts = data.contexts || [];
                specificPrompts = data.specific_prompts || [];
                breedingPrompts = data.breeding_prompts || [];
                currentContextIndex = 0;

                // Create progress bar container if it doesn't exist
                if (!document.getElementById('progress-container')) {
                    createProgressBar(currentEvolutionName);
                }

                // Log startup status
                if (contexts.length > 0) {
                    addActivityLogItem(`✅ ${contexts.length} seed contexts ready`, 'success');
                } else {
                    addActivityLogItem('✅ Evolution started. Seeding in progress...', 'success');
                }

                // Start polling for updates
                isEvolutionRunning = true;
                pollProgress();

                updateContextDisplay();
            } else {
                // Handle error responses
                const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
                console.error("Failed to run evolution:", errorData);

                if (response.status === 409) {
                    // Evolution already running - show helpful message
                    showAlreadyRunningMessage();
                } else if (response.status === 429 || errorData.scope === 'global') {
                    showSystemBusyMessage(errorData.message);
                } else {
                    // Other errors
                    alert(`Error: ${errorData.message || 'Failed to start evolution'}`);
                }
                resetButtonStates();
            }
        } catch (error) {
            console.error("Error running evolution:", error);
            resetButtonStates();
        }
    });

    addListener('stopButton', 'click', async function () {
        console.log("Stopping evolution...");
        const stopButton = document.getElementById('stopButton');

        if (stopButton) {
            stopButton.disabled = true;
            stopButton.textContent = 'Stopping...';
        }

        // Show force stop button after 5 seconds if stop is taking too long
        const forceStopTimeout = setTimeout(() => {
            showForceStopButton();
            addActivityLogItem('⏳ Stop is taking a while - Force Stop available', 'warning');
        }, 5000);

        try {
            const response = await fetch('/api/stop-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Stop request sent:", data.message);
                // Polling will handle final reset
            } else {
                console.error("Failed to stop evolution:", await response.text());
                clearTimeout(forceStopTimeout);
                if (stopButton) {
                    stopButton.disabled = false;
                    stopButton.textContent = 'Stop Evolution';
                }
            }
        } catch (error) {
            console.error("Error stopping evolution:", error);
            clearTimeout(forceStopTimeout);
            if (stopButton) {
                stopButton.disabled = false;
                stopButton.textContent = 'Stop Evolution';
            }
        }
    });

    addListener('downloadButton', 'click', function () {
        if (currentEvolutionData) {
            downloadResults(currentEvolutionData);
        } else {
            alert("No evolution data available to save");
        }
    });

    // --- Force Stop Button ---
    addListener('forceStopButton', 'click', async function () {
        console.log("Force stopping evolution...");
        const forceStopButton = document.getElementById('forceStopButton');
        const stopButton = document.getElementById('stopButton');

        if (forceStopButton) {
            forceStopButton.disabled = true;
            forceStopButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Forcing...';
        }

        try {
            const response = await fetch('/api/force-stop-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Force stop successful:", data);
                addActivityLogItem('⚡ Evolution force stopped - checkpoint saved', 'warning');

                // Store checkpoint ID for resume
                if (data.checkpoint_id) {
                    localStorage.setItem('lastCheckpointId', data.checkpoint_id);
                    showResumeButton(data.checkpoint_id);
                }

                // Reset UI
                resetButtonStates();
                isEvolutionRunning = false;
            } else {
                console.error("Failed to force stop:", await response.text());
            }
        } catch (error) {
            console.error("Error force stopping:", error);
        } finally {
            if (forceStopButton) {
                forceStopButton.disabled = false;
                forceStopButton.innerHTML = '<i class="fas fa-bolt me-1"></i>Force Stop';
                forceStopButton.style.display = 'none';
            }
            if (stopButton) {
                stopButton.style.display = 'none';
            }
        }
    });

    // Resume/Continue buttons now handled via history panel
    // See handleHistoryResume() and handleHistoryContinue() functions

    // --- Template Generation ---
    // Note: generateTemplateBtn listener is set up in viewer.html
    addListener('saveTemplateBtn', 'click', saveAndUseTemplate);
    addListener('reviewEditBtn', 'click', reviewAndEditTemplate);

    // --- State Management ---
    addListener('newEvolutionBtn', 'click', showNewEvolutionMode);
    addListener('evolutionSelect', 'change', handleEvolutionSelection);

    // --- Settings ---
    addListener('saveSettingsBtn', 'click', saveSettings);
    addListener('deleteApiKeyBtn', 'click', deleteApiKey);
    addListener('confirmDeleteKeyBtn', 'click', confirmDeleteApiKey);
    addListener('copyApiKeyBtn', 'click', copyApiKey);

    // Setup button in alert (if present)
    const setupBtn = document.getElementById('setupApiKeyBtn');
    if (setupBtn) {
        setupBtn.addEventListener('click', () => showSidebarView('settings'));
    }

    // Nav buttons
    const navEvolutionBtn = document.getElementById('navEvolutionBtn');
    if (navEvolutionBtn) {
        navEvolutionBtn.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default anchor behavior if it's an <a> tag
            const select = document.getElementById('evolutionSelect');
            if (select && select.value) {
                showEvolutionMode(select.value);
            } else {
                showNewEvolutionMode();
            }
        });
    }

    // --- Keyboard Shortcuts ---
    const ideaTypeInput = document.getElementById('ideaTypeInput');
    if (ideaTypeInput) {
        ideaTypeInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                handleTemplateGeneration();
            }
        });
    }

    // --- Copy Button ---
    addListener('copyIdeaButton', 'click', () => {
        if (!currentModalIdea) return;
        const text = `${currentModalIdea.title || ''}\n\n${currentModalIdea.content || ''}`;
        navigator.clipboard.writeText(text).catch(err => console.error('Copy failed', err));
    });

    // --- Context Navigation ---
    addListener('prevContext', 'click', () => {
        if (currentContextIndex > 0) {
            currentContextIndex--;
            updateContextDisplay();
        }
    });

    addListener('nextContext', 'click', () => {
        if (currentContextIndex < contexts.length - 1) {
            currentContextIndex++;
            updateContextDisplay();
        }
    });

    // --- Contact Form ---
    addListener('sendContactBtn', 'click', async () => {
        const name = document.getElementById('contactName').value.trim();
        const email = document.getElementById('contactEmail').value.trim();
        const message = document.getElementById('contactMessage').value.trim();
        const btn = document.getElementById('sendContactBtn');

        if (!name || !email || !message) {
            alert('Please fill in all fields');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Sending...';

        try {
            const response = await fetch('/api/contact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, message })
            });

            const data = await response.json();

            if (response.ok) {
                // Close modal
                const modalEl = document.getElementById('contactModal');
                const modal = bootstrap.Modal.getInstance(modalEl);
                modal.hide();

                // Clear form
                document.getElementById('contactForm').reset();

                // Show success (you might want a nicer toast notification here)
                alert('Message sent successfully!');
            } else {
                alert(`Error: ${data.message || 'Failed to send message'}`);
            }
        } catch (error) {
            console.error('Error sending contact message:', error);
            alert('An error occurred while sending the message');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Send Message';
        }
    });

    // --- Welcome Modal Logic ---
    const hasSeenWelcome = localStorage.getItem('hasSeenWelcomeModal');
    if (!hasSeenWelcome) {
        // Show modal after a short delay
        setTimeout(() => {
            const welcomeModal = new bootstrap.Modal(document.getElementById('welcomeModal'));
            welcomeModal.show();

            // Set flag when modal is closed
            document.getElementById('welcomeModal').addEventListener('hidden.bs.modal', function () {
                localStorage.setItem('hasSeenWelcomeModal', 'true');
            });
        }, 1000);
    }
});

// Function to restore current evolution from localStorage
async function restoreCurrentEvolution() {
    try {
        console.log("Attempting to restore current evolution from localStorage...");
        let serverProgressData = null;

        // First check if there's actually an evolution running on the server
        try {
            const progressResponse = await fetch('/api/progress');
            if (progressResponse.ok) {
                const progressData = await progressResponse.json();
                console.log("Evolution status from server:", progressData);
                serverProgressData = progressData;

                // Only restore if evolution is actually running
                if (!progressData.is_running) {
                    console.log("No evolution is currently running, skipping localStorage restoration");
                    return false;
                }
            }
        } catch (e) {
            console.log("Could not check evolution status, skipping restoration:", e);
            return false;
        }

        // Check localStorage for current evolution data
        const storedData = localStorage.getItem('currentEvolutionData');
        if (storedData) {
            try {
                const evolutionState = JSON.parse(storedData);
                console.log("Found evolution data in localStorage:", evolutionState);

                // Handle both old format (just history array) and new format (object with history and diversity_history)
                let generationsData, diversityData, tokenCounts, tournamentData;
                if (Array.isArray(evolutionState)) {
                    if (serverProgressData?.evolution_id) {
                        console.log("Legacy stored evolution has no id; using live active run instead");
                        localStorage.removeItem('currentEvolutionData');
                        startPollingActiveEvolution(serverProgressData);
                        return true;
                    }
                    // Old format - just the history array
                    generationsData = evolutionState;
                    diversityData = [];
                    tokenCounts = null;
                    tournamentData = [];
                } else if (evolutionState && evolutionState.history) {
                    const storedEvolutionId = evolutionState.evolution_id || evolutionState.evolutionId || null;
                    const serverEvolutionId = serverProgressData?.evolution_id || null;
                    if (serverEvolutionId && storedEvolutionId !== serverEvolutionId) {
                        console.log(
                            "Stored evolution does not match active run; skipping localStorage restore",
                            { storedEvolutionId, serverEvolutionId }
                        );
                        localStorage.removeItem('currentEvolutionData');
                        startPollingActiveEvolution(serverProgressData);
                        return true;
                    }

                    // New format - object with history and diversity_history
                    generationsData = evolutionState.history;
                    diversityData = evolutionState.diversity_history || [];
                    tokenCounts = evolutionState.token_counts || null;
                    tournamentData = evolutionState.tournament_history || [];
                    // Restore context data if available
                    if (evolutionState.contexts) {
                        contexts = evolutionState.contexts;
                        specificPrompts = evolutionState.specific_prompts || [];
                        breedingPrompts = evolutionState.breeding_prompts || [];
                        currentContextIndex = 0;
                        updateContextDisplay();
                        document.querySelector('.context-navigation').style.display = 'block';
                    }
                } else {
                    console.error("Invalid evolution data format");
                    return false;
                }

                if (generationsData && generationsData.length > 0) {
                    // Render the generations
                    renderGenerations(generationsData);
                    const restoredVersion = Number.isInteger(evolutionState?.history_version)
                        ? evolutionState.history_version
                        : 0;
                    lastRenderedHistoryVersion = restoredVersion;
                    lastStoredHistoryVersion = restoredVersion;

                    // Restore diversity plot if we have diversity data
                    if (diversityData && diversityData.length > 0) {
                        console.log("Restoring diversity plot with data:", diversityData);
                        updateDiversityChart(diversityData);
                        // Ensure proper sizing after restoration
                        setTimeout(ensureDiversityChartSizing, 200);
                    }

                    // Set up currentEvolutionData for download functionality
                    currentEvolutionData = {
                        evolution_id: evolutionState.evolution_id || serverProgressData?.evolution_id || null,
                        evolution_name: evolutionState.evolution_name || serverProgressData?.evolution_name || null,
                        history: generationsData,
                        history_version: restoredVersion,
                        diversity_history: diversityData,
                        contexts: contexts,
                        specific_prompts: specificPrompts,
                        breeding_prompts: breedingPrompts,
                        tournament_history: tournamentData,
                        token_counts: tokenCounts
                    };

                    // Render tournament details if available
                    tournamentHistory = tournamentData || [];
                    renderTournamentDetails(tournamentHistory);

                    // Display token counts if available
                    if (tokenCounts) {
                        console.log("Restoring token counts from localStorage:", tokenCounts);
                        displayTokenCounts(tokenCounts);
                    }

                    // Enable download button
                    const downloadButton = document.getElementById('downloadButton');
                    if (downloadButton) {
                        downloadButton.disabled = false;
                        setupDownloadButton(currentEvolutionData);
                    }

                    console.log("Successfully restored current evolution from localStorage");
                    startPollingActiveEvolution(serverProgressData);
                    return true;
                }
            } catch (e) {
                console.error("Error parsing localStorage evolution data:", e);
            }
        }

        console.log("No current evolution data found in localStorage");
        if (serverProgressData?.is_running) {
            startPollingActiveEvolution(serverProgressData);
            return true;
        }
        return false;
    } catch (error) {
        console.error('Error restoring current evolution:', error);
        return false;
    }
}

// Function to set up the download button properly
function setupDownloadButton(data) {
    console.log("Setting up download button with data:", data);
    const downloadButton = document.getElementById('downloadButton');

    if (!downloadButton) {
        console.error("Download button not found in the DOM");
        return;
    }

    console.log("Download button found:", downloadButton);

    // Remove all existing event listeners by cloning
    const newButton = downloadButton.cloneNode(true);
    downloadButton.parentNode.replaceChild(newButton, downloadButton);

    // Store a reference to the new button
    const updatedButton = document.getElementById('downloadButton');

    // Enable the button
    updatedButton.disabled = false;
    updatedButton.textContent = 'Save Results';

    // Add a single click handler
    updatedButton.onclick = function (event) {
        event.preventDefault();
        console.log("Download button clicked, calling downloadResults with data:", data);
        downloadResults(data);
    };

    console.log("Download button setup complete, button is now:", updatedButton);
}

function applyHistoryIfNew(data, force = false) {
    if (!data || !data.history || data.history.length === 0) return false;

    const hasVersion = Number.isInteger(data.history_version);
    const shouldRender = force || !hasVersion || data.history_version !== lastRenderedHistoryVersion;
    if (!shouldRender) return false;

    renderGenerations(data.history);
    if (hasVersion) {
        lastRenderedHistoryVersion = data.history_version;
    }
    return true;
}

function schedulePersistEvolutionState(stateData) {
    if (!stateData) return;
    if (persistStateTimeout) {
        clearTimeout(persistStateTimeout);
    }
    const stateWithIdentity = {
        ...stateData,
        evolution_id: stateData.evolution_id || currentEvolutionId || null,
        evolution_name: stateData.evolution_name || currentEvolutionName || null
    };
    persistStateTimeout = setTimeout(() => {
        localStorage.setItem('currentEvolutionData', JSON.stringify(stateWithIdentity));
    }, LOCAL_STORAGE_PERSIST_DELAY_MS);
}

// Improve the markdown rendering function with better newline handling
function renderMarkdown(text) {
    if (!text) return '';

    // Normalize line endings
    text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // Escape HTML to prevent XSS
    text = text.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Process code blocks first
    text = text.replace(/```([\s\S]*?)```/g, function (match, code) {
        return '<pre><code>' + code.trim() + '</code></pre>';
    });

    // Process inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Process headings - ensure they're at the start of a line
    text = text.replace(/^(#{1,6})\s+(.*?)$/gm, function (match, hashes, content) {
        const level = hashes.length;
        return `<h${level}>${content.trim()}</h${level}>`;
    });

    // Process bold and italic
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/\_\_(.*?)\_\_/g, '<strong>$1</strong>');
    text = text.replace(/\_(.*?)\_/g, '<em>$1</em>');

    // Process lists - mark each list item
    let lines = text.split('\n');
    let inList = false;
    let listType = null;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Unordered list items
        if (/^\s*[\*\-]\s+/.test(line)) {
            if (!inList || listType !== 'ul') {
                lines[i] = inList ? '</ol><ul><li>' + line.replace(/^\s*[\*\-]\s+/, '') + '</li>' : '<ul><li>' + line.replace(/^\s*[\*\-]\s+/, '') + '</li>';
                inList = true;
                listType = 'ul';
            } else {
                lines[i] = '<li>' + line.replace(/^\s*[\*\-]\s+/, '') + '</li>';
            }
        }
        // Ordered list items
        else if (/^\s*\d+\.\s+/.test(line)) {
            if (!inList || listType !== 'ol') {
                lines[i] = inList ? '</ul><ol><li>' + line.replace(/^\s*\d+\.\s+/, '') + '</li>' : '<ol><li>' + line.replace(/^\s*\d+\.\s+/, '') + '</li>';
                inList = true;
                listType = 'ol';
            } else {
                lines[i] = '<li>' + line.replace(/^\s*\d+\.\s+/, '') + '</li>';
            }
        }
        // Not a list item
        else if (inList && line.trim() === '') {
            lines[i] = listType === 'ul' ? '</ul>' : '</ol>';
            inList = false;
            listType = null;
        }
    }

    // Close any open list
    if (inList) {
        lines.push(listType === 'ul' ? '</ul>' : '</ol>');
    }

    text = lines.join('\n');

    // Process blockquotes
    text = text.replace(/^>\s+(.*?)$/gm, '<blockquote>$1</blockquote>');

    // Process links
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Process paragraphs - any line that's not already a block element
    lines = text.split('\n');
    let inParagraph = false;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Skip if line is empty or already a block element
        if (line === '' ||
            line.startsWith('<h') ||
            line.startsWith('<ul') ||
            line.startsWith('<ol') ||
            line.startsWith('</ul') ||
            line.startsWith('</ol') ||
            line.startsWith('<li') ||
            line.startsWith('<blockquote') ||
            line.startsWith('<pre')) {

            if (inParagraph) {
                // Close the paragraph before this line
                lines[i - 1] += '</p>';
                inParagraph = false;
            }
            continue;
        }

        // If not in paragraph, start one
        if (!inParagraph) {
            lines[i] = '<p>' + lines[i];
            inParagraph = true;
        } else if (i === lines.length - 1 || lines[i + 1].trim() === '') {
            // If this is the last line or next line is empty, close paragraph
            lines[i] += '</p>';
            inParagraph = false;
        }
    }

    // Close any open paragraph
    if (inParagraph) {
        lines[lines.length - 1] += '</p>';
    }

    return lines.join('\n');
}

// Update the card preview function to better handle text
function createCardPreview(text, maxLength = 150) {
    if (!text) return '';

    // Normalize line endings
    text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // Strip markdown syntax for preview
    let preview = text
        .replace(/#{1,6}\s+/g, '') // Remove headings
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
        .replace(/\*(.*?)\*/g, '$1') // Remove italic
        .replace(/`{3}[\s\S]*?`{3}/g, '[Code Block]') // Replace code blocks
        .replace(/`([^`]+)`/g, '$1') // Remove inline code
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1') // Replace links with just text
        .replace(/^>\+\s*/gm, '') // Remove blockquote markers
        .replace(/^\s*[\*\-]\s+/gm, '• ') // Convert list markers to bullets
        .replace(/^\s*\d+\.\s+/gm, '• '); // Convert numbered lists to bullets

    // Replace multiple newlines with a single space
    preview = preview.replace(/\n\s*\n/g, ' ').replace(/\n/g, ' ');

    // Truncate to maxLength
    if (preview.length > maxLength) {
        preview = preview.substring(0, maxLength) + '...';
    }

    return preview;
}

function getBreedingGenotypeEntries(idea) {
    if (!idea || !Array.isArray(idea.breeding_genotypes)) {
        return [];
    }
    return idea.breeding_genotypes.filter(entry => entry && entry.genotype);
}

function createBreedingGenotypeSummary(entries) {
    if (!entries.length) {
        return '';
    }

    const latest = entries[entries.length - 1];
    const latestPreview = createCardPreview(String(latest.genotype || ''), 120);
    const countLabel = entries.length === 1 ? '1 capture' : `${entries.length} captures`;
    const entryItems = entries.map((entry, index) => {
        const sourceGeneration = Number.isInteger(entry.source_generation)
            ? entry.source_generation + 1
            : '?';
        const childGeneration = Number.isInteger(entry.child_generation)
            ? entry.child_generation + 1
            : '?';

        return `
            <li>
                <div class="breeding-genotype-meta">
                    Selection ${index + 1} - Gen ${sourceGeneration} -> Gen ${childGeneration}
                </div>
                <div class="breeding-genotype-text">${escapeHtml(entry.genotype)}</div>
            </li>
        `;
    }).join('');

    return `
        <details class="breeding-genotype-summary">
            <summary>
                <span><i class="fas fa-dna"></i> Genotype</span>
                <span class="breeding-genotype-count">${countLabel}</span>
            </summary>
            <div class="breeding-genotype-latest">${escapeHtml(latestPreview)}</div>
            <ol class="breeding-genotype-list">${entryItems}</ol>
        </details>
    `;
}

// Helper to extract title and content from possibly nested idea structures
// Handles: idea.title/content, idea.idea.title/content, or JSON-encoded content
function getIdeaData(idea) {
    let title = idea.title;
    let content = idea.content;

    // Check if data is nested under an 'idea' property
    if (idea.idea && typeof idea.idea === 'object') {
        title = idea.idea.title || title;
        content = idea.idea.content || content;
    }

    // Handle case where content might be JSON-encoded
    if (typeof content === 'string' && content.startsWith('{') && content.includes('"title"')) {
        try {
            const parsed = JSON.parse(content);
            title = parsed.title || title;
            content = parsed.content || content;
        } catch (e) {
            // Not valid JSON, use as-is
        }
    }

    return {
        title: title || 'Untitled',
        content: content || '',
        // Preserve other properties
        ...idea
    };
}

function getIdeaIslandId(idea) {
    const rawIslandId = idea?.island_id ?? idea?.idea?.island_id;
    const islandId = Number(rawIslandId);
    return Number.isInteger(islandId) && islandId >= 0 ? islandId : null;
}

function getIdeaStableId(idea, generationIndex, ideaIndex) {
    const rawId = idea?.id ?? idea?.idea?.id;
    return rawId ? String(rawId) : `generated:${generationIndex}:${ideaIndex}`;
}

function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === 'function') {
        return window.CSS.escape(String(value));
    }
    return String(value).replace(/["\\]/g, '\\$&');
}

function getTreeParentIds(idea) {
    if (idea?.migration_event && idea?.migrated_from_id) {
        return [String(idea.migrated_from_id)];
    }
    if (idea?.survivor_refined && idea?.survivor_refined_from_id) {
        return [String(idea.survivor_refined_from_id)];
    }
    return Array.isArray(idea?.parent_ids)
        ? idea.parent_ids.filter(Boolean).map(parentId => String(parentId))
        : [];
}

function getTreeEdgeType(idea, parentId) {
    if (idea?.migration_event && String(idea?.migrated_from_id || '') === String(parentId)) {
        return 'migration';
    }
    if (idea?.survivor_refined && String(idea?.survivor_refined_from_id || '') === String(parentId)) {
        return 'survivor_refinement';
    }
    return 'breeding';
}

function getNearestAvailableTreeLane(preferredLane, occupiedLanes) {
    if (!Number.isInteger(preferredLane) || preferredLane < 0) {
        let lane = 0;
        while (occupiedLanes.has(lane)) {
            lane += 1;
        }
        return lane;
    }

    if (!occupiedLanes.has(preferredLane)) {
        return preferredLane;
    }

    for (let distance = 1; distance <= occupiedLanes.size + preferredLane + 2; distance += 1) {
        const left = preferredLane - distance;
        const right = preferredLane + distance;
        if (left >= 0 && !occupiedLanes.has(left)) {
            return left;
        }
        if (!occupiedLanes.has(right)) {
            return right;
        }
    }

    return occupiedLanes.size;
}

function getPreferredParentTreeLane(idea, previousLanesById) {
    const parentLanes = getTreeParentIds(idea)
        .map(parentId => previousLanesById.get(parentId))
        .filter(lane => Number.isInteger(lane));
    if (parentLanes.length === 0) {
        return null;
    }
    const laneSum = parentLanes.reduce((sum, lane) => sum + lane, 0);
    return Math.round(laneSum / parentLanes.length);
}

function buildEvolutionTreeModel(gens) {
    const nodesById = new Map();
    const occurrencesById = new Map();
    const orderedIds = [];

    (gens || []).forEach((generation, generationIndex) => {
        (generation || []).forEach((idea, ideaIndex) => {
            const nodeId = getIdeaStableId(idea, generationIndex, ideaIndex);
            if (!nodesById.has(nodeId)) {
                const islandId = getIdeaIslandId(idea);
                nodesById.set(nodeId, {
                    id: nodeId,
                    title: getIdeaData(idea).title || 'Untitled',
                    idea,
                    firstGeneration: generationIndex,
                    lastGeneration: generationIndex,
                    firstIndex: ideaIndex,
                    islandId,
                    originIslandId: Number.isInteger(Number(idea?.origin_island_id))
                        ? Number(idea.origin_island_id)
                        : islandId,
                    continuationGenerations: [],
                    laneIndex: null
                });
                orderedIds.push(nodeId);
            }

            const node = nodesById.get(nodeId);
            node.lastGeneration = Math.max(node.lastGeneration, generationIndex);
            node.idea = idea;
            if (!occurrencesById.has(nodeId)) {
                occurrencesById.set(nodeId, []);
            }
            occurrencesById.get(nodeId).push(generationIndex);
        });
    });

    const continuations = [];
    occurrencesById.forEach((rawGenerations, nodeId) => {
        const uniqueGenerations = [...new Set(rawGenerations)].sort((a, b) => a - b);
        const node = nodesById.get(nodeId);
        if (!node) return;
        node.continuationGenerations = uniqueGenerations.slice(1);
        for (let i = 1; i < uniqueGenerations.length; i += 1) {
            const previous = uniqueGenerations[i - 1];
            const current = uniqueGenerations[i];
            if (current === previous + 1) {
                continuations.push({
                    id: `${nodeId}:${previous}->${current}`,
                    nodeId,
                    fromGeneration: previous,
                    toGeneration: current
                });
            }
        }
    });

    const edges = [];
    const seenEdges = new Set();
    (gens || []).forEach((generation, generationIndex) => {
        (generation || []).forEach((idea, ideaIndex) => {
            const childId = getIdeaStableId(idea, generationIndex, ideaIndex);
            const childNode = nodesById.get(childId);
            if (!childNode || childNode.firstGeneration !== generationIndex) return;

            getTreeParentIds(idea).forEach(parentId => {
                if (!nodesById.has(parentId)) return;
                const key = `${parentId}->${childId}:${generationIndex}`;
                if (seenEdges.has(key)) return;
                seenEdges.add(key);
                edges.push({
                    id: key,
                    from: parentId,
                    to: childId,
                    generation: generationIndex,
                    type: getTreeEdgeType(idea, parentId)
                });
            });
        });
    });

    let previousLanesById = new Map();
    let laneCount = 0;
    const generationsModel = (gens || []).map((generation, generationIndex) => {
        const nodeIds = [];
        const continuationIds = [];
        const entries = [];
        (generation || []).forEach((idea, ideaIndex) => {
            const nodeId = getIdeaStableId(idea, generationIndex, ideaIndex);
            const node = nodesById.get(nodeId);
            if (!node) return;
            const kind = node.firstGeneration === generationIndex ? 'node' : 'continuation';
            if (node.firstGeneration === generationIndex) {
                nodeIds.push(nodeId);
            } else {
                continuationIds.push(nodeId);
            }
            entries.push({ nodeId, idea, kind });
        });

        const occupiedLanes = new Set();
        const currentLanesById = new Map();
        const slots = [];

        entries.filter(entry => entry.kind === 'continuation').forEach((entry) => {
            const node = nodesById.get(entry.nodeId);
            const preferredLane = previousLanesById.has(entry.nodeId)
                ? previousLanesById.get(entry.nodeId)
                : node?.laneIndex;
            const laneIndex = getNearestAvailableTreeLane(preferredLane, occupiedLanes);
            occupiedLanes.add(laneIndex);
            currentLanesById.set(entry.nodeId, laneIndex);
            if (node && !Number.isInteger(node.laneIndex)) {
                node.laneIndex = laneIndex;
            }
            slots.push({ nodeId: entry.nodeId, kind: 'continuation', laneIndex });
        });

        entries.filter(entry => entry.kind === 'node').forEach((entry) => {
            const node = nodesById.get(entry.nodeId);
            const preferredLane = getPreferredParentTreeLane(entry.idea, previousLanesById);
            const laneIndex = getNearestAvailableTreeLane(preferredLane, occupiedLanes);
            occupiedLanes.add(laneIndex);
            currentLanesById.set(entry.nodeId, laneIndex);
            if (node) {
                node.laneIndex = laneIndex;
            }
            slots.push({ nodeId: entry.nodeId, kind: 'node', laneIndex });
        });

        if (occupiedLanes.size > 0) {
            laneCount = Math.max(laneCount, Math.max(...occupiedLanes) + 1);
        }
        previousLanesById = currentLanesById;

        slots.sort((a, b) => (
            a.laneIndex - b.laneIndex
            || a.kind.localeCompare(b.kind)
            || a.nodeId.localeCompare(b.nodeId)
        ));
        return { generation: generationIndex, nodeIds, continuationIds, slots };
    });

    const parentsById = new Map(orderedIds.map(id => [id, []]));
    const childrenById = new Map(orderedIds.map(id => [id, []]));
    edges.forEach((edge) => {
        if (!parentsById.has(edge.to)) parentsById.set(edge.to, []);
        if (!childrenById.has(edge.from)) childrenById.set(edge.from, []);
        parentsById.get(edge.to).push(edge.from);
        childrenById.get(edge.from).push(edge.to);
    });

    const nodes = orderedIds.map(id => nodesById.get(id));
    nodes.forEach((node) => {
        node.parentIds = parentsById.get(node.id) || [];
        node.childIds = childrenById.get(node.id) || [];
        node.ancestorIds = getTransitiveTreeIds(node.id, parentsById);
        node.descendantIds = getTransitiveTreeIds(node.id, childrenById);
        node.metricGeneration = getTreeMetricGeneration(node);
        node.metricValues = {
            elo: getTreeMetricValue(node.idea || {}, 'elo'),
            diversity: getTreeMetricValue(node.idea || {}, 'diversity'),
            fitness: getTreeMetricValue(node.idea || {}, 'fitness')
        };
        node.fitnessValue = getTreeMetricValue(node.idea || {}, 'fitness');
    });
    const metricStats = getTreeMetricStats(nodes);
    nodes.forEach((node) => {
        node.metricContexts = {
            elo: buildTreeMetricContext(node, metricStats, 'elo'),
            diversity: buildTreeMetricContext(node, metricStats, 'diversity'),
            fitness: buildTreeMetricContext(node, metricStats, 'fitness')
        };
        node.fitnessContext = buildTreeFitnessContext(node, nodesById);
    });

    return {
        nodes,
        nodesById,
        edges,
        continuations,
        generations: generationsModel,
        laneCount,
        fitnessRange: getTreeFitnessRange(nodes),
        metricStats
    };
}

function formatTreeMetric(value, digits = 2) {
    if (value === null || value === undefined || value === '') return '-';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '-';
    return numeric.toFixed(digits);
}

function getTreeMetricValue(idea, key) {
    const selection = idea?.selection_metrics || {};
    const survival = idea?.survival_metrics || {};
    const aliases = {
        elo: [selection.elo, idea?.elo, idea?.ratings?.auto],
        diversity: [selection.diversity, idea?.diversity_score, idea?.diversity],
        fitness: [selection.fitness, survival.survival_score, idea?.fitness]
    };
    const candidates = aliases[key] || [];
    for (const candidate of candidates) {
        const numeric = Number(candidate);
        if (Number.isFinite(numeric)) return numeric;
    }
    return null;
}

function getTreeFitnessRange(nodes) {
    const values = (nodes || [])
        .map(node => getTreeMetricValue(node?.idea || {}, 'fitness'))
        .filter(value => Number.isFinite(value));
    if (values.length === 0) {
        return { min: null, max: null };
    }
    return {
        min: Math.min(...values),
        max: Math.max(...values)
    };
}

function getTreeFitnessStrength(value, fitnessRange) {
    if (!Number.isFinite(value) || !Number.isFinite(fitnessRange?.min) || !Number.isFinite(fitnessRange?.max)) {
        return null;
    }
    if (fitnessRange.max === fitnessRange.min) {
        return 26;
    }
    const normalized = Math.max(0, Math.min(1, (value - fitnessRange.min) / (fitnessRange.max - fitnessRange.min)));
    return 10 + normalized * 34;
}

function getEvolutionTreeMaxGenerationSlots(tree) {
    const slotCounts = (tree?.generations || []).map(row => (row?.slots || []).length);
    return slotCounts.length > 0 ? Math.max(...slotCounts) : 0;
}

function getEvolutionTreeDensity(tree) {
    const laneCount = Math.max(1, Number(tree?.laneCount) || 1);
    const maxGenerationSlots = getEvolutionTreeMaxGenerationSlots(tree);
    const nodeCount = (tree?.nodes || []).length;

    if (laneCount >= 26 || maxGenerationSlots >= 24 || (laneCount >= 18 && nodeCount >= 110)) {
        return 'micro';
    }
    if (laneCount >= 16 || maxGenerationSlots >= 15 || nodeCount >= 90) {
        return 'compact';
    }
    return 'normal';
}

function getTreeNodeAbbreviation(title) {
    const cleanWords = String(title || '')
        .trim()
        .split(/\s+/)
        .map(word => (word.match(/[A-Za-z0-9]/g) || []).join(''))
        .filter(Boolean);

    if (cleanWords.length >= 2) {
        return `${cleanWords[0][0]}${cleanWords[1][0]}`.toUpperCase();
    }
    if (cleanWords.length === 1) {
        return cleanWords[0].slice(0, 2).toUpperCase();
    }
    return 'ID';
}

function getTreeMetricGeneration(node) {
    const idea = node?.idea || {};
    const selectionGeneration = Number(idea?.selection_metrics?.evaluation_generation);
    if (Number.isInteger(selectionGeneration)) {
        return selectionGeneration;
    }
    const survivalGeneration = Number(idea?.survival_metrics?.survived_to_generation);
    if (Number.isInteger(survivalGeneration)) {
        return survivalGeneration;
    }
    return Number.isInteger(node?.lastGeneration) ? node.lastGeneration : (node?.firstGeneration || 0);
}

function getTransitiveTreeIds(startId, adjacency) {
    const seen = new Set();
    const ordered = [];
    const stack = [...(adjacency.get(startId) || [])].reverse();
    while (stack.length > 0) {
        const nodeId = stack.pop();
        if (seen.has(nodeId)) continue;
        seen.add(nodeId);
        ordered.push(nodeId);
        stack.push(...[...(adjacency.get(nodeId) || [])].reverse());
    }
    return ordered;
}

function getTreeMean(values) {
    return values.length > 0
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : null;
}

function getTreeRankDesc(value, values) {
    if (!Number.isFinite(value) || !values.length) return null;
    const sortedValues = [...values].sort((a, b) => b - a);
    const rankIndex = sortedValues.findIndex(candidate => candidate <= value);
    return rankIndex >= 0 ? rankIndex + 1 : sortedValues.length;
}

function getTreeMetricStats(nodes) {
    const stats = {};
    ['elo', 'diversity', 'fitness'].forEach((metric) => {
        const records = (nodes || [])
            .map(node => ({
                generation: getTreeMetricGeneration(node),
                value: getTreeMetricValue(node?.idea || {}, metric)
            }))
            .filter(record => Number.isFinite(record.value));
        const values = records.map(record => record.value);
        const byGeneration = new Map();
        [...new Set(records.map(record => record.generation))].forEach((generation) => {
            const generationValues = records
                .filter(record => record.generation === generation)
                .map(record => record.value);
            const historyValues = records
                .filter(record => record.generation <= generation)
                .map(record => record.value);
            byGeneration.set(generation, {
                generationValues,
                generationMean: getTreeMean(generationValues),
                generationPopulationSize: generationValues.length,
                historyValues,
                historyMean: getTreeMean(historyValues),
                historyPopulationSize: historyValues.length
            });
        });
        stats[metric] = {
            values,
            mean: getTreeMean(values),
            populationSize: values.length,
            byGeneration
        };
    });
    return stats;
}

function buildTreeMetricContext(node, metricStats, metric) {
    const value = getTreeMetricValue(node?.idea || {}, metric);
    const generation = getTreeMetricGeneration(node);
    const globalStats = metricStats?.[metric] || { values: [], populationSize: 0, mean: null, byGeneration: new Map() };
    const scopedStats = globalStats.byGeneration?.get(generation) || {
        generationValues: [],
        generationMean: null,
        generationPopulationSize: 0,
        historyValues: [],
        historyMean: null,
        historyPopulationSize: 0
    };
    return {
        value,
        generation,
        globalRank: getTreeRankDesc(value, globalStats.values || []),
        globalPopulationSize: globalStats.populationSize || 0,
        historyRank: getTreeRankDesc(value, scopedStats.historyValues || []),
        historyPopulationSize: scopedStats.historyPopulationSize || 0,
        generationRank: getTreeRankDesc(value, scopedStats.generationValues || []),
        generationPopulationSize: scopedStats.generationPopulationSize || 0,
        globalMean: globalStats.mean,
        historyMean: scopedStats.historyMean,
        generationMean: scopedStats.generationMean,
        deltaVsGlobalMean: Number.isFinite(value) && Number.isFinite(globalStats.mean)
            ? value - globalStats.mean
            : null,
        deltaVsHistoryMean: Number.isFinite(value) && Number.isFinite(scopedStats.historyMean)
            ? value - scopedStats.historyMean
            : null,
        deltaVsGenerationMean: Number.isFinite(value) && Number.isFinite(scopedStats.generationMean)
            ? value - scopedStats.generationMean
            : null
    };
}

function buildTreeFitnessContext(node, nodesById) {
    const value = getTreeMetricValue(node?.idea || {}, 'fitness');
    const context = node?.metricContexts?.fitness || buildTreeMetricContext(node, null, 'fitness');
    const parentValues = (node?.parentIds || [])
        .map(parentId => nodesById.get(parentId))
        .map(parent => parent ? getTreeMetricValue(parent.idea || {}, 'fitness') : null)
        .filter(parentValue => Number.isFinite(parentValue));
    const parentAverage = parentValues.length > 0
        ? parentValues.reduce((sum, parentValue) => sum + parentValue, 0) / parentValues.length
        : null;
    const descendantCandidates = (node?.descendantIds || [])
        .map(descendantId => {
            const descendant = nodesById.get(descendantId);
            return {
                id: descendantId,
                fitness: descendant ? getTreeMetricValue(descendant.idea || {}, 'fitness') : null
            };
        })
        .filter(candidate => Number.isFinite(candidate.fitness));
    const bestDescendant = descendantCandidates.length > 0
        ? descendantCandidates.reduce((best, candidate) => (
            candidate.fitness > best.fitness ? candidate : best
        ))
        : null;

    return {
        ...context,
        value,
        parentAverage,
        deltaVsParentAverage: Number.isFinite(value) && Number.isFinite(parentAverage)
            ? value - parentAverage
            : null,
        bestDescendantId: bestDescendant?.id || null,
        bestDescendantFitness: bestDescendant?.fitness ?? null,
        deltaVsBestDescendant: Number.isFinite(value) && bestDescendant
            ? bestDescendant.fitness - value
            : null
    };
}

function getTreeNodeFlags(idea) {
    const flags = [];
    if (idea?.oracle_generated) flags.push('Oracle');
    if (idea?.migration_event) flags.push('Migrated');
    if (idea?.survivor_refined) flags.push('Refined');
    if (idea?.elite_selected || idea?.elite_selected_source) flags.push('Creative');
    return flags;
}

function formatTreeSignedMetric(value, digits = 2) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '-';
    const sign = numeric > 0 ? '+' : '';
    return `${sign}${numeric.toFixed(digits)}`;
}

function getTreePercentile(fitnessContext, scope = 'global') {
    const rankKey = scope === 'history' ? 'historyRank' : 'globalRank';
    const sizeKey = scope === 'history' ? 'historyPopulationSize' : 'globalPopulationSize';
    const rank = Number(fitnessContext?.[rankKey]);
    const populationSize = Number(fitnessContext?.[sizeKey]);
    if (!Number.isFinite(rank) || !Number.isFinite(populationSize) || populationSize <= 0) {
        return null;
    }
    if (populationSize === 1) {
        return 100;
    }
    return Math.max(0, Math.min(100, ((populationSize - rank) / (populationSize - 1)) * 100));
}

function getTreeSelectionRelations(tree, nodeId) {
    if (!tree?.nodesById?.has(nodeId)) {
        return {
            ancestors: new Set(),
            descendants: new Set(),
            ancestorEdges: new Set(),
            descendantEdges: new Set()
        };
    }

    const incomingEdges = new Map();
    const outgoingEdges = new Map();
    (tree.edges || []).forEach((edge) => {
        if (!incomingEdges.has(edge.to)) incomingEdges.set(edge.to, []);
        if (!outgoingEdges.has(edge.from)) outgoingEdges.set(edge.from, []);
        incomingEdges.get(edge.to).push(edge);
        outgoingEdges.get(edge.from).push(edge);
    });

    const collect = (startId, edgeMap, direction) => {
        const nodeIds = new Set();
        const edgeIds = new Set();
        const stack = [startId];
        while (stack.length > 0) {
            const currentId = stack.pop();
            (edgeMap.get(currentId) || []).forEach((edge) => {
                const nextId = direction === 'ancestors' ? edge.from : edge.to;
                edgeIds.add(edge.id);
                if (!nodeIds.has(nextId)) {
                    nodeIds.add(nextId);
                    stack.push(nextId);
                }
            });
        }
        return { nodeIds, edgeIds };
    };

    const ancestors = collect(nodeId, incomingEdges, 'ancestors');
    const descendants = collect(nodeId, outgoingEdges, 'descendants');
    return {
        ancestors: ancestors.nodeIds,
        descendants: descendants.nodeIds,
        ancestorEdges: ancestors.edgeIds,
        descendantEdges: descendants.edgeIds
    };
}

function renderTreeMetricItem(label, value, options = {}) {
    return `
        <div class="tree-inspector-metric">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(formatTreeMetric(value, options.digits ?? 2))}</strong>
        </div>
    `;
}

function renderTreeContextItem(label, value) {
    return `
        <div class="tree-inspector-context-item">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `;
}

function renderTreeLineageRows(tree, nodeIds, role) {
    const rows = (nodeIds || []).slice(0, 5).map((nodeId) => {
        const node = tree.nodesById.get(nodeId);
        if (!node) return '';
        const context = node.fitnessContext || {};
        const fitness = getTreeMetricValue(node.idea || {}, 'fitness');
        return `
            <button type="button" class="tree-inspector-lineage-row" data-tree-jump-node="${escapeHtml(nodeId)}">
                <span>${escapeHtml(node.title || 'Untitled')}</span>
                <small>G${node.firstGeneration}${node.lastGeneration > node.firstGeneration ? `-${node.lastGeneration}` : ''}</small>
                <small>Fit ${escapeHtml(formatTreeMetric(fitness, 2))}</small>
                <small>${escapeHtml(formatTreeSignedMetric(context.deltaVsHistoryMean, 2))} hist</small>
            </button>
        `;
    }).join('');
    if (!rows) {
        return '';
    }
    const remaining = Math.max(0, (nodeIds || []).length - 5);
    return `
        <div class="tree-inspector-lineage-group">
            <div class="tree-inspector-section-title">${escapeHtml(role)}</div>
            <div class="tree-inspector-lineage-list">${rows}</div>
            ${remaining > 0 ? `<div class="tree-inspector-more">+${remaining} more</div>` : ''}
        </div>
    `;
}

function renderEvolutionTreeInspector(tree, nodeId) {
    const node = tree?.nodesById?.get(nodeId);
    if (!node) return '';

    const idea = node.idea || {};
    const ideaData = getIdeaData(idea);
    const islandId = Number.isInteger(node.islandId) ? node.islandId : getIdeaIslandId(idea);
    const flags = getTreeNodeFlags(idea);
    const context = node.fitnessContext || {};
    const globalPercentile = getTreePercentile(context, 'global');
    const historyPercentile = getTreePercentile(context, 'history');
    const eloContext = node.metricContexts?.elo || {};
    const diversityContext = node.metricContexts?.diversity || {};
    const generationSpan = node.lastGeneration > node.firstGeneration
        ? `Gen ${node.firstGeneration}-${node.lastGeneration}`
        : `Gen ${node.firstGeneration}`;
    const metricHtml = [
        renderTreeMetricItem('ELO', getTreeMetricValue(idea, 'elo'), { digits: 0 }),
        renderTreeMetricItem('Diversity', getTreeMetricValue(idea, 'diversity'), { digits: 2 }),
        renderTreeMetricItem('Fitness', getTreeMetricValue(idea, 'fitness'), { digits: 2 })
    ].join('');
    const hasFitnessContext = Number.isFinite(context.value);
    const contextHtml = hasFitnessContext
        ? [
            renderTreeContextItem(
                'Global rank',
                context.globalRank && context.globalPopulationSize
                    ? `#${context.globalRank} / ${context.globalPopulationSize}`
                    : '-'
            ),
            renderTreeContextItem(
                'History rank',
                context.historyRank && context.historyPopulationSize
                    ? `#${context.historyRank} / ${context.historyPopulationSize}`
                    : '-'
            ),
            renderTreeContextItem('Global pct', Number.isFinite(globalPercentile) ? `${globalPercentile.toFixed(0)}%` : '-'),
            renderTreeContextItem('History pct', Number.isFinite(historyPercentile) ? `${historyPercentile.toFixed(0)}%` : '-'),
            renderTreeContextItem('Δ global', formatTreeSignedMetric(context.deltaVsGlobalMean, 2)),
            renderTreeContextItem('Δ history', formatTreeSignedMetric(context.deltaVsHistoryMean, 2)),
            renderTreeContextItem('Δ parents', formatTreeSignedMetric(context.deltaVsParentAverage, 2)),
            renderTreeContextItem('Best descendant Δ', formatTreeSignedMetric(context.deltaVsBestDescendant, 2)),
            renderTreeContextItem(
                'ELO global',
                eloContext.globalRank && eloContext.globalPopulationSize
                    ? `#${eloContext.globalRank} / ${eloContext.globalPopulationSize}`
                    : '-'
            ),
            renderTreeContextItem(
                'Diversity global',
                diversityContext.globalRank && diversityContext.globalPopulationSize
                    ? `#${diversityContext.globalRank} / ${diversityContext.globalPopulationSize}`
                    : '-'
            )
        ].join('')
        : '<div class="tree-inspector-empty">Metrics unavailable for this run.</div>';
    const ancestors = [...(node.ancestorIds || [])];
    const descendants = [...(node.descendantIds || [])];
    const lineageHtml = [
        renderTreeLineageRows(tree, ancestors, 'Ancestors'),
        renderTreeLineageRows(tree, descendants, 'Descendants')
    ].join('');
    const metaPills = [
        Number.isInteger(islandId) ? `Island ${islandId + 1}` : null,
        generationSpan,
        ...flags
    ].filter(Boolean);

    return `
        <div class="tree-inspector-header">
            <div>
                <div class="tree-inspector-kicker">Selected idea</div>
                <h3>${escapeHtml(ideaData.title || node.title || 'Untitled')}</h3>
            </div>
            <button type="button" class="tree-inspector-close" data-tree-inspector-close aria-label="Close tree inspector">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="tree-inspector-meta">
            ${metaPills.map(pill => `<span>${escapeHtml(pill)}</span>`).join('')}
        </div>
        <div class="tree-inspector-metrics">${metricHtml}</div>
        <div class="tree-inspector-section-title">Fitness context</div>
        <div class="tree-inspector-context">${contextHtml}</div>
        ${lineageHtml ? `<div class="tree-inspector-lineage">${lineageHtml}</div>` : ''}
        <div class="tree-inspector-section-header">
            <div class="tree-inspector-section-title">Content</div>
            <button type="button" class="tree-inspector-popout" data-tree-inspector-popout>
                <i class="fas fa-up-right-from-square"></i>
                <span>Read</span>
            </button>
        </div>
        <div class="tree-inspector-content markdown-content">${renderMarkdown(ideaData.content || '')}</div>
    `;
}

function updateEvolutionTreeInspector(tree, nodeId) {
    const inspector = document.getElementById('evolution-tree-inspector');
    if (!inspector) return;
    if (!tree?.nodesById?.has(nodeId)) {
        inspector.hidden = true;
        inspector.innerHTML = '';
        return;
    }
    inspector.hidden = false;
    inspector.innerHTML = renderEvolutionTreeInspector(tree, nodeId);
    inspector.querySelector('[data-tree-inspector-close]')?.addEventListener('click', clearEvolutionTreeSelection);
    inspector.querySelector('[data-tree-inspector-popout]')?.addEventListener('click', () => {
        const node = tree.nodesById.get(nodeId);
        if (node) {
            showIdeaModal(getIdeaData(node.idea || {}));
        }
    });
    inspector.querySelectorAll('[data-tree-jump-node]').forEach((button) => {
        button.addEventListener('click', () => selectEvolutionTreeNode(button.dataset.treeJumpNode));
    });
}

function applyEvolutionTreeSelection({ redraw = true } = {}) {
    const tree = currentEvolutionTree;
    const container = document.getElementById('evolution-tree-container');
    const shell = container?.querySelector('.evolution-tree-shell');
    if (!container || !shell || !tree) return;

    const selectedNode = selectedTreeNodeId && tree.nodesById.has(selectedTreeNodeId)
        ? selectedTreeNodeId
        : null;
    const relations = selectedNode
        ? getTreeSelectionRelations(tree, selectedNode)
        : null;

    shell.classList.toggle('has-selection', Boolean(selectedNode));
    container.querySelectorAll('.evolution-tree-node, .evolution-tree-continuation').forEach((el) => {
        const nodeId = el.dataset.nodeId;
        el.classList.remove('tree-selected', 'tree-ancestor', 'tree-descendant', 'tree-muted');
        if (!selectedNode) return;
        if (nodeId === selectedNode) {
            el.classList.add('tree-selected');
        } else if (relations.ancestors.has(nodeId)) {
            el.classList.add('tree-ancestor');
        } else if (relations.descendants.has(nodeId)) {
            el.classList.add('tree-descendant');
        } else {
            el.classList.add('tree-muted');
        }
    });

    updateEvolutionTreeInspector(tree, selectedNode);
    if (redraw) {
        requestAnimationFrame(drawEvolutionTreeConnectors);
    }
}

function selectEvolutionTreeNode(nodeId) {
    if (!nodeId) return;
    selectedTreeNodeId = String(nodeId);
    applyEvolutionTreeSelection();
}

function clearEvolutionTreeSelection() {
    selectedTreeNodeId = null;
    applyEvolutionTreeSelection();
}

function renderEvolutionTree(gens) {
    const container = document.getElementById('evolution-tree-container');
    if (!container) return;

    const signature = safeSignature(gens || []);
    if (signature === lastEvolutionTreeSignature && container.children.length > 0) {
        applyEvolutionTreeSelection({ redraw: false });
        requestAnimationFrame(drawEvolutionTreeConnectors);
        return;
    }
    lastEvolutionTreeSignature = signature;
    container.innerHTML = '';

    if (!gens || gens.length === 0) {
        container.innerHTML = '<div class="evolution-tree-empty">No evolution history available.</div>';
        return;
    }

    const tree = buildEvolutionTreeModel(gens);
    currentEvolutionTree = tree;
    if (selectedTreeNodeId && !tree.nodesById.has(selectedTreeNodeId)) {
        selectedTreeNodeId = null;
    }
    if (tree.nodes.length === 0) {
        container.innerHTML = '<div class="evolution-tree-empty">No tree nodes available.</div>';
        return;
    }

    const shell = document.createElement('div');
    shell.className = 'evolution-tree-shell';
    shell.dataset.laneCount = String(tree.laneCount);
    shell.dataset.maxGenerationSlots = String(getEvolutionTreeMaxGenerationSlots(tree));
    shell.dataset.nodeCount = String(tree.nodes.length);
    shell.dataset.density = getEvolutionTreeDensity(tree);

    const scroll = document.createElement('div');
    scroll.className = 'evolution-tree-scroll';
    scroll.id = 'evolution-tree-scroll';

    const connectorSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    connectorSvg.classList.add('evolution-tree-connectors');
    connectorSvg.setAttribute('aria-hidden', 'true');
    scroll.appendChild(connectorSvg);

    tree.generations.forEach((generationRow) => {
        const row = document.createElement('div');
        row.className = 'evolution-tree-row';
        row.dataset.generation = String(generationRow.generation);

        const label = document.createElement('div');
        label.className = 'evolution-tree-generation-label';
        label.innerHTML = `<span>Gen ${generationRow.generation}</span>`;
        row.appendChild(label);

        const lanes = document.createElement('div');
        lanes.className = 'evolution-tree-lanes';
        lanes.style.gridTemplateColumns = `repeat(${Math.max(1, tree.laneCount)}, minmax(0, 1fr))`;

        const slotsByLane = new Map((generationRow.slots || []).map(slot => [slot.laneIndex, slot]));
        for (let laneIndex = 0; laneIndex < Math.max(1, tree.laneCount); laneIndex += 1) {
            const lane = document.createElement('div');
            lane.className = 'evolution-tree-lane';
            lane.dataset.laneIndex = String(laneIndex);

            const slot = slotsByLane.get(laneIndex);
            const node = slot ? tree.nodesById.get(slot.nodeId) : null;
            if (node && slot.kind === 'node') {
                lane.appendChild(createEvolutionTreeNode(node, tree.fitnessRange));
            } else if (node && slot.kind === 'continuation') {
                lane.appendChild(createEvolutionTreeContinuation(node, generationRow.generation));
            }
            lanes.appendChild(lane);
        }

        row.appendChild(lanes);
        scroll.appendChild(row);
    });

    shell.appendChild(scroll);
    const inspector = document.createElement('aside');
    inspector.className = 'evolution-tree-inspector';
    inspector.id = 'evolution-tree-inspector';
    inspector.hidden = true;
    shell.appendChild(inspector);
    container.appendChild(shell);

    if (!treeResizeListenerAttached) {
        window.addEventListener('resize', () => {
            if (evolutionViewMode === 'tree') {
                drawEvolutionTreeConnectors();
            }
        });
        treeResizeListenerAttached = true;
    }

    applyEvolutionTreeSelection({ redraw: false });
    requestAnimationFrame(drawEvolutionTreeConnectors);
}

function createEvolutionTreeNode(node, fitnessRange) {
    const idea = node.idea || {};
    const islandId = Number.isInteger(node.islandId) ? node.islandId : getIdeaIslandId(idea);
    const fitnessValue = getTreeMetricValue(idea, 'fitness');
    const fitnessStrength = getTreeFitnessStrength(fitnessValue, fitnessRange);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'evolution-tree-node';
    button.dataset.nodeId = node.id;
    button.dataset.hasFitness = fitnessStrength === null ? 'false' : 'true';
    if (fitnessStrength !== null) {
        button.style.setProperty('--tree-fitness-strength', `${fitnessStrength.toFixed(1)}%`);
    }
    if (Number.isInteger(islandId)) {
        button.dataset.islandId = String(islandId);
        button.classList.add(`island-${islandId % 5}`);
    }
    if (idea?.oracle_generated) {
        button.dataset.oracle = 'true';
        button.classList.add('tree-oracle');
    }
    const detailParts = [node.title || 'Untitled'];
    if (Number.isInteger(islandId)) detailParts.push(`Island ${islandId + 1}`);
    if (Number.isFinite(fitnessValue)) detailParts.push(`Fitness ${formatTreeMetric(fitnessValue, 2)}`);
    getTreeNodeFlags(idea).forEach(flag => detailParts.push(flag));
    button.title = detailParts.join(' | ');
    button.setAttribute('aria-label', detailParts.join(', '));

    button.innerHTML = `
        <div class="tree-node-title">${escapeHtml(node.title || 'Untitled')}</div>
        <div class="tree-node-abbrev" aria-hidden="true">${escapeHtml(getTreeNodeAbbreviation(node.title || 'Untitled'))}</div>
    `;
    button.addEventListener('click', (event) => {
        event.stopPropagation();
        selectEvolutionTreeNode(node.id);
    });
    return button;
}

function createEvolutionTreeContinuation(node, generationIndex) {
    const idea = node.idea || {};
    const islandId = Number.isInteger(node.islandId) ? node.islandId : getIdeaIslandId(idea);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'evolution-tree-continuation';
    button.dataset.nodeId = node.id;
    button.dataset.generation = String(generationIndex);
    if (Number.isInteger(islandId)) {
        button.dataset.islandId = String(islandId);
        button.classList.add(`island-${islandId % 5}`);
    }
    if (idea?.oracle_generated) {
        button.dataset.oracle = 'true';
        button.classList.add('tree-oracle');
    }
    button.setAttribute(
        'aria-label',
        `${node.title || 'Idea'} survived unchanged to generation ${generationIndex}`
    );
    button.innerHTML = '<span class="tree-continuation-line"></span>';
    button.addEventListener('click', (event) => {
        event.stopPropagation();
        selectEvolutionTreeNode(node.id);
    });
    return button;
}

function drawEvolutionTreeConnectors() {
    const container = document.getElementById('evolution-tree-container');
    const scroll = document.getElementById('evolution-tree-scroll');
    const svg = scroll?.querySelector('.evolution-tree-connectors');
    if (!container || !scroll || !svg || container.hidden) return;

    const tree = currentEvolutionTree || buildEvolutionTreeModel(generations || []);
    const selectedNode = selectedTreeNodeId && tree.nodesById.has(selectedTreeNodeId)
        ? selectedTreeNodeId
        : null;
    const relations = selectedNode ? getTreeSelectionRelations(tree, selectedNode) : null;
    const scrollRect = scroll.getBoundingClientRect();
    const width = scroll.clientWidth;
    const height = Math.max(scroll.scrollHeight, scroll.clientHeight);
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.innerHTML = '';

    const getAnchor = (selector, side = 'center') => {
        const el = scroll.querySelector(selector);
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        const x = rect.left - scrollRect.left + scroll.scrollLeft + (rect.width / 2);
        const y = rect.top - scrollRect.top + scroll.scrollTop + (
            side === 'bottom' ? rect.height : side === 'top' ? 0 : rect.height / 2
        );
        return { x, y };
    };

    const addPath = (d, className) => {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d);
        path.setAttribute('class', className);
        path.setAttribute('fill', 'none');
        svg.appendChild(path);
    };

    tree.edges.forEach((edge) => {
        const fromNode = tree.nodesById.get(edge.from);
        const toNode = tree.nodesById.get(edge.to);
        if (!fromNode || !toNode) return;
        const escapedFrom = cssEscape(edge.from);
        const escapedTo = cssEscape(edge.to);
        const parentSelector = fromNode.lastGeneration >= edge.generation - 1
            ? `[data-node-id="${escapedFrom}"][data-generation="${edge.generation - 1}"]`
            : `[data-node-id="${escapedFrom}"].evolution-tree-node`;
        const from = getAnchor(parentSelector, 'bottom')
            || getAnchor(`[data-node-id="${escapedFrom}"].evolution-tree-node`, 'bottom');
        const to = getAnchor(`[data-node-id="${escapedTo}"].evolution-tree-node`, 'top');
        if (!from || !to) return;
        const midY = from.y + Math.max(24, (to.y - from.y) * 0.45);
        let relationClass = '';
        if (relations) {
            if (relations.ancestorEdges.has(edge.id)) {
                relationClass = ' tree-connector-lineage-ancestor';
            } else if (relations.descendantEdges.has(edge.id)) {
                relationClass = ' tree-connector-lineage-descendant';
            } else {
                relationClass = ' tree-connector-muted';
            }
        }
        const className = `tree-connector tree-connector-${edge.type || 'breeding'}${relationClass}`;
        addPath(`M ${from.x} ${from.y} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y}`, className);
    });
}

function setEvolutionViewMode(mode) {
    evolutionViewMode = mode === 'tree' ? 'tree' : 'cards';
    const cardsContainer = document.getElementById('generations-container');
    const treeContainer = document.getElementById('evolution-tree-container');
    const toolbar = document.getElementById('evolution-view-toolbar');
    if (toolbar) {
        toolbar.style.display = generations && generations.length > 0 ? 'flex' : 'none';
    }
    if (cardsContainer) {
        cardsContainer.hidden = evolutionViewMode === 'tree';
    }
    if (treeContainer) {
        treeContainer.hidden = evolutionViewMode !== 'tree';
    }
    document.querySelectorAll('.evolution-view-toggle-btn').forEach((button) => {
        const isActive = button.dataset.viewMode === evolutionViewMode;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    if (evolutionViewMode === 'tree') {
        renderEvolutionTree(generations);
    }
}

function syncEvolutionViewMode() {
    const toolbar = document.getElementById('evolution-view-toolbar');
    if (toolbar) {
        toolbar.style.display = generations && generations.length > 0 ? 'flex' : 'none';
    }
    setEvolutionViewMode(evolutionViewMode);
}

function resetEvolutionTreeView() {
    generations = [];
    lastEvolutionTreeSignature = '';
    currentEvolutionTree = null;
    selectedTreeNodeId = null;
    const treeContainer = document.getElementById('evolution-tree-container');
    if (treeContainer) {
        treeContainer.innerHTML = '';
        treeContainer.hidden = true;
    }
    const toolbar = document.getElementById('evolution-view-toolbar');
    if (toolbar) {
        toolbar.style.display = 'none';
    }
    const cardsContainer = document.getElementById('generations-container');
    if (cardsContainer) {
        cardsContainer.hidden = false;
    }
    evolutionViewMode = 'tree';
    document.querySelectorAll('.evolution-view-toggle-btn').forEach((button) => {
        const isActive = button.dataset.viewMode === 'tree';
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
}

// Update the renderGenerations function to use the improved preview
function renderGenerations(gens) {
    generations = gens; // Store the generations globally
    console.log("Received generations:", generations); // Debug log

    if (!generations || generations.length === 0) {
        console.warn("Received empty generations data");
        return;
    }
    syncEvolutionViewMode();
    const container = document.getElementById('generations-container');

    generations.forEach((generation, index) => {
        // Check if this generation section already exists
        let genDiv = document.getElementById(`generation-${index}`);

        if (!genDiv) {
            // Create new generation section if it doesn't exist
            genDiv = document.createElement('div');
            genDiv.className = 'generation-section mb-4';
            genDiv.id = `generation-${index}`;

            // Create generation header with toggle functionality
            const headerDiv = document.createElement('div');
            headerDiv.className = 'generation-header';

            // Add generation title
            const header = document.createElement('h2');
            header.className = 'generation-title';

            // Label the initial population as "Generation 0 (Initial)"
            if (index === 0) {
                header.textContent = `Generation 0 (Initial Population)`;
            } else {
                header.textContent = `Generation ${index}`;
            }

            // Add collapse toggle button
            const toggleButton = document.createElement('button');
            toggleButton.className = 'generation-toggle';
            toggleButton.setAttribute('aria-expanded', 'true');
            toggleButton.setAttribute('aria-label', `Toggle Generation ${index}`);
            toggleButton.onclick = () => toggleGeneration(index);
            toggleButton.innerHTML = '<i class="fas fa-chevron-up"></i>';

            const tournamentChip = document.createElement('button');
            tournamentChip.type = 'button';
            tournamentChip.className = 'generation-tournament-chip';
            tournamentChip.id = `generation-tournament-chip-${index}`;
            tournamentChip.setAttribute('aria-label', `Toggle tournament details for Generation ${index}`);
            tournamentChip.setAttribute('aria-expanded', 'false');
            tournamentChip.setAttribute('aria-controls', `generation-tournament-popout-${index}`);
            tournamentChip.onclick = () => toggleGenerationTournament(index);
            tournamentChip.innerHTML = '<i class="fas fa-trophy"></i><span>Tournament</span>';
            tournamentChip.hidden = true;

            const headerControls = document.createElement('div');
            headerControls.className = 'generation-header-controls';
            headerControls.appendChild(tournamentChip);
            headerControls.appendChild(toggleButton);

            headerDiv.appendChild(header);
            headerDiv.appendChild(headerControls);
            genDiv.appendChild(headerDiv);

            const tournamentPopout = document.createElement('div');
            tournamentPopout.className = 'generation-tournament-popout';
            tournamentPopout.id = `generation-tournament-popout-${index}`;
            tournamentPopout.hidden = true;
            genDiv.appendChild(tournamentPopout);

            // Add ideas container with proper containment
            const contentDiv = document.createElement('div');
            contentDiv.className = 'generation-content';
            contentDiv.id = `generation-content-${index}`;

            const scrollContainer = document.createElement('div');
            scrollContainer.className = 'scroll-container';
            scrollContainer.id = `scroll-container-${index}`;

            // Add a wrapper to ensure proper containment
            const scrollWrapper = document.createElement('div');
            scrollWrapper.className = 'scroll-wrapper';
            scrollWrapper.id = `scroll-wrapper-${index}`;

            scrollWrapper.appendChild(scrollContainer);
            contentDiv.appendChild(scrollWrapper);
            genDiv.appendChild(contentDiv);

            container.appendChild(genDiv);
        }

        renderGenerationTournamentDetails(index);

        // Get the scroll container for this generation
        const scrollContainer = document.getElementById(`scroll-container-${index}`);

        // Process each idea in this generation
        generation.forEach((idea, ideaIndex) => {
            const genotypeEntries = getBreedingGenotypeEntries(idea);
            const islandId = getIdeaIslandId(idea);
            const isOracleGenerated = idea.oracle_generated === true;
            let insertBeforeNode = null;

            // Check if this idea card already exists
            const existingCard = document.getElementById(`idea-${index}-${ideaIndex}`);
            if (existingCard) {
                // An Oracle update might have replaced this idea.
                // We check if the new data for this slot is an Oracle idea.
                const existingGenotypeCount = Number(existingCard.dataset.genotypeCount || '0');
                const genotypeCountChanged = existingGenotypeCount !== genotypeEntries.length;
                const islandChanged = (existingCard.dataset.islandId || '') !== (
                    islandId === null ? '' : String(islandId)
                );
                const migrationChanged = (existingCard.dataset.migrationEvent === 'true') !== (
                    idea.migration_event === true
                );
                const oracleChanged = (existingCard.dataset.oracle === 'true') !== isOracleGenerated;

                // We also check if the existing card in the DOM reflects this.
                // A simple way is to check the button's inner HTML for the icon.
                const lineageButton = existingCard.querySelector('.view-lineage');
                const cardIsAlreadyOracle = lineageButton && lineageButton.innerHTML.includes('fa-eye');

                if ((isOracleGenerated && !cardIsAlreadyOracle) || genotypeCountChanged || islandChanged || migrationChanged || oracleChanged) {
                    // The data is now an Oracle idea, but the card isn't.
                    // This means a replacement happened, or metadata was added after the card rendered.
                    console.log(`Card refresh needed for idea ${ideaIndex} in generation ${index}. Re-rendering card.`);
                    insertBeforeNode = existingCard.nextSibling;
                    existingCard.remove(); // Remove the old card
                } else {
                    // The card exists and is up-to-date, so we skip re-rendering.
                    return;
                }
            }

            // Create a new card for this idea
            const card = document.createElement('div');
            card.className = 'card gen-card';
            card.id = `idea-${index}-${ideaIndex}`;
            card.dataset.genotypeCount = String(genotypeEntries.length);
            if (islandId !== null) {
                card.dataset.islandId = String(islandId);
                card.classList.add(`island-${islandId % 5}`);
            }
            if (idea.migration_event === true) {
                card.dataset.migrationEvent = 'true';
            }
            if (isOracleGenerated) {
                card.dataset.oracle = 'true';
                card.classList.add('oracle-card');
            }

            // Extract idea data (handles nested structures from Firestore)
            const ideaData = getIdeaData(idea);

            // Mark elite ideas with data attribute for styling
            if (ideaData.elite_selected) {
                card.setAttribute('data-elite', 'true');
            }

            // Create a plain text preview for the card
            const plainPreview = createCardPreview(ideaData.content, 150);
            const genotypeSummaryHtml = createBreedingGenotypeSummary(genotypeEntries);
            const islandBadgeHtml = islandId !== null
                ? `<span class="island-badge">Island ${islandId + 1}</span>`
                : '';
            const migrationBadgeHtml = idea.migration_event === true
                ? '<span class="migration-badge">Migrated</span>'
                : '';
            const oracleBadgeHtml = isOracleGenerated
                ? '<span class="oracle-badge">Oracle</span>'
                : '';

            // Add "Prompt" button for initial generation cards
            const viewPromptButton = index === 0 ?
                `<button class="btn btn-outline-info btn-sm view-prompt" title="View Initial Prompt">
                    <i class="fas fa-lightbulb"></i> Prompt
                </button>` : '';

            // Add "Prompt" button for breeding generation cards (if we have breeding prompts)
            const breedingPromptButton = index > 0 ?
                `<button class="btn btn-outline-info btn-sm view-breeding-prompt" title="View Breeding Prompt">
                    <i class="fas fa-lightbulb"></i> Prompt
                </button>` : '';

            // Add "Lineage", "Oracle Analysis", or "Creative Origin" button for non-initial generation cards
            // Check if this is an Oracle-generated idea to show appropriate button text with icon
            const isOracleIdea = idea.oracle_generated && idea.oracle_analysis;
            const isEliteIdea = idea.elite_selected || idea.elite_selected_source;

            // Debug logging - only for elite ideas to help troubleshooting
            if (isEliteIdea || idea.elite_selected_source) {
                console.log(`🌟 ELITE/CREATIVE IDEA DETECTED - Idea ${ideaIndex} in generation ${index}:`, {
                    title: idea.title,
                    elite_selected: idea.elite_selected,
                    elite_selected_source: idea.elite_selected_source,
                    elite_source_id: idea.elite_source_id,
                    elite_source_generation: idea.elite_source_generation,
                    elite_target_generation: idea.elite_target_generation
                });
            }

            let buttonText, buttonTitle, buttonClass;
            if (isEliteIdea) {
                buttonText = '<i class="fas fa-star"></i> Creative';
                buttonTitle = 'View Creative Origin';
                buttonClass = 'view-lineage';
            } else if (isOracleIdea) {
                buttonText = '<i class="fas fa-eye"></i> Oracle';
                buttonTitle = 'View Oracle Analysis';
                buttonClass = 'view-lineage';
            } else {
                buttonText = '<i class="fas fa-project-diagram"></i> Lineage';
                buttonTitle = 'View Lineage';
                buttonClass = 'view-lineage';
            }

            const viewLineageButton = index > 0 ?
                `<button class="btn btn-outline-secondary btn-sm ${buttonClass}" title="${buttonTitle}">
                    ${buttonText}
                </button>` : '';

            card.innerHTML = `
                <div class="card-body">
                    <div class="idea-card-meta">
                        ${islandBadgeHtml}
                        ${migrationBadgeHtml}
                        ${oracleBadgeHtml}
                    </div>
                    <h5 class="card-title">${ideaData.title || 'Untitled'}</h5>
                    <div class="card-text">
                        <p>${plainPreview}</p>
                    </div>
                    ${genotypeSummaryHtml}
                    <div class="card-actions">
                        ${viewPromptButton}
                        ${breedingPromptButton}
                        <button class="btn btn-primary btn-sm view-idea">
                            <i class="fas fa-expand"></i> Full Idea
                        </button>
                        ${viewLineageButton}
                    </div>
                </div>
            `;

            // Add click handler for view button
            const viewButton = card.querySelector('.view-idea');
            viewButton.addEventListener('click', () => {
                showIdeaModal(ideaData);
            });

            // Add click handler for initial prompt button if it exists (generation 0)
            if (index === 0) {
                const viewPromptBtn = card.querySelector('.view-prompt');
                viewPromptBtn.addEventListener('click', () => {
                    // Pass the idea index to show the correct context
                    showPromptModal(ideaIndex, 'initial');
                });
            }

            // Add click handler for breeding prompt button if it exists (generation > 0)
            if (index > 0) {
                const viewBreedingPromptBtn = card.querySelector('.view-breeding-prompt');
                viewBreedingPromptBtn.addEventListener('click', () => {
                    // Pass the generation index and idea index for breeding prompts
                    showPromptModal(ideaIndex, 'breeding', index);
                });
            }

            // Add click handler for view lineage button if it exists
            if (index > 0) {
                const viewLineageBtn = card.querySelector('.view-lineage');
                viewLineageBtn.addEventListener('click', () => {
                    showLineageModal(idea, index);
                });
            }

            if (insertBeforeNode && insertBeforeNode.parentNode === scrollContainer) {
                scrollContainer.insertBefore(card, insertBeforeNode);
            } else {
                scrollContainer.appendChild(card);
            }

            // Log that we've added a new card
            console.log(`Added new idea card: Generation ${index}, Idea ${ideaIndex + 1}`);
        });

        // After all cards are added, check if scrolling is needed
        setTimeout(() => {
            checkScrollOverflow(index);
        }, 100);
    });
}

// Function to check if content overflows and needs scrolling
function checkScrollOverflow(generationIndex) {
    const scrollWrapper = document.getElementById(`scroll-wrapper-${generationIndex}`);
    if (scrollWrapper) {
        const hasOverflow = scrollWrapper.scrollHeight > scrollWrapper.clientHeight;

        // Add or remove attribute to control scroll indicator visibility
        if (hasOverflow) {
            scrollWrapper.removeAttribute('data-no-scroll');
        } else {
            scrollWrapper.setAttribute('data-no-scroll', 'true');
        }

        console.log(`Generation ${generationIndex}: Overflow = ${hasOverflow}, ScrollHeight = ${scrollWrapper.scrollHeight}, ClientHeight = ${scrollWrapper.clientHeight}`);
    }
}

// Add the toggleGeneration function
function toggleGeneration(generationId) {
    const content = document.getElementById(`generation-content-${generationId}`);
    const button = document.querySelector(`#generation-${generationId} .generation-toggle`);
    const icon = button.querySelector('i');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.className = 'fas fa-chevron-up';
        button.setAttribute('aria-expanded', 'true');
    } else {
        content.style.display = 'none';
        icon.className = 'fas fa-chevron-down';
        button.setAttribute('aria-expanded', 'false');
        closeGenerationTournament(generationId);
    }
}

// Update the showIdeaModal function to ensure the modal is properly initialized
function showIdeaModal(idea) {
    // Get the modal element
    const modalElement = document.getElementById('ideaModal');

    // Store the idea for clipboard copying
    currentModalIdea = idea;

    // Set the title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Render the markdown content
    const modalContent = document.getElementById('ideaModalContent');

    // Set the content
    const renderedContent = renderMarkdown(idea.content || '');
    modalContent.innerHTML = renderedContent;

    // Initialize the modal if it hasn't been already
    let modal;
    if (window.bootstrap) {
        modal = new bootstrap.Modal(modalElement);

        // Add event listener to clean up when modal is hidden
        // Use a named function so we can check if it already exists
        const cleanupFunction = function (event) {
            // Remove any lingering backdrop elements
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => {
                backdrop.remove();
            });

            // Reset body classes that might have been added by Bootstrap
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        };

        // We don't want to add multiple cleanup listeners
        // First remove any existing cleanup listener
        modalElement.removeEventListener('hidden.bs.modal', cleanupFunction);

        // Then add our cleanup listener
        // Note: This won't interfere with other named listeners like reopenLineage
        modalElement.addEventListener('hidden.bs.modal', cleanupFunction);

        modal.show();
    } else {
        console.error("Bootstrap is not available. Make sure it's properly loaded.");
        // Fallback for testing - just make the modal visible
        modalElement.style.display = 'block';
    }
}

function updateContextDisplay() {
    const contextDisplay = document.getElementById('contextDisplay');
    const contextContainer = document.getElementById('contextContainer');

    if (contexts.length > 0) {
        // Make sure the container is visible
        contextContainer.style.display = 'block';

        // Use direction prompts if available, otherwise use raw contexts (legacy)
        const displayContent = specificPrompts.length > 0 && specificPrompts[currentContextIndex]
            ? specificPrompts[currentContextIndex]
            : contexts[currentContextIndex];

        const displayTitle = specificPrompts.length > 0 ? "Direction Prompt" : "Context Pool";

        // Format the content into separate items
        const contentItems = displayContent
            .split('\n')
            .filter(item => item.trim())
            .map(item => `<div class="context-item">${item.trim()}</div>`)
            .join('');

        contextDisplay.innerHTML = `
            <div class="context-content">
                <h6 class="mb-3 text-primary">${displayTitle} ${currentContextIndex + 1}</h6>
                ${contentItems}
            </div>
            <div class="context-navigation">
                <div class="context-nav-buttons">
                    <button class="context-nav-btn" id="prevContext" ${currentContextIndex === 0 ? 'disabled' : ''}>
                        ← Previous
                    </button>
                    <span id="contextCounter">${displayTitle} ${currentContextIndex + 1} of ${contexts.length}</span>
                    <button class="context-nav-btn" id="nextContext" ${currentContextIndex === contexts.length - 1 ? 'disabled' : ''}>
                        Next →
                    </button>
                </div>
            </div>
        `;

        // Add event listeners to the newly created buttons
        document.getElementById('prevContext')?.addEventListener('click', () => {
            if (currentContextIndex > 0) {
                currentContextIndex--;
                updateContextDisplay();
            }
        });

        document.getElementById('nextContext')?.addEventListener('click', () => {
            if (currentContextIndex < contexts.length - 1) {
                currentContextIndex++;
                updateContextDisplay();
            }
        });
    } else {
        contextDisplay.innerHTML = '<p class="text-muted">Context will appear here when evolution starts...</p>';
        contextContainer.style.display = 'none';
    }
}

// Clean implementation of downloadResults to prevent double-saving
function downloadResults(data) {
    if (!data) {
        alert('No evolution data available to download');
        return;
    }

    // Use browser's built-in prompt for filename
    const defaultFilename = `evolution_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const filename = prompt("Enter filename to save:", defaultFilename);

    // Exit if user cancels
    if (!filename) {
        console.log("Save cancelled by user");
        return;
    }

    console.log(`Saving to filename: ${filename}`);

    // Include all available data in the save
    const saveData = {
        history: data.history || [],
        diversity_history: data.diversity_history || [],
        contexts: data.contexts || [],
        specific_prompts: data.specific_prompts || specificPrompts || [],
        breeding_prompts: data.breeding_prompts || breedingPrompts || [],
        token_counts: data.token_counts || {},
        timestamp: new Date().toISOString(),
        metadata: {
            total_generations: data.total_generations || (data.history ? data.history.length : 0),
            population_size: data.history && data.history.length > 0 ? data.history[0].length : 0,
            model_used: data.model_type || 'unknown',
            saved_from: 'evolution_viewer'
        }
    };

    console.log("Preparing to save evolution data:", saveData);

    // Disable button to prevent multiple saves
    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        downloadButton.disabled = true;
    }

    // Send to server
    fetch('/api/save-evolution', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            data: saveData,
            filename: filename
        })
    })
        .then(response => {
            console.log("Save response:", response);
            if (!response.ok) {
                throw new Error(`Save failed: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(result => {
            console.log("Save successful:", result);
            alert('Save successful!');
        })
        .catch(error => {
            console.error("Save error:", error);
            alert(`Error saving: ${error.message}`);
        })
        .finally(() => {
            // Re-enable the button
            if (downloadButton) {
                downloadButton.disabled = false;
            }
        });
}

// Global history state
let historyItems = [];
let selectedHistoryItem = null;

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.status === 'success') {
            historyItems = data.items || [];
            renderHistoryList(historyItems);
        } else {
            console.error('Error loading history:', data.message);
            historyItems = [];
            renderHistoryList([]);
        }
    } catch (error) {
        console.error('Error loading history:', error);
        historyItems = [];
        renderHistoryList([]);
    }
}

async function refreshHistoryList() {
    await loadHistory();
}

function renderHistoryList(items, updateGlobal = true) {
    const container = document.getElementById('historyList');
    if (!container) return;

    if (items.length === 0) {
        container.innerHTML = `
            <div class="history-empty">
                <i class="fas fa-folder-open"></i>
                <p>No evolutions found</p>
            </div>
        `;
        return;
    }

    container.innerHTML = items.map(item => {
        const statusClass = `badge-status-${item.status}`;
        const typeClass = `item-type-${item.type}`;
        const date = item.timestamp ? new Date(item.timestamp).toLocaleDateString() : '';
        const time = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        const isSelected = selectedHistoryItem?.id === item.id;

        // Truncate display name (more room in expanded view)
        let displayName = item.display_name || item.id;
        if (displayName.length > 35) {
            displayName = displayName.substring(0, 32) + '...';
        }

        // Build action buttons for expanded state
        let actionsHtml = '';
        if (isSelected) {
            if (item.can_resume) {
                actionsHtml += `<button class="btn btn-sm btn-success" onclick="event.stopPropagation(); handleHistoryResume('${item.id}')">
                    <i class="fas fa-play me-1"></i>Resume
                </button>`;
            }
            if (item.can_continue) {
                actionsHtml += `
                    <div class="continue-action-group d-flex gap-1" id="continueActionGroup-${item.id}">
                        <button class="btn btn-sm btn-info" onclick="event.stopPropagation(); showContinueInputInline('${item.id}', '${item.type}')">
                            <i class="fas fa-forward me-1"></i>Continue
                        </button>
                    </div>`;
            }
            if (item.can_rate) {
                actionsHtml += `<button class="btn btn-sm btn-outline-primary" onclick="event.stopPropagation(); navigateToRatePage('${item.type === 'legacy_saved' ? item.id : ''}')">
                    <i class="fas fa-star me-1"></i>Rate
                </button>`;
            }
            if (item.can_rename) {
                actionsHtml += `<button class="btn btn-sm btn-outline-secondary" onclick="event.stopPropagation(); showRenameInputInline('${item.id}')" title="Rename">
                    <i class="fas fa-edit"></i>
                </button>`;
            }
            if (item.can_delete) {
                actionsHtml += `<button class="btn btn-sm btn-outline-danger" onclick="event.stopPropagation(); confirmDeleteEvolution('${item.id}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>`;
            }
        }

        return `
            <div class="history-item ${isSelected ? 'selected expanded' : ''}"
                 onclick="selectHistoryItem('${item.id}')"
                 data-item-id="${item.id}">
                <div class="item-header">
                    <span class="item-title" id="itemTitle-${item.id}" title="${item.display_name || item.id}">${displayName}</span>
                    <span class="item-type-badge ${typeClass}">${item.type}</span>
                </div>
                <div class="item-meta">
                    <span><i class="fas fa-calendar"></i> ${date} ${time}</span>
                    <span><i class="fas fa-layer-group"></i> Gen ${item.generations}${item.total_generations ? '/' + item.total_generations : ''}</span>
                </div>
                <div class="item-footer">
                    <span class="badge ${statusClass}">${formatStatus(item.status)}</span>
                    ${isSelected ? `<span class="idea-type"><i class="fas fa-tag me-1"></i>${item.idea_type || 'Unknown'}</span>` : ''}
                </div>
                ${isSelected ? `<div class="item-actions">${actionsHtml}</div>` : ''}
            </div>
        `;
    }).join('');
}

function formatStatus(status) {
    const statusMap = {
        'complete': 'Complete',
        'paused': 'Paused',
        'in_progress': 'In Progress',
        'force_stopped': 'Force Stopped',
        'unknown': 'Unknown'
    };
    return statusMap[status] || status;
}

async function selectHistoryItem(itemId) {
    const item = historyItems.find(i => i.id === itemId);
    if (!item) return;

    // Toggle selection - clicking same item deselects
    if (selectedHistoryItem?.id === itemId) {
        selectedHistoryItem = null;
    } else {
        selectedHistoryItem = item;
    }

    // Re-render list with new selection state (expands/collapses in place)
    renderHistoryList(historyItems, false);

    // Load the evolution data into the main content area if selected
    if (selectedHistoryItem) {
        await loadHistoryItemData(item);
    }
}

// Show inline continue input (for history list items)
function showContinueInputInline(itemId, itemType) {
    const group = document.getElementById(`continueActionGroup-${itemId}`);
    if (!group) return;

    group.innerHTML = `
        <input type="number" class="form-control form-control-sm" id="continueGensInput-${itemId}"
               value="3" min="1" max="20" style="width: 55px; text-align: center;"
               onclick="event.stopPropagation()">
        <button class="btn btn-sm btn-info" onclick="event.stopPropagation(); executeContinueInline('${itemId}', '${itemType}')">
            <i class="fas fa-play"></i>
        </button>
        <button class="btn btn-sm btn-outline-secondary" onclick="event.stopPropagation(); cancelContinueInputInline('${itemId}', '${itemType}')">
            <i class="fas fa-times"></i>
        </button>
    `;

    setTimeout(() => {
        const input = document.getElementById(`continueGensInput-${itemId}`);
        if (input) {
            input.focus();
            input.select();
        }
    }, 50);
}

function cancelContinueInputInline(itemId, itemType) {
    const group = document.getElementById(`continueActionGroup-${itemId}`);
    if (!group) return;

    group.innerHTML = `
        <button class="btn btn-sm btn-info" onclick="event.stopPropagation(); showContinueInputInline('${itemId}', '${itemType}')">
            <i class="fas fa-forward me-1"></i>Continue
        </button>
    `;
}

async function executeContinueInline(itemId, itemType) {
    const input = document.getElementById(`continueGensInput-${itemId}`);
    const additionalGens = parseInt(input?.value || '3');

    if (isNaN(additionalGens) || additionalGens < 1) {
        alert("Please enter a valid number of generations (1 or more)");
        return;
    }

    await handleHistoryContinue(itemId, itemType, additionalGens);
}

// Show inline rename input (for history list items)
function showRenameInputInline(itemId) {
    const titleEl = document.getElementById(`itemTitle-${itemId}`);
    if (!titleEl) return;

    const currentName = titleEl.textContent;
    const parent = titleEl.parentNode;

    titleEl.outerHTML = `
        <div class="rename-inline-group d-flex gap-1 flex-grow-1" id="renameGroup-${itemId}">
            <input type="text" class="form-control form-control-sm" id="renameInput-${itemId}"
                   value="${selectedHistoryItem?.display_name || currentName}"
                   onclick="event.stopPropagation()">
            <button class="btn btn-sm btn-success" onclick="event.stopPropagation(); executeRenameInline('${itemId}')">
                <i class="fas fa-check"></i>
            </button>
            <button class="btn btn-sm btn-outline-secondary" onclick="event.stopPropagation(); cancelRenameInline('${itemId}', '${currentName}')">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    setTimeout(() => {
        const input = document.getElementById(`renameInput-${itemId}`);
        if (input) {
            input.focus();
            input.select();
        }
    }, 50);
}

function cancelRenameInline(itemId, originalName) {
    const group = document.getElementById(`renameGroup-${itemId}`);
    if (group) {
        group.outerHTML = `<span class="item-title" id="itemTitle-${itemId}" title="${originalName}">${originalName}</span>`;
    }
}

async function executeRenameInline(itemId) {
    const input = document.getElementById(`renameInput-${itemId}`);
    const newName = input?.value?.trim();

    if (!newName) {
        alert('Please enter a name');
        return;
    }

    try {
        const response = await fetch(`/api/evolution/${itemId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });

        if (response.ok) {
            // Update the item in historyItems
            const item = historyItems.find(i => i.id === itemId);
            if (item) {
                item.display_name = newName;
            }
            if (selectedHistoryItem?.id === itemId) {
                selectedHistoryItem.display_name = newName;
            }
            // Re-render
            renderHistoryList(historyItems, false);
        } else {
            const data = await response.json();
            alert(data.message || 'Failed to rename');
        }
    } catch (error) {
        console.error('Error renaming:', error);
        alert('Error renaming evolution');
    }
}

async function loadHistoryItemData(item) {
    try {
        if (item.type === 'saved' || item.type === 'evolution') {
            // Load from saved evolution file or Firestore
            const response = await fetch(`/api/evolution/${item.id}`);
            if (response.ok) {
                const data = await response.json();
                if (data.data && data.data.history) {
                    // Hide the template entry section
                    const templateEntry = document.getElementById('templateEntry');
                    if (templateEntry) templateEntry.style.display = 'none';

                    // Hide name input section
                    const nameSection = document.getElementById('evolutionNameSection');
                    if (nameSection) nameSection.style.display = 'none';

                    // Hide sidebar to give more canvas space
                    const sidebar = document.querySelector('.left-sidebar');
                    if (sidebar) sidebar.classList.remove('show');

                    // Ensure generations container is visible
                    const genContainer = document.getElementById('generations-container');
                    if (genContainer) genContainer.style.display = 'block';

                    document.getElementById('generations-container').innerHTML = '';
                    renderGenerations(data.data.history);

                    tournamentHistory = data.data.tournament_history || [];
                    renderTournamentDetails(tournamentHistory);

                    if (data.data.diversity_history && data.data.diversity_history.length > 0) {
                        updateDiversityChart(data.data.diversity_history);
                        setTimeout(ensureDiversityChartSizing, 200);
                    } else {
                        resetDiversityPlot();
                    }

                    if (data.data.contexts) {
                        contexts = data.data.contexts;
                        specificPrompts = data.data.specific_prompts || [];
                        breedingPrompts = data.data.breeding_prompts || [];
                        currentContextIndex = 0;
                        updateContextDisplay();
                        document.querySelector('.context-navigation').style.display = 'block';
                    }

                    if (data.data.token_counts) {
                        displayTokenCounts(data.data.token_counts);
                    }

                    const downloadBtn = document.getElementById('downloadButton');
                    if (downloadBtn) {
                        downloadBtn.disabled = false;
                    }
                }
            }
        } else if (item.type === 'checkpoint') {
            // Load from checkpoint
            const response = await fetch(`/api/checkpoints/${item.checkpoint_id}`);
            if (response.ok) {
                const data = await response.json();
                if (data.history) {
                    document.getElementById('generations-container').innerHTML = '';
                    renderGenerations(data.history);

                    tournamentHistory = data.tournament_history || [];
                    renderTournamentDetails(tournamentHistory);

                    if (data.diversity_history && data.diversity_history.length > 0) {
                        updateDiversityChart(data.diversity_history);
                        setTimeout(ensureDiversityChartSizing, 200);
                    } else {
                        resetDiversityPlot();
                    }

                    if (data.contexts) {
                        contexts = data.contexts;
                        specificPrompts = data.specific_prompts || [];
                        breedingPrompts = data.breeding_prompts || [];
                        currentContextIndex = 0;
                        updateContextDisplay();
                        document.querySelector('.context-navigation').style.display = 'block';
                    }

                    document.getElementById('downloadButton').disabled = false;
                }
            }
        }
    } catch (error) {
        console.error('Error loading history item data:', error);
    }
}

async function handleHistoryResume(checkpointId) {
    if (!checkpointId) {
        alert("No checkpoint available to resume from");
        return;
    }

    console.log("Resuming evolution from checkpoint:", checkpointId);

    try {
        const response = await fetch('/api/resume-evolution', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                checkpoint_id: checkpointId,
                additional_generations: 0
            })
        });

        if (response.ok) {
            const data = await response.json();

            // Store checkpoint ID
            localStorage.setItem('lastCheckpointId', data.checkpoint_id || checkpointId);

            // Restore state
            if (data.history) {
                applyHistoryIfNew(data, true);
                if (Number.isInteger(data.history_version)) {
                    lastStoredHistoryVersion = data.history_version;
                }
            }
            if (data.contexts) {
                contexts = data.contexts;
                specificPrompts = data.specific_prompts || [];
                breedingPrompts = data.breeding_prompts || [];
                currentContextIndex = 0;
                updateContextDisplay();
            }

            // Switch to evolution mode (hides template entry, shows evolution view)
            if (typeof showEvolutionMode === 'function') {
                showEvolutionMode();
            } else {
                // Fallback: manually hide template entry
                const templateEntry = document.getElementById('templateEntry');
                if (templateEntry) templateEntry.style.display = 'none';
            }

            // Store evolution name from response
            currentEvolutionName = data.evolution_name || currentEvolutionName;
            currentEvolutionId = data.evolution_id || currentEvolutionId;

            // Create progress bar if needed
            if (!document.getElementById('progress-container')) {
                createProgressBar(currentEvolutionName);
            } else {
                // Update the name in existing progress bar
                const nameEl = document.getElementById('evolution-name');
                if (nameEl && currentEvolutionName) nameEl.textContent = currentEvolutionName;
            }

            // Update UI state to running
            isEvolutionRunning = true;
            evolutionStartTime = Date.now();

            // Set buttons to running state
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            if (startButton) {
                startButton.disabled = true;
                startButton.textContent = 'Running...';
            }
            if (stopButton) {
                stopButton.disabled = false;
                stopButton.style.display = 'block';
            }

            updateElapsedTimeDisplay();

            // Start polling loop
            pollProgress();

            // Clear selection and go back to main view
            clearSelectedEvolution();
            showSidebarView('main');

            addActivityLogItem('🔄 Resuming evolution...', 'info');
        } else {
            const errorData = await response.json().catch(() => ({ message: 'Failed to resume evolution' }));
            console.error("Failed to resume:", errorData);
            if (response.status === 409) {
                showAlreadyRunningMessage();
            } else if (response.status === 429 || errorData.scope === 'global') {
                showSystemBusyMessage(errorData.message);
            } else {
                alert(errorData.message || "Failed to resume evolution. Check console for details.");
            }
            await refreshHistoryList();
        }
    } catch (error) {
        console.error("Error resuming:", error);
        alert("Error resuming evolution: " + error.message);
        await refreshHistoryList();
    }
}

async function handleHistoryContinue(itemId, itemType, additionalGens = 3) {
    if (!itemId) {
        alert("No evolution available to continue");
        return;
    }

    console.log("Continuing evolution for", additionalGens, "more generations");

    // Update the inline action button to show starting state
    const group = document.getElementById(`continueActionGroup-${itemId}`);
    if (group) {
        group.innerHTML = '<span class="text-muted small"><i class="fas fa-spinner fa-spin me-1"></i>Starting...</span>';
    }

    try {
        let response;

        if (itemType === 'checkpoint') {
            // Continue from checkpoint
            response = await fetch('/api/continue-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    checkpoint_id: itemId,
                    additional_generations: additionalGens
                })
            });
        } else {
            // For saved evolutions, use the special load-and-continue endpoint
            response = await fetch('/api/continue-saved-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    evolution_id: itemId,
                    additional_generations: additionalGens
                })
            });
        }

        if (response.ok) {
            const data = await response.json();

            // Store checkpoint ID
            localStorage.setItem('lastCheckpointId', data.checkpoint_id || itemId);

            // Restore state
            if (data.history) {
                applyHistoryIfNew(data, true);
                if (Number.isInteger(data.history_version)) {
                    lastStoredHistoryVersion = data.history_version;
                }
            }
            if (data.contexts) {
                contexts = data.contexts;
                specificPrompts = data.specific_prompts || [];
                breedingPrompts = data.breeding_prompts || [];
                currentContextIndex = 0;
                updateContextDisplay();
            }

            // Switch to evolution mode (hides template entry, shows evolution view)
            if (typeof showEvolutionMode === 'function') {
                showEvolutionMode();
            } else {
                // Fallback: manually hide template entry
                const templateEntry = document.getElementById('templateEntry');
                if (templateEntry) templateEntry.style.display = 'none';
            }

            // Store evolution name from response
            currentEvolutionName = data.evolution_name || currentEvolutionName;
            currentEvolutionId = data.evolution_id || currentEvolutionId;

            // Create progress bar if needed
            if (!document.getElementById('progress-container')) {
                createProgressBar(currentEvolutionName);
            } else {
                // Update the name in existing progress bar
                const nameEl = document.getElementById('evolution-name');
                if (nameEl && currentEvolutionName) nameEl.textContent = currentEvolutionName;
            }

            // Update UI state to running
            isEvolutionRunning = true;
            evolutionStartTime = Date.now();

            // Set buttons to running state
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            if (startButton) {
                startButton.disabled = true;
                startButton.textContent = 'Running...';
            }
            if (stopButton) {
                stopButton.disabled = false;
                stopButton.style.display = 'block';
            }

            updateElapsedTimeDisplay();

            // Start polling loop
            pollProgress();

            // Clear selection and go back to main view
            clearSelectedEvolution();
            showSidebarView('main');

            addActivityLogItem(`📈 Continuing for ${additionalGens} more generations...`, 'info');
        } else {
            const errorData = await response.json().catch(() => ({ message: 'Failed to continue evolution' }));
            console.error("Failed to continue:", errorData);
            if (response.status === 409) {
                showAlreadyRunningMessage();
            } else if (response.status === 429 || errorData.scope === 'global') {
                showSystemBusyMessage(errorData.message);
            } else {
                alert(errorData.message || "Failed to continue evolution. Check console for details.");
            }
            cancelContinueInputInline(itemId, itemType);
        }
    } catch (error) {
        console.error("Error continuing:", error);
        alert("Error continuing evolution: " + error.message);
        cancelContinueInputInline(itemId, itemType);
    }
}

function clearSelectedEvolution() {
    selectedHistoryItem = null;

    const evolutionSelect = document.getElementById('evolutionSelect');
    if (evolutionSelect) {
        evolutionSelect.value = '';
    }

    // Clear selection highlighting in history list
    document.querySelectorAll('.history-item').forEach(el => {
        el.classList.remove('selected');
    });

    // Hide history detail panel
    const detailPanel = document.getElementById('historyDetailPanel');
    if (detailPanel) detailPanel.style.display = 'none';

    // Show the name input section again
    const nameSection = document.getElementById('evolutionNameSection');
    if (nameSection) nameSection.style.display = 'block';

    // Clear the name input
    const nameInput = document.getElementById('evolutionNameInput');
    if (nameInput) nameInput.value = '';
}

// Clear selection in history view
function clearHistorySelection() {
    selectedHistoryItem = null;

    // Re-render list without selection
    renderHistoryList(historyItems, false);

    // Clear main content
    document.getElementById('generations-container').innerHTML = '';
    resetEvolutionTreeView();
    resetDiversityPlot();
}


// Delete evolution with confirmation
async function confirmDeleteEvolution(evolutionId) {
    const item = historyItems.find(i => i.id === evolutionId);
    const name = item?.display_name || evolutionId;

    if (!confirm(`Delete "${name}"?\n\nThis cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/evolution/${evolutionId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            clearHistorySelection();
            await loadHistory();
        } else {
            const data = await response.json();
            alert(data.message || 'Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting:', error);
        alert('Error deleting evolution');
    }
}

function navigateToRatePage(evolutionId) {
    // Switch to rate tab and load the evolution
    const rateTab = document.querySelector('button[onclick*="rate"]') || document.getElementById('navRateBtn');
    if (rateTab) {
        rateTab.click();
    }
    // If evolution ID provided, select it in the rate page
    if (evolutionId) {
        // The rate page will handle loading
    }
}

// Legacy evolutionSelect handler - now handled by history panel
// Keeping for backward compatibility with any code that might set this value
const evolutionSelectEl = document.getElementById('evolutionSelect');
if (evolutionSelectEl) {
    evolutionSelectEl.addEventListener('change', async (e) => {
        const evolutionId = e.target.value;
        if (evolutionId) {
            // Find the item in history and select it
            const item = historyItems.find(i => i.id === evolutionId);
            if (item) {
                await selectHistoryItem(item.id);
            }
        } else {
            clearSelectedEvolution();
            await restoreCurrentEvolution();
        }
    });
}

async function pollProgress() {
    try {
        const response = await fetch('/api/progress');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        let data = await response.json();

        const shouldFetchFullHistory =
            (!data.history && data.history_changed) ||
            (!data.history && data.history_available && (Date.now() - lastFullHistoryFetch) > 15000);

        if (shouldFetchFullHistory) {
            try {
                const fullResponse = await fetch('/api/progress?includeHistory=1');
                if (fullResponse.ok) {
                    const fullData = await fullResponse.json();
                    if (fullData.history) {
                        data = fullData;
                        lastFullHistoryFetch = Date.now();
                    }
                }
            } catch (e) {
                console.log("Full history fetch failed:", e);
            }
        }
        console.log("Progress update:", data); // Add logging to see what's coming from the server

        // Update evolution name if provided
        if (data.evolution_name) {
            currentEvolutionName = data.evolution_name;
            const nameEl = document.getElementById('evolution-name');
            if (nameEl) nameEl.textContent = currentEvolutionName;
        }
        if (data.evolution_id) {
            currentEvolutionId = data.evolution_id;
        }

        // Sync start time from server if available
        if (data.start_time) {
            const serverStartTime = new Date(data.start_time).getTime();
            // detailed check to avoid unnecessary updates but ensure sync
            if (!evolutionStartTime || Math.abs(evolutionStartTime - serverStartTime) > 2000) {
                console.log("Syncing evolution start time from server:", data.start_time);
                evolutionStartTime = serverStartTime;
                if (!elapsedTimeInterval) {
                    startElapsedTimeUpdater();
                }
            }
        }

        // Reset the display only for truly new runs (no history available yet).
        const noInlineHistory = !data.history || data.history.length === 0;
        const noHistoryAvailable = !data.history_available;
        const nothingRenderedYet = lastRenderedHistoryVersion < 0;
        if (data.is_running && noInlineHistory && noHistoryAvailable && nothingRenderedYet) {
            console.log("New evolution detected, resetting UI");
            // Reset generations display but keep progress bar
            const container = document.getElementById('generations-container');
            if (container) {
                container.innerHTML = '';
            }
            resetEvolutionTreeView();
            // Reset diversity plot for new evolution
            resetDiversityPlot();
            showDiversityPlotLoading();

            // Initialize timing for new evolution
            if (!evolutionStartTime) {
                evolutionStartTime = Date.now();
                lastActivityTime = Date.now();
                activityLog = [];
            }
        }

        // Update progress bar and percentage display
        const progressBar = document.getElementById('evolution-progress');
        const progressPercentage = document.getElementById('progress-percentage');

        if (progressBar) {
            // Cap progress at 100% to handle edge cases in calculation
            const progress = Math.min(data.progress || 0, 100);
            const roundedProgress = Math.round(progress * 10) / 10;
            if (roundedProgress !== lastProgressBarValue) {
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
                lastProgressBarValue = roundedProgress;
            }

            // Update prominent percentage display
            if (progressPercentage) {
                const progressLabel = `${Math.round(progress)}%`;
                if (progressPercentage.textContent !== progressLabel) {
                    progressPercentage.textContent = progressLabel;
                }
            }

            // Cap the display at 100%
            if (progress >= 100 && progressBar.style.width !== '100%') {
                progressBar.style.width = '100%';
            }

            // Parse and enhance status messages for activity log
            let activityMessage = '';
            let activityType = 'info';

            if (data.status_message) {
                activityMessage = enhanceStatusMessage(data.status_message, data);
            } else if (data.current_generation === 0) {
                activityMessage = 'Building initial population...';
            } else {
                activityMessage = `Evolving generation ${data.current_generation}...`;
            }

            // Add to activity log if message changed (better duplicate detection)
            if (activityMessage && activityMessage !== previousStatusMessage) {
                addActivityLogItem(activityMessage, activityType);
                previousStatusMessage = activityMessage;
            }

            // Update last activity time on any progress update
            lastActivityTime = Date.now();
            previousProgress = progress;
        }

        // Update status indicator
        if (data.is_running) {
            const genInfo = data.current_generation === 0
                ? 'Initial Population'
                : `Gen ${data.current_generation}/${data.total_generations}`;
            updateEvolutionStatusIndicator(true, `Running: ${genInfo}`);
        }

        // Update UI only when there is a new history version (or when version is unavailable).
        if (data.history && data.history.length > 0) {
            const rendered = applyHistoryIfNew(data);
            if (rendered) {
                const historyVersionToStore = Number.isInteger(data.history_version) ? data.history_version : null;
                const shouldStore =
                    historyVersionToStore === null ||
                    historyVersionToStore !== lastStoredHistoryVersion;
                if (shouldStore) {
                    const evolutionStateToStore = {
                        history: data.history,
                        history_version: historyVersionToStore ?? lastRenderedHistoryVersion,
                        diversity_history: data.diversity_history || [],
                        contexts: data.contexts || contexts,
                        specific_prompts: data.specific_prompts || specificPrompts,
                        breeding_prompts: data.breeding_prompts || breedingPrompts,
                        tournament_history: data.tournament_history || tournamentHistory,
                        token_counts: data.token_counts || null
                    };
                    schedulePersistEvolutionState(evolutionStateToStore);
                    if (historyVersionToStore !== null) {
                        lastStoredHistoryVersion = historyVersionToStore;
                    }
                }
            }
        }

        if (data.tournament_history) {
            const nextTournamentSignature = safeSignature(data.tournament_history);
            if (nextTournamentSignature !== lastTournamentHistorySignature) {
                tournamentHistory = data.tournament_history;
                renderTournamentDetails(tournamentHistory);
                lastTournamentHistorySignature = nextTournamentSignature;
            }
        }

        // Handle diversity updates with activity logging
        if (data.diversity_history && data.diversity_history.length > 0) {
            const latestDiversity = data.diversity_history[data.diversity_history.length - 1];
            // Only log if we have a new diversity calculation
            if (latestDiversity && latestDiversity.diversity_score !== undefined) {
                const diversityGenIndex = data.diversity_history.length - 1;
                const diversityKey = `diversity_gen_${diversityGenIndex}`;
                if (!window._loggedDiversityGens) window._loggedDiversityGens = new Set();
                if (!window._loggedDiversityGens.has(diversityKey)) {
                    window._loggedDiversityGens.add(diversityKey);
                    const score = latestDiversity.diversity_score.toFixed(3);
                    const genLabel = diversityGenIndex === 0 ? 'initial population' : `generation ${diversityGenIndex}`;
                    addActivityLogItem(`📊 Diversity calculated for ${genLabel}: ${score}`, 'info');
                }
            }
            handleDiversityUpdate(data);
        }

        // Handle oracle updates with activity logging
        if (data.oracle_update) {
            const oracleKey = `${currentEvolutionId || 'na'}:${Number.isInteger(data.history_version) ? data.history_version : 'na'}:${data.current_generation || 'na'}:${Math.round((data.progress || 0) * 10) / 10}`;
            if (oracleKey !== lastOracleUpdateKey) {
                addActivityLogItem('🔮 Oracle injected diverse idea into population', 'info');
                lastOracleUpdateKey = oracleKey;
            }
        }

        // Handle elite selection updates
        if (data.elite_selection_update) {
            const eliteKey = `${currentEvolutionId || 'na'}:${Number.isInteger(data.history_version) ? data.history_version : 'na'}:${data.current_generation || 'na'}:${Math.round((data.progress || 0) * 10) / 10}`;
            if (eliteKey !== lastEliteUpdateKey) {
                addActivityLogItem('⭐ Elite idea selected for next generation', 'info');
                lastEliteUpdateKey = eliteKey;
            }
        }

        // Display token counts if available (Live updates)
        if (data.token_counts) {
            displayTokenCounts(data.token_counts);
        }

        if (data.contexts && data.contexts.length > 0) {
            const nextContextSignature = safeSignature({
                contexts: data.contexts,
                specific_prompts: data.specific_prompts || [],
                breeding_prompts: data.breeding_prompts || [],
            });
            if (nextContextSignature !== lastContextSignature) {
                contexts = data.contexts;
                specificPrompts = data.specific_prompts || [];
                breedingPrompts = data.breeding_prompts || [];
                currentContextIndex = 0;
                updateContextDisplay();
                const contextNav = document.querySelector('.context-navigation');
                if (contextNav) {
                    contextNav.style.display = 'block';
                }
                lastContextSignature = nextContextSignature;
            }
        }

        // Continue polling if evolution is still running
        if (data.is_running) {
            isEvolutionRunning = true;
            setTimeout(pollProgress, 1000); // Poll every second
        } else {
            // Evolution complete or stopped
            isEvolutionRunning = false;

            // Mark as complete with proper visual feedback
            // Pass resumable state and checkpoint ID for pause/resume functionality
            markEvolutionComplete(
                data.is_stopped,
                data.is_resumable || false,
                data.checkpoint_id || null
            );

            if (data.history && data.history.length > 0) {
                // Save final state and enable save button
                currentEvolutionData = data;
                applyHistoryIfNew(data, true);

                // Store checkpoint ID if available for future resume/continue
                if (data.checkpoint_id) {
                    localStorage.setItem('lastCheckpointId', data.checkpoint_id);
                }

                // Store the final evolution data in localStorage including diversity data and token counts
                const evolutionStateToStore = {
                    history: data.history,
                    history_version: Number.isInteger(data.history_version) ? data.history_version : lastRenderedHistoryVersion,
                    diversity_history: data.diversity_history || [],
                    contexts: data.contexts || contexts,
                    specific_prompts: data.specific_prompts || specificPrompts,
                    breeding_prompts: data.breeding_prompts || breedingPrompts,
                    tournament_history: data.tournament_history || tournamentHistory,
                    token_counts: data.token_counts || null,
                    checkpoint_id: data.checkpoint_id || null
                };
                schedulePersistEvolutionState(evolutionStateToStore);
                if (Number.isInteger(data.history_version)) {
                    lastStoredHistoryVersion = data.history_version;
                }

                if (data.tournament_history) {
                    tournamentHistory = data.tournament_history;
                    renderTournamentDetails(tournamentHistory);
                }

                // Set up the download button
                setupDownloadButton(data);

                // Display token counts if available
                if (data.token_counts) {
                    displayTokenCounts(data.token_counts);
                    const diagnosticsSummary = getDiagnosticsSummary(data.token_counts);
                    if (diagnosticsSummary) {
                        latestDiagnosticsTokenCounts = data.token_counts;
                        const diagnosticsKey = `${currentEvolutionId || 'na'}:${diagnosticsSummary.signature}`;
                        if (diagnosticsKey !== lastDiagnosticsSummaryKey) {
                            if (diagnosticsSummary.totalCount > 0) {
                                const detailText = diagnosticsSummary.topEntries.length > 0
                                    ? ` (${diagnosticsSummary.topEntries.join(', ')})`
                                    : '';
                                addActivityLogItem(
                                    `🧪 Diagnostics recorded ${diagnosticsSummary.totalCount} recoverable issue(s)${detailText}`,
                                    'warning',
                                    { action: 'show-diagnostics', actionLabel: 'View details' }
                                );
                            } else {
                                addActivityLogItem('🧪 Diagnostics clean: no recoverable parsing/generation issues', 'success');
                            }
                            lastDiagnosticsSummaryKey = diagnosticsKey;
                        }
                    }
                }

                // Final diversity update
                if (data.diversity_history) {
                    handleDiversityUpdate(data);
                }
            }

            // Show completion/stop notification
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');

            if (data.is_stopped) {
                startButton.textContent = 'Evolution Stopped';
                startButton.disabled = false;
                stopButton.style.display = 'none';
                stopButton.disabled = true;
                stopButton.textContent = 'Stop Evolution';

                setTimeout(() => {
                    startButton.textContent = 'Start Evolution';
                }, 3000);
            } else {
                startButton.textContent = 'Evolution Complete!';
                startButton.disabled = false;
                stopButton.style.display = 'none';
                stopButton.disabled = true;
                stopButton.textContent = 'Stop Evolution';

                setTimeout(() => {
                    startButton.textContent = 'Start Evolution';
                }, 2000);
            }

            // Reset tracking variables for next evolution
            previousStatusMessage = '';
            previousProgress = 0;
        }

        // Handle explicit errors from backend
        if (data.error) {
            console.error("Evolution error:", data.error);
            showErrorMessage(`Evolution Error: ${data.error}`);
            addActivityLogItem(`❌ Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error polling progress:', error);
        // Continue polling even if there's an error, but only if evolution is still running
        if (isEvolutionRunning) {
            setTimeout(pollProgress, 2000); // Longer timeout on error
        }
    }
}

// Add a function to show error messages
function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.position = 'fixed';
    errorDiv.style.top = '20px';
    errorDiv.style.left = '50%';
    errorDiv.style.transform = 'translateX(-50%)';
    errorDiv.style.backgroundColor = '#f44336';
    errorDiv.style.color = 'white';
    errorDiv.style.padding = '15px';
    errorDiv.style.borderRadius = '5px';
    errorDiv.style.zIndex = '1001';
    errorDiv.textContent = message;

    document.body.appendChild(errorDiv);

    // Remove after 3 seconds
    setTimeout(() => {
        if (document.body.contains(errorDiv)) {
            document.body.removeChild(errorDiv);
        }
    }, 3000);
}

// Function to reset button states
function resetButtonStates() {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const forceStopButton = document.getElementById('forceStopButton');

    if (startButton) {
        startButton.disabled = false;
        startButton.textContent = 'Start Evolution';
    }

    if (stopButton) {
        stopButton.disabled = true;
        stopButton.textContent = 'Stop Evolution';
        stopButton.style.display = 'none';
    }

    if (forceStopButton) {
        forceStopButton.style.display = 'none';
    }

    // Show name input section again
    const nameSection = document.getElementById('evolutionNameSection');
    if (nameSection) nameSection.style.display = 'block';

    // Clear the name input for next evolution
    const nameInput = document.getElementById('evolutionNameInput');
    if (nameInput) nameInput.value = '';
}

/**
 * Show a user-friendly message when an evolution is already running
 * Offers a button to view the current evolution progress
 */
function showAlreadyRunningMessage() {
    // Check if modal already exists
    let modal = document.getElementById('alreadyRunningModal');
    if (!modal) {
        // Create the modal
        modal = document.createElement('div');
        modal.id = 'alreadyRunningModal';
        modal.className = 'modal fade';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-warning text-dark">
                        <h5 class="modal-title">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Evolution Already Running
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>You already have an evolution in progress. You can only run one evolution at a time.</p>
                        <p class="mb-0 text-muted">Click below to view your current evolution, or wait for it to complete before starting a new one.</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="viewCurrentEvolutionBtn">
                            <i class="fas fa-eye me-1"></i>View Current Evolution
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Add event listener for the view button
        document.getElementById('viewCurrentEvolutionBtn').addEventListener('click', async function () {
            // Close the modal
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) bsModal.hide();

            // Start polling to sync with the running evolution
            isEvolutionRunning = true;
            pollProgress();

            // Show the stop button
            const stopButton = document.getElementById('stopButton');
            if (stopButton) {
                stopButton.disabled = false;
                stopButton.style.display = 'block';
            }

            // Update start button to show running state
            const startButton = document.getElementById('startButton');
            if (startButton) {
                startButton.disabled = true;
                startButton.textContent = 'Running...';
            }

            // Add activity log message
            addActivityLogItem('🔄 Reconnected to running evolution', 'info');
        });
    }

    // Show the modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

/**
 * Show a user-friendly message when the system is at capacity.
 */
function showSystemBusyMessage(message) {
    const text = message || 'System busy. Another evolution is running. Try again shortly.';
    alert(text);
}

/**
 * Check if there's a running evolution and show a persistent banner
 * Called after auth is complete to detect reconnection opportunities
 */
async function checkRunningEvolution() {
    try {
        const response = await fetch('/api/progress');
        if (!response.ok) {
            console.log("Could not check evolution status:", response.status);
            hideRunningEvolutionBanner();
            return;
        }

        const progressData = await response.json();
        console.log("Evolution status check:", progressData);

        if (progressData.is_running) {
            // There's an evolution running - show the banner
            showRunningEvolutionBanner(progressData);
        } else {
            // No evolution running - hide the banner if present
            hideRunningEvolutionBanner();
        }
    } catch (error) {
        console.error("Error checking running evolution:", error);
        hideRunningEvolutionBanner();
    }
}

/**
 * Show the persistent "evolution running" banner at the top of the page
 */
function showRunningEvolutionBanner(progressData) {
    let banner = document.getElementById('runningEvolutionBanner');

    if (!banner) {
        // Create the banner
        banner = document.createElement('div');
        banner.id = 'runningEvolutionBanner';
        banner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            z-index: 9998;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            font-size: 14px;
        `;

        banner.innerHTML = `
            <div class="d-flex align-items-center gap-2">
                <div class="spinner-border spinner-border-sm text-light" role="status">
                    <span class="visually-hidden">Running...</span>
                </div>
                <span id="bannerStatusText">Evolution running...</span>
            </div>
            <div class="d-flex align-items-center gap-2">
                <span id="bannerProgressText" class="badge bg-light text-dark">0%</span>
                <button id="reconnectEvolutionBtn" class="btn btn-sm btn-light">
                    <i class="fas fa-eye me-1"></i>View Evolution
                </button>
            </div>
        `;

        // Insert at the very beginning of body
        document.body.insertBefore(banner, document.body.firstChild);

        // Add top padding to body to prevent content overlap
        document.body.style.paddingTop = (banner.offsetHeight) + 'px';

        // Add click handler for reconnect button
        document.getElementById('reconnectEvolutionBtn').addEventListener('click', reconnectToRunningEvolution);
    }

    // Update status text and progress
    const statusText = document.getElementById('bannerStatusText');
    const progressText = document.getElementById('bannerProgressText');

    if (progressData.evolution_name) {
        statusText.textContent = `"${progressData.evolution_name}" running...`;
    } else {
        statusText.textContent = `Evolution running...`;
    }

    if (progressData.progress !== undefined) {
        progressText.textContent = `${Math.round(progressData.progress)}%`;
    }

    banner.style.display = 'flex';
}

/**
 * Hide the running evolution banner
 */
function hideRunningEvolutionBanner() {
    const banner = document.getElementById('runningEvolutionBanner');
    if (banner) {
        banner.style.display = 'none';
        document.body.style.paddingTop = '0';
    }
}

/**
 * Reconnect to the running evolution
 */
async function reconnectToRunningEvolution() {
    console.log("Reconnecting to running evolution...");

    // Get current progress data to get evolution name
    try {
        const response = await fetch('/api/progress?includeHistory=1');
        if (response.ok) {
            const progressData = await response.json();
            lastFullHistoryFetch = Date.now();

            // Store evolution identity
            if (progressData.evolution_id) {
                currentEvolutionId = progressData.evolution_id;
            }
            if (progressData.evolution_name) {
                currentEvolutionName = progressData.evolution_name;
            }

            if (progressData.history) {
                applyHistoryIfNew(progressData, true);
            }
            if (progressData.contexts) {
                contexts = progressData.contexts;
                specificPrompts = progressData.specific_prompts || [];
                breedingPrompts = progressData.breeding_prompts || [];
                currentContextIndex = 0;
                updateContextDisplay();
            }
        }
    } catch (e) {
        console.log("Could not get progress data for reconnect:", e);
    }

    // Hide the template entry section (the "What kind of ideas do you want to generate?" prompt)
    const templateEntry = document.getElementById('templateEntry');
    if (templateEntry) {
        templateEntry.style.display = 'none';
    }

    // Hide name input section
    const nameSection = document.getElementById('evolutionNameSection');
    if (nameSection) {
        nameSection.style.display = 'none';
    }

    // Create progress bar if it doesn't exist
    if (!document.getElementById('progress-container')) {
        createProgressBar(currentEvolutionName);
    }

    // Start polling to sync with the running evolution
    isEvolutionRunning = true;
    pollProgress();

    // Show the stop button
    const stopButton = document.getElementById('stopButton');
    if (stopButton) {
        stopButton.disabled = false;
        stopButton.style.display = 'block';
    }

    // Update start button to show running state
    const startButton = document.getElementById('startButton');
    if (startButton) {
        startButton.disabled = true;
        startButton.textContent = 'Running...';
    }

    // Add activity log message
    addActivityLogItem('🔄 Reconnected to running evolution', 'info');

    // Hide the banner since we're now watching
    hideRunningEvolutionBanner();
}

// Expose checkRunningEvolution globally so auth_logic.js can call it
window.checkRunningEvolution = checkRunningEvolution;

// Function to show resume button when evolution is paused/stopped
// Now just stores checkpoint ID - the history panel handles the UI
function showResumeButton(checkpointId) {
    if (checkpointId) {
        localStorage.setItem('lastCheckpointId', checkpointId);
    }
    // History panel handles resume/continue UI now
}

// Function to show force stop button after a delay (when regular stop is taking too long)
function showForceStopButton() {
    const forceStopButton = document.getElementById('forceStopButton');
    if (forceStopButton) {
        forceStopButton.style.display = 'block';
    }
}

// Function to check for resumable checkpoints on page load
// Now just logs info - the history panel handles UI for resume/continue
async function checkForResumableCheckpoints() {
    try {
        const response = await fetch('/api/checkpoints');
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'success' && data.checkpoints && data.checkpoints.length > 0) {
                // Find the most recent paused or in_progress checkpoint
                const resumable = data.checkpoints.find(cp =>
                    cp.status === 'paused' || cp.status === 'in_progress' || cp.status === 'force_stopped'
                );
                if (resumable) {
                    console.log("Found resumable checkpoint:", resumable);
                    localStorage.setItem('lastCheckpointId', resumable.id);
                    // Log to activity if UI is ready
                    if (typeof addActivityLogItem === 'function') {
                        try {
                            addActivityLogItem(`💾 Resumable checkpoint found - check History to resume`, 'info');
                        } catch (e) {
                            console.log("Resumable checkpoint:", resumable.id);
                        }
                    }
                    return resumable;
                }
            }
        }
    } catch (error) {
        console.error("Error checking for checkpoints:", error);
    }
    return null;
}

// Function to reset the UI state
function resetUIState() {
    // Clear generations container
    const container = document.getElementById('generations-container');
    if (container) {
        container.innerHTML = '';
    }
    resetEvolutionTreeView();

    // Reset contexts
    contexts = [];
    specificPrompts = [];
    breedingPrompts = [];
    currentContextIndex = 0;

    // Reset context display
    const contextDisplay = document.getElementById('contextDisplay');
    if (contextDisplay) {
        contextDisplay.innerHTML = '<p class="text-muted">Context will appear here when evolution starts...</p>';
    }

    // Hide context container
    const contextContainer = document.getElementById('contextContainer');
    if (contextContainer) {
        contextContainer.style.display = 'none';
    }

    // Hide context navigation
    const contextNav = document.querySelector('.context-navigation');
    if (contextNav) {
        contextNav.style.display = 'none';
    }

    // Stop any existing elapsed time updater FIRST
    stopElapsedTimeUpdater();

    // Reset timing and activity tracking BEFORE creating new progress bar
    evolutionStartTime = null;
    lastActivityTime = null;
    activityLog = [];
    previousStatusMessage = '';
    previousProgress = 0;
    lastProgressBarValue = -1;
    lastContextSignature = '';
    lastTournamentHistorySignature = '';
    lastTokenCountsSignature = '';
    lastOracleUpdateKey = '';
    lastEliteUpdateKey = '';

    // Reset diversity tracking
    window._loggedDiversityGens = new Set();

    // Reset evolution status
    isEvolutionRunning = false;
    currentEvolutionData = null;
    tournamentHistory = [];
    openTournamentPanelGeneration = null;
    tournamentCountTouched = false;
    lastRenderedHistoryVersion = -1;
    lastStoredHistoryVersion = -1;

    // Reset progress bar if it exists, or create a new one
    const existingProgressContainer = document.getElementById('progress-container');
    if (existingProgressContainer) {
        // Remove the old progress container
        existingProgressContainer.remove();
    }
    // Create fresh progress bar with activity log (this starts the timer)
    createProgressBar(currentEvolutionName);

    // Add initial activity for context generation
    addActivityLogItem('🎯 Generating seed contexts...', 'info');
    updateEvolutionStatusIndicator(true, 'Generating contexts...');

    // Clear localStorage data
    if (persistStateTimeout) {
        clearTimeout(persistStateTimeout);
        persistStateTimeout = null;
    }
    localStorage.removeItem('currentEvolutionData');

    // Reset download button
    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        downloadButton.disabled = true;
    }

    // Remove token counts container if it exists
    const tokenCountsContainer = document.getElementById('token-counts-container');
    if (tokenCountsContainer) {
        tokenCountsContainer.innerHTML = '';
    }
}

// Function to create the progress bar with enhanced visual feedback
function createProgressBar(evolutionName = null) {
    currentEvolutionName = evolutionName;

    const progressContainer = document.createElement('div');
    progressContainer.id = 'progress-container';
    progressContainer.className = 'mb-4 evolution-progress-card';
    progressContainer.innerHTML = `
        <div class="evolution-progress-header">
            <div class="evolution-name-section">
                <h4 class="evolution-name" id="evolution-name">${evolutionName || 'Evolution'}</h4>
            </div>
            <div class="evolution-status-section">
            <div class="evolution-status-indicator">
                <div class="evolution-spinner" id="evolution-spinner"></div>
                <span id="evolution-status-text">Initializing...</span>
            </div>
            <div class="evolution-timing" id="evolution-timing">
                <span class="timing-item" id="elapsed-time">
                    <i class="fas fa-clock"></i> <span id="elapsed-time-value">0:00</span>
                </span>
                </div>
            </div>
        </div>
        <div class="progress-container-inner">
            <div class="progress evolution-progress-bar">
                <div id="evolution-progress" class="progress-bar" role="progressbar"
                     style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                <div class="progress-bar-activity" id="progress-bar-activity"></div>
            </div>
            <div class="progress-percentage" id="progress-percentage">0%</div>
        </div>
        <div class="evolution-activity-log" id="evolution-activity-log">
            <div class="activity-log-header">
                <i class="fas fa-stream"></i> Recent Activity
            </div>
            <div class="activity-log-items" id="activity-log-items">
                <div class="activity-item activity-item-placeholder">Waiting for first task...</div>
            </div>
        </div>
    `;

    // Insert progress bar before generations container
    const generationsContainer = document.getElementById('generations-container');
    generationsContainer.parentNode.insertBefore(progressContainer, generationsContainer);

    // Initialize timing
    evolutionStartTime = Date.now();
    lastActivityTime = Date.now();
    activityLog = [];

    // Reset diversity tracking for new evolution
    window._loggedDiversityGens = new Set();

    // Start elapsed time updater immediately
    startElapsedTimeUpdater();

}

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function getTournamentEntryForGeneration(generationIndex) {
    return tournamentHistory.find((entry) => Number(entry.generation) === Number(generationIndex)) || null;
}

function buildTournamentRoundsHtml(rounds) {
    return (rounds || []).map((round) => {
        const bye = round.bye
            ? `<div class="tournament-bye">Bye: ${escapeHtml(round.bye.title || `Idea ${round.bye.idx}`)}</div>`
            : '';
        const pairs = (round.pairs || []).map((pair) => {
            const aTitle = escapeHtml(pair.a_title || `Idea ${pair.a_idx}`);
            const bTitle = escapeHtml(pair.b_title || `Idea ${pair.b_idx}`);
            const aWinner = pair.winner === 'A';
            const bWinner = pair.winner === 'B';
            const tie = pair.winner === 'tie';
            return `
                <div class="tournament-match-card">
                    <div class="match-row ${aWinner ? 'winner' : ''}">
                        <span class="match-name">${aTitle}</span>
                        ${aWinner ? '<span class="match-badge">Winner</span>' : ''}
                    </div>
                    <div class="match-row ${bWinner ? 'winner' : ''}">
                        <span class="match-name">${bTitle}</span>
                        ${bWinner ? '<span class="match-badge">Winner</span>' : ''}
                    </div>
                    ${tie ? '<div class="match-tie">Tie</div>' : ''}
                </div>
            `;
        }).join('');

        return `
            <div class="tournament-round-column">
                <div class="tournament-round-title">Round ${round.round || ''}</div>
                ${bye}
                ${pairs || '<div class="text-muted">No matches recorded.</div>'}
            </div>
        `;
    }).join('');
}

function closeGenerationTournament(generationIndex) {
    const chip = document.getElementById(`generation-tournament-chip-${generationIndex}`);
    const panel = document.getElementById(`generation-tournament-popout-${generationIndex}`);
    if (chip) {
        chip.classList.remove('is-open');
        chip.setAttribute('aria-expanded', 'false');
    }
    if (panel) {
        panel.hidden = true;
    }
    if (openTournamentPanelGeneration === generationIndex) {
        openTournamentPanelGeneration = null;
    }
}

function openGenerationTournament(generationIndex) {
    if (openTournamentPanelGeneration !== null && openTournamentPanelGeneration !== generationIndex) {
        closeGenerationTournament(openTournamentPanelGeneration);
    }
    const chip = document.getElementById(`generation-tournament-chip-${generationIndex}`);
    const panel = document.getElementById(`generation-tournament-popout-${generationIndex}`);
    if (!chip || !panel || chip.hidden) {
        return;
    }
    chip.classList.add('is-open');
    chip.setAttribute('aria-expanded', 'true');
    panel.hidden = false;
    openTournamentPanelGeneration = generationIndex;
}

function toggleGenerationTournament(generationIndex) {
    if (openTournamentPanelGeneration === generationIndex) {
        closeGenerationTournament(generationIndex);
        return;
    }
    openGenerationTournament(generationIndex);
}

function renderGenerationTournamentDetails(generationIndex) {
    const chip = document.getElementById(`generation-tournament-chip-${generationIndex}`);
    const panel = document.getElementById(`generation-tournament-popout-${generationIndex}`);
    if (!chip || !panel) {
        return;
    }

    const entry = getTournamentEntryForGeneration(generationIndex);
    if (!entry) {
        chip.hidden = true;
        panel.hidden = true;
        panel.innerHTML = '';
        if (openTournamentPanelGeneration === generationIndex) {
            openTournamentPanelGeneration = null;
        }
        return;
    }

    chip.hidden = false;

    const roundsHtml = buildTournamentRoundsHtml(entry.rounds || []);
    panel.innerHTML = `
        <div class="generation-tournament-popout-header">
            <h6 class="mb-0">Swiss Tournament Details · Gen ${escapeHtml(entry.generation)}</h6>
            <button type="button" class="generation-tournament-close">Close</button>
        </div>
        <div class="tournament-grid">
            ${roundsHtml || '<div class="text-muted">No rounds recorded.</div>'}
        </div>
    `;

    const closeButton = panel.querySelector('.generation-tournament-close');
    if (closeButton) {
        closeButton.addEventListener('click', () => closeGenerationTournament(generationIndex));
    }

    if (openTournamentPanelGeneration === generationIndex) {
        openGenerationTournament(generationIndex);
    } else {
        closeGenerationTournament(generationIndex);
    }
}

function renderTournamentDetails(history) {
    tournamentHistory = Array.isArray(history) ? history : [];

    if (
        openTournamentPanelGeneration !== null &&
        !getTournamentEntryForGeneration(openTournamentPanelGeneration)
    ) {
        openTournamentPanelGeneration = null;
    }

    const generationSections = document.querySelectorAll('.generation-section');
    generationSections.forEach((section) => {
        const match = section.id.match(/^generation-(\d+)$/);
        if (!match) return;
        const generationIndex = Number(match[1]);
        renderGenerationTournamentDetails(generationIndex);
    });
}

// Function to start the elapsed time updater
function startElapsedTimeUpdater() {
    // Clear any existing interval
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
    }

    // Update immediately on start
    updateElapsedTimeDisplay();

    elapsedTimeInterval = setInterval(() => {
        updateElapsedTimeDisplay();
    }, 1000);
}

// Function to update the elapsed time display
function updateElapsedTimeDisplay() {
    if (!evolutionStartTime) {
        return;
    }

    const elapsed = Date.now() - evolutionStartTime;
    const elapsedTimeEl = document.getElementById('elapsed-time-value');
    if (elapsedTimeEl) {
        elapsedTimeEl.textContent = formatDuration(elapsed);
    }

    // Check for inactivity (more than 5 seconds since last activity)
    const timeSinceActivity = Date.now() - lastActivityTime;
    const activityIndicator = document.getElementById('progress-bar-activity');
    if (activityIndicator) {
        if (timeSinceActivity > 5000) {
            // Show pulsing animation to indicate waiting
            activityIndicator.classList.add('waiting');
        } else {
            activityIndicator.classList.remove('waiting');
        }
    }
}

// Function to stop the elapsed time updater
function stopElapsedTimeUpdater() {
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
        elapsedTimeInterval = null;
    }
}

// Function to format duration in mm:ss or hh:mm:ss
function formatDuration(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// Function to enhance status messages for better readability
function enhanceStatusMessage(rawMessage, data) {
    if (!rawMessage) return '';

    // Parse common patterns and make them more descriptive
    // Order matters - more specific patterns first
    const patterns = [
        {
            match: /Seeding idea (\d+)\/(\d+)/i,
            transform: (m) => `🌱 Creating seed idea ${m[1]} of ${m[2]}`
        },
        {
            // Breeding comes after tournament in gen 1+
            match: /Breeding and refining idea (\d+)\/(\d+)/i,
            transform: (m) => `🧬 Breeding offspring ${m[1]} of ${m[2]}`
        },
        {
            // Refining in gen 0 (initial population polish)
            match: /Refining idea (\d+)\/(\d+)/i,
            transform: (m, data) => {
                // Check if this is gen 0 (initial population) or later
                if (data && data.current_generation === 0) {
                    return `✨ Polishing seed ${m[1]} of ${m[2]}`;
                }
                return `✨ Refining idea ${m[1]} of ${m[2]}`;
            }
        },
        {
            match: /Running Swiss round (\d+)\/(\d+)/i,
            transform: (m) => `🏆 Swiss round ${m[1]} of ${m[2]}`
        },
        {
            match: /Running tournament/i,
            transform: () => '🏆 Running Swiss tournament...'
        }
    ];

    for (const pattern of patterns) {
        const match = rawMessage.match(pattern.match);
        if (match) {
            return pattern.transform(match, data);
        }
    }

    // Return original message if no pattern matches
    return rawMessage;
}

// Function to add an activity to the log
function handleActivityAction(action) {
    if (action === 'show-diagnostics') {
        if (latestDiagnosticsTokenCounts) {
            showTokenDetailsModal(latestDiagnosticsTokenCounts, { focus: 'diagnostics' });
            return;
        }
        addActivityLogItem('Diagnostics details are not available for this run yet.', 'warning');
    }
}

function addActivityLogItem(message, type = 'info', options = {}) {
    lastActivityTime = Date.now();

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const itemId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // Add to activity log array
    activityLog.unshift({
        id: itemId,
        message,
        type,
        timestamp,
        action: options.action || null,
        actionLabel: options.actionLabel || 'View details',
    });

    // Keep only the most recent items
    if (activityLog.length > MAX_ACTIVITY_LOG_ITEMS) {
        activityLog = activityLog.slice(0, MAX_ACTIVITY_LOG_ITEMS);
    }

    // Update the UI
    const logContainer = document.getElementById('activity-log-items');
    if (logContainer) {
        if (!logContainer.dataset.boundActions) {
            logContainer.addEventListener('click', (event) => {
                const button = event.target.closest('.activity-action-btn');
                if (!button) return;
                const action = button.dataset.action;
                if (action) {
                    handleActivityAction(action);
                }
            });
            logContainer.dataset.boundActions = 'true';
        }

        logContainer.innerHTML = activityLog.map(item => `
            <div class="activity-item activity-${item.type}">
                <span class="activity-time">${item.timestamp}</span>
                <span class="activity-message">${item.message}</span>
                ${item.action ? `<button type="button" class="btn btn-link btn-sm activity-action-btn ms-2 p-0 align-baseline" data-action="${escapeHtml(item.action)}">${escapeHtml(item.actionLabel)}</button>` : ''}
            </div>
        `).join('');
    }
}

// Function to update the evolution status indicator
function updateEvolutionStatusIndicator(isRunning, statusText = '') {
    const spinner = document.getElementById('evolution-spinner');
    const statusTextEl = document.getElementById('evolution-status-text');
    const progressCard = document.getElementById('progress-container');

    if (spinner) {
        if (isRunning) {
            spinner.classList.add('spinning');
            spinner.classList.remove('complete', 'stopped');
        } else {
            spinner.classList.remove('spinning');
        }
    }

    if (statusTextEl && statusText && statusTextEl.textContent !== statusText) {
        statusTextEl.textContent = statusText;
    }

    if (progressCard) {
        if (isRunning) {
            progressCard.classList.add('running');
            progressCard.classList.remove('complete', 'stopped');
        }
    }
}

// Function to mark evolution as complete
function markEvolutionComplete(isStopped = false, isResumable = false, checkpointId = null) {
    const spinner = document.getElementById('evolution-spinner');
    const statusTextEl = document.getElementById('evolution-status-text');
    const progressCard = document.getElementById('progress-container');
    const activityIndicator = document.getElementById('progress-bar-activity');
    const forceStopButton = document.getElementById('forceStopButton');

    if (spinner) {
        spinner.classList.remove('spinning');
        spinner.classList.add(isStopped ? 'stopped' : 'complete');
    }

    if (statusTextEl) {
        if (isStopped && isResumable) {
            statusTextEl.textContent = 'Paused';
        } else if (isStopped) {
            statusTextEl.textContent = 'Stopped';
        } else {
            statusTextEl.textContent = 'Complete';
        }
    }

    if (progressCard) {
        progressCard.classList.remove('running');
        progressCard.classList.add(isStopped ? 'stopped' : 'complete');
    }

    if (activityIndicator) {
        activityIndicator.classList.remove('waiting');
    }

    // Hide force stop button
    if (forceStopButton) {
        forceStopButton.style.display = 'none';
    }

    stopElapsedTimeUpdater();

    // Add final activity log entry
    const elapsed = evolutionStartTime ? formatDuration(Date.now() - evolutionStartTime) : 'unknown';
    if (isStopped && isResumable) {
        addActivityLogItem(`⏸️ Paused after ${elapsed} - checkpoint saved`, 'info');
    } else if (isStopped) {
        addActivityLogItem(`⏹️ Stopped after ${elapsed}`, 'warning');
    } else {
        addActivityLogItem(`✅ Completed in ${elapsed}!`, 'success');
    }

    // Store checkpoint info for history panel
    if (checkpointId) {
        localStorage.setItem('lastCheckpointId', checkpointId);
        // Refresh history to show new checkpoint
        if (typeof loadHistory === 'function') {
            loadHistory();
        }
    }
}



// Function to show the context modal
function showContextModal(ideaIndex) {
    console.log(`showContextModal called with ideaIndex: ${ideaIndex}, contexts:`, contexts);

    // Create modal if it doesn't exist
    let contextModal = document.getElementById('contextModal');

    if (!contextModal) {
        // Create the modal element
        contextModal = document.createElement('div');
        contextModal.className = 'modal fade';
        contextModal.id = 'contextModal';
        contextModal.tabIndex = '-1';
        contextModal.setAttribute('aria-labelledby', 'contextModalLabel');
        contextModal.setAttribute('aria-hidden', 'true');

        // Set up the modal HTML structure
        contextModal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="contextModalLabel">Initial Context</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div id="contextModalContent"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;

        // Add the modal to the document body
        document.body.appendChild(contextModal);
    }

    // Get the current context
    const contextModalContent = document.getElementById('contextModalContent');

    if (contexts.length > 0) {
        // Use the provided ideaIndex if available, otherwise use currentContextIndex
        const contextIndex = (ideaIndex !== undefined && ideaIndex < contexts.length) ?
            ideaIndex : currentContextIndex;

        // Use direction prompts if available, otherwise use raw contexts (legacy)
        const displayContent = specificPrompts.length > 0 && specificPrompts[contextIndex]
            ? specificPrompts[contextIndex]
            : contexts[contextIndex];

        const displayTitle = specificPrompts.length > 0 ? "Direction Prompt" : "Context Pool";

        console.log(`Using contextIndex: ${contextIndex}, ${displayTitle.toLowerCase()}: "${displayContent}"`);

        // Update the modal title to show which context we're viewing
        const modalTitle = document.getElementById('contextModalLabel');
        modalTitle.textContent = `${displayTitle} for Idea ${contextIndex + 1}`;

        // Format the content into separate items
        const contextItems = displayContent
            .split('\n')
            .filter(item => item.trim())
            .map(item => `<div class="context-item">${item.trim()}</div>`)
            .join('');

        contextModalContent.innerHTML = `
            <div class="context-content">
                <div class="alert alert-info mb-3">
                    <strong>${displayTitle}:</strong> ${specificPrompts.length > 0 ?
                'This direction prompt was generated from the context pool to create this idea.' :
                'This is the raw context pool used to inspire this idea.'}
                </div>
                ${contextItems}
            </div>
        `;
    } else {
        console.warn('No contexts available');
        contextModalContent.innerHTML = '<p class="text-muted">No context available</p>';
    }

    // Show the modal
    let modal;
    if (window.bootstrap) {
        modal = new bootstrap.Modal(contextModal);

        // Add event listener to clean up when modal is hidden
        // Use a named function so we can check if it already exists
        const cleanupFunction = function (event) {
            // Remove any lingering backdrop elements
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => {
                backdrop.remove();
            });

            // Reset body classes that might have been added by Bootstrap
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        };

        // We don't want to add multiple cleanup listeners
        // First remove any existing cleanup listener
        contextModal.removeEventListener('hidden.bs.modal', cleanupFunction);

        // Then add our cleanup listener
        contextModal.addEventListener('hidden.bs.modal', cleanupFunction);

        modal.show();
    } else {
        console.error("Bootstrap is not available. Make sure it's properly loaded.");
        // Fallback for testing - just make the modal visible
        contextModal.style.display = 'block';
    }
}

// Function to show the lineage modal (or Oracle analysis for Oracle-generated ideas)
function showLineageModal(idea, generationIndex) {
    console.log("Showing lineage for idea:", idea);

    // Create modal if it doesn't exist
    let lineageModal = document.getElementById('lineageModal');

    if (!lineageModal) {
        // Create the modal element
        lineageModal = document.createElement('div');
        lineageModal.className = 'modal fade';
        lineageModal.id = 'lineageModal';
        lineageModal.tabIndex = '-1';
        lineageModal.setAttribute('aria-labelledby', 'lineageModalLabel');
        lineageModal.setAttribute('aria-hidden', 'true');

        // Set up the modal HTML structure
        lineageModal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="lineageModalLabel">Idea Lineage</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div id="lineageModalContent"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;

        // Add the modal to the document body
        document.body.appendChild(lineageModal);
    }

    // Get the lineage modal content element
    const lineageModalContent = document.getElementById('lineageModalContent');

    // Check if this is an elite (most creative) idea first
    if (idea.elite_selected || idea.elite_selected_source) {
        // Determine the type of elite display
        if (idea.elite_selected_source) {
            // This is a SOURCE idea that was selected to be elite
            document.getElementById('lineageModalLabel').textContent = `Selected as Most Creative: ${idea.title || 'Untitled'}`;

            const targetGeneration = idea.elite_target_generation;
            const originHtml = `
                <div class="elite-origin-section">
                    <div class="alert alert-success mb-3">
                        <h6><i class="fas fa-star"></i> Selected as Most Creative</h6>
                        <p class="mb-0">This idea was identified as the most creative and original in Generation ${generationIndex + 1}, with the largest distance from the population centroid. It was selected to pass directly to Generation ${targetGeneration + 1} where it will be refined and formatted.</p>
                    </div>
                    <div class="alert alert-info">
                        <p class="mb-0"><strong>Note:</strong> Look for this idea in Generation ${targetGeneration + 1} to see its refined version.</p>
                    </div>
                </div>
            `;

            lineageModalContent.innerHTML = originHtml;
        } else {
            // This is a REFINED elite idea in the next generation
            document.getElementById('lineageModalLabel').textContent = `Creative Origin: ${idea.title || 'Untitled'}`;

            // Find the source idea from the previous generation
            const sourceGenerationIndex = idea.elite_source_generation;
            const sourceIdeaId = idea.elite_source_id;

            let sourceIdea = null;
            if (sourceGenerationIndex !== undefined && sourceIdeaId && generations[sourceGenerationIndex]) {
                sourceIdea = generations[sourceGenerationIndex].find(sourceCandidate => sourceCandidate.id === sourceIdeaId);
            }

            let originHtml;
            if (sourceIdea) {
                const sourcePreview = createCardPreview(sourceIdea.content, 200);
                originHtml = `
                <div class="elite-origin-section">
                    <div class="alert alert-success mb-3">
                        <h6><i class="fas fa-star"></i> Most Creative Idea Selected</h6>
                        <p class="mb-0">This idea was identified as the most creative and original in Generation ${sourceGenerationIndex + 1}, with the largest distance from the population centroid. It was preserved and refined for the next generation.</p>
                    </div>
                    <div class="elite-origin-content">
                        <h6>Original Source (Generation ${sourceGenerationIndex + 1}):</h6>
                        <div class="card elite-source-card">
                            <div class="card-body">
                                <h6 class="card-title">${sourceIdea.title || 'Untitled'}</h6>
                                <div class="card-preview">
                                    <p>${sourcePreview}</p>
                                </div>
                                <button class="btn btn-sm btn-primary view-source-idea">
                                    View Full Original Idea
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            } else {
                originHtml = `
                <div class="elite-origin-section">
                    <div class="alert alert-success mb-3">
                        <h6><i class="fas fa-star"></i> Most Creative Idea Selected</h6>
                        <p class="mb-0">This idea was identified as the most creative and original, with the largest distance from the population centroid. It was preserved and refined for the next generation.</p>
                    </div>
                    <div class="alert alert-warning">
                        <p class="mb-0">Could not find the original source idea from the previous generation.</p>
                    </div>
                </div>
            `;
            }

            lineageModalContent.innerHTML = originHtml;

            // Add event listener for the view source button if it exists
            const viewSourceBtn = lineageModalContent.querySelector('.view-source-idea');
            if (viewSourceBtn && sourceIdea) {
                viewSourceBtn.addEventListener('click', () => {
                    // Store current idea for reopening
                    const currentIdea = idea;
                    const currentGenIndex = generationIndex;

                    // Close the lineage modal
                    if (window.bootstrap) {
                        const lineageModalInstance = bootstrap.Modal.getInstance(lineageModal);
                        if (lineageModalInstance) {
                            lineageModalInstance.hide();
                        }
                    } else {
                        lineageModal.style.display = 'none';
                    }

                    // Add event listener to reopen lineage modal when idea modal closes
                    setTimeout(() => {
                        const ideaModalElement = document.getElementById('ideaModal');
                        const reopenLineage = function () {
                            ideaModalElement.removeEventListener('hidden.bs.modal', reopenLineage);
                            setTimeout(() => {
                                showLineageModal(currentIdea, currentGenIndex);
                            }, 150);
                        };
                        ideaModalElement.addEventListener('hidden.bs.modal', reopenLineage);

                        // Show the source idea modal
                        showIdeaModal(sourceIdea);
                    }, 150);
                });
            }

            // Show the modal
            let modal;
            if (window.bootstrap) {
                modal = new bootstrap.Modal(lineageModal);
            } else {
                lineageModal.style.display = 'block';
            }
            modal.show();

            return; // Exit early for elite ideas
        }
    } else if (idea.oracle_generated && idea.oracle_analysis) {
        // Show Oracle analysis instead of lineage
        document.getElementById('lineageModalLabel').textContent = `Oracle Analysis: ${idea.title || 'Untitled'}`;

        const analysisHtml = `
            <div class="oracle-analysis-section">
                <div class="alert alert-info mb-3">
                    <h6><i class="fas fa-eye"></i> Oracle Diversity Analysis</h6>
                    <p class="mb-0">This idea was generated by the Oracle agent to increase population diversity by identifying and addressing overused patterns.</p>
                </div>
                <div class="oracle-analysis-content">
                    <h6>Analysis & Recommendations:</h6>
                    <div class="analysis-text" style="white-space: pre-wrap; line-height: 1.6; background-color: #f8f9fa; padding: 1rem; border-radius: 0.375rem; border-left: 4px solid #0d6efd;">
                        ${idea.oracle_analysis}
                    </div>
                </div>
            </div>
        `;

        lineageModalContent.innerHTML = analysisHtml;

        // Show the modal
        let modal;
        if (window.bootstrap) {
            modal = new bootstrap.Modal(lineageModal);
        } else {
            // Fallback if bootstrap is not available
            lineageModal.style.display = 'block';
        }
        modal.show();

        return; // Exit early for Oracle ideas
    }

    // For regular ideas, set the modal title to include the idea title
    document.getElementById('lineageModalLabel').textContent = `Lineage: ${idea.title || 'Untitled'}`;

    // Function to recursively find ancestors
    function findAncestors(currentIdea, currentGenIndex, processedIds = new Set()) {
        // Base case: if we're at generation 0 or the idea has no parents, return empty array
        if (currentGenIndex <= 0 || !currentIdea.parent_ids || currentIdea.parent_ids.length === 0) {
            return [];
        }

        const parentGeneration = generations[currentGenIndex - 1];
        const ancestors = [];

        // Find direct parents
        for (const parentId of currentIdea.parent_ids) {
            // Skip if we've already processed this parent
            if (processedIds.has(parentId)) {
                continue;
            }

            const parentIdea = parentGeneration.find(p => p.id === parentId);
            if (parentIdea) {
                // Add this parent with its generation info
                ancestors.push({
                    idea: parentIdea,
                    generation: currentGenIndex - 1
                });

                // Mark this parent as processed to avoid duplicates in recursive calls
                processedIds.add(parentId);

                // Recursively find this parent's ancestors, passing the updated processedIds
                const parentAncestors = findAncestors(parentIdea, currentGenIndex - 1, processedIds);
                ancestors.push(...parentAncestors);
            }
        }

        return ancestors;
    }

    // Get all ancestors organized by generation
    function getAncestorsByGeneration(currentIdea, currentGenIndex) {
        const allAncestors = findAncestors(currentIdea, currentGenIndex);
        const ancestorsByGen = {};

        // Track all ancestor IDs to avoid duplicates
        const processedIds = new Set();

        // Add direct parent IDs to processed set to avoid duplication
        if (currentIdea.parent_ids) {
            currentIdea.parent_ids.forEach(id => processedIds.add(id));
        }

        // Group ancestors by generation, avoiding duplicates
        for (const ancestor of allAncestors) {
            // Skip if we've already processed this ancestor or if it's a direct parent
            if (processedIds.has(ancestor.idea.id)) {
                continue;
            }

            // Mark this ancestor as processed
            processedIds.add(ancestor.idea.id);

            // Add to the appropriate generation group
            if (!ancestorsByGen[ancestor.generation]) {
                ancestorsByGen[ancestor.generation] = [];
            }
            ancestorsByGen[ancestor.generation].push(ancestor.idea);
        }

        return ancestorsByGen;
    }

    // Check if the idea has parent IDs
    if (idea.parent_ids && idea.parent_ids.length > 0) {
        // Get all ancestors organized by generation
        const ancestorsByGeneration = getAncestorsByGeneration(idea, generationIndex);

        // Find direct parents from the previous generation
        const parentGeneration = generations[generationIndex - 1];
        const directParents = [];

        // Find each parent idea by ID
        for (const parentId of idea.parent_ids) {
            const parentIdea = parentGeneration.find(p => p.id === parentId);
            if (parentIdea) {
                directParents.push(parentIdea);
            }
        }

        if (directParents.length > 0 || Object.keys(ancestorsByGeneration).length > 0) {
            let lineageHtml = '';

            // Add direct parents section
            if (directParents.length > 0) {
                lineageHtml += `
                    <div class="lineage-section mb-4">
                        <h6 class="lineage-generation-title">Direct Parents (Generation ${generationIndex})</h6>
                        <div class="lineage-parent-cards">
                `;

                // Add a card for each direct parent
                directParents.forEach((parent, idx) => {
                    const parentPreview = createCardPreview(parent.content, 100);
                    lineageHtml += createAncestorCard(parent, idx, 'direct-parent');
                });

                lineageHtml += `
                        </div>
                    </div>
                `;
            }

            // Add sections for each generation of ancestors
            const generationNumbers = Object.keys(ancestorsByGeneration).sort((a, b) => b - a); // Sort in descending order

            for (const genNumber of generationNumbers) {
                const ancestors = ancestorsByGeneration[genNumber];

                lineageHtml += `
                    <div class="lineage-section mb-4">
                        <h6 class="lineage-generation-title">Earlier Ancestors (Generation ${parseInt(genNumber) + 1})</h6>
                        <div class="lineage-parent-cards">
                `;

                // Add a card for each ancestor in this generation
                ancestors.forEach((ancestor, idx) => {
                    lineageHtml += createAncestorCard(ancestor, idx, `ancestor-gen-${genNumber}`);
                });

                lineageHtml += `
                        </div>
                    </div>
                `;
            }

            lineageModalContent.innerHTML = lineageHtml;

            // Add event listeners to all "View Full Idea" buttons
            const viewButtons = lineageModalContent.querySelectorAll('.view-ancestor-idea');
            viewButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const genIndex = button.getAttribute('data-gen-index');
                    const ideaIndex = button.getAttribute('data-idea-index');
                    const ancestorType = button.getAttribute('data-ancestor-type');

                    // Store the current idea and generation index for reopening the lineage modal later
                    const currentIdea = idea;
                    const currentGenIndex = generationIndex;

                    // Close the lineage modal before showing the idea modal
                    if (window.bootstrap) {
                        const lineageModalInstance = bootstrap.Modal.getInstance(lineageModal);
                        if (lineageModalInstance) {
                            lineageModalInstance.hide();
                        }
                    } else {
                        // Fallback if bootstrap is not available
                        lineageModal.style.display = 'none';
                    }

                    // Add a small delay to ensure the modal is fully closed before opening the new one
                    setTimeout(() => {
                        // Get the idea modal element
                        const ideaModalElement = document.getElementById('ideaModal');

                        // Add one-time event listener to reopen lineage modal when idea modal is closed
                        const reopenLineage = function () {
                            // Remove this event listener to prevent multiple reopenings
                            ideaModalElement.removeEventListener('hidden.bs.modal', reopenLineage);

                            // Reopen the lineage modal with a small delay
                            setTimeout(() => {
                                showLineageModal(currentIdea, currentGenIndex);
                            }, 150);
                        };

                        // Add the event listener
                        ideaModalElement.addEventListener('hidden.bs.modal', reopenLineage);

                        // Determine which ancestor to show
                        let ancestorIdea;
                        if (ancestorType === 'direct-parent') {
                            ancestorIdea = directParents[ideaIndex];
                        } else {
                            const genNumber = ancestorType.split('-').pop();
                            ancestorIdea = ancestorsByGeneration[genNumber][ideaIndex];
                        }

                        // Show the idea modal
                        showIdeaModal(ancestorIdea);
                    }, 150);
                });
            });
        } else {
            lineageModalContent.innerHTML = `<p>Could not find parent ideas for this idea.</p>`;
        }
    } else {
        lineageModalContent.innerHTML = `<p>This idea has no recorded parent ideas.</p>`;
    }

    // Show the modal
    let modal;
    if (window.bootstrap) {
        modal = new bootstrap.Modal(lineageModal);

        // Add event listener to clean up when modal is hidden
        // Use a named function so we can check if it already exists
        const cleanupFunction = function (event) {
            // Remove any lingering backdrop elements
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => {
                backdrop.remove();
            });

            // Reset body classes that might have been added by Bootstrap
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        };

        // We don't want to add multiple cleanup listeners
        // First remove any existing cleanup listener
        lineageModal.removeEventListener('hidden.bs.modal', cleanupFunction);

        // Then add our cleanup listener
        lineageModal.addEventListener('hidden.bs.modal', cleanupFunction);

        modal.show();
    } else {
        console.error("Bootstrap is not available. Make sure it's properly loaded.");
        // Fallback for testing - just make the modal visible
        lineageModal.style.display = 'block';
    }
}

// Helper function to create an ancestor card
function createAncestorCard(ancestor, index, ancestorType) {
    const preview = createCardPreview(ancestor.content, 100);
    return `
        <div class="card lineage-parent-card mb-3">
            <div class="card-body">
                <h6 class="card-title">${ancestor.title || 'Untitled'}</h6>
                <div class="card-preview">
                    <p>${preview}</p>
                </div>
                <button class="btn btn-sm btn-primary view-ancestor-idea"
                    data-idea-index="${index}"
                    data-ancestor-type="${ancestorType}">
                    <i class="fas fa-expand"></i> View Full Idea
                </button>
            </div>
        </div>
    `;
}

// Function to display token counts
function displayTokenCounts(tokenCounts) {
    console.log("Displaying token counts:", tokenCounts);
    latestDiagnosticsTokenCounts = tokenCounts;
    const nextSignature = safeSignature({
        total: tokenCounts.total,
        total_input: tokenCounts.total_input,
        total_output: tokenCounts.total_output,
        cost: tokenCounts.cost,
    });
    if (nextSignature === lastTokenCountsSignature) {
        return;
    }
    lastTokenCountsSignature = nextSignature;

    // Create or get the token counts container
    let tokenCountsContainer = document.getElementById('token-counts-container');

    if (!tokenCountsContainer) {
        // Create the container if it doesn't exist
        tokenCountsContainer = document.createElement('div');
        tokenCountsContainer.id = 'token-counts-container';
        tokenCountsContainer.className = 'card mt-4';

        // Insert it after the progress container
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer && progressContainer.parentNode) {
            progressContainer.parentNode.insertBefore(tokenCountsContainer, progressContainer.nextSibling);
        } else {
            // Fallback: insert before generations container
            const generationsContainer = document.getElementById('generations-container');
            if (generationsContainer && generationsContainer.parentNode) {
                generationsContainer.parentNode.insertBefore(tokenCountsContainer, generationsContainer);
            }
        }

        tokenCountsContainer.innerHTML = `
            <div class="card-body d-flex justify-content-between align-items-center p-3">
                <div class="d-flex">
                    <div>
                        <h6 class="mb-0">Cost: <strong id="token-cost-value">$0.0000</strong></h6>
                        <small class="text-muted" id="token-total-value">0 tokens</small>
                        <small class="text-muted d-block" id="token-diagnostics-value" style="display:none;"></small>
                    </div>
                    <div id="token-estimated-total-wrap" class="ms-3 border-start ps-3" style="display:none;">
                        <h6 class="mb-0">Est. Total: <strong id="token-estimated-value">$0.0000</strong></h6>
                        <small class="text-muted">Projected</small>
                    </div>
                </div>
                <button id="token-details-btn" class="btn btn-sm btn-outline-primary">
                    <i class="fas fa-info-circle"></i> Details
                </button>
            </div>
        `;

        const detailsButton = tokenCountsContainer.querySelector('#token-details-btn');
        if (detailsButton) {
            detailsButton.addEventListener('click', function () {
                try {
                    const latest = JSON.parse(tokenCountsContainer.dataset.tokenCounts || '{}');
                    showTokenDetailsModal(latest);
                } catch (error) {
                    console.error('Failed to parse token counts for details modal:', error);
                }
            });
        }
    }

    // Store the token data for the modal
    tokenCountsContainer.dataset.tokenCounts = JSON.stringify(tokenCounts);

    // Update in place to avoid replacing DOM every poll.
    const costEl = tokenCountsContainer.querySelector('#token-cost-value');
    if (costEl) {
        costEl.textContent = `$${tokenCounts.cost.total_cost.toFixed(4)}`;
    }

    const totalEl = tokenCountsContainer.querySelector('#token-total-value');
    if (totalEl) {
        totalEl.textContent = `${tokenCounts.total.toLocaleString()} tokens`;
    }

    const diagnosticsEl = tokenCountsContainer.querySelector('#token-diagnostics-value');
    if (diagnosticsEl) {
        const diagnosticsSummary = getDiagnosticsSummary(tokenCounts);
        if (diagnosticsSummary && diagnosticsSummary.totalCount > 0) {
            diagnosticsEl.style.display = '';
            diagnosticsEl.textContent = `Diagnostics: ${diagnosticsSummary.totalCount} events`;
        } else {
            diagnosticsEl.style.display = 'none';
            diagnosticsEl.textContent = '';
        }
    }

    const estWrapEl = tokenCountsContainer.querySelector('#token-estimated-total-wrap');
    const estValueEl = tokenCountsContainer.querySelector('#token-estimated-value');
    if (estWrapEl && estValueEl) {
        if (tokenCounts.cost.estimated_total_cost !== undefined) {
            estWrapEl.style.display = '';
            estValueEl.textContent = `$${tokenCounts.cost.estimated_total_cost.toFixed(4)}`;
        } else {
            estWrapEl.style.display = 'none';
        }
    }
}

// Function to show the token details modal
function showTokenDetailsModal(tokenCounts, options = {}) {
    const focusDiagnostics = options.focus === 'diagnostics';
    // Format the token counts
    const totalTokens = tokenCounts.total.toLocaleString();
    const totalInputTokens = tokenCounts.total_input.toLocaleString();
    const totalOutputTokens = tokenCounts.total_output.toLocaleString();

    // Get cost information
    const costInfo = tokenCounts.cost;
    const totalCost = costInfo.total_cost.toFixed(4);
    const inputCost = costInfo.input_cost.toFixed(4);
    const outputCost = costInfo.output_cost.toFixed(4);

    // Prepare alternative model cost estimates
    let estimatesList = '';
    if (tokenCounts.estimates) {
        estimatesList = Object.values(tokenCounts.estimates).map(e => {
            return `<li class="list-group-item d-flex justify-content-between align-items-center">${e.name}<span>$${e.cost.toFixed(4)} <small class="text-muted">estimate</small></span></li>`;
        }).join('');
    }

    const diagnosticsTotals = tokenCounts?.diagnostics?.totals || {};
    const diagnosticsEntries = Object.entries(diagnosticsTotals)
        .filter(([, value]) => Number.isFinite(value) && value > 0)
        .sort((a, b) => b[1] - a[1]);
    const diagnosticsCount = diagnosticsEntries.reduce((sum, [, value]) => sum + value, 0);
    let diagnosticsSectionHtml = `
        <hr>
        <h6 id="diagnostics-section">Diagnostics</h6>
        <p class="small text-muted mb-0">No recoverable parsing/generation issues were recorded.</p>
    `;
    if (diagnosticsEntries.length > 0) {
        const diagnosticsItems = diagnosticsEntries.map(([key, value]) => (
            `<button type="button" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center small diagnostic-item-btn" data-diagnostic-key="${escapeHtml(key)}" data-diagnostic-total="${value}">${escapeHtml(prettifyDiagnosticKey(key))}<span class="badge bg-warning text-dark rounded-pill">${value}</span></button>`
        )).join('');
        diagnosticsSectionHtml = `
            <hr>
            <h6 id="diagnostics-section">Diagnostics</h6>
            <p class="small text-muted mb-2">${diagnosticsCount} recoverable issue(s) captured across agents.</p>
            <ul class="list-group mb-3">
                ${diagnosticsItems}
            </ul>
            <div id="diagnostics-detail-panel" class="small text-muted border rounded p-2 bg-light">
                Click a diagnostic item to view details.
            </div>
        `;
    }

    // Dynamically build component data for all available components
    const componentData = [];
    const colors = [
        'rgba(54, 162, 235, 0.7)',   // Blue
        'rgba(75, 192, 192, 0.7)',   // Teal
        'rgba(255, 99, 132, 0.7)',   // Red
        'rgba(255, 206, 86, 0.7)',   // Yellow
        'rgba(153, 102, 255, 0.7)',  // Purple
        'rgba(255, 159, 64, 0.7)'    // Orange
    ];
    const borderColors = [
        'rgba(54, 162, 235, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)'
    ];

    let colorIndex = 0;

    // Check for each possible component and add to componentData if it exists
    const possibleComponents = ['ideator', 'formatter', 'critic', 'breeder', 'genotype_encoder', 'diversity_embeddings', 'oracle'];

    for (const componentName of possibleComponents) {
        if (tokenCounts[componentName]) {
            const component = tokenCounts[componentName];
            componentData.push({
                name: componentName.charAt(0).toUpperCase() + componentName.slice(1).replace('_', ' '),
                displayName: componentName === 'genotype_encoder' ? 'Genotype Encoder' :
                    componentName === 'diversity_embeddings' ? 'Diversity Embeddings' :
                    componentName.charAt(0).toUpperCase() + componentName.slice(1),
                total: component.total.toLocaleString(),
                input: component.input.toLocaleString(),
                output: component.output.toLocaleString(),
                model: component.model,
                cost: component.cost.toFixed(4),
                totalValue: component.total,
                costValue: component.cost,
                backgroundColor: colors[colorIndex % colors.length],
                borderColor: borderColors[colorIndex % borderColors.length]
            });
            colorIndex++;
        }
    }

    // Build component breakdown HTML
    let componentBreakdownHtml = '';
    componentData.forEach(comp => {
        componentBreakdownHtml += `
            <li class="list-group-item">
                <div class="d-flex justify-content-between align-items-center">
                    <span>${comp.displayName} <span class="badge bg-secondary">${comp.model}</span></span>
                    <span class="badge bg-primary rounded-pill">${comp.total}</span>
                </div>
                <div class="small text-muted mt-1">
                    <span>Input: ${comp.input}</span> |
                    <span>Output: ${comp.output}</span> |
                    <span>Cost: $${comp.cost}</span>
                </div>
            </li>
        `;
    });

    // Create modal container if it doesn't exist
    let modalContainer = document.getElementById('token-details-modal');
    if (!modalContainer) {
        modalContainer = document.createElement('div');
        modalContainer.id = 'token-details-modal';
        modalContainer.className = 'modal fade';
        modalContainer.tabIndex = '-1';
        modalContainer.setAttribute('aria-labelledby', 'tokenDetailsModalLabel');
        modalContainer.setAttribute('aria-hidden', 'true');
        document.body.appendChild(modalContainer);
    }

    // Set modal content
    modalContainer.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="tokenDetailsModalLabel">Token Usage Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6>Total Tokens: <span class="text-primary">${totalTokens}</span></h6>
                                    <div class="d-flex justify-content-between">
                                        <span>Input: <span class="text-info">${totalInputTokens}</span></span>
                                        <span>Output: <span class="text-success">${totalOutputTokens}</span></span>
                                    </div>
                                    <hr>
                                    <h6>Cost Breakdown:</h6>
                                    <p class="mb-1">Total cost: <strong>$${totalCost}</strong></p>
                                    <div class="small text-muted">
                                        <div>Input: $${inputCost}</div>
                                        <div>Output: $${outputCost}</div>
                                    </div>
                                    <hr>
                                    <h6>Other Model Estimates:</h6>
                                    <ul class="list-group mb-3 small">
                                        ${estimatesList}
                                    </ul>
                                    ${diagnosticsSectionHtml}
                                </div>
                            </div>

                            <h6>Component Breakdown:</h6>
                            <ul class="list-group">
                                ${componentBreakdownHtml}
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <canvas id="tokenPieChart" width="400" height="200"></canvas>
                            </div>
                            <div class="mb-3">
                                <canvas id="tokenBarChart" width="400" height="200"></canvas>
                            </div>
                            <div>
                                <canvas id="tokenCostChart" width="400" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    `;

    // Initialize the modal
    const modal = new bootstrap.Modal(modalContainer);
    modal.show();

    // Create charts after the modal is shown
    modalContainer.addEventListener('shown.bs.modal', function () {
        const diagnosticsPanel = modalContainer.querySelector('#diagnostics-detail-panel');
        const diagnosticsButtons = Array.from(
            modalContainer.querySelectorAll('.diagnostic-item-btn')
        );
        if (diagnosticsPanel && diagnosticsButtons.length > 0) {
            diagnosticsButtons.forEach((button) => {
                button.addEventListener('click', () => {
                    if (button.classList.contains('active')) {
                        button.classList.remove('active');
                        diagnosticsPanel.innerHTML = 'Click a diagnostic item to view details.';
                        return;
                    }

                    diagnosticsButtons.forEach((candidate) => candidate.classList.remove('active'));
                    button.classList.add('active');

                    const key = button.dataset.diagnosticKey || '';
                    const total = Number(button.dataset.diagnosticTotal || 0);
                    const description = getDiagnosticDescription(key);
                    const perAgent = Object.entries(tokenCounts?.diagnostics?.agents || {})
                        .map(([agent, stats]) => {
                            const count = Number((stats && stats[key]) || 0);
                            return { agent, count };
                        })
                        .filter(({ count }) => count > 0)
                        .sort((a, b) => b.count - a.count);

                    const rawEvents = Object.entries(tokenCounts?.diagnostics?.events || {})
                        .flatMap(([agentName, events]) => {
                            if (!Array.isArray(events)) return [];
                            return events.map((event) => ({
                                ...event,
                                agent: event?.agent || agentName,
                            }));
                        })
                        .filter((event) => event && event.key === key)
                        .sort((a, b) => {
                            const at = new Date(a.timestamp || 0).getTime();
                            const bt = new Date(b.timestamp || 0).getTime();
                            return bt - at;
                        });

                    let agentBreakdown = '<li class="text-muted">No per-agent breakdown available.</li>';
                    if (perAgent.length > 0) {
                        agentBreakdown = perAgent
                            .map(({ agent, count }) => `<li>${escapeHtml(agent)}: <strong>${count}</strong></li>`)
                            .join('');
                    }

                    let rawErrorHtml = '<li class="text-muted">No raw diagnostic events captured for this key.</li>';
                    if (rawEvents.length > 0) {
                        rawErrorHtml = rawEvents
                            .slice(0, 15)
                            .map((event) => {
                                const metadata = event.metadata && typeof event.metadata === 'object'
                                    ? ` | meta: ${escapeHtml(JSON.stringify(event.metadata))}`
                                    : '';
                                const detail = event.detail ? escapeHtml(event.detail) : '(no detail)';
                                const timestamp = formatDiagnosticTimestamp(event.timestamp);
                                return `<li><span class="text-muted">${escapeHtml(timestamp)} | ${escapeHtml(event.agent || 'unknown')}</span><br><code class="small">${detail}${metadata}</code></li>`;
                            })
                            .join('');
                    }

                    diagnosticsPanel.innerHTML = `
                        <div class="fw-semibold mb-1">${escapeHtml(prettifyDiagnosticKey(key))} (${total})</div>
                        <div class="mb-2">${escapeHtml(description)}</div>
                        <div class="fw-semibold">By agent</div>
                        <ul class="mb-2 ps-3">${agentBreakdown}</ul>
                        <div class="fw-semibold">Recent raw events</div>
                        <ul class="mb-0 ps-3">${rawErrorHtml}</ul>
                    `;
                });
            });
        }

        if (focusDiagnostics) {
            const diagnosticsEl = modalContainer.querySelector('#diagnostics-section');
            if (diagnosticsEl) {
                diagnosticsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        // Create a pie chart for token distribution by component
        const pieCtx = document.getElementById('tokenPieChart').getContext('2d');
        new Chart(pieCtx, {
            type: 'pie',
            data: {
                labels: componentData.map(comp => comp.displayName),
                datasets: [{
                    data: componentData.map(comp => comp.totalValue),
                    backgroundColor: componentData.map(comp => comp.backgroundColor),
                    borderColor: componentData.map(comp => comp.borderColor),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    title: {
                        display: true,
                        text: 'Token Distribution by Component'
                    }
                }
            }
        });

        // Create a bar chart for costs by component
        const barCtx = document.getElementById('tokenBarChart').getContext('2d');
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: componentData.map(comp => comp.displayName),
                datasets: [{
                    label: 'Cost (USD)',
                    data: componentData.map(comp => comp.costValue),
                    backgroundColor: componentData.map(comp => comp.backgroundColor),
                    borderColor: componentData.map(comp => comp.borderColor),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cost Breakdown by Component'
                    },
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function (value) {
                                return '$' + value.toFixed(4);
                            }
                        }
                    }
                }
            }
        });

        // Create a third chart for input vs output cost breakdown
        const costCtx = document.getElementById('tokenCostChart').getContext('2d');
        new Chart(costCtx, {
            type: 'bar',
            data: {
                labels: ['Input Cost', 'Output Cost', 'Total Cost'],
                datasets: [{
                    label: 'Cost (USD)',
                    data: [
                        parseFloat(inputCost),
                        parseFloat(outputCost),
                        parseFloat(totalCost)
                    ],
                    backgroundColor: ['#36a2eb', '#4bc0c0', '#ff9f40'],
                    borderColor: ['#36a2eb', '#4bc0c0', '#ff9f40'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Input vs Output Cost'
                    },
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function (value) {
                                return '$' + value.toFixed(4);
                            }
                        }
                    }
                }
            }
        });
    }, { once: true });
}

/* ==========================================
   DIVERSITY PLOTTING FUNCTIONALITY
   ========================================== */

// Global variables for diversity plotting
let diversityChart = null;
let diversityData = {
    labels: [],
    overview: {},
    islands: {},
    latest: null
};
let diversityChartMode = 'overview';

function startPollingActiveEvolution(progressData = {}) {
    if (progressData.evolution_id) {
        currentEvolutionId = progressData.evolution_id;
    }
    if (progressData.evolution_name) {
        currentEvolutionName = progressData.evolution_name;
    }
    if (!document.getElementById('progress-container')) {
        createProgressBar(currentEvolutionName);
    }
    if (!progressData.diversity_history || progressData.diversity_history.length === 0) {
        showDiversityPlotLoading();
    }
    isEvolutionRunning = true;
    pollProgress();
}

/**
 * Get chart scales configuration based on current axis mode
 */
function getDiversityChartScales() {
    return {
        x: {
            title: {
                display: true,
                text: 'Generation',
                color: '#64748B',
                font: {
                    size: 14,
                    weight: 'bold'
                }
            },
            grid: {
                color: 'rgba(139, 124, 246, 0.1)',
                lineWidth: 1
            },
            ticks: {
                color: '#64748B',
                font: {
                    size: 12
                },
                callback: function (value, index, values) {
                    const generation = this.getLabelForValue(value);
                    return generation === '0' ? 'Initial' : `Gen ${generation}`;
                }
            }
        },
        y: {
            title: {
                display: true,
                text: diversityChartMode === 'islands'
                    ? 'Diversity / Separation'
                    : 'Diversity Score',
                color: '#64748B',
                font: {
                    size: 14,
                    weight: 'bold'
                }
            },
            grid: {
                color: 'rgba(139, 124, 246, 0.1)',
                lineWidth: 1
            },
            ticks: {
                color: '#64748B',
                font: {
                    size: 12
                }
            },
            beginAtZero: false,
            position: 'left'
        }
    };
}

/**
 * Switch between population and island diversity views.
 */
function setDiversityChartMode(mode) {
    const nextMode = mode === 'islands' ? 'islands' : 'overview';
    const button = document.querySelector(`[data-diversity-mode="${nextMode}"]`);
    if (button?.disabled) return;
    diversityChartMode = nextMode;
    document.querySelectorAll('[data-diversity-mode]').forEach((button) => {
        button.classList.toggle('active', button.dataset.diversityMode === nextMode);
    });

    if (!diversityChart) {
        return;
    }

    diversityChart.data.datasets = getDiversityChartDatasets(diversityData, nextMode);
    diversityChart.options.scales = getDiversityChartScales();
    diversityChart.update('none');
}

/**
 * Initialize the diversity chart
 */
function initializeDiversityChart() {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js is not loaded!');
        return;
    }

    const canvas = document.getElementById('diversity-chart');
    if (!canvas) {
        console.error('❌ Diversity chart canvas not found!');
        return;
    }

    try {
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (diversityChart) {
            diversityChart.destroy();
        }

        // Chart configuration with beautiful styling
        const config = {
            type: 'line',
            data: {
                labels: [],
                datasets: getDiversityChartDatasets(diversityData, diversityChartMode)
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                devicePixelRatio: window.devicePixelRatio || 1,
                layout: {
                    padding: {
                        top: 20,
                        right: 20,
                        bottom: 20,
                        left: 20
                    }
                },
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                size: 12
                            },
                            color: '#64748B'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#1E293B',
                        bodyColor: '#64748B',
                        borderColor: 'rgba(139, 124, 246, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 12,
                        displayColors: true,
                        titleFont: {
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 13
                        },
                        padding: 12,
                        callbacks: {
                            title: function (context) {
                                const generation = context[0].label;
                                return generation === '0' ? 'Initial Population' : `Generation ${generation}`;
                            },
                            label: function (context) {
                                const value = Number(context.parsed.y);
                                return `${context.dataset.label}: ${Number.isFinite(value) ? value.toFixed(4) : '-'}`;
                            }
                        }
                    }
                },
                scales: getDiversityChartScales(),
                animation: {
                    duration: 1000,
                    easing: 'easeInOutCubic'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                elements: {
                    line: {
                        borderCapStyle: 'round',
                        borderJoinStyle: 'round'
                    }
                }
            }
        };

        // Create the chart
        diversityChart = new Chart(ctx, config);
        lastDiversityRenderSignature = '';
        diversityPlotSized = false;

        // Hide loading state
        const loadingElement = document.querySelector('.diversity-loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

    } catch (error) {
        console.error('❌ Error initializing diversity chart:', error);
    }
}

/**
 * Process diversity history data for charting
 * @param {Array} diversityHistory - Array of diversity calculation results
 * @returns {Object} Processed data for charting
 */
function toFiniteDiversityNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function firstFiniteDiversityValue(...values) {
    for (const value of values) {
        const numeric = toFiniteDiversityNumber(value);
        if (numeric !== null) return numeric;
    }
    return null;
}

function getSnapshotGeneration(diversityData, index) {
    const generation = firstFiniteDiversityValue(
        diversityData?.analytics?.current_generation,
        diversityData?.current_generation,
        diversityData?.generation
    );
    return Number.isInteger(generation) ? generation : index;
}

function getCurrentGenerationDiversity(diversityData, generation) {
    const explicit = firstFiniteDiversityValue(
        diversityData?.analytics?.current_generation_diversity,
        diversityData?.current_generation_diversity
    );
    if (explicit !== null) return explicit;

    const generationRows = Array.isArray(diversityData?.generation_diversities)
        ? diversityData.generation_diversities
        : [];
    const row = generationRows.find(genData => Number(genData?.generation) === generation);
    return firstFiniteDiversityValue(row?.diversity_score, row?.score, diversityData?.diversity_score, 0);
}

function getSnapshotIslandRows(diversityData) {
    const analyticsRows = diversityData?.analytics?.islands;
    if (Array.isArray(analyticsRows) && analyticsRows.length > 0) {
        return analyticsRows;
    }
    if (Array.isArray(diversityData?.current_island_diversities)) {
        return diversityData.current_island_diversities;
    }
    if (Array.isArray(diversityData?.island_diversities)) {
        return diversityData.island_diversities;
    }
    return [];
}

function processDiversityData(diversityHistory) {
    if (!diversityHistory || diversityHistory.length === 0) {
        return {
            labels: [],
            overview: { current: [], cumulative: [], drift: [] },
            islands: { meanWithin: [], interIsland: [], separationRatio: [], perIsland: new Map() },
            latest: null,
            hasIslandData: false
        };
    }

    const labels = [];
    const overview = {
        current: [],
        cumulative: [],
        drift: []
    };
    const islands = {
        meanWithin: [],
        interIsland: [],
        separationRatio: [],
        perIsland: new Map()
    };
    let latest = null;
    let hasIslandData = false;

    diversityHistory.forEach((diversityData, index) => {
        if (!diversityData) {
            return;
        }

        if (diversityData.enabled === false || diversityData.error) {
            return;
        }

        const generation = getSnapshotGeneration(diversityData, index);
        const label = String(generation);
        const current = getCurrentGenerationDiversity(diversityData, generation);
        const cumulative = firstFiniteDiversityValue(
            diversityData?.analytics?.cumulative_diversity,
            diversityData?.diversity_score,
            diversityData?.overall_diversity,
            diversityData?.score
        );
        const drift = firstFiniteDiversityValue(
            diversityData?.analytics?.inter_generation_diversity,
            diversityData?.inter_generation_diversity
        );
        const meanWithin = firstFiniteDiversityValue(
            diversityData?.analytics?.mean_island_diversity,
            diversityData?.mean_island_diversity
        );
        const interIsland = firstFiniteDiversityValue(
            diversityData?.analytics?.inter_island_diversity,
            diversityData?.inter_island_diversity
        );
        const separationRatio = firstFiniteDiversityValue(
            diversityData?.analytics?.island_separation_ratio,
            diversityData?.island_separation_ratio
        );

        labels.push(label);
        overview.current.push(current);
        overview.cumulative.push(cumulative);
        overview.drift.push(drift);
        islands.meanWithin.push(meanWithin);
        islands.interIsland.push(interIsland);
        islands.separationRatio.push(separationRatio);

        getSnapshotIslandRows(diversityData).forEach((row) => {
            const islandId = Number(row?.island_id);
            if (!Number.isInteger(islandId)) return;
            const value = firstFiniteDiversityValue(
                row.current_generation_diversity,
                row.diversity_score,
                row.score
            );
            if (!islands.perIsland.has(islandId)) {
                islands.perIsland.set(islandId, []);
            }
            islands.perIsland.get(islandId)[labels.length - 1] = value;
            hasIslandData = true;
        });

        latest = {
            generation,
            current,
            cumulative,
            drift,
            meanWithin,
            interIsland,
            separationRatio,
            islandCount: Number(diversityData?.analytics?.island_count ?? diversityData?.island_count) || null
        };
    });

    islands.perIsland.forEach((series) => {
        for (let i = 0; i < labels.length; i += 1) {
            if (series[i] === undefined) {
                series[i] = null;
            }
        }
    });

    return {
        labels,
        overview,
        islands,
        latest,
        hasIslandData
    };
}

function getDiversityDatasetStyle(color, options = {}) {
    return {
        borderColor: color,
        backgroundColor: color.replace('1)', '0.10)'),
        borderWidth: options.borderWidth ?? 2,
        fill: options.fill ?? false,
        tension: 0.3,
        pointBackgroundColor: color,
        pointBorderColor: '#ffffff',
        pointBorderWidth: 1.5,
        pointRadius: options.pointRadius ?? 4,
        pointHoverRadius: 6,
        borderDash: options.borderDash || [],
        spanGaps: true
    };
}

function getDiversityIslandColor(islandId) {
    const colors = [
        'rgba(37, 99, 235, 1)',
        'rgba(5, 150, 105, 1)',
        'rgba(220, 38, 38, 1)',
        'rgba(124, 58, 237, 1)',
        'rgba(217, 119, 6, 1)'
    ];
    return colors[Math.abs(Number(islandId) || 0) % colors.length];
}

function getDiversityChartDatasets(processedData, mode = diversityChartMode) {
    if (mode === 'islands') {
        const datasets = [
            {
                label: 'Mean Within-Island Diversity',
                data: processedData?.islands?.meanWithin || [],
                ...getDiversityDatasetStyle('rgba(5, 150, 105, 1)', { fill: true, borderWidth: 3 })
            },
            {
                label: 'Inter-Island Separation',
                data: processedData?.islands?.interIsland || [],
                ...getDiversityDatasetStyle('rgba(217, 119, 6, 1)', { borderWidth: 3 })
            }
        ];
        const perIsland = processedData?.islands?.perIsland;
        if (perIsland instanceof Map) {
            [...perIsland.keys()].sort((a, b) => a - b).forEach((islandId) => {
                datasets.push({
                    label: `Island ${islandId + 1}`,
                    data: perIsland.get(islandId),
                    ...getDiversityDatasetStyle(getDiversityIslandColor(islandId), {
                        pointRadius: 3,
                        borderDash: [4, 4]
                    })
                });
            });
        }
        return datasets;
    }

    return [
        {
            label: 'Current Population Diversity',
            data: processedData?.overview?.current || [],
            ...getDiversityDatasetStyle('rgba(79, 70, 229, 1)', { fill: true, borderWidth: 3 })
        },
        {
            label: 'Cumulative Archive Diversity',
            data: processedData?.overview?.cumulative || [],
            ...getDiversityDatasetStyle('rgba(14, 165, 233, 1)', { borderWidth: 2 })
        },
        {
            label: 'Generation Drift',
            data: processedData?.overview?.drift || [],
            ...getDiversityDatasetStyle('rgba(16, 185, 129, 1)', { borderDash: [6, 4] })
        }
    ];
}

function formatDiversityMetric(value, digits = 3) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(digits) : '-';
}

function renderDiversitySummary(latest) {
    const summary = document.getElementById('diversity-summary');
    if (!summary) return;
    if (!latest) {
        summary.innerHTML = '';
        return;
    }

    const items = [
        {
            label: 'Current population',
            value: formatDiversityMetric(latest.current),
            hint: latest.generation === 0 ? 'Initial generation' : `Generation ${latest.generation}`
        },
        {
            label: 'Cumulative archive',
            value: formatDiversityMetric(latest.cumulative),
            hint: 'All ideas seen so far'
        },
        {
            label: 'Inter-island',
            value: formatDiversityMetric(latest.interIsland),
            hint: latest.islandCount && latest.islandCount > 1
                ? `${latest.islandCount} islands`
                : 'Requires 2+ islands'
        },
        {
            label: 'Island separation',
            value: formatDiversityMetric(latest.separationRatio, 2),
            hint: 'Inter / within ratio'
        }
    ];

    summary.innerHTML = items.map(item => `
        <div class="diversity-summary-item">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
            <small>${escapeHtml(item.hint)}</small>
        </div>
    `).join('');
}

function updateDiversityModeAvailability(processedData) {
    const islandsButton = document.querySelector('[data-diversity-mode="islands"]');
    if (!islandsButton) return;
    const hasIslandData = Boolean(processedData?.hasIslandData)
        || (processedData?.latest?.islandCount && processedData.latest.islandCount > 1);
    islandsButton.disabled = !hasIslandData;
    islandsButton.title = hasIslandData
        ? 'Show island-level diversity'
        : 'Island diversity requires an island run';
    if (!hasIslandData && diversityChartMode === 'islands') {
        diversityChartMode = 'overview';
    }
    document.querySelectorAll('[data-diversity-mode]').forEach((button) => {
        button.classList.toggle('active', button.dataset.diversityMode === diversityChartMode);
    });
}

function buildDiversitySignature(processedData) {
    if (!processedData || !processedData.labels) {
        return '';
    }

    return JSON.stringify({
        labels: processedData.labels,
        overview: processedData.overview,
        islands: {
            meanWithin: processedData.islands?.meanWithin || [],
            interIsland: processedData.islands?.interIsland || [],
            separationRatio: processedData.islands?.separationRatio || [],
            perIsland: processedData.islands?.perIsland instanceof Map
                ? [...processedData.islands.perIsland.entries()]
                : []
        },
        mode: diversityChartMode
    });
}

function revealDiversityPlot(processedData) {
    const plotSection = document.getElementById('diversity-plot-section');
    if (!plotSection || !processedData || processedData.labels.length === 0) {
        return;
    }

    plotSection.style.display = 'block';
    const loadingElement = plotSection.querySelector('.diversity-loading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }

    const canvas = document.getElementById('diversity-chart');
    const plotContainer = document.querySelector('.diversity-plot-container');
    if (canvas && plotContainer) {
        plotContainer.style.display = 'block';
        canvas.style.display = 'block';
    }
}

function scheduleDiversityChartResize(delay = 50) {
    window.setTimeout(() => {
        window.requestAnimationFrame(() => {
            if (!diversityChart) return;
            diversityChart.resize();
            diversityChart.update('none');
            diversityPlotSized = true;
        });
    }, delay);
}

/**
 * Update the diversity chart with new data
 * @param {Array} diversityHistory - Updated diversity history
 */
function updateDiversityChart(diversityHistory) {
    if (!diversityChart) {
        initializeDiversityChart();
    }

    if (!diversityChart) {
        console.error('❌ Failed to initialize diversity chart');
        return;
    }

    // Verify the chart is still connected to a valid canvas
    const canvas = document.getElementById('diversity-chart');
    if (!canvas) {
        console.error('❌ Canvas element not found, reinitializing...');
        initializeDiversityChart();
        if (!diversityChart) {
            console.error('❌ Failed to reinitialize diversity chart');
            return;
        }
    } else {
        // Check if chart canvas context is still valid
        try {
            const ctx = canvas.getContext('2d');
            if (!ctx || !diversityChart.canvas) {
                if (diversityChart) {
                    diversityChart.destroy();
                }
                initializeDiversityChart();
            }
        } catch (error) {
            console.error('❌ Canvas context error:', error);
            initializeDiversityChart();
        }
    }

    try {
        // Process the data
        const processedData = processDiversityData(diversityHistory);
        updateDiversityModeAvailability(processedData);
        const nextSignature = buildDiversitySignature(processedData);
        renderDiversitySummary(processedData.latest);
        revealDiversityPlot(processedData);

        // Skip data replacement when unchanged, but still resize because Chart.js
        // may have initialized while the panel was collapsed or hidden.
        if (nextSignature === lastDiversityRenderSignature) {
            if (processedData.labels.length > 0) {
                scheduleDiversityChartResize();
            }
            return false;
        }

        // Update chart data
        diversityChart.data.labels = processedData.labels;
        diversityChart.data.datasets = getDiversityChartDatasets(processedData, diversityChartMode);
        diversityChart.options.scales = getDiversityChartScales();
        diversityData = processedData;
        lastDiversityRenderSignature = nextSignature;

        // Update without animation to avoid visible flicker during polling.
        diversityChart.update('none');

        // Show the diversity plot section if there's data
        const plotSection = document.getElementById('diversity-plot-section');
        if (plotSection && processedData.labels.length > 0) {
            plotSection.style.display = 'block';

            // Hide loading state if showing
            const loadingElement = plotSection.querySelector('.diversity-loading');
            if (loadingElement) {
                loadingElement.style.display = 'none';
            }

            // Ensure proper sizing of canvas and container
            const canvas = document.getElementById('diversity-chart');
            const plotContainer = document.querySelector('.diversity-plot-container');
            const plotSection = document.getElementById('diversity-plot-section');

            if (canvas && plotContainer) {
                // Ensure parent section is visible
                if (plotSection) {
                    plotSection.style.display = 'block';
                }

                // Force proper sizing
                plotContainer.style.display = 'block';
                canvas.style.display = 'block';

                scheduleDiversityChartResize(diversityPlotSized ? 50 : 100);
            }
        }
        return true;
    } catch (error) {
        console.error('❌ Error updating diversity chart:', error);
        return false;
    }
}

/**
 * Show diversity plot section with loading state
 */
function showDiversityPlotLoading() {
    const plotSection = document.getElementById('diversity-plot-section');
    const plotContainer = document.querySelector('.diversity-plot-container');

    if (plotSection && plotContainer) {
        // plotSection.style.display = 'block'; // Don't show until we have data

        // Don't replace the entire content - just add loading overlay
        const existingCanvas = plotContainer.querySelector('#diversity-chart');
        const existingLoading = plotContainer.querySelector('.diversity-loading');

        if (!existingLoading) {
            // Create loading overlay instead of replacing content
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'diversity-loading';
            loadingDiv.innerHTML = `
                <div class="loading-spinner"></div>
                <span>Calculating diversity metrics...</span>
            `;
            plotContainer.appendChild(loadingDiv);
        }

        // Make sure canvas exists
        if (!existingCanvas) {
            const canvas = document.createElement('canvas');
            canvas.id = 'diversity-chart';
            canvas.width = 800;
            canvas.height = 400;
            plotContainer.appendChild(canvas);
        }
    }
}

/**
 * Reset diversity plot to initial state
 */
function resetDiversityPlot() {
    const plotSection = document.getElementById('diversity-plot-section');
    if (plotSection) {
        plotSection.style.display = 'none';
    }

    if (diversityChart) {
        diversityChart.data.labels = [];
        diversityChart.data.datasets = getDiversityChartDatasets(null, diversityChartMode);
        diversityChart.update();
    }

    // Reset global diversity data
    diversityData = {
        labels: [],
        overview: {},
        islands: {},
        latest: null
    };
    renderDiversitySummary(null);
    lastDiversityRenderSignature = '';
    diversityPlotSized = false;
}

/**
 * Ensure diversity chart is properly sized and visible
 */
function ensureDiversityChartSizing() {
    const plotSection = document.getElementById('diversity-plot-section');
    const plotContainer = document.querySelector('.diversity-plot-container');
    const canvas = document.getElementById('diversity-chart');

    if (plotSection && plotContainer && canvas) {
        // Make sure all elements are visible
        plotSection.style.display = 'block';
        plotContainer.style.display = 'block';

        // Set explicit dimensions
        plotContainer.style.height = '450px';
        plotContainer.style.width = '100%';
        plotContainer.style.padding = '20px';
        plotContainer.style.boxSizing = 'border-box';
        plotContainer.style.marginBottom = '2rem';

        canvas.style.display = 'block';
        canvas.style.height = '400px';
        canvas.style.width = '100%';
        canvas.style.maxWidth = '100%';
        canvas.style.maxHeight = '400px';

        // Force chart resize if it exists
        if (diversityChart) {
            setTimeout(() => {
                diversityChart.resize();
                diversityChart.update('none'); // Update without animation for immediate effect
            }, 50);
        }

        console.log('✓ Diversity chart sizing ensured');
    }
}

/**
 * Handle diversity data from progress updates
 * @param {Object} progressData - Progress update data containing diversity_history
 */
function handleDiversityUpdate(progressData) {
    if (progressData && progressData.diversity_history) {
        updateDiversityChart(progressData.diversity_history);
    }
}

// Add event listeners when the document is ready
document.addEventListener('DOMContentLoaded', function () {
    // Initialize diversity chart when page loads
    setTimeout(() => {
        initializeDiversityChart();
    }, 500);

    // Enhance the start evolution function to reset diversity plot
    const startButton = document.getElementById('startButton');
    if (startButton) {
        startButton.addEventListener('click', function () {
            // Reset diversity plot when starting new evolution
            resetDiversityPlot();
            showDiversityPlotLoading();
        });
    }

    document.querySelectorAll('[data-diversity-mode]').forEach((button) => {
        button.addEventListener('click', () => setDiversityChartMode(button.dataset.diversityMode));
    });
});

// Function to show the prompt modal (handles both initial and breeding prompts)
function showPromptModal(ideaIndex, promptType, generationIndex = 0) {
    console.log(`showPromptModal called with ideaIndex: ${ideaIndex}, promptType: ${promptType}, generationIndex: ${generationIndex}`);

    // Create modal if it doesn't exist
    let promptModal = document.getElementById('promptModal');

    if (!promptModal) {
        // Create the modal element
        promptModal = document.createElement('div');
        promptModal.className = 'modal fade';
        promptModal.id = 'promptModal';
        promptModal.tabIndex = '-1';
        promptModal.setAttribute('aria-labelledby', 'promptModalLabel');
        promptModal.setAttribute('aria-hidden', 'true');

        // Set up the modal HTML structure
        promptModal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="promptModalLabel">Prompt</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div id="promptModalContent"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;

        // Add the modal to the document body
        document.body.appendChild(promptModal);
    }

    // Get the modal content element
    const promptModalContent = document.getElementById('promptModalContent');
    const modalTitle = document.getElementById('promptModalLabel');

    let displayContent = '';
    let displayTitle = '';

    if (promptType === 'initial') {
        // Handle initial generation prompts
        if (specificPrompts.length > 0 && specificPrompts[ideaIndex]) {
            displayContent = specificPrompts[ideaIndex];
            displayTitle = `Initial Prompt for Idea ${ideaIndex + 1}`;
        } else if (contexts.length > 0 && contexts[ideaIndex]) {
            displayContent = contexts[ideaIndex];
            displayTitle = `Initial Context for Idea ${ideaIndex + 1}`;
        } else {
            displayContent = 'No prompt available';
            displayTitle = 'Initial Prompt';
        }

        console.log(`Using initial prompt for idea ${ideaIndex}: "${displayContent}"`);
    } else if (promptType === 'breeding') {
        // Handle breeding generation prompts
        const breedingGenIndex = generationIndex - 1; // Generation 1 corresponds to index 0 in breedingPrompts

        if (breedingPrompts.length > breedingGenIndex &&
            breedingPrompts[breedingGenIndex] &&
            breedingPrompts[breedingGenIndex][ideaIndex]) {
            displayContent = breedingPrompts[breedingGenIndex][ideaIndex];
            displayTitle = `Breeding Prompt for Generation ${generationIndex}, Idea ${ideaIndex + 1}`;
        } else {
            displayContent = 'No breeding prompt available for this idea';
            displayTitle = `Breeding Prompt - Generation ${generationIndex}`;
        }

        console.log(`Using breeding prompt for gen ${generationIndex}, idea ${ideaIndex}: "${displayContent}"`);
    }

    // Update the modal title
    modalTitle.textContent = displayTitle;

    // Format the content
    if (displayContent && displayContent !== 'No prompt available' && displayContent !== 'No breeding prompt available for this idea') {
        promptModalContent.innerHTML = `
            <div class="prompt-content">
                <div class="alert alert-info mb-3">
                    <strong>${promptType === 'initial' ? 'Initial' : 'Breeding'} Prompt:</strong>
                    ${promptType === 'initial'
                ? 'This direction prompt was generated from the context pool to create this initial idea.'
                : 'This direction prompt was generated from parent concepts to create this bred idea.'}
                </div>
                <div class="prompt-markdown">${renderMarkdown(displayContent)}</div>
            </div>
        `;
    } else {
        promptModalContent.innerHTML = `<p class="text-muted">${escapeHtml(displayContent)}</p>`;
    }

    // Show the modal
    if (window.bootstrap) {
        const modal = new bootstrap.Modal(promptModal);

        // Add cleanup event listener
        const cleanupFunction = function (event) {
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        };

        promptModal.removeEventListener('hidden.bs.modal', cleanupFunction);
        promptModal.addEventListener('hidden.bs.modal', cleanupFunction);

        modal.show();
    } else {
        console.error("Bootstrap is not available. Make sure it's properly loaded.");
        promptModal.style.display = 'block';
    }
}

const THINKING_UI_CONFIGS = {
    'gemini-2.5-flash-lite': {
        supportsThinking: true,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'off',
        min: 512,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Off for speed/cost. Use Low/Medium/High or a custom token budget when needed.'
    },
    'gemini-2.5-flash': {
        supportsThinking: true,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'off',
        min: 128,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Off for speed/cost. Use Low/Medium/High or a custom token budget when needed.'
    },
    'gemini-2.5-pro': {
        supportsThinking: true,
        allowOff: false,
        allowMedium: true,
        defaultMode: 'low',
        min: 128,
        max: 32768,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Low on Pro models. Increase to Medium/High or set a custom budget for deeper reasoning.'
    },
    'gemini-3-flash-preview': {
        supportsThinking: true,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'minimal',
        offValue: 'minimal',
        offLabel: 'Minimal',
        offDescription: ' - Lowest Gemini 3 thinking level',
        min: 0,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Minimal for speed/cost. Use Low/Medium/High or a custom token budget when needed.'
    },
    'gemini-3.1-flash-lite-preview': {
        supportsThinking: true,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'minimal',
        offValue: 'minimal',
        offLabel: 'Minimal',
        offDescription: ' - Lowest Gemini 3 thinking level',
        min: 0,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Minimal for speed/cost. Use Low/Medium/High or a custom token budget when needed.'
    },
    'gemini-3.1-pro-preview': {
        supportsThinking: true,
        allowOff: false,
        allowMedium: false,
        defaultMode: 'low',
        min: 128,
        max: 32768,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Low on Pro models. Increase to Medium/High or set a custom budget for deeper reasoning.'
    }
};

function getThinkingModelConfig(modelName) {
    return THINKING_UI_CONFIGS[modelName] || null;
}

/**
 * Set up model change listener to show/hide thinking control
 */
function setupModelChangeListener() {
    const modelSelect = document.getElementById('modelType');
    const thinkingBudgetContainer = document.getElementById('thinkingBudgetContainer');
    if (!modelSelect || !thinkingBudgetContainer) return;

    modelSelect.addEventListener('change', function () {
        updateThinkingBudgetVisibility();
    });
    updateThinkingBudgetVisibility();
}

/**
 * Update thinking control visibility based on selected model
 */
function updateThinkingBudgetVisibility() {
    const modelSelect = document.getElementById('modelType');
    const thinkingBudgetContainer = document.getElementById('thinkingBudgetContainer');
    if (!modelSelect || !thinkingBudgetContainer) return;

    const selectedModel = modelSelect.value;
    const config = getThinkingModelConfig(selectedModel);
    if (!config || !config.supportsThinking) {
        thinkingBudgetContainer.style.display = 'none';
        return;
    }

    thinkingBudgetContainer.style.display = 'block';
    configureThinkingBudgetForModel(selectedModel, config);
}

/**
 * Configure thinking controls for a specific model.
 */
function configureThinkingBudgetForModel(modelName, providedConfig = null) {
    const config = providedConfig || getThinkingModelConfig(modelName);
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');
    const thinkingBudgetHelp = document.getElementById('thinkingBudgetHelp');
    const thinkingBudgetMin = document.getElementById('thinkingBudgetMin');
    const thinkingBudgetMax = document.getElementById('thinkingBudgetMax');
    const offOption = document.getElementById('thinkingOffOption');
    const mediumOption = document.getElementById('thinkingMediumOption');
    const offLabel = document.getElementById('thinkingOffLabel');
    const offDescription = document.getElementById('thinkingOffDescription');

    const offRadio = document.getElementById('thinkingOff');
    const lowRadio = document.getElementById('thinkingLow');
    const mediumRadio = document.getElementById('thinkingMedium');
    const highRadio = document.getElementById('thinkingHigh');
    const customRadio = document.getElementById('thinkingCustomBudget');

    if (!config || !thinkingBudgetSlider || !thinkingBudgetHelp || !customRadio) return;

    // Toggle option availability by model capability.
    if (offOption) offOption.style.display = config.allowOff ? 'block' : 'none';
    if (mediumOption) mediumOption.style.display = config.allowMedium ? 'block' : 'none';
    if (offRadio) offRadio.value = config.offValue || 'off';
    if (offLabel) offLabel.textContent = config.offLabel || 'Off';
    if (offDescription) offDescription.textContent = config.offDescription || ' - Prefer speed and lower cost';

    if (offRadio) offRadio.checked = false;
    if (lowRadio) lowRadio.checked = false;
    if (mediumRadio) mediumRadio.checked = false;
    if (highRadio) highRadio.checked = false;
    customRadio.checked = false;

    let effectiveDefault = config.defaultMode;
    if (effectiveDefault === 'off' && !config.allowOff) {
        effectiveDefault = 'low';
    }
    if (effectiveDefault === 'medium' && !config.allowMedium) {
        effectiveDefault = 'low';
    }

    if (offRadio && effectiveDefault === offRadio.value) {
        offRadio.checked = true;
    } else if (effectiveDefault === 'low' && lowRadio) {
        lowRadio.checked = true;
    } else if (effectiveDefault === 'medium' && mediumRadio) {
        mediumRadio.checked = true;
    } else if (effectiveDefault === 'high' && highRadio) {
        highRadio.checked = true;
    } else {
        customRadio.checked = true;
    }

    thinkingBudgetSlider.min = String(config.min);
    thinkingBudgetSlider.max = String(config.max);
    thinkingBudgetSlider.step = String(config.step || 128);
    thinkingBudgetSlider.value = String(config.defaultBudget || config.min || 0);

    if (thinkingBudgetMin) thinkingBudgetMin.textContent = Number(config.min || 0).toLocaleString();
    if (thinkingBudgetMax) thinkingBudgetMax.textContent = Number(config.max || 0).toLocaleString();
    thinkingBudgetHelp.textContent = config.help;

    updateThinkingBudgetMode();
    updateThinkingBudgetDisplay();
}

/**
 * Update custom thinking budget display value.
 */
function updateThinkingBudgetDisplay() {
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');
    const thinkingBudgetValue = document.getElementById('thinkingBudgetValue');
    if (!thinkingBudgetSlider || !thinkingBudgetValue) return;
    const value = parseInt(thinkingBudgetSlider.value || '0', 10);
    thinkingBudgetValue.textContent = value.toLocaleString();
}

/**
 * Handle thinking mode changes (radio buttons).
 */
function updateThinkingBudgetMode() {
    const customContainer = document.getElementById('thinkingBudgetCustomContainer');
    const customRadio = document.getElementById('thinkingCustomBudget');
    if (!customContainer || !customRadio) return;

    if (customRadio.checked) {
        customContainer.style.display = 'block';
        updateThinkingBudgetDisplay();
    } else {
        customContainer.style.display = 'none';
    }
}

/**
 * Get thinking settings for API requests.
 */
function getThinkingSettings() {
    const modelSelect = document.getElementById('modelType');
    if (!modelSelect) {
        return { thinkingLevel: null, thinkingBudget: null };
    }

    const selectedModel = modelSelect.value;
    const config = getThinkingModelConfig(selectedModel);
    if (!config || !config.supportsThinking) {
        return { thinkingLevel: null, thinkingBudget: null };
    }

    const offRadio = document.getElementById('thinkingOff');
    const lowRadio = document.getElementById('thinkingLow');
    const mediumRadio = document.getElementById('thinkingMedium');
    const highRadio = document.getElementById('thinkingHigh');
    const customRadio = document.getElementById('thinkingCustomBudget');
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');

    if (customRadio && customRadio.checked && thinkingBudgetSlider) {
        const budget = parseInt(thinkingBudgetSlider.value || '0', 10);
        return {
            thinkingLevel: null,
            thinkingBudget: Number.isFinite(budget) ? budget : null,
        };
    }
    if (offRadio && offRadio.checked) {
        return { thinkingLevel: offRadio.value || 'off', thinkingBudget: null };
    }
    if (lowRadio && lowRadio.checked) {
        return { thinkingLevel: 'low', thinkingBudget: null };
    }
    if (mediumRadio && mediumRadio.checked) {
        return { thinkingLevel: 'medium', thinkingBudget: null };
    }
    if (highRadio && highRadio.checked) {
        return { thinkingLevel: 'high', thinkingBudget: null };
    }

    if (config.defaultMode === 'custom' && thinkingBudgetSlider) {
        const budget = parseInt(thinkingBudgetSlider.value || '0', 10);
        return {
            thinkingLevel: null,
            thinkingBudget: Number.isFinite(budget) ? budget : null,
        };
    }

    return {
        thinkingLevel: config.defaultMode || 'low',
        thinkingBudget: null,
    };
}

// Sidebar View Navigation
function showSidebarView(viewName) {
    // Hide all views first (move to right)
    document.querySelectorAll('.sidebar-view').forEach(view => {
        view.classList.remove('active');
        view.classList.remove('slide-left');
    });

    // Logic for transitions
    const mainView = document.getElementById('view-main');
    const targetView = document.getElementById(`view-${viewName}`);
    const sidebar = document.querySelector('.controls-panel');

    if (viewName === 'main') {
        mainView.classList.add('active');
        // Collapse sidebar when returning to main
        if (sidebar) sidebar.classList.remove('expanded');
        // Show name input section
        const nameSection = document.getElementById('evolutionNameSection');
        if (nameSection) nameSection.style.display = 'block';
    } else {
        // Move main view to left (parallax)
        mainView.classList.add('slide-left');
        // Bring target view in
        if (targetView) targetView.classList.add('active');

        // Expand sidebar for history view
        if (viewName === 'history') {
            if (sidebar) sidebar.classList.add('expanded');
            // Load/refresh history when entering
            loadHistory();
        } else {
            // Collapse for other views
            if (sidebar) sidebar.classList.remove('expanded');
        }
    }

    // Special handling for settings
    if (viewName === 'settings') {
        checkSettingsStatus();
    }
}

// Expose to window for HTML onclick attributes
window.showSidebarView = showSidebarView;
window.selectHistoryItem = selectHistoryItem;
window.clearHistorySelection = clearHistorySelection;
window.confirmDeleteEvolution = confirmDeleteEvolution;
window.handleHistoryResume = handleHistoryResume;
window.handleHistoryContinue = handleHistoryContinue;
window.showContinueInputInline = showContinueInputInline;
window.executeContinueInline = executeContinueInline;
window.cancelContinueInputInline = cancelContinueInputInline;
window.showRenameInputInline = showRenameInputInline;
window.executeRenameInline = executeRenameInline;
window.cancelRenameInline = cancelRenameInline;
window.navigateToRatePage = navigateToRatePage;

/**
 * Check if API key is set and update UI
 */
async function checkSettingsStatus() {
    try {
        const response = await fetch('/api/settings/status');
        const data = await response.json();

        const inputContainer = document.getElementById('apiKeyInputContainer');
        const configuredContainer = document.getElementById('apiKeyConfiguredContainer');
        const maskedDisplay = document.getElementById('maskedKeyDisplay');

        if (data.masked_key) {
            // Key is set
            if (inputContainer) inputContainer.style.display = 'none';
            if (configuredContainer) configuredContainer.style.display = 'block';
            if (maskedDisplay) maskedDisplay.textContent = data.masked_key;

            // Store masked key for copy function
            window.maskedApiKey = data.masked_key;
            window.fullApiKey = data.api_key;
        } else {
            // No key set
            if (inputContainer) inputContainer.style.display = 'block';
            if (configuredContainer) configuredContainer.style.display = 'none';
        }
    } catch (e) {
        console.error("Error checking settings:", e);
    }
}

/**
 * Save API Key
 */
async function saveSettings() {
    const keyInput = document.getElementById('apiKeyInput');
    const key = keyInput ? keyInput.value.trim() : '';

    if (!key) {
        showStatusInSettings('Please enter an API key', 'warning');
        return;
    }

    const btn = document.getElementById('saveSettingsBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Saving...';

    try {
        const response = await fetch('/api/settings/api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key })
        });

        const data = await response.json();

        if (data.status === 'success') {
            showStatusInSettings('API Key saved successfully!', 'success');
            // Clear input
            keyInput.value = '';
            // Refresh status to switch view
            await checkSettingsStatus();
            // Reload page after a short delay to apply changes
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            showStatusInSettings('Error: ' + data.message, 'danger');
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    } catch (e) {
        showStatusInSettings('Error: ' + e.message, 'danger');
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * Delete API Key
 */
async function deleteApiKey(e) {
    if (e) e.preventDefault();

    // Show the Bootstrap modal
    const deleteModal = new bootstrap.Modal(document.getElementById('deleteKeyModal'));
    deleteModal.show();
}

async function confirmDeleteApiKey() {
    const btn = document.getElementById('confirmDeleteKeyBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Deleting...';

    try {
        const response = await fetch('/api/settings/api-key', {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Hide modal
            const deleteModalEl = document.getElementById('deleteKeyModal');
            const modal = bootstrap.Modal.getInstance(deleteModalEl);
            if (modal) modal.hide();

            showStatusInSettings('API Key deleted.', 'success');
            // Refresh status to switch view
            await checkSettingsStatus();
            // Reload page to clear state
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            const msg = data.message || data.detail || 'Unknown error';
            showStatusInSettings('Error: ' + msg, 'danger');
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    } catch (e) {
        showStatusInSettings('Error: ' + e.message, 'danger');
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * Copy API Key (Masked)
 */
function copyApiKey() {
    const textToCopy = window.fullApiKey || window.maskedApiKey || "";
    navigator.clipboard.writeText(textToCopy).then(() => {
        const btn = document.getElementById('copyApiKeyBtn');
        const originalHTML = btn.innerHTML;

        btn.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
        btn.classList.remove('btn-outline-secondary');
        btn.classList.add('btn-success');

        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.classList.remove('btn-success');
            btn.classList.add('btn-outline-secondary');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        showStatusInSettings('Failed to copy to clipboard', 'danger');
    });
}

// Note: handleTemplateGeneration is defined in viewer.html with proper DOM elements

function showStatusInSettings(msg, type) {
    const el = document.getElementById('settingsStatus');
    if (el) {
        el.className = `alert alert-${type}`;
        el.textContent = msg;
        el.style.display = 'block';
    }
}

function toggleApiKeyVisibility() {
    const input = document.getElementById('apiKeyInput');
    const btn = document.getElementById('toggleApiKeyVisibility');
    const icon = btn ? btn.querySelector('i') : null;

    if (input && icon) {
        if (input.type === 'password') {
            input.type = 'text';
            icon.classList.remove('fa-eye');
            icon.classList.add('fa-eye-slash');
        } else {
            input.type = 'password';
            icon.classList.remove('fa-eye-slash');
            icon.classList.add('fa-eye');
        }
    }
}
