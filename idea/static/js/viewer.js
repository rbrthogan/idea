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
let rankedSortMetric = 'fitness';
let rankedScope = 'history';
let geneCatalogue = null;
let geneScope = 'current';
let selectedGeneId = null;
let lastGeneObservationSignature = '';
let lastEvolutionTreeSignature = '';
let treeResizeListenerAttached = false;
let currentEvolutionTree = null;
let selectedTreeNodeId = null;
const TREE_CONNECTOR_BOUNDARY_GAP = 2;

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

        div.innerHTML = `
            <button type="button" class="template-list-select" aria-current="${template.id === currentType ? 'true' : 'false'}">
                <span class="template-list-title">${escapeHtml(template.name)} ${template.is_system ? '<span class="badge bg-light text-dark border ms-1">System</span>' : '<span class="badge bg-primary ms-1">Custom</span>'}</span>
                <span class="template-list-description">${escapeHtml(template.description || 'No description')}</span>
            </button>
            <div class="template-actions">
                <button type="button" class="btn btn-xs btn-outline-secondary template-edit-button">
                    <i class="fas fa-edit"></i> ${template.is_system ? 'Customize' : 'Edit'}
                </button>
            </div>
        `;
        div.querySelector('.template-list-select')?.addEventListener('click', () => {
            selectTemplate(template.id);
        });
        div.querySelector('.template-edit-button')?.addEventListener('click', event => {
            event.stopPropagation();
            editTemplateMainPage(template.id);
        });

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

let executionSettingsCache = null;

async function loadExecutionSettings(force = false) {
    const data = await window.IdeaWorkbench.loadProviderSettings(force);
    executionSettingsCache = data;
    return data;
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

let initializedWorkspaceOwner = null;
async function initializeAuthenticatedWorkspace() {
    const owner = window.currentUser?.uid;
    if (!owner || initializedWorkspaceOwner === owner) return;
    initializedWorkspaceOwner = owner;
    await loadTemplateTypes({ preserveSelection: true, suppressDefaultSelection: false });
    await loadHistory();
    const restored = await restoreCurrentEvolution();
    if (!restored) await checkForResumableCheckpoints();
}

window.addEventListener('idea-auth-ready', () => {
    if (document.readyState !== 'loading') void initializeAuthenticatedWorkspace();
});
window.addEventListener('idea-auth-cleared', () => {
    initializedWorkspaceOwner = null;
    isEvolutionRunning = false;
    clearTimeout(progressPollTimer);
    stopElapsedTimeUpdater();
});

// Wire controls immediately; load private data only once authentication is ready.
document.addEventListener('DOMContentLoaded', async () => {
    void initializeAuthenticatedWorkspace();

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
    addListener('independentBirthRate', 'input', updateBirthMixHelp);
    addListener('singleParentBirthRate', 'input', updateBirthMixHelp);
    updateBirthMixHelp();
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
    addListener('ageDecayFloor', 'input', () => {
        const el = document.getElementById('ageDecayFloor');
        const valueEl = document.getElementById('ageDecayFloorValue');
        if (el && valueEl) valueEl.textContent = parseFloat(el.value || '0.35').toFixed(2);
    });
    document.querySelectorAll('.evolution-view-toggle-btn').forEach((button) => {
        button.addEventListener('click', () => setEvolutionViewMode(button.dataset.viewMode));
        button.addEventListener('keydown', handleEvolutionViewTabKeydown);
    });
    setupModelChangeListener();

    // --- Evolution Controls ---

    addListener('startButton', 'click', async function () {
        console.log("Starting evolution...");
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');

        updateBirthMixHelp();
        for (const input of document.querySelectorAll('#view-main input, #view-advanced input')) {
            if (input.disabled || input.type === 'hidden' || input.checkValidity()) continue;
            showSidebarView(input.closest('#view-advanced') ? 'advanced' : 'main');
            window.IdeaWorkbench?.openSidebar?.();
            for (let parent = input.parentElement; parent; parent = parent.parentElement) {
                if (parent.tagName === 'DETAILS') parent.open = true;
            }
            input.reportValidity();
            input.focus();
            return;
        }
        document.getElementById('workspaceNotice')?.remove();
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

        // Tournament count (1.0 = ceil(log2(island size)) Swiss rounds)
        const tournamentCount = parseFloat(document.getElementById('tournamentCount').value);

        // Get mutation rate
        const mutationRate = parseFloat(document.getElementById('mutationRate').value);
        const seedContextPoolSizeInput = document.getElementById('seedContextPoolSize');
        const seedContextPoolSize = seedContextPoolSizeInput ? parseInt(seedContextPoolSizeInput.value) : null;
        const replacementRate = parseFloat(document.getElementById('replacementRate').value);
        const fitnessAlpha = parseFloat(document.getElementById('fitnessAlpha').value);
        const ageDecayRate = parseFloat(document.getElementById('ageDecayRate').value);
        const ageDecayFloor = parseFloat(document.getElementById('ageDecayFloor')?.value ?? '0.35');
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
            independentBirthRate: Number(document.getElementById('independentBirthRate')?.value ?? '0.5'),
            singleParentBirthRate: Number(document.getElementById('singleParentBirthRate')?.value ?? '0.25'),
            diversityRepresentation: document.getElementById('diversityRepresentation')?.value ?? 'criterion_summaries',
            seedContextPoolSize: Number.isInteger(seedContextPoolSize) ? seedContextPoolSize : null,
            replacementRate,
            fitnessAlpha,
            ageDecayRate,
            ageDecayFloor,
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

        // Update button states
        if (startButton) {
            startButton.disabled = true;
            startButton.textContent = 'Starting…';
        }
        if (stopButton) {
            stopButton.disabled = true;
            stopButton.style.display = 'none';
        }

        try {
            const response = await fetch('/api/start-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();
                currentEvolutionId = data.evolution_id;
                currentEvolutionName = data.evolution_name;
                resetUIState();
                if (evolutionNameInput) evolutionNameInput.value = '';
                window.IdeaWorkbench?.closeInspector();
                window.IdeaWorkbench?.closeSidebar();
                if (window.onEvolutionStart) window.onEvolutionStart();

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

                if (stopButton) { stopButton.disabled = false; stopButton.style.display = 'block'; }
                startButton.textContent = 'Running…';
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
                    showErrorMessage(apiErrorMessage(errorData, 'Unable to start this experiment. Check your settings and try again.'));
                }
                if (nameSection) nameSection.style.display = 'block';
                resetButtonStates();
            }
        } catch (error) {
            if (nameSection) nameSection.style.display = 'block';
            console.error("Error running evolution:", error);
            showErrorMessage("We could not confirm whether the experiment started. Check History or reconnect before trying again.");
            resetButtonStates();
            void checkRunningEvolution();
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
            if (!isEvolutionRunning) return;
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
                showErrorMessage(apiErrorMessage(await response.json().catch(() => ({})), "Unable to stop the experiment. Try Stop again."));
                clearTimeout(forceStopTimeout);
                if (stopButton) {
                    stopButton.disabled = false;
                    stopButton.textContent = 'Stop Evolution';
                }
            }
        } catch (error) {
            console.error("Error stopping evolution:", error);
            showErrorMessage("Stop could not be confirmed. The experiment may still be running. Reconnect and try Stop again.");
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
        let cancellationPending = false;
        let cancellationConfirmed = false;

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
                cancellationConfirmed = true;
                cancellationPending = Boolean(data.cancel_pending || data.is_running);
                addActivityLogItem(cancellationPending
                    ? 'Stop requested. Requests already running may finish.'
                    : (data.message || 'Evolution stopped.'), 'warning');

                // Store checkpoint ID for resume
                if (!cancellationPending && data.checkpoint_id) {
                    rememberCheckpointId(data.checkpoint_id);
                    showResumeButton(data.checkpoint_id);
                }

                if (cancellationPending) {
                    isEvolutionRunning = true;
                    if (stopButton) {
                        stopButton.disabled = true;
                        stopButton.textContent = 'Stopping…';
                    }
                } else {
                    resetButtonStates();
                    isEvolutionRunning = false;
                }
            } else {
                showErrorMessage(apiErrorMessage(await response.json().catch(() => ({})), "Stop could not be confirmed. Please try again."));
            }
        } catch (error) {
            console.error("Error force stopping:", error);
            showErrorMessage("Stop could not be confirmed. The experiment may still be running. Please try again.");
        } finally {
            if (forceStopButton) {
                forceStopButton.disabled = false;
                forceStopButton.innerHTML = '<i class="fas fa-bolt me-1"></i>Force Stop';
                forceStopButton.style.display = 'none';
            }
            if (stopButton && cancellationConfirmed && !cancellationPending) {
                stopButton.style.display = 'none';
            } else if (stopButton && !cancellationConfirmed) {
                stopButton.style.display = 'block';
                stopButton.disabled = false;
                stopButton.textContent = 'Stop Evolution';
                if (forceStopButton) forceStopButton.style.display = 'block';
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
    addListener('evolutionSelect', 'change', handleEvolutionSelection);

    // --- Settings ---
    addListener('saveSettingsBtn', 'click', saveSettings);
    addListener('apiKeyProviderSelect', 'change', () => {
        const input = document.getElementById('apiKeyInput');
        if (input) input.value = '';
        const status = document.getElementById('settingsStatus');
        if (status) status.style.display = 'none';
        renderApiKeyProviderStatus(executionSettingsCache);
    });
    addListener('deleteApiKeyBtn', 'click', deleteApiKey);
    addListener('confirmDeleteKeyBtn', 'click', confirmDeleteApiKey);

    // Setup button in alert (if present)
    const setupBtn = document.getElementById('setupApiKeyBtn');
    if (setupBtn) {
        setupBtn.addEventListener('click', () => window.IdeaWorkbench?.openProviderSettings());
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

    // The empty workspace teaches the workflow without blocking first use.

});

// Function to restore current evolution from localStorage
async function restoreCurrentEvolution() {
    try {
        console.log("Attempting to restore current evolution from localStorage...");
        let serverProgressData = null;

        // First check whether there is an active server-side run. Completed
        // local state can still be restored below so refresh does not blank the view.
        try {
            const progressResponse = await fetch('/api/progress');
            if (progressResponse.ok) {
                const progressData = await progressResponse.json();
                console.log("Evolution status from server:", progressData);
                serverProgressData = progressData;
            }
        } catch (e) {
            console.log("Could not check evolution status, attempting local restoration:", e);
        }

        // Check localStorage for current evolution data
        const storedData = localStorage.getItem(`idea.${window.currentUser?.uid || 'signed-out'}.currentEvolutionData`);
        if (storedData) {
            try {
                const evolutionState = JSON.parse(storedData);
                console.log("Found evolution data in localStorage:", evolutionState);

                // Handle both old format (just history array) and new format (object with history and diversity_history)
                let generationsData, diversityData, islandDiversityData, tokenCounts, tournamentData, migrationData;
                if (Array.isArray(evolutionState)) {
                    if (serverProgressData?.is_running && serverProgressData?.evolution_id) {
                        console.log("Legacy stored evolution has no id; using live active run instead");
                        localStorage.removeItem(`idea.${window.currentUser?.uid || 'signed-out'}.currentEvolutionData`);
                        startPollingActiveEvolution(serverProgressData);
                        return true;
                    }
                    // Old format - just the history array
                    generationsData = evolutionState;
                    diversityData = [];
                    islandDiversityData = [];
                    tokenCounts = null;
                    tournamentData = [];
                    migrationData = [];
                } else if (evolutionState && evolutionState.history) {
                    const storedEvolutionId = evolutionState.evolution_id || evolutionState.evolutionId || null;
                    const serverEvolutionId = serverProgressData?.evolution_id || null;
                    if (serverProgressData?.is_running && serverEvolutionId && storedEvolutionId !== serverEvolutionId) {
                        console.log(
                            "Stored evolution does not match active run; skipping localStorage restore",
                            { storedEvolutionId, serverEvolutionId }
                        );
                        localStorage.removeItem(`idea.${window.currentUser?.uid || 'signed-out'}.currentEvolutionData`);
                        startPollingActiveEvolution(serverProgressData);
                        return true;
                    }

                    // New format - object with history and diversity_history
                    generationsData = evolutionState.history;
                    diversityData = evolutionState.diversity_history || [];
                    islandDiversityData = evolutionState.island_diversity_history || [];
                    tokenCounts = evolutionState.token_counts || null;
                    tournamentData = evolutionState.tournament_history || [];
                    migrationData = evolutionState.migration_history || [];
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
                    // Artifact URLs are derived while cards render, so restore
                    // the owning run identity before creating any card DOM.
                    currentEvolutionId = evolutionState?.evolution_id
                        || evolutionState?.evolutionId
                        || serverProgressData?.evolution_id
                        || null;
                    currentEvolutionName = evolutionState?.evolution_name
                        || evolutionState?.evolutionName
                        || serverProgressData?.evolution_name
                        || null;

                    // Render the generations
                    selectedGeneId = null;
                    geneScope = 'current';
                    applyGeneCatalogue(evolutionState, true);
                    renderGenerations(generationsData);
                    const restoredVersion = Number.isInteger(evolutionState?.history_version)
                        ? evolutionState.history_version
                        : 0;
                    lastRenderedHistoryVersion = restoredVersion;
                    lastStoredHistoryVersion = restoredVersion;

                    // Restore metric plot if we have diversity data or per-generation population metrics.
                    if ((diversityData && diversityData.length > 0) || (generationsData && generationsData.length > 0)) {
                        console.log("Restoring diversity plot with data:", diversityData);
                        const chartUpdated = updateDiversityChart(diversityData || [], generationsData, islandDiversityData || []);
                        if (chartUpdated) {
                            // Ensure proper sizing after restoration
                            setTimeout(ensureDiversityChartSizing, 200);
                        }
                    }

                    // Set up currentEvolutionData for download functionality
                    currentEvolutionData = {
                        evolution_id: currentEvolutionId,
                        evolution_name: currentEvolutionName,
                        history: generationsData,
                        gene_catalogue: geneCatalogue,
                        history_version: restoredVersion,
                        diversity_history: diversityData,
                        contexts: contexts,
                        specific_prompts: specificPrompts,
                        breeding_prompts: breedingPrompts,
                        tournament_history: tournamentData,
                        island_diversity_history: islandDiversityData,
                        migration_history: migrationData,
                        token_counts: tokenCounts,
                        config: evolutionState?.config || null,
                        is_running: serverProgressData?.is_running ?? evolutionState?.is_running ?? false
                    };
                    if (typeof repairPersistedEvolutionFromClientCache === 'function') {
                        repairPersistedEvolutionFromClientCache(currentEvolutionData);
                    }

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
                    if (serverProgressData?.is_running) {
                        startPollingActiveEvolution(serverProgressData);
                    } else {
                        await retryDeferredEvolutionRepairs(serverProgressData);
                    }
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

    const catalogueChanged = applyGeneCatalogue(data, force);
    const observationSignature = safeSignature(data.history.map(population => population.map(idea => [idea.id, getGeneObservation(idea)])));
    const observationsChanged = observationSignature !== lastGeneObservationSignature;
    const hasVersion = Number.isInteger(data.history_version);
    const shouldRender = force || catalogueChanged || observationsChanged || !hasVersion || data.history_version !== lastRenderedHistoryVersion;
    if (!shouldRender) return false;

    lastGeneObservationSignature = observationSignature;
    renderGenerations(data.history);
    if (currentModalIdea && !document.getElementById('ideaModal')?.hidden) {
        const latest = data.history.flat().filter(idea => idea.id === currentModalIdea.id).at(-1);
        if (latest) renderIdeaModalGenes(latest);
    }
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
        evolution_name: stateData.evolution_name || currentEvolutionName || null,
        gene_catalogue: stateData.gene_catalogue ?? geneCatalogue
    };
    const owner = window.currentUser?.uid;
    persistStateTimeout = setTimeout(() => {
        if (!owner || window.currentUser?.uid !== owner) return;
        try {
            localStorage.setItem(`idea.${owner}.currentEvolutionData`, JSON.stringify(stateWithIdentity));
        } catch (error) {
            console.warn('Browser backup unavailable; server checkpoints remain authoritative.', error);
        }
    }, LOCAL_STORAGE_PERSIST_DELAY_MS);
}

function normalizeMarkdownStructure(text) {
    if (!text) return '';

    const commonSectionHeadings = [
        'Base Game & Twists',
        'Rules & Turn Sequence',
        'Level/Progression',
        'Scoring/Win/Lose',
        'Visuals & Audio',
        'Technical Notes',
        'Minimum Prototype',
        'Emergent Complexity',
        'Concept & Precedent Departure',
        'Precedent Palette Analysis',
        'Departure and Abstracted Principle',
        'Mathematical Mapping',
        'Finite Domain Definition',
        'Domain and Sequence Definition',
        'Core Mapping Functions',
        'Hypothesis, Null & Falsification',
        'Statement of Falsifiable Hypothesis',
        'Confirmatory Hypothesis',
        'Baseline / Null Model',
        'Quantitative Signal Definition',
        'Falsification Criterion',
        'Visual Encoding',
        'Detailed Encoding Rules',
        'Deterministic Implementation Plan',
        'Algorithmic Sequence',
        'Resource Bounds',
        'Artifact & Metric Contract',
        'Produced Files',
        'Interpretation Limits',
        'Confounders and Artifact Mitigation',
        'Win Condition',
        'Stretch Ideas',
        'Core Loop',
        'Components',
        'Mechanics',
        'Controls',
        'Overview',
        'Setup',
        'Summary',
        'Objective',
        'Implementation',
        'Constraints',
        'Features',
        'Description'
    ].sort((a, b) => b.length - a.length);
    const headingPattern = commonSectionHeadings
        .map(heading => heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|');
    const numberedHeadingPattern = `(?:\\d+\\.\\s*)?(?:${headingPattern})`;
    const inlineKnownHeadingPattern = new RegExp(`(?<=[^#\\n])\\s*(#{1,6})\\s*(${numberedHeadingPattern})(?=\\s|[#*:_\\-0-9A-Z("]|$)`, 'g');

    const normalizeSegment = (segment) => {
        let normalized = segment;

        // Repair heading markers that were emitted inline after a sentence.
        normalized = normalized.replace(inlineKnownHeadingPattern, '\n$1 $2');
        normalized = normalized.replace(/^(#{1,6})(?!#)(?=\S)/gm, '$1 ');

        // Repair common inline list markers emitted without line breaks.
        normalized = normalized.replace(/([:.!?])\s*(\d+\.\s+)/g, '$1\n$2');
        normalized = normalized.replace(/([.!?])\s*([*-]\s+)/g, '$1\n$2');

        // Repair common section headings where the heading body follows on the same line.
        normalized = normalized.replace(/^(#{1,6})\s+(.+)$/gm, (match, hashes, content) => {
            const trimmedContent = content.trim();
            const numberMatch = trimmedContent.match(/^(\d+\.\s*)/);
            const numberPrefix = numberMatch ? numberMatch[1] : '';
            const unnumberedContent = numberMatch
                ? trimmedContent.slice(numberMatch[0].length)
                : trimmedContent;
            for (const heading of commonSectionHeadings) {
                if (unnumberedContent === heading) {
                    return `${hashes} ${numberPrefix}${heading}`;
                }
                if (unnumberedContent.startsWith(heading)) {
                    const body = unnumberedContent.slice(heading.length).trim();
                    if (!body || !/^(?:\s|[*:_\-0-9A-Z("])/.test(body)) {
                        continue;
                    }
                    return `${hashes} ${numberPrefix}${heading}\n${body}`;
                }
            }
            return `${hashes} ${trimmedContent}`;
        });

        return normalized;
    };

    return String(text)
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .split(/(```[\s\S]*?```)/g)
        .map(part => part.startsWith('```') ? part : normalizeSegment(part))
        .join('');
}

// Markdown links are deliberately limited to web URLs and same-origin/relative
// navigation. Generated text must never be able to opt into an active protocol.
function decodeMarkdownCodePoint(value, radix) {
    const codePoint = parseInt(value, radix);
    if (!Number.isInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) {
        return '\ufffd';
    }
    return String.fromCodePoint(codePoint);
}

function decodeMarkdownUrlForPolicy(value) {
    let decoded = String(value || '');
    for (let index = 0; index < 3; index++) {
        decoded = decoded
            .replace(/&amp;/gi, '&')
            .replace(/&colon;/gi, ':')
            .replace(/&tab;/gi, '\t')
            .replace(/&newline;/gi, '\n')
            .replace(/&#x([0-9a-f]+);?/gi, (match, hex) => decodeMarkdownCodePoint(hex, 16))
            .replace(/&#([0-9]+);?/g, (match, decimal) => decodeMarkdownCodePoint(decimal, 10));
    }
    return decoded;
}

function sanitizeMarkdownUrl(value) {
    const renderedValue = String(value || '').trim();
    const policyValue = decodeMarkdownUrlForPolicy(renderedValue).trim();
    if (!policyValue || /[\s\u0000-\u001f\u007f]/.test(policyValue) || /[\\"'<>]/.test(policyValue)) {
        return null;
    }
    if (/^https?:\/\//i.test(policyValue)) {
        return renderedValue;
    }
    if (/^(?:\/(?!\/)|\.\.?\/|#|\?)/.test(policyValue)) {
        return renderedValue;
    }
    if (!policyValue.includes(':') && !policyValue.startsWith('//')) {
        return renderedValue;
    }
    return null;
}

// Improve the markdown rendering function with better newline handling.
// This is intentionally a small allow-list renderer, not an HTML passthrough.
function renderMarkdown(text) {
    if (!text) return '';

    text = normalizeMarkdownStructure(text);

    // Escape HTML to prevent XSS
    text = text.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

    // Process code blocks first
    const codeBlocks = [];
    text = text.replace(/```(?:[^\n`]*\n)?([\s\S]*?)```/g, function (match, code) {
        const index = codeBlocks.length;
        codeBlocks.push('<pre><code>' + code.trim() + '</code></pre>');
        return `<pre data-md-code-block="${index}"></pre>`;
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
        else if (inList) {
            const closingTag = listType === 'ul' ? '</ul>' : '</ol>';
            lines[i] = line.trim() === '' ? closingTag : `${closingTag}\n${line}`;
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

    // Process links. Unsafe destinations degrade to their inert label.
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, destination) => {
        const safeDestination = sanitizeMarkdownUrl(destination);
        return safeDestination
            ? `<a href="${safeDestination}" target="_blank" rel="noopener noreferrer">${label}</a>`
            : label;
    });

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

    let rendered = lines.join('\n');
    codeBlocks.forEach((codeBlock, index) => {
        rendered = rendered.split(`<pre data-md-code-block="${index}"></pre>`).join(codeBlock);
    });

    return rendered;
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
                    ${entry.source === 'consolidated' ? ` · Catalogue revision ${escapeHtml(String(entry.gene_pool_revision))}` : entry.source === 'raw' ? ' · Raw genes' : ''}
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

function getIdeaArtifactRefs(idea) {
    const refs = idea?.artifact_refs ?? idea?.idea?.artifact_refs;
    if (!Array.isArray(refs)) return [];
    return refs.filter(ref => (
        ref
        && typeof ref === 'object'
        && typeof ref.id === 'string'
        && typeof ref.role === 'string'
        && typeof ref.media_type === 'string'
    ));
}

function getPrimaryImageArtifact(idea) {
    const imageRefs = getIdeaArtifactRefs(idea).filter(ref => ref.media_type === 'image/png');
    return imageRefs.find(ref => ref.role === 'primary') || imageRefs[0] || null;
}

function getArtifactEvaluationFailure(idea) {
    const evaluations = idea?.evaluations ?? idea?.idea?.evaluations;
    if (!evaluations || typeof evaluations !== 'object') return null;
    const artifactRecords = Object.values(evaluations).filter(record => (
        record
        && record.component_kind === 'llm_code_artifact'
    ));
    return artifactRecords.find(record => ['failed', 'skipped'].includes(record.status))
        || artifactRecords.find(record => ['ok', 'cached'].includes(record.status))
        || null;
}

function getIdeaArtifactSignature(idea) {
    return getIdeaArtifactRefs(idea)
        .map(ref => `${ref.id}:${ref.sha256 || ''}:${ref.content_digest || ''}`)
        .join('|');
}

function getEvolutionArtifactUrl(ref) {
    if (!currentEvolutionId || !ref?.id) return null;
    return `/api/evolution/${encodeURIComponent(currentEvolutionId)}/artifacts/${encodeURIComponent(ref.id)}`;
}

async function hydrateAuthenticatedArtifactImages(container) {
    if (!container) return;
    const images = container.querySelectorAll('img[data-artifact-url]');
    images.forEach(async image => {
        const artifactUrl = image.dataset.artifactUrl;
        image.removeAttribute('data-artifact-url');
        if (!artifactUrl) return;
        try {
            const response = await fetch(artifactUrl, { headers: { Accept: 'image/png' } });
            if (!response.ok) throw new Error(`artifact request failed (${response.status})`);
            const blob = await response.blob();
            if (blob.type && blob.type !== 'image/png') throw new Error('artifact is not a PNG');
            const objectUrl = URL.createObjectURL(blob);
            const revoke = () => URL.revokeObjectURL(objectUrl);
            image.addEventListener('load', revoke, { once: true });
            image.addEventListener('error', revoke, { once: true });
            image.src = objectUrl;
        } catch (error) {
            image.hidden = true;
            const artifactContainer = image.closest('figure, .idea-card-artifact');
            artifactContainer?.classList.add('artifact-unavailable');
            if (
                artifactContainer
                && !artifactContainer.querySelector('.idea-card-artifact-failure')
            ) {
                const failure = document.createElement('div');
                failure.className = 'idea-card-artifact-failure';
                failure.setAttribute('role', 'status');
                failure.textContent = `Visualization unavailable: ${error.message}`;
                artifactContainer.appendChild(failure);
            }
            console.warn('Artifact preview unavailable:', error);
        }
    });
}

function renderCardArtifactPreview(idea) {
    const primary = getPrimaryImageArtifact(idea);
    const artifactUrl = getEvolutionArtifactUrl(primary);
    if (!primary || !artifactUrl) {
        const failure = getArtifactEvaluationFailure(idea);
        if (!failure) return '';
        const execution = failure?.metadata?.execution;
        const finalCause = execution?.repair_loop?.final_cause;
        const error = failure.error
            || finalCause?.message
            || failure?.metadata?.reason
            || 'primary PNG was not produced';
        return `
            <div class="idea-card-artifact-failure" role="status">
                <i class="fas fa-triangle-exclamation" aria-hidden="true"></i>
                <span>Visualization unavailable: ${escapeHtml(error)}</span>
            </div>
        `;
    }
    return `
        <div class="idea-card-artifact">
            <img data-artifact-url="${escapeHtml(artifactUrl)}" alt="Generated visualization" loading="lazy">
        </div>
    `;
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
    if (key.startsWith('eval:')) {
        const metricKey = key.slice(5);
        const records = idea?.evaluations || {};
        const component = Object.keys(records).sort((a, b) => b.length - a.length)
            .find(id => metricKey.startsWith(`${id}.`));
        const record = records[component];
        if (!['ok', 'cached'].includes(record?.status)) return null;
        const value = record?.metrics?.[metricKey.slice(component.length + 1)];
        if (value === null || value === undefined || value === '') return null;
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : null;
    }
    const selection = idea?.selection_metrics || {};
    const survival = idea?.survival_metrics || {};
    const aliases = {
        elo: [selection.elo, idea?.elo, idea?.ratings?.auto],
        diversity: [selection.diversity, idea?.diversity_score, idea?.diversity],
        fitness: [selection.fitness, survival.survival_score, idea?.fitness]
    };
    const candidates = aliases[key] || [];
    for (const candidate of candidates) {
        if (candidate === null || candidate === undefined || candidate === '') continue;
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
    // Every population view selects the same persistent candidate surface.
    const legacyInspector = document.getElementById('evolution-tree-inspector');
    if (legacyInspector) legacyInspector.hidden = true;
    const node = tree?.nodesById?.get(nodeId);
    if (!node) return;
    const idea = getIdeaData(node.idea || {});
    const signature = safeSignature(idea);
    const inspector = document.getElementById('ideaModal');
    if (inspector?.dataset.candidateSignature === signature && !inspector.hidden) return;
    if (inspector) inspector.dataset.candidateSignature = signature;
    showIdeaModal(idea);
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

function assignTreeConnectorTracks(routes) {
    const routesByGeneration = new Map();
    (routes || []).forEach((route) => {
        const generationKey = String(route.generation);
        if (!routesByGeneration.has(generationKey)) {
            routesByGeneration.set(generationKey, []);
        }
        routesByGeneration.get(generationKey).push(route);
    });

    routesByGeneration.forEach((generationRoutes) => {
        const tracks = [];
        generationRoutes
            .map(route => ({
                route,
                left: Math.min(route.from.x, route.to.x),
                right: Math.max(route.from.x, route.to.x)
            }))
            .sort((a, b) => (
                a.left - b.left
                || b.right - a.right
                || a.route.from.x - b.route.from.x
                || a.route.to.x - b.route.to.x
            ))
            .forEach((entry) => {
                let trackIndex = tracks.findIndex(trackRight => entry.left > trackRight + 8);
                if (trackIndex === -1) {
                    trackIndex = tracks.length;
                    tracks.push(entry.right);
                } else {
                    tracks[trackIndex] = entry.right;
                }
                entry.route.track = {
                    index: trackIndex,
                    count: 0
                };
            });

        const trackCount = Math.max(1, tracks.length);
        generationRoutes.forEach((route) => {
            route.track.count = trackCount;
        });
    });

    return routes;
}

function buildTreeSCurvePath(from, to, track = {}) {
    const gap = to.y - from.y;
    const trackCount = Math.max(1, Number(track.count) || 1);
    const trackIndex = Math.max(0, Math.min(trackCount - 1, Number(track.index) || 0));
    const channelRatio = (trackIndex + 1) / (trackCount + 1);
    const channelOffset = gap > 28
        ? Math.max(14, Math.min(gap - 14, gap * channelRatio))
        : gap * channelRatio;
    const channelY = from.y + channelOffset;
    const midX = (from.x + to.x) / 2;
    const direction = to.x >= from.x ? 1 : -1;
    const horizontalDistance = Math.abs(to.x - from.x);
    const bendX = Math.max(14, Math.min(72, horizontalDistance * 0.32));
    const firstControlY = from.y + ((channelY - from.y) * 0.62);
    const secondControlY = to.y - ((to.y - channelY) * 0.62);

    if (horizontalDistance < 4) {
        return `M ${from.x} ${from.y} C ${from.x} ${firstControlY}, ${to.x} ${secondControlY}, ${to.x} ${to.y}`;
    }

    return `M ${from.x} ${from.y} C ${from.x} ${firstControlY}, ${midX - (direction * bendX)} ${channelY}, ${midX} ${channelY} S ${to.x} ${secondControlY}, ${to.x} ${to.y}`;
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
    const latestGeneration = tree.generations.length > 0
        ? tree.generations[tree.generations.length - 1].generation
        : 0;

    const scroll = document.createElement('div');
    scroll.className = 'evolution-tree-scroll';
    scroll.id = 'evolution-tree-scroll';

    const connectorSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    connectorSvg.classList.add('evolution-tree-connectors');
    connectorSvg.setAttribute('aria-hidden', 'true');
    scroll.appendChild(connectorSvg);

    tree.generations.forEach((generationRow) => {
        const isCurrentPopulation = generationRow.generation === latestGeneration;
        const row = document.createElement('div');
        row.className = 'evolution-tree-row';
        row.dataset.generation = String(generationRow.generation);

        const label = document.createElement('div');
        label.className = 'evolution-tree-generation-label';
        const generationText = document.createElement('span');
        generationText.textContent = `Gen ${generationRow.generation}`;
        label.appendChild(generationText);
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
            if (node) {
                lane.appendChild(createEvolutionTreeNode(node, tree.fitnessRange, {
                    generationIndex: generationRow.generation,
                    isCurrentPopulation,
                    isSurvivorStep: slot.kind === 'continuation'
                }));
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

function createEvolutionTreeNode(node, fitnessRange, options = {}) {
    const idea = node.idea || {};
    const islandId = Number.isInteger(node.islandId) ? node.islandId : getIdeaIslandId(idea);
    const fitnessValue = getTreeMetricValue(idea, 'fitness');
    const fitnessStrength = getTreeFitnessStrength(fitnessValue, fitnessRange);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'evolution-tree-node';
    button.dataset.nodeId = node.id;
    if (Number.isInteger(options.generationIndex)) {
        button.dataset.generation = String(options.generationIndex);
    }
    button.dataset.hasFitness = fitnessStrength === null ? 'false' : 'true';
    button.classList.add(options.isCurrentPopulation ? 'tree-current-population' : 'tree-historical');
    if (options.isSurvivorStep) {
        button.classList.add('tree-survivor-step');
    }
    if (fitnessStrength !== null) {
        button.style.setProperty('--tree-fitness-strength', `${fitnessStrength.toFixed(1)}%`);
        const historicalFitnessStrength = Math.max(9, Math.min(28, 5 + (fitnessStrength * 0.52)));
        const historicalBorderStrength = Math.max(42, Math.min(82, 34 + fitnessStrength));
        const historicalTextStrength = Math.max(50, Math.min(78, 44 + (fitnessStrength * 0.72)));
        button.style.setProperty('--tree-historical-fitness-strength', `${historicalFitnessStrength.toFixed(1)}%`);
        button.style.setProperty('--tree-historical-border-strength', `${historicalBorderStrength.toFixed(1)}%`);
        button.style.setProperty('--tree-historical-text-strength', `${historicalTextStrength.toFixed(1)}%`);
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
        `${node.title || 'Idea'} remained in the population at generation ${generationIndex}`
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

    tree.continuations.forEach((continuation) => {
        const escapedNodeId = cssEscape(continuation.nodeId);
        const from = getAnchor(
            `[data-node-id="${escapedNodeId}"][data-generation="${continuation.fromGeneration}"]`,
            'bottom'
        );
        const to = getAnchor(
            `[data-node-id="${escapedNodeId}"][data-generation="${continuation.toGeneration}"]`,
            'top'
        );
        if (!from || !to) return;
        const fromY = from.y + TREE_CONNECTOR_BOUNDARY_GAP;
        const toY = to.y - TREE_CONNECTOR_BOUNDARY_GAP;
        if (toY <= fromY) return;
        const midY = fromY + Math.max(18, (toY - fromY) * 0.5);
        let relationClass = '';
        if (relations) {
            if (continuation.nodeId === selectedNode) {
                relationClass = ' tree-connector-lineage-descendant';
            } else if (relations.ancestors.has(continuation.nodeId) || relations.descendants.has(continuation.nodeId)) {
                relationClass = ' tree-connector-lineage-ancestor';
            } else {
                relationClass = ' tree-connector-muted';
            }
        }
        addPath(
            `M ${from.x} ${fromY} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${toY}`,
            `tree-connector tree-connector-continuation${relationClass}`
        );
    });

    const edgeRoutes = [];
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
        const to = getAnchor(
            `[data-node-id="${escapedTo}"][data-generation="${edge.generation}"]`,
            'top'
        ) || getAnchor(`[data-node-id="${escapedTo}"].evolution-tree-node`, 'top');
        if (!from || !to) return;
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
        edgeRoutes.push({
            edge,
            generation: edge.generation,
            from,
            to,
            className
        });
    });

    assignTreeConnectorTracks(edgeRoutes).forEach((route) => {
        addPath(buildTreeSCurvePath(route.from, route.to, route.track), route.className);
    });
}

function compareRankMetricDesc(left, right) {
    const leftFinite = Number.isFinite(left);
    const rightFinite = Number.isFinite(right);
    if (leftFinite && rightFinite) return right - left;
    if (leftFinite) return -1;
    if (rightFinite) return 1;
    return 0;
}

function normalizeRankMetric(metric) {
    return ['fitness', 'elo', 'diversity'].includes(metric) || (typeof metric === 'string' && metric.startsWith('eval:')) ? metric : 'fitness';
}

function normalizeRankScope(scope) {
    return scope === 'current' ? 'current' : 'history';
}

function normalizeRankOptions(options = {}) {
    if (Number.isInteger(options)) {
        return {
            sortMetric: 'fitness',
            scope: 'current',
            generationIndex: options
        };
    }
    const config = options && typeof options === 'object' ? options : {};
    const defaultSortMetric = typeof rankedSortMetric !== 'undefined' ? rankedSortMetric : 'fitness';
    const defaultScope = typeof rankedScope !== 'undefined' ? rankedScope : 'history';
    return {
        sortMetric: normalizeRankMetric(config.sortMetric ?? defaultSortMetric),
        scope: normalizeRankScope(config.scope ?? defaultScope),
        generationIndex: Number.isInteger(config.generationIndex) ? config.generationIndex : null
    };
}

function getRankGenerationSpan(firstGeneration, lastGeneration) {
    if (!Number.isInteger(firstGeneration) && !Number.isInteger(lastGeneration)) return 'Gen -';
    if (firstGeneration === lastGeneration || !Number.isInteger(lastGeneration)) {
        return `Gen ${firstGeneration}`;
    }
    return `Gen ${firstGeneration}-${lastGeneration}`;
}

function getRankMetricRanks(rows, metric) {
    const values = (rows || [])
        .map(row => row?.[metric])
        .filter(value => Number.isFinite(value));
    return {
        values,
        populationSize: values.length
    };
}

function getRankDesc(value, values) {
    if (!Number.isFinite(value) || !values.length) return null;
    const sortedValues = [...values].sort((a, b) => b - a);
    const rankIndex = sortedValues.findIndex(candidate => candidate <= value);
    return rankIndex >= 0 ? rankIndex + 1 : sortedValues.length;
}

function getRankRowFlags(row) {
    const idea = row?.idea || {};
    const flags = [];
    if (idea?.oracle_generated) flags.push('Oracle');
    if (idea?.migration_event) flags.push('Migrated');
    if (idea?.survivor_refined) flags.push('Refined');
    if (idea?.elite_selected || idea?.elite_selected_source) flags.push('Creative');
    if (row?.occurrenceCount > 1) flags.push('Survivor');
    return flags;
}

function buildEvolutionRankRows(gens, options = {}) {
    const history = Array.isArray(gens) ? gens : [];
    if (history.length === 0) return [];

    const config = normalizeRankOptions(options);
    const selectedGeneration = config.generationIndex !== null
        ? Math.max(0, Math.min(history.length - 1, config.generationIndex))
        : history.length - 1;
    const latestGeneration = Math.max(0, history.length - 1);
    const rowsById = new Map();

    history.forEach((generation, generationIndex) => {
        if (!Array.isArray(generation)) return;
        generation.forEach((idea, ideaIndex) => {
            const id = getIdeaStableId(idea, generationIndex, ideaIndex);
            const ideaData = getIdeaData(idea || {});
            const existing = rowsById.get(id);
            const next = existing || {
                rank: 0,
                id,
                title: ideaData.title || 'Untitled',
                content: ideaData.content || '',
                idea,
                firstGeneration: generationIndex,
                lastGeneration: generationIndex,
                generationIndex,
                firstIndex: ideaIndex,
                ideaIndex,
                generations: [],
                occurrenceCount: 0,
                islandId: getIdeaIslandId(idea),
                fitness: null,
                elo: null,
                diversity: null,
                metricRanks: {}
            };
            next.title = ideaData.title || next.title || 'Untitled';
            next.content = ideaData.content || next.content || '';
            next.idea = idea;
            next.lastGeneration = generationIndex;
            next.generationIndex = generationIndex;
            next.ideaIndex = ideaIndex;
            next.occurrenceCount += 1;
            next.generations.push(generationIndex);
            next.islandId = getIdeaIslandId(idea) ?? next.islandId;
            next.fitness = getTreeMetricValue(idea, 'fitness');
            next.elo = getTreeMetricValue(idea, 'elo');
            next.diversity = getTreeMetricValue(idea, 'diversity');
            if (config.sortMetric.startsWith('eval:')) next[config.sortMetric] = getTreeMetricValue(idea, config.sortMetric);
            rowsById.set(id, next);
        });
    });

    const allRows = Array.from(rowsById.values()).map((row) => ({
        ...row,
        generationSpan: getRankGenerationSpan(row.firstGeneration, row.lastGeneration),
        isCurrent: row.generations.includes(latestGeneration),
        flags: getRankRowFlags(row)
    }));

    [...new Set(['fitness', 'elo', 'diversity', config.sortMetric])].forEach((metric) => {
        const stats = getRankMetricRanks(allRows, metric);
        allRows.forEach((row) => {
            row.metricRanks[metric] = {
                rank: getRankDesc(row[metric], stats.values),
                populationSize: stats.populationSize
            };
        });
    });

    const fallbackMetrics = [config.sortMetric, 'fitness', 'elo', 'diversity']
        .filter((metric, index, list) => list.indexOf(metric) === index);
    const scopedRows = config.scope === 'current'
        ? allRows.filter(row => row.generations.includes(selectedGeneration))
        : allRows;

    return scopedRows.sort((left, right) => {
        for (const metric of fallbackMetrics) {
            const comparison = compareRankMetricDesc(left[metric], right[metric]);
            if (comparison !== 0) return comparison;
        }
        return left.title.localeCompare(right.title) || left.firstGeneration - right.firstGeneration || left.ideaIndex - right.ideaIndex;
    }).map((row, index) => {
        const rankContext = row.metricRanks?.[config.sortMetric] || { rank: null, populationSize: 0 };
        return {
            ...row,
            rank: index + 1,
            sortMetric: config.sortMetric,
            sortValue: row[config.sortMetric],
            metricRank: rankContext.rank,
            metricPopulationSize: rankContext.populationSize,
            scope: config.scope,
            selectedGeneration
        };
    });
}

function formatRankMetric(value, digits = 2) {
    if (!Number.isFinite(value)) return '-';
    return value.toFixed(digits);
}

function formatRankMetricLabel(metric) {
    if (metric.startsWith('eval:')) return metric.slice(5).split('.').slice(1).join('.').replace(/_/g, ' ');
    if (metric === 'elo') return 'ELO';
    return metric.charAt(0).toUpperCase() + metric.slice(1);
}

function formatRankMetricValue(metric, value) {
    return formatRankMetric(value, metric === 'elo' ? 0 : 3);
}

function renderRankScopeLabel(scope, selectedGeneration) {
    return scope === 'current'
        ? `Current generation ${selectedGeneration}`
        : 'All historical ideas';
}

function renderRankButtonClasses(currentValue, buttonValue) {
    return currentValue === buttonValue ? 'active' : '';
}

function renderRankMetricPill(row, metric) {
    const isActive = row.sortMetric === metric;
    const rankContext = row.metricRanks?.[metric] || {};
    const globalRank = rankContext.rank && rankContext.populationSize
        ? `#${rankContext.rank}/${rankContext.populationSize}`
        : '-';
    return `
        <span class="evolution-rank-metric${isActive ? ' active' : ''}">
            <span>${escapeHtml(formatRankMetricLabel(metric))}</span>
            <strong>${escapeHtml(formatRankMetricValue(metric, row[metric]))}</strong>
            <small>${escapeHtml(globalRank)}</small>
        </span>
    `;
}

function renderRankIslandBadge(row) {
    if (!Number.isInteger(row.islandId)) {
        return '<span class="evolution-rank-island evolution-rank-island-empty">No island</span>';
    }
    return `
        <span class="evolution-rank-island island-badge">
            <span class="evolution-rank-island-dot"></span>
            <span>Island ${row.islandId + 1}</span>
        </span>
    `;
}

function renderEvolutionRankedList(gens) {
    const container = document.getElementById('evolution-ranked-container');
    if (!container) return;

    const rows = buildEvolutionRankRows(gens, {
        sortMetric: rankedSortMetric,
        scope: rankedScope
    });
    const selectedGeneration = rows.length > 0 ? rows[0].selectedGeneration : Math.max(0, (gens || []).length - 1);
    if (rows.length === 0) {
        container.innerHTML = '<div class="evolution-ranked-empty">No population metrics available yet.</div>';
        return;
    }

    const evaluatorMetrics = window.IdeaWorkbench?.metricEvidence(rows[0]?.idea, currentEvolutionData?.config?.evaluation || {}) || [];
    container.innerHTML = `
        <div class="evolution-ranked-header">
            <div>
                <div class="evolution-ranked-kicker">${escapeHtml(renderRankScopeLabel(rankedScope, selectedGeneration))}</div>
                <h3>Ranked Ideas</h3>
            </div>
            <div class="evolution-ranked-controls">
                <div class="evolution-ranked-segment" role="group" aria-label="Ranked idea scope">
                    <button type="button" class="${renderRankButtonClasses(rankedScope, 'history')}" data-rank-scope="history">All history</button>
                    <button type="button" class="${renderRankButtonClasses(rankedScope, 'current')}" data-rank-scope="current">Current gen</button>
                </div>
                <div class="evolution-ranked-segment" role="group" aria-label="Ranked idea sort metric">
                    <button type="button" class="${renderRankButtonClasses(rankedSortMetric, 'fitness')}" data-rank-sort="fitness">Fitness</button>
                    <button type="button" class="${renderRankButtonClasses(rankedSortMetric, 'elo')}" data-rank-sort="elo">ELO</button>
                    <button type="button" class="${renderRankButtonClasses(rankedSortMetric, 'diversity')}" data-rank-sort="diversity">Diversity</button>
                    ${evaluatorMetrics.filter(metric => !['quality_elo.quality_elo', 'diversity.diversity'].includes(metric.key)).map(metric => `<button type="button" class="${renderRankButtonClasses(rankedSortMetric, `eval:${metric.key}`)}" data-rank-sort="${escapeHtml(`eval:${metric.key}`)}" title="${escapeHtml(`${metric.evaluator} · higher is better`)}">${escapeHtml(metric.name)}</button>`).join('')}
                </div>
                <div class="evolution-ranked-count">${rows.length} ideas</div>
            </div>
        </div>
        <div class="evolution-ranked-list" role="list" aria-label="Ranked evolution ideas">
            ${rows.map(row => `
                <div role="listitem"><button type="button" class="evolution-rank-row ${Number.isInteger(row.islandId) ? `island-${row.islandId % 5}` : ''}" data-rank-id="${escapeHtml(row.id)}" data-rank-index="${row.rank - 1}">
                    <span class="evolution-rank-number">
                        <strong>#${row.rank}</strong>
                        <small>${escapeHtml(formatRankMetricLabel(row.sortMetric))}</small>
                    </span>
                    <span class="evolution-rank-main">
                        <span class="evolution-rank-title">${escapeHtml(row.title)}</span>
                        <span class="evolution-rank-preview">${escapeHtml((row.content || '').replace(/\s+/g, ' ').trim().slice(0, 180))}</span>
                        <span class="evolution-rank-meta">
                            ${renderRankIslandBadge(row)}
                            <span class="evolution-rank-chip"><i class="fas fa-code-branch"></i>${escapeHtml(row.generationSpan)}</span>
                            <span class="evolution-rank-chip ${row.isCurrent ? 'rank-current' : 'rank-historical'}">${row.isCurrent ? 'Current' : 'Historical'}</span>
                            ${row.flags.map(flag => `<span class="evolution-rank-chip">${escapeHtml(flag)}</span>`).join('')}
                        </span>
                    </span>
                    <span class="evolution-rank-score">
                        <span>Global ${escapeHtml(formatRankMetricLabel(row.sortMetric))}</span>
                        <strong>${row.metricRank && row.metricPopulationSize ? `#${row.metricRank}` : '-'}</strong>
                        <small>${row.metricPopulationSize ? `of ${row.metricPopulationSize}` : 'No metric'}</small>
                    </span>
                    <span class="evolution-rank-metrics">
                        ${renderRankMetricPill(row, 'fitness')}
                        ${renderRankMetricPill(row, 'elo')}
                        ${renderRankMetricPill(row, 'diversity')}
                        ${row.sortMetric.startsWith('eval:') ? renderRankMetricPill(row, row.sortMetric) : ''}
                    </span>
                </button></div>
            `).join('')}
        </div>
    `;

    container.querySelectorAll('[data-rank-sort]').forEach((button) => {
        button.addEventListener('click', () => {
            rankedSortMetric = normalizeRankMetric(button.dataset.rankSort);
            renderEvolutionRankedList(gens);
        });
    });
    container.querySelectorAll('[data-rank-scope]').forEach((button) => {
        button.addEventListener('click', () => {
            rankedScope = normalizeRankScope(button.dataset.rankScope);
            renderEvolutionRankedList(gens);
        });
    });
    container.querySelectorAll('.evolution-rank-row').forEach((button) => {
        const rowIndex = Number(button.dataset.rankIndex);
        const row = rows[rowIndex];
        if (!row) return;
        button.addEventListener('click', () => {
            showIdeaModal(getIdeaData(row.idea || {}));
        });
    });
}

// Gene observations describe candidates; this view never changes selection or calls a model.
function getGeneObservation(idea) {
    const raw = idea?.gene_observation ?? idea?.idea?.gene_observation ?? null;
    const projection = idea?.gene_projection ?? idea?.idea?.gene_projection;
    if (raw?.status === 'ok' && projection?.status === 'ok' && projection.version === 1
        && typeof raw.fingerprint === 'string' && projection.raw_fingerprint === raw.fingerprint
        && Array.isArray(projection.genes) && typeof projection.genotype === 'string') {
        return {...raw, ...projection, raw_genotype: raw.genotype, mapping_status: 'ok', consolidated: true};
    }
    return raw;
}

function buildGeneCatalogueView(history, catalogue, scope = 'current') {
    const all = Array.isArray(history) ? history : [];
    const requested = scope.startsWith('generation:') ? Number(scope.split(':')[1]) : all.length - 1;
    const generationIndex = Number.isInteger(requested) ? Math.max(0, Math.min(all.length - 1, requested)) : all.length - 1;
    const candidates = new Map();
    all.forEach((population, index) => {
        if (scope !== 'history' && index !== generationIndex) return;
        (Array.isArray(population) ? population : []).forEach((idea, slot) => {
            const id = idea?.id ?? idea?.idea?.id ?? `unknown:${index}:${slot}`;
            const previous = candidates.get(String(id));
            candidates.set(String(id), { id: String(id), idea, generation: index, firstGeneration: previous?.firstGeneration ?? index });
        });
    });
    const rows = new Map();
    (Array.isArray(catalogue?.genes) ? catalogue.genes : []).forEach(gene => {
        if (typeof gene?.id !== 'string' || rows.has(gene.id)) return;
        rows.set(gene.id, { ...gene, label: String(gene.label || gene.id), carriers: [] });
    });
    const unclassifiedRows = new Map();
    let analysed = 0, missing = 0, failed = 0, unclassified = 0, unclassifiedGenes = 0, empty = 0, mappingFailed = 0;
    candidates.forEach(candidate => {
        const observation = getGeneObservation(candidate.idea);
        if (!observation) { missing++; return; }
        if (observation.status !== 'ok' || !Array.isArray(observation.genes)) { failed++; return; }
        analysed++;
        if (!observation.genes.length) empty++;
        const memberships = new Set();
        const rawMemberships = new Set();
        let hasUnclassified = false;
        observation.genes.forEach(gene => {
            if (typeof gene?.gene_id === 'string' && rows.has(gene.gene_id)) memberships.add(gene.gene_id);
            else {
                hasUnclassified = true;
                unclassifiedGenes++;
                // Group exact saved wording for display only; never infer a canonical ID.
                const label = typeof gene?.text === 'string' && gene.text.trim() ? gene.text : 'Unnamed extracted gene';
                if (!unclassifiedRows.has(label)) unclassifiedRows.set(label, { label, carriers: [] });
                rawMemberships.add(label);
            }
        });
        if (hasUnclassified) unclassified++;
        if (hasUnclassified && observation.mapping_status === 'failed') mappingFailed++;
        memberships.forEach(id => rows.get(id).carriers.push(candidate));
        rawMemberships.forEach(label => unclassifiedRows.get(label).carriers.push(candidate));
    });
    return {
        rows: [...rows.values()].sort((a, b) => b.carriers.length - a.carriers.length || a.label.localeCompare(b.label)),
        unclassifiedRows: [...unclassifiedRows.values()].sort((a, b) => b.carriers.length - a.carriers.length || a.label.localeCompare(b.label)),
        total: candidates.size, analysed, missing, failed, unclassified, unclassifiedGenes, empty, mappingFailed, generationIndex,
    };
}

// Snapshot membership, not inferred ancestry. Never fill a missing observation with a later one.
function buildGeneHistory(history, catalogue, fitnessValue = (idea, generation) =>
    typeof getFitnessPlotMetricValue === 'function' ? getFitnessPlotMetricValue(idea, 'fitness', generation) : null) {
    const snapshots = (Array.isArray(history) ? history : []).map((_, index) =>
        buildGeneCatalogueView(history, catalogue, `generation:${index}`));
    const seen = new Set();
    const rows = (catalogue?.genes || []).filter((gene, index, genes) =>
        typeof gene?.id === 'string' && genes.findIndex(other => other.id === gene.id) === index)
        .map(gene => ({ ...gene, label: String(gene.label || gene.id), points: [], unique: new Set(), first: null, peak: 0 }));
    const means = candidates => {
        const values = candidates.map(c => fitnessValue(c.idea, c.generation)).filter(v => typeof v === 'number' && Number.isFinite(v));
        return { mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null, measured: values.length };
    };
    const summaries = snapshots.map((view, generation) => {
        const byId = new Map(view.rows.map(row => [row.id, row]));
        const active = new Set(view.rows.filter(row => row.carriers.length).map(row => row.id));
        const added = [...active].filter(id => !seen.has(id));
        added.forEach(id => seen.add(id));
        const previous = generation ? snapshots[generation - 1] : null;
        const previousRows = new Map((previous?.rows || []).map(row => [row.id, row]));
        const complete = view.total > 0 && view.analysed === view.total && view.unclassified === 0;
        const previousComplete = previous && previous.total > 0 && previous.analysed === previous.total && previous.unclassified === 0;
        const lost = previousComplete && complete ? previous.rows.filter(row => row.carriers.length && !active.has(row.id)).length : null;
        const candidates = new Map();
        (history[generation] || []).forEach((idea, slot) => candidates.set(String(idea?.id ?? idea?.idea?.id ?? `unknown:${slot}`), { idea, generation }));
        const populationFitness = means([...candidates.values()]);
        rows.forEach(row => {
            const carriers = byId.get(row.id)?.carriers || [];
            const priorIds = new Set((previousRows.get(row.id)?.carriers || []).map(c => c.id));
            const survived = carriers.filter(c => priorIds.has(c.id)).length;
            const parentLinked = carriers.filter(c => !priorIds.has(c.id)
                && (c.idea.parent_ids || []).some(id => priorIds.has(String(id)))).length;
            carriers.forEach(c => row.unique.add(c.id));
            if (carriers.length && row.first === null) row.first = generation;
            row.peak = Math.max(row.peak, carriers.length);
            row.points.push({ generation, count: carriers.length, analysed: view.analysed, total: view.total,
                survived, parentLinked, connected: survived + parentLinked > 0,
                complete, share: view.analysed ? carriers.length / view.analysed : null,
                ...means(carriers), populationMean: populationFitness.mean, populationMeasured: populationFitness.measured });
        });
        return { generation, active: active.size, added: added.length, lost, cumulative: seen.size,
            total: view.total, analysed: view.analysed, complete, ...populationFitness };
    });
    // Retired vocabulary stays in the catalogue for provenance, but empty
    // tracks must not consume space in the all-genes history overview.
    return { rows: rows.filter(row => row.unique.size).map(row => ({ ...row, unique: row.unique.size })), summaries };
}

function orderGeneThreads(history) {
    return [...history.rows].sort((a, b) => (a.first ?? Infinity) - (b.first ?? Infinity)
        || (b.points[0]?.share || 0) - (a.points[0]?.share || 0)
        || b.peak - a.peak || b.unique - a.unique || a.label.localeCompare(b.label));
}

// Identity accents only: stable across ordering, search and generation changes.
function geneThreadColor(id) {
    const palette = ['#9364a7', '#408a83', '#c17c53', '#6a82b2', '#b56582', '#8a904b', '#527f9a', '#ac7448'];
    let hash = 0;
    for (const char of String(id)) hash = (Math.imul(hash, 31) + char.codePointAt(0)) | 0;
    return palette[(hash >>> 0) % palette.length];
}

function buildGeneThreads(history, width = 900, height = 430) {
    const rows = orderGeneThreads(history);
    const left = 12, right = width - 14, top = 30, bottom = height - 28;
    const step = (right - left) / Math.max(1, history.summaries.length);
    const lane = (bottom - top) / Math.max(1, rows.length);
    // Each snapshot owns a full band, including the first and last snapshots.
    const x = gen => left + (gen + .5) * step;
    const tracks = rows.map((row, index) => {
        const y = top + (index + .5) * lane;
        const half = point => Math.max(.18, Math.min(3, lane * .4) * (point.share || 0));
        const runs = []; let run = [];
        row.points.forEach(point => {
            if (run.length && !point.connected) { runs.push(run); run = []; }
            if (point.count) run.push(point);
            else if (run.length) { runs.push(run); run = []; }
        });
        if (run.length) runs.push(run);
        const path = runs.map(points => {
            const first = points[0], last = points.at(-1);
            // Only trim adjacent occupied bands without an observed carrier link.
            // Keep genuine absences empty and surviving/parent-linked runs seamless.
            const gap = Math.min(3, step * .12);
            const start = Math.max(left, x(first.generation) - step / 2)
                + (row.points[first.generation - 1]?.count ? gap : 0);
            const end = Math.min(right, x(last.generation) + step / 2)
                - (row.points[last.generation + 1]?.count ? gap : 0);
            const upper = [[start, y - half(first)], ...points.map(p => [x(p.generation), y - half(p)]), [end, y - half(last)]];
            const lower = [[end, y + half(last)], ...[...points].reverse().map(p => [x(p.generation), y + half(p)]), [start, y + half(first)]];
            // Smooth width changes without moving lanes or overshooting measured occupancy.
            const edge = points => points.slice(1).map((p, i) => {
                const previous = points[i], middle = (previous[0] + p[0]) / 2;
                return `C${middle.toFixed(2)},${previous[1].toFixed(2)} ${middle.toFixed(2)},${p[1].toFixed(2)} ${p.map(n => n.toFixed(2)).join(',')}`;
            }).join('');
            return `M${upper[0].map(n => n.toFixed(2)).join(',')}${edge(upper)}L${lower[0].map(n => n.toFixed(2)).join(',')}${edge(lower)}Z`;
        }).join('');
        const lastSeen = row.points.findLastIndex(point => point.count > 0);
        return { row, y, path, lastSeen, start: row.first === null ? left : x(row.first) - step / 2, end: lastSeen < 0 ? right : x(lastSeen) + step / 2 };
    });
    // Explicit ticks keep 100+ generations on the same axis without a column per generation.
    const generations = [...new Set(Array.from({length: Math.min(6, history.summaries.length)}, (_, i) =>
        Math.round(i * (history.summaries.length - 1) / Math.max(1, Math.min(6, history.summaries.length) - 1))))];
    return { tracks, left, right, top, bottom, width, height, lane, step, generations, x };
}

function geneTrendPath(points, value, width = 210, height = 90, min = 0, max = 1) {
    let path = '', connected = false;
    points.forEach((point, index) => {
        const v = value(point);
        if (typeof v !== 'number' || !Number.isFinite(v)) { connected = false; return; }
        const x = geneTrendX(index, points.length, width);
        const y = 8 + (height - 24) * (1 - (v - min) / Math.max(.0001, max - min));
        path += `${connected ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
        connected = true;
    });
    return path;
}

function geneTrendX(index, count, width = 210) {
    return count === 1 ? (24 + width - 8) / 2 : 24 + index * (width - 32) / Math.max(1, count - 1);
}

function renderGeneTrend(row, metric, generation) {
    const occupancy = metric === 'occupancy';
    const value = point => occupancy ? (point.complete ? point.share : null) : point.mean;
    const values = row.points.flatMap(p => [value(p), ...(!occupancy ? [p.populationMean] : [])]).filter(v => typeof v === 'number' && Number.isFinite(v));
    const min = occupancy ? 0 : Math.min(0, ...values), max = occupancy ? 1 : Math.max(1, ...values);
    const selected = row.points[generation];
    const x = geneTrendX(generation, row.points.length);
    const exact = occupancy
        ? selected?.complete ? `${selected.count} / ${selected.total}` : `${selected?.count ?? 0} known`
        : !selected?.count || selected.mean === null ? '—' : selected.mean.toFixed(3);
    const note = occupancy ? selected?.complete ? '' : `${selected?.analysed ?? 0}/${selected?.total ?? 0} analysed · mapping incomplete`
        : selected?.count ? `${selected.measured}/${selected.count} scored · population ${selected.populationMean === null ? '—' : selected.populationMean.toFixed(3)}` : 'No carriers';
    const description = row.points.map(p => `Gen ${p.generation}: ${occupancy ? p.complete ? `${p.count} of ${p.total}` : `${p.count} known carriers; incomplete analysis` : p.mean === null ? 'fitness unavailable' : `${p.mean.toFixed(3)}, ${p.measured} of ${p.count} scored`}`).join('; ');
    const latest = row.points[row.points.length - 1];
    const latestCount = latest.complete ? `${latest.count} / ${latest.total}` : `${latest.count} known`;
    const counts = occupancy ? `<div class="gene-occupancy-counts">
        ${generation !== row.points.length - 1 ? `<span title="Carriers in selected generation ${generation}"><small>Gen ${generation}</small><strong>${escapeHtml(exact)}</strong></span>` : ''}
        <span title="Carriers in latest generation ${latest.generation}${latest.complete ? '' : '; mapping incomplete'}"><small>Now</small><strong>${escapeHtml(latestCount)}</strong></span>
        <span title="Unique observed carriers across all generations; each idea counted once"><small>Ever</small><strong>${row.unique}</strong></span>
    </div>` : '';
    return `<section class="gene-trend"><div class="gene-trend-label"><h5>${occupancy ? 'Occupancy' : 'Carrier fitness'}</h5>${occupancy ? '' : `<strong>${escapeHtml(exact)}</strong>`}</div>${counts}
        <svg viewBox="0 0 210 112" role="img" aria-label="${occupancy ? 'Occupancy' : 'Mean carrier fitness'} by generation"><title>${escapeHtml(description)}</title>
            <text x="0" y="12">${occupancy ? '100%' : max.toFixed(1)}</text><text x="9" y="99">${occupancy ? '0' : min.toFixed(1)}</text>
            <path class="gene-trend-axis" d="M24,8V96H202"/>
            ${!occupancy ? `<path class="gene-trend-population" d="${geneTrendPath(row.points, p => p.populationMean, 210, 112, min, max)}"/>${row.points.map((p, i) => {
                if (typeof p.populationMean !== 'number' || !Number.isFinite(p.populationMean)) return '';
                const px = geneTrendX(i, row.points.length), py = 8 + 88 * (1 - (p.populationMean - min) / (max - min));
                return `<path class="gene-trend-population" d="M${px - 2},${py}h4"><title>Population: ${p.populationMean.toFixed(3)}</title></path>`;
            }).join('')}` : ''}
            <path class="gene-trend-line" d="${geneTrendPath(row.points, value, 210, 112, min, max)}"/>
            ${row.points.map((p, i) => {
                const val = value(p);
                if (typeof val !== 'number' || !Number.isFinite(val)) return '';
                return `<circle class="gene-trend-dot" cx="${geneTrendX(i, row.points.length)}" cy="${8 + 88 * (1 - (val - min) / (max - min))}" r="${i === generation ? 3.5 : 2}"><title>${escapeHtml(`Gen ${i}: ${val.toFixed(3)}`)}</title></circle>`;
            }).join('')}
            <path class="gene-trend-cursor" d="M${x},8V96"/>
            ${row.points.length === 1 ? '<text x="113" y="110" text-anchor="middle">Gen 0</text>' : `<text x="24" y="110">0</text><text x="202" y="110" text-anchor="end">${row.points.length - 1}</text>`}
        </svg>${note ? `<p>${escapeHtml(note)}</p>` : ''}</section>`;
}

function renderGeneThreadDetail(row, generation) {
    const point = row.points[generation];
    const continuity = generation > 0 && point?.count
        ? `<p class="gene-continuity" title="Links to carriers in the preceding snapshot, using saved IDs and parent links through the current gene vocabulary. Shared parent–child annotations do not prove the gene was selected during breeding. Gaps mean no observed link; missing mappings can also cause gaps.">${point.survived} surviving · ${point.parentLinked} parent-linked</p>` : '';
    return `<p class="gene-trend-subject">${escapeHtml(row.label)}</p>${renderGeneTrend(row, 'occupancy', generation)}${continuity}${renderGeneTrend(row, 'fitness', generation)}`;
}

// Keep one entry per carrier, including ideas that have since left the population.
// Use its latest carrying snapshot for DNA; never invent membership in missing data.
function geneHistoryCarriers(gens, geneId) {
    const carriers = new Map();
    gens.forEach((population, generation) => {
        (population || []).forEach((idea, slot) => {
            const id = String(idea?.id ?? idea?.idea?.id ?? `unknown:${generation}:${slot}`);
            const candidate = {id, idea, generation};
            if (!geneStoryIds(candidate).has(geneId)) return;
            const previous = carriers.get(id);
            carriers.set(id, {...candidate, firstGeneration: previous?.firstGeneration ?? generation});
        });
    });
    return [...carriers.values()].sort((a, b) => a.firstGeneration - b.firstGeneration);
}

function geneStoryIds(candidate) {
    const observation = getGeneObservation(candidate?.idea);
    return new Set(observation?.status === 'ok' && Array.isArray(observation.genes)
        ? observation.genes.map(g => g.gene_id).filter(id => typeof id === 'string') : []);
}

function geneHistoryIdeas(gens) {
    const ideas = new Map();
    gens.forEach((population, generation) => (population || []).forEach((idea, slot) => {
        const id = String(idea?.id ?? idea?.idea?.id ?? `unknown:${generation}:${slot}`);
        ideas.set(id, {id, idea, generation, firstGeneration: ideas.get(id)?.firstGeneration ?? generation});
    }));
    return [...ideas.values()].sort((a, b) => a.firstGeneration - b.firstGeneration);
}

function geneExplorerCandidate(gens, state) {
    if (!state.candidateId) return null;
    const snapshot = gens[state.candidateGeneration]?.find(idea => String(idea?.id ?? idea?.idea?.id) === state.candidateId);
    return snapshot ? {id: state.candidateId, idea: snapshot, generation: state.candidateGeneration}
        : geneHistoryIdeas(gens).find(candidate => candidate.id === state.candidateId);
}

function geneExplorerCandidates(gens, row, state) {
    const query = (state.ideaSearch || '').trim().toLocaleLowerCase();
    if (query) return geneHistoryIdeas(gens).filter(candidate =>
        String(getIdeaData(candidate.idea).title || 'Untitled').toLocaleLowerCase().includes(query));
    // Hovering a gene previews it without replacing the selected idea's list.
    return geneHistoryCarriers(gens, (state.candidateId && state.candidateGeneId) || row.id);
}

function changeGeneExplorerFocus(state, id, generation, mode = 'hover') {
    if (mode === 'hover' && state.pinnedGeneId) return false;
    state.focusId = id;
    state.focusGeneration = generation;
    state.previewGeneId = state.candidateId && mode !== 'pin' ? id : null;
    if (mode === 'pin') {
        state.pinnedGeneId = id; state.candidateId = null;
        state.candidateGeneId = null; state.candidateGeneration = null;
    }
    if (mode === 'keyboard' && state.pinnedGeneId) state.pinnedGeneId = id;
    return true;
}

function clearGeneExplorerSelection(state) {
    state.pinnedGeneId = null;
    state.candidateId = null;
    state.candidateGeneId = null;
    state.candidateGeneration = null;
    state.previewGeneId = null;
}

function endGeneExplorerPreview(state) {
    if (!state.previewGeneId) return false;
    state.previewGeneId = null;
    if (state.candidateGeneId) state.focusId = state.candidateGeneId;
    return true;
}

function renderGeneStories(candidates, row, state) {
    const searching = Boolean((state.ideaSearch || '').trim());
    return `<div class="gene-stories-heading"><h5>Ideas <span title="${searching ? 'Matching ideas across all history' : 'Unique carriers across all history'}">${candidates.length}${searching ? ' found' : ''}</span></h5></div>
        <ul class="gene-story-list">${candidates.map((candidate, index) => {
            const title = getIdeaData(candidate.idea).title || 'Untitled';
            return `<li class="${geneStoryIds(candidate).has(row.id) ? 'is-carrier' : ''} ${state.candidateId === candidate.id ? 'is-selected' : ''}">
                <button type="button" class="gene-story-select" data-gene-story="${index}" aria-pressed="${state.candidateId === candidate.id}" title="${escapeHtml(title)}"><span class="gene-story-dot" aria-hidden="true"></span><span>${escapeHtml(title)}</span><span class="visually-hidden"> · highlight DNA</span></button>
                <small class="gene-idea-generation" title="${searching ? 'First observed' : 'First observed carrying this gene'} in generation ${candidate.firstGeneration}">G${candidate.firstGeneration}</small>
                <button type="button" class="gene-story-read" data-gene-story-read="${index}" aria-label="Read ${escapeHtml(title)}" title="Read idea">↗</button></li>`;
        }).join('')}</ul>${candidates.length ? '' : `<p class="gene-explanation">${searching ? 'No matching ideas.' : 'No observed carriers.'}</p>`}`;
}

function geneExplorerHeading(row, candidates, state) {
    const candidate = candidates.find(c => c.id === state.candidateId);
    if (candidate && state.previewGeneId) return {
        name: row.label,
        summary: `${geneStoryIds(candidate).has(row.id) ? 'In' : 'Not in'} selected idea · ${getIdeaData(candidate.idea).title || 'Untitled'}`
    };
    return candidate ? {name: getIdeaData(candidate.idea).title || 'Untitled', summary: getGeneObservation(candidate.idea)?.status === 'ok' ? (geneStoryIds(candidate).size ? `DNA · ${geneStoryIds(candidate).size} mapped genes highlighted` : 'Gene mapping pending · use ↗ to inspect raw genes') : 'Gene analysis unavailable'}
        : {name: row.label, summary: `${state.pinnedGeneId ? 'Pinned · ' : ''}${row.unique} unique carrier${row.unique === 1 ? '' : 's'}`};
}

function renderGeneHistory(history, state, gens = []) {
    if (!history.summaries.length || !history.rows.length) return '';
    const containerWidth = typeof document !== 'undefined' ? document.getElementById('evolution-genes-container')?.clientWidth || 1000 : 1000;
    const width = Math.max(260, containerWidth - (containerWidth <= 600 ? 0 : containerWidth <= 900 ? 180 : 440));
    const model = buildGeneThreads(history, width);
    const query = state.search.trim().toLocaleLowerCase();
    const matches = new Set(model.tracks.filter(t => [t.row.label, ...(t.row.aliases || [])].some(label => String(label).toLocaleLowerCase().includes(query))).map(t => t.row.id));
    if (state.pinnedGeneId && !history.rows.some(r => r.id === state.pinnedGeneId)) state.pinnedGeneId = null;
    const activeIndex = Math.max(0, model.tracks.findIndex(t => t.row.id === (state.pinnedGeneId || state.focusId || selectedGeneId)));
    const active = model.tracks[activeIndex].row;
    const generation = Math.max(0, Math.min(history.summaries.length - 1, state.focusGeneration ?? history.summaries.length - 1));
    const candidate = geneExplorerCandidate(gens, state);
    if (state.candidateId && !candidate) clearGeneExplorerSelection(state);
    if (candidate && !state.candidateGeneId) state.candidateGeneId = active.id;
    const candidates = geneExplorerCandidates(gens, active, state);
    const dna = geneStoryIds(candidate);
    state.focusId = active.id; state.focusGeneration = generation;
    const heading = geneExplorerHeading(active, candidate ? [candidate] : [], state);
    return `<section class="gene-history" aria-label="Gene history" style="--gene-active-color:${geneThreadColor(active.id)}">
        <div class="gene-thread-toolbar"><p><strong>${model.tracks.length} genes</strong> · ${history.summaries.length} generation${history.summaries.length === 1 ? '' : 's'}</p>
            <div class="gene-thread-actions"><button type="button" id="gene-thread-pin" aria-pressed="${Boolean(state.pinnedGeneId)}">${state.pinnedGeneId ? 'Pinned' : 'Pin gene'}</button><button type="button" id="gene-thread-clear" ${state.pinnedGeneId || state.candidateId ? '' : 'disabled'}>Clear <kbd>Esc</kbd></button>
            <label><span class="visually-hidden">Find a gene</span><input id="gene-history-search" class="form-control form-control-sm" type="search" value="${escapeHtml(state.search)}" placeholder="Find a gene…" list="gene-thread-options"><datalist id="gene-thread-options">${model.tracks.map(t => `<option value="${escapeHtml(t.row.label)}"></option>`).join('')}</datalist></label></div></div>
        <div class="gene-river-layout"><aside class="gene-stories" id="gene-thread-stories" aria-label="Ideas and their DNA">
            <label class="gene-idea-search"><span class="visually-hidden">Search all ideas by title</span><input id="gene-idea-search" class="form-control form-control-sm" type="search" value="${escapeHtml(state.ideaSearch || '')}" placeholder="Find an idea…" autocomplete="off"></label>
            <div id="gene-story-results">${renderGeneStories(candidates, active, state)}</div></aside><div class="gene-river-main">
            <div class="gene-thread-heading"><h4 id="gene-thread-name" aria-live="polite">${escapeHtml(heading.name)}</h4><p id="gene-thread-summary">${escapeHtml(heading.summary)}</p></div>
            <svg id="gene-thread-canvas" viewBox="0 0 ${model.width} ${model.height}" preserveAspectRatio="none" tabindex="0" role="group" aria-label="${model.tracks.length} gene paths. Arrow keys explore. Click or Enter pins a gene. Escape clears selection." style="height:${model.height}px">
                ${model.generations.map(gen => `<path class="gene-thread-grid" d="M${model.x(gen)},22V${model.bottom + 4}"/><text class="gene-thread-axis-label" x="${model.x(gen)}" y="14" text-anchor="middle">Gen ${gen}</text>`).join('')}
                ${model.tracks.map((track, index) => `<g data-gene-track="${index}" style="--gene-color:${geneThreadColor(track.row.id)}" class="gene-thread ${matches.has(track.row.id) ? '' : 'is-search-dimmed'} ${index === activeIndex ? 'is-active' : ''} ${state.candidateId ? dna.has(track.row.id) ? 'is-dna' : 'is-dna-dimmed' : ''}"><title>${escapeHtml(track.row.label)} · Gaps: no observed carrier link</title><path class="gene-thread-ribbon" d="${track.path}"/></g>`).join('')}
                <path id="gene-thread-time-guide" class="gene-thread-time-guide" d="M${model.x(generation)},24V${model.bottom}"/>
            </svg><p class="gene-thread-hint" id="gene-thread-hint">${query ? `${matches.size} matching · all ${model.tracks.length} visible` : ''}</p>
        </div><aside class="gene-thread-detail" id="gene-thread-detail" aria-label="Selected gene trends">${renderGeneThreadDetail(active, generation)}</aside></div>
        <p class="gene-history-footer">Click a gene to pin · Select an idea to see its DNA <span title="Each gene keeps the same accent color across views. Colors repeat; they do not encode generation or fitness.">Color = gene · Width = occupancy</span></p>
    </section>`;
}

function bindGeneThreads(container, history, state, gens) {
    const canvas = container.querySelector('#gene-thread-canvas');
    if (!canvas) return;
    const root = container.querySelector('.gene-history');
    const model = buildGeneThreads(history, Number(canvas.viewBox.baseVal.width));
    const groups = [...canvas.querySelectorAll('[data-gene-track]')];
    let activeIndex = Math.max(0, model.tracks.findIndex(t => t.row.id === state.focusId));
    let generation = state.focusGeneration ?? history.summaries.length - 1;
    const sync = () => {
        const row = model.tracks[activeIndex].row;
        const candidates = geneExplorerCandidates(gens, row, state);
        const candidate = geneExplorerCandidate(gens, state);
        const dna = geneStoryIds(candidate);
        groups.forEach((group, index) => {
            group.classList.toggle('is-active', index === activeIndex);
            group.classList.toggle('is-dna', Boolean(state.candidateId) && dna.has(model.tracks[index].row.id));
            group.classList.toggle('is-dna-dimmed', Boolean(state.candidateId) && !dna.has(model.tracks[index].row.id));
        });
        if (groups[activeIndex]) canvas.append(groups[activeIndex]);
        const heading = geneExplorerHeading(row, candidate ? [candidate] : [], state);
        container.querySelector('#gene-thread-name').textContent = heading.name;
        container.querySelector('#gene-thread-summary').textContent = heading.summary;
        root.style.setProperty('--gene-active-color', geneThreadColor(row.id));
        container.querySelector('#gene-thread-time-guide').setAttribute('d', `M${model.x(generation)},24V${model.bottom}`);
        container.querySelector('#gene-thread-detail').innerHTML = renderGeneThreadDetail(row, generation);
        const stories = container.querySelector('#gene-thread-stories');
        const scroll = stories.querySelector('.gene-story-list')?.scrollTop || 0;
        const results = stories.querySelector('#gene-story-results');
        const html = renderGeneStories(candidates, row, state);
        if (results.innerHTML !== html) results.innerHTML = html;
        if (stories.querySelector('.gene-story-list')) stories.querySelector('.gene-story-list').scrollTop = scroll;
        const pin = container.querySelector('#gene-thread-pin');
        pin.textContent = state.pinnedGeneId ? 'Pinned' : 'Pin gene';
        pin.setAttribute('aria-pressed', String(Boolean(state.pinnedGeneId)));
        container.querySelector('#gene-thread-clear').disabled = !state.pinnedGeneId && !state.candidateId;
        canvas.setAttribute('aria-label', `${row.label}, Gen ${generation}: ${row.points[generation].count}/${row.points[generation].analysed} carriers. ${state.candidateId ? `${dna.size} idea genes highlighted.` : state.pinnedGeneId ? 'Pinned.' : ''} Arrow keys explore; Enter pins; Escape clears.`);
    };
    const update = (index, gen, mode = 'hover') => {
        index = Math.max(0, Math.min(model.tracks.length - 1, index));
        gen = Math.max(0, Math.min(history.summaries.length - 1, gen));
        if (mode === 'hover' && index === activeIndex && gen === generation && (!state.candidateId || state.previewGeneId)) return;
        if (!changeGeneExplorerFocus(state, model.tracks[index].row.id, gen, mode)) return;
        activeIndex = index; generation = gen;
        if (mode === 'pin') selectedGeneId = state.pinnedGeneId;
        sync();
    };
    const hit = event => {
        const rect = canvas.getBoundingClientRect();
        const y = (event.clientY - rect.top) * model.height / rect.height;
        const x = (event.clientX - rect.left) * model.width / rect.width;
        if (y < model.top || y > model.bottom || x < model.left || x > model.right) return null;
        const targetTrack = event.target.closest?.('[data-gene-track]');
        return [targetTrack ? Number(targetTrack.dataset.geneTrack) : Math.floor((y - model.top) / model.lane), Math.floor((x - model.left) / model.step)];
    };
    canvas.addEventListener('pointermove', event => {
        if (event.pointerType === 'touch' || state.pinnedGeneId) return;
        const position = hit(event); if (position) update(...position);
    });
    canvas.addEventListener('pointerleave', () => {
        if (endGeneExplorerPreview(state)) {
            const index = model.tracks.findIndex(track => track.row.id === state.focusId);
            if (index >= 0) activeIndex = index;
            sync();
        }
    });
    canvas.addEventListener('click', event => {
        const position = hit(event); if (position) update(...position, 'pin');
    });
    canvas.addEventListener('keydown', event => {
        const changes = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], Home: [-activeIndex, 0], End: [model.tracks.length - activeIndex - 1, 0] };
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); update(activeIndex, generation, 'pin'); }
        else if (changes[event.key]) { event.preventDefault(); update(activeIndex + changes[event.key][0], generation + changes[event.key][1], 'keyboard'); }
    });
    const clear = () => { clearGeneExplorerSelection(state); selectedGeneId = null; sync(); };
    root.addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); clear(); canvas.focus(); } });
    root.addEventListener('click', event => {
        const button = event.target.closest('button');
        if (!button) return;
        if (button.id === 'gene-thread-clear') { clear(); canvas.focus(); }
        else if (button.id === 'gene-thread-pin') {
            if (state.pinnedGeneId) clear(); else update(activeIndex, generation, 'pin');
        } else if (button.hasAttribute('data-gene-story-read')) {
            const candidate = geneExplorerCandidates(gens, model.tracks[activeIndex].row, state)[Number(button.dataset.geneStoryRead)];
            if (candidate) showIdeaModal(getIdeaData(candidate.idea));
        } else if (button.hasAttribute('data-gene-story')) {
            const index = Number(button.dataset.geneStory);
            const candidate = geneExplorerCandidates(gens, model.tracks[activeIndex].row, state)[index];
            if (!candidate) return;
            state.candidateId = state.candidateId === candidate.id ? null : candidate.id;
            state.candidateGeneration = state.candidateId ? candidate.generation : null;
            state.previewGeneId = null;
            state.pinnedGeneId = null; selectedGeneId = null;
            const dna = geneStoryIds(candidate);
            if (state.candidateId && !dna.has(model.tracks[activeIndex].row.id)) {
                const first = model.tracks.findIndex(t => dna.has(t.row.id));
                if (first >= 0) { activeIndex = first; state.focusId = model.tracks[first].row.id; }
            }
            state.candidateGeneId = state.candidateId ? model.tracks[activeIndex].row.id : null;
            sync();
            root.querySelector(`[data-gene-story="${index}"]`)?.focus();
        }
    });
    const ideaSearch = container.querySelector('#gene-idea-search');
    ideaSearch?.addEventListener('input', event => {
        state.ideaSearch = event.target.value;
        sync();
    });
    ideaSearch?.addEventListener('keydown', event => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        container.querySelector('[data-gene-story]')?.click();
    });
    const search = container.querySelector('#gene-history-search');
    let searchMatches = model.tracks.map((track, index) => [track.row.label, ...(track.row.aliases || [])].some(label => String(label).toLocaleLowerCase().includes(state.search.trim().toLocaleLowerCase())) ? index : -1).filter(i => i >= 0);
    search?.addEventListener('input', event => {
        state.search = event.target.value;
        const query = state.search.trim().toLocaleLowerCase();
        searchMatches = model.tracks.map((track, index) => [track.row.label, ...(track.row.aliases || [])].some(label => String(label).toLocaleLowerCase().includes(query)) ? index : -1).filter(i => i >= 0);
        const matched = new Set(searchMatches);
        groups.forEach((group, index) => group.classList.toggle('is-search-dimmed', !matched.has(index)));
        container.querySelector('#gene-thread-hint').textContent = query ? `${searchMatches.length} matching · all ${model.tracks.length} visible` : '';
        if (searchMatches.length && query) update(searchMatches[0], generation);
    });
    search?.addEventListener('keydown', event => {
        if (event.key === 'Enter') { event.preventDefault(); update(searchMatches[0] ?? activeIndex, generation, 'pin'); canvas.focus(); }
    });
}

function applyGeneCatalogue(data, reset = false) {
    const supplied = data && Object.prototype.hasOwnProperty.call(data, 'gene_catalogue');
    if (!supplied && !reset) return false;
    if (reset) getGeneExplorerState.state = null;
    const next = supplied && data.gene_catalogue && Array.isArray(data.gene_catalogue.genes) ? data.gene_catalogue : null;
    const changed = safeSignature(geneCatalogue) !== safeSignature(next);
    geneCatalogue = next;
    if (selectedGeneId && !geneCatalogue?.genes.some(gene => gene.id === selectedGeneId)) selectedGeneId = null;
    return changed;
}

function getGeneExplorerState() {
    return getGeneExplorerState.state ||= { mode: 'history', search: '', ideaSearch: '', focusId: null, focusGeneration: null, pinnedGeneId: null, candidateId: null, candidateGeneration: null, candidateGeneId: null, previewGeneId: null };
}

function renderGeneCatalogue(gens = generations) {
    const container = document.getElementById('evolution-genes-container');
    if (!container) return;
    const view = buildGeneCatalogueView(gens, geneCatalogue, geneScope);
    const selected = view.rows.find(row => row.id === selectedGeneId);
    const explorer = getGeneExplorerState();
    if (typeof ResizeObserver === 'function' && !container.geneMapObserver) {
        let previousWidth = container.clientWidth;
        container.geneMapObserver = new ResizeObserver(() => {
            const width = container.clientWidth;
            if (width > 0 && Math.abs(width - previousWidth) > 2) {
                previousWidth = width;
                if (!container.hidden) renderGeneCatalogue(generations);
            }
        });
        container.geneMapObserver.observe(container);
    }
    const history = buildGeneHistory(gens, geneCatalogue);
    const selectedHistory = history.rows.find(row => row.id === selectedGeneId);
    const selectedPoint = geneScope !== 'history' ? selectedHistory?.points[view.generationIndex] : null;
    const selectedStats = selectedPoint ? `<p class="gene-coverage"><strong>Generation ${selectedPoint.generation} · ${selectedPoint.count} observed carriers / ${selectedPoint.analysed} analysed${selectedPoint.complete ? '' : ' · incomplete coverage'}</strong><span>Mean carrier fitness: ${selectedPoint.mean === null ? 'unavailable' : selectedPoint.mean.toFixed(3)} (${selectedPoint.measured}/${selectedPoint.count} measured). Generation mean: ${selectedPoint.populationMean === null ? 'unavailable' : selectedPoint.populationMean.toFixed(3)} (${selectedPoint.populationMeasured}/${selectedPoint.total} measured).</span></p>` : selected ? '<p class="gene-explanation">Unique carriers across history. Choose a generation in the timeline to compare fitness within the same population.</p>' : '';
    const scopeOptions = [
        ['current', 'Current generation'], ['history', 'All history · unique candidates'],
        ...(Array.isArray(gens) ? gens : []).map((_, index) => [`generation:${index}`, `Generation ${index}${index === 0 ? ' · initial' : ''}`]),
    ];
    const coverage = `${view.analysed} of ${view.total} unique candidates analysed`;
    const omissions = [`${view.missing} not analysed`, `${view.failed} failed`, `${view.unclassified} with unclassified genes`];
    container.innerHTML = `
        <div class="gene-catalogue-header">
            <div><h3>Gene pool</h3></div>
            <label class="gene-scope-control" ${explorer.mode === 'history' ? 'hidden' : ''}>Carrier scope<select id="gene-scope-select" class="form-select form-select-sm">${scopeOptions.map(([value, label]) => `<option value="${escapeHtml(value)}" ${geneScope === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select></label>
        </div>
        ${geneCatalogue?.consolidation?.revision ? `<p class="gene-explanation">Catalogue revision ${escapeHtml(String(geneCatalogue.consolidation.revision))} · history shown through this vocabulary.</p>` : ''}
        ${geneCatalogue?.mapping_status === 'failed' ? `<p class="gene-error" role="status">Consolidation unavailable. Saved factors are retained; new candidates may have raw genes only.</p>` : ''}
        ${explorer.mode === 'catalogue' || view.missing || view.failed || view.unclassified ? `<p class="gene-coverage" role="status"><strong>${escapeHtml(coverage)}</strong><span>${escapeHtml(omissions.join(' · '))}</span></p>` : ''}
        <div class="gene-view-switch" role="group" aria-label="Gene view"><button type="button" data-gene-view="history" aria-pressed="${explorer.mode === 'history'}">History</button><button type="button" data-gene-view="catalogue" aria-pressed="${explorer.mode === 'catalogue'}">Catalogue</button></div>
        ${explorer.mode === 'history' ? renderGeneHistory(history, explorer, gens) : ''}
        ${view.rows.length && explorer.mode === 'catalogue' ? `
            <div class="gene-table-scroll"><table class="gene-catalogue-table">
                <caption>Observed carriers / analysed candidates. A candidate can carry several genes; repeated survivors count once. Missing or unclassified observations may hide additional carriers.</caption>
                <thead><tr><th scope="col">Gene</th><th scope="col">Carriers / analysed</th><th scope="col">First seen</th></tr></thead>
                <tbody>${view.rows.map((row, index) => `<tr ${selectedGeneId === row.id ? 'class="gene-selected"' : ''}><th scope="row"><button type="button" class="gene-label-button" data-gene-row="${index}" aria-controls="gene-carriers" aria-expanded="${selectedGeneId === row.id}">${escapeHtml(row.label)}</button></th><td>${view.analysed ? `${row.carriers.length} / ${view.analysed}` : 'Not measured'}</td><td>${Number.isInteger(row.first_seen_generation) ? `Gen ${row.first_seen_generation}` : 'Unknown'}</td></tr>`).join('')}</tbody>
            </table></div>` : !view.rows.length && !view.unclassifiedRows.length ? `<div class="gene-empty"><h4>${view.analysed ? 'No catalogued genes yet' : 'No gene observations yet'}</h4><p>${view.analysed ? `${view.empty} analysed candidates have no extracted genes; ${view.unclassifiedGenes} extracted genes are unclassified.` : 'This run has no successful gene analysis in the selected scope. Older candidates may not have been analysed.'} Opening this view does not analyse or backfill candidates.</p></div>` : ''}
        <section id="gene-carriers" class="gene-carriers" ${selected && explorer.mode === 'catalogue' ? '' : 'hidden'} aria-labelledby="gene-carriers-title">
            ${selected ? `<div class="gene-carriers-header"><div><p class="gene-kicker">Carrying this gene</p><h4 id="gene-carriers-title" tabindex="-1">${escapeHtml(selected.label)}</h4></div><button type="button" class="btn btn-sm btn-outline-secondary" id="gene-clear-selection">Clear selection</button></div>
                ${selectedStats}
                ${Array.isArray(selected.aliases) && selected.aliases.length ? `<details class="gene-aliases"><summary>Observed wording (${selected.aliases.length})</summary><ul>${selected.aliases.map(alias => `<li>${escapeHtml(String(alias))}</li>`).join('')}</ul></details>` : ''}
                ${selected.carriers.length ? `<ul class="gene-carrier-list">${selected.carriers.map((candidate, index) => `<li><button type="button" data-gene-carrier="${index}"><span>${escapeHtml(getIdeaData(candidate.idea).title || 'Untitled')}</span><small>Gen ${candidate.generation} · Open candidate</small></button></li>`).join('')}</ul>` : '<p class="gene-explanation">No observed carriers in this scope. Try all history to see earlier candidates.</p>'}` : ''}
        </section>
        ${view.unclassifiedRows.length && explorer.mode === 'catalogue' ? `<section class="gene-unmapped" aria-labelledby="gene-unmapped-title">
            <h4 id="gene-unmapped-title" class="gene-inspector-title">Unmapped genes</h4>
            <p class="gene-explanation">These extracted genes have not been matched to the catalogue. ${view.mappingFailed ? `Catalogue matching failed for ${view.mappingFailed} candidate${view.mappingFailed === 1 ? '' : 's'}. ` : ''}Their original wording is preserved; select a candidate to inspect it.</p>
            <div class="gene-table-scroll"><table class="gene-catalogue-table gene-unmapped-table">
                <caption>${view.unclassifiedRows.length} distinct raw terms · identical wording grouped for display only.</caption>
                <thead><tr><th scope="col">Raw gene · unmapped</th><th scope="col">Candidates</th></tr></thead>
                <tbody>${view.unclassifiedRows.map((row, rowIndex) => `<tr><th scope="row">${escapeHtml(row.label)}</th><td><ul class="gene-carrier-list">${row.carriers.map((candidate, candidateIndex) => `<li><button type="button" data-unmapped-gene="${rowIndex}" data-unmapped-carrier="${candidateIndex}"><span>${escapeHtml(getIdeaData(candidate.idea).title || 'Untitled')}</span><small>Gen ${candidate.generation} · Open candidate</small></button></li>`).join('')}</ul></td></tr>`).join('')}</tbody>
            </table></div>
        </section>` : ''}`;
    container.querySelectorAll('[data-gene-view]').forEach(button => button.addEventListener('click', () => {
        explorer.mode = button.dataset.geneView;
        renderGeneCatalogue(gens);
        container.querySelector(`[data-gene-view="${explorer.mode}"]`)?.focus();
    }));
    bindGeneThreads(container, history, explorer, gens);
    container.querySelector('#gene-scope-select')?.addEventListener('change', event => {
        geneScope = event.target.value;
        renderGeneCatalogue(gens);
        container.querySelector('#gene-scope-select')?.focus();
    });
    container.querySelectorAll('[data-gene-row]').forEach(button => button.addEventListener('click', () => {
        selectedGeneId = view.rows[Number(button.dataset.geneRow)]?.id || null;
        explorer.pinnedGeneId = selectedGeneId; explorer.focusId = selectedGeneId; explorer.candidateId = null;
        renderGeneCatalogue(gens);
        document.getElementById('gene-carriers-title')?.focus();
    }));
    container.querySelector('#gene-clear-selection')?.addEventListener('click', () => {
        const rowIndex = view.rows.findIndex(row => row.id === selectedGeneId);
        clearGeneExplorerSelection(explorer);
        selectedGeneId = null;
        renderGeneCatalogue(gens);
        (container.querySelector(`[data-gene-row="${rowIndex}"]`) || container.querySelector('[data-gene-view="history"]'))?.focus();
    });
    container.querySelectorAll('[data-gene-carrier]').forEach(button => button.addEventListener('click', () => {
        const candidate = selected?.carriers[Number(button.dataset.geneCarrier)];
        if (candidate) showIdeaModal(getIdeaData(candidate.idea));
    }));
    container.querySelectorAll('[data-unmapped-carrier]').forEach(button => button.addEventListener('click', () => {
        const row = view.unclassifiedRows[Number(button.dataset.unmappedGene)];
        const candidate = row?.carriers[Number(button.dataset.unmappedCarrier)];
        if (candidate) showIdeaModal(getIdeaData(candidate.idea));
    }));
}

function openGeneCatalogue(geneId) {
    selectedGeneId = geneId;
    const state = getGeneExplorerState();
    state.mode = 'history'; state.focusId = geneId; state.pinnedGeneId = geneId; state.candidateId = null;
    // Include historical candidates when navigating from any candidate's inspector.
    geneScope = 'history';
    window.IdeaWorkbench?.closeInspector();
    setEvolutionViewMode('genes');
    document.getElementById('gene-thread-canvas')?.focus();
}

function renderIdeaModalGenes(idea) {
    const panel = document.getElementById('ideaModalGenes');
    if (!panel) return;
    const observation = getGeneObservation(idea);
    const rows = new Map((geneCatalogue?.genes || []).map(gene => [gene.id, gene]));
    if (!observation) {
        panel.innerHTML = '<p class="gene-explanation">No gene analysis saved for this idea.</p>';
        return;
    }
    if (observation.status !== 'ok' || !Array.isArray(observation.genes)) {
        panel.innerHTML = `<h3 class="gene-inspector-title">Gene analysis unavailable</h3>${observation.error ? `<p class="gene-error">${escapeHtml(String(observation.error))}</p>` : ''}`;
        return;
    }
    const normalize = value => String(value || '').trim().toLocaleLowerCase().replace(/\s+/g, ' ').replace(/[.;:]+$/, '');
    const uniqueTerms = terms => [...new Map(terms.filter(term => typeof term === 'string' && term.trim())
        .map(term => [normalize(term), term.trim()])).values()];
    const entries = new Map();
    observation.genes.forEach((gene, index) => {
        const canonical = rows.get(gene.gene_id);
        const label = String(canonical?.label || gene.text || canonical?.id || 'Unnamed gene');
        const key = canonical ? `id:${canonical.id}` : `raw:${normalize(label)}`;
        if (!entries.has(key)) entries.set(key, {canonical, label, index, genes: []});
        entries.get(key).genes.push(gene);
    });
    const history = typeof generations !== 'undefined' && Array.isArray(generations) ? generations : [];
    const reasoning = [];
    const visibleTerms = new Set();
    const unchanged = new Set(['carried forward unchanged', 'unchanged', 'exact match', 'identity mapping', 'no change']);
    const items = [...entries.values()].map(({canonical, label, index, genes}) => {
        const decisions = genes.flatMap(gene => Array.isArray(gene.decisions) ? gene.decisions.filter(d => d && typeof d === 'object') : []);
        const sources = uniqueTerms(genes.flatMap(gene => [gene.text, ...(Array.isArray(gene.source_terms) ? gene.source_terms : [])])
            .concat(decisions.map(d => d.source))).filter(source => normalize(source) !== normalize(label));
        [label, ...sources].forEach(term => visibleTerms.add(normalize(term)));
        const unresolved = uniqueTerms(decisions.map(d => d.unresolved));
        const notes = uniqueTerms(decisions.filter(d => typeof d.reason === 'string' && !unchanged.has(normalize(d.reason))
            && [d.source, d.input].some(term => term && normalize(term) !== normalize(label))).map(d => d.reason));
        if (sources.length || unresolved.length || notes.length) reasoning.push({label, sources, unresolved, notes});
        const carriers = canonical ? geneHistoryCarriers(history, canonical.id) : [];
        const count = carriers.length;
        const first = count ? Math.min(...carriers.map(carrier => carrier.firstGeneration)) : null;
        const content = `<span class="gene-inspector-dot" aria-hidden="true"></span><span class="gene-inspector-name">${escapeHtml(label)}</span>
            ${count ? `<span class="gene-inspector-count" title="First observed in generation ${first}; ${count} unique ideas across saved history">First G${first} · ${count} idea${count === 1 ? '' : 's'}</span>` : ''}`;
        return `<li style="--gene-color:${geneThreadColor(canonical?.id || label)}">
            ${canonical ? `<button type="button" class="gene-inspector-link" data-candidate-gene="${index}" aria-label="${escapeHtml(label)}${count ? `, first seen in generation ${first}, ${count} idea${count === 1 ? '' : 's'} across history` : ''}. Explore gene">${content}</button>` : `<div class="gene-inspector-link">${content}<span class="gene-inspector-count">Unmapped</span></div>`}
        </li>`;
    });
    // Keep provenance available once, outside the main gene list.
    const raw = typeof (observation.raw_genotype ?? observation.genotype) === 'string'
        ? (observation.raw_genotype ?? observation.genotype).trim() : '';
    const extraRaw = raw && raw.split(';').some(term => normalize(term) && !visibleTerms.has(normalize(term)));
    panel.innerHTML = items.length ? `<div class="gene-inspector-heading"><span>${items.length} gene${items.length === 1 ? '' : 's'}</span>
        ${observation.mapping_status === 'failed' ? '<span>Mapping incomplete</span>' : ''}</div>
        <ul class="candidate-gene-list">${items.join('')}</ul><p class="gene-inspector-legend" title="Accent colors repeat and do not encode generation or fitness.">Colors match the gene timeline.</p>` : '<p class="gene-explanation">No genes were extracted.</p>';
    if (reasoning.length || extraRaw) {
        panel.innerHTML += `<details class="gene-inspector-details"><summary>Mapping details</summary>
            ${extraRaw ? `<p class="gene-inspector-original">${escapeHtml(raw)}</p>` : ''}
            ${reasoning.map(({label, sources, unresolved, notes}) => `<p><strong>${escapeHtml(label)}</strong>${[
                sources.length ? `From ${sources.map(escapeHtml).join(' · ')}` : '',
                ...notes.map(escapeHtml),
                ...unresolved.map(note => `Review: ${escapeHtml(note)}`),
            ].filter(Boolean).join('<br>')}</p>`).join('')}
        </details>`;
    }
    panel.querySelectorAll('[data-candidate-gene]').forEach(button => button.addEventListener('click', () => {
        const gene = observation.genes[Number(button.dataset.candidateGene)];
        if (gene && rows.has(gene.gene_id)) openGeneCatalogue(gene.gene_id);
    }));
}

function getRovingTabIndex(currentIndex, key, tabCount) {
    const count = Math.max(0, Math.trunc(Number(tabCount) || 0));
    if (count === 0 || currentIndex < 0 || currentIndex >= count) {
        return null;
    }
    if (key === 'Home') return 0;
    if (key === 'End') return count - 1;
    if (key === 'ArrowRight' || key === 'ArrowDown') return (currentIndex + 1) % count;
    if (key === 'ArrowLeft' || key === 'ArrowUp') return (currentIndex - 1 + count) % count;
    return null;
}

function handleEvolutionViewTabKeydown(event) {
    const tabs = Array.from(document.querySelectorAll('.evolution-view-toggle-btn[role="tab"]'));
    const targetIndex = getRovingTabIndex(tabs.indexOf(event.currentTarget), event.key, tabs.length);
    if (targetIndex === null) return;

    event.preventDefault();
    const targetTab = tabs[targetIndex];
    setEvolutionViewMode(targetTab.dataset.viewMode);
    targetTab.focus();
}

function setEvolutionViewMode(mode) {
    evolutionViewMode = ['tree', 'ranked', 'genes'].includes(mode) ? mode : 'cards';
    const cardsContainer = document.getElementById('generations-container');
    const treeContainer = document.getElementById('evolution-tree-container');
    const rankedContainer = document.getElementById('evolution-ranked-container');
    const genesContainer = document.getElementById('evolution-genes-container');
    const toolbar = document.getElementById('evolution-view-toolbar');
    if (toolbar) {
        toolbar.style.display = generations && generations.length > 0 ? 'flex' : 'none';
    }
    if (cardsContainer) {
        cardsContainer.hidden = evolutionViewMode !== 'cards';
    }
    if (treeContainer) {
        treeContainer.hidden = evolutionViewMode !== 'tree';
    }
    if (rankedContainer) {
        rankedContainer.hidden = evolutionViewMode !== 'ranked';
    }
    if (genesContainer) genesContainer.hidden = evolutionViewMode !== 'genes';
    document.querySelectorAll('.evolution-view-toggle-btn').forEach((button) => {
        const isActive = button.dataset.viewMode === evolutionViewMode;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
        button.tabIndex = isActive ? 0 : -1;
    });
    if (evolutionViewMode === 'tree') {
        renderEvolutionTree(generations);
    }
    if (evolutionViewMode === 'ranked') {
        renderEvolutionRankedList(generations);
    }
    if (evolutionViewMode === 'genes') renderGeneCatalogue(generations);
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
    const rankedContainer = document.getElementById('evolution-ranked-container');
    const genesContainer = document.getElementById('evolution-genes-container');
    if (rankedContainer) {
        rankedContainer.innerHTML = '';
        rankedContainer.hidden = true;
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
    rankedSortMetric = 'fitness';
    rankedScope = 'history';
    geneCatalogue = null;
    geneScope = 'current';
    selectedGeneId = null;
    lastGeneObservationSignature = '';
    currentModalIdea = null;
    if (genesContainer) { genesContainer.innerHTML = ''; genesContainer.hidden = true; }
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
            const artifactSignature = getIdeaArtifactSignature(idea);
            const geneSignature = safeSignature([idea.id, getGeneObservation(idea)]);
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
                const artifactsChanged = (existingCard.dataset.artifactSignature || '') !== artifactSignature;
                const genesChanged = existingCard.dataset.geneSignature !== geneSignature;

                // We also check if the existing card in the DOM reflects this.
                // A simple way is to check the button's inner HTML for the icon.
                const lineageButton = existingCard.querySelector('.view-lineage');
                const cardIsAlreadyOracle = lineageButton && lineageButton.innerHTML.includes('fa-eye');

                if ((isOracleGenerated && !cardIsAlreadyOracle) || genotypeCountChanged || islandChanged || migrationChanged || oracleChanged || artifactsChanged || genesChanged) {
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
            card.dataset.artifactSignature = artifactSignature;
            card.dataset.geneSignature = geneSignature;
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
            const artifactCount = getIdeaArtifactRefs(ideaData).length;
            const artifactBadgeHtml = artifactCount > 0
                ? `<span class="artifact-badge"><i class="fas fa-image" aria-hidden="true"></i>${artifactCount} artifact${artifactCount === 1 ? '' : 's'}</span>`
                : '';
            const artifactPreviewHtml = renderCardArtifactPreview(ideaData);

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
                        ${artifactBadgeHtml}
                    </div>
                    <h5 class="card-title"></h5>
                    ${artifactPreviewHtml}
                    <div class="card-text">
                        <p class="idea-preview-text"></p>
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
            card.querySelector('.card-title').textContent = ideaData.title || 'Untitled';
            card.querySelector('.idea-preview-text').textContent = plainPreview;

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
            hydrateAuthenticatedArtifactImages(card);

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

function rememberModalFocusOrigin(modalElement) {
    if (!modalElement) return;
    const activeElement = document.activeElement;
    modalElement._focusReturnTarget = activeElement && activeElement !== document.body
        ? activeElement
        : null;
    if (modalElement.dataset.focusReturnBound === 'true') return;

    modalElement.dataset.focusReturnBound = 'true';
    modalElement.addEventListener('hidden.bs.modal', () => {
        const target = modalElement._focusReturnTarget;
        modalElement._focusReturnTarget = null;
        if (!target?.isConnected || target.disabled || typeof target.focus !== 'function') return;
        target.focus({ preventScroll: true });
    });
}

function clearIdeaDetailElement(element) {
    if (element) element.replaceChildren();
}

function formatArtifactBytes(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return 'size unknown';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function buildArtifactFigure(ref, ideaTitle) {
    const artifactUrl = getEvolutionArtifactUrl(ref);
    if (!artifactUrl) return null;
    const figure = document.createElement('figure');
    figure.className = 'idea-artifact-figure';
    const image = document.createElement('img');
    image.dataset.artifactUrl = artifactUrl;
    image.alt = `Visualization for ${ideaTitle || 'idea'}`;
    image.loading = 'lazy';
    const caption = document.createElement('figcaption');
    caption.textContent = `${ref.role} · PNG · ${formatArtifactBytes(ref.size_bytes)}`;
    figure.append(image, caption);
    return figure;
}

function activateIdeaModalTab(tabName) {
    const modal = document.getElementById('ideaModal');
    if (!modal) return;
    modal.dataset.activeTab = tabName;
    modal.querySelectorAll('[data-idea-modal-tab]').forEach(button => {
        const selected = !button.hidden && button.dataset.ideaModalTab === tabName;
        button.setAttribute('aria-selected', selected ? 'true' : 'false');
        button.tabIndex = selected ? 0 : -1;
    });
    modal.querySelectorAll('[data-idea-modal-panel]').forEach(panel => {
        panel.hidden = panel.dataset.ideaModalPanel !== tabName;
    });
    if (tabName === 'artifacts') hydrateAuthenticatedArtifactImages(document.getElementById('ideaModalArtifacts'));
    if (tabName === 'code') document.getElementById('ideaModalCode')?._loadSource?.();
}

function configureIdeaModalTabs(availableTabs) {
    const tabList = document.getElementById('ideaModalTabs');
    if (!tabList) return;
    const available = new Set(availableTabs);
    const buttons = [...tabList.querySelectorAll('[data-idea-modal-tab]')];
    buttons.forEach(button => {
        button.hidden = !available.has(button.dataset.ideaModalTab);
        button.onclick = () => activateIdeaModalTab(button.dataset.ideaModalTab);
        button.onkeydown = event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            const visible = buttons.filter(candidate => !candidate.hidden);
            const currentIndex = visible.indexOf(button);
            let nextIndex = currentIndex;
            if (event.key === 'ArrowRight') nextIndex = (currentIndex + 1) % visible.length;
            if (event.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + visible.length) % visible.length;
            if (event.key === 'Home') nextIndex = 0;
            if (event.key === 'End') nextIndex = visible.length - 1;
            event.preventDefault();
            const next = visible[nextIndex];
            activateIdeaModalTab(next.dataset.ideaModalTab);
            next.focus();
        };
    });
    tabList.hidden = available.size <= 1;
    activateIdeaModalTab('overview');
}

function renderIdeaModalArtifacts(idea) {
    const refs = getIdeaArtifactRefs(idea);
    const images = refs.filter(ref => ref.media_type === 'image/png');
    const primary = getPrimaryImageArtifact(idea);
    const primaryContainer = document.getElementById('ideaModalPrimaryArtifact');
    const gallery = document.getElementById('ideaModalArtifacts');
    clearIdeaDetailElement(primaryContainer);
    clearIdeaDetailElement(gallery);

    if (primary) {
        const figure = buildArtifactFigure(primary, idea.title);
        if (figure) {
            primaryContainer.appendChild(figure);
            primaryContainer.hidden = false;
        } else {
            primaryContainer.hidden = true;
        }
    } else {
        primaryContainer.hidden = true;
    }

    images.forEach(ref => {
        const figure = buildArtifactFigure(ref, idea.title);
        if (figure) gallery.appendChild(figure);
    });
    refs.filter(ref => ref.media_type !== 'image/png').forEach(ref => {
        const row = document.createElement('div');
        row.className = 'idea-artifact-row';
        row.textContent = `${ref.role} · ${ref.media_type} · ${formatArtifactBytes(ref.size_bytes)}`;
        gallery.appendChild(row);
    });

    hydrateAuthenticatedArtifactImages(primaryContainer);
    return refs.length > 0;
}

function renderIdeaModalCode(idea) {
    const panel = document.getElementById('ideaModalCode');
    clearIdeaDetailElement(panel);
    panel._loadSource = null;
    const sourceRef = getIdeaArtifactRefs(idea).find(ref => (
        ref.role === 'source' || ref.media_type === 'text/x-python'
    ));
    const artifactUrl = getEvolutionArtifactUrl(sourceRef);
    if (!sourceRef || !artifactUrl) return false;

    const output = document.createElement('pre');
    output.className = 'idea-code-output';
    output.setAttribute('aria-label', 'Generated Python source');
    output.textContent = 'Select Code & run to load generated source.';
    panel.dataset.artifactId = sourceRef.id;
    panel.appendChild(output);
    panel._loadSource = () => {
    panel._loadSource = null;
    output.textContent = 'Loading generated source…';
    fetch(artifactUrl, { headers: { Accept: 'text/plain' } })
        .then(response => {
            if (!response.ok) throw new Error(`source request failed (${response.status})`);
            return response.text();
        })
        .then(source => {
            if (panel.dataset.artifactId === sourceRef.id) output.textContent = source;
        })
        .catch(error => {
            if (panel.dataset.artifactId === sourceRef.id) {
                output.textContent = `Generated source is unavailable: ${error.message}`;
            }
        });
    };
    return true;
}

function appendEvaluationDetail(list, label, value) {
    if (value === undefined || value === null || value === '') return;
    const term = document.createElement('dt');
    term.textContent = label;
    const detail = document.createElement('dd');
    detail.textContent = String(value);
    list.append(term, detail);
}

function renderIdeaModalEvaluations(idea) {
    const panel = document.getElementById('ideaModalEvaluations');
    clearIdeaDetailElement(panel);
    const evaluations = idea?.evaluations && typeof idea.evaluations === 'object'
        ? idea.evaluations
        : {};
    const entries = Object.entries(evaluations);
    const selection = idea?.selection_metrics && typeof idea.selection_metrics === 'object'
        ? idea.selection_metrics
        : null;
    const diversitySummary = idea?.diversity_summary;
    const hasDiversitySummary = Array.isArray(diversitySummary?.summaries) && diversitySummary.summaries.length > 0;
    if (entries.length === 0 && !selection && !hasDiversitySummary) return false;

    const explanation = document.createElement('p');
    explanation.className = 'idea-evaluation-note';
    explanation.textContent = 'Scoring evidence used to rank this idea.';
    panel.appendChild(explanation);
    if (hasDiversitySummary) {
        const disclosure = document.createElement('details');
        disclosure.className = 'idea-evaluation-disclosure';
        const toggle = document.createElement('summary');
        toggle.textContent = 'Diversity descriptions';
        const section = document.createElement('article');
        section.className = 'idea-evaluation-card';
        const heading = document.createElement('p');
        heading.className = 'idea-evaluation-note';
        heading.textContent = 'These descriptions are compared to estimate how different this idea is from the population.';
        const descriptions = document.createElement('dl');
        diversitySummary.summaries.forEach((summary, index) => {
            appendEvaluationDetail(descriptions, diversitySummary.criteria?.[index] || `Criterion ${index+1}`, summary);
        });
        section.append(heading, descriptions);
        disclosure.append(toggle, section);
        panel.appendChild(disclosure);
    }

    entries.forEach(([componentId, record]) => {
        const card = document.createElement('article');
        card.className = 'idea-evaluation-card';
        const heading = document.createElement('h3');
        heading.textContent = componentId.replaceAll('_', ' ');
        const details = document.createElement('dl');
        appendEvaluationDetail(details, 'Status', record?.status || 'unknown');
        const metrics = record?.metrics && typeof record.metrics === 'object'
            ? Object.entries(record.metrics).map(([name, value]) => `${name}: ${value}`).join(', ')
            : '';
        appendEvaluationDetail(details, 'Metrics', metrics);
        appendEvaluationDetail(details, 'Duration', Number.isFinite(Number(record?.duration_ms))
            ? `${Number(record.duration_ms)} ms`
            : null);
        appendEvaluationDetail(details, 'Error', record?.error);
        const execution = record?.metadata?.execution;
        if (execution && typeof execution === 'object') {
            const repairLoop = execution?.repair_loop;
            const finalCause = repairLoop?.final_cause;
            appendEvaluationDetail(details, 'Failure stage', execution.stage || finalCause?.stage);
            appendEvaluationDetail(details, 'Container exit', execution.container_exit_code);
            appendEvaluationDetail(
                details,
                'Execution attempts',
                execution.execution_attempts ?? repairLoop?.execution_attempts,
            );
            appendEvaluationDetail(
                details,
                'Repair attempts',
                execution.repair_attempts ?? repairLoop?.repair_attempts,
            );
            appendEvaluationDetail(details, 'Detail', execution.detail || finalCause?.message);
        }
        const provenance = record?.metadata?.provenance;
        if (provenance && typeof provenance === 'object') {
            appendEvaluationDetail(details, 'Execution attempts', provenance.execution_attempts);
            appendEvaluationDetail(details, 'Repair attempts', provenance.repair_attempts);
            appendEvaluationDetail(details, 'Repair provider', provenance.repair_provider);
        }
        card.append(heading, details);
        panel.appendChild(card);
    });
    return true;
}

// Update the showIdeaModal function to ensure the modal is properly initialized
function showIdeaModal(idea) {
    // Get the modal element
    const modalElement = document.getElementById('ideaModal');

    // Store the idea for clipboard copying and keep selection shared between views.
    const previousId = currentModalIdea?.id;
    const previousTab = modalElement.querySelector('[data-idea-modal-tab][aria-selected="true"]')?.dataset.ideaModalTab;
    currentModalIdea = idea;
    if (idea.id && currentEvolutionTree?.nodesById?.has(String(idea.id))) selectedTreeNodeId = String(idea.id);
    modalElement.dataset.candidateSignature = safeSignature(idea);

    // Set the title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Render the markdown content
    const modalContent = document.getElementById('ideaModalContent');

    // Set the content and reveal only detail tabs backed by candidate data.
    const renderedContent = renderMarkdown(idea.content || '');
    modalContent.innerHTML = renderedContent;
    const hasArtifacts = renderIdeaModalArtifacts(idea);
    const hasCode = renderIdeaModalCode(idea);
    const hasEvaluations = renderIdeaModalEvaluations(idea);
    renderIdeaModalGenes(idea);
    configureIdeaModalTabs([
        'overview',
        'genes',
        ...(hasArtifacts ? ['artifacts'] : []),
        ...(hasCode ? ['code'] : []),
        ...(hasEvaluations ? ['evaluations'] : []),
    ]);
    if (previousId === idea.id && previousTab && !modalElement.querySelector(`[data-idea-modal-tab="${previousTab}"]`)?.hidden) activateIdeaModalTab(previousTab);

    window.IdeaWorkbench?.renderMetrics(
        document.getElementById('candidateMetricEvidence'),
        idea,
        currentEvolutionData?.config?.evaluation || {},
    );
    if (window.IdeaWorkbench) {
        window.IdeaWorkbench.openInspector(modalElement);
    } else {
        modalElement.hidden = false;
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
            .map(item => `<div class="context-item">${escapeHtml(item.trim())}</div>`)
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
        gene_catalogue: data.gene_catalogue ?? geneCatalogue,
        diversity_history: data.diversity_history || [],
        island_diversity_history: data.island_diversity_history || [],
        migration_history: data.migration_history || [],
        tournament_history: data.tournament_history || tournamentHistory || [],
        contexts: data.contexts || [],
        specific_prompts: data.specific_prompts || specificPrompts || [],
        breeding_prompts: data.breeding_prompts || breedingPrompts || [],
        token_counts: data.token_counts || {},
        config: data.config || currentEvolutionData?.config || {},
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
const repairedEvolutionCacheIds = new Set();
const deferredEvolutionCacheStates = new Map();

function getCachedEvolutionState() {
    if (currentEvolutionData?.history?.length) {
        return currentEvolutionData;
    }
    try {
        const storedData = localStorage.getItem(`idea.${window.currentUser?.uid || 'signed-out'}.currentEvolutionData`);
        return storedData ? JSON.parse(storedData) : null;
    } catch (error) {
        console.warn('Unable to read cached evolution state:', error);
        return null;
    }
}

function getCachedEvolutionId(evolutionState) {
    return evolutionState?.evolution_id || evolutionState?.evolutionId || currentEvolutionId || null;
}

function getEvolutionGenerationCount(evolutionState) {
    const explicitGeneration = evolutionState?.current_generation;
    if (Number.isFinite(explicitGeneration)) {
        return explicitGeneration;
    }
    const history = evolutionState?.history;
    return Array.isArray(history) && history.length > 0 ? history.length - 1 : 0;
}

function getEvolutionTotalIdeas(history) {
    if (!Array.isArray(history)) return 0;
    return history.reduce((total, generation) => {
        return total + (Array.isArray(generation) ? generation.length : 0);
    }, 0);
}

function normalizeCachedEvolutionState(evolutionState, evolutionId = null) {
    if (!evolutionState?.history?.length) return null;
    const normalizedId = evolutionId || getCachedEvolutionId(evolutionState);
    const currentGeneration = getEvolutionGenerationCount(evolutionState);
    const totalGenerations = Number.isFinite(evolutionState.total_generations)
        ? evolutionState.total_generations
        : Math.max(currentGeneration, evolutionState.history.length - 1);
    return {
        ...evolutionState,
        evolution_id: normalizedId,
        current_generation: currentGeneration,
        total_generations: totalGenerations,
        status: evolutionState.status || (evolutionState.is_running ? 'in_progress' : 'complete')
    };
}

function cachedEvolutionIsNewer(evolutionState, persistedData = null) {
    if (!evolutionState?.history?.length) return false;
    const persistedHistory = persistedData?.history;
    const persistedLength = Array.isArray(persistedHistory) ? persistedHistory.length : 0;
    const cachedLength = evolutionState.history.length;
    if (cachedLength !== persistedLength) {
        return cachedLength > persistedLength;
    }
    return getEvolutionGenerationCount(evolutionState) > getEvolutionGenerationCount(persistedData || {});
}

async function repairPersistedEvolutionFromClientCache(evolutionState) {
    const normalized = normalizeCachedEvolutionState(evolutionState);
    const evolutionId = normalized?.evolution_id;
    if (!evolutionId || !normalized?.history?.length) {
        return;
    }
    if (isEvolutionRunning || normalized.is_running || deferredEvolutionCacheStates.has(evolutionId)) {
        deferEvolutionRepair(normalized);
        return;
    }
    if (repairedEvolutionCacheIds.has(evolutionId)) return;
    repairedEvolutionCacheIds.add(evolutionId);
    try {
        const response = await fetch(`/api/evolution/${encodeURIComponent(evolutionId)}/repair`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: normalized })
        });
        if (response.status === 409) {
            const payload = await response.json();
            if (payload.status === 'conflict') {
                // Another tab may have started a worker after this browser's
                // last progress check. Wait for a confirmed stopped state.
                deferEvolutionRepair(normalized);
                return;
            }
        }
        if (!response.ok) {
            repairedEvolutionCacheIds.delete(evolutionId);
        }
    } catch (error) {
        repairedEvolutionCacheIds.delete(evolutionId);
        console.warn('Unable to repair persisted evolution from local cache:', error);
    }
}

function deferEvolutionRepair(evolutionState) {
    const evolutionId = getCachedEvolutionId(evolutionState);
    const previous = deferredEvolutionCacheStates.get(evolutionId);
    if (!previous || !cachedEvolutionIsNewer(previous, evolutionState)) {
        deferredEvolutionCacheStates.set(evolutionId, evolutionState);
    }
}

async function retryDeferredEvolutionRepairs(serverState) {
    if (serverState?.is_running !== false || isEvolutionRunning) return;
    for (const [evolutionId, cached] of [...deferredEvolutionCacheStates]) {
        const sameRun = getCachedEvolutionId(serverState) === evolutionId;
        const latest = sameRun && serverState.history?.length && !cachedEvolutionIsNewer(cached, serverState)
            ? serverState : cached;
        const stopped = { ...latest, evolution_id: evolutionId, is_running: false };
        if (sameRun) {
            stopped.status = typeof serverState.status === 'string' ? serverState.status
                : serverState.is_stopped || serverState.is_resumable ? 'paused' : 'complete';
        }
        deferredEvolutionCacheStates.delete(evolutionId);
        repairedEvolutionCacheIds.delete(evolutionId);
        await repairPersistedEvolutionFromClientCache(stopped);
    }
}

async function repairCurrentEvolutionIfNewer(evolutionId, persistedData = null) {
    const rawCached = getCachedEvolutionState();
    if (getCachedEvolutionId(rawCached) !== evolutionId) return null;
    const cached = normalizeCachedEvolutionState(rawCached);
    if (cachedEvolutionIsNewer(cached, persistedData)) {
        await repairPersistedEvolutionFromClientCache(cached);
        return cached;
    }
    return null;
}

function overlayHistoryItemsWithCachedEvolution(items) {
    const cached = normalizeCachedEvolutionState(getCachedEvolutionState());
    const cachedId = getCachedEvolutionId(cached);
    if (!cachedId || !cached?.history?.length) return items;

    let found = false;
    const overlaid = items.map(item => {
        if (item.id !== cachedId) return item;
        found = true;
        if (item.execution_archived || item.type === 'legacy_checkpoint') return item;
        if (!cachedEvolutionIsNewer(cached, { history: new Array((item.generations || 0) + 1), current_generation: item.generations || 0 })) {
            return item;
        }
        const status = cached.status || 'complete';
        const nextItem = {
            ...item,
            status,
            generations: cached.current_generation,
            total_generations: Math.max(cached.total_generations || 0, item.total_generations || 0),
            total_ideas: getEvolutionTotalIdeas(cached.history),
            config: cached.config || item.config || {},
            can_use_config: Boolean((cached.config && Object.keys(cached.config).length > 0) || item.can_use_config),
            can_resume: ['paused', 'in_progress', 'force_stopped', 'error'].includes(status),
            can_continue: status === 'complete',
            can_rate: cached.current_generation > 0
        };
        repairPersistedEvolutionFromClientCache(cached);
        return nextItem;
    });

    if (!found) {
        const status = cached.status || 'complete';
        overlaid.unshift({
            id: cachedId,
            type: 'evolution',
            status,
            timestamp: new Date().toISOString(),
            display_name: cached.evolution_name || cached.name || cachedId,
            generations: cached.current_generation,
            total_generations: cached.total_generations,
            pop_size: Array.isArray(cached.history.at(-1)) ? cached.history.at(-1).length : 0,
            idea_type: 'Unknown',
            model_type: 'Unknown',
            total_ideas: getEvolutionTotalIdeas(cached.history),
            config: cached.config || {},
            can_use_config: Boolean(cached.config && Object.keys(cached.config).length > 0),
            can_resume: ['paused', 'in_progress', 'force_stopped', 'error'].includes(status),
            can_continue: status === 'complete',
            can_rate: cached.current_generation > 0,
            can_rename: true,
            can_delete: true
        });
        repairPersistedEvolutionFromClientCache(cached);
    }

    return overlaid;
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.status === 'success') {
            historyItems = overlayHistoryItemsWithCachedEvolution(data.items || []);
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

function getModelDisplayName(modelId) {
    const options = Array.from(document.querySelectorAll('#modelType option'));
    const option = options.find(opt => opt.value === String(modelId || ''));
    return option ? option.textContent.trim() : (modelId || 'Unknown');
}

function getTemplateDisplayName(templateId) {
    const template = allTemplates.find(t => t.id === templateId);
    return template?.name || templateId || 'Unknown';
}

function formatConfigNumber(value, digits = 2) {
    if (value === null || value === undefined || value === '') return '-';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return String(value);
    return numeric.toFixed(digits);
}

function formatConfigInteger(value) {
    if (value === null || value === undefined || value === '') return '-';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return String(value);
    return String(Math.trunc(numeric));
}

function formatConfigBoolean(value) {
    return value === true ? 'Yes' : 'No';
}

function formatThinkingConfig(config) {
    if (config.thinking_budget !== null && config.thinking_budget !== undefined) {
        return `Budget ${Number(config.thinking_budget).toLocaleString()} tokens`;
    }
    if (config.thinking_level) {
        return String(config.thinking_level).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    return 'Provider default';
}

function renderConfigRow(label, value) {
    return `
        <div class="config-detail-row">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `;
}

function renderConfigSection(title, rows) {
    return `
        <div class="config-detail-section">
            <div class="config-detail-section-title">${escapeHtml(title)}</div>
            ${rows.map(([label, value]) => renderConfigRow(label, value)).join('')}
        </div>
    `;
}

function buildEvolutionConfigDetailsHtml(config = {}, item = {}) {
    if (!config || Object.keys(config).length === 0) {
        return `
            <div class="history-config-details">
                <div class="config-detail-empty">No saved config metadata available.</div>
            </div>
        `;
    }

    const generationCount = config.generations ?? item.total_generations;
    const populationSize = config.pop_size ?? item.pop_size;
    const localSizes = [];
    if (config.independent_birth_rate !== undefined && Number(populationSize) > 0) {
        const total = Number(populationSize);
        const islands = Math.max(1, Math.min(Number(config.island_count) || 1, Math.max(1, Math.floor(total/2))));
        for (let idx = 0; idx < islands; idx++) localSizes.push(Math.floor(total/islands) + (idx < total%islands ? 1 : 0));
    }
    const logarithmicRounds = config.tournament_rounds_policy === 'swiss_log2';
    const localRoundCounts = localSizes.map(size => {
        const fullRounds = Math.max(1, logarithmicRounds ? Math.ceil(Math.log2(size)) : size - 1);
        return Math.max(1, Math.floor(Number(config.tournament_count) * fullRounds + 0.5));
    });
    const sections = [
        renderConfigSection('Basic', [
            ['Template', getTemplateDisplayName(config.idea_type || item.idea_type)],
            ['Model', getModelDisplayName(config.model_type || item.model_type)],
            ['Population Size', formatConfigInteger(populationSize)],
            ['Generations', formatConfigInteger(generationCount)],
            ['Max Budget', config.max_budget === null || config.max_budget === undefined ? '-' : `$${formatConfigNumber(config.max_budget, 2)}`],
        ]),
        renderConfigSection('LLM', [
            ['Creative Temperature', supportsSamplingControls(config.model_type, config.thinking_level) ? formatConfigNumber(config.creative_temp, 2) : 'Provider controlled'],
            ['Top P', supportsSamplingControls(config.model_type, config.thinking_level) ? formatConfigNumber(config.top_p, 2) : 'Provider controlled'],
            ['Seed Context Pool Size', formatConfigInteger(config.seed_context_pool_size)],
            ['Thinking', formatThinkingConfig(config)],
        ]),
        renderConfigSection('Tournament', [
            ['Tournament Count', formatConfigNumber(config.tournament_count, 2)],
            ['Round Budget', logarithmicRounds ? 'Logarithmic Swiss rounds' : 'Legacy: group size minus one'],
            ['Full Tournament Rounds', formatConfigInteger(config.full_tournament_rounds)],
            ['Target Tournament Rounds', formatConfigInteger(config.tournament_rounds)],
            ['Rounds per Island', localRoundCounts.length ? localRoundCounts.join(', ') : 'Legacy global sizing'],
        ]),
        renderConfigSection('Breeding and Selection', [
            ['Mutation Rate', formatConfigNumber(config.mutation_rate, 2)],
            ['Births from Scratch', formatConfigNumber(config.independent_birth_rate ?? 0, 2)],
            ['Single-parent Births', formatConfigNumber(config.single_parent_birth_rate ?? 0, 2)],
            ['Crossover Births', formatConfigNumber(Math.max(0, 1 - (config.independent_birth_rate ?? 0) - (config.single_parent_birth_rate ?? 0)), 2)],
            ['Diversity Representation', config.diversity_representation === 'criterion_summaries' ? 'Per-criterion summaries' : 'Full text (legacy)'],
            ['Replacement Rate', formatConfigNumber(config.replacement_rate, 2)],
            ['Fitness Alpha', formatConfigNumber(config.fitness_alpha, 2)],
            ['Age Decay Rate', formatConfigNumber(config.age_decay_rate, 2)],
            ['Age Decay Floor', formatConfigNumber(config.age_decay_floor, 2)],
            ['Refine All Ideas', formatConfigBoolean(config.refine_all_ideas)],
            ['Survivor Refinement', formatConfigNumber(config.survivor_refinement_rate, 2)],
        ]),
        renderConfigSection('Islands', [
            ['Island Count', formatConfigInteger(config.island_count)],
            ['Migration Probability', formatConfigNumber(config.migration_rate, 2)],
            ['Migration Novelty Protection', formatConfigNumber(config.migration_novelty_protection_quantile, 2)],
        ]),
    ];

    return `<div class="history-config-details">${sections.join('')}</div>`;
}

function updateBirthMixHelp() {
    const fresh = document.getElementById('independentBirthRate');
    const single = document.getElementById('singleParentBirthRate');
    const help = document.getElementById('birthMixHelp');
    if (!fresh || !single || !help) return;
    const a = Number(fresh.value), b = Number(single.value);
    const valid = Number.isFinite(a) && Number.isFinite(b) && a >= 0 && b >= 0 && a + b <= 1;
    const error = valid ? '' : 'Birth fractions must be between 0 and 1 and add up to at most 1.';
    fresh.setCustomValidity(error);
    single.setCustomValidity(error);
    help.textContent = valid
        ? `Of new births: ${(a*100).toFixed(0)}% fresh, ${(b*100).toFixed(0)}% single-parent, ${((1-a-b)*100).toFixed(0)}% crossover. Small counts are rounded across the whole generation. Oracle adds one direct replacement per generation.`
        : error;
}

function hasSavedConfig(config) {
    return Boolean(config && typeof config === 'object' && Object.keys(config).length > 0);
}

function toSafeCssToken(value, fallback = 'unknown') {
    const token = String(value || fallback).toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
    return token || fallback;
}

function findHistoryItemElement(itemId) {
    return Array.from(document.querySelectorAll('#historyList .history-item'))
        .find(element => element.dataset.itemId === String(itemId)) || null;
}

function renderRecentHistory(items) {
    const container = document.getElementById('recentHistoryList');
    if (!container) return;
    container.replaceChildren();
    if (!items.length) {
        const hint = document.createElement('p');
        hint.className = 'recent-history-empty';
        hint.textContent = 'Start your first experiment. Saved runs will be kept here for comparison.';
        container.appendChild(hint);
        return;
    }
    items.slice(0, 3).forEach(item => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'recent-experiment';
        const name = document.createElement('strong');
        name.textContent = item.display_name || item.name || item.id;
        const meta = document.createElement('small');
        meta.textContent = `${formatStatus(item.status)} · Generation ${item.generations ?? 0}`;
        button.append(name, meta);
        button.addEventListener('click', () => {
            selectedHistoryItem = item;
            renderHistoryList(historyItems, false);
            void loadHistoryItemData(item);
            window.IdeaWorkbench?.closeSidebar();
        });
        container.appendChild(button);
    });
}

function renderHistoryList(items, updateGlobal = true) {
    renderRecentHistory(items);
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
        const statusClass = `badge-status-${toSafeCssToken(item.status)}`;
        const typeClass = `item-type-${toSafeCssToken(item.type)}`;
        const date = item.timestamp ? new Date(item.timestamp).toLocaleDateString() : '';
        const time = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        const isSelected = String(selectedHistoryItem?.id ?? '') === String(item.id ?? '');
        const canUseConfig = !item.execution_archived && (item.can_use_config || hasSavedConfig(item.config));

        // Truncate display name (more room in expanded view)
        let displayName = String(item.display_name || item.id || 'Untitled');
        if (displayName.length > 35) {
            displayName = displayName.substring(0, 32) + '...';
        }

        // Build action buttons for expanded state
        let actionsHtml = '';
        if (isSelected && item.execution_archived) actionsHtml += '<p class="small text-muted mb-2">Archived experiment · saved results remain available.</p>';
        if (isSelected) {
            if (item.can_resume && !item.execution_archived) {
                actionsHtml += `<button class="btn btn-sm btn-success" data-history-action="resume">
                    <i class="fas fa-play me-1"></i>Resume
                </button>`;
            }
            if (canUseConfig) {
                actionsHtml += `<button class="btn btn-sm btn-outline-success" data-history-action="use-config">
                    <i class="fas fa-sliders-h me-1"></i>Use Config
                </button>`;
            }
            if (item.can_continue && !item.execution_archived) {
                actionsHtml += `
                    <div class="continue-action-group d-flex gap-1">
                        <button class="btn btn-sm btn-info" data-history-action="continue">
                            <i class="fas fa-forward me-1"></i>Continue
                        </button>
                    </div>`;
            }
            if (item.can_rate) {
                actionsHtml += `<button class="btn btn-sm btn-outline-primary" data-history-action="rate">
                    <i class="fas fa-star me-1"></i>Rate
                </button>`;
            }
            if (item.can_rename) {
                actionsHtml += `<button class="btn btn-sm btn-outline-secondary" data-history-action="rename" title="Rename">
                    <i class="fas fa-edit"></i>
                </button>`;
            }
            if (item.can_delete) {
                actionsHtml += `<button class="btn btn-sm btn-outline-danger" data-history-action="delete" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>`;
            }
        }

        return `
            <div class="history-item ${isSelected ? 'selected expanded' : ''}"
                 data-item-id="${escapeHtml(item.id)}">
                <button type="button" class="history-item-select" aria-expanded="${isSelected ? 'true' : 'false'}">
                    <span class="item-header">
                        <span class="item-title" title="${escapeHtml(item.display_name || item.id)}">${escapeHtml(displayName)}</span>
                        <span class="item-type-badge ${typeClass}">${escapeHtml(item.type)}</span>
                    </span>
                    <span class="item-meta">
                        <span><i class="fas fa-calendar"></i> ${escapeHtml(date)} ${escapeHtml(time)}</span>
                        <span><i class="fas fa-layer-group"></i> Gen ${escapeHtml(item.generations)}${item.total_generations ? '/' + escapeHtml(item.total_generations) : ''}</span>
                    </span>
                    <span class="item-footer">
                        <span class="badge ${statusClass}">${escapeHtml(formatStatus(item.status))}</span>
                        ${isSelected ? `<span class="idea-type"><i class="fas fa-tag me-1"></i>${escapeHtml(item.idea_type || 'Unknown')}</span>` : ''}
                    </span>
                </button>
                ${isSelected ? buildEvolutionConfigDetailsHtml(item.config || {}, item) : ''}
                ${isSelected ? `<div class="item-actions">${actionsHtml}</div>` : ''}
            </div>
        `;
    }).join('');

    container.querySelectorAll('.history-item-select').forEach(button => {
        button.addEventListener('click', () => {
            const element = button.closest('.history-item');
            if (element) selectHistoryItem(element.dataset.itemId);
        });
    });
    container.querySelectorAll('[data-history-action]').forEach(button => {
        button.addEventListener('click', event => {
            event.stopPropagation();
            const element = button.closest('.history-item');
            const itemId = element?.dataset.itemId;
            const item = historyItems.find(candidate => String(candidate.id) === itemId);
            if (!item) return;

            const actions = {
                resume: () => handleHistoryResume(item.id),
                'use-config': () => handleUseHistoryConfig(item.id),
                continue: () => showContinueInputInline(item.id, item.type),
                rate: () => navigateToRatePage(item.type === 'legacy_saved' ? item.id : ''),
                rename: () => showRenameInputInline(item.id),
                delete: () => confirmDeleteEvolution(item.id),
            };
            actions[button.dataset.historyAction]?.();
        });
    });
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
    const item = historyItems.find(i => String(i.id) === String(itemId));
    if (!item) return;

    // Toggle selection - clicking same item deselects
    if (String(selectedHistoryItem?.id ?? '') === String(itemId)) {
        selectedHistoryItem = null;
    } else {
        selectedHistoryItem = item;
    }

    // Re-render list with new selection state (expands/collapses in place)
    renderHistoryList(historyItems, false);
    findHistoryItemElement(itemId)?.querySelector('.history-item-select')?.focus();

    // Load the evolution data into the main content area if selected
    if (selectedHistoryItem) {
        await loadHistoryItemData(item);
    }
}

// Show inline continue input (for history list items)
function showContinueInputInline(itemId, itemType) {
    const group = findHistoryItemElement(itemId)?.querySelector('.continue-action-group');
    if (!group) return;

    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'form-control form-control-sm continue-generations-input';
    input.value = '3';
    input.min = '1';
    input.max = '20';
    input.style.width = '55px';
    input.style.textAlign = 'center';
    input.addEventListener('click', event => event.stopPropagation());

    const confirmButton = document.createElement('button');
    confirmButton.type = 'button';
    confirmButton.className = 'btn btn-sm btn-info';
    confirmButton.setAttribute('aria-label', 'Continue evolution');
    confirmButton.innerHTML = '<i class="fas fa-play"></i>';
    confirmButton.addEventListener('click', event => {
        event.stopPropagation();
        executeContinueInline(itemId, itemType);
    });

    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.className = 'btn btn-sm btn-outline-secondary';
    cancelButton.setAttribute('aria-label', 'Cancel');
    cancelButton.innerHTML = '<i class="fas fa-times"></i>';
    cancelButton.addEventListener('click', event => {
        event.stopPropagation();
        cancelContinueInputInline(itemId, itemType);
    });

    group.replaceChildren(input, confirmButton, cancelButton);

    setTimeout(() => {
        input.focus();
        input.select();
    }, 50);
}

function cancelContinueInputInline(itemId, itemType) {
    const group = findHistoryItemElement(itemId)?.querySelector('.continue-action-group');
    if (!group) return;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-sm btn-info';
    button.innerHTML = '<i class="fas fa-forward me-1"></i>Continue';
    button.addEventListener('click', event => {
        event.stopPropagation();
        showContinueInputInline(itemId, itemType);
    });
    group.replaceChildren(button);
}

async function executeContinueInline(itemId, itemType) {
    const input = findHistoryItemElement(itemId)?.querySelector('.continue-generations-input');
    const additionalGens = parseInt(input?.value || '3');

    if (isNaN(additionalGens) || additionalGens < 1) {
        alert("Please enter a valid number of generations (1 or more)");
        return;
    }

    await handleHistoryContinue(itemId, itemType, additionalGens);
}

// Show inline rename input (for history list items)
function showRenameInputInline(itemId) {
    const titleEl = findHistoryItemElement(itemId)?.querySelector('.item-title');
    if (!titleEl) return;

    const currentName = titleEl.textContent;
    const parent = titleEl.parentNode;

    const group = document.createElement('div');
    group.className = 'rename-inline-group d-flex gap-1 flex-grow-1';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-control form-control-sm rename-input';
    input.value = selectedHistoryItem?.display_name || currentName;
    input.addEventListener('click', event => event.stopPropagation());

    const confirmButton = document.createElement('button');
    confirmButton.type = 'button';
    confirmButton.className = 'btn btn-sm btn-success';
    confirmButton.setAttribute('aria-label', 'Save name');
    confirmButton.innerHTML = '<i class="fas fa-check"></i>';
    confirmButton.addEventListener('click', event => {
        event.stopPropagation();
        executeRenameInline(itemId);
    });

    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.className = 'btn btn-sm btn-outline-secondary';
    cancelButton.setAttribute('aria-label', 'Cancel rename');
    cancelButton.innerHTML = '<i class="fas fa-times"></i>';
    cancelButton.addEventListener('click', event => {
        event.stopPropagation();
        cancelRenameInline(itemId, currentName);
    });

    group.append(input, confirmButton, cancelButton);
    parent.replaceChild(group, titleEl);

    setTimeout(() => {
        input.focus();
        input.select();
    }, 50);
}

function cancelRenameInline(itemId, originalName) {
    const group = findHistoryItemElement(itemId)?.querySelector('.rename-inline-group');
    if (group) {
        const title = document.createElement('span');
        title.className = 'item-title';
        title.title = originalName;
        title.textContent = originalName;
        group.replaceWith(title);
    }
}

async function executeRenameInline(itemId) {
    const input = findHistoryItemElement(itemId)?.querySelector('.rename-input');
    const newName = input?.value?.trim();

    if (!newName) {
        alert('Please enter a name');
        return;
    }

    try {
        const response = await fetch(`/api/evolution/${encodeURIComponent(itemId)}/rename`, {
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

function setFormControlValue(id, value, formatter = null) {
    if (value === null || value === undefined) return;
    const el = document.getElementById(id);
    if (!el) return;
    el.value = String(value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));

    if (formatter) {
        const valueEl = document.getElementById(`${id}Value`);
        if (valueEl) valueEl.textContent = formatter(value);
    }
}

function setCheckboxValue(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.checked = value === true;
    el.dispatchEvent(new Event('change', { bubbles: true }));
}

function applyThinkingConfigToForm(config) {
    updateThinkingBudgetVisibility();

    const thinkingBudget = config.thinking_budget;
    const thinkingLevel = config.thinking_level;
    const customRadio = document.getElementById('thinkingCustomBudget');
    const offRadio = document.getElementById('thinkingOff');
    const lowRadio = document.getElementById('thinkingLow');
    const mediumRadio = document.getElementById('thinkingMedium');
    const highRadio = document.getElementById('thinkingHigh');
    const xhighRadio = document.getElementById('thinkingXHigh');
    const maxRadio = document.getElementById('thinkingMax');
    const budgetSlider = document.getElementById('thinkingBudgetSlider');

    if (thinkingBudget !== null && thinkingBudget !== undefined && customRadio && budgetSlider) {
        customRadio.checked = true;
        budgetSlider.value = String(thinkingBudget);
        updateThinkingBudgetMode();
        updateThinkingBudgetDisplay();
        return;
    }

    const normalizedLevel = String(thinkingLevel || '').toLowerCase();
    if (['off', 'none', 'minimal'].includes(normalizedLevel) && offRadio) {
        offRadio.checked = true;
    } else if (normalizedLevel === 'low' && lowRadio) {
        lowRadio.checked = true;
    } else if (normalizedLevel === 'medium' && mediumRadio) {
        mediumRadio.checked = true;
    } else if (normalizedLevel === 'high' && highRadio) {
        highRadio.checked = true;
    } else if (normalizedLevel === 'xhigh' && xhighRadio) {
        xhighRadio.checked = true;
    } else if (normalizedLevel === 'max' && maxRadio) {
        maxRadio.checked = true;
    }
    updateThinkingBudgetMode();
}

function applyEvolutionConfigToForm(config = {}, sourceName = '') {
    if (!config || Object.keys(config).length === 0) {
        alert('This evolution does not have saved config metadata.');
        return false;
    }

    if (config.idea_type) {
        const selected = typeof selectTemplate === 'function' ? selectTemplate(config.idea_type, false) : false;
        if (!selected) {
            const ideaTypeInput = document.getElementById('ideaType');
            if (ideaTypeInput) ideaTypeInput.value = config.idea_type;
            const display = document.getElementById('ideaTypeDisplay');
            if (display) {
                display.textContent = config.idea_type;
                display.classList.remove('text-muted', 'fst-italic');
            }
        }
    }

    setFormControlValue('popSize', config.pop_size);
    setFormControlValue('generations', config.generations);
    if (config.model_type && !window.IdeaWorkbench?.selectModel('modelType', config.model_type, { fallback: true })) {
        const modelSelect = document.getElementById('modelType');
        if (Array.from(modelSelect?.options || []).some(option => option.value === config.model_type)) {
            setFormControlValue('modelType', config.model_type);
        } else {
            addActivityLogItem(`Saved model ${config.model_type} is no longer available. Using ${modelSelect?.selectedOptions?.[0]?.textContent || modelSelect?.value || 'the current model'}.`, 'warning');
        }
    }
    setFormControlValue('creativeTemp', config.creative_temp, value => String(value));
    setFormControlValue('topP', config.top_p, value => Number(value).toFixed(2));
    setFormControlValue('seedContextPoolSize', config.seed_context_pool_size, value => String(parseInt(value, 10)));
    setFormControlValue('tournamentCount', config.tournament_count, value => Number(value).toFixed(2));
    setFormControlValue('mutationRate', config.mutation_rate, value => Number(value).toFixed(2));
    setFormControlValue('independentBirthRate', config.independent_birth_rate ?? 0.5, value => Number(value).toFixed(2));
    setFormControlValue('singleParentBirthRate', config.single_parent_birth_rate ?? 0.25, value => Number(value).toFixed(2));
    setFormControlValue('diversityRepresentation', config.diversity_representation ?? 'criterion_summaries');
    updateBirthMixHelp();
    setFormControlValue('replacementRate', config.replacement_rate, value => Number(value).toFixed(2));
    setFormControlValue('fitnessAlpha', config.fitness_alpha, value => Number(value).toFixed(2));
    setFormControlValue('ageDecayRate', config.age_decay_rate, value => Number(value).toFixed(2));
    setFormControlValue('ageDecayFloor', config.age_decay_floor, value => Number(value).toFixed(2));
    setFormControlValue('survivorRefinementRate', config.survivor_refinement_rate, value => Number(value).toFixed(2));
    setFormControlValue('islandCount', config.island_count, value => String(parseInt(value, 10)));
    setFormControlValue('migrationRate', config.migration_rate, value => Number(value).toFixed(2));
    setFormControlValue('migrationNoveltyProtectionQuantile', config.migration_novelty_protection_quantile, value => Number(value).toFixed(2));

    const maxBudgetEl = document.getElementById('maxBudget');
    if (maxBudgetEl) {
        maxBudgetEl.value = config.max_budget === null || config.max_budget === undefined ? '' : String(config.max_budget);
    }

    setCheckboxValue('refineAllIdeas', config.refine_all_ideas);
    const artifactSwitch = document.getElementById('enableVisualizationArtifacts');
    if (artifactSwitch) {
        const components = Array.isArray(config.evaluation?.components)
            ? config.evaluation.components
            : [];
        const hasArtifactProducer = components.some(component => (
            component?.enabled !== false && component?.kind === 'llm_code_artifact'
        ));
        artifactSwitch.checked = !artifactSwitch.disabled && hasArtifactProducer;
    }
    updateSurvivorRefinementControls();
    applyThinkingConfigToForm(config);
    tournamentCountTouched = true;

    const nameSection = document.getElementById('evolutionNameSection');
    if (nameSection) nameSection.style.display = 'block';
    const nameInput = document.getElementById('evolutionNameInput');
    if (nameInput) {
        nameInput.value = '';
        if (sourceName) nameInput.placeholder = `New run from ${sourceName}`;
    }

    const templateEntry = document.getElementById('templateEntry');
    if (templateEntry) templateEntry.style.display = 'block';

    showSidebarView('main');
    addActivityLogItem(`Loaded config from ${sourceName || 'saved evolution'}`, 'info');
    return true;
}

async function getHistoryItemConfig(itemId) {
    const item = historyItems.find(i => i.id === itemId);
    if (!item) return null;
    if (hasSavedConfig(item.config)) {
        return item.config;
    }
    if (currentEvolutionData?.evolution_id === itemId && currentEvolutionData?.config) {
        item.config = currentEvolutionData.config;
        item.can_use_config = true;
        return item.config;
    }
    if (item.type === 'evolution' || item.type === 'saved') {
        const response = await fetch(`/api/evolution/${itemId}`);
        if (response.ok) {
            const data = await response.json();
            const config = data?.data?.config;
            if (config && Object.keys(config).length > 0) {
                item.config = config;
                item.can_use_config = true;
                return config;
            }
        }
    }
    return null;
}

async function handleUseHistoryConfig(itemId) {
    try {
        const item = historyItems.find(i => i.id === itemId);
        const config = await getHistoryItemConfig(itemId);
        if (!config) {
            alert('This evolution does not have saved config metadata.');
            return;
        }
        applyEvolutionConfigToForm(config, item?.display_name || itemId);
    } catch (error) {
        console.error('Error applying saved config:', error);
        alert('Unable to apply saved config: ' + error.message);
    }
}

async function loadHistoryItemData(item) {
    try {
        if (item.type === 'saved' || item.type === 'evolution' || item.type === 'legacy_checkpoint') {
            // Load from saved evolution file or Firestore
            const response = await fetch(`/api/evolution/${item.id}`);
            if (response.ok) {
                const data = await response.json();
                let evolutionData = data.data;
                // Legacy checkpoints stay read-only until Resume creates a copy.
                const cachedEvolution = item.type === 'legacy_checkpoint'
                    ? null : await repairCurrentEvolutionIfNewer(item.id, evolutionData);
                if (cachedEvolution) {
                    evolutionData = cachedEvolution;
                }
                if (evolutionData && evolutionData.history) {
                    clearTokenCounts();
                    // Set identity before rendering because artifact previews
                    // build authenticated URLs synchronously during card creation.
                    currentEvolutionId = data.id || evolutionData.evolution_id || item.id;
                    currentEvolutionName = data.name
                        || evolutionData.evolution_name
                        || evolutionData.name
                        || item.display_name
                        || null;
                    evolutionData.evolution_id = currentEvolutionId;
                    evolutionData.evolution_name = currentEvolutionName;

                    if (evolutionData.config) {
                        item.config = evolutionData.config;
                        item.can_use_config = true;
                        renderHistoryList(historyItems, false);
                    }
                    // Hide the template entry section
                    const templateEntry = document.getElementById('templateEntry');
                    if (templateEntry) templateEntry.style.display = 'none';
                    const library = document.getElementById('existingTemplates');
                    if (library) library.style.display = 'none';
                    document.getElementById('newEvolutionBtn')?.style.setProperty('display', 'inline-flex');
                    window.IdeaWorkbench?.closeInspector();

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
                    selectedGeneId = null;
                    geneScope = 'current';
                    applyGeneCatalogue(evolutionData, true);
                    renderGenerations(evolutionData.history);
                    currentEvolutionData = evolutionData;

                    tournamentHistory = evolutionData.tournament_history || [];
                    renderTournamentDetails(tournamentHistory);

                    if ((evolutionData.diversity_history && evolutionData.diversity_history.length > 0)
                        || (evolutionData.history && evolutionData.history.length > 0)) {
                        const chartUpdated = updateDiversityChart(
                            evolutionData.diversity_history || [],
                            evolutionData.history,
                            evolutionData.island_diversity_history || []
                        );
                        if (chartUpdated) {
                            setTimeout(ensureDiversityChartSizing, 200);
                        }
                    } else {
                        resetDiversityPlot();
                    }

                    if (evolutionData.contexts) {
                        contexts = evolutionData.contexts;
                        specificPrompts = evolutionData.specific_prompts || [];
                        breedingPrompts = evolutionData.breeding_prompts || [];
                        currentContextIndex = 0;
                        updateContextDisplay();
                        document.querySelector('.context-navigation').style.display = 'block';
                    }

                    if (evolutionData.token_counts) {
                        displayTokenCounts(evolutionData.token_counts);
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
                    clearTokenCounts();
                    document.getElementById('generations-container').innerHTML = '';
                    selectedGeneId = null;
                    geneScope = 'current';
                    applyGeneCatalogue(data, true);
                    renderGenerations(data.history);
                    currentEvolutionData = data;

                    tournamentHistory = data.tournament_history || [];
                    renderTournamentDetails(tournamentHistory);

                    if ((data.diversity_history && data.diversity_history.length > 0)
                        || (data.history && data.history.length > 0)) {
                        const chartUpdated = updateDiversityChart(
                            data.diversity_history || [],
                            data.history,
                            data.island_diversity_history || []
                        );
                        if (chartUpdated) {
                            setTimeout(ensureDiversityChartSizing, 200);
                        }
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
            rememberCheckpointId(data.checkpoint_id || checkpointId);

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
    await repairCurrentEvolutionIfNewer(itemId);

    // Update the inline action button to show starting state
    const group = findHistoryItemElement(itemId)?.querySelector('.continue-action-group');
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
            rememberCheckpointId(data.checkpoint_id || itemId);

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
        el.querySelector('.history-item-select')?.setAttribute('aria-expanded', 'false');
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

let progressPollTimer = null;
let progressPollInFlight = false;
let progressPollRequested = false;
let progressConnectionLost = false;

function scheduleProgressPoll(delay) {
    clearTimeout(progressPollTimer);
    progressPollTimer = setTimeout(pollProgress, delay);
}

async function pollProgress() {
    if (progressPollInFlight) { progressPollRequested = true; return; }
    clearTimeout(progressPollTimer);
    progressPollInFlight = true;
    try {
        const response = await fetch('/api/progress');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        let data = await response.json();
        if (progressConnectionLost) {
            progressConnectionLost = false;
            document.getElementById('workspaceNotice')?.remove();
            addActivityLogItem('Connection restored. Progress is up to date.', 'info');
        }

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
                progressBar.setAttribute('aria-valuetext', `${Math.round(progress)}% complete`);
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
            currentEvolutionData = {
                ...(currentEvolutionData || {}),
                ...data,
                history: data.history,
                diversity_history: data.diversity_history || currentEvolutionData?.diversity_history || [],
                island_diversity_history: data.island_diversity_history || currentEvolutionData?.island_diversity_history || [],
                migration_history: data.migration_history || currentEvolutionData?.migration_history || [],
                tournament_history: data.tournament_history || currentEvolutionData?.tournament_history || tournamentHistory
            };
            if (rendered) {
                const historyVersionToStore = Number.isInteger(data.history_version) ? data.history_version : null;
                const shouldStore =
                    !!data.gene_catalogue ||
                    historyVersionToStore === null ||
                    historyVersionToStore !== lastStoredHistoryVersion;
                if (shouldStore) {
                    const evolutionStateToStore = {
                        history: data.history,
                        history_version: historyVersionToStore ?? lastRenderedHistoryVersion,
                        diversity_history: data.diversity_history || [],
                        island_diversity_history: data.island_diversity_history || [],
                        migration_history: data.migration_history || [],
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
                handleDiversityUpdate(data);
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
            scheduleProgressPoll(1500);
        } else {
            // Evolution complete or stopped
            isEvolutionRunning = false;

            // Mark as complete with proper visual feedback
            // Pass resumable state and checkpoint ID for pause/resume functionality
            markEvolutionComplete(
                data.is_stopped,
                data.is_resumable || false,
                data.checkpoint_id || null,
                data.error || (data.status === 'error' || data.status === 'failed' ? 'This experiment failed. Open History to inspect saved progress.' : null),
                data.stop_reason || null
            );

            if (data.history && data.history.length > 0) {
                // Save final state and enable save button
                currentEvolutionData = data;
                applyHistoryIfNew(data, true);

                // Store checkpoint ID if available for future resume/continue
                if (data.checkpoint_id) {
                    rememberCheckpointId(data.checkpoint_id);
                }

                // Store the final evolution data in localStorage including diversity data and token counts
                const evolutionStateToStore = {
                    history: data.history,
                    history_version: Number.isInteger(data.history_version) ? data.history_version : lastRenderedHistoryVersion,
                    diversity_history: data.diversity_history || [],
                    island_diversity_history: data.island_diversity_history || [],
                    migration_history: data.migration_history || [],
                    contexts: data.contexts || contexts,
                    specific_prompts: data.specific_prompts || specificPrompts,
                    breeding_prompts: data.breeding_prompts || breedingPrompts,
                    tournament_history: data.tournament_history || tournamentHistory,
                    token_counts: data.token_counts || null,
                    config: data.config || currentEvolutionData?.config || null,
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

            resetButtonStates();
            if (data.error || ['error', 'failed'].includes(data.status)) {
                showErrorMessage(`${data.error || 'The experiment failed.'} ${data.is_resumable ? 'Open History to resume from the saved checkpoint.' : 'Open History to inspect any saved progress.'}`);
            }

            // Reset tracking variables for next evolution
            previousStatusMessage = '';
            previousProgress = 0;
            await retryDeferredEvolutionRepairs(data);
        }

    } catch (error) {
        console.error('Error polling progress:', error);
        if (isEvolutionRunning) {
            if (!progressConnectionLost) {
                progressConnectionLost = true;
                showErrorMessage('Connection lost. The experiment may still be running. Reconnecting automatically…');
                updateEvolutionStatusIndicator(true, 'Reconnecting…');
            }
            scheduleProgressPoll(4000);
        }
    } finally {
        progressPollInFlight = false;
        if (progressPollRequested && isEvolutionRunning) scheduleProgressPoll(0);
        progressPollRequested = false;
    }
}

function apiErrorMessage(data, fallback) {
    if (typeof data?.message === 'string') return data.message;
    if (typeof data?.detail === 'string') return data.detail;
    if (Array.isArray(data?.detail)) {
        return data.detail.map(item => `${(item.loc || []).filter(part => part !== 'body').join(' → ')}: ${item.msg}`).join('; ');
    }
    return fallback;
}

function showErrorMessage(message) {
    let notice = document.getElementById('workspaceNotice');
    if (!notice) {
        notice = document.createElement('div');
        notice.id = 'workspaceNotice';
        notice.className = 'workspace-notice alert alert-danger';
        notice.setAttribute('role', 'alert');
        const text = document.createElement('span');
        text.className = 'workspace-notice-text';
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'btn-close';
        close.setAttribute('aria-label', 'Dismiss message');
        close.addEventListener('click', () => notice.remove());
        notice.append(text, close);
        (document.getElementById('mainContent') || document.body).prepend(notice);
    }
    notice.querySelector('.workspace-notice-text').textContent = message;
    notice.scrollIntoView({ block: 'nearest' });
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
        modal.setAttribute('aria-labelledby', 'alreadyRunningModalLabel');
        modal.setAttribute('aria-describedby', 'alreadyRunningModalDescription');
        modal.setAttribute('aria-hidden', 'true');
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-warning text-dark">
                        <h5 class="modal-title" id="alreadyRunningModalLabel">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Evolution Already Running
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p id="alreadyRunningModalDescription">You already have an evolution in progress. You can only run one evolution at a time.</p>
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
    rememberModalFocusOrigin(modal);
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
            await retryDeferredEvolutionRepairs(progressData);
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
            background: var(--primary);
            color: var(--text-on-primary);
            padding: 10px 20px;
            z-index: 9998;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            box-shadow: 0 2px 10px color-mix(in srgb, var(--text-primary) 30%, transparent);
            font-size: 14px;
            font-family: Figtree, sans-serif;
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
        rememberCheckpointId(checkpointId);
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
                    rememberCheckpointId(resumable.id);
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

function clearTokenCounts() {
    document.getElementById('token-counts-container')?.remove();
    latestDiagnosticsTokenCounts = null;
    lastTokenCountsSignature = '';
}

// Function to reset the UI state
function resetUIState() {
    window.IdeaWorkbench?.closeInspector();
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
    try {
        localStorage.removeItem(`idea.${window.currentUser?.uid || 'signed-out'}.currentEvolutionData`);
    } catch (error) {
        console.warn('Browser backup unavailable; starting with server state.', error);
    }

    // Reset download button
    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        downloadButton.disabled = true;
    }

    // Remove token counts container if it exists
    clearTokenCounts();
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
                <h4 class="evolution-name" id="evolution-name"></h4>
            </div>
            <div class="evolution-status-section">
            <div class="evolution-status-indicator">
                <div class="evolution-spinner" id="evolution-spinner"></div>
                <span id="evolution-status-text" role="status" aria-live="polite" aria-atomic="true">Initializing...</span>
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
                     style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"
                     aria-label="Evolution progress" aria-valuetext="0% complete"></div>
                <div class="progress-bar-activity" id="progress-bar-activity"></div>
            </div>
            <div class="progress-percentage" id="progress-percentage" aria-hidden="true">0%</div>
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
    progressContainer.querySelector('#evolution-name').textContent = evolutionName || 'Evolution';

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
                <div class="tournament-round-title">Round ${escapeHtml(round.round || '')}</div>
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
            <div class="activity-item activity-${toSafeCssToken(item.type, 'info')}">
                <span class="activity-time">${escapeHtml(item.timestamp)}</span>
                <span class="activity-message">${escapeHtml(item.message)}</span>
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
            spinner.classList.remove('complete', 'stopped', 'failed');
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
            progressCard.classList.remove('complete', 'stopped', 'failed');
        }
    }
}

// Function to mark evolution as complete
function markEvolutionComplete(isStopped = false, isResumable = false, checkpointId = null, error = null, stopReason = null) {
    const spinner = document.getElementById('evolution-spinner');
    const statusTextEl = document.getElementById('evolution-status-text');
    const progressCard = document.getElementById('progress-container');
    const activityIndicator = document.getElementById('progress-bar-activity');
    const forceStopButton = document.getElementById('forceStopButton');

    if (spinner) {
        spinner.classList.remove('spinning', 'complete', 'stopped', 'failed');
        spinner.classList.add(error ? 'failed' : isStopped ? 'stopped' : 'complete');
    }

    if (statusTextEl) {
        if (error) {
            statusTextEl.textContent = isResumable ? 'Failed · checkpoint saved' : 'Failed';
        } else if (isStopped && isResumable) {
            statusTextEl.textContent = stopReason === 'budget' ? 'Paused · budget reached' : 'Paused';
        } else if (isStopped) {
            statusTextEl.textContent = 'Stopped';
        } else {
            statusTextEl.textContent = 'Complete';
        }
    }

    if (progressCard) {
        progressCard.classList.remove('running', 'complete', 'stopped', 'failed');
        progressCard.classList.add(error ? 'failed' : isStopped ? 'stopped' : 'complete');
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
    if (error) {
        addActivityLogItem(`Failed after ${elapsed}: ${error}`, 'error');
    } else if (isStopped && isResumable) {
        addActivityLogItem(stopReason === 'budget'
            ? `Paused after ${elapsed}: budget reached; checkpoint saved. Use this configuration to start a new experiment with a larger budget.`
            : `Paused after ${elapsed} · checkpoint saved`, 'info');
    } else if (isStopped) {
        addActivityLogItem(`⏹️ Stopped after ${elapsed}`, 'warning');
    } else {
        addActivityLogItem(`✅ Completed in ${elapsed}!`, 'success');
    }

    // Store checkpoint info for history panel
    if (checkpointId) {
        rememberCheckpointId(checkpointId);
        // Refresh history to show new checkpoint
        if (typeof loadHistory === 'function') {
            loadHistory();
        }
    }
}



// This is a convenience cache; a storage failure must not interrupt a saved run.
function rememberCheckpointId(checkpointId) {
    const owner = window.currentUser?.uid;
    if (!owner || !checkpointId) return;
    try {
        localStorage.setItem(`idea.${owner}.lastCheckpointId`, checkpointId);
    } catch (error) {
        console.warn('Checkpoint saved on the server; browser shortcut unavailable.', error);
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
            .map(item => `<div class="context-item">${escapeHtml(item.trim())}</div>`)
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

            const parsedTargetGeneration = Number(idea.elite_target_generation);
            const targetGeneration = Number.isInteger(parsedTargetGeneration)
                ? parsedTargetGeneration
                : generationIndex;
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
            const parsedSourceGeneration = Number(idea.elite_source_generation);
            const sourceGenerationIndex = Number.isInteger(parsedSourceGeneration) && parsedSourceGeneration >= 0
                ? parsedSourceGeneration
                : null;
            const sourceIdeaId = idea.elite_source_id;

            let sourceIdea = null;
            if (sourceGenerationIndex !== null && sourceIdeaId && generations[sourceGenerationIndex]) {
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
                                <h6 class="card-title">${escapeHtml(sourceIdea.title || 'Untitled')}</h6>
                                <div class="card-preview">
                                    <p>${escapeHtml(sourcePreview)}</p>
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
                    <div class="analysis-text" style="white-space: pre-wrap; line-height: 1.6; background-color: var(--surface-alt); padding: 1rem; border-radius: 0.375rem; border-left: 4px solid var(--primary);"></div>
                </div>
            </div>
        `;

        lineageModalContent.innerHTML = analysisHtml;
        lineageModalContent.querySelector('.analysis-text').textContent = idea.oracle_analysis;

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
                <h6 class="card-title">${escapeHtml(ancestor.title || 'Untitled')}</h6>
                <div class="card-preview">
                    <p>${escapeHtml(preview)}</p>
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
        total_thoughts: tokenCounts.total_thoughts,
        total_cached_input: tokenCounts.total_cached_input,
        cost: tokenCounts.cost,
    });
    if (nextSignature === lastTokenCountsSignature
        && document.getElementById('token-counts-container')?.querySelector('#token-cost-value')) {
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
    const totalThoughtTokens = (tokenCounts.total_thoughts || 0).toLocaleString();
    const totalCachedInputTokens = (tokenCounts.total_cached_input || 0).toLocaleString();
    const totalToolInputTokens = (tokenCounts.total_tool_input || 0).toLocaleString();

    // Get cost information
    const costInfo = tokenCounts.cost;
    const totalCost = costInfo.total_cost.toFixed(4);
    const inputCost = costInfo.input_cost.toFixed(4);
    const outputCost = costInfo.output_cost.toFixed(4);
    const thoughtCost = (costInfo.thought_cost || 0).toFixed(4);
    const cachedInputCost = (costInfo.cached_input_cost || 0).toFixed(4);

    // Prepare alternative model cost estimates
    let estimatesList = '';
    if (tokenCounts.estimates) {
        estimatesList = Object.values(tokenCounts.estimates).map(e => {
            return `<li class="list-group-item d-flex justify-content-between align-items-center">${escapeHtml(e.name)}<span>$${e.cost.toFixed(4)} <small class="text-muted">estimate</small></span></li>`;
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
    const chartTheme = getViewerChartTheme();
    const borderColors = [chartTheme.violet, chartTheme.green, chartTheme.blue,
        chartTheme.coral, chartTheme.amber, chartTheme.text];
    const colors = borderColors.map(color => viewerChartColorWithAlpha(color, 0.7));

    let colorIndex = 0;

    // Keep core agents in their familiar order, then include dynamically
    // registered evaluators such as artifact_render.
    const coreComponents = ['ideator', 'formatter', 'critic', 'breeder', 'genotype_encoder', 'diversity_embeddings', 'oracle'];
    const possibleComponents = [
        ...coreComponents.filter(componentName => tokenCounts[componentName]),
        ...Object.keys(tokenCounts.models || {}).filter(componentName => (
            tokenCounts[componentName] && !coreComponents.includes(componentName)
        )),
    ];
    const componentDisplayNames = {
        genotype_encoder: 'Genotype Encoder',
        diversity_embeddings: 'Diversity Embeddings',
        artifact_render: 'Artifact Render / Code Execution',
        restored_usage: 'Restored Usage',
    };

    for (const componentName of possibleComponents) {
        if (tokenCounts[componentName]) {
            const component = tokenCounts[componentName];
            const defaultDisplayName = componentName
                .split('_')
                .map(part => part.charAt(0).toUpperCase() + part.slice(1))
                .join(' ');
            const displayName = componentDisplayNames[componentName] || defaultDisplayName;
            componentData.push({
                name: displayName,
                displayName,
                total: component.total.toLocaleString(),
                input: component.input.toLocaleString(),
                output: component.output.toLocaleString(),
                thoughts: (component.thoughts || 0).toLocaleString(),
                cachedInput: (component.cached_input || 0).toLocaleString(),
                toolInput: (component.tool_input || 0).toLocaleString(),
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
        const thoughtHtml = comp.thoughts !== '0' ? `<span>Thoughts: ${comp.thoughts}</span> |` : '';
        const cachedHtml = comp.cachedInput !== '0' ? `<span>Cached input: ${comp.cachedInput}</span> |` : '';
        const toolInputHtml = comp.toolInput !== '0' ? `<span>Tool/intermediate input: ${comp.toolInput}</span> |` : '';
        componentBreakdownHtml += `
            <li class="list-group-item">
                <div class="d-flex justify-content-between align-items-center">
                    <span>${escapeHtml(comp.displayName)} <span class="badge bg-secondary">${escapeHtml(comp.model)}</span></span>
                    <span class="badge bg-primary rounded-pill">${comp.total}</span>
                </div>
                <div class="small text-muted mt-1">
                    <span>Input: ${comp.input}</span> |
                    <span>Output: ${comp.output}</span> |
                    ${thoughtHtml}
                    ${cachedHtml}
                    ${toolInputHtml}
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
                                    <div class="small text-muted mt-1">
                                        <span>Thoughts: ${totalThoughtTokens}</span>
                                        ${tokenCounts.total_cached_input ? `<span class="ms-2">Cached input: ${totalCachedInputTokens}</span>` : ''}
                                        ${tokenCounts.total_tool_input ? `<span class="ms-2">Tool/intermediate input: ${totalToolInputTokens}</span>` : ''}
                                    </div>
                                    <hr>
                                    <h6>Cost Breakdown:</h6>
                                    <p class="mb-1">Total cost: <strong>$${totalCost}</strong></p>
                                    <div class="small text-muted">
                                        <div>Input: $${inputCost}</div>
                                        ${tokenCounts.total_cached_input ? `<div>Cached input read: $${cachedInputCost}</div>` : ''}
                                        <div>Output: $${outputCost}</div>
                                        <div>Thoughts: $${thoughtCost}</div>
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

        // Create a third chart for input/output/thinking cost breakdown
        const costCtx = document.getElementById('tokenCostChart').getContext('2d');
        new Chart(costCtx, {
            type: 'bar',
            data: {
                labels: ['Input Cost', 'Output Cost', 'Thinking Cost', 'Total Cost'],
                datasets: [{
                    label: 'Cost (USD)',
                    data: [
                        parseFloat(inputCost),
                        parseFloat(outputCost),
                        parseFloat(thoughtCost),
                        parseFloat(totalCost)
                    ],
                    backgroundColor: [chartTheme.violet, chartTheme.green, chartTheme.blue, chartTheme.coral],
                    borderColor: [chartTheme.violet, chartTheme.green, chartTheme.blue, chartTheme.coral],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Input vs Output vs Thinking Cost'
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
let diversityData = createEmptyFitnessPlotData();
let fitnessPlotMetric = 'fitness';
let fitnessPlotScope = 'population';

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
    const metricConfig = getFitnessPlotMetricConfig(fitnessPlotMetric);
    return {
        x: {
            title: {
                display: true,
                text: 'Generation',
                color: getViewerChartTheme().secondary,
                font: {
                    size: 14,
                    weight: 'bold'
                }
            },
            grid: {
                color: viewerChartColorWithAlpha(getViewerChartTheme().violet, 0.1),
                lineWidth: 1
            },
            ticks: {
                color: getViewerChartTheme().secondary,
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
                text: metricConfig.axisLabel,
                color: getViewerChartTheme().secondary,
                font: {
                    size: 14,
                    weight: 'bold'
                }
            },
            grid: {
                color: viewerChartColorWithAlpha(getViewerChartTheme().violet, 0.1),
                lineWidth: 1
            },
            ticks: {
                color: getViewerChartTheme().secondary,
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
 * Switch between population metric views.
 */
function setFitnessPlotMetric(metric) {
    const nextMetric = normalizeFitnessPlotMetric(metric);
    const button = document.querySelector(`[data-fitness-metric="${nextMetric}"]`);
    if (button?.disabled) return;
    fitnessPlotMetric = nextMetric;
    if (fitnessPlotMetric !== 'diversity') {
        fitnessPlotScope = 'population';
    }
    document.querySelectorAll('[data-fitness-metric]').forEach((button) => {
        button.classList.toggle('active', button.dataset.fitnessMetric === nextMetric);
    });
    updateFitnessScopeAvailability(diversityData);

    if (!diversityChart) {
        return;
    }

    diversityChart.data.datasets = getDiversityChartDatasets(diversityData, nextMetric, fitnessPlotScope);
    diversityChart.options.scales = getDiversityChartScales();
    renderDiversitySummary(diversityData.latest, diversityData);
    lastDiversityRenderSignature = buildDiversitySignature(diversityData);
    diversityChart.update('none');
}

function setDiversityChartMode(mode) {
    if (mode === 'islands') {
        setFitnessPlotMetric('diversity');
        setFitnessPlotScope('islands');
        return;
    }
    if (mode === 'overview') {
        setFitnessPlotMetric('diversity');
        setFitnessPlotScope('population');
        return;
    }
    setFitnessPlotMetric(mode);
}

function setFitnessPlotScope(scope) {
    const nextScope = normalizeFitnessPlotScope(scope);
    if (nextScope === 'islands' && (fitnessPlotMetric !== 'diversity' || !hasIslandDiversityData(diversityData))) {
        return;
    }
    fitnessPlotScope = nextScope;
    updateFitnessScopeAvailability(diversityData);

    if (!diversityChart) {
        return;
    }

    diversityChart.data.datasets = getDiversityChartDatasets(diversityData, fitnessPlotMetric, fitnessPlotScope);
    diversityChart.options.scales = getDiversityChartScales();
    renderDiversitySummary(diversityData.latest, diversityData);
    lastDiversityRenderSignature = buildDiversitySignature(diversityData);
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

    // Progress can arrive before the deferred page-load initializer. Keep the
    // populated chart instead of replacing it with an empty generation axis.
    if (diversityChart?.canvas === canvas) return;

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
                labels: [...(diversityData?.labels || [])],
                datasets: getDiversityChartDatasets(diversityData, fitnessPlotMetric, fitnessPlotScope)
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
                            color: getViewerChartTheme().secondary
                        }
                    },
                    tooltip: {
                        backgroundColor: getViewerChartTheme().surface,
                        titleColor: getViewerChartTheme().text,
                        bodyColor: getViewerChartTheme().secondary,
                        borderColor: getViewerChartTheme().border,
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
                                return `${context.dataset.label}: ${formatFitnessPlotMetric(value, fitnessPlotMetric)}`;
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
    if (value === null || value === undefined) return null;
    if (typeof value === 'string' && value.trim() === '') return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function toFiniteDiversityInteger(value) {
    const numeric = toFiniteDiversityNumber(value);
    return Number.isInteger(numeric) ? numeric : null;
}

function firstFiniteDiversityValue(...values) {
    for (const value of values) {
        const numeric = toFiniteDiversityNumber(value);
        if (numeric !== null) return numeric;
    }
    return null;
}

function getSnapshotGeneration(diversityData, index) {
    const generation = toFiniteDiversityInteger(firstFiniteDiversityValue(
        diversityData?.analytics?.current_generation,
        diversityData?.current_generation,
        diversityData?.generation
    ));
    return generation !== null ? generation : index;
}

function normalizeIslandDiversityHistory(islandDiversityHistory) {
    return Array.isArray(islandDiversityHistory) ? islandDiversityHistory : [];
}

function getIslandDiversityRows(snapshot) {
    const candidates = [
        snapshot?.current_islands,
        snapshot?.current_island_diversities,
        snapshot?.analytics?.islands,
        snapshot?.islands
    ];
    for (const candidate of candidates) {
        if (Array.isArray(candidate)) return candidate;
    }
    return [];
}

function snapshotHasIslandDiversity(snapshot) {
    if (!snapshot || snapshot.enabled === false || snapshot.error) return false;
    return (
        toFiniteDiversityNumber(snapshot.mean_island_diversity) !== null ||
        toFiniteDiversityNumber(snapshot.inter_island_diversity) !== null ||
        toFiniteDiversityNumber(snapshot.island_separation_ratio) !== null ||
        getIslandDiversityRows(snapshot).length > 0
    );
}

function buildIslandDiversitySnapshotMap(diversityHistory, islandDiversityHistory) {
    const snapshots = new Map();
    (Array.isArray(diversityHistory) ? diversityHistory : []).forEach((snapshot, index) => {
        if (!snapshotHasIslandDiversity(snapshot)) return;
        snapshots.set(getSnapshotGeneration(snapshot, index), snapshot);
    });
    normalizeIslandDiversityHistory(islandDiversityHistory).forEach((snapshot, index) => {
        if (!snapshotHasIslandDiversity(snapshot)) return;
        snapshots.set(getSnapshotGeneration(snapshot, index), {
            ...(snapshots.get(getSnapshotGeneration(snapshot, index)) || {}),
            ...snapshot
        });
    });
    return snapshots;
}

function getFitnessPlotMetricKeys() {
    return ['fitness', 'elo', 'diversity'];
}

function normalizeFitnessPlotMetric(metric) {
    return getFitnessPlotMetricKeys().includes(metric) ? metric : 'fitness';
}

function normalizeFitnessPlotScope(scope) {
    return scope === 'islands' ? 'islands' : 'population';
}

function getViewerChartTheme() {
    const styles = typeof getComputedStyle === 'function' && typeof document !== 'undefined' && document.documentElement
        ? getComputedStyle(document.documentElement) : null;
    const token = (name, fallback) => styles?.getPropertyValue(name).trim() || fallback;
    const theme = {
        violet: token('--chart-violet', '#7146d8'),
        coral: token('--chart-coral', '#c45d28'),
        green: token('--chart-green', '#27836a'),
        amber: token('--chart-amber', '#9b7019'),
        blue: token('--chart-blue', '#4378b8'),
        text: token('--text-primary', '#342151'),
        secondary: token('--text-secondary', '#695c78'),
        border: token('--border', '#ded4e8'),
        surface: token('--surface', '#fcf9f5')
    };
    if (typeof Chart !== 'undefined' && Chart.defaults) {
        Chart.defaults.font.family = "'Figtree', sans-serif";
        Chart.defaults.color = theme.secondary;
        Chart.defaults.borderColor = theme.border;
        const plugins = Chart.defaults.plugins;
        if (plugins?.title) plugins.title.color = theme.text;
        if (plugins?.legend?.labels) plugins.legend.labels.color = theme.secondary;
        if (plugins?.tooltip) {
            plugins.tooltip.backgroundColor = theme.surface;
            plugins.tooltip.titleColor = theme.text;
            plugins.tooltip.bodyColor = theme.secondary;
            plugins.tooltip.borderColor = theme.border;
        }
    }
    return theme;
}

function viewerChartColorWithAlpha(color, alpha) {
    const hex = /^#([a-f\d]{6})$/i.exec(color);
    if (hex) {
        const channels = [0, 2, 4].map(offset => parseInt(hex[1].slice(offset, offset + 2), 16));
        return `rgba(${channels.join(', ')}, ${alpha})`;
    }
    const rgb = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(color);
    return rgb ? `rgba(${rgb.slice(1).join(', ')}, ${alpha})` : color;
}

function getFitnessPlotMetricConfig(metric) {
    const chartTheme = getViewerChartTheme();
    const configs = {
        fitness: {
            label: 'Fitness',
            axisLabel: 'Fitness Score',
            averageLabel: 'Average Fitness',
            bestLabel: 'Best Fitness',
            lowLabel: 'Lowest Fitness',
            digits: 3,
            colors: {
                average: chartTheme.violet,
                best: chartTheme.green,
                low: chartTheme.coral
            }
        },
        elo: {
            label: 'ELO',
            axisLabel: 'ELO Rating',
            averageLabel: 'Average ELO',
            bestLabel: 'Highest ELO',
            lowLabel: 'Lowest ELO',
            digits: 0,
            colors: {
                average: chartTheme.violet,
                best: chartTheme.green,
                low: chartTheme.coral
            }
        },
        diversity: {
            label: 'Diversity',
            axisLabel: 'Diversity Score',
            averageLabel: 'Average Diversity',
            bestLabel: 'Highest Diversity',
            lowLabel: 'Lowest Diversity',
            digits: 3,
            colors: {
                average: chartTheme.violet,
                best: chartTheme.green,
                low: chartTheme.coral
            }
        }
    };
    return configs[normalizeFitnessPlotMetric(metric)];
}

function createEmptyFitnessMetricSeries() {
    return { average: [], best: [], low: [], spread: [], count: [] };
}

function createEmptyIslandDiversitySeries() {
    return {
        mean: [],
        inter: [],
        separation: [],
        islandSeries: {},
        latest: null
    };
}

function createEmptyFitnessPlotData() {
    return {
        labels: [],
        metrics: {
            fitness: createEmptyFitnessMetricSeries(),
            elo: createEmptyFitnessMetricSeries(),
            diversity: createEmptyFitnessMetricSeries()
        },
        islandDiversity: createEmptyIslandDiversitySeries(),
        latest: null
    };
}

function normalizeGenerationHistory(generationHistory) {
    if (Array.isArray(generationHistory)) return generationHistory;
    if (Array.isArray(generationHistory?.history)) return generationHistory.history;
    if (Array.isArray(generationHistory?.generations)) return generationHistory.generations;
    return [];
}

function getGenerationPopulation(generationData) {
    if (Array.isArray(generationData)) return generationData;
    if (Array.isArray(generationData?.ideas)) return generationData.ideas;
    if (Array.isArray(generationData?.population)) return generationData.population;
    return [];
}

function getFitnessPlotSelectionGeneration(idea) {
    return toFiniteDiversityInteger(idea?.selection_metrics?.evaluation_generation);
}

function getFitnessPlotGenerationMetricRecord(idea, generation) {
    const requestedGeneration = toFiniteDiversityInteger(generation);
    if (requestedGeneration === null) return null;

    const generationMetrics = idea?.generation_metrics;
    if (!generationMetrics) return null;

    let record = null;
    if (Array.isArray(generationMetrics)) {
        record = generationMetrics.find((row) => {
            const rowGeneration = firstFiniteDiversityValue(row?.evaluation_generation, row?.generation);
            return rowGeneration === requestedGeneration;
        }) || null;
    } else if (typeof generationMetrics === 'object') {
        record = generationMetrics[String(requestedGeneration)] || generationMetrics[requestedGeneration] || null;
    }

    if (!record || typeof record !== 'object') return null;
    const recordGeneration = firstFiniteDiversityValue(record.evaluation_generation, record.generation);
    if (recordGeneration !== null && recordGeneration !== requestedGeneration) return null;
    return record;
}

function getFitnessPlotMetricValue(idea, metric, generation = null) {
    const selection = idea?.selection_metrics || {};
    const ratings = idea?.ratings || {};
    const requestedGeneration = toFiniteDiversityInteger(generation);
    const component = metric === 'diversity' ? 'diversity' : metric === 'elo' ? 'quality_elo' : null;
    const evaluation = component ? idea?.evaluations?.[component] : null;
    if (evaluation && evaluation.status !== 'ok'
        && (requestedGeneration === null || toFiniteDiversityInteger(evaluation.generation) === requestedGeneration)) return null;
    const generationRecord = getFitnessPlotGenerationMetricRecord(idea, requestedGeneration);
    if (generationRecord) {
        const snapshotValue = firstFiniteDiversityValue(generationRecord[metric]);
        if (snapshotValue !== null) return snapshotValue;
    }

    const selectionGeneration = getFitnessPlotSelectionGeneration(idea);
    const hasGenerationStampedSelection = selectionGeneration !== null && requestedGeneration !== null;
    const selectionMatchesGeneration = !hasGenerationStampedSelection || selectionGeneration === requestedGeneration;

    if (!selectionMatchesGeneration) {
        return null;
    }

    const selectionOnlyAliases = {
        fitness: [selection.fitness],
        elo: [selection.elo],
        diversity: [selection.diversity]
    };
    const fallbackAliases = {
        fitness: [selection.fitness, idea?.fitness],
        elo: [selection.elo, idea?.elo, ratings.auto, ratings.manual],
        diversity: [selection.diversity, idea?.diversity_score, idea?.diversity]
    };
    const aliases = hasGenerationStampedSelection
        ? selectionOnlyAliases
        : fallbackAliases;
    const candidates = aliases[metric] || [];
    for (const candidate of candidates) {
        const numeric = toFiniteDiversityNumber(candidate);
        if (numeric !== null) return numeric;
    }
    return null;
}

function summarizeFitnessPlotValues(values, fallbackValue = null) {
    const finiteValues = (values || [])
        .map(value => toFiniteDiversityNumber(value))
        .filter(value => value !== null);
    if (finiteValues.length === 0) {
        const fallback = toFiniteDiversityNumber(fallbackValue);
        if (fallback !== null) {
            return {
                average: fallback,
                best: fallback,
                low: fallback,
                spread: 0,
                count: 1
            };
        }
        return { average: null, best: null, low: null, spread: null, count: 0 };
    }

    const total = finiteValues.reduce((sum, value) => sum + value, 0);
    const best = Math.max(...finiteValues);
    const low = Math.min(...finiteValues);
    return {
        average: total / finiteValues.length,
        best,
        low,
        spread: best - low,
        count: finiteValues.length
    };
}

function buildFitnessPlotSummaries(population, generation) {
    const summaries = {};
    getFitnessPlotMetricKeys().forEach((metric) => {
        const values = population.map(idea => getFitnessPlotMetricValue(idea, metric, generation));
        summaries[metric] = { ...summarizeFitnessPlotValues(values), total: population.length };
    });
    return summaries;
}

function appendFitnessPlotSummaries(processedData, generation, summaries, options = {}) {
    const hasMetricData = getFitnessPlotMetricKeys().some(metric => summaries[metric]?.count > 0);
    if (!hasMetricData && !options.includeEmpty) return false;

    processedData.labels.push(String(generation));
    getFitnessPlotMetricKeys().forEach((metric) => {
        const summary = summaries[metric] || summarizeFitnessPlotValues([]);
        processedData.metrics[metric].average.push(summary.average);
        processedData.metrics[metric].best.push(summary.best);
        processedData.metrics[metric].low.push(summary.low);
        processedData.metrics[metric].spread.push(summary.spread);
        processedData.metrics[metric].count.push(summary.count);
    });

    processedData.latest = {
        generation,
        ...summaries
    };
    return true;
}

function ensureFitnessPlotGeneration(processedData, generation) {
    const label = String(generation);
    if (processedData.labels.includes(label)) return;
    const emptySummaries = buildFitnessPlotSummaries([], generation);
    appendFitnessPlotSummaries(processedData, generation, emptySummaries, { includeEmpty: true });
}

function buildIslandDiversitySeries(labels, diversityHistory, islandDiversityHistory) {
    const islandSnapshots = buildIslandDiversitySnapshotMap(diversityHistory, islandDiversityHistory);
    const islandIds = new Set();
    islandSnapshots.forEach((snapshot) => {
        getIslandDiversityRows(snapshot).forEach((row) => {
            const islandId = toFiniteDiversityInteger(row?.island_id);
            if (islandId !== null) islandIds.add(islandId);
        });
    });

    const sortedIslandIds = Array.from(islandIds).sort((a, b) => a - b);
    const islandSeries = {};
    sortedIslandIds.forEach((islandId) => {
        islandSeries[String(islandId)] = labels.map(() => null);
    });

    const series = {
        mean: [],
        inter: [],
        separation: [],
        islandSeries,
        latest: null
    };

    labels.forEach((label, index) => {
        const generation = toFiniteDiversityInteger(label);
        const snapshot = generation === null ? null : islandSnapshots.get(generation);
        const mean = firstFiniteDiversityValue(
            snapshot?.mean_island_diversity,
            snapshot?.analytics?.mean_island_diversity
        );
        const inter = firstFiniteDiversityValue(
            snapshot?.inter_island_diversity,
            snapshot?.analytics?.inter_island_diversity
        );
        const separation = firstFiniteDiversityValue(
            snapshot?.island_separation_ratio,
            snapshot?.analytics?.island_separation_ratio
        );

        series.mean.push(mean);
        series.inter.push(inter);
        series.separation.push(separation);

        const islandRows = getIslandDiversityRows(snapshot);
        islandRows.forEach((row) => {
            const islandId = toFiniteDiversityInteger(row?.island_id);
            if (islandId === null) return;
            const key = String(islandId);
            if (!series.islandSeries[key]) {
                series.islandSeries[key] = labels.map(() => null);
            }
            series.islandSeries[key][index] = firstFiniteDiversityValue(row?.diversity_score);
        });

        if (snapshot) {
            series.latest = {
                generation,
                mean,
                inter,
                separation,
                islandCount: islandRows.length
            };
        }
    });

    return series;
}

function processDiversityData(diversityHistory, generationHistory = null, islandDiversityHistory = []) {
    const processedData = createEmptyFitnessPlotData();
    const historicalRows = Array.isArray(diversityHistory) ? diversityHistory : [];
    const historicalSnapshot = [...historicalRows].reverse().find(snapshot => snapshot?.history_sampling || snapshot?.metadata?.history_sampling);
    processedData.historySampling = historicalSnapshot?.history_sampling || historicalSnapshot?.metadata?.history_sampling || null;
    const generationRows = normalizeGenerationHistory(generationHistory);

    if (generationRows.length > 0) {
        generationRows.forEach((generationData, index) => {
            const generation = toFiniteDiversityInteger(generationData?.generation) ?? index;
            const population = getGenerationPopulation(generationData);
            const summaries = buildFitnessPlotSummaries(population, generation);
            appendFitnessPlotSummaries(processedData, generation, summaries, { includeEmpty: population.length > 0 });
        });
    } else {
        (Array.isArray(diversityHistory) ? diversityHistory : []).forEach((snapshot, index) => {
            if (!snapshot || snapshot.enabled === false || snapshot.error) return;
            const generation = getSnapshotGeneration(snapshot, index);
            const summaries = buildFitnessPlotSummaries([], generation);
            appendFitnessPlotSummaries(processedData, generation, summaries);
        });
    }

    const islandSnapshots = buildIslandDiversitySnapshotMap(diversityHistory, islandDiversityHistory);
    islandSnapshots.forEach((_snapshot, generation) => {
        ensureFitnessPlotGeneration(processedData, generation);
    });
    processedData.islandDiversity = buildIslandDiversitySeries(
        processedData.labels,
        diversityHistory,
        islandDiversityHistory
    );
    return processedData;
}

function getDiversityDatasetStyle(color, options = {}) {
    return {
        borderColor: color,
        backgroundColor: viewerChartColorWithAlpha(color, 0.10),
        borderWidth: options.borderWidth ?? 2,
        fill: options.fill ?? false,
        tension: 0.3,
        pointBackgroundColor: color,
        pointBorderColor: getViewerChartTheme().surface,
        pointBorderWidth: 1.5,
        pointRadius: options.pointRadius ?? 4,
        pointHoverRadius: 6,
        borderDash: options.borderDash || [],
        spanGaps: true
    };
}

function getIslandDiversityColor(index) {
    const theme = getViewerChartTheme();
    const colors = [theme.violet, theme.green, theme.coral, theme.amber, theme.blue];
    return colors[index % colors.length];
}

function getIslandDiversityDatasets(processedData) {
    const islandData = processedData?.islandDiversity || createEmptyIslandDiversitySeries();
    const datasets = [
        {
            label: 'Mean Island Diversity',
            data: islandData.mean || [],
            ...getDiversityDatasetStyle(getViewerChartTheme().violet, { fill: true, borderWidth: 3 })
        },
        {
            label: 'Inter-Island Diversity',
            data: islandData.inter || [],
            ...getDiversityDatasetStyle(getViewerChartTheme().green, { borderWidth: 2.5, borderDash: [7, 4] })
        }
    ];

    Object.keys(islandData.islandSeries || {})
        .sort((a, b) => Number(a) - Number(b))
        .forEach((islandId, index) => {
            datasets.push({
                label: `Island ${Number(islandId) + 1}`,
                data: islandData.islandSeries[islandId] || [],
                ...getDiversityDatasetStyle(getIslandDiversityColor(index), {
                    borderWidth: 2,
                    pointRadius: 3
                })
            });
        });

    return datasets;
}

function getDiversityChartDatasets(processedData, mode = fitnessPlotMetric, scope = fitnessPlotScope) {
    const metric = normalizeFitnessPlotMetric(mode);
    if (metric === 'diversity' && normalizeFitnessPlotScope(scope) === 'islands') {
        return getIslandDiversityDatasets(processedData);
    }
    const config = getFitnessPlotMetricConfig(metric);
    const series = processedData?.metrics?.[metric] || createEmptyFitnessMetricSeries();
    return [
        {
            label: config.averageLabel,
            data: series.average || [],
            ...getDiversityDatasetStyle(config.colors.average, { fill: true, borderWidth: 3 })
        },
        {
            label: config.bestLabel,
            data: series.best || [],
            ...getDiversityDatasetStyle(config.colors.best, { borderWidth: 2.5 })
        },
        {
            label: config.lowLabel,
            data: series.low || [],
            ...getDiversityDatasetStyle(config.colors.low, { borderDash: [6, 4] })
        }
    ];
}

function formatDiversityMetric(value, digits = 3) {
    const numeric = toFiniteDiversityNumber(value);
    return numeric !== null ? numeric.toFixed(digits) : '—';
}

function formatFitnessPlotMetric(value, metric = fitnessPlotMetric, digits = null) {
    const config = getFitnessPlotMetricConfig(metric);
    const resolvedDigits = Number.isInteger(digits) ? digits : config.digits;
    return formatDiversityMetric(value, resolvedDigits);
}

function renderDiversitySummary(latest, processedData = diversityData) {
    const samplingNotice = window.IdeaWorkbench?.renderHistorySamplingNote(
        document.getElementById('diversitySamplingNote'), processedData?.historySampling,
    ) || '';
    window.IdeaWorkbench?.renderChartTable(
        document.getElementById('fitnessChartTable'),
        processedData?.labels || [],
        getDiversityChartDatasets(processedData, fitnessPlotMetric, fitnessPlotScope),
        `${getFitnessPlotMetricConfig(fitnessPlotMetric).label} by generation${samplingNotice ? `. ${samplingNotice}` : ''}`,
    );
    const summary = document.getElementById('diversity-summary');
    if (!summary) return;
    if (fitnessPlotMetric === 'diversity' && fitnessPlotScope === 'islands') {
        renderIslandDiversitySummary(processedData?.islandDiversity?.latest || null);
        return;
    }
    if (!latest) {
        summary.innerHTML = '';
        return;
    }

    const config = getFitnessPlotMetricConfig(fitnessPlotMetric);
    const metricSummary = latest[fitnessPlotMetric] || {};
    const count = Number(metricSummary.count) || 0;
    const generationHint = latest.generation === 0 ? 'Initial generation' : `Generation ${latest.generation}`;
    const items = [
        {
            label: `Average ${config.label}`,
            value: formatFitnessPlotMetric(metricSummary.average, fitnessPlotMetric),
            hint: `${count}/${metricSummary.total ?? count} measured · ${generationHint}`
        },
        {
            label: config.bestLabel,
            value: formatFitnessPlotMetric(metricSummary.best, fitnessPlotMetric),
            hint: 'Top population value'
        },
        {
            label: config.lowLabel,
            value: formatFitnessPlotMetric(metricSummary.low, fitnessPlotMetric),
            hint: 'Lowest population value'
        },
        {
            label: 'Spread',
            value: formatFitnessPlotMetric(metricSummary.spread, fitnessPlotMetric),
            hint: 'Best minus lowest'
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

function renderIslandDiversitySummary(latestIsland) {
    const summary = document.getElementById('diversity-summary');
    if (!summary) return;
    if (!latestIsland) {
        summary.innerHTML = '';
        return;
    }

    const generationHint = latestIsland.generation === 0
        ? 'Initial generation'
        : `Generation ${latestIsland.generation}`;
    const items = [
        {
            label: 'Mean Island Diversity',
            value: formatFitnessPlotMetric(latestIsland.mean, 'diversity'),
            hint: generationHint
        },
        {
            label: 'Inter-Island Diversity',
            value: formatFitnessPlotMetric(latestIsland.inter, 'diversity'),
            hint: 'Distance between island centroids'
        },
        {
            label: 'Separation Ratio',
            value: formatDiversityMetric(latestIsland.separation, 2),
            hint: 'Inter-island divided by mean island diversity'
        },
        {
            label: 'Islands',
            value: Number.isFinite(Number(latestIsland.islandCount)) ? String(latestIsland.islandCount) : '-',
            hint: 'Tracked island populations'
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

function hasFitnessPlotMetricData(processedData, metric) {
    const series = processedData?.metrics?.[metric];
    return Boolean(series?.average?.some(value => toFiniteDiversityNumber(value) !== null));
}

function hasIslandDiversityData(processedData) {
    const islandData = processedData?.islandDiversity;
    if (!islandData) return false;
    if ((islandData.mean || []).some(value => toFiniteDiversityNumber(value) !== null)) return true;
    if ((islandData.inter || []).some(value => toFiniteDiversityNumber(value) !== null)) return true;
    return Object.values(islandData.islandSeries || {}).some(series =>
        (series || []).some(value => toFiniteDiversityNumber(value) !== null)
    );
}

function updateFitnessScopeAvailability(processedData) {
    const buttons = document.querySelectorAll('[data-fitness-scope]');
    if (!buttons.length) return;

    const islandAvailable = fitnessPlotMetric === 'diversity' && hasIslandDiversityData(processedData);
    if (fitnessPlotScope === 'islands' && !islandAvailable) {
        fitnessPlotScope = 'population';
    }

    buttons.forEach((button) => {
        const scope = normalizeFitnessPlotScope(button.dataset.fitnessScope);
        const enabled = scope === 'population' || islandAvailable;
        button.disabled = !enabled;
        button.title = enabled
            ? (scope === 'population' ? 'Show population summary' : 'Show per-island diversity')
            : 'Island diversity values are not available for this metric';
        button.classList.toggle('active', scope === fitnessPlotScope);
    });
}

function updateFitnessMetricAvailability(processedData) {
    const buttons = document.querySelectorAll('[data-fitness-metric]');
    if (!buttons.length) {
        updateFitnessScopeAvailability(processedData);
        return;
    }

    const availableMetrics = getFitnessPlotMetricKeys()
        .filter(metric => hasFitnessPlotMetricData(processedData, metric) || (
            metric === 'diversity' && hasIslandDiversityData(processedData)
        ));
    if (availableMetrics.length > 0 && !availableMetrics.includes(fitnessPlotMetric)) {
        fitnessPlotMetric = availableMetrics[0];
    }

    buttons.forEach((button) => {
        const metric = normalizeFitnessPlotMetric(button.dataset.fitnessMetric);
        const hasData = availableMetrics.includes(metric);
        button.disabled = availableMetrics.length > 0 && !hasData;
        button.title = hasData
            ? `Show ${getFitnessPlotMetricConfig(metric).label}`
            : `No ${getFitnessPlotMetricConfig(metric).label} values available yet`;
        button.classList.toggle('active', metric === fitnessPlotMetric);
    });
    updateFitnessScopeAvailability(processedData);
}

function buildDiversitySignature(processedData) {
    if (!processedData || !processedData.labels) {
        return '';
    }

    return JSON.stringify({
        labels: processedData.labels,
        metrics: processedData.metrics,
        islandDiversity: processedData.islandDiversity,
        historySampling: processedData.historySampling,
        metric: fitnessPlotMetric,
        scope: fitnessPlotScope
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
 * @param {Array} generationHistory - Population history with per-idea metrics
 * @param {Array} islandDiversityHistory - Per-island diversity snapshots
 */
function updateDiversityChart(diversityHistory, generationHistory = null, islandDiversityHistory = []) {
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
            if (!ctx || diversityChart.canvas !== canvas) {
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
        const processedData = processDiversityData(diversityHistory, generationHistory, islandDiversityHistory);
        updateFitnessMetricAvailability(processedData);
        const nextSignature = buildDiversitySignature(processedData);
        if (processedData.labels.length === 0) {
            diversityChart.data.labels = [];
            diversityChart.data.datasets = getDiversityChartDatasets(processedData, fitnessPlotMetric, fitnessPlotScope);
            diversityChart.options.scales = getDiversityChartScales();
            diversityChart.update('none');
            diversityData = processedData;
            lastDiversityRenderSignature = nextSignature;
            renderDiversitySummary(null, processedData);
            const plotSection = document.getElementById('diversity-plot-section');
            if (plotSection) {
                plotSection.style.display = 'none';
            }
            return false;
        }
        renderDiversitySummary(processedData.latest, processedData);
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
        diversityChart.data.datasets = getDiversityChartDatasets(processedData, fitnessPlotMetric, fitnessPlotScope);
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
                <span>Calculating population metrics...</span>
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
        diversityChart.data.datasets = getDiversityChartDatasets(null, fitnessPlotMetric, fitnessPlotScope);
        diversityChart.update();
    }

    // Reset global diversity data
    diversityData = createEmptyFitnessPlotData();
    fitnessPlotMetric = 'fitness';
    fitnessPlotScope = 'population';
    updateFitnessMetricAvailability(diversityData);
    renderDiversitySummary(null, diversityData);
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
    if (progressData && (progressData.diversity_history || progressData.history)) {
        const renderedHistory = Array.isArray(generations) && generations.length > 0 ? generations : null;
        const generationHistory =
            progressData.history ||
            currentEvolutionData?.history ||
            renderedHistory ||
            null;
        const islandDiversityHistory =
            progressData.island_diversity_history ||
            currentEvolutionData?.island_diversity_history ||
            [];
        updateDiversityChart(
            progressData.diversity_history || currentEvolutionData?.diversity_history || [],
            generationHistory,
            islandDiversityHistory
        );
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

    document.querySelectorAll('[data-fitness-metric]').forEach((button) => {
        button.addEventListener('click', () => setFitnessPlotMetric(button.dataset.fitnessMetric));
    });
    document.querySelectorAll('[data-fitness-scope]').forEach((button) => {
        button.addEventListener('click', () => setFitnessPlotScope(button.dataset.fitnessScope));
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

const FALLBACK_THINKING_UI_CONFIGS = {
    'gemini-2.5-flash-lite': {
        supportsThinking: true,
        supportsBudget: true,
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
        supportsBudget: true,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'high',
        min: 0,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is High dynamic thinking. Use Off for speed/cost or set a custom token budget when needed.'
    },
    'gemini-2.5-pro': {
        supportsThinking: true,
        supportsBudget: true,
        allowOff: false,
        allowMedium: true,
        defaultMode: 'high',
        min: 128,
        max: 32768,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is High dynamic thinking. Use Low/Medium or set a custom budget to tune latency and depth.'
    },
    'gemini-3-flash-preview': {
        supportsThinking: true,
        supportsBudget: false,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'low',
        offValue: 'minimal',
        offLabel: 'Minimal',
        offDescription: ' - Lowest Gemini 3 thinking level',
        min: 0,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Low. Use Medium/High for deeper reasoning or Minimal to reduce latency and cost.'
    },
    'gemini-3.1-flash-lite-preview': {
        supportsThinking: true,
        supportsBudget: false,
        allowOff: true,
        allowMedium: true,
        defaultMode: 'low',
        offValue: 'minimal',
        offLabel: 'Minimal',
        offDescription: ' - Lowest Gemini 3 thinking level',
        min: 0,
        max: 24576,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Low. Use Medium/High for deeper reasoning or Minimal to reduce latency and cost.'
    },
    'gemini-3.1-pro-preview': {
        supportsThinking: true,
        supportsBudget: false,
        allowOff: false,
        allowMedium: true,
        defaultMode: 'low',
        min: 128,
        max: 32768,
        step: 128,
        defaultBudget: 1024,
        help: 'Default is Low. Use Medium/High for deeper reasoning.'
    }
};

function buildThinkingHelp(config, lowestLevel) {
    if (config.supports_thinking_level && !config.supports_thinking_budget) {
        const allowedLevels = Array.isArray(config.allowed_levels) ? config.allowed_levels : [];
        if (allowedLevels.includes('max')) {
            return 'Default is Medium. Use None/Low for latency or High/Extra High/Max for quality-first workloads.';
        }
        if (lowestLevel === 'minimal') {
            return 'Default is Low. Use Medium/High for deeper reasoning or Minimal to reduce latency and cost.';
        }
        return 'Default is Low. Use Medium/High for deeper reasoning.';
    }

    if (config.default_level === 'high') {
        return 'Default is High dynamic thinking. Use lower presets or a custom token budget to tune latency and depth.';
    }
    if (config.allow_off) {
        return 'Default is Off for speed/cost. Use Low/Medium/High or a custom token budget when needed.';
    }
    return 'Default is Low. Use Medium/High or a custom token budget for deeper reasoning.';
}

function buildThinkingUiConfig(config) {
    if (!config || !config.supports_thinking) {
        return { supportsThinking: false };
    }

    const allowedLevels = Array.isArray(config.allowed_levels) ? config.allowed_levels : [];
    const lowestLevel = allowedLevels.includes('minimal')
        ? 'minimal'
        : (allowedLevels.includes('none')
            ? 'none'
            : (allowedLevels.includes('off') || config.allow_off ? 'off' : null));
    const minBudget = Number(config.min_budget ?? 0);
    const maxBudget = Number(config.max_budget ?? 0);
    let defaultBudget = Number(config.default_budget);
    if (!Number.isFinite(defaultBudget) || defaultBudget < minBudget) {
        defaultBudget = Math.min(Math.max(1024, minBudget), maxBudget || 1024);
    }

    return {
        supportsThinking: true,
        supportsBudget: config.supports_thinking_budget === true,
        allowOff: lowestLevel !== null,
        allowMedium: allowedLevels.length === 0 || allowedLevels.includes('medium'),
        allowXHigh: allowedLevels.includes('xhigh'),
        allowMax: allowedLevels.includes('max'),
        defaultMode: config.default_level || (config.default_budget !== null && config.default_budget !== undefined ? 'custom' : 'low'),
        offValue: lowestLevel || 'off',
        offLabel: lowestLevel === 'minimal' ? 'Minimal' : (lowestLevel === 'none' ? 'None' : 'Off'),
        offDescription: lowestLevel === 'minimal'
            ? ' - Lowest Gemini 3 thinking level'
            : ' - Prefer speed and lower cost',
        min: Number.isFinite(minBudget) ? minBudget : 0,
        max: Number.isFinite(maxBudget) ? maxBudget : 0,
        step: 128,
        defaultBudget,
        help: buildThinkingHelp(config, lowestLevel)
    };
}

function buildThinkingUiConfigs(serverConfigs) {
    if (!serverConfigs || typeof serverConfigs !== 'object') {
        return null;
    }

    return Object.fromEntries(
        Object.entries(serverConfigs).map(([modelName, config]) => [
            modelName,
            buildThinkingUiConfig(config)
        ])
    );
}

const THINKING_UI_CONFIGS = buildThinkingUiConfigs(window.IDEA_THINKING_MODEL_CONFIG) || FALLBACK_THINKING_UI_CONFIGS;

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
    window.IdeaWorkbench?.configureModelPicker({
        modelId: 'modelType', providerId: 'providerSelect',
        models: Array.from(modelSelect.options).map(option => ({ id: option.value, name: option.textContent, provider: option.dataset.provider })),
        defaultModel: window.IDEA_DEFAULT_MODEL,
    });

    modelSelect.addEventListener('change', function () {
        updateThinkingBudgetVisibility();
        if (typeof window.checkApiKeyStatus === 'function') {
            void window.checkApiKeyStatus();
        }
    });
    updateThinkingBudgetVisibility();
}

/**
 * Update thinking control visibility based on selected model
 */
function supportsSamplingControls(model, thinkingLevel) {
    const capabilities = window.IDEA_THINKING_MODEL_CONFIG?.[String(model || '').replace(/^models\//, '')];
    const support = capabilities?.supports_sampling_parameters;
    return support !== false && (support !== 'reasoning_none_only' || thinkingLevel === 'none');
}

function updateSamplingControls() {
    const model = document.getElementById('modelType');
    if (!model) return;
    const enabled = supportsSamplingControls(model.value, getThinkingSettings().thinkingLevel);
    for (const id of ['creativeTemp', 'topP']) {
        const control = document.getElementById(id);
        if (control) control.disabled = !enabled;
        const value = document.getElementById(`${id}Value`);
        if (value && control) value.textContent = enabled ? control.value : 'Provider controlled';
    }
    const notice = document.getElementById('samplingControlsNotice');
    if (notice) {
        notice.hidden = enabled;
        notice.textContent = enabled ? '' : 'This model controls sampling. Temperature and top P do not apply with these settings.';
    }
    const geneNotice = document.getElementById('geneAnalysisModelNotice');
    if (geneNotice) {
        geneNotice.textContent = 'Gene analysis uses the selected model and thinking settings, within the run budget.';
    }
}

function updateThinkingBudgetVisibility() {
    const modelSelect = document.getElementById('modelType');
    const thinkingBudgetContainer = document.getElementById('thinkingBudgetContainer');
    if (!modelSelect || !thinkingBudgetContainer) return;

    const selectedModel = modelSelect.value;
    const selectedProvider = modelSelect.selectedOptions?.[0]?.dataset?.provider;
    const providerSelect = document.getElementById('apiKeyProviderSelect');
    if (providerSelect && ['openai', 'google_generative_ai'].includes(selectedProvider)) {
        providerSelect.value = selectedProvider;
        renderApiKeyProviderStatus(executionSettingsCache);
    }
    const config = getThinkingModelConfig(selectedModel);
    if (!config || !config.supportsThinking) {
        thinkingBudgetContainer.style.display = 'none';
        updateSamplingControls();
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
    const xhighOption = document.getElementById('thinkingXHighOption');
    const maxOption = document.getElementById('thinkingMaxOption');
    const offLabel = document.getElementById('thinkingOffLabel');
    const offDescription = document.getElementById('thinkingOffDescription');

    const offRadio = document.getElementById('thinkingOff');
    const lowRadio = document.getElementById('thinkingLow');
    const mediumRadio = document.getElementById('thinkingMedium');
    const highRadio = document.getElementById('thinkingHigh');
    const xhighRadio = document.getElementById('thinkingXHigh');
    const maxRadio = document.getElementById('thinkingMax');
    const customRadio = document.getElementById('thinkingCustomBudget');
    const customBudgetOption = document.getElementById('thinkingCustomBudgetOption') || customRadio?.closest('.form-check');

    if (!config || !thinkingBudgetSlider || !thinkingBudgetHelp || !customRadio) return;

    // Toggle option availability by model capability.
    const supportsBudget = config.supportsBudget !== false;
    if (offOption) offOption.style.display = config.allowOff ? 'block' : 'none';
    if (mediumOption) mediumOption.style.display = config.allowMedium ? 'block' : 'none';
    if (xhighOption) xhighOption.style.display = config.allowXHigh ? 'block' : 'none';
    if (maxOption) maxOption.style.display = config.allowMax ? 'block' : 'none';
    if (customBudgetOption) customBudgetOption.style.display = supportsBudget ? 'block' : 'none';
    if (offRadio) offRadio.value = config.offValue || 'off';
    if (offLabel) offLabel.textContent = config.offLabel || 'Off';
    if (offDescription) offDescription.textContent = config.offDescription || ' - Prefer speed and lower cost';

    if (offRadio) offRadio.checked = false;
    if (lowRadio) lowRadio.checked = false;
    if (mediumRadio) mediumRadio.checked = false;
    if (highRadio) highRadio.checked = false;
    if (xhighRadio) xhighRadio.checked = false;
    if (maxRadio) maxRadio.checked = false;
    customRadio.checked = false;

    let effectiveDefault = config.defaultMode;
    if (effectiveDefault === 'off' && !config.allowOff) {
        effectiveDefault = 'low';
    }
    if (effectiveDefault === 'medium' && !config.allowMedium) {
        effectiveDefault = 'low';
    }
    if (effectiveDefault === 'custom' && !supportsBudget) {
        effectiveDefault = 'high';
    }

    if (offRadio && effectiveDefault === offRadio.value) {
        offRadio.checked = true;
    } else if (effectiveDefault === 'low' && lowRadio) {
        lowRadio.checked = true;
    } else if (effectiveDefault === 'medium' && mediumRadio) {
        mediumRadio.checked = true;
    } else if (effectiveDefault === 'high' && highRadio) {
        highRadio.checked = true;
    } else if (effectiveDefault === 'xhigh' && xhighRadio) {
        xhighRadio.checked = true;
    } else if (effectiveDefault === 'max' && maxRadio) {
        maxRadio.checked = true;
    } else if (supportsBudget) {
        customRadio.checked = true;
    } else if (highRadio) {
        highRadio.checked = true;
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
    updateSamplingControls();
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
    const xhighRadio = document.getElementById('thinkingXHigh');
    const maxRadio = document.getElementById('thinkingMax');
    const customRadio = document.getElementById('thinkingCustomBudget');
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');

    if (config.supportsBudget !== false && customRadio && customRadio.checked && thinkingBudgetSlider) {
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
    if (xhighRadio && xhighRadio.checked) {
        return { thinkingLevel: 'xhigh', thinkingBudget: null };
    }
    if (maxRadio && maxRadio.checked) {
        return { thinkingLevel: 'max', thinkingBudget: null };
    }

    if (config.supportsBudget !== false && config.defaultMode === 'custom' && thinkingBudgetSlider) {
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
    window.IdeaWorkbench?.selectSidebarView(viewName);
    const nameSection = document.getElementById('evolutionNameSection');
    if (nameSection && viewName === 'main') nameSection.style.display = 'block';
    if (viewName === 'history') loadHistory();
    if (viewName === 'settings') return checkSettingsStatus();
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
let settingsStatusVersion = 0;
let settingsSaveInFlight = false;
async function checkSettingsStatus() {
    const version = ++settingsStatusVersion;
    const panel = document.getElementById('view-settings');
    const save = document.getElementById('saveSettingsBtn');
    if (save) save.disabled = true;
    panel?.setAttribute('aria-busy', 'true');
    document.getElementById('apiKeyInputContainer')?.style.setProperty('display', 'none');
    document.getElementById('apiKeyConfiguredContainer')?.style.setProperty('display', 'none');
    showStatusInSettings('Checking provider key…', 'info');
    try {
        const data = await loadExecutionSettings(true);
        if (version !== settingsStatusVersion) return false;
        const status = document.getElementById('settingsStatus');
        if (status) { status.hidden = true; status.style.display = 'none'; }
        renderApiKeyProviderStatus(data);
        return data;
    } catch (error) {
        if (version !== settingsStatusVersion) return false;
        console.error('Error checking settings:', error);
        showStatusInSettings('Could not refresh key status. Check your connection and reopen Settings to retry.', 'danger');
        return false;
    } finally {
        if (version === settingsStatusVersion) {
            panel?.removeAttribute('aria-busy');
            if (save) save.disabled = settingsSaveInFlight;
        }
    }
}

function selectedApiKeyProvider() {
    return document.getElementById('apiKeyProviderSelect')?.value || 'google_generative_ai';
}

function apiKeyProviderName(provider) {
    return provider === 'openai' ? 'OpenAI' : 'Gemini';
}

function renderApiKeyProviderStatus(data) {
    data = data || window.IdeaWorkbench?.providerSettingsSnapshot();
    if (data) window.IdeaWorkbench?.setProviderSettings(data);
    const provider = selectedApiKeyProvider();
    const providerName = apiKeyProviderName(provider);
    const status = data?.api_keys?.[provider] || { configured: false, masked_key: null };
    const inputContainer = document.getElementById('apiKeyInputContainer');
    const configuredContainer = document.getElementById('apiKeyConfiguredContainer');
    const maskedDisplay = document.getElementById('maskedKeyDisplay');
    const configuredProviderName = document.getElementById('configuredProviderName');
    const providerLabel = document.getElementById('apiKeyProviderLabel');
    const keyInput = document.getElementById('apiKeyInput');

    if (providerLabel) providerLabel.textContent = `${providerName} API Key`;
    if (configuredProviderName) configuredProviderName.textContent = providerName;
    if (keyInput) keyInput.placeholder = `Enter ${providerName} API key`;
    if (inputContainer) inputContainer.style.display = status.configured ? 'none' : 'block';
    if (configuredContainer) configuredContainer.style.display = status.configured ? 'block' : 'none';
    if (maskedDisplay) maskedDisplay.textContent = status.masked_key || '...';
    const source = status.source || data?.key_sources?.[provider];
    const sourceNote = document.getElementById('apiKeySourceNote');
    if (sourceNote) sourceNote.textContent = source === 'environment'
        ? 'Using this machine’s local environment key. Replace it only if you want to save a different key.'
        : 'Saved securely for this provider. Switching models reuses this key.';
    const remove = document.getElementById('deleteApiKeyBtn');
    if (remove) remove.hidden = source === 'environment';
    const cancel = document.getElementById('cancelApiKeyEdit');
    if (cancel) cancel.hidden = true;
    if (status.error_code === 'stored_key_unreadable') {
        showStatusInSettings(status.message || 'This saved key cannot be opened. Enter a replacement key to use this provider.', 'warning');
    }
}

/**
 * Save API Key
 */
async function saveSettings() {
    if (settingsSaveInFlight) return;
    const keyInput = document.getElementById('apiKeyInput');
    const key = keyInput ? keyInput.value.trim() : '';

    if (!key) {
        showStatusInSettings('Please enter an API key', 'warning');
        return;
    }

    settingsSaveInFlight = true;
    const provider = selectedApiKeyProvider();
    const btn = document.getElementById('saveSettingsBtn');
    const originalText = btn.innerHTML;
    const providerSelect = document.getElementById('apiKeyProviderSelect');
    if (providerSelect) providerSelect.disabled = true;
    keyInput.disabled = true;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Saving...';

    try {
        const response = await fetch('/api/settings/api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key, provider })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Clear input
            keyInput.value = '';
            // Refresh status to switch view
            const refreshed = await checkSettingsStatus();
            const notice = window.IdeaWorkbench.providerKeySaveNotice(provider, refreshed);
            showStatusInSettings(notice.message, notice.type);
            if (typeof window.checkApiKeyStatus === 'function') {
                await window.checkApiKeyStatus();
            }
        } else {
            showStatusInSettings(apiErrorMessage(data, 'The key could not be saved. Try again.'), 'danger');
        }
    } catch (e) {
        showStatusInSettings('Error: ' + e.message, 'danger');
    } finally {
        settingsSaveInFlight = false;
        btn.disabled = false;
        if (providerSelect) providerSelect.disabled = false;
        keyInput.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * Delete API Key
 */
async function deleteApiKey(e) {
    if (e) e.preventDefault();

    const providerName = apiKeyProviderName(selectedApiKeyProvider());
    const deleteProviderName = document.getElementById('deleteKeyProviderName');
    if (deleteProviderName) deleteProviderName.textContent = providerName;

    // Show the Bootstrap modal
    const deleteModal = new bootstrap.Modal(document.getElementById('deleteKeyModal'));
    deleteModal.show();
}

async function confirmDeleteApiKey() {
    const provider = selectedApiKeyProvider();
    const providerName = apiKeyProviderName(provider);
    const btn = document.getElementById('confirmDeleteKeyBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Deleting...';

    try {
        const response = await fetch(`/api/settings/api-key?provider=${encodeURIComponent(provider)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Hide modal
            const deleteModalEl = document.getElementById('deleteKeyModal');
            const modal = bootstrap.Modal.getInstance(deleteModalEl);
            if (modal) modal.hide();

            showStatusInSettings(`${providerName} API key deleted.`, 'success');
            // Refresh status to switch view
            await checkSettingsStatus();
            if (typeof window.checkApiKeyStatus === 'function') {
                await window.checkApiKeyStatus();
            }
        } else {
            const msg = data.message || data.detail || 'Unknown error';
            showStatusInSettings('Error: ' + msg, 'danger');
        }
    } catch (e) {
        showStatusInSettings('Error: ' + e.message, 'danger');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// Note: handleTemplateGeneration is defined in viewer.html with proper DOM elements

function showStatusInSettings(msg, type) {
    const el = document.getElementById('settingsStatus');
    if (el) {
        el.className = `alert alert-${type}`;
        el.textContent = msg;
        el.style.display = 'block';
        el.hidden = false;
        el.setAttribute('role', type === 'danger' ? 'alert' : 'status');
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
