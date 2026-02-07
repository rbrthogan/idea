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
const LOCAL_STORAGE_PERSIST_DELAY_MS = 1200;

// Track previous status message to detect changes for activity log
let previousStatusMessage = '';
let previousProgress = 0;

/**
 * Load available templates and populate the idea type dropdown
 */
let allTemplates = []; // Store templates globally
const LAST_TEMPLATE_STORAGE_KEY = 'lastSelectedTemplateId';

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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

        // Get thinking budget value (only for Gemini 2.5 models)
        const thinkingBudget = getThinkingBudgetValue();

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
            thinkingBudget,
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

        // First check if there's actually an evolution running on the server
        try {
            const progressResponse = await fetch('/api/progress');
            if (progressResponse.ok) {
                const progressData = await progressResponse.json();
                console.log("Evolution status from server:", progressData);

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
                    // Old format - just the history array
                    generationsData = evolutionState;
                    diversityData = [];
                    tokenCounts = null;
                    tournamentData = [];
                } else if (evolutionState && evolutionState.history) {
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
                    return true;
                }
            } catch (e) {
                console.error("Error parsing localStorage evolution data:", e);
            }
        }

        console.log("No current evolution data found in localStorage");
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
    persistStateTimeout = setTimeout(() => {
        localStorage.setItem('currentEvolutionData', JSON.stringify(stateData));
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

// Update the renderGenerations function to use the improved preview
function renderGenerations(gens) {
    generations = gens; // Store the generations globally
    console.log("Received generations:", generations); // Debug log

    if (!generations || generations.length === 0) {
        console.warn("Received empty generations data");
        return;
    }
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

            headerDiv.appendChild(header);
            headerDiv.appendChild(toggleButton);
            genDiv.appendChild(headerDiv);

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

        // Get the scroll container for this generation
        const scrollContainer = document.getElementById(`scroll-container-${index}`);

        // Process each idea in this generation
        generation.forEach((idea, ideaIndex) => {
            // Check if this idea card already exists
            const existingCard = document.getElementById(`idea-${index}-${ideaIndex}`);
            if (existingCard) {
                // An Oracle update might have replaced this idea.
                // We check if the new data for this slot is an Oracle idea.
                const isOracleIdea = idea.oracle_generated === true;

                // We also check if the existing card in the DOM reflects this.
                // A simple way is to check the button's inner HTML for the icon.
                const lineageButton = existingCard.querySelector('.view-lineage');
                const cardIsAlreadyOracle = lineageButton && lineageButton.innerHTML.includes('fa-eye');

                if (isOracleIdea && !cardIsAlreadyOracle) {
                    // The data is now an Oracle idea, but the card isn't.
                    // This means a replacement happened. We must re-render the card completely.
                    console.log(`Oracle replacement detected for idea ${ideaIndex} in generation ${index}. Re-rendering card.`);
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

            // Extract idea data (handles nested structures from Firestore)
            const ideaData = getIdeaData(idea);

            // Mark elite ideas with data attribute for styling
            if (ideaData.elite_selected) {
                card.setAttribute('data-elite', 'true');
            }

            // Create a plain text preview for the card
            const plainPreview = createCardPreview(ideaData.content, 150);

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
                    <h5 class="card-title">${ideaData.title || 'Untitled'}</h5>
                    <div class="card-text">
                        <p>${plainPreview}</p>
                    </div>
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

            scrollContainer.appendChild(card);

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

    // For debugging
    console.log("Rendering markdown for:", idea.content);

    // Set the content
    const renderedContent = renderMarkdown(idea.content || '');
    modalContent.innerHTML = renderedContent;

    // For debugging
    console.log("Rendered content:", renderedContent);

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

        // Use specific prompts if available (translation layer), otherwise use raw contexts (legacy)
        const displayContent = specificPrompts.length > 0 && specificPrompts[currentContextIndex]
            ? specificPrompts[currentContextIndex]
            : contexts[currentContextIndex];

        const displayTitle = specificPrompts.length > 0 ? "Specific Prompt" : "Context Pool";

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

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

async function loadEvolutions() {
    // Legacy function - now redirects to loadHistory
    await loadHistory();
}

// Global history state
let historyItems = [];
let selectedHistoryItem = null;
let currentHistoryFilter = 'all';

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

function filterHistory(filter) {
    currentHistoryFilter = filter;

    // Update button states
    ['all', 'saved', 'checkpoint'].forEach(f => {
        const btn = document.getElementById(`historyFilter${f.charAt(0).toUpperCase() + f.slice(1)}`);
        if (btn) {
            btn.classList.toggle('active', f === filter);
        }
    });

    // Filter and render
    let filtered = historyItems;
    if (filter !== 'all') {
        filtered = historyItems.filter(item => item.type === filter);
    }
    renderHistoryList(filtered, false);
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
        document.getElementById('selectedCheckpointId').value = item.id;
    }
}

function updateSelectedEvolutionCard(item) {
    const card = document.getElementById('selectedEvolutionCard');
    if (!card || !item) {
        if (card) card.style.display = 'none';
        return;
    }

    card.style.display = 'block';

    // Update status badge
    const statusBadge = document.getElementById('selectedEvolutionStatus');
    statusBadge.className = `badge badge-status-${item.status}`;
    statusBadge.textContent = formatStatus(item.status);

    // Update date
    const dateEl = document.getElementById('selectedEvolutionDate');
    if (item.timestamp) {
        dateEl.textContent = new Date(item.timestamp).toLocaleDateString();
    } else {
        dateEl.textContent = '';
    }

    // Update name
    const nameEl = document.getElementById('selectedEvolutionName');
    nameEl.textContent = item.display_name || item.id;
    nameEl.title = item.display_name || item.id;

    // Update meta
    const metaEl = document.getElementById('selectedEvolutionMeta');
    metaEl.textContent = `Gen ${item.generations}${item.total_generations ? '/' + item.total_generations : ''} • ${item.idea_type || 'Unknown type'}`;

    // Update actions
    const actionsEl = document.getElementById('selectedEvolutionActions');
    let actionsHtml = '';

    // Store checkpoint ID for actions
    if (item.checkpoint_id) {
        document.getElementById('selectedCheckpointId').value = item.checkpoint_id;
    } else if (item.type === 'saved') {
        // For saved evolutions, we might need to create a checkpoint first
        document.getElementById('selectedCheckpointId').value = '';
    }

    if (item.can_resume) {
        actionsHtml += `<button class="btn btn-sm btn-success flex-grow-1" onclick="handleHistoryResume('${item.checkpoint_id || item.id}')">
            <i class="fas fa-play me-1"></i>Resume
        </button>`;
    }
    if (item.can_continue) {
        // Inline continue with generation input
        actionsHtml += `
            <div class="continue-action-group d-flex gap-1 flex-grow-1" id="continueActionGroup">
                <button class="btn btn-sm btn-info flex-grow-1" id="continueBtn" onclick="showContinueInput('${item.checkpoint_id || item.id}', '${item.type}')">
                    <i class="fas fa-forward me-1"></i>Continue
                </button>
            </div>`;
    }
    if (item.can_rate) {
        actionsHtml += `<button class="btn btn-sm btn-outline-primary" onclick="navigateToRatePage('${item.type === 'saved' ? item.id : ''}')">
            <i class="fas fa-star me-1"></i>Rate
        </button>`;
    }

    // Add rename button for evolutions that support it
    if (item.can_rename) {
        actionsHtml += `<button class="btn btn-sm btn-outline-secondary" onclick="showRenameInput('${item.id}')" title="Rename">
            <i class="fas fa-edit"></i>
        </button>`;
    }

    // Add delete button
    if (item.can_delete) {
        actionsHtml += `<button class="btn btn-sm btn-outline-danger" onclick="confirmDeleteEvolution('${item.id}')" title="Delete">
            <i class="fas fa-trash"></i>
        </button>`;
    }

    actionsEl.innerHTML = actionsHtml || '<span class="text-muted small">No actions available</span>';

    // Store the item ID in hidden field
    document.getElementById('evolutionSelect').value = item.type === 'saved' ? item.id : '';
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

// Legacy functions for main view selected card (keeping for compatibility)
function showContinueInput(itemId, itemType) {
    showContinueInputInline(itemId, itemType);
}

function cancelContinueInput(itemId, itemType) {
    cancelContinueInputInline(itemId, itemType);
}

async function executeContinue(itemId, itemType) {
    await executeContinueInline(itemId, itemType);
}

// Show rename input
function showRenameInput(evolutionId) {
    const nameEl = document.getElementById('selectedEvolutionName');
    if (!nameEl) return;

    const currentName = nameEl.textContent;
    nameEl.outerHTML = `
        <div class="rename-input-group d-flex gap-1" id="renameInputGroup">
            <input type="text" class="form-control form-control-sm" id="renameInput" value="${currentName}" style="min-width: 120px;">
            <button class="btn btn-sm btn-success" onclick="executeRename('${evolutionId}')">
                <i class="fas fa-check"></i>
            </button>
            <button class="btn btn-sm btn-outline-secondary" onclick="cancelRename('${currentName}')">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    // Focus and select input
    setTimeout(() => {
        const input = document.getElementById('renameInput');
        if (input) {
            input.focus();
            input.select();
        }
    }, 50);
}

// Cancel rename
function cancelRename(originalName) {
    const group = document.getElementById('renameInputGroup');
    if (group) {
        group.outerHTML = `<div class="fw-medium text-truncate" id="selectedEvolutionName" style="max-width: 180px;">${originalName}</div>`;
    }
}

// Execute rename
async function executeRename(evolutionId) {
    const input = document.getElementById('renameInput');
    const newName = input?.value?.trim();

    if (!newName) {
        alert('Please enter a name');
        return;
    }

    try {
        const response = await fetch(`/api/evolution/${evolutionId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });

        if (response.ok) {
            // Update the display
            cancelRename(newName);
            // Refresh history list
            await loadHistory();
            // Update the selected item
            if (selectedHistoryItem) {
                selectedHistoryItem.display_name = newName;
            }
        } else {
            const data = await response.json();
            alert(data.message || 'Failed to rename');
        }
    } catch (error) {
        console.error('Error renaming:', error);
        alert('Error renaming evolution');
    }
}

// Confirm and delete evolution
async function confirmDeleteEvolution(evolutionId) {
    if (!confirm('Are you sure you want to delete this evolution? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/evolution/${evolutionId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Clear selection and refresh
            clearSelectedEvolution();
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

                    if (data.data.tournament_history) {
                        tournamentHistory = data.data.tournament_history;
                        renderTournamentDetails(tournamentHistory);
                    }

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

                    if (data.tournament_history) {
                        tournamentHistory = data.tournament_history;
                        renderTournamentDetails(tournamentHistory);
                    }

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

    // Update the action button
    const actionsEl = document.getElementById('selectedEvolutionActions');
    if (actionsEl) {
        actionsEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Resuming...';
    }

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

    // Update the action button to show starting state
    const group = document.getElementById('continueActionGroup');
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
            // Restore the continue button
            cancelContinueInput(itemId, itemType);
        }
    } catch (error) {
        console.error("Error continuing:", error);
        alert("Error continuing evolution: " + error.message);
        // Restore the continue button
        cancelContinueInput(itemId, itemType);
    }
}

function clearSelectedEvolution() {
    selectedHistoryItem = null;

    // Hide selected evolution card in main view
    const mainCard = document.getElementById('selectedEvolutionCard');
    if (mainCard) mainCard.style.display = 'none';

    document.getElementById('evolutionSelect').value = '';
    document.getElementById('selectedCheckpointId').value = '';

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

// Modify existing loadCurrentEvolution function
async function loadCurrentEvolution() {
    // ... existing code ...
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

        // Check if this is a new evolution (history is empty but is_running is true)
        if (data.is_running && (!data.history || data.history.length === 0)) {
            console.log("New evolution detected, resetting UI");
            // Reset generations display but keep progress bar
            const container = document.getElementById('generations-container');
            if (container) {
                container.innerHTML = '';
            }
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
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);

            // Update prominent percentage display
            if (progressPercentage) {
                progressPercentage.textContent = `${Math.round(progress)}%`;
            }

            // Cap the display at 100%
            if (progress >= 100) {
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
            tournamentHistory = data.tournament_history;
            renderTournamentDetails(tournamentHistory);
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
            addActivityLogItem('🔮 Oracle injected diverse idea into population', 'info');
        }

        // Handle elite selection updates
        if (data.elite_selection_update) {
            addActivityLogItem('⭐ Elite idea selected for next generation', 'info');
        }

        // Display token counts if available (Live updates)
        if (data.token_counts) {
            displayTokenCounts(data.token_counts);
        }

        if (data.contexts && data.contexts.length > 0) {
            contexts = data.contexts;
            specificPrompts = data.specific_prompts || [];
            breedingPrompts = data.breeding_prompts || [];
            currentContextIndex = 0;
            updateContextDisplay();
            document.querySelector('.context-navigation').style.display = 'block';
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

// Add debouncing to save operations
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Use debounced version of save function
const debouncedSave = debounce(saveEvolution, 300);

async function handleSave() {
    // Show loading indicator
    showLoadingIndicator();

    try {
        // Perform save operation
        await saveEvolution();

        // Update UI after save completes
        updateUIAfterSave();
    } catch (error) {
        console.error("Save failed:", error);
        showErrorMessage("Save failed. Please try again.");
    } finally {
        // Always hide loading indicator
        hideLoadingIndicator();
    }
}

// Add these utility functions for modal overlay management
function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.zIndex = '1000';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';

    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.style.width = '50px';
    spinner.style.height = '50px';
    spinner.style.border = '5px solid #f3f3f3';
    spinner.style.borderTop = '5px solid #3498db';
    spinner.style.borderRadius = '50%';
    spinner.style.animation = 'spin 1s linear infinite';

    // Add keyframes for spinner animation
    const style = document.createElement('style');
    style.textContent = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
    document.head.appendChild(style);

    overlay.appendChild(spinner);
    document.body.appendChild(overlay);
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        document.body.removeChild(overlay);
    }
}

// Replace the existing showLoadingIndicator and hideLoadingIndicator functions
function showLoadingIndicator() {
    showLoadingOverlay();
    // Disable all interactive elements
    document.querySelectorAll('button, input, select').forEach(el => {
        el.dataset.wasDisabled = el.disabled;
        el.disabled = true;
    });
}

function hideLoadingIndicator() {
    hideLoadingOverlay();
    // Re-enable elements that weren't disabled before
    document.querySelectorAll('button, input, select').forEach(el => {
        if (el.dataset.wasDisabled !== 'true') {
            el.disabled = false;
        }
        delete el.dataset.wasDisabled;
    });
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

// Update the saveEvolution function to use the new overlay
async function saveEvolution(data) {
    showLoadingIndicator();

    try {
        // Get the filename from the UI or generate one
        const filename = document.getElementById('saveFilename')?.value ||
            `evolution_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;

        const response = await fetch('/api/save-evolution', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: data || currentEvolutionData,
                filename: filename
            })
        });

        if (!response.ok) {
            throw new Error(`Save failed: ${response.statusText}`);
        }

        // Show success message
        showSuccessMessage('Evolution saved successfully!');

        return await response.json();
    } catch (error) {
        console.error('Error saving evolution:', error);
        showErrorMessage(`Save failed: ${error.message}`);
        throw error;
    } finally {
        hideLoadingIndicator();
    }
}

// Add a success message function
function showSuccessMessage(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.style.position = 'fixed';
    successDiv.style.top = '20px';
    successDiv.style.left = '50%';
    successDiv.style.transform = 'translateX(-50%)';
    successDiv.style.backgroundColor = '#4CAF50';
    successDiv.style.color = 'white';
    successDiv.style.padding = '15px';
    successDiv.style.borderRadius = '5px';
    successDiv.style.zIndex = '1001';
    successDiv.textContent = message;

    document.body.appendChild(successDiv);

    // Remove after 3 seconds
    setTimeout(() => {
        if (document.body.contains(successDiv)) {
            document.body.removeChild(successDiv);
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

// Function to show continue button (for completed evolutions only)
// Now just stores checkpoint ID - the history panel handles the UI
function showContinueButton(checkpointId) {
    if (checkpointId) {
        localStorage.setItem('lastCheckpointId', checkpointId);
    }
    // History panel handles resume/continue UI now
}

// Function to hide resume/continue buttons
// Now a no-op - history panel handles the UI
function hideResumeButtons() {
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

    // Reset diversity tracking
    window._loggedDiversityGens = new Set();

    // Reset evolution status
    isEvolutionRunning = false;
    currentEvolutionData = null;
    tournamentHistory = [];
    tournamentCountTouched = false;
    lastRenderedHistoryVersion = -1;
    lastStoredHistoryVersion = -1;

    // Reset progress bar if it exists, or create a new one
    const existingProgressContainer = document.getElementById('progress-container');
    if (existingProgressContainer) {
        // Remove the old progress container
        existingProgressContainer.remove();
    }
    const tournamentContainer = document.getElementById('tournament-details-container');
    if (tournamentContainer) {
        tournamentContainer.remove();
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

    // Ensure tournament details container is present
    ensureTournamentDetailsContainer();
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

function ensureTournamentDetailsContainer() {
    let container = document.getElementById('tournament-details-container');
    if (container) {
        return container;
    }

    container = document.createElement('div');
    container.id = 'tournament-details-container';
    container.className = 'tournament-details-card mb-4';
    container.innerHTML = `
        <div class="tournament-details-header">
            <h5 class="mb-0">Swiss Tournament Details</h5>
            <button type="button" class="btn btn-sm btn-outline-secondary" id="tournament-details-toggle">Hide</button>
        </div>
        <div class="tournament-details-body" id="tournament-details-body" style="display: block;">
            <div class="text-muted">No tournament data yet.</div>
        </div>
    `;

    const generationsContainer = document.getElementById('generations-container');
    generationsContainer.parentNode.insertBefore(container, generationsContainer);

    const toggleBtn = container.querySelector('#tournament-details-toggle');
    const body = container.querySelector('#tournament-details-body');
    if (toggleBtn && body) {
        toggleBtn.addEventListener('click', () => {
            const isHidden = body.style.display === 'none';
            body.style.display = isHidden ? 'block' : 'none';
            toggleBtn.textContent = isHidden ? 'Hide' : 'Show';
        });
    }

    return container;
}

function renderTournamentDetails(history) {
    const container = ensureTournamentDetailsContainer();
    const body = container.querySelector('#tournament-details-body');
    if (!body) return;

    if (!history || history.length === 0) {
        body.innerHTML = '<div class="text-muted">No tournament data yet.</div>';
        return;
    }

    const html = history.map((entry, idx) => {
        const genLabel = `Gen ${entry.generation}`;
        const rounds = (entry.rounds || []).map((round) => {
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

        const openAttr = idx === history.length - 1 ? 'open' : '';
        return `
            <details class="tournament-gen" ${openAttr}>
                <summary>${genLabel}</summary>
                <div class="tournament-grid">
                    ${rounds || '<div class="text-muted">No rounds recorded.</div>'}
                </div>
            </details>
        `;
    }).join('');

    body.innerHTML = html;
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
function addActivityLogItem(message, type = 'info') {
    lastActivityTime = Date.now();

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Add to activity log array
    activityLog.unshift({ message, type, timestamp });

    // Keep only the most recent items
    if (activityLog.length > MAX_ACTIVITY_LOG_ITEMS) {
        activityLog = activityLog.slice(0, MAX_ACTIVITY_LOG_ITEMS);
    }

    // Update the UI
    const logContainer = document.getElementById('activity-log-items');
    if (logContainer) {
        logContainer.innerHTML = activityLog.map(item => `
            <div class="activity-item activity-${item.type}">
                <span class="activity-time">${item.timestamp}</span>
                <span class="activity-message">${item.message}</span>
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

    if (statusTextEl && statusText) {
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

        // Use specific prompts if available (translation layer), otherwise use raw contexts (legacy)
        const displayContent = specificPrompts.length > 0 && specificPrompts[contextIndex]
            ? specificPrompts[contextIndex]
            : contexts[contextIndex];

        const displayTitle = specificPrompts.length > 0 ? "Specific Prompt" : "Context Pool";

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
                'This is the specific prompt generated from the context pool to create this idea.' :
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
    }

    // Get cost information
    const totalCost = tokenCounts.cost.total_cost.toFixed(4);
    const totalTokens = tokenCounts.total.toLocaleString();

    // Get estimated total cost if available
    let estimatedCostHtml = '';
    if (tokenCounts.cost.estimated_total_cost !== undefined) {
        const estimatedCost = tokenCounts.cost.estimated_total_cost.toFixed(4);
        estimatedCostHtml = `<div class="ms-3 border-start ps-3">
            <h6 class="mb-0">Est. Total: <strong>$${estimatedCost}</strong></h6>
            <small class="text-muted">Projected</small>
        </div>`;
    }

    // Store the token data for the modal
    tokenCountsContainer.dataset.tokenCounts = JSON.stringify(tokenCounts);

    // Update the container content with a simple cost display
    tokenCountsContainer.innerHTML = `
        <div class="card-body d-flex justify-content-between align-items-center p-3">
            <div class="d-flex">
                <div>
                    <h6 class="mb-0">Cost: <strong>$${totalCost}</strong></h6>
                    <small class="text-muted">${totalTokens} tokens</small>
                </div>
                ${estimatedCostHtml}
            </div>
            <button id="token-details-btn" class="btn btn-sm btn-outline-primary">
                <i class="fas fa-info-circle"></i> Details
            </button>
        </div>
    `;

    // Add event listener to the details button
    document.getElementById('token-details-btn').addEventListener('click', function () {
        showTokenDetailsModal(tokenCounts);
    });
}

// Function to show the token details modal
function showTokenDetailsModal(tokenCounts) {
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
    const possibleComponents = ['ideator', 'formatter', 'critic', 'breeder', 'genotype_encoder', 'oracle'];

    for (const componentName of possibleComponents) {
        if (tokenCounts[componentName]) {
            const component = tokenCounts[componentName];
            componentData.push({
                name: componentName.charAt(0).toUpperCase() + componentName.slice(1).replace('_', ' '),
                displayName: componentName === 'genotype_encoder' ? 'Genotype Encoder' :
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
    overall: [],
    perGeneration: [],
    interGeneration: []
};
let isSplitAxes = false;

/**
 * Get chart scales configuration based on current axis mode
 */
function getDiversityChartScales() {
    const baseScales = {
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
                text: isSplitAxes ? 'Diversity Score (Population & Per-Gen)' : 'Diversity Score',
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

    // Add second y-axis if in split mode
    if (isSplitAxes) {
        baseScales.y1 = {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
                display: true,
                text: 'Inter-Generation Diversity',
                color: 'rgba(34, 197, 94, 1)',
                font: {
                    size: 14,
                    weight: 'bold'
                }
            },
            grid: {
                drawOnChartArea: false,
            },
            ticks: {
                color: 'rgba(34, 197, 94, 1)',
                font: {
                    size: 12
                }
            },
            beginAtZero: false
        };
    }

    return baseScales;
}

/**
 * Toggle between combined and split y-axes
 */
function toggleDiversityAxes() {
    if (!diversityChart) {
        console.warn('Diversity chart not initialized');
        return;
    }

    isSplitAxes = !isSplitAxes;

    // Update button text
    const toggleButton = document.getElementById('diversity-axis-toggle');
    if (toggleButton) {
        const icon = toggleButton.querySelector('i');
        const text = isSplitAxes ? ' Combined Axes' : ' Split Axes';
        toggleButton.innerHTML = `<i class="fas fa-arrows-alt-v"></i>${text}`;
    }

    // Update dataset yAxisID for inter-generation diversity
    if (isSplitAxes) {
        diversityChart.data.datasets[2].yAxisID = 'y1';
    } else {
        diversityChart.data.datasets[2].yAxisID = 'y';
    }

    // Update scales configuration
    diversityChart.options.scales = getDiversityChartScales();

    // Update the chart
    diversityChart.update('active');
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
                datasets: [
                    {
                        label: 'Overall Population Diversity',
                        data: [],
                        borderColor: 'rgba(139, 124, 246, 1)',
                        backgroundColor: 'rgba(139, 124, 246, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: 'rgba(139, 124, 246, 1)',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: 'rgba(139, 124, 246, 1)',
                        pointHoverBorderColor: '#ffffff',
                        pointHoverBorderWidth: 3
                    },
                    {
                        label: 'Per-Generation Diversity',
                        data: [],
                        borderColor: 'rgba(244, 114, 182, 1)',
                        backgroundColor: 'rgba(244, 114, 182, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: 'rgba(244, 114, 182, 1)',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: 'rgba(244, 114, 182, 1)',
                        pointHoverBorderColor: '#ffffff',
                        pointHoverBorderWidth: 3,
                        borderDash: [5, 5]
                    },
                    {
                        label: 'Inter-Generation Diversity',
                        data: [],
                        borderColor: 'rgba(34, 197, 94, 1)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: 'rgba(34, 197, 94, 1)',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: 'rgba(34, 197, 94, 1)',
                        pointHoverBorderColor: '#ffffff',
                        pointHoverBorderWidth: 3,
                        borderDash: [10, 3],
                        spanGaps: true, // Skip null values instead of breaking the line
                        yAxisID: 'y'
                    }
                ]
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
                                const value = parseFloat(context.parsed.y).toFixed(4);
                                const axisLabel = isSplitAxes && context.datasetIndex === 2 ?
                                    ' (right axis)' : (isSplitAxes ? ' (left axis)' : '');
                                return `${context.dataset.label}: ${value}${axisLabel}`;
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
                },
                onResize: function (chart, size) {
                    // Ensure proper sizing on resize
                    chart.canvas.style.height = '400px';
                    chart.canvas.style.width = '100%';
                }
            }
        };

        // Create the chart
        diversityChart = new Chart(ctx, config);

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
function processDiversityData(diversityHistory) {
    if (!diversityHistory || diversityHistory.length === 0) {
        return { labels: [], overall: [], perGeneration: [], interGeneration: [] };
    }

    const labels = [];
    const overallScores = [];
    const perGenerationScores = [];
    const interGenerationScores = [];

    console.log("Processing diversity history:", diversityHistory);

    diversityHistory.forEach((diversityData, index) => {
        // Check if we have valid diversity data
        if (!diversityData) {
            console.log(`Skipping index ${index}: Invalid data`);
            return;
        }

        // Handle case where diversity calculation is disabled or failed
        if (diversityData.enabled === false) {
            console.log(`Skipping index ${index}: Diversity calculation disabled`);
            return;
        }

        if (diversityData.error) {
            console.log(`Skipping index ${index}: Error in diversity data:`, diversityData.error);
            return;
        }

        // Add to charts - we have valid data
        labels.push(index.toString());

        // Try different property names for diversity score
        const overallScore = diversityData.diversity_score ||
            diversityData.overall_diversity ||
            diversityData.score || 0;
        overallScores.push(overallScore);

        // Try to find per-generation diversity
        let perGenScore = 0;
        if (diversityData.generation_diversities && Array.isArray(diversityData.generation_diversities)) {
            // Find the diversity for the current generation
            const currentGenData = diversityData.generation_diversities.find(gen => gen.generation === index);
            perGenScore = currentGenData ? (currentGenData.diversity_score || currentGenData.score || 0) : 0;
        } else if (diversityData.current_generation_diversity) {
            perGenScore = diversityData.current_generation_diversity;
        } else {
            // Fallback: use overall score for per-generation as well
            perGenScore = overallScore;
        }

        perGenerationScores.push(perGenScore);

        // Extract inter-generation diversity (skip None/null values)
        const interGenScore = diversityData.inter_generation_diversity;
        if (interGenScore !== null && interGenScore !== undefined) {
            interGenerationScores.push(interGenScore);
        } else {
            interGenerationScores.push(null); // Chart.js can handle null values
        }
    });

    return {
        labels,
        overall: overallScores,
        perGeneration: perGenerationScores,
        interGeneration: interGenerationScores
    };
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

        // Update chart data
        diversityChart.data.labels = processedData.labels;
        diversityChart.data.datasets[0].data = processedData.overall;
        diversityChart.data.datasets[1].data = processedData.perGeneration;
        diversityChart.data.datasets[2].data = processedData.interGeneration;

        // Update the chart with animation
        diversityChart.update('default');

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
                plotContainer.style.height = '450px';
                plotContainer.style.width = '100%';
                plotContainer.style.padding = '20px';
                plotContainer.style.boxSizing = 'border-box';

                // Ensure canvas takes full available space
                canvas.style.display = 'block';
                canvas.style.width = '100%';
                canvas.style.height = '400px';
                canvas.style.maxWidth = '100%';

                // Force chart to resize to container
                setTimeout(() => {
                    if (diversityChart) {
                        diversityChart.resize();
                    }
                }, 100);
            }
        }
    } catch (error) {
        console.error('❌ Error updating diversity chart:', error);
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
        diversityChart.data.datasets[0].data = [];
        diversityChart.data.datasets[1].data = [];
        diversityChart.data.datasets[2].data = [];
        diversityChart.update();
    }

    // Reset global diversity data
    diversityData = {
        overall: [],
        perGeneration: []
    };
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
        // Ensure proper sizing after update
        setTimeout(ensureDiversityChartSizing, 100);
    }
}

/**
 * Initialize diversity plot when evolution starts
 */
function initializeDiversityPlot() {
    // Make sure the canvas is ready
    setTimeout(() => {
        const canvas = document.getElementById('diversity-chart');
        if (canvas) {
            // Restore canvas element if it was replaced
            const plotContainer = document.querySelector('.diversity-plot-container');
            if (plotContainer && !plotContainer.querySelector('canvas')) {
                plotContainer.innerHTML = '<canvas id="diversity-chart" width="800" height="400"></canvas>';
            }
            initializeDiversityChart();
        }
    }, 100);
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

    // Add event listener for diversity axis toggle
    const axisToggleButton = document.getElementById('diversity-axis-toggle');
    if (axisToggleButton) {
        axisToggleButton.addEventListener('click', toggleDiversityAxes);
    }
});

/**
 * Debug function to diagnose diversity chart issues
 */
function debugDiversityChart() {
    console.log('🔍 DIVERSITY CHART DEBUG INFO');
    console.log('================================');

    const plotSection = document.getElementById('diversity-plot-section');
    const plotContainer = document.querySelector('.diversity-plot-container');
    const canvas = document.getElementById('diversity-chart');

    console.log('Plot section exists:', !!plotSection);
    console.log('Plot section display:', plotSection?.style.display);
    console.log('Plot container exists:', !!plotContainer);
    console.log('Canvas exists:', !!canvas);
    console.log('Canvas dimensions:', canvas ? `${canvas.offsetWidth}x${canvas.offsetHeight}` : 'N/A');
    console.log('Chart instance exists:', !!diversityChart);
    console.log('Chart canvas valid:', diversityChart?.canvas ? 'Yes' : 'No');

    if (canvas) {
        const rect = canvas.getBoundingClientRect();
        console.log('Canvas bounding rect:', rect);
    }

    // Check localStorage
    const storedData = localStorage.getItem('currentEvolutionData');
    if (storedData) {
        try {
            const data = JSON.parse(storedData);
            console.log('LocalStorage data format:', Array.isArray(data) ? 'Old (array)' : 'New (object)');
            if (data.diversity_history) {
                console.log('Diversity history entries:', data.diversity_history.length);
            }
        } catch (e) {
            console.log('LocalStorage parse error:', e.message);
        }
    } else {
        console.log('No localStorage data found');
    }

    console.log('================================');

    // Try to fix sizing if chart exists
    if (diversityChart && canvas) {
        console.log('Attempting to fix chart sizing...');
        ensureDiversityChartSizing();
    }
}

// Make debug function available globally for browser console
window.debugDiversityChart = debugDiversityChart;

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
        const contentItems = displayContent
            .split('\n')
            .filter(item => item.trim())
            .map(item => `<div class="prompt-item">${item.trim()}</div>`)
            .join('');

        promptModalContent.innerHTML = `
            <div class="prompt-content">
                <div class="alert alert-info mb-3">
                    <strong>${promptType === 'initial' ? 'Initial' : 'Breeding'} Prompt:</strong>
                    ${promptType === 'initial'
                ? 'This is the specific prompt used to create this initial idea from the context pool.'
                : 'This is the specific prompt generated from parent concepts to create this bred idea.'}
                </div>
                ${contentItems}
            </div>
        `;
    } else {
        promptModalContent.innerHTML = `<p class="text-muted">${displayContent}</p>`;
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

/**
 * Set up model change listener to show/hide thinking budget control
 */
function setupModelChangeListener() {
    const modelSelect = document.getElementById('modelType');
    const thinkingBudgetContainer = document.getElementById('thinkingBudgetContainer');

    if (modelSelect && thinkingBudgetContainer) {
        modelSelect.addEventListener('change', function () {
            updateThinkingBudgetVisibility();
        });

        // Initialize visibility on page load
        updateThinkingBudgetVisibility();
    }
}

/**
 * Update thinking budget control visibility based on selected model
 */
function updateThinkingBudgetVisibility() {
    const modelSelect = document.getElementById('modelType');
    const thinkingBudgetContainer = document.getElementById('thinkingBudgetContainer');
    const dynamicRadio = document.getElementById('thinkingDynamic');
    const customContainer = document.getElementById('thinkingBudgetCustomContainer');

    if (!modelSelect || !thinkingBudgetContainer) return;

    const selectedModel = modelSelect.value;
    const supportsThinking = selectedModel.includes('2.5') || selectedModel.includes('3-pro');

    if (supportsThinking) {
        thinkingBudgetContainer.style.display = 'block';

        // Reset to dynamic mode when switching models
        if (dynamicRadio) {
            dynamicRadio.checked = true;
        }
        if (customContainer) {
            customContainer.style.display = 'none';
        }

        configureThinkingBudgetForModel(selectedModel);
    } else {
        thinkingBudgetContainer.style.display = 'none';
    }
}

/**
 * Configure thinking budget for specific model
 */
function configureThinkingBudgetForModel(modelName) {
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');
    const thinkingBudgetHelp = document.getElementById('thinkingBudgetHelp');
    const thinkingDisabledOption = document.getElementById('thinkingDisabledOption');
    const thinkingBudgetMin = document.getElementById('thinkingBudgetMin');
    const thinkingBudgetMax = document.getElementById('thinkingBudgetMax');

    if (!thinkingBudgetSlider || !thinkingBudgetHelp || !thinkingDisabledOption) return;

    // Configuration based on model specifications
    const configs = {
        'gemini-2.5-pro': {
            min: 128,
            max: 32768,
            default: 128,  // Minimum possible since can't disable
            defaultMode: 'custom',  // Use custom mode with minimum value
            canDisable: false,
            help: 'Minimum thinking budget: 128 tokens (cannot be disabled for Pro model)'
        },
        'gemini-3-pro-preview': {
            min: 128,
            max: 32768,
            default: 128,  // Minimum possible since can't disable
            defaultMode: 'custom',  // Use custom mode with minimum value
            canDisable: false,
            help: 'Minimum thinking budget: 128 tokens (cannot be disabled for Pro model)'
        },
        'gemini-2.5-flash': {
            min: 128,  // Custom range starts at 128
            max: 24576,
            default: 0,  // Disabled by default
            defaultMode: 'disabled',  // Use disabled mode
            canDisable: true,
            help: 'Thinking disabled by default (0-24576 tokens available for custom)'
        },
        'gemini-2.5-flash-lite': {
            min: 512,  // Custom range starts at 512
            max: 24576,
            default: 0,  // Disabled by default
            defaultMode: 'disabled',  // Use disabled mode
            canDisable: true,
            help: 'Thinking disabled by default (0-24576 tokens available for custom)'
        }
    };

    const config = configs[modelName];
    if (!config) return;

    // Show/hide disabled option based on model capability
    if (config.canDisable) {
        thinkingDisabledOption.style.display = 'block';
    } else {
        thinkingDisabledOption.style.display = 'none';
    }

    // Set the default mode based on model configuration
    const dynamicRadio = document.getElementById('thinkingDynamic');
    const disabledRadio = document.getElementById('thinkingDisabled');
    const customRadio = document.getElementById('thinkingCustom');

    // Clear all selections first
    if (dynamicRadio) dynamicRadio.checked = false;
    if (disabledRadio) disabledRadio.checked = false;
    if (customRadio) customRadio.checked = false;

    // Set the appropriate default based on config
    if (config.defaultMode === 'disabled' && config.canDisable) {
        disabledRadio.checked = true;
    } else if (config.defaultMode === 'custom') {
        customRadio.checked = true;
    } else {
        // Fallback to dynamic
        dynamicRadio.checked = true;
    }

    // Update slider range and default value
    thinkingBudgetSlider.min = config.min;
    thinkingBudgetSlider.max = config.max;

    // Set slider to model's default value (128 for Pro, doesn't matter for disabled modes)
    if (config.defaultMode === 'custom') {
        thinkingBudgetSlider.value = config.default;
    } else {
        // Set to a reasonable value for when user switches to custom later
        thinkingBudgetSlider.value = Math.max(config.min, 512);
    }

    // Update step size based on range (larger steps for larger ranges)
    const range = config.max - config.min;
    if (range > 10000) {
        thinkingBudgetSlider.step = 256;  // Larger steps for big ranges
    } else if (range > 5000) {
        thinkingBudgetSlider.step = 128;
    } else {
        thinkingBudgetSlider.step = 64;
    }

    // Update display elements
    thinkingBudgetHelp.textContent = config.help;
    if (thinkingBudgetMin) {
        thinkingBudgetMin.textContent = config.min.toLocaleString();
    }
    if (thinkingBudgetMax) {
        thinkingBudgetMax.textContent = config.max.toLocaleString();
    }

    // Update the mode display (show/hide custom slider)
    updateThinkingBudgetMode();

    // Update the slider value display
    updateThinkingBudgetDisplay();
}

/**
 * Update thinking budget display value
 */
function updateThinkingBudgetDisplay() {
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');
    const thinkingBudgetValue = document.getElementById('thinkingBudgetValue');

    if (!thinkingBudgetSlider || !thinkingBudgetValue) return;

    const value = parseInt(thinkingBudgetSlider.value);
    thinkingBudgetValue.textContent = value.toLocaleString();
}

/**
 * Handle thinking budget mode changes (radio buttons)
 */
function updateThinkingBudgetMode() {
    const customContainer = document.getElementById('thinkingBudgetCustomContainer');
    const dynamicRadio = document.getElementById('thinkingDynamic');
    const disabledRadio = document.getElementById('thinkingDisabled');
    const customRadio = document.getElementById('thinkingCustom');

    if (!customContainer || !dynamicRadio || !customRadio) return;

    if (customRadio.checked) {
        // Show custom slider
        customContainer.style.display = 'block';
        updateThinkingBudgetDisplay();
    } else {
        // Hide custom slider
        customContainer.style.display = 'none';
    }
}

/**
 * Get thinking budget value for API request
 */
function getThinkingBudgetValue() {
    const modelSelect = document.getElementById('modelType');
    const dynamicRadio = document.getElementById('thinkingDynamic');
    const disabledRadio = document.getElementById('thinkingDisabled');
    const customRadio = document.getElementById('thinkingCustom');
    const thinkingBudgetSlider = document.getElementById('thinkingBudgetSlider');

    if (!modelSelect) return null;

    const selectedModel = modelSelect.value;
    const supportsThinking = selectedModel.includes('2.5') || selectedModel.includes('3-pro');

    if (!supportsThinking) {
        return null; // Don't send thinking budget for non-thinking models
    }

    // Determine value based on selected mode
    if (dynamicRadio && dynamicRadio.checked) {
        return -1; // Dynamic thinking
    } else if (disabledRadio && disabledRadio.checked) {
        return 0; // Disabled thinking
    } else if (customRadio && customRadio.checked && thinkingBudgetSlider) {
        return parseInt(thinkingBudgetSlider.value); // Custom value
    }

    // Default to dynamic if nothing selected
    return -1;
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
