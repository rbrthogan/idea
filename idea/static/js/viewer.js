// static/js/viewer.js

// static/js/viewer.js

// CSS styles for lineage modal and buttons are now in viewer.css


let pollingInterval;
let isEvolutionRunning = false;
let currentContextIndex = 0;
let contexts = [];
let specificPrompts = [];
let breedingPrompts = [];  // Store breeding prompts for each generation
let currentEvolutionId = null;
let currentEvolutionData = null;
let generations = [];
let currentModalIdea = null;

/**
 * Load available templates and populate the idea type dropdown
 */
let allTemplates = []; // Store templates globally

/**
 * Load available templates and populate the sidebar list
 */
async function loadTemplateTypes() {
    try {
        const response = await fetch('/api/template-types');
        const data = await response.json();

        if (data.status === 'success') {
            allTemplates = data.templates;
            renderTemplateList(allTemplates);

            // Set default selection if none selected
            const currentType = document.getElementById('ideaType').value;
            if (!currentType && allTemplates.length > 0) {
                selectTemplate(allTemplates[0].id, false); // Don't slide back on initial load
            }
        } else {
            console.error('Error loading template types:', data.message);
            renderTemplateList([]);
        }
    } catch (error) {
        console.error('Error loading template types:', error);
        renderTemplateList([]);
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
            <h6>${template.name}</h6>
            <p>${template.description || 'No description'}</p>
            <div class="template-actions">
                <button class="btn btn-xs btn-outline-secondary" onclick="editTemplate('${template.id}')">
                    <i class="fas fa-edit"></i> Edit
                </button>
            </div>
        `;

        container.appendChild(div);
    });
}

function selectTemplate(id, slideBack = true) {
    const template = allTemplates.find(t => t.id === id);
    if (!template) return;

    // Update hidden input
    document.getElementById('ideaType').value = id;

    // Update selector text
    const display = document.getElementById('ideaTypeDisplay');
    if (display) {
        display.textContent = template.name;
        display.classList.remove('text-muted', 'fst-italic');
    }

    // Re-render list to update active state
    renderTemplateList(allTemplates);

    if (slideBack) {
        showSidebarView('main');
    }
}

async function editTemplate(id) {
    try {
        const response = await fetch(`/api/templates/${id}`);
        if (response.ok) {
            const template = await response.json();
            currentTemplate = template;
            document.getElementById('templateEditor').value = JSON.stringify(template, null, 2);
            const modal = new bootstrap.Modal(document.getElementById('templateModal'));
            modal.show();
        }
    } catch (e) {
        console.error("Error fetching template:", e);
    }
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
window.editTemplate = editTemplate;
window.filterTemplates = filterTemplates;

// Initialize
// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    // Load initial data
    await loadTemplateTypes();
    await loadEvolutions();

    // Check if we have a current evolution running or saved
    await restoreCurrentEvolution();

    // Start polling for progress
    // pollProgress(); // Managed by restoreCurrentEvolution / startEvolution
    // setInterval(pollProgress, 1000);

    // Helper for safe event listeners
    const addListener = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, handler);
    };

    // --- Evolution Controls ---

    addListener('startButton', 'click', async function () {
        console.log("Starting evolution...");
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');

        const popSize = parseInt(document.getElementById('popSize').value);
        const generations = parseInt(document.getElementById('generations').value);
        const ideaType = document.getElementById('ideaType').value;
        const modelType = document.getElementById('modelType').value;

        const creativeTemp = parseFloat(document.getElementById('creativeTemp').value);
        const topP = parseFloat(document.getElementById('topP').value);

        // Get tournament values
        const tournamentSize = parseInt(document.getElementById('tournamentSize').value);
        const tournamentComparisons = parseInt(document.getElementById('tournamentComparisons').value);

        // Get thinking budget value (only for Gemini 2.5 models)
        const thinkingBudget = getThinkingBudgetValue();

        // Get max budget
        const maxBudgetInput = document.getElementById('maxBudget');
        const maxBudget = maxBudgetInput && maxBudgetInput.value ? parseFloat(maxBudgetInput.value) : null;

        const requestBody = {
            popSize,
            generations,
            ideaType,
            modelType,
            creativeTemp,
            topP,
            tournamentSize,
            tournamentComparisons,
            thinkingBudget,
            maxBudget
        };

        console.log("Request body JSON:", JSON.stringify(requestBody));

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

                // Reset contexts and index
                contexts = data.contexts || [];
                specificPrompts = data.specific_prompts || [];
                breedingPrompts = data.breeding_prompts || [];
                currentContextIndex = 0;

                // Create progress bar container if it doesn't exist
                if (!document.getElementById('progress-container')) {
                    createProgressBar();
                }

                // Start polling for updates
                isEvolutionRunning = true;
                pollProgress();

                updateContextDisplay();
            } else {
                console.error("Failed to run evolution:", await response.text());
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
                if (stopButton) {
                    stopButton.disabled = false;
                    stopButton.textContent = 'Stop Evolution';
                }
            }
        } catch (error) {
            console.error("Error stopping evolution:", error);
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

    // --- Template Generation ---
    addListener('generateTemplateBtn', 'click', handleTemplateGeneration);
    addListener('saveTemplateBtn', 'click', saveAndUseTemplate);
    addListener('reviewEditBtn', 'click', reviewAndEditTemplate);

    // --- State Management ---
    addListener('newEvolutionBtn', 'click', showNewEvolutionMode);
    addListener('evolutionSelect', 'change', handleEvolutionSelection);

    // --- Settings ---
    addListener('saveSettingsBtn', 'click', saveSettings);
    addListener('toggleApiKeyVisibility', 'click', toggleApiKeyVisibility);

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
                let generationsData, diversityData, tokenCounts;
                if (Array.isArray(evolutionState)) {
                    // Old format - just the history array
                    generationsData = evolutionState;
                    diversityData = [];
                    tokenCounts = null;
                } else if (evolutionState && evolutionState.history) {
                    // New format - object with history and diversity_history
                    generationsData = evolutionState.history;
                    diversityData = evolutionState.diversity_history || [];
                    tokenCounts = evolutionState.token_counts || null;
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
                        diversity_history: diversityData,
                        contexts: contexts,
                        specific_prompts: specificPrompts,
                        breeding_prompts: breedingPrompts,
                        token_counts: tokenCounts
                    };

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
        .replace(/^>+\s*/gm, '') // Remove blockquote markers
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
            scrollWrapper.style.width = '100%';
            scrollWrapper.style.overflow = 'auto';

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

            // Mark elite ideas with data attribute for styling
            if (idea.elite_selected) {
                card.setAttribute('data-elite', 'true');
            }

            // Create a plain text preview for the card
            const plainPreview = createCardPreview(idea.content, 150);

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
                    <h5 class="card-title">${idea.title || 'Untitled'}</h5>
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
                showIdeaModal(idea);
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
    const response = await fetch('/api/evolutions');
    const evolutions = await response.json();

    const select = document.getElementById('evolutionSelect');
    select.innerHTML = '<option value="">Current Evolution</option>';

    evolutions.forEach(evolution => {
        const option = document.createElement('option');
        option.value = evolution.id;
        option.textContent = `Evolution from ${new Date(evolution.timestamp).toLocaleString()} - ${evolution.filename}`;
        select.appendChild(option);
    });
}

document.getElementById('evolutionSelect').addEventListener('change', async (e) => {
    const evolutionId = e.target.value;
    if (evolutionId) {
        const response = await fetch(`/api/evolution/${evolutionId}`);
        if (response.ok) {
            const data = await response.json();
            if (data.data && data.data.history) {
                // Clear existing content
                document.getElementById('generations-container').innerHTML = '';

                // Render the generations
                renderGenerations(data.data.history);

                // Restore diversity plot if we have diversity data
                if (data.data.diversity_history && data.data.diversity_history.length > 0) {
                    console.log("Restoring diversity plot from saved evolution:", data.data.diversity_history);
                    updateDiversityChart(data.data.diversity_history);
                    // Ensure proper sizing after restoration
                    setTimeout(ensureDiversityChartSizing, 200);
                } else {
                    console.log("No diversity data found in saved evolution, resetting plot");
                    resetDiversityPlot();
                }

                // Update contexts if available
                if (data.data.contexts) {
                    contexts = data.data.contexts;
                    specificPrompts = data.data.specific_prompts || [];
                    breedingPrompts = data.data.breeding_prompts || [];
                    currentContextIndex = 0;
                    updateContextDisplay();
                    document.querySelector('.context-navigation').style.display = 'block';
                }

                // Display token counts if available
                if (data.data.token_counts) {
                    console.log("Displaying token counts from saved evolution:", data.data.token_counts);
                    displayTokenCounts(data.data.token_counts);
                }

                // Enable download button
                document.getElementById('downloadButton').disabled = false;
            }
        } else {
            console.error('Failed to load evolution:', await response.text());
        }
    } else {
        // Load current evolution from localStorage when "Current Evolution" is selected
        console.log('Loading current evolution from localStorage...');
        const restored = await restoreCurrentEvolution();

        if (!restored) {
            // Clear display if no current evolution available
            document.getElementById('generations-container').innerHTML = '';
            document.getElementById('contextDisplay').innerHTML =
                '<p class="text-muted">Context will appear here when evolution starts...</p>';
            document.querySelector('.context-navigation').style.display = 'none';
            document.getElementById('downloadButton').disabled = true;

            // Reset diversity plot when no current evolution is available
            resetDiversityPlot();
        }
    }
});

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

        const data = await response.json();
        console.log("Progress update:", data); // Add logging to see what's coming from the server

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
        }

        // Update progress bar
        const progressBar = document.getElementById('evolution-progress');
        const progressStatus = document.getElementById('progress-status');

        if (progressBar && progressStatus) {
            const progress = data.progress || 0;
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            progressBar.textContent = `${Math.round(progress)}%`;

            if (data.current_generation === 0) {
                progressStatus.textContent = `Generating Generation 0 (Initial Population)... (${Math.round(progress)}%)`;
            } else {
                progressStatus.textContent = `Generating Generation ${data.current_generation}... (${Math.round(progress)}%)`;
            }
        }

        // Always update UI with current progress if there's history data
        if (data.history && data.history.length > 0) {
            console.log("Rendering generations from progress update");
            renderGenerations(data.history);

            // Store the current evolution data in localStorage including diversity data and token counts
            const evolutionStateToStore = {
                history: data.history,
                diversity_history: data.diversity_history || [],
                contexts: data.contexts || contexts,
                specific_prompts: data.specific_prompts || specificPrompts,
                breeding_prompts: data.breeding_prompts || breedingPrompts,
                token_counts: data.token_counts || null
            };
            localStorage.setItem('currentEvolutionData', JSON.stringify(evolutionStateToStore));
        }

        // Handle diversity updates
        if (data.diversity_history) {
            handleDiversityUpdate(data);
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

            if (data.history && data.history.length > 0) {
                // Save final state and enable save button
                currentEvolutionData = data;
                renderGenerations(data.history);

                // Store the final evolution data in localStorage including diversity data and token counts
                const evolutionStateToStore = {
                    history: data.history,
                    diversity_history: data.diversity_history || [],
                    contexts: data.contexts || contexts,
                    specific_prompts: data.specific_prompts || specificPrompts,
                    breeding_prompts: data.breeding_prompts || breedingPrompts,
                    token_counts: data.token_counts || null
                };
                localStorage.setItem('currentEvolutionData', JSON.stringify(evolutionStateToStore));

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

            // Update progress status
            if (progressStatus) {
                if (data.is_stopped) {
                    progressStatus.textContent = data.stop_message || 'Evolution stopped';
                } else {
                    progressStatus.textContent = 'Evolution complete!';
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
        }

        // Handle explicit errors from backend
        if (data.error) {
            console.error("Evolution error:", data.error);
            showErrorMessage(`Evolution Error: ${data.error}`);
            if (progressStatus) {
                progressStatus.textContent = `Error: ${data.error}`;
                progressStatus.classList.add('text-danger');
            }
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

    if (startButton) {
        startButton.disabled = false;
        startButton.textContent = 'Start Evolution';
    }

    if (stopButton) {
        stopButton.disabled = true;
        stopButton.textContent = 'Stop Evolution';
        stopButton.style.display = 'none';
    }
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

    // Reset progress bar if it exists
    const progressBar = document.getElementById('evolution-progress');
    const progressStatus = document.getElementById('progress-status');
    if (progressBar && progressStatus) {
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressBar.textContent = '0%';
        progressStatus.textContent = 'Starting evolution...';
    } else {
        // Create progress bar if it doesn't exist
        createProgressBar();
    }

    // Reset evolution status
    isEvolutionRunning = false;
    currentEvolutionData = null;

    // Clear localStorage data
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

// Function to create the progress bar
function createProgressBar() {
    const progressContainer = document.createElement('div');
    progressContainer.id = 'progress-container';
    progressContainer.className = 'mb-4';
    progressContainer.innerHTML = `
        <div class="progress">
            <div id="evolution-progress" class="progress-bar" role="progressbar"
                 style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
        </div>
        <div id="progress-status" class="text-center mt-2">Starting evolution...</div>
    `;

    // Insert progress bar before generations container
    const generationsContainer = document.getElementById('generations-container');
    generationsContainer.parentNode.insertBefore(progressContainer, generationsContainer);
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

    diversityHistory.forEach((diversityData, index) => {
        // Check if we have valid diversity data
        if (!diversityData) {
            return;
        }

        // Handle case where diversity calculation is disabled or failed
        if (diversityData.enabled === false) {
            return;
        }

        if (diversityData.error) {
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

            if (canvas && plotContainer) {
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
        plotSection.style.display = 'block';

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

    if (viewName === 'main') {
        mainView.classList.add('active');
    } else {
        // Move main view to left (parallax)
        mainView.classList.add('slide-left');
        // Bring target view in
        if (targetView) targetView.classList.add('active');
    }

    // Special handling for settings
    if (viewName === 'settings') {
        checkSettingsStatus();
    }
}

// Expose to window for HTML onclick attributes
window.showSidebarView = showSidebarView;

async function checkSettingsStatus() {
    try {
        const response = await fetch('/api/settings/status');
        const data = await response.json();

        const input = document.getElementById('apiKeyInput');
        if (data.masked_key && input) {
            input.placeholder = "API Key is set (" + data.masked_key + ")";
        }
    } catch (e) {
        console.error("Error checking settings:", e);
    }
}

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
            showStatusInSettings('API Key saved successfully! Reloading...', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 1500);
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

async function handleTemplateGeneration() {
    const ideaTypeInput = document.getElementById('ideaTypeInput');
    const ideaType = ideaTypeInput.value.trim();
    if (!ideaType) return;

    const generateBtn = document.getElementById('generateTemplateBtn');
    const originalText = generateBtn.innerHTML;
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';

    try {
        const response = await fetch('/api/generate-template', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idea_type: ideaType })
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Update hidden input
            document.getElementById('ideaType').value = ideaType;

            // Update new status card
            const display = document.getElementById('ideaTypeDisplay');

            if (display) {
                display.textContent = ideaType;
                display.classList.remove('text-muted', 'fst-italic');
            }

            // Show editor
            currentTemplate = data.template;
            document.getElementById('templateEditor').value = JSON.stringify(data.template, null, 2);

            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('templateModal'));
            modal.show();
        } else {
            alert('Error generating template: ' + data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error generating template');
    } finally {
        generateBtn.disabled = false;
        generateBtn.innerHTML = originalText;
    }
}

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
