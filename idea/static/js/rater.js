function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

let currentPair = null;
let currentEvolutionId = null;
let ideasDb = {};
let currentRatingType = 'auto';
let currentGenerationView = 'death';
let autoRatingInProgress = false;
let currentModalIdea = null;
let currentEvolutionAnalytics = {
    history: [],
    diversity_history: [],
    fitness_alpha: 0.7
};
let chartFocusModalInstance = null;
let focusedChartInstance = null;
let pendingFocusedChartConfig = null;
const chartFocusLookup = {
    eloChart: () => window.eloChart,
    diversityChart: () => window.diversityChart,
    fitnessChart: () => window.fitnessChart
};

// Sorting state
let currentSortColumn = 'fitness';
let currentSortDirection = 'desc';
let originalIdeasOrder = [];
let autoRatingProgressPollTimer = null;

// Wait for Firebase auth to be ready before loading data
function waitForAuth(callback, timeout = 5000) {
    const startTime = Date.now();

    function checkAuth() {
        if (window.currentUser) {
            console.log("Auth ready, loading data...");
            callback();
        } else if (Date.now() - startTime > timeout) {
            console.warn("Auth timeout, attempting to load anyway...");
            callback();
        } else {
            setTimeout(checkAuth, 100);
        }
    }

    checkAuth();
}

// Fetch a random pair on load
window.addEventListener("load", () => {
    // Show loading state
    document.getElementById('titleA').textContent = "Loading ideas...";
    document.getElementById('contentA').innerHTML = "<p>Please wait while we load the ideas for comparison.</p>";
    document.getElementById('titleB').textContent = "";
    document.getElementById('contentB').innerHTML = "";

    // Disable voting buttons initially
    document.querySelectorAll('.vote-btn').forEach(btn => {
        btn.disabled = true;
    });

    toggleRatingMode('manual'); // Start in manual mode

    // Wait for auth before fetching data
    waitForAuth(() => {
        loadEvolutions();
        loadCurrentEvolution();
        loadModels();
    });

    // Set up retry mechanism for refreshPair
    setTimeout(() => {
        if (Object.keys(ideasDb).length < 2) {
            console.log("No ideas loaded yet, retrying refreshPair...");
            refreshPair();
        }
    }, 5000);
});

// Helper to extract title and content from possibly nested idea structures
// Handles: idea.title/content, idea.idea.title/content, or JSON-encoded content
function getIdeaData(idea) {
    const sourceIdea = (idea && typeof idea === 'object') ? idea : {};
    let title = sourceIdea.title;
    let content = sourceIdea.content;

    // Check if data is nested under an 'idea' property
    if (sourceIdea.idea && typeof sourceIdea.idea === 'object') {
        title = sourceIdea.idea.title || title;
        content = sourceIdea.idea.content || content;
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
        // Preserve other properties
        ...sourceIdea,
        title: title || 'Untitled',
        content: content || ''
    };
}

function normalizeIdeaRecord(idea, fallbackId = null) {
    const sourceIdea = (idea && typeof idea === 'object') ? idea : {};
    const normalized = getIdeaData(sourceIdea);

    if (!normalized.id) {
        normalized.id = fallbackId || generateUUID();
    }

    const legacyRating = typeof normalized.ratings === 'number'
        ? normalizeToFiniteNumber(normalized.ratings)
        : null;
    const autoRating = normalizeToFiniteNumber(normalized.ratings?.auto)
        ?? legacyRating
        ?? normalizeToFiniteNumber(normalized.elo)
        ?? 1500;
    const manualRating = normalizeToFiniteNumber(normalized.ratings?.manual)
        ?? legacyRating
        ?? 1500;
    normalized.ratings = {
        auto: Math.round(autoRating),
        manual: Math.round(manualRating)
    };
    normalized.elo = normalized.ratings.auto;

    const totalMatches = normalizeToFiniteNumber(normalized.match_count) ?? 0;
    const autoMatches = normalizeToFiniteNumber(normalized.auto_match_count) ?? 0;
    const manualMatches = normalizeToFiniteNumber(normalized.manual_match_count) ?? 0;
    normalized.match_count = Math.max(0, Math.trunc(totalMatches));
    normalized.auto_match_count = Math.max(0, Math.trunc(autoMatches));
    normalized.manual_match_count = Math.max(0, Math.trunc(manualMatches));

    return normalized;
}

function setManualSwissStatus(swissStatus) {
    const statusEl = document.getElementById('manualSwissStatus');
    if (!statusEl) {
        return;
    }

    if (!swissStatus) {
        statusEl.textContent = 'Swiss pairing active';
        return;
    }

    const roundNumber = swissStatus.round_number ?? 0;
    const remainingPairs = swissStatus.remaining_pairs_in_round;
    const byeIdeaId = swissStatus.bye_idea_id;

    let text = `Round ${roundNumber}`;
    if (typeof remainingPairs === 'number') {
        text += ` · ${remainingPairs} pairs remaining`;
    }
    if (byeIdeaId) {
        text += ` · Bye: ${byeIdeaId}`;
    }
    statusEl.textContent = text;
}

function normalizeGenerationValue(generation) {
    if (generation === null || generation === undefined || generation === '') {
        return null;
    }
    const value = Number(generation);
    if (!Number.isFinite(value)) {
        return null;
    }
    return Math.trunc(value);
}

function normalizeToFiniteNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function clampUnitInterval(value, fallback = 0.7) {
    const parsed = normalizeToFiniteNumber(value);
    if (parsed === null) {
        return fallback;
    }
    return Math.max(0, Math.min(1, parsed));
}

function setCurrentEvolutionAnalytics(rawEvolutionData) {
    const evolutionData = rawEvolutionData && typeof rawEvolutionData === 'object'
        ? rawEvolutionData
        : {};

    const history = Array.isArray(evolutionData.history) ? evolutionData.history : [];
    const diversityHistory = Array.isArray(evolutionData.diversity_history)
        ? evolutionData.diversity_history
        : [];
    const config = evolutionData.config && typeof evolutionData.config === 'object'
        ? evolutionData.config
        : {};

    currentEvolutionAnalytics = {
        history,
        diversity_history: diversityHistory,
        fitness_alpha: clampUnitInterval(
            config.fitness_alpha ?? evolutionData.fitness_alpha,
            0.7
        )
    };
}

function flattenGenerationHistory(generationsData, fallbackPrefix = 'idea') {
    const flattened = [];
    (generationsData || []).forEach((generation, genIndex) => {
        (generation || []).forEach((idea, ideaIndex) => {
            const fallbackId = `${fallbackPrefix}_${genIndex}_${ideaIndex}`;
            const normalizedIdea = normalizeIdeaRecord(idea, fallbackId);
            flattened.push({
                ...normalizedIdea,
                id: normalizedIdea.id || fallbackId,
                elo: normalizedIdea.elo ?? normalizedIdea.ratings?.auto ?? 1500,
                generation: normalizedIdea.generation !== undefined ? normalizedIdea.generation : genIndex
            });
        });
    });
    return flattened;
}

function getIdeaGenerationValue(idea, generationView = currentGenerationView) {
    const birth = normalizeGenerationValue(idea.birth_generation);
    const death = normalizeGenerationValue(idea.death_generation);
    const fallback = normalizeGenerationValue(idea.generation);

    if (generationView === 'birth') {
        return birth ?? death ?? fallback;
    }
    return death ?? birth ?? fallback;
}

function getIdeaGenerationLabel(idea, generationView = currentGenerationView) {
    const generation = getIdeaGenerationValue(idea, generationView);
    return generation !== null ? formatGenerationLabel(generation) : '?';
}

async function initializeRating(ideas, shouldRefreshPair = true) {
    // Reset state
    ideasDb = {};
    ideas = (ideas || []).map((idea, index) => normalizeIdeaRecord(idea, `idea_${index}`));

    const lifecycleById = {};

    // Initialize ideas database with IDs
    ideas.forEach(idea => {
        if (!idea.id) {
            idea.id = generateUUID();
        }
        if (!Number.isFinite(Number(idea.elo))) {
            idea.elo = normalizeToFiniteNumber(idea.ratings?.auto) ?? 1500;
        }

        const observed = normalizeGenerationValue(idea.generation);
        const explicitBirth = normalizeGenerationValue(idea.birth_generation);
        const explicitDeath = normalizeGenerationValue(idea.death_generation);
        const birthCandidate = explicitBirth ?? observed ?? explicitDeath;
        const deathCandidate = explicitDeath ?? observed ?? explicitBirth;

        if (!lifecycleById[idea.id]) {
            lifecycleById[idea.id] = {
                birth: birthCandidate,
                death: deathCandidate
            };
        } else {
            const current = lifecycleById[idea.id];
            if (birthCandidate !== null) {
                current.birth = current.birth === null ? birthCandidate : Math.min(current.birth, birthCandidate);
            }
            if (deathCandidate !== null) {
                current.death = current.death === null ? deathCandidate : Math.max(current.death, deathCandidate);
            }
        }
    });

    ideas.forEach(idea => {
        const lifecycle = lifecycleById[idea.id];
        if (lifecycle) {
            if (lifecycle.birth !== null) {
                idea.birth_generation = lifecycle.birth;
            }
            if (lifecycle.death !== null) {
                idea.death_generation = lifecycle.death;
            }
            if (idea.generation === undefined || idea.generation === null || idea.generation === '') {
                idea.generation = lifecycle.death ?? lifecycle.birth ?? 0;
            }
        }
        ideasDb[idea.id] = normalizeIdeaRecord(idea, idea.id);
    });

    syncAutoTournamentCountUI();

    if (shouldRefreshPair) {
        await refreshPair();
    }
}

// Add the markdown rendering function from viewer.js
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
            line.startsWith('<pre') ||
            line.startsWith('<blockquote')) {

            // Close paragraph if we were in one
            if (inParagraph) {
                lines[i - 1] += '</p>';
                inParagraph = false;
            }
            continue;
        }

        // Start a new paragraph if we're not in one
        if (!inParagraph) {
            lines[i] = '<p>' + lines[i];
            inParagraph = true;
        }
    }

    // Close the last paragraph if needed
    if (inParagraph) {
        lines[lines.length - 1] += '</p>';
    }

    return lines.join('\n');
}

// Update refreshPair function to use the backend API for efficient pair selection
async function refreshPair() {
    try {
        // Get all ideas
        const ideas = Object.values(ideasDb);

        if (ideas.length < 2) {
            console.log('Not enough ideas to compare, waiting for data to load...');

            // Show loading message instead of alert
            document.getElementById('titleA').textContent = "Loading ideas...";
            document.getElementById('contentA').innerHTML = "<p>Please wait while we load the ideas for comparison.</p>";
            document.getElementById('titleB').textContent = "";
            document.getElementById('contentB').innerHTML = "";

            // Disable voting buttons
            document.querySelectorAll('.vote-btn').forEach(btn => {
                btn.disabled = true;
            });

            return;
        }

        // Enable voting buttons
        document.querySelectorAll('.vote-btn').forEach(btn => {
            btn.disabled = false;
        });

        // Get next Swiss pair from backend API
        try {
            const response = await fetch('/api/get-efficient-pair', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    evolution_id: currentEvolutionId
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to get efficient pair: ${response.status}`);
            }

            const data = await response.json();
            if (!data.idea_a || !data.idea_b) {
                throw new Error('Incomplete pair payload');
            }
            const ideaA = normalizeIdeaRecord(data.idea_a, data.idea_a?.id);
            const ideaB = normalizeIdeaRecord(data.idea_b, data.idea_b?.id);
            setManualSwissStatus(data.swiss_status);

            // Update local database with the returned ideas (they may have been modified)
            if (ideaA && ideaA.id) {
                ideasDb[ideaA.id] = ideaA;
            }
            if (ideaB && ideaB.id) {
                ideasDb[ideaB.id] = ideaB;
            }

            currentPair = [ideaA, ideaB];

            // Extract nested idea data
            const ideaDataA = getIdeaData(ideaA);
            const ideaDataB = getIdeaData(ideaB);

            // Display the ideas without showing ratings
            document.getElementById('titleA').textContent = ideaDataA.title;
            document.getElementById('titleB').textContent = ideaDataB.title;

            document.getElementById('contentA').innerHTML = renderMarkdown(ideaDataA.content);
            document.getElementById('contentB').innerHTML = renderMarkdown(ideaDataB.content);

            console.log(`Swiss pair selected: ${ideaA.id} vs ${ideaB.id}`);

        } catch (error) {
            console.error('Error getting Swiss pair, falling back to random selection:', error);
            setManualSwissStatus(null);

            // Fallback to random selection if API call fails
            const [ideaA, ideaB] = getRandomPair(ideas);
            currentPair = [ideaA, ideaB];

            // Extract nested idea data
            const ideaDataA = getIdeaData(ideaA);
            const ideaDataB = getIdeaData(ideaB);

            // Display the ideas without showing ratings
            document.getElementById('titleA').textContent = ideaDataA.title;
            document.getElementById('titleB').textContent = ideaDataB.title;

            document.getElementById('contentA').innerHTML = renderMarkdown(ideaDataA.content);
            document.getElementById('contentB').innerHTML = renderMarkdown(ideaDataB.content);
        }

        // Show the comparison view
        document.querySelector('.ideas-container').style.display = 'flex';
        document.getElementById('autoRatingResults').style.display = 'none';

    } catch (error) {
        console.error('Error refreshing pair:', error);
    }
}

// Fix the manual rating functionality
async function vote(outcome) {
    if (!currentPair) return;

    try {
        const response = await fetch('/api/submit-rating', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                idea_a_id: currentPair[0].id,
                idea_b_id: currentPair[1].id,
                outcome: outcome,
                evolution_id: currentEvolutionId
            })
        });

        if (!response.ok) {
            throw new Error('Failed to submit rating');
        }

        const result = await response.json();
        console.log('Rating submitted:', result);

        // Update local ELO ratings and match counts for manual ratings
        if (result.updated_elos) {
            Object.entries(result.updated_elos).forEach(([id, elo]) => {
                if (ideasDb[id]) {
                    // Initialize ratings object if needed
                    if (!ideasDb[id].ratings) {
                        ideasDb[id].ratings = { auto: ideasDb[id].elo || 1500, manual: 1500 };
                    }

                    // Update the manual rating
                    ideasDb[id].ratings.manual = elo;

                    console.log(`Updated manual rating for idea ${id} to ${elo}`);
                }
            });
        }

        // Update match counts if provided by the server
        if (result.updated_match_counts) {
            Object.entries(result.updated_match_counts).forEach(([id, counts]) => {
                if (ideasDb[id]) {
                    ideasDb[id].match_count = counts.total;
                    ideasDb[id].manual_match_count = counts.manual;
                    ideasDb[id].auto_match_count = counts.auto;
                    console.log(`Updated match counts for idea ${id}: total=${counts.total}, manual=${counts.manual}, auto=${counts.auto}`);
                }
            });
        } else {
            // Fallback to incrementing match counts locally if not provided by the server
            currentPair.forEach(idea => {
                ideasDb[idea.id].match_count = (ideasDb[idea.id].match_count || 0) + 1;
                ideasDb[idea.id].manual_match_count = (ideasDb[idea.id].manual_match_count || 0) + 1;
            });
        }

        // Get next pair
        await refreshPair();
    } catch (error) {
        console.error('Error submitting rating:', error);
        alert('Error: ' + error.message);
    }
}

// Update the showRanking function to support different rating types
async function showRanking() {
    try {
        // Hide the idea comparison view
        document.querySelector('.ideas-container').style.display = 'none';

        // Show results section and reset
        const resultsSection = document.getElementById('autoRatingResults');
        resultsSection.style.display = 'block';
        document.getElementById('ratingProgress').style.display = 'none';

        // Get all ideas
        const ideas = Object.values(ideasDb).map((idea, index) =>
            normalizeIdeaRecord(idea, `ranking-${index}`)
        );
        ideas.forEach((idea) => {
            ideasDb[idea.id] = idea;
        });

        // Calculate total match counts for each type
        const totalMatches = ideas.reduce((sum, idea) => sum + (idea.match_count || 0), 0);
        const totalAutoMatches = ideas.reduce((sum, idea) => sum + (idea.auto_match_count || 0), 0);
        const totalManualMatches = ideas.reduce((sum, idea) => sum + (idea.manual_match_count || 0), 0);

        // Sort ideas based on rating type
        if (currentRatingType === 'auto') {
            ideas.sort((a, b) => {
                const aRating = a.ratings?.auto || a.elo || 1500;
                const bRating = b.ratings?.auto || b.elo || 1500;
                return bRating - aRating;
            });
            document.getElementById('ratingStats').innerHTML = `
                <p>Auto ratings (LLM-based)</p>
                <p>Total auto comparisons: ${totalAutoMatches / 2}</p>
            `;
        } else if (currentRatingType === 'manual') {
            ideas.sort((a, b) => {
                const aRating = a.ratings?.manual || 1500;
                const bRating = b.ratings?.manual || 1500;
                return bRating - aRating;
            });
            document.getElementById('ratingStats').innerHTML = `
                <p>Manual ratings (human-based)</p>
                <p>Total manual comparisons: ${totalManualMatches / 2}</p>
            `;
        } else if (currentRatingType === 'diff') {
            // Sort by absolute difference between auto and manual
            ideas.sort((a, b) => {
                const aDiff = Math.abs((a.ratings?.auto || 1500) - (a.ratings?.manual || 1500));
                const bDiff = Math.abs((b.ratings?.auto || 1500) - (b.ratings?.manual || 1500));
                return bDiff - aDiff;
            });
            document.getElementById('ratingStats').innerHTML = `
                <p>Difference between auto and manual ratings</p>
                <p>Total comparisons: ${totalMatches / 2} (Auto: ${totalAutoMatches / 2}, Manual: ${totalManualMatches / 2})</p>
            `;
        }

        // Update the rankings table with the sorted ideas
        updateRankingsTable(ideas);
    } catch (error) {
        console.error('Error showing ranking:', error);
    }
}

// Toggle between manual and auto rating modes
function toggleRatingMode(mode) {
    const manualControls = document.getElementById('manualRatingControls');
    const autoControls = document.getElementById('autoRatingControls');

    if (mode === 'manual') {
        manualControls.style.display = 'block';
        autoControls.style.display = 'none';
        document.getElementById('manualModeBtn').classList.add('active');
        document.getElementById('autoModeBtn').classList.remove('active');
    } else {
        manualControls.style.display = 'none';
        autoControls.style.display = 'block';
        document.getElementById('manualModeBtn').classList.remove('active');
        document.getElementById('autoModeBtn').classList.add('active');
    }
}

function showMeanElo() {
    fetch("/mean-elo-per-generation")
        .then((res) => {
            if (!res.ok) {
                throw new Error("Failed to get mean ELO: " + res.statusText);
            }
            return res.json();
        })
        .then((meanElo) => {
            const resultsContainer = document.getElementById("results-container");
            resultsContainer.innerHTML = "<h2>Mean ELO per Generation</h2>";

            const list = document.createElement("ul");
            for (const genIndex in meanElo) {
                const item = document.createElement("li");
                item.textContent = `Generation ${genIndex}: ${meanElo[genIndex].toFixed(2)}`;
                list.appendChild(item);
            }
            resultsContainer.appendChild(list);
        })
        .catch((error) => {
            console.error(error);
            alert(error.message);
        });
}

// Add event listener for keydown events
document.addEventListener("keydown", (event) => {
    switch (event.key) {
        case "ArrowLeft":
            vote("A");
            break;
        case "ArrowRight":
            vote("B");
            break;
        case " ":
            event.preventDefault(); // Prevent default spacebar scrolling
            vote("tie");
            break;
        default:
            break;
    }
});

async function loadCurrentEvolution() {
    try {
        console.log("Loading current evolution...");

        // First check if there's actually an evolution running on the server
        let shouldRestoreFromLocalStorage = false;
        try {
            const progressResponse = await fetch('/api/progress');
            if (progressResponse.ok) {
                const progressData = await progressResponse.json();
                console.log("Evolution status from server:", progressData);

                // Only restore localStorage data if evolution is actually running
                if (progressData.is_running) {
                    shouldRestoreFromLocalStorage = true;
                } else {
                    console.log("No evolution is currently running, skipping localStorage restoration");
                }
            }
        } catch (e) {
            console.log("Could not check evolution status, skipping localStorage restoration:", e);
        }

        // Check localStorage for current evolution data (only if evolution is running)
        if (shouldRestoreFromLocalStorage) {
            const storedData = localStorage.getItem('currentEvolutionData');
            if (storedData) {
                try {
                    const evolutionState = JSON.parse(storedData);
                    console.log("Loaded evolution state from localStorage:", evolutionState);

                    // Handle both old format (just history array) and new format (object with history and diversity_history)
                    let generationsData;
                    if (Array.isArray(evolutionState)) {
                        // Old format - just the history array
                        generationsData = evolutionState;
                        setCurrentEvolutionAnalytics({ history: generationsData });
                    } else if (evolutionState && evolutionState.history) {
                        // New format - object with history and diversity_history
                        generationsData = evolutionState.history;
                        setCurrentEvolutionAnalytics(evolutionState);
                    } else {
                        console.error("Invalid evolution data format");
                        // Continue to API call if localStorage parsing fails
                        throw new Error("Invalid evolution data format");
                    }

                    if (generationsData && generationsData.length > 0) {
                        const ideas = flattenGenerationHistory(generationsData, 'current-local');
                        console.log("Processed ideas from localStorage:", ideas);
                        await initializeRating(ideas, false);
                        await refreshPair(); // Ensure refreshPair is called
                        return; // Exit early if we successfully loaded from localStorage
                    }
                } catch (e) {
                    console.error("Error parsing localStorage data:", e);
                    // Continue to API call if localStorage parsing fails
                }
            }
        }

        // If localStorage doesn't have data, try the API
        const response = await fetch('/api/generations');

        if (response.ok) {
            const generations = await response.json();
            console.log("Loaded generations from API:", generations);

            if (generations && generations.length > 0) {
                setCurrentEvolutionAnalytics({ history: generations });
                const ideas = flattenGenerationHistory(generations, 'current-api');
                console.log("Processed ideas from API:", ideas);
                await initializeRating(ideas, false);

                // Refresh the pair now that we have ideas
                await refreshPair();
            } else {
                console.warn("No generations found or empty generations array");
                document.getElementById("titleA").textContent = "No ideas to rate";
                document.getElementById("contentA").textContent = "Please run an evolution first or select a saved evolution.";
                document.getElementById("titleB").textContent = "";
                document.getElementById("contentB").textContent = "";

                // Do not retry automatically to avoid overwriting user selection
                console.log("No running evolution found. Waiting for user selection.");
            }
        } else {
            console.error("Failed to load generations:", await response.text());
        }
    } catch (error) {
        console.error('Error loading current evolution:', error);
    }
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
    currentEvolutionId = evolutionId;

    // Show loading state
    document.getElementById('titleA').textContent = "Loading ideas...";
    document.getElementById('contentA').innerHTML = "<p>Please wait while we load the ideas for comparison.</p>";
    document.getElementById('titleB').textContent = "";
    document.getElementById('contentB').innerHTML = "";

    // Disable voting buttons while loading
    document.querySelectorAll('.vote-btn').forEach(btn => {
        btn.disabled = true;
    });

    try {
        if (evolutionId) {
            console.log(`Loading evolution ${evolutionId}...`);
            const response = await fetch(`/api/evolution/${evolutionId}`);
            if (response.ok) {
                const data = await response.json();
                console.log('Loaded evolution data:', data);

                if (data.data && data.data.history) {
                    setCurrentEvolutionAnalytics(data.data);
                    const ideas = flattenGenerationHistory(data.data.history, `evo-${evolutionId}`);
                    console.log('Processed ideas:', ideas);

                    await initializeRating(ideas, false);
                    // IMPORTANT: Call refreshPair after initialization
                    await refreshPair();
                } else {
                    console.error('Invalid evolution data structure:', data);
                    document.getElementById('titleA').textContent = "No ideas found";
                    document.getElementById('contentA').innerHTML = "<p>This evolution contains no ideas to rate.</p>";
                }
            } else {
                console.error('Failed to load evolution:', await response.text());
                document.getElementById('titleA').textContent = "Error loading evolution";
                document.getElementById('contentA').innerHTML = "<p>Failed to load the selected evolution. Please try again.</p>";
            }
        } else {
            // Load current evolution
            console.log('Loading current evolution...');
            await loadCurrentEvolution();
        }
    } catch (error) {
        console.error('Error in evolution selector:', error);
        document.getElementById('titleA').textContent = "Error";
        document.getElementById('contentA').innerHTML = "<p>An error occurred while loading the evolution. Please try again.</p>";
    }
});

// Add this function to load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        if (response.ok) {
            const data = await response.json();
            const modelSelect = document.getElementById('modelSelect');

            // Clear existing options
            modelSelect.innerHTML = '';

            // Add models to dropdown
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                if (model.id === data.default) {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
        } else {
            console.error('Failed to load models:', await response.text());
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Helper function to update the progress bar
function updateProgressBar(percent) {
    const roundedPercent = Math.round(percent);
    const percentStr = `${roundedPercent}%`;
    console.log(`Updating progress bar to ${percentStr}`);

    const progressBar = document.getElementById('ratingProgress');
    if (!progressBar) {
        console.error("Progress bar element not found!");
        return;
    }

    // Debug the current state
    console.log(`Before update - width: ${progressBar.style.width}, aria-valuenow: ${progressBar.getAttribute('aria-valuenow')}`);

    // Ensure the progress bar is visible
    progressBar.style.display = 'block';

    // Update the width with !important to override any conflicting styles
    progressBar.style.cssText = `width: ${percentStr} !important; transition: width 0.3s ease;`;
    progressBar.setAttribute('aria-valuenow', roundedPercent);

    // Force a reflow to ensure the browser updates the progress bar
    void progressBar.offsetHeight;

    // Debug the updated state
    console.log(`After update - width: ${progressBar.style.width}, aria-valuenow: ${progressBar.getAttribute('aria-valuenow')}`);
}

function stopAutoRatingProgressPolling() {
    if (autoRatingProgressPollTimer) {
        clearTimeout(autoRatingProgressPollTimer);
        autoRatingProgressPollTimer = null;
    }
}

function getOutcomeCounts(results) {
    const counts = { A: 0, B: 0, tie: 0 };
    (results || []).forEach((result) => {
        if (!result || typeof result.outcome !== 'string') {
            return;
        }
        if (Object.prototype.hasOwnProperty.call(counts, result.outcome)) {
            counts[result.outcome] += 1;
        }
    });
    return counts;
}

function normalizeTournamentCountValue(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return 1.0;
    }
    return Math.max(0.25, Math.round(parsed * 4) / 4);
}

function getAutoTournamentEstimate() {
    const countEl = document.getElementById('autoTournamentCount');
    const ideaCount = Object.keys(ideasDb).length;
    const fullTournamentRounds = Math.max(1, ideaCount - 1);
    const tournamentCount = normalizeTournamentCountValue(countEl ? countEl.value : 1.0);
    const targetRounds = Math.max(1, Math.round(tournamentCount * fullTournamentRounds));
    const matchesPerRound = Math.max(1, Math.floor(ideaCount / 2));
    const estimatedTotalMatches = Math.max(1, targetRounds * matchesPerRound);

    return {
        ideaCount,
        tournamentCount,
        fullTournamentRounds,
        targetRounds,
        matchesPerRound,
        estimatedTotalMatches
    };
}

function syncAutoTournamentCountUI() {
    const countEl = document.getElementById('autoTournamentCount');
    const valueEl = document.getElementById('autoTournamentCountValue');
    const hintEl = document.getElementById('autoTournamentRoundsHint');
    if (!countEl) {
        return;
    }

    const normalizedCount = normalizeTournamentCountValue(countEl.value);
    countEl.value = String(normalizedCount);

    const estimate = getAutoTournamentEstimate();
    if (valueEl) {
        valueEl.textContent = estimate.tournamentCount.toFixed(2);
    }
    if (hintEl) {
        const roundsLabel = estimate.targetRounds === 1 ? 'round' : 'rounds';
        hintEl.textContent = `${estimate.tournamentCount.toFixed(2)}x tournament = ${estimate.targetRounds} Swiss ${roundsLabel} (${estimate.fullTournamentRounds} rounds = 1 full tournament).`;
    }
}

const autoTournamentCountInput = document.getElementById('autoTournamentCount');
if (autoTournamentCountInput) {
    autoTournamentCountInput.addEventListener('input', syncAutoTournamentCountUI);
}
syncAutoTournamentCountUI();

// Start Swiss auto-rating run
const startAutoRatingBtn = document.getElementById('startAutoRating');
startAutoRatingBtn.addEventListener('click', async function () {
    if (autoRatingInProgress) {
        return; // prevent multiple simultaneous runs
    }

    const evolutionId = document.getElementById('evolutionSelect').value;
    const modelId = document.getElementById('modelSelect').value;
    const tournamentEstimate = getAutoTournamentEstimate();
    const tournamentCount = tournamentEstimate.tournamentCount;

    if (!evolutionId) {
        alert('Please select an evolution first');
        return;
    }

    if (!Number.isFinite(tournamentCount) || tournamentCount < 0.25) {
        alert('Please select a valid tournament count (minimum 0.25).');
        return;
    }

    autoRatingInProgress = true;
    startAutoRatingBtn.disabled = true;
    startAutoRatingBtn.textContent = 'Auto Swiss Rating...';

    // Hide the idea comparison view
    document.querySelector('.ideas-container').style.display = 'none';

    // Clear any previous results
    document.getElementById('results-container').innerHTML = '';

    // Show results section and reset
    const resultsSection = document.getElementById('autoRatingResults');
    resultsSection.style.display = 'block';

    console.log("Initializing progress bar for auto-rating");

    // Get the progress bar element and ensure it's properly reset
    const progressBar = document.getElementById('ratingProgress');
    if (progressBar) {
        // Reset the progress bar to 0% with a clean slate
        progressBar.style.cssText = 'width: 0% !important; transition: width 0.3s ease;';
        progressBar.setAttribute('aria-valuenow', '0');
        progressBar.style.display = 'block';

        // Force a reflow to ensure the browser updates the progress bar
        void progressBar.offsetHeight;

        console.log("Progress bar initialized to 0%");
    } else {
        console.error("Progress bar element not found during initialization!");
    }

    document.getElementById('ratingStats').innerHTML = 'Processing Swiss rounds...';
    document.getElementById('rankingsTable').innerHTML = '';

    try {
        const ratingStatsEl = document.getElementById('ratingStats');
        let expectedRounds = tournamentEstimate.targetRounds;
        let expectedTotalMatches = tournamentEstimate.estimatedTotalMatches;
        let allResults = [];
        let finalIdeas = [];
        let finalTokenCounts = null;
        let progressPollingActive = true;
        let lastProgressVersion = -1;
        const pollProgress = async () => {
            if (!progressPollingActive) {
                return;
            }
            try {
                const progressResponse = await fetch('/api/auto-rate/progress');
                if (progressResponse.ok) {
                    const progressData = await progressResponse.json();
                    if (!progressData || !progressData.is_running) {
                        return;
                    }

                    const progressVersion = Number(progressData.version ?? 0);
                    if (progressVersion === lastProgressVersion) {
                        return;
                    }
                    lastProgressVersion = progressVersion;

                    const requestedRounds = Math.max(
                        1,
                        Math.trunc(Number(progressData.requested_rounds) || expectedRounds)
                    );
                    expectedRounds = requestedRounds;
                    expectedTotalMatches = Math.max(
                        1,
                        Math.trunc(Number(progressData.total_matches) || expectedTotalMatches)
                    );

                    const completedRounds = Math.max(
                        0,
                        Math.trunc(Number(progressData.completed_rounds) || 0)
                    );
                    const completedMatches = Math.max(
                        0,
                        Math.trunc(Number(progressData.completed_matches) || 0)
                    );
                    const completedComparisons = Math.max(
                        0,
                        Math.trunc(Number(progressData.completed_comparisons) || 0)
                    );
                    const progressPercent = Number(progressData.progress);
                    const computedPercent = (completedMatches / Math.max(1, expectedTotalMatches)) * 100;
                    const safeProgress = Number.isFinite(progressPercent)
                        ? Math.min(99, Math.max(0, progressPercent))
                        : Math.min(99, Math.max(0, computedPercent));
                    updateProgressBar(safeProgress);

                    const progressIdeas = (progressData.ideas || []).map((idea, ideaIndex) =>
                        normalizeIdeaRecord(idea, `auto-live-${ideaIndex}`)
                    );
                    if (progressIdeas.length > 0) {
                        finalIdeas = progressIdeas;
                        progressIdeas.forEach((idea) => {
                            if (idea.id) {
                                ideasDb[idea.id] = normalizeIdeaRecord({
                                    ...(ideasDb[idea.id] || {}),
                                    ...idea
                                }, idea.id);
                            }
                        });
                        updateRankingsTable(progressIdeas);
                    }

                    const winCounts = progressData.win_counts || { A: 0, B: 0, tie: 0 };
                    const liveTournamentCount = normalizeTournamentCountValue(
                        progressData.tournament_count ?? tournamentCount
                    );
                    const fullRounds = Math.max(
                        1,
                        Math.trunc(Number(progressData.full_tournament_rounds) || tournamentEstimate.fullTournamentRounds)
                    );

                    if (ratingStatsEl) {
                        ratingStatsEl.innerHTML = `
                            <p>Tournament count: ${liveTournamentCount.toFixed(2)}</p>
                            <p>Swiss rounds: ${Math.min(completedRounds, requestedRounds)}/${requestedRounds}</p>
                            <p>Resolved matches: ${completedMatches}/${expectedTotalMatches}</p>
                            <p>Total comparisons completed: ${completedComparisons}</p>
                            <p>A wins: ${Number(winCounts.A) || 0}</p>
                            <p>B wins: ${Number(winCounts.B) || 0}</p>
                            <p>Ties: ${Number(winCounts.tie) || 0}</p>
                            <p class="text-muted mb-0">1 full tournament = ${fullRounds} Swiss rounds</p>
                        `;
                    }
                }
            } catch (pollError) {
                console.warn('Error polling auto-rating progress:', pollError);
            } finally {
                if (progressPollingActive) {
                    autoRatingProgressPollTimer = setTimeout(pollProgress, 350);
                }
            }
        };

        pollProgress();

        const response = await fetch('/api/auto-rate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tournamentCount: tournamentCount,
                evolutionId: evolutionId,
                modelId: modelId,
                skipSave: false
            })
        });
        progressPollingActive = false;
        stopAutoRatingProgressPolling();

        if (!response.ok) {
            let errorMessage = 'Failed to perform automated Swiss rating';
            try {
                const errorPayload = await response.json();
                if (errorPayload?.message) {
                    errorMessage = errorPayload.message;
                } else if (errorPayload?.detail) {
                    errorMessage = errorPayload.detail;
                }
            } catch (_) {
                // Keep default message when response isn't JSON
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        allResults = data.results || [];
        const completedRounds = Math.max(
            0,
            Math.trunc(Number(data.completed_rounds) || Number(data.target_tournament_rounds) || expectedRounds)
        );
        expectedRounds = Math.max(
            1,
            Math.trunc(Number(data.target_tournament_rounds) || Number(data.completed_rounds) || expectedRounds)
        );
        const resolvedMatches = Math.max(
            0,
            Math.trunc(Number(data.new_comparisons) || allResults.length)
        );
        const totalCompletedComparisons = Math.max(
            0,
            Math.trunc(Number(data.completed_comparisons) || resolvedMatches)
        );
        expectedTotalMatches = Math.max(
            1,
            Math.trunc(Number(data.target_tournament_rounds) || expectedRounds)
            * Math.max(1, Math.floor(Object.keys(ideasDb).length / 2))
        );
        finalIdeas = (data.ideas || finalIdeas).map((idea, ideaIndex) =>
            normalizeIdeaRecord(idea, `auto-${ideaIndex}`)
        );
        finalTokenCounts = data.token_counts || null;

        finalIdeas.forEach(idea => {
            if (idea.id) {
                ideasDb[idea.id] = normalizeIdeaRecord({
                    ...(ideasDb[idea.id] || {}),
                    ...idea
                }, idea.id);
            }
        });

        updateRankingsTable(finalIdeas);

        updateProgressBar(100);

        const stats = ratingStatsEl || document.getElementById('ratingStats');
        const finalOutcomeCounts = getOutcomeCounts(allResults);
        const finalTournamentCount = normalizeTournamentCountValue(data.tournament_count ?? tournamentCount);
        const finalFullRounds = Math.max(
            1,
            Math.trunc(Number(data.full_tournament_rounds) || tournamentEstimate.fullTournamentRounds)
        );
        let statsHtml = `
            <p>Tournament count: ${finalTournamentCount.toFixed(2)}</p>
            <p>Completed: ${completedRounds}/${expectedRounds} Swiss rounds</p>
            <p>Resolved matches: ${resolvedMatches}</p>
            <p>Total comparisons completed: ${totalCompletedComparisons}</p>
            <p>A wins: ${finalOutcomeCounts.A}</p>
            <p>B wins: ${finalOutcomeCounts.B}</p>
            <p>Ties: ${finalOutcomeCounts.tie}</p>
            <p class="text-muted mb-0">1 full tournament = ${finalFullRounds} Swiss rounds</p>
        `;

        if (finalTokenCounts) {
            const totalCost = finalTokenCounts.cost.total_cost.toFixed(4);
            const totalTokens = finalTokenCounts.total.toLocaleString();

            statsHtml += `
                <hr>
                <p><strong>Auto-rating Cost: $${totalCost}</strong></p>
                <p>Tokens used: ${totalTokens}</p>
                <button id="autorating-cost-details-btn" class="btn btn-sm btn-outline-primary">
                    <i class="fas fa-info-circle"></i> Cost Details
                </button>
            `;
        }

        stats.innerHTML = statsHtml;

        const costDetailsBtn = document.getElementById('autorating-cost-details-btn');
        if (costDetailsBtn && finalTokenCounts) {
            costDetailsBtn.addEventListener('click', function () {
                showAutoratingCostModal(finalTokenCounts);
            });
        }

    } catch (error) {
        console.error('Error during auto-rating:', error);
        stopAutoRatingProgressPolling();
        document.getElementById('ratingStats').innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
    } finally {
        stopAutoRatingProgressPolling();
        startAutoRatingBtn.textContent = 'Start Auto Swiss Rating';
        startAutoRatingBtn.disabled = false;
        autoRatingInProgress = false;
    }
});

// Helper function to update the rankings table
function updateRankingsTable(ideas) {
    const normalizedIdeas = (ideas || []).map((idea, index) =>
        normalizeIdeaRecord(idea, `ranked-${index}`)
    );
    decorateIdeasWithTableMetrics(normalizedIdeas);

    // Store ideas in a global variable for modal access/sorting
    window.rankedIdeas = normalizedIdeas;

    // Store original order for sorting
    if (originalIdeasOrder.length === 0) {
        originalIdeasOrder = [...normalizedIdeas];
    }

    // Set up sorting event listeners
    setupTableSorting();

    // Default to overall fitness descending
    if (!currentSortColumn) {
        currentSortColumn = 'fitness';
        currentSortDirection = 'desc';
    }
    updateSortIndicators();
    sortTable(currentSortColumn, currentSortDirection);

    // Create the ELO chart
    createEloChart(window.rankedIdeas || normalizedIdeas, currentRatingType);
}

// Table sorting functionality
function setupTableSorting() {
    const headers = document.querySelectorAll('.sortable-header');

    headers.forEach(header => {
        if (header.dataset.sortBound === 'true') {
            return;
        }
        header.dataset.sortBound = 'true';
        header.addEventListener('click', function (e) {
            e.preventDefault();

            const sortColumn = this.dataset.sort;

            // If clicking the same column, toggle direction
            if (currentSortColumn === sortColumn) {
                currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                currentSortColumn = sortColumn;
                currentSortDirection = 'asc';
            }

            // Update visual indicators
            updateSortIndicators();

            // Sort the table
            sortTable(sortColumn, currentSortDirection);
        });
    });
}

function updateSortIndicators() {
    const headers = document.querySelectorAll('.sortable-header');

    headers.forEach(header => {
        const arrow = header.querySelector('.sort-arrow');
        const column = header.dataset.sort;

        if (column === currentSortColumn) {
            arrow.className = `sort-arrow active ${currentSortDirection}`;
        } else {
            arrow.className = 'sort-arrow inactive';
        }
    });
}

function sortTable(column, direction) {
    if (!window.rankedIdeas) return;

    const ideas = [...window.rankedIdeas];

    ideas.sort((a, b) => {
        let aValue, bValue;

        switch (column) {
            case 'rank':
                // Find the original index in the rankedIdeas array
                aValue = window.rankedIdeas.indexOf(a);
                bValue = window.rankedIdeas.indexOf(b);
                break;

            case 'title':
                aValue = (a.title || 'Untitled').toLowerCase();
                bValue = (b.title || 'Untitled').toLowerCase();
                break;

            case 'rating':
                aValue = getRatingValueByType(a, currentRatingType);
                bValue = getRatingValueByType(b, currentRatingType);
                if (currentRatingType === 'diff') {
                    aValue = Math.abs(aValue);
                    bValue = Math.abs(bValue);
                }
                break;

            case 'matches':
                if (currentRatingType === 'auto') {
                    aValue = a.auto_match_count || 0;
                    bValue = b.auto_match_count || 0;
                } else if (currentRatingType === 'manual') {
                    aValue = a.manual_match_count || 0;
                    bValue = b.manual_match_count || 0;
                } else {
                    aValue = a.match_count || 0;
                    bValue = b.match_count || 0;
                }
                break;

            case 'diversity':
                aValue = getIdeaTableDiversityScore(a);
                bValue = getIdeaTableDiversityScore(b);
                break;

            case 'fitness':
                aValue = getIdeaTableFitnessScore(a);
                bValue = getIdeaTableFitnessScore(b);
                break;

            case 'birth_generation':
                aValue = getIdeaGenerationValue(a, 'birth');
                bValue = getIdeaGenerationValue(b, 'birth');
                aValue = aValue !== null ? aValue : -1;
                bValue = bValue !== null ? bValue : -1;
                break;

            case 'death_generation':
                aValue = getIdeaGenerationValue(a, 'death');
                bValue = getIdeaGenerationValue(b, 'death');
                aValue = aValue !== null ? aValue : -1;
                bValue = bValue !== null ? bValue : -1;
                break;

            case 'generation':
                // Backward-compatible alias
                aValue = getIdeaGenerationValue(a, 'death');
                bValue = getIdeaGenerationValue(b, 'death');
                aValue = aValue !== null ? aValue : -1;
                bValue = bValue !== null ? bValue : -1;
                break;

            default:
                return 0;
        }

        // Handle string vs numeric comparison
        if (typeof aValue === 'string' && typeof bValue === 'string') {
            return direction === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        }

        // Keep missing numeric values at the bottom regardless of direction
        const aFinite = Number.isFinite(aValue);
        const bFinite = Number.isFinite(bValue);
        if (!aFinite && !bFinite) {
            return 0;
        }
        if (!aFinite) {
            return 1;
        }
        if (!bFinite) {
            return -1;
        }

        return direction === 'asc' ? aValue - bValue : bValue - aValue;
    });

    // Update the table with sorted ideas
    window.rankedIdeas = ideas;
    updateRankingsTableContent(ideas);
}

function updateRankingsTableContent(ideas) {
    const normalizedIdeas = (ideas || []).map((idea, index) =>
        normalizeIdeaRecord(idea, `ranked-sort-${index}`)
    );
    decorateIdeasWithTableMetrics(normalizedIdeas);

    const rankingsTable = document.getElementById('rankingsTable');
    rankingsTable.innerHTML = ''; // Clear existing content

    normalizedIdeas.forEach((idea, index) => {
        const row = document.createElement('tr');
        row.className = 'idea-row';
        row.dataset.ideaIndex = index;
        row.style.cursor = 'pointer';

        row.innerHTML = buildRankingRowHtml(idea, index);

        // Add click event to show the idea details
        row.addEventListener('click', function () {
            showIdeaDetails(index);
        });

        rankingsTable.appendChild(row);
    });
}

function getIdeaRawDiversityScore(idea, diversityLookup = null) {
    const directCandidates = [
        idea?.diversity_score,
        idea?.diversity,
        idea?.selection_metrics?.diversity,
        idea?.survival_metrics?.diversity
    ];
    for (const candidate of directCandidates) {
        const parsed = normalizeToFiniteNumber(candidate);
        if (parsed !== null) {
            return parsed;
        }
    }

    const generation = getIdeaGenerationValue(idea, 'death')
        ?? getIdeaGenerationValue(idea, 'birth')
        ?? normalizeGenerationValue(idea?.generation);
    if (generation === null) {
        return null;
    }

    const lookup = diversityLookup || buildGenerationDiversityLookup(currentEvolutionAnalytics?.diversity_history ?? []);
    if (Object.prototype.hasOwnProperty.call(lookup.zeroBasedLookup, generation)) {
        return normalizeToFiniteNumber(lookup.zeroBasedLookup[generation]);
    }
    if (Object.prototype.hasOwnProperty.call(lookup.oneBasedLookup, generation)) {
        return normalizeToFiniteNumber(lookup.oneBasedLookup[generation]);
    }

    return null;
}

function getExplicitIdeaFitnessScore(idea) {
    const directCandidates = [
        idea?.overall_fitness,
        idea?.fitness,
        idea?.fitness_score,
        idea?.selection_metrics?.fitness,
        idea?.selection_metrics?.overall_fitness,
        idea?.survival_score,
        idea?.survival_metrics?.survival_score
    ];
    for (const candidate of directCandidates) {
        const parsed = normalizeToFiniteNumber(candidate);
        if (parsed !== null) {
            return parsed;
        }
    }
    return null;
}

function getIdeaTableDiversityScore(idea) {
    return normalizeToFiniteNumber(idea?.table_diversity_score)
        ?? getIdeaRawDiversityScore(idea);
}

function getIdeaTableFitnessScore(idea) {
    return getExplicitIdeaFitnessScore(idea)
        ?? normalizeToFiniteNumber(idea?.table_overall_fitness);
}

function decorateIdeasWithTableMetrics(ideas) {
    const rankedIdeas = Array.isArray(ideas) ? ideas : [];
    if (rankedIdeas.length === 0) {
        return rankedIdeas;
    }

    const diversityLookup = buildGenerationDiversityLookup(currentEvolutionAnalytics?.diversity_history ?? []);
    const diversityValues = rankedIdeas.map((idea) => getIdeaRawDiversityScore(idea, diversityLookup));
    const autoRatings = rankedIdeas.map((idea) =>
        normalizeToFiniteNumber(idea?.ratings?.auto) ?? normalizeToFiniteNumber(idea?.elo) ?? 1500
    );

    const normalizedRatings = normalizeSeriesMinMax(autoRatings);
    const normalizedDiversity = normalizeSeriesMinMax(diversityValues);
    const fitnessAlpha = clampUnitInterval(currentEvolutionAnalytics?.fitness_alpha, 0.7);

    rankedIdeas.forEach((idea, index) => {
        const explicitFitness = getExplicitIdeaFitnessScore(idea);
        const ratingComponent = normalizedRatings[index];
        const diversityComponent = normalizedDiversity[index];

        let fallbackFitness = null;
        if (Number.isFinite(ratingComponent) && Number.isFinite(diversityComponent)) {
            fallbackFitness = (fitnessAlpha * ratingComponent) + ((1 - fitnessAlpha) * diversityComponent);
        } else if (Number.isFinite(ratingComponent)) {
            fallbackFitness = ratingComponent;
        } else if (Number.isFinite(diversityComponent)) {
            fallbackFitness = diversityComponent;
        }

        idea.table_diversity_score = diversityValues[index];
        idea.table_overall_fitness = explicitFitness !== null ? explicitFitness : fallbackFitness;
    });

    return rankedIdeas;
}

function formatRankingMetric(value, decimals = 4) {
    const parsed = normalizeToFiniteNumber(value);
    if (parsed === null) {
        return '—';
    }
    return parsed.toFixed(decimals);
}

function getIdeaRatingCellHtml(idea) {
    const autoRating = normalizeToFiniteNumber(idea?.ratings?.auto) ?? normalizeToFiniteNumber(idea?.elo) ?? 1500;
    const manualRating = normalizeToFiniteNumber(idea?.ratings?.manual) ?? 1500;
    const diffRating = autoRating - manualRating;

    if (currentRatingType === 'manual') {
        return `<td>${Math.round(manualRating)}</td>`;
    }
    if (currentRatingType === 'diff') {
        const diffClass = diffRating > 0 ? 'text-success' : (diffRating < 0 ? 'text-danger' : '');
        const diffSign = diffRating > 0 ? '+' : '';
        return `<td class="${diffClass}">${diffSign}${Math.round(diffRating)} (A:${Math.round(autoRating)}/M:${Math.round(manualRating)})</td>`;
    }
    return `<td>${Math.round(autoRating)}</td>`;
}

function getIdeaMatchCountByType(idea) {
    if (currentRatingType === 'auto') {
        return idea?.auto_match_count || 0;
    }
    if (currentRatingType === 'manual') {
        return idea?.manual_match_count || 0;
    }
    return idea?.match_count || 0;
}

function buildRankingRowHtml(idea, index) {
    const diversityScore = getIdeaTableDiversityScore(idea);
    const overallFitness = getIdeaTableFitnessScore(idea);
    const birthGeneration = getIdeaGenerationLabel(idea, 'birth');
    const deathGeneration = getIdeaGenerationLabel(idea, 'death');

    return `
        <td>${index + 1}</td>
        <td>${idea.title || 'Untitled'}</td>
        ${getIdeaRatingCellHtml(idea)}
        <td>${getIdeaMatchCountByType(idea)}</td>
        <td>${formatRankingMetric(diversityScore, 4)}</td>
        <td>${formatRankingMetric(overallFitness, 4)}</td>
        <td>${birthGeneration}</td>
        <td>${deathGeneration}</td>
    `;
}

// Add this function to show idea details in a modal
function showIdeaDetails(ideaIndex) {
    if (!window.rankedIdeas || !window.rankedIdeas[ideaIndex]) {
        console.error('Idea not found');
        return;
    }

    const idea = window.rankedIdeas[ideaIndex];
    currentModalIdea = idea;

    // Set modal title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Set modal content with markdown rendering
    const modalContent = document.getElementById('modalIdeaContent');
    modalContent.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h4>${idea.title || 'Untitled'}</h4>
                <p><strong>Generation (${currentGenerationView === 'birth' ? 'Birth' : 'Death'} view):</strong> ${getIdeaGenerationLabel(idea)}</p>
                <p><strong>Birth Generation:</strong> ${getIdeaGenerationLabel(idea, 'birth')}</p>
                <p><strong>Death Generation:</strong> ${getIdeaGenerationLabel(idea, 'death')}</p>
                <p><strong>ELO Rating:</strong> ${idea.elo}</p>
                <hr>
                <div class="idea-content">
                    ${renderMarkdown(idea.content || '')}
                </div>
            </div>
        </div>
    `;

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('ideaModal'));
    modal.show();
}

// Add event listener for the return button when the document loads
document.addEventListener('DOMContentLoaded', function () {
    // Connect the return button to the showComparisonView function
    const returnButton = document.getElementById('returnToComparison');
    if (returnButton) {
        returnButton.addEventListener('click', showComparisonView);
    }

    const copyButton = document.getElementById('copyIdeaButton');
    if (copyButton) {
        copyButton.addEventListener('click', () => {
            if (!currentModalIdea) return;
            const text = `${currentModalIdea.title || ''}\n\n${currentModalIdea.content || ''}`;
            navigator.clipboard.writeText(text).catch(err => console.error('Copy failed', err));
        });
    }

    initializeChartFocusInteractions();
});

// Make sure the showComparisonView function is properly defined
function showComparisonView() {
    console.log('Showing comparison view');
    document.querySelector('.ideas-container').style.display = 'flex';
    document.getElementById('autoRatingResults').style.display = 'none';
    document.getElementById('results-container').innerHTML = '';

    // Refresh the pair to show new ideas
    refreshPair();
}

// Add function to toggle between rating types
function toggleRatingType(type) {
    currentRatingType = type;

    // Reset sorting state when switching rating types
    currentSortColumn = 'fitness';
    currentSortDirection = 'desc';

    // Update button states
    document.getElementById('autoRatingTypeBtn').classList.remove('active');
    document.getElementById('manualRatingTypeBtn').classList.remove('active');
    document.getElementById('diffRatingTypeBtn').classList.remove('active');

    document.getElementById(`${type}RatingTypeBtn`).classList.add('active');

    // If we're viewing rankings, refresh them with the new type
    if (document.getElementById('autoRatingResults').style.display === 'block') {
        // Update the rankings table without reloading everything
        if (window.rankedIdeas) {
            console.log(`Refreshing rankings table with rating type: ${type}`);
            // Reset to original order first
            originalIdeasOrder = [];
            showRanking();
        } else {
            showRanking();
        }
    }
}

function toggleGenerationView(view) {
    currentGenerationView = view === 'birth' ? 'birth' : 'death';

    const birthBtn = document.getElementById('birthGenerationViewBtn');
    const deathBtn = document.getElementById('deathGenerationViewBtn');
    if (birthBtn && deathBtn) {
        birthBtn.classList.remove('active');
        deathBtn.classList.remove('active');
        document.getElementById(`${currentGenerationView}GenerationViewBtn`).classList.add('active');
    }

    if (document.getElementById('autoRatingResults').style.display === 'block') {
        showRanking();
    }
}

// Function to show autorating cost details modal
function showAutoratingCostModal(tokenCounts) {
    // Format the token counts
    const totalTokens = tokenCounts.total.toLocaleString();
    const totalInputTokens = tokenCounts.total_input.toLocaleString();
    const totalOutputTokens = tokenCounts.total_output.toLocaleString();

    // Component totals and details for critic
    const criticTotal = tokenCounts.critic.total.toLocaleString();
    const criticInput = tokenCounts.critic.input.toLocaleString();
    const criticOutput = tokenCounts.critic.output.toLocaleString();
    const criticModel = tokenCounts.critic.model;
    const criticCost = tokenCounts.critic.cost.toFixed(4);

    // Get cost information
    const costInfo = tokenCounts.cost;
    const totalCost = costInfo.total_cost.toFixed(4);
    const inputCost = costInfo.input_cost.toFixed(4);
    const outputCost = costInfo.output_cost.toFixed(4);

    // Prepare alternative cost estimates using the critic token counts
    let estimatesList = '';
    if (tokenCounts.estimates) {
        estimatesList = Object.values(tokenCounts.estimates).map(e => {
            return `<li class="list-group-item d-flex justify-content-between align-items-center">${e.name}<span>$${e.cost.toFixed(4)} <small class="text-muted">estimate</small></span></li>`;
        }).join('');
    }

    // Create modal container if it doesn't exist
    let modalContainer = document.getElementById('autorating-cost-modal');
    if (!modalContainer) {
        modalContainer = document.createElement('div');
        modalContainer.id = 'autorating-cost-modal';
        modalContainer.className = 'modal fade';
        modalContainer.tabIndex = '-1';
        modalContainer.setAttribute('aria-labelledby', 'autoratingCostModalLabel');
        modalContainer.setAttribute('aria-hidden', 'true');
        document.body.appendChild(modalContainer);
    }

    // Set modal content
    modalContainer.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="autoratingCostModalLabel">Auto-rating Cost Details</h5>
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
                                <li class="list-group-item">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span>Critic <span class="badge bg-secondary">${criticModel}</span></span>
                                        <span class="badge bg-primary rounded-pill">${criticTotal}</span>
                                    </div>
                                    <div class="small text-muted mt-1">
                                        <span>Input: ${criticInput}</span> |
                                        <span>Output: ${criticOutput}</span> |
                                        <span>Cost: $${criticCost}</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-4">
                                <canvas id="autoratingTokenPieChart" width="400" height="250"></canvas>
                            </div>
                            <div>
                                <canvas id="autoratingTokenBarChart" width="400" height="250"></canvas>
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
        // For autorating, we only have the critic component, so create simpler charts

        // Create a pie chart for input vs output tokens
        const pieCtx = document.getElementById('autoratingTokenPieChart').getContext('2d');
        new Chart(pieCtx, {
            type: 'pie',
            data: {
                labels: ['Input Tokens', 'Output Tokens'],
                datasets: [{
                    data: [
                        tokenCounts.total_input,
                        tokenCounts.total_output
                    ],
                    backgroundColor: [
                        '#36a2eb',
                        '#4bc0c0'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Token Distribution (Input vs Output)'
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // Create a bar chart for costs
        const barCtx = document.getElementById('autoratingTokenBarChart').getContext('2d');
        new Chart(barCtx, {
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
                    backgroundColor: ['#36a2eb', '#4bc0c0', '#ff9f40']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cost Breakdown'
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

// Add function to reset ratings
async function resetRatings(type) {
    const evolutionId = document.getElementById('evolutionSelect').value;

    if (!evolutionId) {
        alert('Please select an evolution first');
        return;
    }

    if (!confirm(`Are you sure you want to reset ${type} ratings to their default value (1500) and clear all match counts?`)) {
        return;
    }

    try {
        const response = await fetch('/api/reset-ratings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                evolutionId: evolutionId,
                ratingType: type
            })
        });

        if (!response.ok) {
            throw new Error('Failed to reset ratings and match counts');
        }

        const data = await response.json();

        // Reload the current evolution
        await loadEvolution(evolutionId);

        // Show a success message
        alert(data.message);

        // If we're viewing rankings, refresh them
        if (document.getElementById('autoRatingResults').style.display === 'block') {
            showRanking();
        }
    } catch (error) {
        console.error('Error resetting ratings and match counts:', error);
        alert('Error: ' + error.message);
    }
}

// Update the showIdeaDetails function to show both rating types
function showIdeaDetails(ideaIndex) {
    if (!window.rankedIdeas || !window.rankedIdeas[ideaIndex]) {
        console.error('Idea not found');
        return;
    }

    const idea = window.rankedIdeas[ideaIndex];
    currentModalIdea = idea;

    // Get the rating values
    const autoRating = idea.ratings?.auto || idea.elo || 1500;
    const manualRating = idea.ratings?.manual || 1500;
    const diffRating = autoRating - manualRating;
    const diffClass = diffRating > 0 ? 'text-success' : (diffRating < 0 ? 'text-danger' : '');
    const diffSign = diffRating > 0 ? '+' : '';

    // Set modal title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Set modal content with markdown rendering
    const modalContent = document.getElementById('modalIdeaContent');
    modalContent.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h4>${idea.title || 'Untitled'}</h4>
                <p><strong>Generation (${currentGenerationView === 'birth' ? 'Birth' : 'Death'} view):</strong> ${getIdeaGenerationLabel(idea)}</p>
                <p><strong>Birth Generation:</strong> ${getIdeaGenerationLabel(idea, 'birth')}</p>
                <p><strong>Death Generation:</strong> ${getIdeaGenerationLabel(idea, 'death')}</p>
                <p><strong>Auto Rating:</strong> ${autoRating}</p>
                <p><strong>Manual Rating:</strong> ${manualRating}</p>
                <p><strong>Difference:</strong> <span class="${diffClass}">${diffSign}${diffRating}</span></p>
                <hr>
                <div class="idea-content">
                    ${renderMarkdown(idea.content || '')}
                </div>
            </div>
        </div>
    `;

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('ideaModal'));
    modal.show();
}

async function loadEvolution(evolutionId) {
    try {
        const response = await fetch(`/api/evolution/${evolutionId}`);
        if (!response.ok) {
            throw new Error('Failed to load evolution');
        }

        const data = await response.json();
        currentEvolutionId = evolutionId;
        setCurrentEvolutionAnalytics(data.data);

        const allIdeas = flattenGenerationHistory(data.data.history, `load-${evolutionId}`);

        allIdeas.forEach(idea => {
            if (!idea.ratings) {
                idea.ratings = {
                    auto: idea.elo || 1500,
                    manual: 1500
                };
            } else if (typeof idea.ratings === 'number') {
                const oldElo = idea.ratings;
                idea.ratings = {
                    auto: oldElo,
                    manual: oldElo
                };
            }

            idea.elo = idea.ratings.auto;
            idea.match_count = idea.match_count || 0;
            idea.auto_match_count = idea.auto_match_count || 0;
            idea.manual_match_count = idea.manual_match_count || 0;
        });

        await initializeRating(allIdeas);

        return data;
    } catch (error) {
        console.error('Error loading evolution:', error);
        return null;
    }
}

// Helper function to format generation labels consistently
function formatGenerationLabel(generation) {
    if (generation === 0 || generation === '0') {
        return 'Initial Population';
    }
    return `Gen ${generation}`;
}

// Keep the original random pair function as a fallback
function getRandomPair(ideas) {
    if (ideas.length < 2) {
        throw new Error('Not enough ideas to form a pair');
    }

    // Get first random idea
    const index1 = Math.floor(Math.random() * ideas.length);
    const idea1 = ideas[index1];

    // Get second random idea (different from the first)
    let index2;
    do {
        index2 = Math.floor(Math.random() * ideas.length);
    } while (index2 === index1);

    const idea2 = ideas[index2];

    return [idea1, idea2];
}

function getRatingValueByType(idea, ratingType) {
    const autoRating = normalizeToFiniteNumber(idea?.ratings?.auto) ?? normalizeToFiniteNumber(idea?.elo) ?? 1500;
    const manualRating = normalizeToFiniteNumber(idea?.ratings?.manual) ?? 1500;

    if (ratingType === 'manual') {
        return manualRating;
    }
    if (ratingType === 'diff') {
        return autoRating - manualRating;
    }
    return autoRating;
}

function normalizeSeriesMinMax(values) {
    const finiteValues = values.filter(value => Number.isFinite(value));
    if (finiteValues.length === 0) {
        return values.map(() => null);
    }

    const minValue = Math.min(...finiteValues);
    const maxValue = Math.max(...finiteValues);

    if (Math.abs(maxValue - minValue) < 1e-12) {
        return values.map(value => (Number.isFinite(value) ? 0.5 : null));
    }

    return values.map(value => {
        if (!Number.isFinite(value)) {
            return null;
        }
        return (value - minValue) / (maxValue - minValue);
    });
}

function buildGenerationDiversityLookup(diversityHistory) {
    const zeroBasedLookup = {};
    const oneBasedLookup = {};

    if (!Array.isArray(diversityHistory)) {
        return { zeroBasedLookup, oneBasedLookup };
    }

    diversityHistory.forEach((snapshot, snapshotIndex) => {
        if (!snapshot || typeof snapshot !== 'object') {
            return;
        }

        if (Array.isArray(snapshot.generation_diversities)) {
            snapshot.generation_diversities.forEach((genData) => {
                const generation = normalizeGenerationValue(genData?.generation);
                const score = normalizeToFiniteNumber(genData?.diversity_score);
                if (generation === null || score === null) {
                    return;
                }
                zeroBasedLookup[generation] = score;
                oneBasedLookup[generation + 1] = score;
            });
        }

        const inferredGeneration = normalizeGenerationValue(snapshot.generation) ?? snapshotIndex;
        const inferredScore = normalizeToFiniteNumber(snapshot.current_generation_diversity)
            ?? normalizeToFiniteNumber(snapshot.diversity_score);
        if (inferredScore !== null) {
            zeroBasedLookup[inferredGeneration] = inferredScore;
            oneBasedLookup[inferredGeneration + 1] = inferredScore;
        }
    });

    return { zeroBasedLookup, oneBasedLookup };
}

function getGenerationDiversitySeries(generations) {
    const diversityHistory = currentEvolutionAnalytics?.diversity_history ?? [];
    const { zeroBasedLookup, oneBasedLookup } = buildGenerationDiversityLookup(diversityHistory);
    return generations.map((generation) => {
        const normalizedGeneration = normalizeGenerationValue(generation);
        if (normalizedGeneration === null) {
            return null;
        }
        if (Object.prototype.hasOwnProperty.call(zeroBasedLookup, normalizedGeneration)) {
            return zeroBasedLookup[normalizedGeneration];
        }
        if (Object.prototype.hasOwnProperty.call(oneBasedLookup, normalizedGeneration)) {
            return oneBasedLookup[normalizedGeneration];
        }
        return null;
    });
}

function cloneConfigValue(value) {
    if (Array.isArray(value)) {
        return value.map(cloneConfigValue);
    }
    if (value && typeof value === 'object') {
        const cloned = {};
        Object.keys(value).forEach((key) => {
            cloned[key] = cloneConfigValue(value[key]);
        });
        return cloned;
    }
    return value;
}

function destroyFocusedChart() {
    if (typeof Chart === 'undefined') {
        focusedChartInstance = null;
        return;
    }
    if (focusedChartInstance instanceof Chart) {
        focusedChartInstance.destroy();
        focusedChartInstance = null;
    }
}

function renderFocusedChart() {
    if (typeof Chart === 'undefined') {
        return;
    }
    if (!pendingFocusedChartConfig) {
        return;
    }

    const canvas = document.getElementById('chartFocusCanvas');
    if (!canvas) {
        return;
    }

    destroyFocusedChart();

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        return;
    }

    focusedChartInstance = new Chart(ctx, pendingFocusedChartConfig);
    pendingFocusedChartConfig = null;
}

function openFocusedChart(chartId, chartTitle) {
    if (typeof Chart === 'undefined') {
        return;
    }
    const chartGetter = chartFocusLookup[chartId];
    const sourceChart = typeof chartGetter === 'function' ? chartGetter() : null;
    if (!(sourceChart instanceof Chart)) {
        return;
    }

    const modalElement = document.getElementById('chartFocusModal');
    const modalTitle = document.getElementById('chartFocusModalLabel');
    if (!modalElement) {
        return;
    }

    const focusedOptions = cloneConfigValue(sourceChart.options || {});
    focusedOptions.responsive = true;
    focusedOptions.maintainAspectRatio = false;

    pendingFocusedChartConfig = {
        type: sourceChart.config.type,
        data: cloneConfigValue(sourceChart.data),
        options: focusedOptions
    };

    if (modalTitle) {
        modalTitle.textContent = chartTitle || 'Chart Focus';
    }

    if (!chartFocusModalInstance) {
        chartFocusModalInstance = new bootstrap.Modal(modalElement);
        modalElement.addEventListener('shown.bs.modal', renderFocusedChart);
        modalElement.addEventListener('hidden.bs.modal', () => {
            pendingFocusedChartConfig = null;
            destroyFocusedChart();
        });
    }

    chartFocusModalInstance.show();
    if (modalElement.classList.contains('show')) {
        renderFocusedChart();
    }
}

function initializeChartFocusInteractions() {
    const triggers = document.querySelectorAll('.chart-focus-trigger[data-focus-chart]');
    triggers.forEach((trigger) => {
        if (trigger.dataset.chartFocusReady === 'true') {
            return;
        }
        trigger.dataset.chartFocusReady = 'true';

        const openChart = () => {
            const chartId = trigger.dataset.focusChart;
            const chartTitle = trigger.dataset.chartTitle;
            openFocusedChart(chartId, chartTitle);
        };

        trigger.addEventListener('click', (event) => {
            event.preventDefault();
            openChart();
        });

        trigger.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') {
                return;
            }
            event.preventDefault();
            openChart();
        });
    });
}

function createDiversityChart(labels, diversitySeries) {
    const canvas = document.getElementById('diversityChart');
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext('2d');
    if (window.diversityChart instanceof Chart) {
        window.diversityChart.destroy();
    }

    const hasDiversityData = diversitySeries.some(value => Number.isFinite(value));
    window.diversityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Diversity Score',
                    data: diversitySeries,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: hasDiversityData ? 'Diversity by Generation' : 'Diversity by Generation (not available)'
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            if (!Number.isFinite(context.raw)) {
                                return `${context.dataset.label}: n/a`;
                            }
                            return `${context.dataset.label}: ${context.raw.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Diversity Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                }
            }
        }
    });
}

function createFitnessChart(labels, combinedFitnessSeries, normalizedEloSeries, normalizedDiversitySeries, fitnessAlpha) {
    const canvas = document.getElementById('fitnessChart');
    if (!canvas) {
        return;
    }

    const formulaEl = document.getElementById('fitnessFormula');
    if (formulaEl) {
        formulaEl.textContent = `fitness = ${fitnessAlpha.toFixed(2)} × norm(mean ELO) + ${(1 - fitnessAlpha).toFixed(2)} × norm(diversity)`;
    }

    const ctx = canvas.getContext('2d');
    if (window.fitnessChart instanceof Chart) {
        window.fitnessChart.destroy();
    }

    const hasFitnessData = combinedFitnessSeries.some(value => Number.isFinite(value));
    window.fitnessChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Combined Fitness',
                    data: combinedFitnessSeries,
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    tension: 0.2
                },
                {
                    label: 'Normalized Mean ELO',
                    data: normalizedEloSeries,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderDash: [6, 4],
                    tension: 0.2
                },
                {
                    label: 'Normalized Diversity',
                    data: normalizedDiversitySeries,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderDash: [6, 4],
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: hasFitnessData ? 'Combined Fitness by Generation' : 'Combined Fitness by Generation (insufficient data)'
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            if (!Number.isFinite(context.raw)) {
                                return `${context.dataset.label}: n/a`;
                            }
                            return `${context.dataset.label}: ${context.raw.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Normalized Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                }
            }
        }
    });
}

// Fix the createEloChart function to handle the case when there's no existing chart
function createEloChart(ideas, ratingType) {
    const generationGroups = {};

    ideas.forEach(idea => {
        const generation = getIdeaGenerationValue(idea, currentGenerationView);
        const normalizedGeneration = generation !== null ? generation : 0;
        if (!generationGroups[normalizedGeneration]) {
            generationGroups[normalizedGeneration] = [];
        }
        generationGroups[normalizedGeneration].push(idea);
    });

    const generations = Object.keys(generationGroups)
        .map(generation => Number(generation))
        .filter(generation => Number.isFinite(generation))
        .sort((a, b) => a - b);

    const maxElos = [];
    const meanElos = [];
    const medianElos = [];
    const meanAutoElos = [];

    generations.forEach(generation => {
        const genIdeas = generationGroups[generation] || [];
        const ratings = genIdeas
            .map(idea => getRatingValueByType(idea, ratingType))
            .filter(value => Number.isFinite(value));

        if (ratings.length === 0) {
            maxElos.push(null);
            meanElos.push(null);
            medianElos.push(null);
        } else {
            const maxElo = Math.max(...ratings);
            const meanElo = ratings.reduce((sum, elo) => sum + elo, 0) / ratings.length;
            const sortedRatings = [...ratings].sort((a, b) => a - b);
            const medianElo = sortedRatings.length % 2 === 0
                ? (sortedRatings[sortedRatings.length / 2 - 1] + sortedRatings[sortedRatings.length / 2]) / 2
                : sortedRatings[Math.floor(sortedRatings.length / 2)];
            maxElos.push(maxElo);
            meanElos.push(meanElo);
            medianElos.push(medianElo);
        }

        const autoRatings = genIdeas
            .map(idea => getRatingValueByType(idea, 'auto'))
            .filter(value => Number.isFinite(value));
        if (autoRatings.length === 0) {
            meanAutoElos.push(null);
        } else {
            meanAutoElos.push(autoRatings.reduce((sum, elo) => sum + elo, 0) / autoRatings.length);
        }
    });

    const generationLabels = generations.map(generation => formatGenerationLabel(generation));
    const ctx = document.getElementById('eloChart').getContext('2d');

    if (window.eloChart instanceof Chart) {
        window.eloChart.destroy();
    }

    window.eloChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: generationLabels,
            datasets: [
                {
                    label: 'Max ELO',
                    data: maxElos,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Mean ELO',
                    data: meanElos,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Median ELO',
                    data: medianElos,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${ratingType.charAt(0).toUpperCase() + ratingType.slice(1)} Ratings by Generation`
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            if (!Number.isFinite(context.raw)) {
                                return `${context.dataset.label}: n/a`;
                            }
                            return `${context.dataset.label}: ${Math.round(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'ELO Rating'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                }
            }
        }
    });

    const analyticsHistory = Array.isArray(currentEvolutionAnalytics?.history)
        ? currentEvolutionAnalytics.history
        : [];
    let fitnessGenerations = generations;
    let meanAutoFitnessSeries = meanAutoElos;

    if (analyticsHistory.length > 0) {
        fitnessGenerations = analyticsHistory.map((_, index) => index);
        meanAutoFitnessSeries = analyticsHistory.map((generation) => {
            const autoRatings = (generation || [])
                .map(idea => getRatingValueByType(idea, 'auto'))
                .filter(value => Number.isFinite(value));
            if (autoRatings.length === 0) {
                return null;
            }
            return autoRatings.reduce((sum, elo) => sum + elo, 0) / autoRatings.length;
        });
    }

    const fitnessLabels = fitnessGenerations.map(generation => formatGenerationLabel(generation));
    const diversitySeries = getGenerationDiversitySeries(fitnessGenerations);
    createDiversityChart(fitnessLabels, diversitySeries);

    const normalizedEloSeries = normalizeSeriesMinMax(meanAutoFitnessSeries);
    const normalizedDiversitySeries = normalizeSeriesMinMax(diversitySeries);
    const fitnessAlpha = clampUnitInterval(currentEvolutionAnalytics?.fitness_alpha, 0.7);
    const combinedFitnessSeries = fitnessGenerations.map((_, index) => {
        const eloComponent = normalizedEloSeries[index];
        const diversityComponent = normalizedDiversitySeries[index];
        if (!Number.isFinite(eloComponent) || !Number.isFinite(diversityComponent)) {
            return null;
        }
        return (fitnessAlpha * eloComponent) + ((1 - fitnessAlpha) * diversityComponent);
    });

    createFitnessChart(
        fitnessLabels,
        combinedFitnessSeries,
        normalizedEloSeries,
        normalizedDiversitySeries,
        fitnessAlpha
    );
}
