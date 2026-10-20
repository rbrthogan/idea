function getRaterChartTheme() {
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

function raterChartColorWithAlpha(color, alpha) {
    const hex = /^#([a-f\d]{6})$/i.exec(color);
    if (hex) {
        const channels = [0, 2, 4].map(offset => parseInt(hex[1].slice(offset, offset + 2), 16));
        return `rgba(${channels.join(', ')}, ${alpha})`;
    }
    const rgb = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(color);
    return rgb ? `rgba(${rgb.slice(1).join(', ')}, ${alpha})` : color;
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

let currentPair = null;
let currentEvolutionId = null;
let ratingSelectionVersion = 0;
let pairRequestVersion = 0;
let voteRequestVersion = 0;
let evolutionListRequestVersion = 0;
let modelRequestVersion = 0;
let autoRatingRequestVersion = 0;
let pairContext = null;
let pairLoading = false;
let ratingSelectionLoading = false;

function captureRatingContext() {
    return { owner: window.currentUser?.uid, selection: ratingSelectionVersion, evolutionId: currentEvolutionId };
}

function isRatingContextCurrent(context) {
    return Boolean(context.owner)
        && context.owner === window.currentUser?.uid
        && context.selection === ratingSelectionVersion
        && context.evolutionId === currentEvolutionId;
}

function updateManualRatingControls() {
    const ready = currentPair && pairContext && isRatingContextCurrent(pairContext)
        && !pairLoading && !ratingSelectionLoading && !voteInFlight;
    document.querySelectorAll('.vote-btn').forEach(button => { button.disabled = !ready; });
    const retry = document.getElementById('retryManualPair');
    if (retry) retry.disabled = pairLoading || ratingSelectionLoading || voteInFlight;
}

function beginRatingSelection(evolutionId) {
    ratingSelectionVersion += 1;
    pairRequestVersion += 1;
    voteRequestVersion += 1;
    currentEvolutionId = evolutionId || null;
    currentPair = null;
    pairContext = null;
    pairLoading = false;
    ratingSelectionLoading = true;
    voteInFlight = false;
    ideasDb = {};
    currentModalIdea = null;
    window.rankedIdeas = [];
    window.IdeaWorkbench?.closeInspector?.();
    document.getElementById('manualRatingFeedback').hidden = true;
    document.getElementById('titleA').textContent = 'Loading ideas…';
    document.getElementById('contentA').textContent = 'Loading this experiment for comparison.';
    document.getElementById('titleB').textContent = '';
    document.getElementById('contentB').textContent = '';
    ['artifactA', 'artifactB'].forEach(id => { document.getElementById(id)?.replaceChildren(); });
    document.querySelector('.ideas-container').style.display = 'flex';
    document.getElementById('autoRatingResults').style.display = 'none';
    setManualSwissStatus(null);
    updateManualRatingControls();
    return captureRatingContext();
}
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

// Authentication may finish well after page load; do not time out into anonymous reads.
let initializedRatingOwner = null;
function initializeAuthenticatedRating() {
    const owner = window.currentUser?.uid;
    if (!owner || initializedRatingOwner === owner) return;
    initializedRatingOwner = owner;
    document.getElementById('evolutionSelect').innerHTML = '<option value="">Current Evolution</option>';
    void loadEvolutions();
    void loadCurrentEvolution();
    void loadModels();
}
window.addEventListener('idea-auth-ready', () => {
    if (document.readyState === 'complete') initializeAuthenticatedRating();
});
window.addEventListener('idea-auth-cleared', () => {
    initializedRatingOwner = null;
    stopAutoRatingProgressPolling();
    autoRatingInProgress = false;
    autoRatingRequestVersion += 1;
    const autoButton = document.getElementById('startAutoRating');
    if (autoButton) { autoButton.disabled = false; autoButton.textContent = 'Start Auto Swiss Rating'; }
    beginRatingSelection(null);
    document.getElementById('evolutionSelect').innerHTML = '<option value="">Current Evolution</option>';
    ratingSelectionLoading = false;
    document.getElementById('titleA').textContent = 'Sign in to compare ideas';
    document.getElementById('contentA').textContent = '';
    updateManualRatingControls();
});

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

    initializeAuthenticatedRating();
    document.getElementById('retryManualPair')?.addEventListener('click', async () => {
        if (!currentEvolutionId) await loadCurrentEvolution();
        else if (Object.keys(ideasDb).length < 2) await loadEvolution(currentEvolutionId);
        else await refreshPair();
    });
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

function updateAutoratingBillingWarning(message) {
    const warning = document.getElementById('autoratingBillingWarning');
    if (!warning) return;
    const text = typeof message === 'string' ? message.trim() : '';
    if (warning.textContent !== text) warning.textContent = text;
    warning.hidden = !text;
}

function announceRatingStatus(message) {
    const announcement = document.getElementById('ratingStatusAnnouncement');
    const text = String(message || '').trim();
    if (announcement && text && announcement.textContent !== text) {
        announcement.textContent = text;
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
        evaluation: config.evaluation || null,
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

function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

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

function normalizeMarkdownStructure(text) {
    if (!text) return '';

    const commonSectionHeadings = [
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
        'Overview',
        'Summary',
        'Implementation',
        'Constraints',
        'Description'
    ].sort((a, b) => b.length - a.length);
    const headingPattern = commonSectionHeadings
        .map(heading => heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|');
    const numberedHeadingPattern = `(?:\\d+\\.\\s*)?(?:${headingPattern})`;
    const inlineKnownHeadingPattern = new RegExp(`(?<=[^#\\n])\\s*(#{1,6})\\s*(${numberedHeadingPattern})(?=\\s|[#*:_\\-0-9A-Z("]|$)`, 'g');

    const normalizeSegment = (segment) => {
        let normalized = segment;
        normalized = normalized.replace(inlineKnownHeadingPattern, '\n$1 $2');
        normalized = normalized.replace(/^(#{1,6})(?!#)(?=\S)/gm, '$1 ');
        normalized = normalized.replace(/([:.!?])\s*(\d+\.\s+)/g, '$1\n$2');
        normalized = normalized.replace(/([.!?])\s*([*-]\s+)/g, '$1\n$2');
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

// Add the small allow-list markdown rendering function from viewer.js.
function renderMarkdown(text) {
    if (!text) return '';

    text = normalizeMarkdownStructure(text);

    // Escape HTML to prevent XSS
    text = text.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

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

// Pair selection is bound to the selected run and signed-in account.
async function refreshPair() {
    if (voteInFlight || ratingSelectionLoading) return;
    const context = captureRatingContext();
    const request = ++pairRequestVersion;
    const stillCurrent = () => request === pairRequestVersion && isRatingContextCurrent(context);
    currentPair = null;
    pairContext = null;
    pairLoading = true;
    updateManualRatingControls();
    try {
        if (!context.owner || !context.evolutionId || Object.keys(ideasDb).length < 2) {
            document.getElementById('titleA').textContent = 'No comparison available';
            document.getElementById('contentA').textContent = 'Choose an experiment with at least two saved ideas.';
            document.getElementById('titleB').textContent = '';
            document.getElementById('contentB').textContent = '';
            return;
        }
        const response = await fetch('/api/get-efficient-pair', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ evolution_id: context.evolutionId })
        });
        if (!stillCurrent()) return;
        if (!response.ok) throw new Error(`Failed to get efficient pair: ${response.status}`);
        const data = await response.json();
        if (!stillCurrent()) return;
        if (!data.idea_a?.id || !data.idea_b?.id || data.idea_a.id === data.idea_b.id || !data.pair_token) {
            throw new Error('Incomplete pair payload');
        }
        const ideaA = normalizeIdeaRecord(data.idea_a, data.idea_a.id);
        const ideaB = normalizeIdeaRecord(data.idea_b, data.idea_b.id);
        ideasDb[ideaA.id] = ideaA;
        ideasDb[ideaB.id] = ideaB;
        currentPair = [ideaA, ideaB];
        pairContext = { ...context, pairToken: data.pair_token };
        setManualSwissStatus(data.swiss_status);
        const ideaDataA = getIdeaData(ideaA);
        const ideaDataB = getIdeaData(ideaB);
        document.getElementById('titleA').textContent = ideaDataA.title;
        document.getElementById('titleB').textContent = ideaDataB.title;
        document.getElementById('contentA').innerHTML = renderMarkdown(ideaDataA.content);
        document.getElementById('contentB').innerHTML = renderMarkdown(ideaDataB.content);
        window.IdeaWorkbench?.renderPrimaryArtifact(document.getElementById('artifactA'), ideaA, context.evolutionId);
        window.IdeaWorkbench?.renderPrimaryArtifact(document.getElementById('artifactB'), ideaB, context.evolutionId);
        document.getElementById('manualRatingFeedback').hidden = true;
        document.querySelector('.ideas-container').style.display = 'flex';
        document.getElementById('autoRatingResults').style.display = 'none';
    } catch (error) {
        if (!stillCurrent()) return;
        console.error('Error getting Swiss pair:', error);
        setManualSwissStatus(null);
        currentPair = null;
        pairContext = null;
        showManualRatingFeedback('Could not load a pair. Check your connection and use Reload pair.');
    } finally {
        if (stillCurrent()) {
            pairLoading = false;
            updateManualRatingControls();
        }
    }
}

// A vote retains its pair, run and owner even if the selection changes while saving.
let voteInFlight = false;
async function vote(outcome) {
    if (!currentPair || !pairContext || !isRatingContextCurrent(pairContext)
        || voteInFlight || pairLoading || ratingSelectionLoading) return;
    const context = captureRatingContext();
    const votedPair = currentPair.slice();
    const pairToken = pairContext.pairToken;
    const request = ++voteRequestVersion;
    const stillCurrent = () => request === voteRequestVersion && isRatingContextCurrent(context);
    voteInFlight = true;
    updateManualRatingControls();
    document.getElementById('manualRatingFeedback').hidden = true;
    announceRatingStatus('Saving your rating…');
    try {
        const response = await fetch('/api/submit-rating', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                idea_a_id: votedPair[0].id,
                idea_b_id: votedPair[1].id,
                outcome,
                evolution_id: context.evolutionId,
                pair_token: pairToken
            })
        });
        if (!stillCurrent()) return;
        if (response.status === 409) {
            voteInFlight = false;
            await refreshPair();
            if (stillCurrent() && currentPair && pairContext && isRatingContextCurrent(pairContext)) {
                showManualRatingFeedback('Your rating was not saved because this comparison changed. The current pair has been reloaded.');
            } else if (stillCurrent()) {
                showManualRatingFeedback('Your rating was not saved. Could not load the current pair. Use Reload pair to retry.');
            }
            return;
        }
        if (!response.ok) throw new Error('Failed to submit rating');
        const result = await response.json();
        if (!stillCurrent()) return;
        if (result.updated_elos) {
            Object.entries(result.updated_elos).forEach(([id, elo]) => {
                if (!ideasDb[id]) return;
                if (!ideasDb[id].ratings) ideasDb[id].ratings = { auto: ideasDb[id].elo || 1500, manual: 1500 };
                ideasDb[id].ratings.manual = elo;
            });
        }
        if (result.updated_match_counts) {
            Object.entries(result.updated_match_counts).forEach(([id, counts]) => {
                if (!ideasDb[id]) return;
                ideasDb[id].match_count = counts.total;
                ideasDb[id].manual_match_count = counts.manual;
                ideasDb[id].auto_match_count = counts.auto;
            });
        } else {
            votedPair.forEach(idea => {
                if (!ideasDb[idea.id]) return;
                ideasDb[idea.id].match_count = (ideasDb[idea.id].match_count || 0) + 1;
                ideasDb[idea.id].manual_match_count = (ideasDb[idea.id].manual_match_count || 0) + 1;
            });
        }
        voteInFlight = false;
        await refreshPair();
    } catch (error) {
        if (!stillCurrent()) return;
        console.error('Error submitting rating:', error);
        currentPair = null;
        pairContext = null;
        showManualRatingFeedback('Rating could not be confirmed. Reload the pair to check the saved state before voting again.');
    } finally {
        if (stillCurrent()) {
            voteInFlight = false;
            updateManualRatingControls();
        }
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
    const votes = document.getElementById('manualVoteBar');
    if (votes) votes.hidden = mode !== 'manual';

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

// Add event listener for keydown events
document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest('input, textarea, select, button, a, [contenteditable], .candidate-inspector, .modal')) return;
    if (document.querySelector('.ideas-container')?.style.display === 'none' || document.getElementById('manualVoteBar')?.hidden) return;
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

function showManualRatingFeedback(message) {
    const feedback = document.getElementById('manualRatingFeedback');
    if (!feedback) return;
    feedback.querySelector('span').textContent = message;
    feedback.hidden = false;
    announceRatingStatus(message);
}

async function loadCurrentEvolution() {
    const context = beginRatingSelection(null);
    try {
        const progressResponse = await fetch('/api/progress');
        if (!isRatingContextCurrent(context)) return;
        if (!progressResponse.ok) throw new Error('Could not load the current experiment.');
        const progress = await progressResponse.json();
        if (!isRatingContextCurrent(context) || document.getElementById('evolutionSelect').value) return;
        currentEvolutionId = progress.evolution_id || null;
        context.evolutionId = currentEvolutionId;
        if (progress.is_running) {
            document.getElementById('titleA').textContent = 'Evolution is still running';
            document.getElementById('contentA').textContent = 'Return to Evolution to follow its progress. Rating becomes available after it completes or stops.';
            return;
        }
        if (!currentEvolutionId) {
            document.getElementById('titleA').textContent = 'Choose an experiment';
            document.getElementById('contentA').textContent = 'Select a saved evolution to compare its ideas, or start an experiment in Evolution.';
            return;
        }
        const response = await fetch(`/api/evolution/${encodeURIComponent(context.evolutionId)}`);
        if (!isRatingContextCurrent(context)) return;
        if (!response.ok) throw new Error('Could not load this experiment. Use Reload pair to retry.');
        const payload = await response.json();
        if (!isRatingContextCurrent(context) || document.getElementById('evolutionSelect').value) return;
        const data = payload.data || {};
        if (!data.history?.length) throw new Error('This experiment has no saved ideas yet.');
        setCurrentEvolutionAnalytics(data);
        await initializeRating(flattenGenerationHistory(data.history, `current-${context.evolutionId}`), false);
        if (!isRatingContextCurrent(context)) return;
        ratingSelectionLoading = false;
        await refreshPair();
    } catch (error) {
        if (isRatingContextCurrent(context)) showManualRatingFeedback(error.message || 'Could not load this experiment. Please retry.');
    } finally {
        if (isRatingContextCurrent(context)) {
            ratingSelectionLoading = false;
            updateManualRatingControls();
        }
    }
}

async function loadEvolutions() {
    const owner = window.currentUser?.uid;
    const request = ++evolutionListRequestVersion;
    const stillCurrent = () => owner && owner === window.currentUser?.uid && request === evolutionListRequestVersion;
    try {
        const response = await fetch('/api/evolutions');
        if (!stillCurrent()) return;
        if (!response.ok) throw new Error('Could not load saved experiments. Refresh the page to retry.');
        const evolutions = await response.json();
        if (!stillCurrent()) return;
        const select = document.getElementById('evolutionSelect');
        const selected = select.value;
        select.innerHTML = '<option value="">Current Evolution</option>';
        evolutions.forEach(evolution => {
            const option = document.createElement('option');
            option.value = evolution.id;
            option.textContent = evolution.filename || `Evolution from ${new Date(evolution.timestamp).toLocaleString()}`;
            select.appendChild(option);
        });
        if (selected && Array.from(select.options).some(option => option.value === selected)) select.value = selected;
    } catch (error) {
        if (stillCurrent()) showManualRatingFeedback(error.message);
    }
}

document.getElementById('evolutionSelect').addEventListener('change', async (event) => {
    const evolutionId = event.target.value;
    if (evolutionId) await loadEvolution(evolutionId);
    else await loadCurrentEvolution();
});

// Add this function to load available models
async function loadModels() {
    const owner = window.currentUser?.uid;
    const request = ++modelRequestVersion;
    const stillCurrent = () => owner && owner === window.currentUser?.uid && request === modelRequestVersion;
    try {
        const response = await fetch('/api/models');
        if (!stillCurrent()) return;
        if (response.ok) {
            const data = await response.json();
            if (!stillCurrent()) return;
            const modelSelect = document.getElementById('modelSelect');

            window.IdeaWorkbench?.configureModelPicker({
                modelId: 'modelSelect', providerId: 'ratingProviderSelect',
                models: data.models || [], defaultModel: data.default,
            });
            await window.checkApiKeyStatus?.();
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
    progressBar.setAttribute('aria-valuetext', `${roundedPercent}% complete`);

    const announcementStep = roundedPercent >= 100 ? 100 : Math.floor(roundedPercent / 10) * 10;
    if (progressBar.dataset.announcedStep !== String(announcementStep)) {
        progressBar.dataset.announcedStep = String(announcementStep);
        announceRatingStatus(`Automated rating ${announcementStep}% complete`);
    }

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

    if (ratingSelectionLoading || currentEvolutionId !== evolutionId) {
        showManualRatingFeedback('Wait for the selected experiment to load before starting a rating run.');
        return;
    }
    const context = captureRatingContext();
    const request = ++autoRatingRequestVersion;
    const stillCurrent = () => request === autoRatingRequestVersion && isRatingContextCurrent(context);
    autoRatingInProgress = true;
    startAutoRatingBtn.disabled = true;
    startAutoRatingBtn.textContent = 'Auto Swiss Rating...';
    announceRatingStatus('Automated Swiss rating started');
    updateAutoratingBillingWarning(null);

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
        progressBar.setAttribute('aria-valuetext', '0% complete');
        delete progressBar.dataset.announcedStep;
        progressBar.style.display = 'block';

        // Force a reflow to ensure the browser updates the progress bar
        void progressBar.offsetHeight;

        console.log("Progress bar initialized to 0%");
    } else {
        console.error("Progress bar element not found during initialization!");
    }

    document.getElementById('ratingStats').innerHTML = 'Processing Swiss rounds...';
    document.getElementById('rankingsTable').innerHTML = '';

    let progressPollingActive = true;
    try {
        const ratingStatsEl = document.getElementById('ratingStats');
        let expectedRounds = tournamentEstimate.targetRounds;
        let expectedTotalMatches = tournamentEstimate.estimatedTotalMatches;
        let allResults = [];
        let finalIdeas = [];
        let finalTokenCounts = null;
        let lastProgressVersion = -1;
        const pollProgress = async () => {
            if (!progressPollingActive || !stillCurrent()) return;
            try {
                const progressResponse = await fetch('/api/auto-rate/progress');
                if (!stillCurrent()) return;
                if (progressResponse.ok) {
                    const progressData = await progressResponse.json();
                    if (!stillCurrent()) return;
                    if (progressData?.billing_warning) updateAutoratingBillingWarning(progressData.billing_warning);
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
                if (progressPollingActive && stillCurrent()) {
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
        if (request === autoRatingRequestVersion) stopAutoRatingProgressPolling();
        if (!stillCurrent()) return;

        if (!response.ok) {
            let errorMessage = 'Failed to perform automated Swiss rating';
            try {
                const errorPayload = await response.json();
                if (!stillCurrent()) return;
                if (errorPayload?.billing_warning) updateAutoratingBillingWarning(errorPayload.billing_warning);
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
        if (!stillCurrent()) return;
        if (data.billing_warning) updateAutoratingBillingWarning(data.billing_warning);
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
        if (!stillCurrent()) return;
        console.error('Error during auto-rating:', error);
        stopAutoRatingProgressPolling();
        document.getElementById('ratingStats').innerHTML = `<p class="text-danger">Error: ${escapeHtml(error.message)}</p>`;
        announceRatingStatus(`Automated rating failed: ${error.message}`);
    } finally {
        progressPollingActive = false;
        if (request === autoRatingRequestVersion) {
            stopAutoRatingProgressPolling();
            startAutoRatingBtn.textContent = 'Start Auto Swiss Rating';
            startAutoRatingBtn.disabled = false;
            autoRatingInProgress = false;
        }
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
        const columnHeader = header.closest('th');
        const isActive = column === currentSortColumn;

        if (isActive) {
            arrow.className = `sort-arrow active ${currentSortDirection}`;
        } else {
            arrow.className = 'sort-arrow inactive';
        }
        if (columnHeader) {
            columnHeader.setAttribute(
                'aria-sort',
                isActive ? (currentSortDirection === 'asc' ? 'ascending' : 'descending') : 'none'
            );
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

        row.innerHTML = buildRankingRowHtml(idea, index);

        row.querySelector('.idea-details-trigger')?.addEventListener('click', function () {
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
        <td><button type="button" class="idea-details-trigger">${escapeHtml(idea.title || 'Untitled')}</button></td>
        ${getIdeaRatingCellHtml(idea)}
        <td>${escapeHtml(getIdeaMatchCountByType(idea))}</td>
        <td>${escapeHtml(formatRankingMetric(diversityScore, 4))}</td>
        <td>${escapeHtml(formatRankingMetric(overallFitness, 4))}</td>
        <td>${escapeHtml(birthGeneration)}</td>
        <td>${escapeHtml(deathGeneration)}</td>
    `;
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
    const totalThoughtTokens = (tokenCounts.total_thoughts || 0).toLocaleString();
    const totalCachedInputTokens = (tokenCounts.total_cached_input || 0).toLocaleString();

    // Component totals and details for critic
    const criticTotal = tokenCounts.critic.total.toLocaleString();
    const criticInput = tokenCounts.critic.input.toLocaleString();
    const criticOutput = tokenCounts.critic.output.toLocaleString();
    const criticThoughts = (tokenCounts.critic.thoughts || 0).toLocaleString();
    const criticCachedInput = (tokenCounts.critic.cached_input || 0).toLocaleString();
    const criticModel = tokenCounts.critic.model;
    const criticCost = tokenCounts.critic.cost.toFixed(4);

    // Get cost information
    const costInfo = tokenCounts.cost;
    const totalCost = costInfo.total_cost.toFixed(4);
    const inputCost = costInfo.input_cost.toFixed(4);
    const outputCost = costInfo.output_cost.toFixed(4);
    const thoughtCost = (costInfo.thought_cost || 0).toFixed(4);
    const cachedInputCost = (costInfo.cached_input_cost || 0).toFixed(4);

    // Prepare alternative cost estimates using the critic token counts
    let estimatesList = '';
    if (tokenCounts.estimates) {
        estimatesList = Object.values(tokenCounts.estimates).map(e => {
            return `<li class="list-group-item d-flex justify-content-between align-items-center">${escapeHtml(e.name)}<span>$${e.cost.toFixed(4)} <small class="text-muted">estimate</small></span></li>`;
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
                                    <div class="small text-muted mt-1">
                                        <span>Thoughts: ${totalThoughtTokens}</span>
                                        ${tokenCounts.total_cached_input ? `<span class="ms-2">Cached input: ${totalCachedInputTokens}</span>` : ''}
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
                                </div>
                            </div>

                            <h6>Component Breakdown:</h6>
                            <ul class="list-group">
                                <li class="list-group-item">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span>Critic <span class="badge bg-secondary">${escapeHtml(criticModel)}</span></span>
                                        <span class="badge bg-primary rounded-pill">${criticTotal}</span>
                                    </div>
                                    <div class="small text-muted mt-1">
                                        <span>Input: ${criticInput}</span> |
                                        <span>Output: ${criticOutput}</span> |
                                        ${tokenCounts.critic.thoughts ? `<span>Thoughts: ${criticThoughts}</span> |` : ''}
                                        ${tokenCounts.critic.cached_input ? `<span>Cached input: ${criticCachedInput}</span> |` : ''}
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
    rememberModalFocusOrigin(modalContainer);
    const modal = new bootstrap.Modal(modalContainer);
    modal.show();

    // Create charts after the modal is shown
    modalContainer.addEventListener('shown.bs.modal', function () {
        // For autorating, we only have the critic component, so create simpler charts

        // Create a pie chart for input vs output/thinking tokens
        const pieCtx = document.getElementById('autoratingTokenPieChart').getContext('2d');
        new Chart(pieCtx, {
            type: 'pie',
            data: {
                labels: ['Input Tokens', 'Output Tokens', 'Thinking Tokens'],
                datasets: [{
                    data: [
                        tokenCounts.total_input,
                        tokenCounts.total_output,
                        tokenCounts.total_thoughts || 0
                    ],
                    backgroundColor: [
                        getRaterChartTheme().violet,
                        getRaterChartTheme().green,
                        getRaterChartTheme().blue
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Token Distribution'
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
                labels: ['Input Cost', 'Output Cost', 'Thinking Cost', 'Total Cost'],
                datasets: [{
                    label: 'Cost (USD)',
                    data: [
                        parseFloat(inputCost),
                        parseFloat(outputCost),
                        parseFloat(thoughtCost),
                        parseFloat(totalCost)
                    ],
                    backgroundColor: [getRaterChartTheme().violet, getRaterChartTheme().green, getRaterChartTheme().blue, getRaterChartTheme().coral]
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

    const context = captureRatingContext();
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

        if (!isRatingContextCurrent(context)) return;
        if (!response.ok) {
            throw new Error('Failed to reset ratings and match counts');
        }

        const data = await response.json();
        if (!isRatingContextCurrent(context)) return;

        // Refresh only if the user is still viewing the run that was reset.
        const reloaded = await loadEvolution(evolutionId);
        if (!reloaded) return;

        // Show a success message
        alert(data.message);

        // If we're viewing rankings, refresh them
        if (document.getElementById('autoRatingResults').style.display === 'block') {
            showRanking();
        }
    } catch (error) {
        if (!isRatingContextCurrent(context)) return;
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
                <h4 class="idea-details-title"></h4>
                <p><strong>Generation (${currentGenerationView === 'birth' ? 'Birth' : 'Death'} view):</strong> <span data-idea-detail="generation"></span></p>
                <p><strong>Birth Generation:</strong> <span data-idea-detail="birth-generation"></span></p>
                <p><strong>Death Generation:</strong> <span data-idea-detail="death-generation"></span></p>
                <p><strong>Auto Rating:</strong> <span data-idea-detail="auto-rating"></span></p>
                <p><strong>Manual Rating:</strong> <span data-idea-detail="manual-rating"></span></p>
                <p><strong>Difference:</strong> <span class="${diffClass}" data-idea-detail="rating-difference"></span></p>
                <hr>
                <div class="idea-content">
                    ${renderMarkdown(idea.content || '')}
                </div>
            </div>
        </div>
    `;
    modalContent.querySelector('.idea-details-title').textContent = idea.title || 'Untitled';
    modalContent.querySelector('[data-idea-detail="generation"]').textContent = getIdeaGenerationLabel(idea);
    modalContent.querySelector('[data-idea-detail="birth-generation"]').textContent = getIdeaGenerationLabel(idea, 'birth');
    modalContent.querySelector('[data-idea-detail="death-generation"]').textContent = getIdeaGenerationLabel(idea, 'death');
    modalContent.querySelector('[data-idea-detail="auto-rating"]').textContent = autoRating;
    modalContent.querySelector('[data-idea-detail="manual-rating"]').textContent = manualRating;
    modalContent.querySelector('[data-idea-detail="rating-difference"]').textContent = `${diffSign}${diffRating}`;

    // Show the modal
    const modalElement = document.getElementById('ideaModal');
    rememberModalFocusOrigin(modalElement);
    window.IdeaWorkbench?.renderMetrics(document.getElementById('candidateMetricEvidence'), idea, currentEvolutionAnalytics?.evaluation || {});
    window.IdeaWorkbench?.renderPrimaryArtifact(document.getElementById('raterIdeaPrimaryArtifact'), idea, currentEvolutionId);
    window.IdeaWorkbench?.openInspector(modalElement);
}

async function loadEvolution(evolutionId) {
    const context = beginRatingSelection(evolutionId);
    try {
        const response = await fetch(`/api/evolution/${encodeURIComponent(evolutionId)}`);
        if (!isRatingContextCurrent(context)) return null;
        if (!response.ok) throw new Error('Could not load this experiment. Use Reload pair to retry.');
        const data = await response.json();
        if (!isRatingContextCurrent(context)) return null;
        if (!data.data?.history?.length) throw new Error('This experiment has no saved ideas yet.');
        setCurrentEvolutionAnalytics(data.data);
        await initializeRating(flattenGenerationHistory(data.data.history, `load-${evolutionId}`), false);
        if (!isRatingContextCurrent(context)) return null;
        ratingSelectionLoading = false;
        await refreshPair();
        return isRatingContextCurrent(context) ? data : null;
    } catch (error) {
        if (isRatingContextCurrent(context)) showManualRatingFeedback(error.message || 'Could not load this experiment. Please retry.');
        return null;
    } finally {
        if (isRatingContextCurrent(context)) {
            ratingSelectionLoading = false;
            updateManualRatingControls();
        }
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

    rememberModalFocusOrigin(modalElement);
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
                    borderColor: getRaterChartTheme().violet,
                    backgroundColor: raterChartColorWithAlpha(getRaterChartTheme().violet, 0.2),
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

function createFitnessChart(labels, combinedFitnessSeries, normalizedEloSeries, normalizedDiversitySeries, fitnessAlpha, usesSavedFitness = false) {
    const canvas = document.getElementById('fitnessChart');
    if (!canvas) {
        return;
    }

    const formulaEl = document.getElementById('fitnessFormula');
    if (formulaEl) {
        formulaEl.textContent = usesSavedFitness
            ? 'Mean of saved candidate fitness. Inspect a candidate for evaluator weights and evidence.'
            : `Legacy estimate: ${fitnessAlpha.toFixed(2)} × norm(mean ELO) + ${(1 - fitnessAlpha).toFixed(2)} × norm(diversity)`;
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
                    label: usesSavedFitness ? 'Saved selection fitness' : 'Legacy fitness estimate',
                    data: combinedFitnessSeries,
                    borderColor: getRaterChartTheme().violet,
                    backgroundColor: raterChartColorWithAlpha(getRaterChartTheme().violet, 0.2),
                    tension: 0.2
                },
                {
                    label: 'Normalized Mean ELO',
                    data: normalizedEloSeries,
                    borderColor: getRaterChartTheme().blue,
                    borderDash: [6, 4],
                    tension: 0.2
                },
                {
                    label: 'Normalized Diversity',
                    data: normalizedDiversitySeries,
                    borderColor: getRaterChartTheme().green,
                    borderDash: [6, 4],
                    tension: 0.2
                }
            ].filter((_, index) => !usesSavedFitness || index === 0)
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: usesSavedFitness ? 'Saved Selection Fitness by Generation' : (hasFitnessData ? 'Legacy Fitness Estimate' : 'Fitness (insufficient data)')
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
                    ...(usesSavedFitness ? {} : { max: 1 }),
                    title: {
                        display: true,
                        text: usesSavedFitness ? 'Selection fitness' : 'Normalized estimate'
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
                    borderColor: getRaterChartTheme().green,
                    backgroundColor: raterChartColorWithAlpha(getRaterChartTheme().green, 0.2),
                    tension: 0.1
                },
                {
                    label: 'Mean ELO',
                    data: meanElos,
                    borderColor: getRaterChartTheme().violet,
                    backgroundColor: raterChartColorWithAlpha(getRaterChartTheme().violet, 0.2),
                    tension: 0.1
                },
                {
                    label: 'Median ELO',
                    data: medianElos,
                    borderColor: getRaterChartTheme().blue,
                    backgroundColor: raterChartColorWithAlpha(getRaterChartTheme().blue, 0.2),
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

    const savedFitnessSeries = fitnessGenerations.map(generation => {
        const values = (analyticsHistory[generation] || [])
            .map(idea => window.IdeaWorkbench?.finite(idea?.selection_metrics?.fitness ?? idea?.fitness))
            .filter(value => Number.isFinite(value));
        return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    });
    const usesSavedFitness = savedFitnessSeries.some(value => Number.isFinite(value));
    const displayedFitnessSeries = usesSavedFitness ? savedFitnessSeries : combinedFitnessSeries;
    createFitnessChart(
        fitnessLabels,
        displayedFitnessSeries,
        normalizedEloSeries,
        normalizedDiversitySeries,
        fitnessAlpha,
        usesSavedFitness
    );
    const samplingNotice = window.IdeaWorkbench?.renderHistorySamplingNote(
        document.getElementById('raterDiversitySamplingNote'), currentEvolutionAnalytics?.diversity_history,
    ) || '';
    window.IdeaWorkbench?.renderChartTable(
        document.getElementById('raterChartTable'), fitnessLabels,
        [
            { label: 'Mean automatic Elo', data: meanAutoFitnessSeries },
            { label: 'Diversity', data: diversitySeries },
            { label: usesSavedFitness ? 'Saved selection fitness' : 'Legacy fitness estimate', data: displayedFitnessSeries },
        ], `Comparison analytics by generation${samplingNotice ? `. ${samplingNotice}` : ''}`,
    );
}
