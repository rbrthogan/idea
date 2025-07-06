function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

let currentPair = null;
let currentEvolutionId = null;
let ideasDb = {};
let currentRatingType = 'auto';
let autoRatingInProgress = false;
let currentModalIdea = null;

// Sorting state
let currentSortColumn = null;
let currentSortDirection = 'asc';
let originalIdeasOrder = [];

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

  // Load data
  loadEvolutions();
  loadCurrentEvolution();
  loadModels();
  toggleRatingMode('manual'); // Start in manual mode

  // Set up retry mechanism for refreshPair
  setTimeout(() => {
    if (Object.keys(ideasDb).length < 2) {
      console.log("No ideas loaded yet, retrying refreshPair...");
      refreshPair();
    }
  }, 3000);
});

async function initializeRating(ideas, shouldRefreshPair = true) {
    // Reset state
    ideasDb = {};

    // Initialize ideas database with IDs
    ideas.forEach(idea => {
        if (!idea.id) {
            idea.id = generateUUID();
        }
        if (!idea.elo) {
            idea.elo = 1500;
        }
        ideasDb[idea.id] = idea;
    });

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
    text = text.replace(/```([\s\S]*?)```/g, function(match, code) {
        return '<pre><code>' + code.trim() + '</code></pre>';
    });

    // Process inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Process headings - ensure they're at the start of a line
    text = text.replace(/^(#{1,6})\s+(.*?)$/gm, function(match, hashes, content) {
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
                lines[i-1] += '</p>';
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

        // Get efficient pair from backend API
        const eloRangeInput = document.getElementById('manualEloRange');
        const eloRange = eloRangeInput ? parseInt(eloRangeInput.value) || 100 : 100;

        try {
            const response = await fetch('/api/get-efficient-pair', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    evolution_id: currentEvolutionId,
                    elo_range: eloRange
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to get efficient pair: ${response.status}`);
            }

            const data = await response.json();
            const ideaA = data.idea_a;
            const ideaB = data.idea_b;

            // Update local database with the returned ideas (they may have been modified)
            if (ideaA && ideaA.id) {
                ideasDb[ideaA.id] = ideaA;
            }
            if (ideaB && ideaB.id) {
                ideasDb[ideaB.id] = ideaB;
            }

            currentPair = [ideaA, ideaB];

            // Display the ideas without showing ratings
            document.getElementById('titleA').textContent = ideaA.title || 'Untitled';
            document.getElementById('titleB').textContent = ideaB.title || 'Untitled';

            document.getElementById('contentA').innerHTML = renderMarkdown(ideaA.content || '');
            document.getElementById('contentB').innerHTML = renderMarkdown(ideaB.content || '');

            console.log(`Efficient pair selected: ${ideaA.id} vs ${ideaB.id}`);

        } catch (error) {
            console.error('Error getting efficient pair, falling back to random selection:', error);

            // Fallback to random selection if API call fails
            const [ideaA, ideaB] = getRandomPair(ideas);
            currentPair = [ideaA, ideaB];

            // Display the ideas without showing ratings
            document.getElementById('titleA').textContent = ideaA.title || 'Untitled';
            document.getElementById('titleB').textContent = ideaB.title || 'Untitled';

            document.getElementById('contentA').innerHTML = renderMarkdown(ideaA.content || '');
            document.getElementById('contentB').innerHTML = renderMarkdown(ideaB.content || '');
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
        const ideas = Object.values(ideasDb);

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

        // First, check localStorage for current evolution data
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
                } else if (evolutionState && evolutionState.history) {
                    // New format - object with history and diversity_history
                    generationsData = evolutionState.history;
                } else {
                    console.error("Invalid evolution data format");
                    // Continue to API call if localStorage parsing fails
                    throw new Error("Invalid evolution data format");
                }

                if (generationsData && generationsData.length > 0) {
                    // Flatten all generations into a single array of ideas
                    const ideas = generationsData.flat().map(idea => ({
                        ...idea,
                        id: generateUUID(),
                        elo: 1500
                    }));
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

        // If localStorage doesn't have data, try the API
        const response = await fetch('/api/generations');

        if (response.ok) {
            const generations = await response.json();
            console.log("Loaded generations from API:", generations);

            if (generations && generations.length > 0) {
                // Flatten all generations into a single array of ideas
                const ideas = generations.flat().map(idea => ({
                    ...idea,
                    id: generateUUID(),
                    elo: 1500
                }));
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

                // Retry after a delay if no ideas were found
                setTimeout(() => {
                    console.log("Retrying to load current evolution...");
                    loadCurrentEvolution();
                }, 2000);
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
                    // Flatten all generations into a single array of ideas
                    const ideas = data.data.history.flat().map((idea, index) => ({
                        ...idea,
                        id: idea.id || generateUUID(),
                        elo: idea.elo || idea.ratings?.auto || 1500,
                        // Add generation info if not present
                        generation: idea.generation !== undefined ? idea.generation : Math.floor(index / (data.data.history[0]?.length || 1))
                    }));
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

// Update the auto-rating event listener to include clickable ideas and generation info
const startAutoRatingBtn = document.getElementById('startAutoRating');
startAutoRatingBtn.addEventListener('click', async function() {
    if (autoRatingInProgress) {
        return; // prevent multiple simultaneous runs
    }

    const numComparisons = parseInt(document.getElementById('numComparisons').value);
    const evolutionId = document.getElementById('evolutionSelect').value;
    const modelId = document.getElementById('modelSelect').value;
    const eloRange = parseInt(document.getElementById('eloRange').value || 100);

    if (!evolutionId) {
        alert('Please select an evolution first');
        return;
    }

    autoRatingInProgress = true;
    startAutoRatingBtn.disabled = true;
    startAutoRatingBtn.textContent = 'Auto Rating...';

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

    document.getElementById('ratingStats').innerHTML = 'Processing...';
    document.getElementById('rankingsTable').innerHTML = '';

    try {
        // Break the process into smaller chunks (e.g., 5 comparisons per request)
        const chunkSize = 5;
        const chunks = Math.ceil(numComparisons / chunkSize);
        let completedComparisons = 0;
        let totalCompletedComparisons = 0;
        let allResults = [];
        let finalIdeas = [];
        let finalTokenCounts = null;
        let accumulatedTokenCounts = {
            total: 0,
            total_input: 0,
            total_output: 0,
            cost: {
                input_cost: 0,
                output_cost: 0,
                total_cost: 0,
                currency: 'USD'
            },
            critic: {
                total: 0,
                input: 0,
                output: 0,
                model: null,
                cost: 0
            },
            models: {},
            estimates: {}
        };

        for (let i = 0; i < chunks; i++) {
            // Calculate how many comparisons to do in this chunk
            const comparisonsInChunk = Math.min(chunkSize, numComparisons - completedComparisons);

            // Make the API call for this chunk
            const response = await fetch('/api/auto-rate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    numComparisons: comparisonsInChunk,
                    evolutionId: evolutionId,
                    modelId: modelId,
                    eloRange: eloRange,
                    skipSave: false // Always save to ensure match counts are tracked properly
                })
            });

            if (!response.ok) {
                throw new Error('Failed to perform automated rating');
            }

            const data = await response.json();

            // Add results from this chunk
            allResults = allResults.concat(data.results || []);

            // Track new comparisons completed in this chunk
            const newComparisons = data.new_comparisons || data.results?.length || 0;
            completedComparisons += newComparisons;

            // Get the total completed comparisons (including previous ones)
            totalCompletedComparisons = data.completed_comparisons || completedComparisons;

            // Store the latest ideas data
            finalIdeas = data.ideas || [];

                        // Accumulate token count data from each chunk
            if (data.token_counts) {
                const chunkTokens = data.token_counts;

                // Accumulate totals
                accumulatedTokenCounts.total += chunkTokens.total || 0;
                accumulatedTokenCounts.total_input += chunkTokens.total_input || 0;
                accumulatedTokenCounts.total_output += chunkTokens.total_output || 0;

                // Accumulate costs
                if (chunkTokens.cost) {
                    accumulatedTokenCounts.cost.input_cost += chunkTokens.cost.input_cost || 0;
                    accumulatedTokenCounts.cost.output_cost += chunkTokens.cost.output_cost || 0;
                    accumulatedTokenCounts.cost.total_cost += chunkTokens.cost.total_cost || 0;
                }

                // Accumulate critic data
                if (chunkTokens.critic) {
                    accumulatedTokenCounts.critic.total += chunkTokens.critic.total || 0;
                    accumulatedTokenCounts.critic.input += chunkTokens.critic.input || 0;
                    accumulatedTokenCounts.critic.output += chunkTokens.critic.output || 0;
                    accumulatedTokenCounts.critic.cost += chunkTokens.critic.cost || 0;

                    // Set model info from first chunk
                    if (!accumulatedTokenCounts.critic.model && chunkTokens.critic.model) {
                        accumulatedTokenCounts.critic.model = chunkTokens.critic.model;
                    }
                }

                // Set models info from first chunk
                if (chunkTokens.models && Object.keys(accumulatedTokenCounts.models).length === 0) {
                    accumulatedTokenCounts.models = chunkTokens.models;
                }

                // Accumulate estimates by updating them using the accumulated token counts
                if (chunkTokens.estimates) {
                    for (const [modelId, estimate] of Object.entries(chunkTokens.estimates)) {
                        if (!accumulatedTokenCounts.estimates[modelId]) {
                            accumulatedTokenCounts.estimates[modelId] = {
                                name: estimate.name,
                                cost: 0
                            };
                        }
                        accumulatedTokenCounts.estimates[modelId].cost += estimate.cost || 0;
                    }
                }

                // Store the accumulated counts as the final token counts
                finalTokenCounts = accumulatedTokenCounts;
            }

            // Update the local ideasDb with the latest data from the server
            finalIdeas.forEach(idea => {
                if (idea.id && ideasDb[idea.id]) {
                    // Update the existing idea in the database with the latest data
                    ideasDb[idea.id].elo = idea.elo;
                    ideasDb[idea.id].ratings = idea.ratings;
                    ideasDb[idea.id].match_count = idea.match_count || 0;
                    ideasDb[idea.id].auto_match_count = idea.auto_match_count || 0;
                    ideasDb[idea.id].manual_match_count = idea.manual_match_count || 0;
                }
            });

            // Update progress bar - based on our requested comparisons
            const progressPercent = Math.min(100, (completedComparisons / numComparisons) * 100);
            updateProgressBar(progressPercent);

            // Update stats during processing
            const stats = document.getElementById('ratingStats');
            stats.innerHTML = `
                <p>Progress: ${completedComparisons}/${numComparisons} comparisons</p>
                <p>Total comparisons completed: ${totalCompletedComparisons}</p>
                <p>A wins: ${allResults.filter(r => r.outcome === 'A').length}</p>
                <p>B wins: ${allResults.filter(r => r.outcome === 'B').length}</p>
                <p>Ties: ${allResults.filter(r => r.outcome === 'tie').length}</p>
            `;

            // Update the rankings table after each chunk to show progress
            updateRankingsTable(finalIdeas);
        }

        // Final update to ensure everything is in sync
        // Update the local ideasDb with the final data
        finalIdeas.forEach(idea => {
            if (idea.id && ideasDb[idea.id]) {
                // Update the existing idea in the database with the latest data
                ideasDb[idea.id].elo = idea.elo;
                ideasDb[idea.id].ratings = idea.ratings;
                ideasDb[idea.id].match_count = idea.match_count || 0;
                ideasDb[idea.id].auto_match_count = idea.auto_match_count || 0;
                ideasDb[idea.id].manual_match_count = idea.manual_match_count || 0;
            }
        });

        // Update the rankings table with the final data
        updateRankingsTable(finalIdeas);

        // Ensure progress bar shows 100% at the end
        console.log("Setting final progress to 100%");
        updateProgressBar(100);

        // Add a small delay to ensure the UI updates
        await new Promise(resolve => setTimeout(resolve, 300));

        // Update final stats including cost information
        const stats = document.getElementById('ratingStats');
        let statsHtml = `
            <p>Completed: ${completedComparisons}/${numComparisons} comparisons</p>
            <p>Total comparisons completed: ${totalCompletedComparisons}</p>
            <p>A wins: ${allResults.filter(r => r.outcome === 'A').length}</p>
            <p>B wins: ${allResults.filter(r => r.outcome === 'B').length}</p>
            <p>Ties: ${allResults.filter(r => r.outcome === 'tie').length}</p>
        `;

        // Add cost information if available
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

        // Add event listener for cost details button if it exists
        const costDetailsBtn = document.getElementById('autorating-cost-details-btn');
        if (costDetailsBtn && finalTokenCounts) {
            costDetailsBtn.addEventListener('click', function() {
                showAutoratingCostModal(finalTokenCounts);
            });
        }

    } catch (error) {
        console.error('Error during auto-rating:', error);
        document.getElementById('ratingStats').innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
    } finally {
        startAutoRatingBtn.textContent = 'Start Auto Rating';
        startAutoRatingBtn.disabled = false;
        autoRatingInProgress = false;
    }
});

// Helper function to update the rankings table
function updateRankingsTable(ideas) {
    // Show rankings with clickable rows and generation info
    const rankingsTable = document.getElementById('rankingsTable');
    rankingsTable.innerHTML = ''; // Clear existing content

    // Store ideas in a global variable for modal access
    window.rankedIdeas = ideas;

    // Store original order for sorting
    if (originalIdeasOrder.length === 0) {
        originalIdeasOrder = [...ideas];
    }

    ideas.forEach((idea, index) => {
        const row = document.createElement('tr');
        row.className = 'idea-row';
        row.dataset.ideaIndex = index;
        row.style.cursor = 'pointer';

        // Get the appropriate rating values
        const autoRating = idea.ratings?.auto || idea.elo || 1500;
        const manualRating = idea.ratings?.manual || 1500;
        const diffRating = autoRating - manualRating;

        // Get the appropriate match count based on the current rating type
        let matchCount = 0;
        if (currentRatingType === 'auto') {
            matchCount = idea.auto_match_count || 0;

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td>${autoRating}</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        } else if (currentRatingType === 'manual') {
            matchCount = idea.manual_match_count || 0;

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td>${manualRating}</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        } else {
            // For 'diff' or any other type, use the total match count
            matchCount = idea.match_count || 0;

            // Add color coding for diff
            const diffClass = diffRating > 0 ? 'text-success' : (diffRating < 0 ? 'text-danger' : '');
            const diffSign = diffRating > 0 ? '+' : '';

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td class="${diffClass}">${diffSign}${diffRating} (A:${autoRating}/M:${manualRating})</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        }

        // Add click event to show the idea details
        row.addEventListener('click', function() {
            showIdeaDetails(index);
        });

        rankingsTable.appendChild(row);
    });

    // Set up sorting event listeners
    setupTableSorting();

    // Create the ELO chart
    createEloChart(ideas, currentRatingType);
}

// Table sorting functionality
function setupTableSorting() {
    const headers = document.querySelectorAll('.sortable-header');

    headers.forEach(header => {
        header.addEventListener('click', function(e) {
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
                if (currentRatingType === 'auto') {
                    aValue = a.ratings?.auto || a.elo || 1500;
                    bValue = b.ratings?.auto || b.elo || 1500;
                } else if (currentRatingType === 'manual') {
                    aValue = a.ratings?.manual || 1500;
                    bValue = b.ratings?.manual || 1500;
                } else {
                    // For diff, sort by absolute difference
                    const aAuto = a.ratings?.auto || a.elo || 1500;
                    const aManual = a.ratings?.manual || 1500;
                    const bAuto = b.ratings?.auto || b.elo || 1500;
                    const bManual = b.ratings?.manual || 1500;
                    aValue = Math.abs(aAuto - aManual);
                    bValue = Math.abs(bAuto - bManual);
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

            case 'generation':
                aValue = a.generation !== undefined ? a.generation : -1;
                bValue = b.generation !== undefined ? b.generation : -1;
                break;

            default:
                return 0;
        }

        // Handle string vs numeric comparison
        if (typeof aValue === 'string' && typeof bValue === 'string') {
            return direction === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        } else {
            return direction === 'asc' ? aValue - bValue : bValue - aValue;
        }
    });

    // Update the table with sorted ideas
    window.rankedIdeas = ideas;
    updateRankingsTableContent(ideas);
}

function updateRankingsTableContent(ideas) {
    const rankingsTable = document.getElementById('rankingsTable');
    rankingsTable.innerHTML = ''; // Clear existing content

    ideas.forEach((idea, index) => {
        const row = document.createElement('tr');
        row.className = 'idea-row';
        row.dataset.ideaIndex = index;
        row.style.cursor = 'pointer';

        // Get the appropriate rating values
        const autoRating = idea.ratings?.auto || idea.elo || 1500;
        const manualRating = idea.ratings?.manual || 1500;
        const diffRating = autoRating - manualRating;

        // Get the appropriate match count based on the current rating type
        let matchCount = 0;
        if (currentRatingType === 'auto') {
            matchCount = idea.auto_match_count || 0;

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td>${autoRating}</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        } else if (currentRatingType === 'manual') {
            matchCount = idea.manual_match_count || 0;

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td>${manualRating}</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        } else {
            // For 'diff' or any other type, use the total match count
            matchCount = idea.match_count || 0;

            // Add color coding for diff
            const diffClass = diffRating > 0 ? 'text-success' : (diffRating < 0 ? 'text-danger' : '');
            const diffSign = diffRating > 0 ? '+' : '';

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td class="${diffClass}">${diffSign}${diffRating} (A:${autoRating}/M:${manualRating})</td>
                <td>${matchCount}</td>
                <td>${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</td>
            `;
        }

        // Add click event to show the idea details
        row.addEventListener('click', function() {
            showIdeaDetails(index);
        });

        rankingsTable.appendChild(row);
    });
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
                <p><strong>Generation:</strong> ${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</p>
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
document.addEventListener('DOMContentLoaded', function() {
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
    currentSortColumn = null;
    currentSortDirection = 'asc';

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
                            callback: function(value) {
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
                <p><strong>Generation:</strong> ${idea.generation !== undefined ? formatGenerationLabel(idea.generation) : '?'}</p>
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

        // Extract all ideas from all generations
        const allIdeas = [];
        data.data.history.forEach((generation, genIndex) => {
            generation.forEach(idea => {
                // Add generation info
                idea.generation = genIndex;

                // Ensure idea has an ID
                if (!idea.id) {
                    idea.id = generateUUID();
                }

                // Initialize ratings structure if needed
                if (!idea.ratings) {
                    idea.ratings = {
                        auto: idea.elo || 1500,
                        manual: 1500
                    };
                } else if (typeof idea.ratings === 'number') {
                    // Convert old format
                    const oldElo = idea.ratings;
                    idea.ratings = {
                        auto: oldElo,
                        manual: oldElo
                    };
                }

                // Ensure backward compatibility
                idea.elo = idea.ratings.auto;

                // Ensure match count properties exist
                idea.match_count = idea.match_count || 0;
                idea.auto_match_count = idea.auto_match_count || 0;
                idea.manual_match_count = idea.manual_match_count || 0;

                allIdeas.push(idea);
            });
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

// Fix the createEloChart function to handle the case when there's no existing chart
function createEloChart(ideas, ratingType) {
    // Group ideas by generation
    const generationGroups = {};

    ideas.forEach(idea => {
        const gen = idea.generation || 0;
        if (!generationGroups[gen]) {
            generationGroups[gen] = [];
        }
        generationGroups[gen].push(idea);
    });

    // Calculate max, mean, and median ELO for each generation
    const generations = Object.keys(generationGroups).sort((a, b) => a - b);
    const maxElos = [];
    const meanElos = [];
    const medianElos = [];

    generations.forEach(gen => {
        const genIdeas = generationGroups[gen];

        // Get the appropriate rating field based on type
        let ratings;
        if (ratingType === 'auto') {
            ratings = genIdeas.map(idea => idea.ratings?.auto || idea.elo || 1500);
        } else if (ratingType === 'manual') {
            ratings = genIdeas.map(idea => idea.ratings?.manual || 1500);
        } else if (ratingType === 'diff') {
            ratings = genIdeas.map(idea => {
                const auto = idea.ratings?.auto || idea.elo || 1500;
                const manual = idea.ratings?.manual || 1500;
                return auto - manual;
            });
        }

        const maxElo = Math.max(...ratings);
        const meanElo = ratings.reduce((sum, elo) => sum + elo, 0) / ratings.length;

        // Calculate median ELO
        const sortedRatings = [...ratings].sort((a, b) => a - b);
        const medianElo = sortedRatings.length % 2 === 0
            ? (sortedRatings[sortedRatings.length / 2 - 1] + sortedRatings[sortedRatings.length / 2]) / 2
            : sortedRatings[Math.floor(sortedRatings.length / 2)];

        maxElos.push(maxElo);
        meanElos.push(meanElo);
        medianElos.push(medianElo);
    });

    // Create the chart
    const ctx = document.getElementById('eloChart').getContext('2d');

    // Safely destroy existing chart if it exists
    if (window.eloChart instanceof Chart) {
        window.eloChart.destroy();
    }

    // Create new chart
    window.eloChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: generations.map(gen => gen === '0' ? 'Initial Population' : `Gen ${gen}`),
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
            plugins: {
                title: {
                    display: true,
                    text: `${ratingType.charAt(0).toUpperCase() + ratingType.slice(1)} Ratings by Generation`
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
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
}
