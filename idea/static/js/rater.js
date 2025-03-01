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

// Fetch a random pair on load
window.addEventListener("load", () => {
  refreshPair();
  loadEvolutions();
  loadCurrentEvolution();
  loadModels();
  toggleRatingMode('manual'); // Start in manual mode
});

async function initializeRating(ideas) {
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

    await refreshPair();
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

// Update refreshPair function to not show ratings in the title
async function refreshPair() {
    try {
        // Get all ideas
        const ideas = Object.values(ideasDb);

        if (ideas.length < 2) {
            alert('Not enough ideas to compare');
            return;
        }

        // Randomly select two different ideas
        const [ideaA, ideaB] = getRandomPair(ideas);
        currentPair = [ideaA, ideaB];

        // Display the ideas without showing ratings
        document.getElementById('titleA').textContent = ideaA.title || 'Untitled';
        document.getElementById('titleB').textContent = ideaB.title || 'Untitled';

        document.getElementById('proposalA').innerHTML = renderMarkdown(ideaA.proposal || '');
        document.getElementById('proposalB').innerHTML = renderMarkdown(ideaB.proposal || '');

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

        // Update local ELO ratings for manual ratings
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

        // Get next pair
        await refreshPair();
    } catch (error) {
        console.error('Error submitting rating:', error);
        alert('Error: ' + error.message);
    }
}

// Update the showRanking function to include the chart
async function showRanking() {
    try {
        // Hide the idea comparison view
        document.querySelector('.ideas-container').style.display = 'none';

        // Show results section and reset
        const resultsSection = document.getElementById('autoRatingResults');
        resultsSection.style.display = 'block';
        document.getElementById('ratingProgress').style.display = 'none';
        document.getElementById('ratingStats').innerHTML = '<p>Current manual ratings</p>';

        // Get all ideas and sort by ELO
        const ideas = Object.values(ideasDb).sort((a, b) => b.elo - a.elo);

        // Store ideas in a global variable for modal access
        window.rankedIdeas = ideas;

        // Show rankings with clickable rows
        const rankingsTable = document.getElementById('rankingsTable');
        rankingsTable.innerHTML = ''; // Clear existing content

        ideas.forEach((idea, index) => {
            const row = document.createElement('tr');
            row.className = 'idea-row';
            row.dataset.ideaIndex = index;
            row.style.cursor = 'pointer';

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${idea.title || 'Untitled'}</td>
                <td>${idea.elo}</td>
                <td>${idea.generation || '?'}</td>
            `;

            // Add click event to show the idea details
            row.addEventListener('click', function() {
                showIdeaDetails(index);
            });

            rankingsTable.appendChild(row);
        });

        // After populating the rankings table, create the chart
        createEloChart(ideas, currentRatingType);
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
        const response = await fetch('/api/generations');

        if (response.ok) {
            const generations = await response.json();
            console.log("Loaded generations:", generations);

            if (generations && generations.length > 0) {
                // Flatten all generations into a single array of ideas
                const ideas = generations.flat().map(idea => ({
                    ...idea,
                    id: generateUUID(),
                    elo: 1500
                }));
                console.log("Processed ideas:", ideas);
                await initializeRating(ideas);
            } else {
                console.warn("No generations found or empty generations array");
                document.getElementById("titleA").textContent = "No ideas to rate";
                document.getElementById("proposalA").textContent = "Please run an evolution first or select a saved evolution.";
                document.getElementById("titleB").textContent = "";
                document.getElementById("proposalB").textContent = "";
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

    if (evolutionId) {
        const response = await fetch(`/api/evolution/${evolutionId}`);
        if (response.ok) {
            const data = await response.json();
            if (data.data && data.data.history) {
                // Flatten all generations into a single array of ideas
                const ideas = data.data.history.flat().map(idea => ({
                    ...idea,
                    id: idea.id || generateUUID()
                }));
                await initializeRating(ideas);
            }
        } else {
            console.error('Failed to load evolution:', await response.text());
        }
    } else {
        // Load current evolution
        await loadCurrentEvolution();
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

// Update the auto-rating event listener to include clickable ideas and generation info
document.getElementById('startAutoRating').addEventListener('click', async function() {
    const numComparisons = parseInt(document.getElementById('numComparisons').value);
    const evolutionId = document.getElementById('evolutionSelect').value;
    const modelId = document.getElementById('modelSelect').value;

    if (!evolutionId) {
        alert('Please select an evolution first');
        return;
    }

    // Hide the idea comparison view
    document.querySelector('.ideas-container').style.display = 'none';

    // Clear any previous results
    document.getElementById('results-container').innerHTML = '';

    // Show results section and reset
    const resultsSection = document.getElementById('autoRatingResults');
    resultsSection.style.display = 'block';
    document.getElementById('ratingProgress').style.width = '0%';
    document.getElementById('ratingStats').innerHTML = 'Processing...';
    document.getElementById('rankingsTable').innerHTML = '';

    try {
        // Break the process into smaller chunks (e.g., 5 comparisons per request)
        const chunkSize = 5;
        const chunks = Math.ceil(numComparisons / chunkSize);
        let completedComparisons = 0;
        let allResults = [];

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
                    skipSave: (i < chunks - 1) // Only save on the last chunk
                })
            });

            if (!response.ok) {
                throw new Error('Failed to perform automated rating');
            }

            const data = await response.json();

            // Add results from this chunk
            allResults = allResults.concat(data.results);
            completedComparisons += data.results.length;

            // Update progress bar
            const progressPercent = (completedComparisons / numComparisons) * 100;
            document.getElementById('ratingProgress').style.width = `${progressPercent}%`;

            // Update stats during processing
            const stats = document.getElementById('ratingStats');
            stats.innerHTML = `
                <p>Progress: ${completedComparisons}/${numComparisons} comparisons</p>
                <p>A wins: ${allResults.filter(r => r.outcome === 'A').length}</p>
                <p>B wins: ${allResults.filter(r => r.outcome === 'B').length}</p>
                <p>Ties: ${allResults.filter(r => r.outcome === 'tie').length}</p>
            `;

            // If this is the last chunk, show the final rankings with enhanced table
            if (i === chunks - 1) {
                // Show rankings with clickable rows and generation info
                const rankingsTable = document.getElementById('rankingsTable');
                rankingsTable.innerHTML = ''; // Clear existing content

                // Store ideas in a global variable for modal access
                window.rankedIdeas = data.ideas;

                data.ideas.forEach((idea, index) => {
                    const row = document.createElement('tr');
                    row.className = 'idea-row';
                    row.dataset.ideaIndex = index;
                    row.style.cursor = 'pointer';

                    row.innerHTML = `
                        <td>${index + 1}</td>
                        <td>${idea.title || 'Untitled'}</td>
                        <td>${idea.elo}</td>
                        <td>Gen ${idea.generation || '?'}</td>
                    `;

                    // Add click event to show the idea details
                    row.addEventListener('click', function() {
                        showIdeaDetails(index);
                    });

                    rankingsTable.appendChild(row);
                });
            }
        }

    } catch (error) {
        console.error('Error during auto rating:', error);
        document.getElementById('ratingStats').innerHTML = `Error: ${error.message}`;
    }
});

// Add this function to show idea details in a modal
function showIdeaDetails(ideaIndex) {
    if (!window.rankedIdeas || !window.rankedIdeas[ideaIndex]) {
        console.error('Idea not found');
        return;
    }

    const idea = window.rankedIdeas[ideaIndex];

    // Set modal title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Set modal content with markdown rendering
    const modalContent = document.getElementById('modalIdeaContent');
    modalContent.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h4>${idea.title || 'Untitled'}</h4>
                <p><strong>Generation:</strong> ${idea.generation || '?'}</p>
                <p><strong>ELO Rating:</strong> ${idea.elo}</p>
                <hr>
                <div class="idea-proposal">
                    ${renderMarkdown(idea.proposal || '')}
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

    // Update button states
    document.getElementById('autoRatingTypeBtn').classList.remove('active');
    document.getElementById('manualRatingTypeBtn').classList.remove('active');
    document.getElementById('diffRatingTypeBtn').classList.remove('active');

    document.getElementById(`${type}RatingTypeBtn`).classList.add('active');

    // If we're viewing rankings, refresh them with the new type
    if (document.getElementById('autoRatingResults').style.display === 'block') {
        showRanking();
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

        // Sort ideas based on rating type
        if (currentRatingType === 'auto') {
            ideas.sort((a, b) => {
                const aRating = a.ratings?.auto || a.elo || 1500;
                const bRating = b.ratings?.auto || b.elo || 1500;
                return bRating - aRating;
            });
            document.getElementById('ratingStats').innerHTML = '<p>Auto ratings (LLM-based)</p>';
        } else if (currentRatingType === 'manual') {
            ideas.sort((a, b) => {
                const aRating = a.ratings?.manual || 1500;
                const bRating = b.ratings?.manual || 1500;
                return bRating - aRating;
            });
            document.getElementById('ratingStats').innerHTML = '<p>Manual ratings (human-based)</p>';
        } else if (currentRatingType === 'diff') {
            // Sort by absolute difference between auto and manual
            ideas.sort((a, b) => {
                const aDiff = Math.abs((a.ratings?.auto || 1500) - (a.ratings?.manual || 1500));
                const bDiff = Math.abs((b.ratings?.auto || 1500) - (b.ratings?.manual || 1500));
                return bDiff - aDiff;
            });
            document.getElementById('ratingStats').innerHTML = '<p>Difference between auto and manual ratings</p>';
        }

        // Store ideas in a global variable for modal access
        window.rankedIdeas = ideas;

        // Show rankings with clickable rows
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

            if (currentRatingType === 'auto') {
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${idea.title || 'Untitled'}</td>
                    <td>${autoRating}</td>
                    <td>Gen ${idea.generation || '?'}</td>
                `;
            } else if (currentRatingType === 'manual') {
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${idea.title || 'Untitled'}</td>
                    <td>${manualRating}</td>
                    <td>Gen ${idea.generation || '?'}</td>
                `;
            } else if (currentRatingType === 'diff') {
                // Add color coding for diff
                const diffClass = diffRating > 0 ? 'text-success' : (diffRating < 0 ? 'text-danger' : '');
                const diffSign = diffRating > 0 ? '+' : '';

                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${idea.title || 'Untitled'}</td>
                    <td class="${diffClass}">${diffSign}${diffRating} (A:${autoRating}/M:${manualRating})</td>
                    <td>Gen ${idea.generation || '?'}</td>
                `;
            }

            // Add click event to show the idea details
            row.addEventListener('click', function() {
                showIdeaDetails(index);
            });

            rankingsTable.appendChild(row);
        });

        // Create the ELO chart
        createEloChart(ideas, currentRatingType);
    } catch (error) {
        console.error('Error showing ranking:', error);
    }
}

// Add function to reset ratings
async function resetRatings(type) {
    const evolutionId = document.getElementById('evolutionSelect').value;

    if (!evolutionId) {
        alert('Please select an evolution first');
        return;
    }

    if (!confirm(`Are you sure you want to reset ${type} ratings to their default value (1500)?`)) {
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
            throw new Error('Failed to reset ratings');
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
        console.error('Error resetting ratings:', error);
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
                <p><strong>Generation:</strong> ${idea.generation || '?'}</p>
                <p><strong>Auto Rating:</strong> ${autoRating}</p>
                <p><strong>Manual Rating:</strong> ${manualRating}</p>
                <p><strong>Difference:</strong> <span class="${diffClass}">${diffSign}${diffRating}</span></p>
                <hr>
                <div class="idea-proposal">
                    ${renderMarkdown(idea.proposal || '')}
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
                idea.generation = genIndex + 1;

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

// Add this function to get a random pair of ideas
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

    // Calculate max and mean ELO for each generation
    const generations = Object.keys(generationGroups).sort((a, b) => a - b);
    const maxElos = [];
    const meanElos = [];

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

        maxElos.push(maxElo);
        meanElos.push(meanElo);
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
            labels: generations.map(gen => `Gen ${gen}`),
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
