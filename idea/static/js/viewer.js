// static/js/viewer.js

let pollingInterval;
let isEvolutionRunning = false;
let currentContextIndex = 0;
let contexts = [];
let currentEvolutionId = null;
let currentEvolutionData = null;
let generations = [];

document.addEventListener("DOMContentLoaded", () => {
    console.log("viewer.js loaded!");

    const startButton = document.getElementById('startButton');
    if (startButton) {
        startButton.onclick = async function() {
            console.log("Starting evolution...");
            const popSize = document.getElementById('popSize').value;
            const generations = document.getElementById('generations').value;
            const ideaType = document.getElementById('ideaType').value;
            const modelType = document.getElementById('modelType').value;
            const contextType = document.getElementById('contextType').value;

            // Clear previous results
            const container = document.getElementById('generations-container');
            container.innerHTML = '';
            startButton.disabled = true;
            startButton.textContent = 'Running...';

            try {
                console.log("Sending request with:", { popSize, generations, ideaType, modelType, contextType });
                const response = await fetch('/api/start-evolution', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        popSize,
                        generations,
                        ideaType,
                        modelType,
                        contextType
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log("Received response data:", data);

                    // Reset contexts and index
                    contexts = data.contexts || [];
                    currentContextIndex = 0;
                    console.log("Loaded contexts:", contexts);

                    if (data.history && Array.isArray(data.history)) {
                        renderGenerations(data.history);
                        updateContextDisplay();  // Update context display after loading new data
                        // Enable download button
                        const downloadButton = document.getElementById('downloadButton');
                        downloadButton.disabled = false;
                        downloadButton.textContent = 'Save Results';
                        downloadButton.onclick = () => downloadResults(data);
                    } else {
                        console.error("Invalid history data:", data);
                    }
                } else {
                    console.error("Failed to run evolution:", await response.text());
                }
            } catch (error) {
                console.error("Error running evolution:", error);
            } finally {
                startButton.disabled = false;
                startButton.textContent = 'Start Evolution';
            }
        };
    }

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

    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        downloadButton.disabled = true;
    }
});

function renderGenerations(gens) {
    generations = gens; // Store the generations globally
    console.log("Received generations:", generations); // Debug log

    if (!generations || generations.length === 0) {
        console.warn("Received empty generations data");
        return;
    }
    const container = document.getElementById('generations-container');
    container.innerHTML = '';

    generations.forEach((generation, index) => {
        const genDiv = document.createElement('div');
        genDiv.className = 'generation-section mb-4';

        // Add generation header
        const header = document.createElement('h2');
        header.className = 'generation-title';
        header.textContent = `Generation ${index + 1}`;
        genDiv.appendChild(header);

        // Add ideas container
        const scrollContainer = document.createElement('div');
        scrollContainer.className = 'scroll-container';

        // Add idea cards
        generation.forEach((idea, ideaIndex) => {
            const card = document.createElement('div');
            card.className = 'card gen-card';
            card.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">${idea.title || 'Untitled'}</h5>
                    <button class="btn btn-primary btn-sm view-idea">View</button>
                </div>
            `;

            // Add click handler for view button
            const viewButton = card.querySelector('.view-idea');
            viewButton.addEventListener('click', () => {
                showIdeaModal(idea);
            });

            scrollContainer.appendChild(card);
        });

        genDiv.appendChild(scrollContainer);
        container.appendChild(genDiv);
    });
}

function showIdeaModal(idea) {
    const modal = new bootstrap.Modal(document.getElementById('ideaModal'));
    document.getElementById('ideaModalLabel').textContent = idea.title;
    document.getElementById('ideaModalContent').textContent = idea.proposal;
    modal.show();
}

function updateContextDisplay() {
    const contextDisplay = document.getElementById('contextDisplay');

    if (contexts.length > 0) {
        // Format the context text into separate items
        const contextItems = contexts[currentContextIndex]
            .split('\n')
            .filter(item => item.trim())
            .map(item => `<div class="context-item">${item.trim()}</div>`)
            .join('');

        contextDisplay.innerHTML = `
            <div class="context-content">
                ${contextItems}
            </div>
            <div class="context-navigation">
                <div class="context-nav-buttons">
                    <button class="context-nav-btn" id="prevContext" ${currentContextIndex === 0 ? 'disabled' : ''}>
                        ← Previous
                    </button>
                    <span id="contextCounter">Context ${currentContextIndex + 1} of ${contexts.length}</span>
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
    }
}

function downloadResults(data) {
    const modal = new bootstrap.Modal(document.getElementById('saveDataModal'));
    const defaultFilename = `evolution-results-${new Date().toISOString().replace(/:/g, '-')}.json`;
    document.getElementById('saveFilename').value = defaultFilename;

    document.getElementById('confirmSave').onclick = async () => {
        const filename = document.getElementById('saveFilename').value;
        const resultsData = {
            history: data.history.map(generation =>
                generation.map(idea => ({
                    ...idea,
                    id: generateUUID(),
                    elo: 1500
                }))
            ),
            contexts: contexts  // Add contexts to saved data
        };

        try {
            const response = await fetch('/api/save-evolution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: resultsData,
                    filename: filename
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Show success feedback
            const downloadButton = document.getElementById('downloadButton');
            const originalText = downloadButton.textContent;
            downloadButton.textContent = 'Saved!';
            setTimeout(() => {
                downloadButton.textContent = originalText;
            }, 2000);

            modal.hide();
        } catch (error) {
            console.error('Error saving file:', error);
            alert('Error saving file: ' + error.message);
        }
    };

    modal.show();
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
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

                // Update contexts if available
                if (data.data.contexts) {
                    contexts = data.data.contexts;
                    currentContextIndex = 0;
                    updateContextDisplay();
                    document.querySelector('.context-navigation').style.display = 'block';
                }

                // Enable download button
                document.getElementById('downloadButton').disabled = false;
            }
        } else {
            console.error('Failed to load evolution:', await response.text());
        }
    } else {
        // Clear display for current evolution
        document.getElementById('generations-container').innerHTML = '';
        document.getElementById('contextDisplay').innerHTML =
            '<p class="text-muted">Context will appear here when evolution starts...</p>';
        document.querySelector('.context-navigation').style.display = 'none';
        document.getElementById('downloadButton').disabled = true;
    }
});

// Modify existing loadCurrentEvolution function
async function loadCurrentEvolution() {
    // ... existing code ...
}

// Call this when page loads
loadEvolutions();

async function pollProgress() {
    try {
        const response = await fetch('/api/progress');
        const data = await response.json();

        if (data.is_running) {
            // Update UI with current progress
            if (data.history && data.history.length > 0) {
                renderGenerations(data.history);
            }
            if (data.contexts) {
                contexts = data.contexts;
                currentContextIndex = 0;
                updateContextDisplay();
                document.querySelector('.context-navigation').style.display = 'block';
            }
            // Continue polling
            setTimeout(pollProgress, 1000);
        } else if (data.history && data.history.length > 0) {
            // Evolution complete - save final state and enable save button
            currentEvolutionData = data;
            renderGenerations(data.history);
            document.getElementById('downloadButton').disabled = false;

            // Show completion notification
            const startButton = document.getElementById('startButton');
            startButton.textContent = 'Evolution Complete!';
            startButton.disabled = false;
            setTimeout(() => {
                startButton.textContent = 'Start Evolution';
            }, 2000);
        }
    } catch (error) {
        console.error('Error polling progress:', error);
    }
}

function downloadResults() {
    if (!currentEvolutionData) {
        console.warn("No evolution data available to save");
        return;
    }

    // Ensure we have the data structure we need
    const evolutionData = {
        history: currentEvolutionData.history,
        contexts: currentEvolutionData.contexts,
        // Add any other relevant data you want to save
    };

    // Create and trigger download
    const dataStr = JSON.stringify(evolutionData);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = window.URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'evolution_results.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}

document.getElementById('startButton').addEventListener('click', async () => {
    const button = document.getElementById('startButton');
    button.disabled = true;
    button.textContent = 'Running...';

    // Reset state
    currentEvolutionData = null;
    document.getElementById('downloadButton').disabled = true;
    document.getElementById('generations-container').innerHTML = '';

    // Get configuration
    const config = {
        popSize: document.getElementById('popSize').value,
        generations: document.getElementById('generations').value,
        ideaType: document.getElementById('ideaType').value,
        modelType: document.getElementById('modelType').value,
        contextType: document.getElementById('contextType').value
    };

    try {
        const response = await fetch('/api/start-evolution', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Failed to start evolution');
        }

        // Start polling for progress
        pollProgress();
    } catch (error) {
        console.error('Error:', error);
        button.disabled = false;
        button.textContent = 'Start Evolution';
        alert('Error starting evolution: ' + error.message);
    }
});

// Add click handler for download button
document.getElementById('downloadButton').addEventListener('click', function() {
    if (currentEvolutionData && currentEvolutionData.history && currentEvolutionData.history.length > 0) {
        downloadResults();
    } else {
        console.warn("No evolution data available to save");
    }
});
