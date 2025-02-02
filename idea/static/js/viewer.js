// static/js/viewer.js

let pollingInterval;
let isEvolutionRunning = false;
let currentContextIndex = 0;
let contexts = [];

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
});

function renderGenerations(history) {
    console.log("Starting renderGenerations with:", history);
    const container = document.getElementById('generations-container');
    container.innerHTML = '';

    history.forEach((generation, index) => {
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
    const navigation = document.querySelector('.context-navigation');

    console.log('Updating context display:', {
        currentIndex: currentContextIndex,
        totalContexts: contexts.length,
        currentContext: contexts[currentContextIndex]
    });

    if (contexts.length > 0) {
        // Format the context text for better readability
        const contextText = contexts[currentContextIndex]
            .split(',')
            .map(word => word.trim())
            .join('\n');

        contextDisplay.innerHTML = `
            <div class="context-content">
                <pre>${contextText}</pre>
            </div>
            <div class="context-navigation mt-3">
                <div class="d-flex justify-content-between align-items-center">
                    <button class="btn btn-sm btn-outline-primary" id="prevContext" ${currentContextIndex === 0 ? 'disabled' : ''}>
                        &larr; Previous
                    </button>
                    <span id="contextCounter">Context ${currentContextIndex + 1} of ${contexts.length}</span>
                    <button class="btn btn-sm btn-outline-primary" id="nextContext" ${currentContextIndex === contexts.length - 1 ? 'disabled' : ''}>
                        Next &rarr;
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
