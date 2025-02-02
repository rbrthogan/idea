// static/js/viewer.js

let pollingInterval;
let isEvolutionRunning = false;

document.addEventListener("DOMContentLoaded", () => {
    console.log("viewer.js loaded!");

    const startButton = document.getElementById('startButton');
    if (startButton) {
        startButton.onclick = async function() {
            console.log("Starting evolution...");
            const popSize = document.getElementById('popSize').value;
            const generations = document.getElementById('generations').value;
            const ideaType = document.getElementById('ideaType').value;

            // Clear previous results
            const container = document.getElementById('generations-container');
            container.innerHTML = '';
            startButton.disabled = true;
            startButton.textContent = 'Running...';

            try {
                console.log("Sending request with:", { popSize, generations, ideaType });
                const response = await fetch('/api/start-evolution', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        popSize,
                        generations,
                        ideaType
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log("Received response data:", data);
                    if (data.history && Array.isArray(data.history)) {
                        renderGenerations(data.history);
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
