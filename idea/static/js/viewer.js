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

    // Set up temperature sliders
    setupTemperatureSliders();

    const startButton = document.getElementById('startButton');
    if (startButton) {
        startButton.onclick = async function() {
            console.log("Starting evolution...");
            const popSize = document.getElementById('popSize').value;
            const generations = document.getElementById('generations').value;
            const ideaType = document.getElementById('ideaType').value;
            const modelType = document.getElementById('modelType').value;

            // Get temperature values
            const ideatorTemp = document.getElementById('ideatorTemp').value;
            const criticTemp = document.getElementById('criticTemp').value;
            const breederTemp = document.getElementById('breederTemp').value;

            console.log("Temperature values being sent:", {
                ideatorTemp,
                criticTemp,
                breederTemp
            });

            // Reset UI state
            resetUIState();

            // Disable start button
            startButton.disabled = true;
            startButton.textContent = 'Running...';

            try {
                console.log("Sending request with:", {
                    popSize, generations, ideaType, modelType,
                    ideatorTemp, criticTemp, breederTemp
                });

                const requestBody = {
                    popSize,
                    generations,
                    ideaType,
                    modelType,
                    ideatorTemp,
                    criticTemp,
                    breederTemp
                };

                console.log("Request body JSON:", JSON.stringify(requestBody));

                const response = await fetch('/api/start-evolution', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log("Received response data:", data);

                    // Reset contexts and index
                    contexts = data.contexts || [];
                    currentContextIndex = 0;
                    console.log("Loaded contexts:", contexts);

                    // Create progress bar container if it doesn't exist
                    if (!document.getElementById('progress-container')) {
                        createProgressBar();
                    }

                    // Start polling for updates
                    isEvolutionRunning = true;
                    pollProgress();

                    updateContextDisplay();  // Update context display after loading new data
                } else {
                    console.error("Failed to run evolution:", await response.text());
                    startButton.disabled = false;
                    startButton.textContent = 'Start Evolution';
                }
            } catch (error) {
                console.error("Error running evolution:", error);
                startButton.disabled = false;
                startButton.textContent = 'Start Evolution';
            }
        };
    }

    // Add direct event listener to download button
    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        console.log("Found download button in DOMContentLoaded, adding click listener");
        downloadButton.addEventListener('click', function() {
            console.log("Download button clicked directly");
            if (currentEvolutionData) {
                console.log("Using currentEvolutionData:", currentEvolutionData);
                downloadResults(currentEvolutionData);
            } else {
                console.error("No evolution data available");
                alert("No evolution data available to save");
            }
        });
    } else {
        console.error("Download button not found in DOMContentLoaded");
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
    updatedButton.onclick = function(event) {
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
            line.startsWith('<li') ||
            line.startsWith('<blockquote') ||
            line.startsWith('<pre')) {

            if (inParagraph) {
                // Close the paragraph before this line
                lines[i-1] += '</p>';
                inParagraph = false;
            }
            continue;
        }

        // If not in paragraph, start one
        if (!inParagraph) {
            lines[i] = '<p>' + lines[i];
            inParagraph = true;
        } else if (i === lines.length - 1 || lines[i+1].trim() === '') {
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

            // Add generation header with appropriate label
            const header = document.createElement('h2');
            header.className = 'generation-title';

            // Label the initial population as "Generation 0 (Initial)"
            if (index === 0) {
                header.textContent = `Generation 1 (Initial Population)`;
            } else {
                header.textContent = `Generation ${index + 1}`;
            }

            genDiv.appendChild(header);

            // Add ideas container with proper containment
            const scrollContainer = document.createElement('div');
            scrollContainer.className = 'scroll-container';
            scrollContainer.id = `scroll-container-${index}`;

            // Add a wrapper to ensure proper containment
            const scrollWrapper = document.createElement('div');
            scrollWrapper.className = 'scroll-wrapper';
            scrollWrapper.style.width = '100%';
            scrollWrapper.style.overflow = 'hidden';

            scrollWrapper.appendChild(scrollContainer);
            genDiv.appendChild(scrollWrapper);

            container.appendChild(genDiv);
        }

        // Get the scroll container for this generation
        const scrollContainer = document.getElementById(`scroll-container-${index}`);

        // Process each idea in this generation
        generation.forEach((idea, ideaIndex) => {
            // Check if this idea card already exists
            const existingCard = document.getElementById(`idea-${index}-${ideaIndex}`);
            if (existingCard) {
                // Card already exists, no need to recreate it
                return;
            }

            // Create a new card for this idea
            const card = document.createElement('div');
            card.className = 'card gen-card';
            card.id = `idea-${index}-${ideaIndex}`;

            // Set a fixed width to prevent stretching
            card.style.minWidth = '280px';
            card.style.maxWidth = '280px';
            card.style.flex = '0 0 auto';

            // Create a plain text preview for the card
            const plainPreview = createCardPreview(idea.proposal, 150);

            // Add "View Context" button for initial generation cards
            const viewContextButton = index === 0 ?
                `<button class="btn btn-outline-secondary btn-sm view-context me-2">View Context</button>` : '';

            card.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">${idea.title || 'Untitled'}</h5>
                    <div class="card-preview">
                        <p>${plainPreview}</p>
                    </div>
                    <div class="card-actions">
                        ${viewContextButton}
                        <button class="btn btn-primary btn-sm view-idea">View Full Idea</button>
                    </div>
                </div>
            `;

            // Add click handler for view button
            const viewButton = card.querySelector('.view-idea');
            viewButton.addEventListener('click', () => {
                showIdeaModal(idea);
            });

            // Add click handler for view context button if it exists
            if (index === 0) {
                const viewContextBtn = card.querySelector('.view-context');
                viewContextBtn.addEventListener('click', () => {
                    showContextModal();
                });
            }

            scrollContainer.appendChild(card);

            // Log that we've added a new card
            console.log(`Added new idea card: Generation ${index}, Idea ${ideaIndex + 1}`);
        });
    });
}

// Update the showIdeaModal function to ensure the modal is properly initialized
function showIdeaModal(idea) {
    // Get the modal element
    const modalElement = document.getElementById('ideaModal');

    // Set the title
    document.getElementById('ideaModalLabel').textContent = idea.title || 'Untitled';

    // Render the markdown content
    const modalContent = document.getElementById('ideaModalContent');

    // For debugging
    console.log("Rendering markdown for:", idea.proposal);

    // Set the content
    const renderedContent = renderMarkdown(idea.proposal || '');
    modalContent.innerHTML = renderedContent;

    // For debugging
    console.log("Rendered content:", renderedContent);

    // Initialize the modal if it hasn't been already
    let modal;
    if (window.bootstrap) {
        modal = new bootstrap.Modal(modalElement);
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
        contextContainer.style.display = 'none';
    }
}

// Clean implementation of downloadResults to prevent double-saving
function downloadResults(data) {
    console.log("Download function called with data:", data);

    // Ensure we have valid data
    if (!data || (!data.history && !data.contexts)) {
        console.error("Invalid data for saving:", data);
        alert("No data available to save");
        return;
    }

    // Use browser's built-in prompt for simplicity
    const filename = prompt("Enter filename to save:", `evolution_${new Date().toISOString().replace(/[:.]/g, '-')}.json`);

    // Exit if user cancels
    if (!filename) {
        console.log("Save cancelled by user");
        return;
    }

    console.log(`Saving to filename: ${filename}`);

    // Disable the download button to prevent double-clicks
    const downloadButton = document.getElementById('downloadButton');
    if (downloadButton) {
        downloadButton.disabled = true;
    }

    // Prepare the data for saving
    const saveData = {
        history: data.history || [],
        contexts: data.contexts || [],
        metadata: {
            timestamp: new Date().toISOString(),
            generations: data.total_generations || 0
        }
    };

    console.log("Prepared save data:", saveData);

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

            // Store the current evolution data in localStorage
            localStorage.setItem('currentEvolutionData', JSON.stringify(data.history));
        }

        if (data.contexts && data.contexts.length > 0) {
            contexts = data.contexts;
            currentContextIndex = 0;
            updateContextDisplay();
            document.querySelector('.context-navigation').style.display = 'block';
        }

        // Continue polling if evolution is still running
        if (data.is_running) {
            isEvolutionRunning = true;
            setTimeout(pollProgress, 1000); // Poll every second
        } else {
            // Evolution complete
            isEvolutionRunning = false;

            if (data.history && data.history.length > 0) {
                // Save final state and enable save button
                currentEvolutionData = data;
                renderGenerations(data.history);

                // Store the final evolution data in localStorage
                localStorage.setItem('currentEvolutionData', JSON.stringify(data.history));

                // Set up the download button
                setupDownloadButton(data);
            }

            // Update progress status
            if (progressStatus) {
                progressStatus.textContent = 'Evolution complete!';
            }

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
        // Continue polling even if there's an error, but only if evolution is still running
        if (isEvolutionRunning) {
            setTimeout(pollProgress, 2000); // Longer timeout on error
        }
    }
}

// Add debouncing to save operations
function debounce(func, wait) {
  let timeout;
  return function(...args) {
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

// Function to reset the UI state
function resetUIState() {
    // Clear generations container
    const container = document.getElementById('generations-container');
    if (container) {
        container.innerHTML = '';
    }

    // Reset contexts
    contexts = [];
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

// Function to set up temperature sliders
function setupTemperatureSliders() {
    const sliders = [
        { id: 'ideatorTemp', valueId: 'ideatorTempValue', defaultValue: 1.0 },
        { id: 'criticTemp', valueId: 'criticTempValue', defaultValue: 0.7 },
        { id: 'breederTemp', valueId: 'breederTempValue', defaultValue: 1.0 }
    ];

    sliders.forEach(slider => {
        const sliderElement = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);

        if (sliderElement && valueElement) {
            // Set initial value
            sliderElement.value = slider.defaultValue;
            valueElement.textContent = slider.defaultValue;

            // Log when slider changes (don't add new event listeners as we're using inline handlers)
            console.log(`Slider ${slider.id} initialized with value ${slider.defaultValue}`);
        } else {
            console.error(`Could not find elements for slider ${slider.id} or value display ${slider.valueId}`);
        }
    });

    // Log initial values to verify
    console.log("Initial temperature values:", {
        ideator: document.getElementById('ideatorTemp')?.value,
        critic: document.getElementById('criticTemp')?.value,
        breeder: document.getElementById('breederTemp')?.value
    });
}

// Function to show the context modal
function showContextModal() {
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
        // Format the context text into separate items
        const contextItems = contexts[currentContextIndex]
            .split('\n')
            .filter(item => item.trim())
            .map(item => `<div class="context-item">${item.trim()}</div>`)
            .join('');

        contextModalContent.innerHTML = `
            <div class="context-content">
                ${contextItems}
            </div>
        `;
    } else {
        contextModalContent.innerHTML = '<p class="text-muted">No context available</p>';
    }

    // Initialize and show the modal
    const modal = new bootstrap.Modal(contextModal);
    modal.show();
}
