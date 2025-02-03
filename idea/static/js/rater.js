let currentPair = null;
let currentEvolutionId = null;
let ideasDb = {};

// Fetch a random pair on load
window.addEventListener("load", () => {
  refreshPair();
  loadEvolutions();
  loadCurrentEvolution();
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

async function refreshPair() {
    if (Object.keys(ideasDb).length < 2) {
        document.getElementById("titleA").textContent = "No ideas to rate";
        document.getElementById("proposalA").textContent = "";
        document.getElementById("titleB").textContent = "";
        document.getElementById("proposalB").textContent = "";
        return;
    }

    // Get random pair
    const ids = Object.keys(ideasDb);
    const idA = ids[Math.floor(Math.random() * ids.length)];
    let idB;
    do {
        idB = ids[Math.floor(Math.random() * ids.length)];
    } while (idB === idA);

    currentPair = [ideasDb[idA], ideasDb[idB]];

    // Display the pair
    document.getElementById("titleA").textContent = currentPair[0].title;
    document.getElementById("proposalA").textContent = currentPair[0].proposal;
    document.getElementById("titleB").textContent = currentPair[1].title;
    document.getElementById("proposalB").textContent = currentPair[1].proposal;
}

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

        if (response.ok) {
            const result = await response.json();
            // Update local ELO ratings
            if (result.updated_elos) {
                Object.entries(result.updated_elos).forEach(([id, elo]) => {
                    if (ideasDb[id]) {
                        ideasDb[id].elo = elo;
                    }
                });
            }
            await refreshPair();
        }
    } catch (error) {
        console.error('Error submitting rating:', error);
    }
}

function showRanking() {
  fetch("/ranking")
    .then((res) => {
      if (!res.ok) {
        throw new Error("Failed to get ranking: " + res.statusText);
      }
      return res.json();
    })
    .then((ranking) => {
      const resultsContainer = document.getElementById("results-container");
      resultsContainer.innerHTML = "<h2>Ranking</h2>";

      const table = document.createElement("table");
      const header = document.createElement("tr");
      header.innerHTML = "<th>ELO</th><th>Title</th><th>Proposal</th>";
      table.appendChild(header);

      ranking.forEach((idea) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${idea.elo.toFixed(2)}</td>
          <td>${idea.title}</td>
          <td>${idea.proposal}</td>
        `;
        table.appendChild(row);
      });

      resultsContainer.appendChild(table);
    })
    .catch((error) => {
      console.error(error);
      alert(error.message);
    });
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
        const response = await fetch('/api/generations');
        if (response.ok) {
            const generations = await response.json();
            if (generations.length > 0) {
                // Flatten all generations into a single array of ideas
                const ideas = generations.flat().map(idea => ({
                    ...idea,
                    id: generateUUID(),
                    elo: 1500
                }));
                await initializeRating(ideas);
            }
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
