let ideaA = null;
let ideaB = null;

// Fetch a random pair on load
window.addEventListener("load", () => {
  refreshPair();
});

function refreshPair() {
  fetch("/random-pair")
    .then((res) => {
      if (!res.ok) {
        throw new Error("Failed to get random pair: " + res.statusText);
      }
      return res.json();
    })
    .then((data) => {
      ideaA = data.ideaA;
      ideaB = data.ideaB;
      document.getElementById("titleA").textContent = ideaA.title;
      document.getElementById("proposalA").textContent = ideaA.proposal;
      document.getElementById("titleB").textContent = ideaB.title;
      document.getElementById("proposalB").textContent = ideaB.proposal;

      // Clear any leftover results
      document.getElementById("results-container").innerHTML = "";
    })
    .catch((error) => {
      console.error(error);
      alert(error.message);
    });
}

function vote(outcome) {
  // outcome can be "A", "B", or "tie"
  if (!ideaA || !ideaB) return;

  const payload = {
    idea_a_id: ideaA.id,
    idea_b_id: ideaB.id,
    outcome: outcome,
  };

  fetch("/rate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error("Failed to post rating: " + res.statusText);
      }
      return res.json();
    })
    .then((data) => {
      // ELO updated
      console.log(data);
      // Optionally, auto-refresh the pair
      refreshPair();
    })
    .catch((err) => {
      console.error(err);
      alert(err.message);
    });
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

// Add these functions
function showLoadDialog() {
    // Fetch available files from data directory
    fetch('/api/list-evolution-files')
        .then(res => res.json())
        .then(files => {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';

            files.forEach(file => {
                const button = document.createElement('button');
                button.className = 'btn btn-link';
                button.textContent = file;
                button.onclick = () => loadEvolutionData(file);
                fileList.appendChild(button);
            });

            new bootstrap.Modal(document.getElementById('loadDataModal')).show();
        });
}

function loadEvolutionData(filename) {
    fetch(`/api/load-evolution/${filename}`)
        .then(res => res.json())
        .then(data => {
            // Initialize the rater with the loaded data
            if (data && data.history) {
                const ideas = data.history.flat();
                IDEAS_DB = ideas.reduce((acc, idea) => {
                    acc[idea.id] = idea;
                    return acc;
                }, {});
                refreshPair();
            }
        });
}
