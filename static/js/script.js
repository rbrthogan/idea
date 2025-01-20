// static/js/script.js

// On page load, we fetch the generation data and build the UI.
document.addEventListener("DOMContentLoaded", init);

async function init() {
  const generations = await fetchGenerations();
  renderGenerations(generations);
}

async function fetchGenerations() {
  const response = await fetch("/api/generations");
  if (!response.ok) {
    console.error("Failed to fetch generations:", response.statusText);
    return [];
  }
  return response.json();
}

function renderGenerations(generations) {
  const container = document.getElementById("generations-container");
  container.innerHTML = ""; // Clear existing content

  generations.forEach((genData, genIndex) => {
    // Title for this generation
    const genTitle = document.createElement("h3");
    genTitle.className = "generation-title";
    genTitle.textContent = `Generation ${genIndex + 1}`;
    container.appendChild(genTitle);

    // A horizontal scroll container for proposals
    const scrollContainer = document.createElement("div");
    scrollContainer.className = "scroll-container";

    // Create a card for each proposal
    genData.forEach((proposal, propIndex) => {
      const card = document.createElement("div");
      card.className = "card gen-card shadow-sm";

      card.innerHTML = `
        <div class="card-body">
          <h5 class="card-title">${proposal.title}</h5>
          <button class="btn btn-primary mt-3" data-gen="${genIndex}" data-prop="${propIndex}">
            View More
          </button>
        </div>
      `;

      scrollContainer.appendChild(card);
    });

    container.appendChild(scrollContainer);
  });

  // Attach click listeners to "View More" buttons
  attachViewMoreHandlers(generations);
}

function attachViewMoreHandlers(generations) {
  const buttons = document.querySelectorAll(".btn[data-gen]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const genIndex = parseInt(btn.getAttribute("data-gen"), 10);
      const propIndex = parseInt(btn.getAttribute("data-prop"), 10);
      const proposal = generations[genIndex][propIndex];
      showProposalModal(proposal.title, proposal.proposal);
    });
  });
}

function showProposalModal(title, proposalText) {
  // Set modal title
  const modalTitle = document.getElementById("proposalModalLabel");
  modalTitle.textContent = title;

  // Replace newlines with <br> for formatting
  const modalBody = document.getElementById("proposalModalBody");
  modalBody.innerHTML = proposalText.replace(/\n/g, "<br/>");

  // Show the modal (Bootstrap 5)
  const proposalModal = new bootstrap.Modal(document.getElementById("proposalModal"));
  proposalModal.show();
}
