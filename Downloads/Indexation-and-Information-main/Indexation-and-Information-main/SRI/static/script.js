const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const statusBar = document.getElementById('status');
const resultsList = document.getElementById('resultsList');
const docModal = document.getElementById('docModal');
const modalClose = document.querySelector('.modal-close');

const modelNames = {
    'cosine': 'Cosinus',
    'boolean': 'Booléen',
    'boolean_extended': 'Booléen étendu',
    'lukasiewicz': 'Lukasiewicz',
    'kraft': 'Kraft',
    'jaccard': 'Jaccard',
    'dice': 'Dice',
    'euclidean': 'Euclidienne'
};

// Event listeners
searchBtn.addEventListener('click', performSearch);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

document.querySelectorAll('input[name="model"]').forEach(radio => {
    radio.addEventListener('change', performSearch);
});

modalClose.addEventListener('click', closeModal);
docModal.addEventListener('click', (e) => {
    if (e.target === docModal) closeModal();
});

async function performSearch() {
    const query = queryInput.value.trim();
    if (!query) return;

    const model = document.querySelector('input[name="model"]:checked').value;
    
    statusBar.textContent = 'Recherche en cours...';
    resultsList.innerHTML = '';

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, model })
        });

        const data = await response.json();
        displayResults(data, model);
    } catch (error) {
        console.error('Erreur:', error);
        statusBar.textContent = 'Erreur lors de la recherche';
    }
}

function displayResults(data, model) {
    const { results, count, elapsed } = data;
    
    if (count === 0) {
        statusBar.textContent = 'Aucun résultat trouvé.';
        resultsList.innerHTML = '<div class="no-results">Aucun résultat trouvé pour votre recherche.</div>';
        return;
    }

    statusBar.textContent = `${count} résultat(s) (${elapsed} secondes) - Modèle : ${modelNames[model]}`;

    resultsList.innerHTML = '';
    
    results.forEach((result) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        const ext = result.filename.split('.').pop().toUpperCase();
        const scorePct = result.score;
        let scoreClass = 'score-high';
        if (scorePct < 50) scoreClass = 'score-medium';
        if (scorePct < 20) scoreClass = 'score-low';

        card.innerHTML = `
            <div class="result-meta">
                <span>📄 ${result.filename} • ${ext} • Score: ${scorePct}%</span>
            </div>
            <div class="result-title">${result.filename.replace(/_/g, ' ').replace('.txt', '').replace('.pdf', '').split(/(?=[A-Z])/).join(' ')}</div>
            <div class="result-snippet">${result.snippet}</div>
            <div class="score-bar">
                <span class="score-tag ${scoreClass}">Pertinence: ${scorePct}%</span>
            </div>
        `;

        card.addEventListener('click', () => showDocument(result));
        resultsList.appendChild(card);
    });
}

async function showDocument(result) {
    try {
        const response = await fetch(`/api/document/${Array.from(document.querySelectorAll('.result-card')).indexOf(event.currentTarget) + 1}`);
        
        if (!response.ok) {
            // For now, use the data we already have
            displayDocumentModal(result);
            return;
        }

        const doc = await response.json();
        displayDocumentModal(doc);
    } catch (error) {
        displayDocumentModal(result);
    }
}

function displayDocumentModal(doc) {
    const ext = doc.filename.split('.').pop().toUpperCase();
    const title = doc.filename.replace(/_/g, ' ').replace(/\.[^.]+$/, '').split(/(?=[A-Z])/).join(' ').toUpperCase();
    
    document.getElementById('docTitle').textContent = title;
    document.getElementById('docExt').textContent = `📄 ${ext}`;
    document.getElementById('docPath').textContent = `📁 ${doc.filename}`;
    document.getElementById('docBody').textContent = doc.content;
    document.getElementById('docFooter').textContent = `Fichier: ${doc.filepath} • Cliquez en dehors ou sur la croix pour fermer`;
    
    docModal.classList.add('show');
}

function closeModal() {
    docModal.classList.remove('show');
}

// Focus on search input on page load
window.addEventListener('load', () => {
    queryInput.focus();
});
