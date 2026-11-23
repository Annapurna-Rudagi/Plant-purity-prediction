document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://127.0.0.1:5000/predict';
    const CIRCUMFERENCE = 2 * Math.PI * 75;

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const scanOverlay = document.getElementById('scanOverlay');
    const resultsSection = document.getElementById('resultsSection');
    const validResultGrid = document.getElementById('validResultGrid');
    const errorCard = document.getElementById('errorCard');

    const imagePreview = document.getElementById('imagePreview');
    const metaFilename = document.getElementById('metaFilename');
    const metaConfidence = document.getElementById('metaConfidence');

    const donutArc = document.getElementById('donutArc');
    const centerConfidence = document.getElementById('centerConfidence');
    const probLegend = document.getElementById('probLegend');

    const predictionHighlight = document.getElementById('predictionHighlight');
    const finalPredictionText = document.getElementById('finalPredictionText');
    const resultNote = document.getElementById('resultNote');
    const errorMsg = document.getElementById('errorMsg');
    const errorTitle = document.getElementById('errorTitle');
    const errorPillLabel = document.getElementById('errorPillLabel');

    const resetBtn = document.getElementById('resetBtn');
    const resetBtnError = document.getElementById('resetBtnError');

    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    });
    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    resetBtn.addEventListener('click', resetToUpload);
    resetBtnError.addEventListener('click', resetToUpload);

    function resetToUpload() {
        resultsSection.classList.add('hidden');
        fileInput.value = '';
    }

    // ---- Core flow: send the real image to the real API. No fallback, no simulation. ----
    async function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a JPG or PNG image.');
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        imagePreview.src = objectUrl;
        metaFilename.textContent = file.name;

        scanOverlay.classList.add('active');
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(API_URL, { method: 'POST', body: formData });
            const data = await response.json();
            scanOverlay.classList.remove('active');

            if (!response.ok || data.error) {
                showError(data.error || 'The server returned an error.', 'REQUEST FAILED', 'Could Not Get a Prediction');
                return;
            }

            // Honest pre-check result: this is not treated as an "error" — it's a
            // real, valid response from the server saying the image doesn't look
            // like a seedling. Shown with different wording, not a failure state.
            if (data.is_seedling_image === false) {
                showError(data.note, 'NOT A SEEDLING IMAGE', 'This Doesn\'t Look Like a Maize Seedling');
                return;
            }

            displayResult(data);
        } catch (err) {
            scanOverlay.classList.remove('active');
            showError(
                'Could not reach the prediction server. Make sure api_server.py is running ' +
                'at http://127.0.0.1:5000 before uploading an image. (No result is shown here ' +
                'unless the real model actually responds.)',
                'SERVER UNREACHABLE',
                'Could Not Reach the Server'
            );
        }
    }

    // Display-name mapping: the model still predicts female/hybrid/male
    // internally, but the interface shows these friendlier labels instead.
    const DISPLAY_NAMES = {
        female: "Pure",
        hybrid: "Hybrid",
        male: "Impure",
    };

    function toDisplayName(className) {
        const key = className.toLowerCase();
        return DISPLAY_NAMES[key] || className;
    }

    function displayResult(data) {
        resultsSection.classList.remove('hidden');
        validResultGrid.classList.remove('hidden');
        errorCard.classList.add('hidden');

        const isConfident = data.confident;
        const confidencePct = parseFloat(data.confidence);

        metaConfidence.textContent = data.confidence;
        centerConfidence.textContent = data.confidence;
        finalPredictionText.textContent = toDisplayName(data.prediction);

        if (isConfident) {
            predictionHighlight.classList.remove('uncertain');
            resultNote.textContent = `Classified as ${toDisplayName(data.prediction)} with ${data.confidence} confidence.`;
        } else {
            predictionHighlight.classList.add('uncertain');
            resultNote.textContent = data.note || `Model confidence (${data.confidence}) was below the trust threshold.`;
        }

        // Animate the donut to the top class's real confidence
        const offset = CIRCUMFERENCE - (CIRCUMFERENCE * confidencePct) / 100;
        donutArc.style.strokeDashoffset = CIRCUMFERENCE;
        setTimeout(() => { donutArc.style.strokeDashoffset = offset; }, 50);

        // Render every class's real probability — no class is hidden or invented
        probLegend.innerHTML = '';
        const probs = data.all_class_probabilities;
        Object.keys(probs).forEach((className) => {
            const pct = parseFloat(probs[className]);
            const row = document.createElement('div');
            row.className = 'legend-row';
            row.innerHTML = `
                <span class="legend-name">${toDisplayName(className)}</span>
                <div class="legend-track"><div class="legend-fill" style="width:${pct}%"></div></div>
                <span class="legend-pct">${probs[className]}</span>
            `;
            probLegend.appendChild(row);
        });
    }

    function showError(message, pillLabel, title) {
        resultsSection.classList.remove('hidden');
        validResultGrid.classList.add('hidden');
        errorCard.classList.remove('hidden');
        errorMsg.textContent = message;
        errorPillLabel.textContent = pillLabel || 'REQUEST FAILED';
        errorTitle.textContent = title || 'Could Not Get a Prediction';
    }
});
