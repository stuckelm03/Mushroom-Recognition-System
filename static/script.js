let myMap = null;
let gbifLayer = null;

const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const imagePreview = document.getElementById('image-preview');
const resultContainer = document.getElementById('result-container');
const predictionText = document.getElementById('prediction-text');
const urlInput = document.getElementById('imageUrlInput');
const urlBtn = document.getElementById('urlSubmitBtn');

dropArea.addEventListener('click', () => fileElem.click());
fileElem.addEventListener('change', function() { handleFiles(this.files); });

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
});

dropArea.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));

window.addEventListener('paste', e => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let index in items) {
        if (items[index].kind === 'file') {
            const blob = items[index].getAsFile();
            handleFiles([new File([blob], "pasted.png", { type: items[index].type })]);
        }
    }
});

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function() {
        imagePreview.src = reader.result;
        imagePreview.style.display = 'block';
    }
    const formData = new FormData();
    formData.append('file', file);
    sendData(formData);
}

urlBtn.addEventListener('click', () => {
    const url = urlInput.value.trim();
    if (url === "") return;
    imagePreview.src = url;
    imagePreview.style.display = 'block';
    const formData = new FormData();
    formData.append('image_url', url);
    sendData(formData);
});

function sendData(formData) {
    resultContainer.style.display = 'block';
    predictionText.innerText = "Trwa analiza...";
    predictionText.style.color = "#f39c12";
    document.getElementById('wiki-info').style.display = 'none';

    fetch('/predict', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            predictionText.innerText = "Błąd: " + data.error;
            predictionText.style.color = "#e74c3c";
        } else {
            predictionText.innerText = data.prediction;
            predictionText.style.color = "#27ae60";
            
            const wikiInfo = document.getElementById('wiki-info');
            const toxicityBadge = document.getElementById('toxicity-badge');
            const wikiSummary = document.getElementById('wiki-summary');
            const wikiLink = document.getElementById('wiki-link');
            const gallery = document.getElementById('reference-gallery');
            const mapContainer = document.getElementById('map-container');
            
            wikiInfo.style.display = 'block';
            wikiSummary.innerText = data.summary;
            
            toxicityBadge.innerText = data.toxicity;
            toxicityBadge.className = 'badge ' + data.toxicity_class;

            if (data.url) {
                wikiLink.href = data.url;
                wikiLink.style.display = 'inline-block';
            } else {
                wikiLink.style.display = 'none';
            }

            gallery.innerHTML = '';
            if (data.images && data.images.length > 0) {
                data.images.forEach(imgUrl => {
                    const imgElement = document.createElement('img');
                    imgElement.src = imgUrl;
                    imgElement.alt = "Zdjęcie referencyjne " + data.prediction;
                    imgElement.onclick = () => window.open(imgUrl, '_blank');
                    gallery.appendChild(imgElement);
                });
            }

            mapContainer.style.display = 'block';
            
            if (!myMap) {
                myMap = L.map('map').setView([40, -20], 2);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap'
                }).addTo(myMap);
            }

            if (gbifLayer) {
                myMap.removeLayer(gbifLayer);
            }

            fetch(`https://api.gbif.org/v1/species/match?name=${encodeURIComponent(data.prediction)}`)
                .then(res => res.json())
                .then(gbifData => {
                    if (gbifData.usageKey) {
                        const tileUrl = `https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}@1x.png?srs=EPSG:3857&taxonKey=${gbifData.usageKey}&style=classic.point`;
                        gbifLayer = L.tileLayer(tileUrl, {
                            attribution: 'Dane występowania: GBIF',
                            opacity: 0.8
                        }).addTo(myMap);
                    }
                });
        }
    })
    .catch(error => {
        predictionText.innerText = "Błąd połączenia";
        predictionText.style.color = "#e74c3c";
    });
}