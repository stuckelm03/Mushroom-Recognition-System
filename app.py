import os
import io
import base64
import requests
import urllib3
import urllib.parse
import torch
import torch.nn as nn
import re
from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)

MUSHROOM_KNOWLEDGE = {
    "Volvopluteus gloiocephalus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Agaricus augustus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Amanita amerirubescens": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Amanita calyptrodermsa": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Armillaria mellea": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Armillaria tabescens": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Artomyces pyxidatus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Bolbitius titubans": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Boletus pallidus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Boletus rex-veris": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Cantharellus californicus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Cantharellus cinnabarinus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Cerioporus squamosus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Chlorophyllum brunneum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Clitocybe nuda": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Coprinellus micaceus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Coprinus comatus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Flammulina velutipes": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Entoloma abortivum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Ganoderma applanatum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Ganoderma oregonense": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Grifola frondosa": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Hericium coralloides": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Hericium erinaceus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Hypomyces lactifluorum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Ischnoderma resinosum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Laccaria ochropurpurea": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Lacrymaria lacrymabunda": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Lactarius indigo": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Laetiporus sulphureus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Lycoperdon perlatum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Lycoperdon pyriforme": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Mycena haematopus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Pleurotus ostreatus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Pleurotus pulmonarius": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Pluteus cervinus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Psathyrella candolleana": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Pseudohydnum gelatinosum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Psilocybe cyanescens": {"toxicity": "✅ Jadalny / Halucynogenny", "class": "safe"},
    "Psilocybe muliercula": {"toxicity": "✅ Jadalny / Halucynogenny", "class": "safe"},
    "Psilocybe pelliculosa": {"toxicity": "✅ Jadalny / Halucynogenny", "class": "safe"},
    "Psilocybe zapotecorum": {"toxicity": "✅ Jadalny / Halucynogenny", "class": "safe"},
    "Retiboletus ornatipes": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Sarcomyxa serotina": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Stropharia ambigua": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Stropharia rugosoannulata": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Suillus americanus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Suillus luteus": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Suillus spraguei": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Tricholoma murrillianum": {"toxicity": "✅ Jadalny", "class": "safe"},
    "Clathrus archeri": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Tylopilus rubrobrunneus": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Tylopilus felleus": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Coprinopsis lagopus": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Crucibulum laeve": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Cryptoporus volvatus": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Fomitopsis mounceae": {"toxicity": "⚠️ Toksyczny", "class": "danger"},
    "Ganoderma curtisii": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Ganoderma tsugae": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Gliophorus psittacinus": {"toxicity": "⚠️ Lekko trujący", "class": "danger"},
    "Gloeophyllum sepiarium": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Gymnopilus luteofolius": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Laricifomes officinalis": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Leucoagaricus americanus": {"toxicity": "❌ Odradzany", "class": "unknown"},
    "Leucoagaricus leucothites": {"toxicity": "⚠️ Lekko trujący", "class": "danger"},
    "Lycogala epidendrum": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Mycena leaiana": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Panaeolus foenisecii": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Panellus stipticus": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Phaeolus schweinitzii": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Phyllotopsis nidulans": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Psilocybe caerulescens": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Psilocybe cubensis": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Psilocybe neoxalapensis": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Schizophyllum commune": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Stereum ostrea": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Tapinella atrotomentosa": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Trametes versicolor": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Trametes gibbosa": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Trametes betulina": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Trichaptum biforme": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Tricholomopsis rutilans": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Tubaria furfuracea": {"toxicity": "❌ Niejadalny", "class": "unknown"},
    "Agaricus xanthodermus": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita augusta": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita brunnescens": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita flavoconia": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita muscaria": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita persicina": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Amanita velosa": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Chlorophyllum molybdites": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Daedaleopsis confragosa": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Galerina marginata": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Hygrophoropsis aurantiaca": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Hypholoma fasciculare": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Hypholoma lateritium": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Leratiomyces ceres": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Omphalotus illudens": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Omphalotus olivascens": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Panaeolus cinctulus": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Panaeolus papilionaceus": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Phlebia tremellosa": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Psilocybe allenii": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Psilocybe azurescens": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Psilocybe aztecorum": {"toxicity": "☠️ TRUJĄCY", "class": "danger"},
    "Psilocybe ovoideocystidiata": {"toxicity": "☠️ TRUJĄCY", "class": "danger"}
}

try:
    with open('classes.txt', 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    classes = [f"Klasa_{i}" for i in range(95)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

try:
    model.load_state_dict(torch.load('mushroom_model.pth', map_location=device, weights_only=True))
except Exception:
    pass

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_mushroom_info(mushroom_folder_name):
    clean_name = ''.join([i for i in mushroom_folder_name if not i.isdigit()]).strip('_ ')
    clean_name = clean_name.replace('_', ' ')
    
    info = {
        "name": clean_name,
        "summary": "Brak szczegółowych informacji w polskiej Wikipedii.",
        "toxicity": "Brak danych / Zawsze sprawdzaj w atlasie!",
        "toxicity_class": "unknown",
        "images": [],
        "url": ""
    }
    
    if clean_name in MUSHROOM_KNOWLEDGE:
        info["toxicity"] = MUSHROOM_KNOWLEDGE[clean_name]["toxicity"]
        info["toxicity_class"] = MUSHROOM_KNOWLEDGE[clean_name]["class"]
        
    headers = {
        'User-Agent': 'RozpoznawanieGrzybowApp/1.0 (Educational Project; Python Requests)'
    }
    
    try:
        search_url = "https://pl.wikipedia.org/w/api.php"
        search_params = {
            "action": "query", "list": "search", "srsearch": clean_name, 
            "format": "json", "utf8": 1, "srlimit": 1
        }
        
        s_req = requests.get(search_url, params=search_params, headers=headers, timeout=5, verify=False)
        s_req.raise_for_status()
        search_res = s_req.json()
        
        if 'query' in search_res and search_res['query'].get('search'):
            best_title = search_res['query']['search'][0]['title']
            
            page_params = {
                "action": "query", "prop": "extracts|info", "inprop": "url",
                "titles": best_title, "format": "json", 
                "explaintext": 1, "redirects": 1
            }
            
            p_req = requests.get(search_url, params=page_params, headers=headers, timeout=5, verify=False)
            p_req.raise_for_status()
            page_res = p_req.json()
            
            pages = page_res.get('query', {}).get('pages', {})
            if pages:
                page_id = list(pages.keys())[0]
                if page_id != "-1":
                    page_data = pages[page_id]
                    info["url"] = page_data.get('fullurl', '')
                    
                    extract = page_data.get('extract', '').strip()
                    extract = re.sub(r'==.*?==', '', extract).strip()
                    if extract:
                        info["summary"] = extract[:800] + "..." if len(extract) > 800 else extract
            
            safe_title = urllib.parse.quote(best_title.replace(' ', '_'))
            img_url_api = f"https://pl.wikipedia.org/api/rest_v1/page/media-list/{safe_title}"
            
            i_req = requests.get(img_url_api, headers=headers, timeout=5, verify=False)
            if i_req.status_code == 200:
                media_list = i_req.json().get('items', [])
                for item in media_list:
                    if item.get('type') == 'image':
                        src = item.get('srcset', [{}])[0].get('src', '')
                        if src and not any(x in src.lower() for x in ['icon', 'logo', 'symbol', 'question_book', 'map', 'status_iucn', 'red_list', 'ambox']):
                            if src.lower().endswith(('.png', '.jpg', '.jpeg')):
                                info["images"].append("https:" + src if src.startswith('//') else src)
                                if len(info["images"]) >= 4:
                                    break
                                    
    except requests.exceptions.Timeout:
        info["summary"] = "Przekroczono czas oczekiwania na odpowiedź. Serwery Wikipedii są przeciążone."
        print(f"\n[BŁĄD API WIKIPEDII]: Timeout\n")
    except requests.exceptions.ConnectionError:
        info["summary"] = "Błąd połączenia z siecią. Sprawdź swój dostęp do Internetu."
        print(f"\n[BŁĄD API WIKIPEDII]: Connection Error\n")
    except Exception as e:
        print(f"\n[BŁĄD API WIKIPEDII]: {e}\n")
            
    if clean_name not in MUSHROOM_KNOWLEDGE and info["summary"] not in ["Brak szczegółowych informacji w polskiej Wikipedii.", "Błąd połączenia z siecią. Sprawdź swój dostęp do Internetu.", "Przekroczono czas oczekiwania na odpowiedź. Serwery Wikipedii są przeciążone."]:
        full_text_lower = info["summary"].lower()
        if any(word in full_text_lower for word in ["śmiertelnie trując"]):
            info["toxicity"] = "☠️ Śmiertelnie trujący (AI)"
            info["toxicity_class"] = "danger"
        elif any(word in full_text_lower for word in [" trując", "grzyb trujący"]):
            info["toxicity"] = "⚠️ Trujący (AI)"
            info["toxicity_class"] = "danger"
        elif any(word in full_text_lower for word in ["niejadaln", "padlin", "cuchną"]):
            info["toxicity"] = "❌ Niejadalny (AI)"
            info["toxicity_class"] = "danger"
        elif any(word in full_text_lower for word in [" jadaln", "grzyb jadalny", "smaczn"]):
            info["toxicity"] = "✅ Jadalny (AI)"
            info["toxicity_class"] = "safe"
                
    return info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = None
    try:
        if 'image_url' in request.form and request.form['image_url'].strip() != '':
            url = request.form['image_url'].strip()
            
            if url.startswith('data:image'):
                header, encoded = url.split(',', 1)
                img_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                response = requests.get(url, timeout=5, verify=False)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
        elif 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
        else:
            return jsonify({'error': 'Nie podano pliku ani linku'})

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = outputs.max(1)
            
        result_raw = classes[predicted.item()]
        mushroom_data = get_mushroom_info(result_raw)
        
        return jsonify({
            'prediction': mushroom_data['name'],
            'summary': mushroom_data['summary'],
            'toxicity': mushroom_data['toxicity'],
            'toxicity_class': mushroom_data['toxicity_class'],
            'images': mushroom_data['images'],
            'url': mushroom_data['url']
        })
        
    except Exception as e:
        return jsonify({'error': f"Nie udało się przetworzyć pliku. Sprawdź format lub link."})

if __name__ == '__main__':
    app.run(debug=True)