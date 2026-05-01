# 🍄 Mushroom Recognition System
### System identyfikacji 95 gatunków grzybów z wykorzystaniem Deep Learning

Kompleksowy system doradczy oparty na głębokim uczeniu maszynowym, pozwalający na błyskawiczną identyfikację gatunków grzybów, weryfikację ich toksyczności oraz wizualizację globalnego występowania.

---

## 🌟 O projekcie
Projekt zrealizowany w ramach przedmiotu **Systemy rozpoznawania obrazów** na **Politechnice Świętokrzyskiej**. System łączy nowoczesne techniki wizji komputerowej z danymi z otwartych baz biologicznych.

### Główne funkcjonalności:
* **Klasyfikacja AI:** Rozpoznawanie 95 gatunków grzybów (architektura **ResNet50**).
* **Bezpieczeństwo:** Hybrydowy system weryfikacji toksyczności (lokalna baza wiedzy + dane zewnętrzne).
* **Ekosystem danych:** Integracja z API Wikipedii (opisy) oraz GBIF (mapy występowania).
* **Reprodukowalność:** Skrypty do automatycznego przygotowania danych i stratyfikowanego podziału.

---

## 📊 Metodologia i Wyniki (Ewaluacja Testowa)

W celu uzyskania rzetelnych wyników zastosowano **stratyfikowany podział danych** (osobno dla każdej klasy):
* **Zbiór Treningowy:** 70%
* **Zbiór Walidacyjny:** 15%
* **Niezależny Zbiór Testowy:** 15%

### Wyniki na zbiorze testowym (Unseen Data):
| Metryka | Wynik |
| :--- | :--- |
| **Accuracy (Dokładność)** | **83.26%** |
| **Macro Precision** | **83.11%** |
| **Macro F1-Score** | **82.74%** |

*Model wykazuje wysoką zdolność generalizacji, a główne pomyłki zachodzą w obrębie gatunków konwergentnych wizualnie (np. Pleurotus ostreatus vs P. pulmonarius).*

---

## 🛠 Technologie
* **AI:** Python 3.10, PyTorch, Torchvision (ResNet50)
* **Backend:** Flask (REST API)
* **Frontend:** JS (Vanilla), Leaflet.js, HTML5/CSS3
* **Dane:** Kaggle (Mushroom dataset), iNaturalist (nowa klasa: Clathrus archeri)

---

## 📋 Instalacja i Uruchomienie

1. **Sklonuj repozytorium:**
   ```bash
   git clone [https://github.com/stuckelm03/Mushroom-Recognition-System.git](https://github.com/stuckelm03/Mushroom-Recognition-System.git)
   ```
2. **Zainstaluj biblioteki:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Uruchom:**
   ```bash
   python app.py
   ```
   
## 👥 Autorzy

* **Mateusz Z.**

* **Maciej S.**



*Projekt realizowany na Wydziale Elektrotechniki, Automatyki i Informatyki Politechniki Świętokrzyskiej.*
