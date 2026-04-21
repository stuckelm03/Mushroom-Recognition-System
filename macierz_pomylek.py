import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Przygotowanie transformacji (takie samo jak w aplikacji)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Wczytywanie bazy zdjęć (data_folder)...")
# Wskazujemy na główny folder ze zdjęciami, który masz na stacjonarce
full_dataset = datasets.ImageFolder('data_folder/MO_94')

# Pobieramy klasy bezpośrednio z nazw folderów - dzięki temu na 100% będzie 95 klas!
classes = full_dataset.classes
print(f"Wykryto {len(classes)} klas grzybów w folderze.")

# Blokada "ziarna" losowości, aby zbiór testowy był zbliżony do tego z uczenia
torch.manual_seed(42) 
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
_, test_data = random_split(full_dataset, [train_size, test_size])
test_data.dataset.transform = test_transform

# Ładowanie danych po 32 zdjęcia na raz
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# 2. Budowanie modelu i ładowanie Twojego mózgu (.pth)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używam procesora: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Wczytujemy mózg, który wytrenowałeś (MUSI BYĆ WYTRENOWANY NA NOWO DLA 95 KLAS!)
model.load_state_dict(torch.load('mushroom_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 3. Zbieranie wyników i predykcji
y_true = []
y_pred = []

print("Trwa testowanie modelu na zdjęciach... To potrwa kilka chwil.")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 4. Generowanie i rysowanie wielkiej macierzy
print("Obliczanie matematycznej macierzy pomyłek...")
cm = confusion_matrix(y_true, y_pred)

# Ustawiamy GIGANTYCZNY rozmiar wykresu
plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)

plt.xlabel('Przewidziana klasa (Predykcja AI)', fontsize=25, labelpad=20)
plt.ylabel('Prawdziwa klasa (Rzeczywistość)', fontsize=25, labelpad=20)
# Zmieniony tytuł na 95 klas:
plt.title('Macierz Pomyłek (Confusion Matrix) dla 95 klas grzybów', fontsize=40, pad=30)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Zapis do wysokiej jakości pliku PNG
plt.savefig('macierz_pomylek_95.png', bbox_inches='tight', dpi=150)
print("Gotowe! Twoja macierz została zapisana jako 'macierz_pomylek_95.png' w tym folderze.")