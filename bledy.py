import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

# 1. Transformacje
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Wczytywanie bazy zdjęć...")
# Pamiętaj o dobrej nazwie folderu!
full_dataset = datasets.ImageFolder('data_folder/MO_94')
classes = full_dataset.classes
print(f"Wykryto {len(classes)} klas grzybów w folderze.")

# Zbieramy po jednym prawidłowym zdjęciu dla każdej klasy jako "wzór"
print("Przygotowywanie bazy zdjęć referencyjnych...")
reference_images_paths = {}
for path, label in full_dataset.samples:
    if label not in reference_images_paths:
        reference_images_paths[label] = path
    if len(reference_images_paths) == len(classes):
        break

def load_ref_image(path):
    img = Image.open(path).convert('RGB')
    return test_transform(img)

# Podział bazy
torch.manual_seed(42) 
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
_, test_data = random_split(full_dataset, [train_size, test_size])
test_data.dataset.transform = test_transform

test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Budowanie modelu
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('mushroom_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Funkcja cofająca normalizację, żeby obrazki ładnie wyglądały
def denormalize_img(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 3. Szukanie 10 błędów
errors = []
print("Szukam 10 pomyłek modelu...")

with torch.no_grad():
    for images, labels in test_loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, preds = outputs.max(1)
        
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                errors.append({
                    'wrong_img': images[i],
                    'true_idx': labels[i].item(),
                    'true_name': classes[labels[i].item()],
                    'pred_name': classes[preds[i].item()]
                })
                if len(errors) == 10:
                    break
        if len(errors) == 10:
            break

# 4. Rysowanie i zapisywanie w dwóch plikach (po 5 błędów w każdym)
print("Generowanie plansz porównawczych...")
paczki = [errors[:5], errors[5:]]

for numer_paczki, paczka in enumerate(paczki):
    # Tworzymy dużą planszę: 5 rzędów, 2 kolumny
    fig, axes = plt.subplots(5, 2, figsize=(12, 22))
    fig.suptitle(f'Analiza błędów modelu - Część {numer_paczki + 1}', fontsize=22, fontweight='bold')

    for row_idx, error_data in enumerate(paczka):
        
        # LEWA KOLUMNA: Błąd AI
        ax_wrong = axes[row_idx, 0]
        ax_wrong.imshow(denormalize_img(error_data['wrong_img']))
        tytul_bład = f"BŁĘDNA PREDYKCJA AI\nRozpoznano jako: {error_data['pred_name']}"
        ax_wrong.set_title(tytul_bład, fontsize=12, color='darkred', pad=10)
        ax_wrong.axis('off')
        
        # PRAWA KOLUMNA: Zdjęcie referencyjne poprawnego grzyba
        ax_ref = axes[row_idx, 1]
        ref_img_tensor = load_ref_image(reference_images_paths[error_data['true_idx']])
        ax_ref.imshow(denormalize_img(ref_img_tensor))
        tytul_ok = f"WZÓR GATUNKU\nPrawdziwy grzyb to: {error_data['true_name']}"
        ax_ref.set_title(tytul_ok, fontsize=12, color='darkgreen', pad=10)
        ax_ref.axis('off')

    # Ten parametr zapobiega nachodzeniu tekstów na siebie!
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    nazwa_pliku = f'porownanie_bledow_{numer_paczki + 1}.png'
    plt.savefig(nazwa_pliku, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano plik '{nazwa_pliku}'.")

print("Zakończono pomyślnie! Pliki są gotowe do wklejenia w sprawozdanie.")