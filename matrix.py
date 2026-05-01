import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

DATA_DIR = r"E:\Studia\Grzyby\dataset_split" 
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = "mushroom_model.pth"

BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Trwa analiza zbioru testowego (95 klas)...")
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Generowanie macierzy pomyłek...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(25, 22))
sns.heatmap(cm, 
            annot=False, 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar_kws={'label': 'Liczba predykcji'})

plt.title('Pełna Macierz Pomyłek - Zbiór Testowy (95 klas)', fontsize=20, fontweight='bold')
plt.xlabel('Predykcja AI', fontsize=15)
plt.ylabel('Rzeczywisty gatunek', fontsize=15)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout()

output_file = 'macierz_pomylek_pelna_test.png'
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Sukces! Pełna macierz została zapisana jako: {output_file}")