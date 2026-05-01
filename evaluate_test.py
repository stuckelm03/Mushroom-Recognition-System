import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"E:\Studia\Grzyby\dataset_split" 
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = os.path.join(BASE_DIR, 'mushroom_model.pth')

BATCH_SIZE = 32

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"Ładowanie ZBIORU TESTOWEGO z: {TEST_DIR}...")
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)
print(f"Liczba próbek testowych: {len(test_dataset)}")
print(f"Liczba klas: {num_classes}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

try:
    print(f"Wczytywanie wag z {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model załadowany pomyślnie.")
except Exception as e:
    print(f"BŁĄD Wczytywania modelu: {e}")
    print("Upewnij się, że plik mushroom_model.pth istnieje w tym samym folderze co ten skrypt.")
    exit()

model = model.to(device)
model.eval() 

print("Rozpoczynam ewaluację na zbiorze testowym... To chwilę potrwa.")
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

print("Ewaluacja zakończona. Generowanie raportów...")

print("\n" + "="*50)
print("RAPORT Z WYNIKAMI (SKOPIUJ TO DO PREZENTACJI):")
print("="*50)

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

accuracy = report_dict['accuracy']
macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
macro_f1 = report_dict['macro avg']['f1-score']

print(f"Accuracy (Dokładność ogólna) na teście: {accuracy * 100:.2f}%")
print(f"Precision (Precyzja Macro):             {macro_precision * 100:.2f}%")
print(f"Recall (Czułość Macro):                 {macro_recall * 100:.2f}%")
print(f"F1-Score (Wynik F1 Macro):              {macro_f1 * 100:.2f}%")
print("="*50 + "\n")

df_report = pd.DataFrame(report_dict).transpose()
df_classes = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
worst_classes = df_classes.sort_values(by='f1-score', ascending=True).head(7)

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=worst_classes['f1-score'], y=worst_classes.index, palette='Reds_r')
plt.title('Gatunki o najniższym F1-Score (Zbiór TESTOWY)', fontsize=14, fontweight='bold')
plt.xlabel('F1-Score', fontsize=12)
plt.ylabel('Gatunek', fontsize=12)
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(worst_classes['f1-score']):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'real_test_f1_scores_worst.png'), dpi=300)
plt.close()
print("Zapisano: 'real_test_f1_scores_worst.png'")


cm = confusion_matrix(y_true, y_pred)
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0) 

flat_indices = np.argsort(cm_errors, axis=None)[::-1]
top_confused_indices = set()

for idx in flat_indices:
    if cm_errors.flat[idx] > 0:
        row, col = np.unravel_index(idx, cm_errors.shape)
        top_confused_indices.add(row)
        top_confused_indices.add(col)
    if len(top_confused_indices) >= 5: 
        break

top_confused_indices = list(top_confused_indices)
sub_cm = cm[np.ix_(top_confused_indices, top_confused_indices)]
sub_class_names = [class_names[i] for i in top_confused_indices]
short_names = [f"{name.split()[0][0]}. {name.split()[1]}" if len(name.split()) > 1 else name for name in sub_class_names]

plt.figure(figsize=(8, 6))
sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=short_names, yticklabels=short_names, 
            cbar_kws={'label': 'Liczba predykcji'}, annot_kws={"size": 12, "weight": "bold"})
plt.title('Najczęściej mylone gatunki (Zbiór TESTOWY)', fontsize=14, fontweight='bold')
plt.xlabel('Predykcja sieci (AI)', fontsize=12, fontweight='bold')
plt.ylabel('Rzeczywisty gatunek', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'real_test_confusion_matrix.png'), dpi=300)
plt.close()
print("Zapisano: 'real_test_confusion_matrix.png'")