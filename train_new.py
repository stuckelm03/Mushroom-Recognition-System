import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"E:\Studia\Grzyby\dataset_split" 
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "mushroom_model.pth")
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 15
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"--- Uruchamianie treningu (Seed: {SEED}) ---")


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train')) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sprzęt użyty do uczenia: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

since = time.time()
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoka {epoch+1}/{NUM_EPOCHS}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
        else:
            model.eval()   

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
        else:
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc.item())

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f"*** Nowy najlepszy model! Zapisywanie wag... ***")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

time_elapsed = time.time() - since
print(f'\nTrening zakończony w {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

print("Generowanie wykresów z przebiegu uczenia...")
epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Trening Loss', marker='o')
plt.plot(epochs_range, history['val_loss'], label='Walidacja Loss', marker='o')
plt.title('Funkcja straty (Loss)')
plt.xlabel('Epoka')
plt.ylabel('Loss (Cross Entropy)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, np.array(history['train_acc']) * 100, label='Trening Accuracy', marker='s', color='green')
plt.plot(epochs_range, np.array(history['val_acc']) * 100, label='Walidacja Accuracy', marker='s', color='red')
plt.title('Dokładność (Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Skuteczność (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_curves.png'), dpi=300)
plt.close()
print("Zapisano wykres do pliku 'training_curves.png'. Gotowe!")