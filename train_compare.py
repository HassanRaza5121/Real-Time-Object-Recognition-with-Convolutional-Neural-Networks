# train_compare.py
import os
import time
import copy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------
# Config
# -----------------------
DATA_DIR = "dataset_split"  # change if needed
OUT_DIR = "models_out"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 4
INPUT_SIZE = 224  # for transfer learning; baseline will be resized to this as well
BATCH_SIZE = 16   # reduce to 8 or 4 if you don't have GPU/RAM
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 10

# Use seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------
# Data transforms
# -----------------------
# Basic (no augmentation) transforms for baseline validation/test
basic_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet mean/std works well even for small datasets
                         [0.229, 0.224, 0.225])
])

# Augmented transforms for Model 2 (training)
augmented_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# For validation/test (no augmentation)
val_transforms = basic_transforms

# -----------------------
# Datasets & Dataloaders
# -----------------------
def make_dataloaders(train_transform, val_transform, batch_size=BATCH_SIZE):
    data_dir = Path(DATA_DIR)
    train_ds = datasets.ImageFolder(data_dir/"train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir/"val", transform=val_transform)
    test_ds = datasets.ImageFolder(data_dir/"test", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names

# -----------------------
# Model 1: Baseline CNN
# A small custom CNN (lightweight)
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------
# Model 3: Transfer learning (MobileNetV2 recommended)
# -----------------------
def get_mobilenetv2(num_classes=NUM_CLASSES, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# -----------------------
# Training & evaluation utilities
# -----------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=NUM_EPOCHS, model_name="model"):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loader, val_loader = dataloaders['train'], dataloaders['val']

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {model_name}")
        print("-"*30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

                if i % PRINT_EVERY == 0 and i > 0:
                    print(f"[{phase}] batch {i}/{len(loader)} loss: {loss.item():.4f}")

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{model_name}_best.pth"))
                print(f"Saved best model for {model_name} with acc {best_acc:.4f}")

        if scheduler:
            scheduler.step()

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s. Best val Acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    print("\nTest classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)
    return all_labels, all_preds, cm

# -----------------------
# Main: Train Model1, Model2, Model3
# -----------------------
def main():
    # MODEL 1: Baseline (no augmentation)
    print("Preparing dataloaders for Model 1 (baseline)")
    train_loader_1, val_loader_1, test_loader_1, class_names = make_dataloaders(basic_transforms, val_transforms)
    dataloaders_1 = {'train': train_loader_1, 'val': val_loader_1}

    model1 = SmallCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
    # simple step LR
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=8, gamma=0.5)

    print("\nTraining Model 1 (SmallCNN) - baseline")
    model1 = train_model(model1, dataloaders_1, criterion, optimizer1, scheduler1, num_epochs=15, model_name="smallcnn_baseline")

    print("\nEvaluating Model 1 on test set")
    evaluate_model(model1, test_loader_1, class_names)

    # MODEL 2: Data-Augmented CNN (same architecture but augmented loader)
    print("\nPreparing dataloaders for Model 2 (data augmentation)")
    train_loader_2, val_loader_2, test_loader_2, _ = make_dataloaders(augmented_transforms, val_transforms)
    dataloaders_2 = {'train': train_loader_2, 'val': val_loader_2}

    model2 = SmallCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=8, gamma=0.5)

    print("\nTraining Model 2 (SmallCNN + augmentation)")
    model2 = train_model(model2, dataloaders_2, criterion, optimizer2, scheduler2, num_epochs=20, model_name="smallcnn_augmented")

    print("\nEvaluating Model 2 on test set")
    evaluate_model(model2, test_loader_2, class_names)

    # MODEL 3: Transfer learning (MobileNetV2)
    print("\nPreparing dataloaders for Model 3 (transfer learning)")
    train_loader_3, val_loader_3, test_loader_3, _ = make_dataloaders(augmented_transforms, val_transforms)
    dataloaders_3 = {'train': train_loader_3, 'val': val_loader_3}

    model3 = get_mobilenetv2(num_classes=len(class_names), pretrained=True).to(DEVICE)

    # Strategy: freeze features, train classifier first, then unfreeze some layers and fine-tune
    for param in model3.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer3 = optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=6, gamma=0.5)

    print("\nTraining Model 3 (MobileNetV2) - classifier only")
    model3 = train_model(model3, dataloaders_3, criterion, optimizer3, scheduler3, num_epochs=8, model_name="mobilenet_classifier_only")

    # Unfreeze last few layers for fine-tuning
    print("\nUnfreezing last MobileNet layers for fine-tuning")
    for name, param in model3.named_parameters():
        if "features.18" in name or "features.17" in name or "features.16" in name:
            param.requires_grad = True

    optimizer3_ft = optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler3_ft = optim.lr_scheduler.StepLR(optimizer3_ft, step_size=5, gamma=0.5)

    print("\nFine-tuning MobileNetV2")
    model3 = train_model(model3, dataloaders_3, criterion, optimizer3_ft, scheduler3_ft, num_epochs=12, model_name="mobilenet_finetuned")

    print("\nEvaluating Model 3 on test set")
    evaluate_model(model3, test_loader_3, class_names)

if __name__ == "__main__":
    print("Device:", DEVICE)
    main()
