import os
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


# ======================================================
# CONFIGURATION
# ======================================================

DATASET_PATH = r"F:\Projects\Road_damage_classification\model_train"
OUTPUT_PATH = r"F:\Projects\Road_damage_classification\model_train_result"

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ======================================================
# TRANSFORMS
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ======================================================
# LOAD DATASET
# ======================================================

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes
num_classes = len(class_names)

targets = np.array(dataset.targets)

train_idx, temp_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.30,
    stratify=targets,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=targets[temp_idx],
    random_state=42
)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset   = torch.utils.data.Subset(dataset, val_idx)
test_dataset  = torch.utils.data.Subset(dataset, test_idx)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)


# ======================================================
# MODELS
# ======================================================

def get_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),

        nn.Linear(64*56*56, 256),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(256, num_classes)
    )


def get_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_vit():
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )


def get_swin():
    return timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes
    )


models_dict = {

    #"CNN": get_cnn(),

    #"ResNet18": get_resnet18(),

    "ViT": get_vit(),

    "Swin": get_swin()
}


# ======================================================
# PLOT TRAINING CURVES
# ======================================================

def plot_training_curves(name, train_losses, val_losses, train_accs, val_accs):

    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        os.path.join(OUTPUT_PATH, f"{name}_training_curves.png")
    )

    plt.close()


# ======================================================
# PLOT ROC CURVE
# ======================================================

def plot_roc_curve(name, all_labels, all_probs):

    y_true = label_binarize(all_labels, classes=range(num_classes))
    y_score = np.array(all_probs)

    plt.figure()

    for i in range(num_classes):

        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],'k--')

    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.savefig(
        os.path.join(OUTPUT_PATH, f"{name}_roc_curve.png")
    )

    plt.close()


# ======================================================
# TRAIN FUNCTION WITH EARLY STOPPING
# ======================================================

def train_model(model, name):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_val_acc = 0

    # Early stopping variables
    patience = 5
    patience_counter = 0
    best_weights = None

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(1)

            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total


        # VALIDATION
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)


        print(
            f"{name} | Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


        # Early stopping logic
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0
            best_weights = model.state_dict()

            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_PATH, f"{name}_best.pth")
            )

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(f"Early stopping triggered for {name}")
                break


    # Restore best model
    if best_weights is not None:

        model.load_state_dict(best_weights)


    plot_training_curves(
        name,
        train_losses,
        val_losses,
        train_accs,
        val_accs
    )

    evaluate_model(model, name)


# ======================================================
# EVALUATION
# ======================================================

def evaluate_model(model, name):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())


    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    with open(
        os.path.join(OUTPUT_PATH, f"{name}_report.txt"),
        "w"
    ) as f:
        f.write(report)


    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{name} Confusion Matrix")

    plt.savefig(
        os.path.join(OUTPUT_PATH, f"{name}_cm.png")
    )

    plt.close()


    plot_roc_curve(name, all_labels, all_probs)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("Using device:", DEVICE)

    for name, model in models_dict.items():

        print(f"\nTraining {name}")

        train_model(model, name)

    print("\nTraining Completed")
