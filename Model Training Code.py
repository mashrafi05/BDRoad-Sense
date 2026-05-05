import os
import csv
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

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# ======================================================
# CONFIGURATION
# ======================================================

TRAIN_PATH  = r"I:\Road Damage Detection\Data\BDRoad-Sense\augmented_train"
VAL_PATH    = r"I:\Road Damage Detection\Data\BDRoad-Sense\val"
TEST_PATH   = r"I:\Road Damage Detection\Data\BDRoad-Sense\test"
OUTPUT_PATH = r"I:\Road Damage Detection\Data\Results"

BATCH_SIZE  = 32          # default for CNN and ResNet18

# ViT and Swin use smaller batches to avoid OOM — change to 4 if still crashing
MODEL_BATCH_SIZES = {
    "CNN":      32,
    "ResNet18": 32,
    "ViT":      8,
    "Swin":     8,
}

EPOCHS      = 30
LR          = 1e-4
PATIENCE    = 6
NUM_RUNS    = 3                          # number of times each model is trained

# Seeds for each run — different seeds = different random init & shuffle
RUN_SEEDS   = [42, 123, 777]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---- Output sub-folders ----
CM_CSV_DIR  = os.path.join(OUTPUT_PATH, "confusion_matrices")   # CSV files
CKPT_DIR    = os.path.join(OUTPUT_PATH, "checkpoints")          # resume checkpoints
RUNS_DIR    = os.path.join(OUTPUT_PATH, "runs")                 # per-run plots & reports

for d in [OUTPUT_PATH, CM_CSV_DIR, CKPT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)


# ======================================================
# SEED HELPER
# ======================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ======================================================
# TRANSFORMS
# ======================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ======================================================
# DATASETS
# ======================================================

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_PATH,   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(TEST_PATH,  transform=val_test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Device      : {DEVICE}")
print(f"Classes     : {class_names}")
print(f"Train imgs  : {len(train_dataset)}")
print(f"Val imgs    : {len(val_dataset)}")
print(f"Test imgs   : {len(test_dataset)}")
print(f"Runs        : {NUM_RUNS}  (seeds: {RUN_SEEDS})\n")
print("Batch sizes per model:")
for mname, bs in MODEL_BATCH_SIZES.items():
    print(f"  {mname:<12} {bs}")
print()


# ======================================================
# DATA LOADERS
# ======================================================

def get_train_loader(model_name):
    """Returns a train DataLoader with the correct batch size for the given model."""
    bs = MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE)
    return DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

# Val and test loaders are shared — small batch is fine for inference
val_loader = DataLoader(val_dataset,  batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)


# ======================================================
# CLASS WEIGHTS
# ======================================================

class_counts = np.array([
    len([f for f in os.listdir(os.path.join(TRAIN_PATH, c))
         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"))])
    for c in class_names
])
total = class_counts.sum()
class_weights = torch.tensor(
    total / (num_classes * class_counts), dtype=torch.float32
).to(DEVICE)

print("Class weights:")
for cname, w in zip(class_names, class_weights.cpu().numpy()):
    print(f"  {cname:<20} {w:.4f}")
print()


# ======================================================
# MODEL DEFINITIONS
# ======================================================

def get_cnn(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

def get_resnet18(num_classes):
    m = models.resnet18(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_vit(num_classes):
    return timm.create_model("vit_base_patch16_224",
                              pretrained=True, num_classes=num_classes)

def get_swin(num_classes):
    return timm.create_model("swin_base_patch4_window7_224",
                              pretrained=True, num_classes=num_classes)

MODEL_BUILDERS = {
    "CNN":      get_cnn,
    "ResNet18": get_resnet18,
    "ViT":      get_vit,
    "Swin":     get_swin,
}


# ======================================================
# CHECKPOINT HELPERS
# ======================================================

def ckpt_path(name, run):
    return os.path.join(CKPT_DIR, f"{name}_run{run}_resume.ckpt")

def save_checkpoint(name, run, epoch, model, optimizer, scheduler,
                    best_val_acc, patience_counter,
                    train_losses, val_losses, train_accs, val_accs):
    torch.save({
        "epoch": epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_val_acc":     best_val_acc,
        "patience_counter": patience_counter,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "train_accs":       train_accs,
        "val_accs":         val_accs,
    }, ckpt_path(name, run))

def load_checkpoint(name, run, model, optimizer, scheduler):
    path = ckpt_path(name, run)
    if not os.path.isfile(path):
        print(f"    [Checkpoint] No checkpoint — starting from epoch 1.")
        return 0, 0.0, 0, [], [], [], []
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"    [Checkpoint] Resumed from epoch {ckpt['epoch']+1}  "
          f"(best_val_acc={ckpt['best_val_acc']:.4f})")
    return (ckpt["epoch"] + 1, ckpt["best_val_acc"], ckpt["patience_counter"],
            ckpt["train_losses"], ckpt["val_losses"],
            ckpt["train_accs"],   ckpt["val_accs"])

def delete_checkpoint(name, run):
    path = ckpt_path(name, run)
    if os.path.isfile(path):
        os.remove(path)

def run_best_pth(name, run):
    return os.path.join(OUTPUT_PATH, f"{name}_run{run}_best.pth")

def run_is_done(name, run):
    """True if this specific run already completed."""
    return os.path.isfile(run_best_pth(name, run)) and \
           os.path.isfile(os.path.join(RUNS_DIR, f"{name}_run{run}_report.txt"))


# ======================================================
# PLOTS  (saved into RUNS_DIR with run index)
# ======================================================

def plot_training_curves(name, run, train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.title(f"{name} Run {run} — Loss"); plt.xlabel("Epoch")
    plt.ylabel("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs,   label="Val Accuracy")
    plt.title(f"{name} Run {run} — Accuracy"); plt.xlabel("Epoch")
    plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, f"{name}_run{run}_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(name, run, cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} Run {run} — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, f"{name}_run{run}_cm.png"), dpi=150)
    plt.close()


def plot_roc_curve(name, run, all_labels, all_probs):
    y_true  = label_binarize(all_labels, classes=range(num_classes))
    y_score = np.array(all_probs)
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.title(f"{name} Run {run} — ROC Curve")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, f"{name}_run{run}_roc.png"), dpi=150)
    plt.close()


# ======================================================
# SAVE CONFUSION MATRIX AS CSV
# ======================================================

def save_cm_csv(name, cm):
    """
    Saves the confusion matrix to /confusion_matrices/{name}_confusion_matrix.csv
    Rows = True class, Columns = Predicted class.
    This is the MEAN confusion matrix across all runs (rounded to int).
    """
    path = os.path.join(CM_CSV_DIR, f"{name}_confusion_matrix.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True \\ Predicted"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + [int(v) for v in row])
    print(f"  [CSV] Confusion matrix saved → {path}")


# ======================================================
# EVALUATE  (returns acc, cm, labels, probs)
# ======================================================

def evaluate_model(model, name, run):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)

    print(f"    Test Accuracy : {overall_acc*100:.2f}%")
    print(report)

    with open(os.path.join(RUNS_DIR, f"{name}_run{run}_report.txt"), "w") as f:
        f.write(f"Model : {name}  |  Run : {run}\n")
        f.write(f"Test Accuracy : {overall_acc*100:.2f}%\n\n")
        f.write(report)

    plot_confusion_matrix(name, run, cm)
    plot_roc_curve(name, run, all_labels, all_probs)

    return overall_acc, cm


# ======================================================
# TRAIN ONE RUN
# ======================================================

def train_one_run(name, run, seed):
    set_seed(seed)

    # Build the train loader with the correct batch size for this model
    train_loader = get_train_loader(name)

    model     = MODEL_BUILDERS[name](num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    (start_epoch, best_val_acc, patience_counter,
     train_losses, val_losses,
     train_accs,   val_accs) = load_checkpoint(name, run, model, optimizer, scheduler)

    best_weights = None
    best_pth     = run_best_pth(name, run)

    if start_epoch > 0 and os.path.isfile(best_pth):
        best_weights = torch.load(best_pth, map_location=DEVICE)
        print(f"    [Checkpoint] Best weights reloaded.")

    bs = MODEL_BATCH_SIZES.get(name, BATCH_SIZE)
    print(f"    Batch size : {bs}")

    for epoch in range(start_epoch, EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = train_correct / train_total

        # ---- VALIDATE ----
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"    Epoch {epoch+1:>3}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_pth)
            print(f"      ✓ Best saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                save_checkpoint(name, run, epoch, model, optimizer, scheduler,
                                best_val_acc, patience_counter,
                                train_losses, val_losses, train_accs, val_accs)
                break

        save_checkpoint(name, run, epoch, model, optimizer, scheduler,
                        best_val_acc, patience_counter,
                        train_losses, val_losses, train_accs, val_accs)

    if best_weights:
        model.load_state_dict(best_weights)

    plot_training_curves(name, run, train_losses, val_losses, train_accs, val_accs)

    print(f"\n    Evaluating {name} Run {run} on test set ...")
    test_acc, cm = evaluate_model(model, name, run)

    delete_checkpoint(name, run)
    return test_acc, cm


# ======================================================
# FINAL PLOTS — mean CM heatmap & comparison bar chart
# ======================================================

def plot_mean_cm(name, mean_cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} — Mean Confusion Matrix ({NUM_RUNS} runs)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"{name}_mean_cm.png"), dpi=150)
    plt.close()


def plot_final_comparison(summary):
    """Bar chart: mean ± std for all models."""
    names  = list(summary.keys())
    means  = [summary[n]["mean"] * 100 for n in names]
    stds   = [summary[n]["std"]  * 100 for n in names]
    colors = ["steelblue", "coral", "seagreen", "orchid"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=8,
                   color=colors, edgecolor="black", width=0.5,
                   error_kw={"elinewidth": 2, "ecolor": "black"})
    for bar, m, s in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.8,
                 f"{m:.2f}%\n±{s:.2f}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.ylim(0, 115)
    plt.title(f"Model Comparison — Mean ± Std over {NUM_RUNS} Runs", fontsize=13)
    plt.ylabel("Test Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison_mean_std.png"), dpi=150)
    plt.close()


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print(f"Using device : {DEVICE}")
    print(f"Output       : {OUTPUT_PATH}")
    print(f"CM CSVs      : {CM_CSV_DIR}\n")

    # summary[model_name] = {"runs": [...], "mean": x, "std": x, "cms": [...]}
    summary = {}

    for name in MODEL_BUILDERS:

        print(f"\n{'#'*65}")
        print(f"#  MODEL : {name}  (batch size: {MODEL_BATCH_SIZES.get(name, BATCH_SIZE)})")
        print(f"{'#'*65}")

        run_accs = []
        run_cms  = []

        for run in range(1, NUM_RUNS + 1):
            seed = RUN_SEEDS[run - 1]

            print(f"\n  ── Run {run}/{NUM_RUNS}  (seed={seed}) ──")

            if run_is_done(name, run):
                # Run already completed — reload saved acc from report
                report_path = os.path.join(RUNS_DIR, f"{name}_run{run}_report.txt")
                with open(report_path) as f:
                    for line in f:
                        if line.startswith("Test Accuracy"):
                            acc = float(line.split(":")[1].strip().replace("%", "")) / 100
                            break
                model = MODEL_BUILDERS[name](num_classes).to(DEVICE)
                pth   = run_best_pth(name, run)
                model.load_state_dict(torch.load(pth, map_location=DEVICE))
                _, cm = evaluate_model(model, name, run)
                print(f"    [SKIP] Run {run} already done — acc={acc*100:.2f}%")
                run_accs.append(acc)
                run_cms.append(cm)
                continue

            acc, cm = train_one_run(name, run, seed)
            run_accs.append(acc)
            run_cms.append(cm)

        # ---- Stats ----
        mean_acc = np.mean(run_accs)
        std_acc  = np.std(run_accs)
        mean_cm  = np.mean(run_cms, axis=0)

        summary[name] = {
            "runs": run_accs,
            "mean": mean_acc,
            "std":  std_acc,
            "mean_cm": mean_cm,
        }

        # ---- Print per-run + stats ----
        print(f"\n  ┌─ {name} Results ────────────────────────")
        for i, acc in enumerate(run_accs, 1):
            print(f"  │  Run {i} : {acc*100:.4f}%")
        print(f"  │  Mean  : {mean_acc*100:.4f}%")
        print(f"  │  Std   : {std_acc*100:.4f}%")
        print(f"  └──────────────────────────────────────────")

        # ---- Save mean CM heatmap ----
        plot_mean_cm(name, mean_cm)

        # ---- Save mean CM as CSV ----
        save_cm_csv(name, mean_cm)

        # ---- Save best .pth as the run with highest accuracy ----
        best_run = int(np.argmax(run_accs)) + 1
        best_src = run_best_pth(name, best_run)
        best_dst = os.path.join(OUTPUT_PATH, f"{name}_best.pth")
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"  Best overall model (run {best_run}) → {name}_best.pth")

    # ======================================================
    # FINAL SUMMARY TABLE
    # ======================================================

    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY  ({NUM_RUNS} runs each)")
    print(f"{'='*65}")
    header = f"  {'Model':<12}" + "".join(f"  Run{i:<6}" for i in range(1, NUM_RUNS+1))
    header += f"  {'Mean':>8}   {'Std':>8}"
    print(header)
    print("-" * 65)

    for name, stats in summary.items():
        row = f"  {name:<12}"
        for acc in stats["runs"]:
            row += f"  {acc*100:>6.2f}%"
        row += f"  {stats['mean']*100:>7.2f}%   {stats['std']*100:>7.4f}%"
        print(row)

    print("=" * 65)

    # Save summary table as CSV
    summary_csv = os.path.join(OUTPUT_PATH, "training_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model"] + [f"Run_{i}" for i in range(1, NUM_RUNS+1)] + ["Mean_%", "Std_%"]
        )
        for name, stats in summary.items():
            writer.writerow(
                [name]
                + [f"{a*100:.4f}" for a in stats["runs"]]
                + [f"{stats['mean']*100:.4f}", f"{stats['std']*100:.4f}"]
            )
    print(f"\n  Summary CSV saved → {summary_csv}")

    # Final comparison bar chart
    plot_final_comparison(summary)

    print(f"\nAll results saved to : {OUTPUT_PATH}")
    print(f"Confusion matrix CSVs: {CM_CSV_DIR}")
    print("Training Complete.")