import os
import cv2
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# pip install albumentations opencv-python

import albumentations as A

# ======================================================
# CONFIGURATION
# ======================================================

SOURCE_DIR = r"I:\Road Damage Detection\Data\BDRoad Sense - Split\train"
OUTPUT_DIR = r"I:\Road Damage Detection\Data\BDRoad Sense - Split\train_augmented"

SEED = 42

# Target image count per class
CLASS_TARGETS = {
    "Major Damage":  1500,
    "Minor Damage":  1500,
    "Normal Road":   1500,
    "Speed Breaker": 1400,
    "Manhole":       1300,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

random.seed(SEED)
np.random.seed(SEED)

# ======================================================
# AUGMENTATION PIPELINE
# High diversity — each call returns a DIFFERENT result
# ======================================================

def get_augmentation_pipeline():
    return A.Compose([

        # --- Geometric ---
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(1024, 1024),
            scale=(0.85, 1.0),
            p=0.4
        ),

        # --- Color / Brightness ---
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
            p=0.8
        ),

        # --- Blur (occasional) ---
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.25),

        # --- Road texture distortion ---
        A.OneOf([
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
            A.ElasticTransform(
                alpha=80,
                sigma=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
        ], p=0.5),

        # --- Noise (occasional) ---
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),

        # Final resize to ensure all outputs are 1024x1024
        A.Resize(1024, 1024),
    ])


# ======================================================
# HELPER — load image
# ======================================================

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ======================================================
# MAIN
# ======================================================

def augment_dataset():
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    print(f"Source : {SOURCE_DIR}")
    print(f"Output : {OUTPUT_DIR}\n")

    summary = defaultdict(lambda: {"original": 0, "augmented": 0, "total": 0})

    for cls_name, target in CLASS_TARGETS.items():
        cls_src  = source_path / cls_name
        cls_dest = output_path / cls_name
        cls_dest.mkdir(parents=True, exist_ok=True)

        if not cls_src.exists():
            print(f"[WARNING] Class folder not found: {cls_src}")
            continue

        # Collect source images
        images = sorted([
            f for f in cls_src.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        n_original = len(images)
        summary[cls_name]["original"] = n_original

        if n_original == 0:
            print(f"[WARNING] No images found for class: {cls_name}")
            continue

        # --- Step 1: Copy all originals to output ---
        for img_path in images:
            dest = cls_dest / img_path.name
            if not dest.exists():
                img = load_image(img_path)
                save_image(img, dest)

        # --- Step 2: Augment until target is reached ---
        n_needed = max(0, target - n_original)
        aug_count = 0

        if n_needed == 0:
            print(f"  {cls_name:<20} already at/above target ({n_original}), only originals copied.")
            summary[cls_name]["augmented"] = 0
            summary[cls_name]["total"] = n_original
            continue

        print(f"  {cls_name:<20} originals={n_original}  need={n_needed}  target={target}")

        transform = get_augmentation_pipeline()

        # Cycle through source images repeatedly
        image_cycle = images * (n_needed // n_original + 2)
        random.shuffle(image_cycle)

        for i, img_path in enumerate(image_cycle):
            if aug_count >= n_needed:
                break

            try:
                img = load_image(img_path)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            augmented = transform(image=img)["image"]

            aug_filename = f"aug_{aug_count:05d}_{img_path.stem}{img_path.suffix}"
            save_image(augmented, cls_dest / aug_filename)
            aug_count += 1

        summary[cls_name]["augmented"] = aug_count
        summary[cls_name]["total"]     = n_original + aug_count

    # ======================================================
    # SUMMARY TABLE
    # ======================================================

    print(f"\n{'='*65}")
    print(f"  AUGMENTATION SUMMARY")
    print(f"{'='*65}")
    print(f"{'Class':<22} {'Original':>9} {'Augmented':>10} {'Total':>7} {'Target':>7}")
    print("-" * 65)

    grand_orig = grand_aug = grand_total = 0
    for cls_name, counts in summary.items():
        target = CLASS_TARGETS.get(cls_name, "-")
        print(
            f"  {cls_name:<20} {counts['original']:>9} "
            f"{counts['augmented']:>10} {counts['total']:>7} {target:>7}"
        )
        grand_orig  += counts["original"]
        grand_aug   += counts["augmented"]
        grand_total += counts["total"]

    print("-" * 65)
    print(f"  {'TOTAL':<20} {grand_orig:>9} {grand_aug:>10} {grand_total:>7}")
    print("=" * 65)
    print(f"\nAugmented dataset saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    augment_dataset()