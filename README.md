# BDRoad-Sense Dataset

## Dataset Structure

```
BDRoad-Sense/
│── Original_images/    # Resized original images (1024×1024)
│── augmented_images/   # Augmented images
│── metadata.csv
```

---

## Data Format

* Image format: .JPG
* Resolution: 1024 × 1024
* Original images: 6,350 (resized)
* Total images: 12,687 (including augmented)
* Metadata: CSV file with image path, class label, location, area type, and device

---

## License

This dataset is released under the **[specify license, e.g., CC BY 4.0]**.

---

## How to Download

1. Go to the repository: https://data.mendeley.com/datasets/z3nx8n396g/2
2. Download & use


## How to Use

1. Extract/download the dataset
2. Load `metadata.csv`
3. Use image paths to access images
4. Train/test models using preferred frameworks (e.g., PyTorch, TensorFlow)
