Augmentation code used:

import os
import cv2
import numpy as np
import random

# ----------------------------
# 🔹 Basic Augmentation Functions
# ----------------------------

def random_brightness_contrast(image):
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-30, 30)    # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def slight_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def random_rotation(image):
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# ----------------------------
# 🔹 Augmentation Pipeline
# ----------------------------

def apply_augmentations(image, include_rotation=False):
    image = random_brightness_contrast(image)
    image = slight_gaussian_blur(image)
    if include_rotation:
        image = random_rotation(image)
    return image

# ----------------------------
# 🔹 Class-wise Augmentation
# ----------------------------

def process_folder(input_dir, output_dir, num_augments, include_rotation=False):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):

            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            base_name = os.path.splitext(filename)[0]

            for i in range(num_augments):
                augmented = apply_augmentations(image, include_rotation)
                save_path = os.path.join(output_dir, f"{base_name}_aug{i+1}.jpg")
                cv2.imwrite(save_path, augmented)

            print(f"✅ {filename} → {num_augments} augmented images generated")

# ----------------------------
# 🔹 MAIN
# ----------------------------

if __name__ == "__main__":

    # 🔹 INPUT FOLDERS
    major_input = r"C:\Users\mashr\OneDrive\Desktop\Road Final DATA\1024px major damage(2062)"
    manhole_input = r"C:\Users\mashr\OneDrive\Desktop\Road Final DATA\1024px Manhole(555)"
    minor_input = r"C:\Users\mashr\OneDrive\Desktop\Road Final DATA\1024px Minor Damage(1468)"
    normal_input = r"C:\Users\mashr\OneDrive\Desktop\Road Final DATA\1024px Normal Road(1326)"
    speed_input = r"C:\Users\mashr\OneDrive\Desktop\Road Final DATA\1024px Speed breaker(939)"

    # 🔹 OUTPUT FOLDERS
    major_output = r"C:\Users\mashr\OneDrive\Desktop\Road Damage Augmented dataset\Augmented Major Damage(2062)"
    manhole_output = r"C:\Users\mashr\OneDrive\Desktop\Road Damage Augmented dataset\Augmented Manhole(2220)"
    minor_output = r"C:\Users\mashr\OneDrive\Desktop\Road Damage Augmented dataset\Augmented Minor Damage(2936)"
    normal_output = r"C:\Users\mashr\OneDrive\Desktop\Road Damage Augmented dataset\Augmented Normal Road(2652)"
    speed_output = r"C:\Users\mashr\OneDrive\Desktop\Road Damage Augmented dataset\Augmented Speed Breaker(2817)"

    # 🔹 Apply Augmentation Rules

    # Major → 1x
    process_folder(major_input, major_output, num_augments=1, include_rotation=False)

    # Manhole → 4x (includes rotation)
    process_folder(manhole_input, manhole_output, num_augments=4, include_rotation=True)

    # Minor → 2x
    process_folder(minor_input, minor_output, num_augments=2, include_rotation=False)

    # Normal → 2x
    process_folder(normal_input, normal_output, num_augments=2, include_rotation=False)

    # Speed Breaker → 3x
    process_folder(speed_input, speed_output, num_augments=3, include_rotation=False)

    print("\n🎉 All Class-wise Augmentations Completed!")
