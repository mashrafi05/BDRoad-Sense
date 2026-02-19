import os
from PIL import Image

def resize_if_larger_than_1024(input_dir, output_dir, max_size=1024):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if filename.lower().endswith(('.jpg', '.jpeg')):
            try:
                with Image.open(file_path) as img:

                    width, height = img.size

                    # Resize only if width or height > 1024
                    if width > max_size or height > max_size:
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        print(f"🔽 Resized: {filename} ({width}x{height} → {img.size})")
                    else:
                        print(f"✔ Kept Original: {filename}")

                    new_filename = os.path.splitext(filename)[0] + ".jpg"
                    output_path = os.path.join(output_dir, new_filename)

                    img.save(output_path, "JPEG", quality=95)

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\mashr\Downloads\revised major-20260217T194317Z-1-001\revised major"
    output_directory = r"C:\Users\mashr\OneDrive\Desktop\1024px major damage"

    resize_if_larger_than_1024(input_directory, output_directory)
