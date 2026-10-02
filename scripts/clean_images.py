import os
from PIL import Image

def clean_corrupted_images(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception as e:
                    print(f"Removing corrupted image: {filepath} ({e})")
                    os.remove(filepath)
                    count += 1
    print(f"Removed {count} corrupted images.")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    clean_corrupted_images(processed_dir)
