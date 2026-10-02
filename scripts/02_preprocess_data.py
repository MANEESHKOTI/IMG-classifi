import os
import shutil
import json
from sklearn.model_selection import train_test_split
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def preprocess_data():
    if not os.path.exists(RAW_DATA_DIR):
        print("Raw data directory not found. Please run 01_fetch_data.py first.")
        return

    # Scan the data/raw/ directory
    species_dirs = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    species_dirs.sort()
    
    if not species_dirs:
        print("No species data found in data/raw/")
        return

    # Create class mapping
    class_map = {species: idx for idx, species in enumerate(species_dirs)}
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    with open(os.path.join(PROCESSED_DATA_DIR, 'class_map.json'), 'w') as f:
        json.dump(class_map, f, indent=4)
        
    print(f"Created class map with {len(class_map)} species.")

    # Create processed directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for species in species_dirs:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, species), exist_ok=True)

    # Stratified split for each species
    for species in species_dirs:
        species_path = os.path.join(RAW_DATA_DIR, species)
        images = [f for f in os.listdir(species_path) if f.endswith('.jpg')]
        
        if not images:
            continue
            
        # 70% train, 15% val, 15% test
        # First split into train and temp (70 / 30)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        # Then split temp into val and test (50 / 50 -> 15% / 15% overall)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        splits_dict = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split, imgs in splits_dict.items():
            for img in imgs:
                src = os.path.join(species_path, img)
                dst = os.path.join(PROCESSED_DATA_DIR, split, species, img)
                shutil.copy2(src, dst)
                
        print(f"Split {species}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
    print("Data preprocessing complete.")

if __name__ == '__main__':
    preprocess_data()
