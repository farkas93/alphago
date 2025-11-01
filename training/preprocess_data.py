# training/preprocess_data.py - Updated to scan data folder directly
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.sgf_parser import parse_sgf_file
from game.features import FeatureExtractor, augment_features
from config import DATA_DIR, PROCESSED_DIR, BOARD_SIZE

def preprocess_data():
    """
    Scans data/ for KGS folders, processes SGF files directly
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Find all KGS folders
    kgs_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("kgs-")]
    
    if not kgs_folders:
        print("No KGS folders found in data/")
        return
    
    print(f"Found {len(kgs_folders)} KGS archive folders")
    print(f"Target board size: {BOARD_SIZE}x{BOARD_SIZE}")
    
    # Collect all SGF files from all KGS folders
    all_sgf_files = []
    for folder in kgs_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        sgf_files = list(Path(folder_path).rglob("*.sgf"))
        all_sgf_files.extend(sgf_files)
        print(f"  {folder}: {len(sgf_files)} SGF files")
    
    print(f"\nTotal SGF files found: {len(all_sgf_files)}")

    extractor = FeatureExtractor()
    all_features = []
    all_labels = []
    
    games_processed = 0
    games_skipped = 0

    for sgf_file in tqdm(all_sgf_files, desc="Processing SGF files"):
        samples = parse_sgf_file(str(sgf_file), extractor)

        if not samples:
            games_skipped += 1
            continue
        
        games_processed += 1

        for features, move in samples:
            # Augment data with 8 symmetries
            augmented_data = augment_features(features, BOARD_SIZE)
            
            for aug_features, transform_fn in augmented_data:
                aug_move = transform_fn(move)
                
                # Convert move (row, col) to a single integer label
                label = aug_move[0] * BOARD_SIZE + aug_move[1]
                
                all_features.append(aug_features)
                all_labels.append(label)

    if not all_features:
        print("\n⚠️  No valid training samples extracted!")
        print("Make sure SGF files contain games with the target board size.")
        return

    # Convert to numpy arrays
    features_np = np.array(all_features, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.int64)

    # Save to a compressed .npz file
    output_path = os.path.join(PROCESSED_DIR, f"supervised_data_{BOARD_SIZE}x{BOARD_SIZE}.npz")
    np.savez_compressed(output_path, features=features_np, labels=labels_np)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Games processed: {games_processed}")
    print(f"Games skipped: {games_skipped}")
    print(f"Total training samples (with augmentation): {len(features_np)}")
    print(f"Features shape: {features_np.shape}")
    print(f"Labels shape: {labels_np.shape}")
    print(f"Data saved to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    preprocess_data()