# training/preprocess_data.py - Memory-efficient version
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.sgf_parser import parse_sgf_file
from game.features import FeatureExtractor, augment_features
from config import DATA_DIR, PROCESSED_DIR, BOARD_SIZE

def preprocess_data(chunk_size=50000, use_augmentation=True):
    """
    Process SGF files in chunks to avoid memory issues
    
    Args:
        chunk_size: Number of samples per chunk file
        use_augmentation: If False, don't use 8x augmentation (saves memory)
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
    print(f"Augmentation: {'Enabled (8x)' if use_augmentation else 'Disabled'}")
    print(f"Chunk size: {chunk_size} samples")
    
    # Collect all SGF files
    all_sgf_files = []
    for folder in kgs_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        sgf_files = list(Path(folder_path).rglob("*.sgf"))
        all_sgf_files.extend(sgf_files)
        print(f"  {folder}: {len(sgf_files)} SGF files")
    
    print(f"\nTotal SGF files found: {len(all_sgf_files)}")

    extractor = FeatureExtractor()
    
    # Chunk tracking
    chunk_features = []
    chunk_labels = []
    chunk_idx = 0
    total_samples = 0
    
    games_processed = 0
    games_skipped = 0

    for sgf_file in tqdm(all_sgf_files, desc="Processing SGF files"):
        samples = parse_sgf_file(str(sgf_file), extractor)

        if not samples:
            games_skipped += 1
            continue
        
        games_processed += 1

        for features, move in samples:
            if use_augmentation:
                # Augment with 8 symmetries
                augmented_data = augment_features(features, BOARD_SIZE)
                
                for aug_features, transform_fn in augmented_data:
                    aug_move = transform_fn(move)
                    label = aug_move[0] * BOARD_SIZE + aug_move[1]
                    
                    chunk_features.append(aug_features)
                    chunk_labels.append(label)
            else:
                # No augmentation - just use original
                label = move[0] * BOARD_SIZE + move[1]
                chunk_features.append(features)
                chunk_labels.append(label)
            
            # Save chunk when it reaches chunk_size
            if len(chunk_features) >= chunk_size:
                save_chunk(chunk_features, chunk_labels, chunk_idx)
                total_samples += len(chunk_features)
                chunk_features = []
                chunk_labels = []
                chunk_idx += 1

    # Save final chunk
    if chunk_features:
        save_chunk(chunk_features, chunk_labels, chunk_idx)
        total_samples += len(chunk_features)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Games processed: {games_processed}")
    print(f"Games skipped: {games_skipped}")
    print(f"Total chunks saved: {chunk_idx + 1}")
    print(f"Total training samples: {total_samples}")
    print(f"Samples per chunk: ~{chunk_size}")
    print(f"{'='*60}\n")

def save_chunk(features, labels, chunk_idx):
    """Save a chunk of processed data"""
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)
    
    output_path = os.path.join(
        PROCESSED_DIR, 
        f"supervised_data_{BOARD_SIZE}x{BOARD_SIZE}_chunk{chunk_idx:03d}.npz"
    )
    np.savez_compressed(output_path, features=features_np, labels=labels_np)
    
    # Calculate size
    size_mb = (features_np.nbytes + labels_np.nbytes) / (1024 * 1024)
    print(f"\nSaved chunk {chunk_idx}: {len(features)} samples ({size_mb:.1f} MB)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Go game data')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Number of samples per chunk (default: 50000)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable 8x data augmentation to save memory')
    
    args = parser.parse_args()
    
    preprocess_data(
        chunk_size=args.chunk_size,
        use_augmentation=not args.no_augmentation
    )