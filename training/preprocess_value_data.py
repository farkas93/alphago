# training/preprocess_value_data.py
"""
Preprocess SGF files for Value Network training
Extracts (board_state, game_outcome) pairs
"""
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.sgf_parser_value import parse_sgf_for_value
from game.features import FeatureExtractor, augment_features
from config import DATA_DIR, PROCESSED_DIR, BOARD_SIZE

def preprocess_value_data(chunk_size=50000, sample_rate=0.3, use_augmentation=False):
    """
    Process SGF files for value network training
    
    Args:
        chunk_size: Number of samples per chunk file
        sample_rate: Fraction of positions to sample from each game (0.3 = 30%)
                    Lower rate prevents overfitting to single games
        use_augmentation: Whether to use 8x augmentation (warning: 8x more data!)
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
    print(f"Sample rate: {sample_rate} (sampling {sample_rate*100:.0f}% of positions per game)")
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
    
    # Outcome distribution tracking
    wins = 0
    losses = 0
    draws = 0

    for sgf_file in tqdm(all_sgf_files, desc="Processing SGF files"):
        samples = parse_sgf_for_value(str(sgf_file), extractor, sample_rate=sample_rate)

        if not samples:
            games_skipped += 1
            continue
        
        games_processed += 1

        for features, outcome in samples:
            # Track outcome distribution
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1
            
            if use_augmentation:
                # Augment with 8 symmetries
                augmented_data = augment_features(features, BOARD_SIZE)
                
                for aug_features, transform_fn in augmented_data:
                    chunk_features.append(aug_features)
                    chunk_labels.append(outcome)  # Outcome doesn't change with symmetry
            else:
                # No augmentation
                chunk_features.append(features)
                chunk_labels.append(outcome)
            
            # Save chunk when it reaches chunk_size
            if len(chunk_features) >= chunk_size:
                save_value_chunk(chunk_features, chunk_labels, chunk_idx)
                total_samples += len(chunk_features)
                chunk_features = []
                chunk_labels = []
                chunk_idx += 1

    # Save final chunk
    if chunk_features:
        save_value_chunk(chunk_features, chunk_labels, chunk_idx)
        total_samples += len(chunk_features)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Games processed: {games_processed}")
    print(f"Games skipped: {games_skipped}")
    print(f"Total chunks saved: {chunk_idx + 1}")
    print(f"Total training samples: {total_samples}")
    print(f"\nOutcome distribution:")
    total_outcomes = wins + losses + draws
    if total_outcomes > 0:
        print(f"  Wins:   {wins} ({100*wins/total_outcomes:.1f}%)")
        print(f"  Losses: {losses} ({100*losses/total_outcomes:.1f}%)")
        print(f"  Draws:  {draws} ({100*draws/total_outcomes:.1f}%)")
    print(f"{'='*60}\n")

def save_value_chunk(features, labels, chunk_idx):
    """Save a chunk of value network training data"""
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)  # Note: float32 for regression
    
    output_path = os.path.join(
        PROCESSED_DIR, 
        f"value_data_{BOARD_SIZE}x{BOARD_SIZE}_chunk{chunk_idx:03d}.npz"
    )
    np.savez_compressed(output_path, features=features_np, labels=labels_np)
    
    # Calculate size
    size_mb = (features_np.nbytes + labels_np.nbytes) / (1024 * 1024)
    print(f"\nSaved value chunk {chunk_idx}: {len(features)} samples ({size_mb:.1f} MB)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Value Network data')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Number of samples per chunk (default: 50000)')
    parser.add_argument('--sample-rate', type=float, default=0.3,
                        help='Fraction of positions to sample per game (default: 0.3)')
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable 8x data augmentation')
    
    args = parser.parse_args()
    
    preprocess_value_data(
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        use_augmentation=args.augmentation
    )