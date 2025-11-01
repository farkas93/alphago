# training/data_loader.py
"""
Data loader that handles chunked .npz files
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import PROCESSED_DIR, BOARD_SIZE

class ChunkedGoDataset(Dataset):
    """
    Dataset that loads chunks on-demand to save memory
    """
    
    def __init__(self, processed_dir=PROCESSED_DIR, board_size=BOARD_SIZE):
        self.processed_dir = processed_dir
        self.board_size = board_size
        
        # Find all chunk files
        self.chunk_files = sorted([
            f for f in os.listdir(processed_dir) 
            if f.startswith(f"supervised_data_{board_size}x{board_size}_chunk") and f.endswith('.npz')
        ])
        
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {processed_dir}")
        
        print(f"Found {len(self.chunk_files)} chunk files")
        
        # Load chunk sizes
        self.chunk_sizes = []
        self.cumulative_sizes = [0]
        
        for chunk_file in self.chunk_files:
            path = os.path.join(processed_dir, chunk_file)
            data = np.load(path)
            size = len(data['labels'])
            self.chunk_sizes.append(size)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
        
        self.total_size = self.cumulative_sizes[-1]
        print(f"Total samples: {self.total_size:,}")
        
        # Cache for current chunk
        self.current_chunk_idx = None
        self.current_features = None
        self.current_labels = None
    
    def __len__(self):
        return self.total_size
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk into memory"""
        if chunk_idx != self.current_chunk_idx:
            path = os.path.join(self.processed_dir, self.chunk_files[chunk_idx])
            data = np.load(path)
            self.current_features = data['features']
            self.current_labels = data['labels']
            self.current_chunk_idx = chunk_idx
    
    def __getitem__(self, idx):
        # Find which chunk this index belongs to
        chunk_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes[1:]):
            if idx < cumsum:
                chunk_idx = i
                break
        
        # Load chunk if needed
        self._load_chunk(chunk_idx)
        
        # Get local index within chunk
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        # Get sample
        features = torch.from_numpy(self.current_features[local_idx])
        label = torch.tensor(self.current_labels[local_idx], dtype=torch.long)
        
        return features, label


def create_data_loaders(batch_size=128, train_split=0.95, num_workers=4):
    """
    Create train and validation data loaders
    
    Args:
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    dataset = ChunkedGoDataset()
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    train_loader, val_loader = create_data_loaders(batch_size=32)
    
    # Get one batch
    features, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")