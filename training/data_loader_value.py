# training/data_loader_value.py
"""
Streaming data loader for Value Network training
"""
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from config import PROCESSED_DIR, BOARD_SIZE

class StreamingValueDataset(IterableDataset):
    """
    Streaming dataset for value network training
    Loads (features, outcome) pairs
    """
    
    def __init__(self, processed_dir=PROCESSED_DIR, board_size=BOARD_SIZE, shuffle_chunks=True):
        super(StreamingValueDataset).__init__()
        self.processed_dir = processed_dir
        self.board_size = board_size
        self.shuffle_chunks = shuffle_chunks
        
        # Find all value chunk files
        self.chunk_files = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir) 
            if f.startswith(f"value_data_{board_size}x{board_size}_chunk") and f.endswith('.npz')
        ])
        
        if not self.chunk_files:
            raise ValueError(f"No value chunk files found in {processed_dir}")
        
        print(f"Found {len(self.chunk_files)} value chunk files")
        
        # Calculate total size
        total_samples = 0
        for chunk_file in self.chunk_files:
            data = np.load(chunk_file)
            total_samples += len(data['labels'])
            data.close()
        
        self.total_samples = total_samples
        print(f"Total value samples: {total_samples:,}")
    
    def __iter__(self):
        """Iterate through all chunks, yielding samples one by one"""
        
        # Optionally shuffle chunk order
        chunk_order = list(range(len(self.chunk_files)))
        if self.shuffle_chunks:
            np.random.shuffle(chunk_order)
        
        # Process each chunk
        for chunk_idx in chunk_order:
            chunk_file = self.chunk_files[chunk_idx]
            
            # Load chunk
            data = np.load(chunk_file)
            features = data['features']
            labels = data['labels']
            
            # Shuffle samples within chunk
            indices = np.arange(len(labels))
            if self.shuffle_chunks:
                np.random.shuffle(indices)
            
            # Yield samples one by one
            for idx in indices:
                feature = torch.from_numpy(features[idx])
                label = torch.tensor(labels[idx], dtype=torch.float32)
                yield feature, label
            
            # Explicitly delete to free memory
            del features, labels, data
    
    def __len__(self):
        return self.total_samples


def create_value_streaming_loaders(batch_size=128, num_workers=0):
    """
    Create streaming data loaders for value network
    """
    
    train_dataset = StreamingValueDataset(shuffle_chunks=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader


if __name__ == "__main__":
    # Test streaming loader
    print("Testing value network data loader...")
    train_loader = create_value_streaming_loaders(batch_size=32)
    
    print("\nIterating through first 3 batches...")
    for i, (features, labels) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Features: {features.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label range: [{labels.min().item():.2f}, {labels.max().item():.2f}]")
        print(f"  Label mean: {labels.mean().item():.4f}")
        if i >= 2:
            break
    
    print("\n✓ Value network data loader working!")