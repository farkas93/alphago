# training/train_policy.py - Updated for streaming
"""
Train the policy network with supervised learning (streaming version)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.policy_net import PolicyNetwork
from training.data_loader import create_streaming_loaders  # Changed import
from config import CHECKPOINT_DIR, POLICY_MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS

def train_policy_network(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train the policy network with streaming data"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # Create model
    model = PolicyNetwork().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Data loader (streaming - no val split)
    print("\nCreating streaming data loader...")
    train_loader = create_streaming_loaders(batch_size=batch_size, num_workers=0)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    best_train_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc="Training", total=None)  # total=None for streaming
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= batch_count
        train_acc = 100. * train_correct / train_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Samples processed: {train_total:,}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'policy_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
        }, checkpoint_path)
        
        # Save best model
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
            }, POLICY_MODEL_PATH)
            print(f"  ✓ Saved best model (train_acc: {train_acc:.2f}%)")
        
        # Learning rate schedule
        scheduler.step()
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best training accuracy: {best_train_acc:.2f}%")
    print(f"Model saved to: {POLICY_MODEL_PATH}")
    print(f"{'='*60}\n")

def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_history.png'))
    print(f"\nTraining plot saved to {CHECKPOINT_DIR}/training_history.png")

if __name__ == "__main__":
    train_policy_network()