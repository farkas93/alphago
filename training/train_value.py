# training/train_value.py
"""
Train the value network with supervised learning
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

from models.value_net import ValueNetwork
from training.data_loader_value import create_value_streaming_loaders
from config import CHECKPOINT_DIR, VALUE_MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, \
                   USE_MLFLOW, MLFLOW_UPLOAD_TEST, MLFLOW_DEFAULT_URI

def train_value_network(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, 
                       mlflow_uri=None, experiment_name="AlphaGo-Value-Network"):
    """Train the value network with streaming data and MLflow tracking"""
    
    # Setup MLflow
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"MLflow tracking URI: {mlflow_uri}")
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"value_net_19x19_epochs{epochs}"):
        
        if MLFLOW_UPLOAD_TEST:
            print(f"Current Run Artifact URI: {mlflow.get_artifact_uri()}")
            print("\nRunning instant artifact upload test...")
            dummy_model_state = {'test_tensor': torch.randn(5, 5)}
            dummy_path = os.path.join(CHECKPOINT_DIR, 'dummy_artifact_test_value.pth')
            torch.save(dummy_model_state, dummy_path)
            mlflow.log_artifact(dummy_path, artifact_path="validation_test")
            os.remove(dummy_path)
            print("✓ Instant artifact upload test successful!")

        # Log hyperparameters
        mlflow.log_params({
            "board_size": 19,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "scheduler": "StepLR",
            "scheduler_step": 5,
            "scheduler_gamma": 0.5,
        })
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))
        
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        
        # Create model
        model = ValueNetwork().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        mlflow.log_param("model_parameters", num_params)
        
        # Loss and optimizer (MSE for regression)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Data loader
        print("\nCreating streaming data loader...")
        train_loader = create_value_streaming_loaders(batch_size=batch_size, num_workers=0)
        
        # Training history
        history = {
            'train_loss': [],
            'train_mae': []  # Mean Absolute Error for interpretability
        }
        
        best_train_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            train_total = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc="Training", total=None)
            for features, labels in pbar:
                features = features.to(device)
                labels = labels.to(device).unsqueeze(1)  # Shape: (batch, 1)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                train_mae += torch.abs(outputs - labels).mean().item()
                train_total += labels.size(0)
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{torch.abs(outputs - labels).mean().item():.4f}'
                })
                
                # Log batch metrics every 100 batches
                if batch_count % 100 == 0:
                    mlflow.log_metrics({
                        "batch_loss": loss.item(),
                        "batch_mae": torch.abs(outputs - labels).mean().item(),
                    }, step=epoch * 10000 + batch_count)
            
            train_loss /= batch_count
            train_mae /= batch_count
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)
            
            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "epoch_loss": train_loss,
                "epoch_mae": train_mae,
                "learning_rate": scheduler.get_last_lr()[0],
                "samples_processed": train_total,
            }, step=epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss (MSE): {train_loss:.4f} | Train MAE: {train_mae:.4f}")
            print(f"  Samples processed: {train_total:,}")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'value_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            
            # Log checkpoint as artifact every 5 epochs
            if (epoch + 1) % 5 == 0:
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            
            # Save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                }, VALUE_MODEL_PATH)
                print(f"  ✓ Saved best model (train_loss: {train_loss:.4f})")
                
                # Log best model to MLflow
                mlflow.log_artifact(VALUE_MODEL_PATH, artifact_path="best_model")
                mlflow.log_metric("best_train_loss", best_train_loss)
            
            # Learning rate schedule
            scheduler.step()
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Plot training history
        plot_path = plot_training_history(history)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # Log final model
        mlflow.pytorch.log_model(model, "final_model")
        
        # Log final metrics
        mlflow.log_metrics({
            "final_train_loss": train_loss,
            "final_train_mae": train_mae,
            "best_train_loss": best_train_loss,
        })
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best training loss: {best_train_loss:.4f}")
        print(f"Model saved to: {VALUE_MODEL_PATH}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"{'='*60}\n")

def plot_training_history(history):
    """Plot training curves and return path"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss (MSE)', marker='o', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Value Network Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE
    ax2.plot(history['train_mae'], label='Train MAE', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Value Network Training MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, 'value_training_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nTraining plot saved to {plot_path}")
    
    return plot_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AlphaGo Value Network')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    default_uri = MLFLOW_DEFAULT_URI if USE_MLFLOW else None
    parser.add_argument('--mlflow-uri', type=str, default=default_uri,
                        help='MLflow tracking server URI')
    parser.add_argument('--experiment-name', type=str, default='AlphaGo-Value-Network',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    train_value_network(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name
    )