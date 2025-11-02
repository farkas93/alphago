# config.py (NEW FILE - centralized configuration)
"""Configuration for AlphaGo training"""

# Board settings
BOARD_SIZE = 19  # Changed from 9 to 19

# Feature extraction
HISTORY_LENGTH = 8
NUM_FEATURE_PLANES = 17  # Can expand to 48 later

# Neural network architecture
RESIDUAL_BLOCKS = 10  # For 19x19, we'll use deeper network
FILTERS = 128  # Increased from what we'd use for 9x9

# Training hyperparameters
BATCH_SIZE = 128  # Adjust based on GPU memory
LEARNING_RATE = 0.001
EPOCHS = 20

# Data paths
DATA_DIR = "data"
SGF_DIR = "data"
PROCESSED_DIR = "data/processed"

# Model checkpoints
CHECKPOINT_DIR = "checkpoints"
POLICY_MODEL_PATH = "checkpoints/policy_net_19x19.pth"
VALUE_MODEL_PATH = "checkpoints/value_net_19x19.pth"

USE_MLFLOW=True
MLFLOW_UPLOAD_TEST=True
MLFLOW_DEFAULT_URI="https://mlflow.chezombor.com/"