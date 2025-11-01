# game/features.py
import numpy as np
from typing import Tuple
from game.go_board import GoBoard

class FeatureExtractor:
    """
    AlphaGo feature planes for 9x9 board
    Total: 48 planes (can reduce for faster training)
    """
    
    def __init__(self, history_length: int = 8):
        self.history_length = history_length
        # Full AlphaGo: 48 planes
        # Simplified version: 17 planes (what we'll use for 9x9)
        self.num_planes = 17
        
    def extract_features(self, board: GoBoard) -> np.ndarray:
        """
        Extract feature planes from board state
        Returns: (num_planes, board_size, board_size) numpy array
        
        Planes:
        0-7: Current player stones (last 8 moves)
        8-15: Opponent stones (last 8 moves)
        16: Color to play (all 1s for black, all 0s for white)
        
        Full AlphaGo also includes:
        - Liberties (1, 2, 3, 4+ for each player)
        - Capture size
        - Self-atari size
        - Liberties after move
        - Ladder capture/escape
        - Sensibleness
        - Zeros (bias)
        """
        size = board.size
        features = np.zeros((self.num_planes, size, size), dtype=np.float32)
        
        # Get board history (pad if necessary)
        history = board.history[-self.history_length:] if board.history else []
        history = [np.zeros((size, size))] * (self.history_length - len(history)) + history
        history.append(board.board.copy())  # Current position
        
        current_player = board.current_player
        
        # Planes 0-7: Current player's stones over last 8 positions
        for i, hist_board in enumerate(history[-self.history_length:]):
            features[i] = (hist_board == current_player).astype(np.float32)
        
        # Planes 8-15: Opponent's stones over last 8 positions
        for i, hist_board in enumerate(history[-self.history_length:]):
            features[8 + i] = (hist_board == -current_player).astype(np.float32)
        
        # Plane 16: Color to play (1 if black to play, 0 if white)
        features[16] = np.ones((size, size)) * (current_player == 1)
        
        return features
    
    def extract_full_features(self, board: GoBoard) -> np.ndarray:
        """
        Full 48-plane feature set (use this for better performance later)
        """
        size = board.size
        features = np.zeros((48, size, size), dtype=np.float32)
        
        # Basic features (0-16) same as above
        features[:17] = self.extract_features(board)
        
        current_player = board.current_player
        
        # Planes 17-20: Liberties for current player (1, 2, 3, 4+)
        for row in range(size):
            for col in range(size):
                if board.board[row, col] == current_player:
                    libs = board._count_liberties(row, col)
                    if libs == 1:
                        features[17, row, col] = 1
                    elif libs == 2:
                        features[18, row, col] = 1
                    elif libs == 3:
                        features[19, row, col] = 1
                    elif libs >= 4:
                        features[20, row, col] = 1
        
        # Planes 21-24: Liberties for opponent
        for row in range(size):
            for col in range(size):
                if board.board[row, col] == -current_player:
                    libs = board._count_liberties(row, col)
                    if libs == 1:
                        features[21, row, col] = 1
                    elif libs == 2:
                        features[22, row, col] = 1
                    elif libs == 3:
                        features[23, row, col] = 1
                    elif libs >= 4:
                        features[24, row, col] = 1
        
        # Plane 25: Ones (bias plane)
        features[25] = np.ones((size, size))
        
        # Planes 26-47: Additional advanced features (ladder, capture, etc.)
        # We'll implement these later for optimization
        
        return features
    
    def board_to_tensor(self, board: GoBoard, device='cuda') -> 'torch.Tensor':
        """Convert board to PyTorch tensor ready for network"""
        import torch
        features = self.extract_features(board)
        return torch.from_numpy(features).unsqueeze(0).to(device)  # Add batch dimension


def augment_features(features: np.ndarray, board_size: int = 9) -> list:
    """
    Generate 8 symmetries (4 rotations + 4 reflections) for data augmentation
    Returns list of (features, move_transform_fn) tuples
    """
    augmented = []
    
    # Original
    augmented.append((features, lambda m: m))
    
    # 90° rotation
    rot90 = np.rot90(features, k=1, axes=(1, 2))
    augmented.append((rot90, lambda m: (m[1], board_size - 1 - m[0])))
    
    # 180° rotation
    rot180 = np.rot90(features, k=2, axes=(1, 2))
    augmented.append((rot180, lambda m: (board_size - 1 - m[0], board_size - 1 - m[1])))
    
    # 270° rotation
    rot270 = np.rot90(features, k=3, axes=(1, 2))
    augmented.append((rot270, lambda m: (board_size - 1 - m[1], m[0])))
    
    # Horizontal flip
    flip_h = np.flip(features, axis=2)
    augmented.append((flip_h, lambda m: (m[0], board_size - 1 - m[1])))
    
    # Vertical flip
    flip_v = np.flip(features, axis=1)
    augmented.append((flip_v, lambda m: (board_size - 1 - m[0], m[1])))
    
    # Diagonal flip
    diag = np.transpose(features, (0, 2, 1))
    augmented.append((diag, lambda m: (m[1], m[0])))
    
    # Anti-diagonal flip
    anti_diag = np.flip(np.transpose(features, (0, 2, 1)), axis=(1, 2))
    augmented.append((anti_diag, lambda m: (board_size - 1 - m[1], board_size - 1 - m[0])))
    
    return augmented