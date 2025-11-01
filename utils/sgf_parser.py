# utils/sgf_parser.py - Fixed version
import os
import numpy as np
from sgfmill import sgf, sgf_moves
from game.go_board import GoBoard
from game.features import FeatureExtractor
from typing import List, Tuple, Optional
from config import BOARD_SIZE

def parse_sgf_file(file_path: str, extractor: FeatureExtractor) -> Optional[List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """
    Parses a single SGF file and returns a list of (features, move) tuples.
    Returns None if the game is not the target board size or has errors.
    """
    try:
        with open(file_path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:
        # Debug: print which files fail
        # print(f"Failed to parse {file_path}: {e}")
        return None

    # Get game properties
    try:
        board_size = game.get_size()
    except (ValueError, KeyError) as e:
        # print(f"Failed to get size from {file_path}: {e}")
        return None

    # Only process games matching our target board size
    if board_size != BOARD_SIZE:
        return None

    # Initialize board with correct size
    board = GoBoard(size=board_size)
    training_samples = []
    
    # Handle handicap stones (AB property in root)
    root = game.get_root()
    if root.has_property('AB'):  # Added Black handicap stones
        handicap_points = root.get('AB')
        for point in handicap_points:
            if point is not None:
                row, col = point
                board.board[row, col] = 1  # Place black stones
    
    if root.has_property('AW'):  # Added White stones (rare)
        white_points = root.get('AW')
        for point in white_points:
            if point is not None:
                row, col = point
                board.board[row, col] = -1  # Place white stones

    # Iterate through game moves
    try:
        for node in game.get_main_sequence():
            if not node.has_property('B') and not node.has_property('W'):
                continue  # Skip nodes without moves
            
            try:
                # Get move from node
                move = node.get_move()
                if move is None:
                    continue  # Skip pass moves
                
                color, coords = move
                if coords is None:
                    continue  # Pass move
                
                row, col = coords
                
                # Extract features before playing the move
                features = extractor.extract_features(board)
                label = (row, col)
                training_samples.append((features, label))
                
                # Play the move
                if not board.play_move(row, col):
                    # Invalid move in SGF, skip rest of game
                    break
                    
            except (ValueError, AttributeError) as e:
                # Skip problematic moves
                continue
                
    except Exception as e:
        # print(f"Error processing moves in {file_path}: {e}")
        return None if not training_samples else training_samples

    return training_samples if training_samples else None