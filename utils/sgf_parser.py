# utils/sgf_parser.py
import os
import numpy as np
from sgfmill import sgf, sgf_moves
from game.go_board import GoBoard
from game.features import FeatureExtractor
from typing import List, Tuple, Optional

def parse_sgf_file(file_path: str, extractor: FeatureExtractor) -> Optional[List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """
    Parses a single SGF file and returns a list of (features, move) tuples.
    Returns None if the game is not 9x9 or has errors.
    """
    try:
        with open(file_path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception:
        return None  # Skip corrupted files

    # Get game properties
    try:
        board_size = game.get_size()
        winner_node = game.get_root()
        winner = winner_node.get("RE")
        komi = float(winner_node.get("KM"))
    except (ValueError, KeyError):
        return None # Skip games with missing info

    # We only want 9x9 games for this project
    if board_size != 9:
        return None

    board = GoBoard(size=board_size)
    training_samples = []

    # Iterate through game moves
    for node in game.get_main_sequence():
        try:
            color, move = sgf_moves.get_move_from_sgf(node.get_raw_move(), board_size)
        except (ValueError, AttributeError):
            continue  # Skip non-move nodes

        if move is None:  # Pass move
            continue

        row, col = move
        
        # Before playing the move, extract the features for the current state
        features = extractor.extract_features(board)
        
        # The label is the move that was played
        label = (row, col)
        
        training_samples.append((features, label))
        
        # Play the move to update the board state for the next iteration
        board.play_move(row, col)

    return training_samples

def sgf_coord_to_num(coord: str, board_size: int = 9) -> Tuple[int, int]:
    """Converts SGF coordinate (e.g., 'pd') to numerical (row, col)"""
    col_str = "abcdefghijklmnopqrstuvwxyz"
    col = col_str.find(coord[0])
    row = col_str.find(coord[1])
    return row, col