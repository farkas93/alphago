# utils/sgf_parser_value.py
"""
SGF parser for Value Network training data
Extracts (board_state, game_outcome) pairs
"""
import os
import numpy as np
from sgfmill import sgf, sgf_moves
from game.go_board import GoBoard
from game.features import FeatureExtractor
from typing import List, Tuple, Optional
from config import BOARD_SIZE

def parse_sgf_for_value(file_path: str, extractor: FeatureExtractor, 
                        sample_rate: float = 1.0) -> Optional[List[Tuple[np.ndarray, float]]]:
    """
    Parse SGF file and extract (board_state, outcome) pairs for value network training
    
    Args:
        file_path: Path to SGF file
        extractor: Feature extractor
        sample_rate: Fraction of positions to sample (1.0 = all positions)
    
    Returns:
        List of (features, outcome) tuples where outcome is in [-1, 1]
        -1 = loss for current player, +1 = win for current player, 0 = draw
    """
    try:
        with open(file_path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception:
        return None

    # Get game properties
    try:
        board_size = game.get_size()
        root = game.get_root()
        
        # Get game result (e.g., "B+Resign", "W+3.5", "B+R", "0" for draw)
        result_str = root.get("RE")
        if not result_str:
            return None
            
    except (ValueError, KeyError):
        return None

    # Only process games matching our target board size
    if board_size != BOARD_SIZE:
        return None
    
    # Parse game result
    # Format: "B+..." = Black wins, "W+..." = White wins, "0" or "Draw" = draw
    if result_str.startswith("B+"):
        black_won = True
    elif result_str.startswith("W+"):
        black_won = False
    elif result_str in ["0", "Draw", "Jigo"]:
        black_won = None  # Draw
    else:
        return None  # Unknown result format

    # Initialize board
    board = GoBoard(size=board_size)
    training_samples = []
    
    # Handle handicap stones
    if root.has_property('AB'):
        handicap_points = root.get('AB')
        for point in handicap_points:
            if point is not None:
                row, col = point
                board.board[row, col] = 1
    
    if root.has_property('AW'):
        white_points = root.get('AW')
        for point in white_points:
            if point is not None:
                row, col = point
                board.board[row, col] = -1

    # Iterate through game moves
    move_count = 0
    try:
        for node in game.get_main_sequence():
            if not node.has_property('B') and not node.has_property('W'):
                continue
            
            try:
                move = node.get_move()
                if move is None:
                    continue
                
                color, coords = move
                if coords is None:
                    continue
                
                row, col = coords
                
                # Sample positions based on sample_rate
                if np.random.random() < sample_rate:
                    # Extract features for current position
                    features = extractor.extract_features(board)
                    
                    # Determine outcome from perspective of current player
                    current_player = board.current_player  # 1 = Black, -1 = White
                    
                    if black_won is None:
                        outcome = 0.0  # Draw
                    elif (current_player == 1 and black_won) or (current_player == -1 and not black_won):
                        outcome = 1.0  # Current player wins
                    else:
                        outcome = -1.0  # Current player loses
                    
                    training_samples.append((features, outcome))
                
                # Play the move
                if not board.play_move(row, col):
                    break
                
                move_count += 1
                    
            except (ValueError, AttributeError):
                continue
                
    except Exception:
        return None if not training_samples else training_samples

    return training_samples if training_samples else None


def get_game_outcome_stats(processed_dir, board_size=BOARD_SIZE):
    """
    Quick scan to get statistics on game outcomes
    Useful for understanding data distribution
    """
    from pathlib import Path
    
    kgs_folders = [f for f in os.listdir(processed_dir) if f.startswith("kgs-")]
    
    all_sgf_files = []
    for folder in kgs_folders:
        folder_path = os.path.join(processed_dir, folder)
        sgf_files = list(Path(folder_path).rglob("*.sgf"))
        all_sgf_files.extend(sgf_files)
    
    black_wins = 0
    white_wins = 0
    draws = 0
    unknown = 0
    
    print(f"Scanning {len(all_sgf_files)} SGF files for outcome statistics...")
    
    for sgf_file in all_sgf_files[:1000]:  # Sample first 1000 games
        try:
            with open(sgf_file, "rb") as f:
                game = sgf.Sgf_game.from_bytes(f.read())
            
            if game.get_size() != board_size:
                continue
            
            root = game.get_root()
            result_str = root.get("RE")
            
            if not result_str:
                unknown += 1
            elif result_str.startswith("B+"):
                black_wins += 1
            elif result_str.startswith("W+"):
                white_wins += 1
            elif result_str in ["0", "Draw", "Jigo"]:
                draws += 1
            else:
                unknown += 1
                
        except:
            unknown += 1
    
    total = black_wins + white_wins + draws
    if total > 0:
        print(f"\nGame Outcome Statistics (from {total} valid games):")
        print(f"  Black wins: {black_wins} ({100*black_wins/total:.1f}%)")
        print(f"  White wins: {white_wins} ({100*white_wins/total:.1f}%)")
        print(f"  Draws: {draws} ({100*draws/total:.1f}%)")
        print(f"  Unknown: {unknown}")


if __name__ == "__main__":
    from config import DATA_DIR
    
    # Get outcome statistics
    get_game_outcome_stats(DATA_DIR)