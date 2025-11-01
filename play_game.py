# play_game.py
"""Quick test to verify everything works"""
from game.go_board import GoBoard
from game.features import FeatureExtractor
from game.renderer import GoRenderer

def test_board_logic():
    """Test basic Go rules"""
    board = GoBoard(9)
    
    # Test basic moves
    assert board.play_move(4, 4), "Center move should be valid"
    assert board.play_move(4, 5), "Adjacent move should be valid"
    assert not board.play_move(4, 4), "Occupied position should be invalid"
    
    # Test capture
    board2 = GoBoard(9)
    board2.play_move(0, 1)  # Black
    board2.play_move(0, 0)  # White
    board2.play_move(1, 0)  # Black
    assert board2.board[0, 0] == 0, "White stone should be captured"
    
    print("✓ Board logic tests passed")

def test_features():
    """Test feature extraction"""
    board = GoBoard(9)
    board.play_move(4, 4)
    board.play_move(4, 5)
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(board)
    
    assert features.shape == (17, 9, 9), f"Wrong shape: {features.shape}"
    assert features.dtype == np.float32, "Wrong dtype"
    
    print("✓ Feature extraction tests passed")

if __name__ == "__main__":
    import numpy as np
    test_board_logic()
    test_features()
    
    print("\nLaunching interactive game...")
    print("Controls:")
    print("  - Click to place stones")
    print("  - L: Toggle legal move hints")
    print("  - R: Reset game")
    print("  - ESC: Quit")
    
    board = GoBoard(9)
    renderer = GoRenderer(board)
    renderer.run_interactive()