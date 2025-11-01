# play_game.py
"""Test game with different agent configurations"""
from game.go_board import GoBoard
from game.renderer import GoRenderer
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

def play_human_vs_random():
    """Human (Black) vs Random Agent (White)"""
    print("Mode: Human (Black) vs Random Agent (White)")
    board = GoBoard(9)
    renderer = GoRenderer(board, white_agent=RandomAgent())
    renderer.run_interactive()

def play_human_vs_heuristic():
    """Human (Black) vs Heuristic Agent (White)"""
    print("Mode: Human (Black) vs Heuristic Agent (White)")
    board = GoBoard(9)
    renderer = GoRenderer(board, white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_random_vs_heuristic():
    """Random Agent (Black) vs Heuristic Agent (White) - Watch AI battle"""
    print("Mode: Random Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(9)
    renderer = GoRenderer(board, 
                         black_agent=RandomAgent(),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_heuristic_vs_heuristic():
    """Heuristic vs Heuristic - Watch AI battle"""
    print("Mode: Heuristic Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(9)
    renderer = GoRenderer(board,
                         black_agent=HeuristicAgent(),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

if __name__ == "__main__":
    print("Choose game mode:")
    print("1. Human vs Random Agent")
    print("2. Human vs Heuristic Agent")
    print("3. Random vs Heuristic (AI battle)")
    print("4. Heuristic vs Heuristic (AI battle)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        play_human_vs_random()
    elif choice == "2":
        play_human_vs_heuristic()
    elif choice == "3":
        play_random_vs_heuristic()
    elif choice == "4":
        play_heuristic_vs_heuristic()
    else:
        print("Invalid choice, defaulting to Human vs Heuristic")
        play_human_vs_heuristic()