# play_game.py
"""Test game with different agent configurations"""
from game.go_board import GoBoard
from game.renderer import GoRenderer
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.policy_agent import PolicyAgent
from agents.mcts_agent import MCTSAgent
from config import BOARD_SIZE

def play_human_vs_random():
    """Human (Black) vs Random Agent (White)"""
    print("Mode: Human (Black) vs Random Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board, white_agent=RandomAgent())
    renderer.run_interactive()

def play_human_vs_heuristic():
    """Human (Black) vs Heuristic Agent (White)"""
    print("Mode: Human (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board, white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_human_vs_policy():
    """Human (Black) vs Policy Agent (White)"""
    print("Mode: Human (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board, white_agent=PolicyAgent())
    renderer.run_interactive()

def play_random_vs_heuristic():
    """Random Agent (Black) vs Heuristic Agent (White) - Watch AI battle"""
    print("Mode: Random Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board, 
                         black_agent=RandomAgent(),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_heuristic_vs_heuristic():
    """Heuristic vs Heuristic - Watch AI battle"""
    print("Mode: Heuristic Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board,
                         black_agent=HeuristicAgent(),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_policy_vs_heuristic():
    """Policy Agent vs Heuristic - Watch AI battle"""
    print("Mode: Policy Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board,
                         black_agent=PolicyAgent(),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_human_vs_mcts():
    """Human (Black) vs MCTS Agent (White)"""
    print("Mode: Human (Black) vs MCTS Agent (White)")
    board = GoBoard(BOARD_SIZE)
    # Use fewer simulations for interactive play (faster)
    renderer = GoRenderer(board, white_agent=MCTSAgent(num_simulations=400))
    renderer.run_interactive()

def play_mcts_vs_heuristic():
    """MCTS Agent vs Heuristic - Watch AI battle"""
    print("Mode: MCTS Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board,
                         black_agent=MCTSAgent(num_simulations=400),
                         white_agent=HeuristicAgent())
    renderer.run_interactive()

def play_mcts_vs_policy():
    """MCTS Agent vs Heuristic - Watch AI battle"""
    print("Mode: MCTS Agent (Black) vs Heuristic Agent (White)")
    board = GoBoard(BOARD_SIZE)
    renderer = GoRenderer(board,
                         black_agent=MCTSAgent(num_simulations=400),
                         white_agent=PolicyAgent())
    renderer.run_interactive()

if __name__ == "__main__":
    print("Choose game mode:")
    print("1. Human vs Random Agent")
    print("2. Human vs Heuristic Agent")
    print("3. Human vs Policy Agent")
    print("4. Policy Agent vs Heuristic (AI battle)")
    print("5 Human vs MCTS Agent")
    print("6. MCTS vs Heuristic (AI battle)")
    print("7. MCTS vs Policy (AI battle)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        play_human_vs_random()
    elif choice == "2":
        play_human_vs_heuristic()
    elif choice == "3":
        play_human_vs_policy()
    elif choice == "4":
        play_policy_vs_heuristic()
    elif choice == "5":
        play_human_vs_mcts()
    elif choice == "6":
        play_mcts_vs_heuristic()
    elif choice == "7":
        play_mcts_vs_policy()
    else:
        print("Invalid choice, defaulting to Human vs Heuristic")
        play_human_vs_heuristic()