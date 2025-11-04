# agents/mcts_agent.py
"""
Agent that uses MCTS with neural networks
"""
import torch
from agents.base_agent import Agent
from game.go_board import GoBoard
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from mcts.mcts import MCTS
from config import POLICY_MODEL_PATH, VALUE_MODEL_PATH

class MCTSAgent(Agent):
    """
    Agent that uses Monte Carlo Tree Search with policy and value networks
    """
    
    def __init__(self, policy_model_path=POLICY_MODEL_PATH, 
                 value_model_path=VALUE_MODEL_PATH,
                 num_simulations=800, 
                 c_puct=1.0,
                 temperature=0.0):
        """
        Initialize MCTS Agent
        
        Args:
            policy_model_path: Path to trained policy network
            value_model_path: Path to trained value network
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            temperature: Move selection temperature (0 = deterministic)
        """
        super().__init__(f"MCTS Agent ({num_simulations} sims)")
        
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        print(f"Loading policy network from {policy_model_path}...")
        self.policy_net = PolicyNetwork().to(self.device)
        policy_checkpoint = torch.load(policy_model_path, map_location=self.device)
        self.policy_net.load_state_dict(policy_checkpoint['model_state_dict'])
        self.policy_net.eval()
        
        print(f"Loading value network from {value_model_path}...")
        self.value_net = ValueNetwork().to(self.device)
        value_checkpoint = torch.load(value_model_path, map_location=self.device)
        self.value_net.load_state_dict(value_checkpoint['model_state_dict'])
        self.value_net.eval()
        
        # Create MCTS
        self.mcts = MCTS(
            self.policy_net,
            self.value_net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=self.device
        )
        
        print(f"✓ MCTS Agent initialized")
    
    def select_move(self, board: GoBoard):
        """Select move using MCTS"""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        
        # Run MCTS search
        root = self.mcts.search(board)
        
        # Get best move
        move = self.mcts.get_best_move(root, temperature=self.temperature)
        
        return move


if __name__ == "__main__":
    # Test MCTS agent
    print("Testing MCTS Agent...")
    
    from game.go_board import GoBoard
    
    # Create agent with fewer simulations for testing
    agent = MCTSAgent(num_simulations=100)
    
    # Test on empty board
    board = GoBoard(19)
    print("\nSelecting move on empty board...")
    move = agent.select_move(board)
    print(f"Selected move: {move}")
    
    # Play a few moves
    print("\nPlaying 3 moves...")
    for i in range(3):
        move = agent.select_move(board)
        if move:
            board.play_move(move[0], move[1])
            print(f"Move {i+1}: {move}")
    
    print("\n✓ MCTS Agent test complete!")