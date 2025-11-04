# mcts/mcts.py
"""
Monte Carlo Tree Search
Combines policy network, value network, and tree search
"""
import numpy as np
import torch
from typing import Tuple, Optional
from game.go_board import GoBoard
from game.features import FeatureExtractor
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from mcts.node import MCTSNode

class MCTS:
    """
    Monte Carlo Tree Search guided by neural nets

    The algorithm:
    1. Selection: Start at root. select child with highest UCB until leaf
    2. Expansion: Expand leaf node using policy network
    3. Evaluation: Evaluate position using value network
    4. Backup: Propagate value back up the tree
    """

    def __init__(self, policy_net: PolicyNetwork, value_net: ValueNetwork,
                 num_simulations: int = 800, c_puct: float = 1.0,
                 device: str = 'cuda'):
        """
        Initialize MCTS
        
        Args:
            policy_net: Trained policy network
            value_net: Trained value network
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant (higher = more exploration)
            device: 'cuda' or 'cpu'
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

        self.feature_extractor = FeatureExtractor()

        self.policy_net.eval()
        self.value_net.eval()


    def search(self, board: GoBoard) -> MCTSNode:
        """
        Run MCTS search from given board position
        
        Args:
            board: Current board state
        
        Returns:
            Root node after search (contains visit statistics)
        """
        root = MCTSNode(board.copy())
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # greedily select path of children with highest usb scores.
            while not node.is_leaf() and not node.board.is_game_over():
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            value = self._expand_and_evaluate(node)

            self._backup(search_path, value)
        
        return root
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand leaf node and evaluate position
        
        Args:
            node: Leaf node to expand
        
        Returns:
            Value of position from perspective of current player
        """
        if node.board.is_game_over():
            return 0.0
        
        legal_moves = node.board.get_legal_moves()

        if not legal_moves:
            return 0.0
        
        features = self.feature_extractor.extract_features(node.board)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits = self.policy_net(features_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

            legal_mask = np.zeros(node.board.size * node.board.size, dtype=bool)
            for move in legal_moves:
                move_idx = move[0] * node.board.size + move[1]
                legal_mask[move_idx] = True

                policy_probs = policy_probs * legal_mask
                prob_sum = policy_probs.sum()
                if prob_sum > 0:
                    policy_probs = policy_probs / prob_sum
                else:
                    # Fallback: uniform over legal moves
                    policy_probs = legal_mask.astype(np.float32) / len(legal_moves)
                
                value = self.value_net(features_tensor).cpu().item()

            node.expand(policy_probs, legal_moves)
            
            return value
        