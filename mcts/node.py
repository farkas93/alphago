# mcts/node.py
"""
MCTS Tree Node
Stores statistics and children for Monte Carlo Tree Search
"""
import numpy as np
from typing import Optional, Dict, Tuple
from game.go_board import GoBoard

class MCTSNode:
    """
    Node in the MCTS tree
    
    Attributes:
        board: GoBoard state at this node
        parent: Parent node (None for root)
        move: Move that led to this node (row, col)
        prior_prob: Prior probability from policy network P(a|s)
        visit_count: Number of times this node was visited (N)
        total_value: Sum of values from all visits (W)
        children: Dictionary of move -> child node
    """
    def __init__(self, board: GoBoard, parent: Optional['MCTSNode'] = None,
                 move: Optional[Tuple[int, int]] = None, prior_prob: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob # From policy network

        self.children: Dict[Tuple[int, int], MCTSNode] = {}

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children expanded yet)"""
        return not self.children
    
    def is_root(self) -> bool:
        """Check if the node is root"""
        return self.parent is None
    
    def get_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for this node
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Where:
        - Q(s,a) = average value (exploitation)
        - P(s,a) = prior probability from policy network
        - N(s) = parent visit count
        - N(s,a) = this node's visit count
        - c_puct = exploration constant (higher = more exploration)
        
        Args:
            c_puct: Exploration constant (typically 1.0-5.0)
        
        Returns:
            UCB score (higher = should explore this node)
        """
        if self.parent is None:
            return 0.0

        q_value = self.get_value()
        u_value = c_puct * self.prior_prob * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select child with highest UCB score

        Args: c_puct: 
            Exploration constant
        Returns:
            Child node with highest UCB score
        """
        return max(self.children.values(), key=lambda child: child.get_ucb_score(c_puct))
    
    def expand(self, policy_probs: np.ndarray, legal_moves: list) -> None:
        """
        Expand node by adding all legal moves as children
        
        Args:
            policy_probs: Policy network output (361,) for 19x19
            legal_moves: List of legal moves [(row, col), ...]
        """
        for move in legal_moves:
            if move not in self.children:
                board_cp = self.board.copy()
                board_cp.play_move(move[0], move[1])

                move_idx = move[0] * self.board.size + move[1]
                prior = policy_probs[move_idx]

                child = MCTSNode(board_cp, parent=self, move=move, prior_prob=prior)
                self.children[move] = child
    
    def update(self, value: float) -> None:
        """
        Update node statistics (backprop)

        Args:
            value: Value to add (from perspective of player who made the move)
        """
        self.visit_count += 1
        self.total_value += value

    def get_visit_distribution(self) -> np.ndarray:
        """
        Get visit count distribution over all possible moves
        Used for training and move selection
        
        Returns:
            Array of shape (board_size * board_size, ) with visit counts
        """

        distribution = np.zeros(self.board.size * self.board.size, dtype=np.float32)
        for move, child in self.children.items():
            move_idx = move[0] * self.board.size + move[1]
            distribution[move_idx] = child.visit_count
        return distribution

    def __repr__(self):
        return (f"MCTSNode(move={self.move}, visits={self.visit_count}, "
                f"value={self.get_value():.3f}, prior={self.prior_prob:.3f}, "
                f"children={len(self.children)})")