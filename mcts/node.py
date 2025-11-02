# mcts/node.py
class MCTSNode:
    def __init__(self, board_state, parent=None, prior_prob=0):
        self.board_state = board_state
        self.parent = parent
        self.children = {}  # move -> child node
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob  # From policy network
    
    def ucb_score(self, c_puct=1.0):
        """Upper Confidence Bound for Trees"""
        # Balances exploration vs exploitation
        pass