# mcts/mcts.py (skeleton)
"""
Monte Carlo Tree Search guided by neural networks
"""
class MCTS:
    def __init__(self, policy_net, value_net):
        self.policy_net = policy_net
        self.value_net = value_net
    
    def search(self, board, num_simulations=800):
        # 1. Selection - traverse tree using UCB
        # 2. Expansion - add new node
        # 3. Evaluation - use value_net + rollout
        # 4. Backup - propagate value up tree
        pass
    
    def get_best_move(self):
        # Return move with most visits
        pass