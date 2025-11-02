# mcts/mcts.py
class MCTS:
    def __init__(self, policy_net, value_net, num_simulations=800):
        self.policy_net = policy_net
        self.value_net = value_net
        self.num_simulations = num_simulations
    
    def search(self, board):
        root = MCTSNode(board)
        
        for _ in range(self.num_simulations):
            node = root
            # 1. Select
            # 2. Expand
            # 3. Evaluate
            # 4. Backup
        
        return self.get_best_move(root)