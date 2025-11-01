# models/value_net.py (skeleton)
"""
Value Network - predicts win probability from position
Very similar to policy net but outputs single scalar
"""
class ValueNetwork(nn.Module):
    def __init__(self):
        # Same conv tower as policy net
        # Different head: outputs single value in [-1, 1]
        pass
    
    def forward(self, x):
        # Returns win probability for current player
        pass