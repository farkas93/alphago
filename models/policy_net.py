# models/policy_net.py
"""
Policy Network for AlphaGo
Predicts move probabilities P(a|s) from board state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BOARD_SIZE, NUM_FEATURE_PLANES, RESIDUAL_BLOCKS, FILTERS

class ResidualBlock(nn.Module):
    """Residual block with 2 conv layers + skip connection"""

    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)
    
class PolicyNetwork(nn.Module):
    """
    AlphaGo Policy Network
    Input: (batch, 17, 19, 19) feature planes
    Output: (batch, 361) move probabilities
    """

    def __init__(self, input_channels=NUM_FEATURE_PLANES,
                 num_filters=FILTERS,
                 num_residual_blocks=RESIDUAL_BLOCKS,
                 board_size=BOARD_SIZE):
        super().__init__()

        self.board_size = board_size
        self.output_size = board_size * board_size
        
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])

        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*board_size*board_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.residual_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        return self.policy_fc(policy)
    

    def predict_move(self, board_state, temperature=1.0):
        """
        Predict move probabilities for a single board state
        
        Args:
            board_state: (17, 19, 19) tensor
            temperature: Controls randomness (1.0 = normal, <1 = more deterministic)
        
        Returns:
            move_probs: (361,) probability distribution
        """
        self.eval()
        with torch.no_grad():
            if board_state.dim() == 3:
                board_state = board_state.unsqueeze(0) # Add batch dimension
            
            logits = self.forward(board_state)

            logits = logits / temperature
            move_probs = F.softmax(logits, dim=1)

            return move_probs.squeeze(0)
        
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the network
    print("Testing Policy Network...")
    
    model = PolicyNetwork()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, NUM_FEATURE_PLANES, BOARD_SIZE, BOARD_SIZE)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, {BOARD_SIZE * BOARD_SIZE})")
    
    # Test prediction
    single_board = torch.randn(NUM_FEATURE_PLANES, BOARD_SIZE, BOARD_SIZE)
    move_probs = model.predict_move(single_board)
    print(f"\nMove probabilities shape: {move_probs.shape}")
    print(f"Sum of probabilities: {move_probs.sum().item():.4f}")
    print(f"Max probability: {move_probs.max().item():.4f}")
