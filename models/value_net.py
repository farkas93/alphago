# models/value_net.py (skeleton)
"""
Value Network - predicts win probability from position
Very similar to policy net but outputs single scalar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_block import ResidualBlock
from config import BOARD_SIZE, NUM_FEATURE_PLANES, RESIDUAL_BLOCKS, FILTERS, \
                   VALUE_HEAD_FILTERS, VALUE_HEAD_HIDDEN_SIZE

class ValueNetwork(nn.Module):
    """
    Input: (batch, 17, 19, 19) feature planes
    Output: (batch, 1) win probability for the current player [-1, 1]
    """

    def __init__(self, input_channels=NUM_FEATURE_PLANES, 
                 num_filters=FILTERS, 
                 num_residual_blocks=RESIDUAL_BLOCKS,
                 board_size=BOARD_SIZE,
                 value_head_filters=VALUE_HEAD_FILTERS,
                 value_head_hidden_size=VALUE_HEAD_HIDDEN_SIZE):
        super().__init__()

        self.board_size = board_size
        
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])

        self.value_conv = nn.Conv2d(num_filters, value_head_filters, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_head_filters)
        
        self.value_fc1 = nn.Linear(value_head_filters * board_size * board_size, value_head_hidden_size)
        self.value_fc2 = nn.Linear(value_head_hidden_size, 1) 

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.residual_blocks:
            x = block(x)
        
        value = F.relu(self.policy_bn(self.policy_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        return  torch.tanh(self.value_fc2(value))

        
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the network
    print("Testing Value Network...")
    
    model = ValueNetwork()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, NUM_FEATURE_PLANES, BOARD_SIZE, BOARD_SIZE)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 1)")
    print(f"Output values (should be between -1 and 1):\n{output}")
