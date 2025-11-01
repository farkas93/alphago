# agents/policy_agent.py
"""Agent that uses trained policy network"""
import torch
from agents.base_agent import Agent
from game.go_board import GoBoard
from game.features import FeatureExtractor
from models.policy_net import PolicyNetwork
from config import POLICY_MODEL_PATH

class PolicyAgent(Agent):
    def __init__(self, model_path=POLICY_MODEL_PATH, temperature=1.0):
        super().__init__("Policy Network Agent")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = FeatureExtractor()
        self.temperature = temperature
        
        # Load trained model
        self.model = PolicyNetwork().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded policy network from {model_path}")
    
    def select_move(self, board: GoBoard):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        
        # Extract features
        features = self.extractor.extract_features(board)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Get move probabilities
        with torch.no_grad():
            move_probs = self.model.predict_move(features_tensor.squeeze(0), self.temperature)
        
        # Filter to legal moves only
        move_probs_np = move_probs.cpu().numpy()
        legal_probs = []
        for row, col in legal_moves:
            idx = row * board.size + col
            legal_probs.append((move_probs_np[idx], (row, col)))
        
        # Sort by probability and pick best
        legal_probs.sort(reverse=True)
        return legal_probs[0][1]  # Return best legal move