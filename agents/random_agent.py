# agents/random_agent.py
import random
from agents.base_agent import Agent
from game.go_board import GoBoard
from typing import Tuple, Optional

class RandomAgent(Agent):
    """Agent that plays random legal moves"""
    
    def __init__(self):
        super().__init__("Random Agent")
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Select a random legal move"""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)