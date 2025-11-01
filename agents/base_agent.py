# agents/base_agent.py
from abc import ABC, abstractmethod
from game.go_board import GoBoard
from typing import Tuple, Optional

class Agent(ABC):
    """Base class for Go agents"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Select a move given the current board state"""
        pass
    
    def __str__(self):
        return self.name

