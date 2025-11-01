# game/go_board.py
import numpy as np
from typing import Tuple, Set, Optional, List

class GoBoard:
    """9x9 Go board with Chinese rules"""
    
    def __init__(self, size: int = 9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0=empty, 1=black, -1=white
        self.current_player = 1  # Black starts
        self.ko_point = None  # For ko rule
        self.history = []  # Board state history
        self.captured = {1: 0, -1: 0}  # Captured stones count
        
    def copy(self) -> 'GoBoard':
        """Deep copy of board state"""
        new_board = GoBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        new_board.captured = self.captured.copy()
        return new_board
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is legal"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row, col] != 0:
            return False
        if self.ko_point == (row, col):
            return False
        
        # Check for suicide (not allowed unless capturing)
        test_board = self.copy()
        test_board.board[row, col] = self.current_player
        
        # If move has liberties, it's valid
        if test_board._count_liberties(row, col) > 0:
            return True
        
        # Check if it captures opponent stones
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_r, adj_c = row + dr, col + dc
            if (0 <= adj_r < self.size and 0 <= adj_c < self.size and
                test_board.board[adj_r, adj_c] == -self.current_player):
                if test_board._count_liberties(adj_r, adj_c) == 0:
                    return True  # Captures opponent
        
        return False  # Suicide move
    
    def play_move(self, row: int, col: int) -> bool:
        """Execute a move, return True if successful"""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.ko_point = None
        
        # Capture opponent stones
        captured_stones = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_r, adj_c = row + dr, col + dc
            if (0 <= adj_r < self.size and 0 <= adj_c < self.size and
                self.board[adj_r, adj_c] == -self.current_player):
                if self._count_liberties(adj_r, adj_c) == 0:
                    captured = self._remove_group(adj_r, adj_c)
                    captured_stones.extend(captured)
        
        # Ko rule: if exactly one stone captured, mark ko point
        if len(captured_stones) == 1:
            self.ko_point = captured_stones[0]
        
        self.captured[self.current_player] += len(captured_stones)
        self.history.append(self.board.copy())
        self.current_player = -self.current_player
        return True
    
    def _count_liberties(self, row: int, col: int) -> int:
        """Count liberties (empty adjacent points) for a group"""
        color = self.board[row, col]
        if color == 0:
            return 0
        
        group = self._get_group(row, col)
        liberties = set()
        
        for r, c in group:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adj_r, adj_c = r + dr, c + dc
                if (0 <= adj_r < self.size and 0 <= adj_c < self.size and
                    self.board[adj_r, adj_c] == 0):
                    liberties.add((adj_r, adj_c))
        
        return len(liberties)
    
    def _get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all stones in the connected group"""
        color = self.board[row, col]
        if color == 0:
            return set()
        
        group = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adj_r, adj_c = r + dr, c + dc
                if (0 <= adj_r < self.size and 0 <= adj_c < self.size and
                    self.board[adj_r, adj_c] == color and (adj_r, adj_c) not in group):
                    stack.append((adj_r, adj_c))
        
        return group
    
    def _remove_group(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Remove a captured group, return positions"""
        group = self._get_group(row, col)
        for r, c in group:
            self.board[r, c] = 0
        return list(group)
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves for current player"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col):
                    moves.append((row, col))
        return moves
    
    def is_game_over(self) -> bool:
        """Simple end condition: no legal moves"""
        return len(self.get_legal_moves()) == 0