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
    
    def _get_territory(self, row: int, col: int, visited: set) -> Tuple[Set[Tuple[int, int]], int]:
        """
        Find connected empty region and determine owner
        Returns: (set of positions, owner)
        owner: 1 (black), -1 (white), 0 (neutral/dame)
        """
        territory = set()
        border_colors = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in territory or (r, c) in visited:
                continue
            if not (0 <= r < self.size and 0 <= c < self.size):
                continue
            
            if self.board[r, c] == 0:  # Empty
                territory.add((r, c))
                # Check all 4 neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
            else:  # Stone found at border
                border_colors.add(self.board[r, c])
        
        # Determine owner based on border colors
        if len(border_colors) == 1:
            owner = border_colors.pop()  # Territory belongs to this color
        else:
            owner = 0  # Neutral (borders both colors or no stones)
        
        return territory, owner

    def score_game(self, komi: float = 6.5) -> Tuple[float, float, str]:
        """
        Score the game using area scoring (Chinese rules)
        Returns: (black_score, white_score, winner)
        komi: compensation points for white (typically 6.5 or 7.5)
        """
        black_score = 0
        white_score = komi  # White gets komi compensation
        
        # Count stones and territory
        visited = set()
        
        for row in range(self.size):
            for col in range(self.size):
                if (row, col) in visited:
                    continue
                
                if self.board[row, col] == 1:  # Black stone
                    black_score += 1
                    visited.add((row, col))
                elif self.board[row, col] == -1:  # White stone
                    white_score += 1
                    visited.add((row, col))
                else:  # Empty point - determine territory
                    territory, owner = self._get_territory(row, col, visited)
                    if owner == 1:
                        black_score += len(territory)
                    elif owner == -1:
                        white_score += len(territory)
                    # If owner is 0, it's neutral territory (no points)
                    visited.update(territory)
        
        # Add captured stones to score
        black_score += self.captured[1]
        white_score += self.captured[-1]
        
        # Determine winner
        if black_score > white_score:
            winner = f"Black wins by {black_score - white_score:.1f}"
        elif white_score > black_score:
            winner = f"White wins by {white_score - black_score:.1f}"
        else:
            winner = "Draw"
        
        return black_score, white_score, winner
    
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