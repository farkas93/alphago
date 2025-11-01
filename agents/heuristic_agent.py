# agents/heuristic_agent.py
import random
from agents.base_agent import Agent
from game.go_board import GoBoard
from typing import Tuple, Optional, List

class HeuristicAgent(Agent):
    """
    Simple heuristic-based agent that:
    1. Captures opponent stones if possible
    2. Saves own stones from capture
    3. Otherwise plays randomly
    """
    
    def __init__(self):
        super().__init__("Heuristic Agent")
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Select move based on simple heuristics"""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        
        # Priority 1: Capture opponent stones
        capture_moves = self._find_capture_moves(board, legal_moves)
        if capture_moves:
            return random.choice(capture_moves)
        
        # Priority 2: Save own stones from atari (1 liberty)
        save_moves = self._find_save_moves(board, legal_moves)
        if save_moves:
            return random.choice(save_moves)
        
        # Priority 3: Avoid self-atari (don't put own stones in danger)
        safe_moves = self._avoid_self_atari(board, legal_moves)
        if safe_moves:
            return random.choice(safe_moves)
        
        # Fallback: random move
        return random.choice(legal_moves)
    
    def _find_capture_moves(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find moves that capture opponent stones"""
        capture_moves = []
        
        for move in legal_moves:
            # Simulate the move
            test_board = board.copy()
            test_board.play_move(move[0], move[1])
            
            # Check if we captured anything (captured count increased)
            if test_board.captured[board.current_player] > board.captured[board.current_player]:
                capture_moves.append(move)
        
        return capture_moves
    
    def _find_save_moves(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find moves that save friendly stones in atari"""
        save_moves = []
        current_player = board.current_player
        
        # Find all friendly groups in atari (1 liberty)
        groups_in_atari = []
        visited = set()
        
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row, col] == current_player and (row, col) not in visited:
                    group = board._get_group(row, col)
                    visited.update(group)
                    if board._count_liberties(row, col) == 1:
                        groups_in_atari.append(group)
        
        # Find moves that add liberties to groups in atari
        for move in legal_moves:
            for group in groups_in_atari:
                # Check if move is adjacent to group
                for stone_row, stone_col in group:
                    if abs(move[0] - stone_row) + abs(move[1] - stone_col) == 1:
                        save_moves.append(move)
                        break
        
        return save_moves
    
    def _avoid_self_atari(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter out moves that put own stones in atari"""
        safe_moves = []
        
        for move in legal_moves:
            test_board = board.copy()
            test_board.board[move[0], move[1]] = board.current_player
            
            # Check if this move results in 1 liberty (atari)
            liberties = test_board._count_liberties(move[0], move[1])
            
            # Allow atari if it captures opponent
            is_capture = False
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adj_r, adj_c = move[0] + dr, move[1] + dc
                if (0 <= adj_r < board.size and 0 <= adj_c < board.size and
                    board.board[adj_r, adj_c] == -board.current_player):
                    if board._count_liberties(adj_r, adj_c) == 1:
                        is_capture = True
                        break
            
            if liberties > 1 or is_capture:
                safe_moves.append(move)
        
        return safe_moves if safe_moves else legal_moves