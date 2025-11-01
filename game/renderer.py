# game/renderer.py
import pygame
import numpy as np
from game.go_board import GoBoard
from typing import Optional, Tuple
from agents.base_agent import Agent

class GoRenderer:
    """Pygame-based Go board renderer with mouse interaction"""
    
    def __init__(self, board: GoBoard, cell_size: int = 60, 
                 black_agent: Optional['Agent'] = None,
                 white_agent: Optional['Agent'] = None):
        self.board = board
        self.cell_size = cell_size
        self.margin = cell_size
        self.width = board.size * cell_size + 2 * self.margin
        self.height = board.size * cell_size + 2 * self.margin + 100  # Extra space for info

        self.black_agent = black_agent
        self.white_agent = white_agent
        self.ai_move_delay = 500  # milliseconds delay for AI moves
        self.last_ai_move_time = 0
        
        # Colors
        self.COLOR_BG = (220, 179, 92)  # Wood color
        self.COLOR_LINE = (0, 0, 0)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_LEGAL = (0, 255, 0, 100)  # Transparent green
        self.COLOR_LAST_MOVE = (255, 0, 0)
        self.COLOR_GAME_OVER = (255, 0, 0)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('AlphaGo 9x9 - Training Board')
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.show_legal_moves = True
        self.last_move = None
        self.game_over = False 
        
    def draw_board(self):
        """Draw the Go board"""
        self.screen.fill(self.COLOR_BG)
        
        # Draw grid lines
        for i in range(self.board.size):
            # Vertical lines
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_LINE, 
                           (x, self.margin), 
                           (x, self.margin + (self.board.size - 1) * self.cell_size), 2)
            # Horizontal lines
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_LINE,
                           (self.margin, y),
                           (self.margin + (self.board.size - 1) * self.cell_size, y), 2)
        
        # Draw star points (for 9x9: center and 4 corners near edges)
        star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for row, col in star_points:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.COLOR_LINE, (x, y), 4)
        
        # Draw stones
        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.board[row, col] != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = self.COLOR_BLACK if self.board.board[row, col] == 1 else self.COLOR_WHITE
                    pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 2 - 3)
                    
                    # Draw border for white stones
                    if self.board.board[row, col] == -1:
                        pygame.draw.circle(self.screen, self.COLOR_LINE, (x, y), 
                                         self.cell_size // 2 - 3, 2)
        
        # Highlight last move
        if self.last_move:
            row, col = self.last_move
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.COLOR_LAST_MOVE, (x, y), 8, 3)
        
        # Show legal moves (small dots)
        if self.show_legal_moves:
            legal_moves = self.board.get_legal_moves()
            for row, col in legal_moves:
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                pygame.draw.circle(self.screen, self.COLOR_LEGAL, (x, y), 5)
        
        # Draw info panel
        self._draw_info_panel()
        
        pygame.display.flip()
    
    def _draw_info_panel(self):
        """Draw game information at bottom"""
        y_start = self.margin + self.board.size * self.cell_size + 20
        
        # Game over message
        if self.game_over:
            text = self.font.render("GAME OVER!", True, self.COLOR_GAME_OVER)
            text_rect = text.get_rect(center=(self.width // 2, y_start))
            self.screen.blit(text, text_rect)
            y_start += 40
        else:
            # Current player with agent name if applicable
            player_text = "Black" if self.board.current_player == 1 else "White"
            agent = self.black_agent if self.board.current_player == 1 else self.white_agent
            if agent:
                player_text += f" ({agent.name})"  # Show agent name
            text = self.font.render(f"Turn: {player_text}", True, self.COLOR_LINE)
            self.screen.blit(text, (20, y_start))
        
        # Captures
        captures_text = f"Captures - B: {self.board.captured[1]}  W: {self.board.captured[-1]}"
        text = self.small_font.render(captures_text, True, self.COLOR_LINE)
        self.screen.blit(text, (20, y_start + 40))
        
        # Instructions
        inst_text = "Click to play | L: Toggle legal moves | R: Reset | ESC: Quit"
        text = self.small_font.render(inst_text, True, self.COLOR_LINE)
        self.screen.blit(text, (20, y_start + 65))
    
    def _make_ai_move(self):
        """Let AI agent make a move if it's their turn"""
        if self.game_over:
            return
        
        current_agent = None
        if self.board.current_player == 1 and self.black_agent:
            current_agent = self.black_agent
        elif self.board.current_player == -1 and self.white_agent:
            current_agent = self.white_agent
        
        if current_agent:
            move = current_agent.select_move(self.board)
            if move:
                row, col = move
                if self.board.play_move(row, col):
                    self.last_move = (row, col)
                    player = "Black" if self.board.current_player == -1 else "White"
                    print(f"Move played: {player} ({current_agent.name}) at ({row}, {col})")
                    
                    if self.board.is_game_over():
                        self.game_over = True
                        print("Game Over! No more legal moves.")
            else:
                self.game_over = True
                print("Game Over! Agent has no moves.")

    def get_board_position(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to board coordinates"""
        x, y = mouse_pos
        
        # Check if click is within board area
        if (x < self.margin - self.cell_size // 2 or 
            x > self.margin + self.board.size * self.cell_size - self.cell_size // 2 or
            y < self.margin - self.cell_size // 2 or 
            y > self.margin + self.board.size * self.cell_size - self.cell_size // 2):
            return None
        
        # Find nearest intersection
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        
        # Clamp to board boundaries
        row = max(0, min(self.board.size - 1, row))
        col = max(0, min(self.board.size - 1, col))
        
        return (row, col)
    
    
    def run_interactive(self):
        """Run interactive game loop with optional AI agents"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_l:
                        self.show_legal_moves = not self.show_legal_moves
                    elif event.key == pygame.K_r:
                        self.board = GoBoard(self.board.size)
                        self.last_move = None
                        self.game_over = False
                        self.last_ai_move_time = current_time  # Reset AI timer
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.game_over:
                        # Only allow human moves if no agent is controlling current player
                        current_agent = (self.black_agent if self.board.current_player == 1 
                                       else self.white_agent)
                        if not current_agent:  # Human player
                            pos = self.get_board_position(event.pos)
                            if pos:
                                row, col = pos
                                if self.board.play_move(row, col):
                                    self.last_move = (row, col)
                                    player = "Black" if self.board.current_player == -1 else "White"
                                    print(f"Move played: {player} at ({row}, {col})")
                                    
                                    if self.board.is_game_over():
                                        self.game_over = True
                                        print("Game Over! No more legal moves.")
                                    else:
                                        self.last_ai_move_time = current_time  # Start AI timer
            
            # AI move with delay for visibility
            if not self.game_over and current_time - self.last_ai_move_time > self.ai_move_delay:
                current_agent = (self.black_agent if self.board.current_player == 1 
                               else self.white_agent)
                if current_agent:
                    self._make_ai_move()
                    self.last_ai_move_time = current_time
            
            self.draw_board()
            clock.tick(30)
        
        pygame.quit()


# Example usage / test script
if __name__ == "__main__":
    from game.go_board import GoBoard
    from game.features import FeatureExtractor
    
    # Create board
    board = GoBoard(size=9)
    
    # Test features
    extractor = FeatureExtractor()
    features = extractor.extract_features(board)
    print(f"Feature shape: {features.shape}")  # Should be (17, 9, 9)
    
    # Launch interactive game
    renderer = GoRenderer(board)
    renderer.run_interactive()