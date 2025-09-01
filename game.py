# game.py
import pygame
import chess
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, SQUARE_SIZE, WHITE_COLOR, BLACK_COLOR, HIGHLIGHT_COLOR, MOVE_HIGHLIGHT_COLOR, FPS
from ai import ChessAI
from ai2 import ChessAI2

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.board = chess.Board()  # python-chess board object
        self.selected_square = None  # Selected square as (row, col)
        self.valid_moves = []        # List of valid move target squares for the selected piece
        self.ai = ChessAI(depth=5, time_limit=7)   # AI opponent (adjust depth as needed)
        self.piece_images = self.load_images()

    def load_images(self):
        """
        Load piece images from the assets folder using full names.
        Mapping between python‑chess symbols and full names:
          White: 'P' (pawn), 'N' (knight), 'B' (bishop), 'R' (rook), 'Q' (queen), 'K' (king)
          Black: 'p', 'n', 'b', 'r', 'q', 'k'
        Files are expected to be named like:
          assets/white-pawn.png, assets/black-queen.png, etc.
        """
        piece_images = {}
        mapping = {
            'P': 'pawn',
            'N': 'knight',
            'B': 'bishop',
            'R': 'rook',
            'Q': 'queen',
            'K': 'king',
            'p': 'pawn',
            'n': 'knight',
            'b': 'bishop',
            'r': 'rook',
            'q': 'queen',
            'k': 'king'
        }
        for symbol, name in mapping.items():
            color = "white" if symbol.isupper() else "black"
            image_path = f"assets/{color}-{name}.png"
            piece_images[symbol] = pygame.transform.scale(
                pygame.image.load(image_path), (SQUARE_SIZE, SQUARE_SIZE)
            )
        return piece_images

    def draw_board(self):
        """
        Draws the chess board with alternating square colors.
        Also highlights the selected square and valid move squares.
        """
        for row in range(8):
            for col in range(8):
                color = WHITE_COLOR if (row + col) % 2 == 0 else BLACK_COLOR
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                # Highlight the selected square
                if self.selected_square == (row, col):
                    pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 3)
                # Highlight valid move target squares
                if (row, col) in self.valid_moves:
                    pygame.draw.rect(self.screen, MOVE_HIGHLIGHT_COLOR, rect, 3)

    def draw_pieces(self):
        """
        Draw each piece on the board.
        Converts python‑chess square indices to board (row, col) positions.
        """
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)  # Flip rank so that rank 1 is at the bottom
                symbol = piece.symbol()  # e.g., 'P' for white pawn, 'q' for black queen
                if symbol in self.piece_images:
                    self.screen.blit(self.piece_images[symbol], (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def get_square_under_mouse(self, pos):
        """
        Convert a pixel position to board coordinates (row, col).
        """
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return (row, col)

    def handle_player_move(self, pos):
        """
        Handles a mouse click: selects a piece or attempts a move.
        If no piece is selected, it selects a white piece.
        If a piece is already selected, it attempts to move to the clicked square.
        """
        row, col = self.get_square_under_mouse(pos)
        clicked_square = chess.square(col, 7 - row)  # Convert to python‑chess square index

        if self.selected_square is None:
            # No piece selected yet; try to select a white piece.
            piece = self.board.piece_at(clicked_square)
            if piece is not None and piece.color == chess.WHITE:
                self.selected_square = (row, col)
                self.valid_moves = []
                for move in self.board.legal_moves:
                    if move.from_square == clicked_square:
                        target_row = 7 - chess.square_rank(move.to_square)
                        target_col = chess.square_file(move.to_square)
                        self.valid_moves.append((target_row, target_col))
        else:
            # A piece is already selected; try to make a move.
            from_square = chess.square(self.selected_square[1], 7 - self.selected_square[0])
            move = chess.Move(from_square, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.valid_moves = []
                # Let the AI make its move if the game is not over.
                if not self.board.is_game_over():
                    ai_move = self.ai.choose_move(self.board)
                    self.board.push(ai_move)
            else:
                # If the move is illegal, clear the selection.
                self.selected_square = None
                self.valid_moves = []

    def run(self):
        """
        Main game loop: handle events, update display, and cap the frame rate.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_player_move(pygame.mouse.get_pos())

            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()
            self.clock.tick(FPS)
