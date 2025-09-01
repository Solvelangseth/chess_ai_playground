import chess
import math
import time

class ChessAI:
    def __init__(self, depth=3, time_limit=5):
        """
        depth: Maximum search depth.
        time_limit: Maximum time (in seconds) to search.
        """
        self.depth = depth
        self.time_limit = time_limit
        self.transposition_table = {}
        
        # Material values (scaled to avoid float issues)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-Square Tables (from white's perspective).
        # For black, the table is mirrored vertically.
        self.pawn_table = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-20,-20, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ]
    
    def evaluate(self, board):
        """
        Evaluate the board using both material count and positional bonuses.
        Positive score favors White; negative favors Black.
        """
        if board.is_checkmate():
            # If checkmate, assign a large negative/positive value based on who is mated.
            return -99999 if board.turn else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        evaluation = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Base value for the piece.
                value = self.piece_values[piece.piece_type]
                # Add piece-square table bonus.
                value += self.get_piece_square_value(piece, square)
                evaluation += value if piece.color == chess.WHITE else -value
        
        # Mobility bonus: favor positions with more legal moves.
        mobility = len(list(board.legal_moves))
        evaluation += mobility if board.turn == chess.WHITE else -mobility

        return evaluation

    def get_piece_square_value(self, piece, square):
        """Return the bonus from the appropriate piece-square table."""
        if piece.piece_type == chess.PAWN:
            table = self.pawn_table
        elif piece.piece_type == chess.KNIGHT:
            table = self.knight_table
        elif piece.piece_type == chess.BISHOP:
            table = self.bishop_table
        elif piece.piece_type == chess.ROOK:
            table = self.rook_table
        elif piece.piece_type == chess.QUEEN:
            table = self.queen_table
        elif piece.piece_type == chess.KING:
            table = self.king_table
        else:
            return 0

        # For black pieces, mirror the table vertically.
        index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        return table[index]

    def choose_move(self, board):
        """
        Use iterative deepening with alpha-beta pruning to choose the best move.
        Stops searching when time runs out.
        """
        best_move = None
        start_time = time.time()
        current_depth = 1

        while current_depth <= self.depth:
            score, move = self.alphabeta(board, current_depth, -math.inf, math.inf, board.turn, start_time)
            # If time has expired, break out.
            if time.time() - start_time > self.time_limit:
                break
            if move is not None:
                best_move = move
            current_depth += 1
        return best_move

    def alphabeta(self, board, depth, alpha, beta, maximizing_player, start_time):
        """
        Alpha-beta pruning algorithm with a simple transposition table.
        Returns a tuple (evaluation, best_move).
        """
        # Check for time expiration.
        if time.time() - start_time > self.time_limit:
            return self.evaluate(board), None

        # Transposition table lookup using FEN string.
        board_key = board.fen()
        if board_key in self.transposition_table:
            stored_depth, stored_eval = self.transposition_table[board_key]
            if stored_depth >= depth:
                return stored_eval, None

        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in self.order_moves(board):
                board.push(move)
                eval, _ = self.alphabeta(board, depth - 1, alpha, beta, False, start_time)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff.
            self.transposition_table[board_key] = (depth, max_eval)
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in self.order_moves(board):
                board.push(move)
                eval, _ = self.alphabeta(board, depth - 1, alpha, beta, True, start_time)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff.
            self.transposition_table[board_key] = (depth, min_eval)
            return min_eval, best_move

    def order_moves(self, board):
        """
        Order moves to improve search efficiency.
        A simple heuristic: prioritize capture moves.
        """
        moves = list(board.legal_moves)
        # Capture moves come first.
        moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        return moves
