import chess
import math
import time
import random

class ChessAI2:
    def __init__(self, depth=4, time_limit=10):
        """
        Initialize an advanced chess AI.
        
        Args:
            depth: Maximum search depth (default: 4)
            time_limit: Maximum time in seconds for move selection (default: 10)
        """
        self.depth = depth
        self.time_limit = time_limit
        self.transposition_table = {}
        self.killer_moves = [[None for _ in range(2)] for _ in range(100)]  # Store killer moves by depth
        self.history_table = {}  # For history heuristic
        
        # Material values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-Square Tables (optimized)
        self.pawn_table = [
             0,   0,   0,   0,   0,   0,   0,   0,
            50,  50,  50,  50,  50,  50,  50,  50,
            10,  10,  20,  30,  30,  20,  10,  10,
             5,   5,  10,  25,  25,  10,   5,   5,
             0,   0,   0,  20,  20,   0,   0,   0,
             5,  -5, -10,   0,   0, -10,  -5,   5,
             5,  10,  10, -20, -20,  10,  10,   5,
             0,   0,   0,   0,   0,   0,   0,   0
        ]
        
        self.knight_table = [
           -50, -40, -30, -30, -30, -30, -40, -50,
           -40, -20,   0,   5,   5,   0, -20, -40,
           -30,   5,  10,  15,  15,  10,   5, -30,
           -30,   0,  15,  20,  20,  15,   0, -30,
           -30,   5,  15,  20,  20,  15,   5, -30,
           -30,   0,  10,  15,  15,  10,   0, -30,
           -40, -20,   0,   0,   0,   0, -20, -40,
           -50, -40, -30, -30, -30, -30, -40, -50
        ]
        
        self.bishop_table = [
           -20, -10, -10, -10, -10, -10, -10, -20,
           -10,   5,   0,   0,   0,   0,   5, -10,
           -10,  10,  10,  10,  10,  10,  10, -10,
           -10,   0,  10,  10,  10,  10,   0, -10,
           -10,   5,   5,  10,  10,   5,   5, -10,
           -10,   0,   5,  10,  10,   5,   0, -10,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -20, -10, -10, -10, -10, -10, -10, -20
        ]
        
        self.rook_table = [
              0,   0,   0,   5,   5,   0,   0,   0,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
              5,  10,  10,  10,  10,  10,  10,   5,
              0,   0,   0,   0,   0,   0,   0,   0
        ]
        
        self.queen_table = [
           -20, -10, -10,  -5,  -5, -10, -10, -20,
           -10,   0,   5,   0,   0,   0,   0, -10,
           -10,   5,   5,   5,   5,   5,   0, -10,
             0,   0,   5,   5,   5,   5,   0,  -5,
            -5,   0,   5,   5,   5,   5,   0,  -5,
           -10,   0,   5,   5,   5,   5,   0, -10,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -20, -10, -10,  -5,  -5, -10, -10, -20
        ]
        
        # King tables (midgame and endgame)
        self.king_middle_table = [
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -20, -30, -30, -40, -40, -30, -30, -20,
           -10, -20, -20, -20, -20, -20, -20, -10,
            20,  20,   0,   0,   0,   0,  20,  20,
            20,  30,  10,   0,   0,  10,  30,  20
        ]
        
        self.king_end_table = [
           -50, -40, -30, -20, -20, -30, -40, -50,
           -30, -20, -10,   0,   0, -10, -20, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -30,   0,   0,   0,   0, -30, -30,
           -50, -30, -30, -30, -30, -30, -30, -50
        ]
        
        # Advanced evaluation parameters
        self.pawn_structure_penalties = {
            'isolated': -10,
            'doubled': -15,
            'backward': -8,
            'passed_bonus': [0, 5, 10, 20, 35, 60, 100, 200]  # By rank
        }
        
        # Opening book - common strong first moves
        self.opening_book = {
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': ['e2e4', 'd2d4', 'g1f3', 'c2c4'],
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1': ['e7e5', 'c7c5', 'e7e6', 'c7c6']
        }
        
        # Endgame recognition threshold (total material excluding kings)
        self.endgame_threshold = 1500
    
    def evaluate(self, board):
        """
        Advanced position evaluation function.
        """
        if board.is_checkmate():
            return -99999 if board.turn else 99999
            
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
            
        # Material count
        white_material = 0
        black_material = 0
        
        # Piece counts for endgame detection
        white_pieces = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0}
        black_pieces = {'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0}
        
        # For pawn structure evaluation
        white_pawns_file = [0] * 8
        black_pawns_file = [0] * 8
        
        # Total material (for endgame detection)
        total_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            value = self.piece_values[piece.piece_type]
            
            # Track piece counts
            if piece.color == chess.WHITE:
                white_material += value
                symbol = piece.symbol().upper()
                if symbol in white_pieces:
                    white_pieces[symbol] += 1
                    
                # Track pawn files for structure evaluation
                if piece.piece_type == chess.PAWN:
                    file_idx = chess.square_file(square)
                    white_pawns_file[file_idx] += 1
            else:
                black_material += value
                symbol = piece.symbol().lower()
                if symbol in black_pieces:
                    black_pieces[symbol] += 1
                    
                # Track pawn files for structure evaluation
                if piece.piece_type == chess.PAWN:
                    file_idx = chess.square_file(square)
                    black_pawns_file[file_idx] += 1
            
            # Add positional bonuses
            positional_value = self.get_piece_square_value(piece, square, white_material + black_material)
            if piece.color == chess.WHITE:
                white_material += positional_value
            else:
                black_material += positional_value
                
            total_material += value
        
        # Phase detection (endgame vs midgame)
        is_endgame = total_material <= self.endgame_threshold
        
        # Basic material difference
        evaluation = white_material - black_material
        
        # Piece development and control (early game)
        if not is_endgame and white_pieces['P'] + black_pieces['p'] > 12:
            # Knights and bishops developed
            if board.piece_at(chess.B1) is None and white_pieces['N'] > 0:
                evaluation += 10  # Knight developed
            if board.piece_at(chess.G1) is None and white_pieces['N'] > 0:
                evaluation += 10  # Knight developed
            if board.piece_at(chess.C1) is None and white_pieces['B'] > 0:
                evaluation += 10  # Bishop developed
            if board.piece_at(chess.F1) is None and white_pieces['B'] > 0:
                evaluation += 10  # Bishop developed
                
            if board.piece_at(chess.B8) is None and black_pieces['n'] > 0:
                evaluation -= 10  # Knight developed
            if board.piece_at(chess.G8) is None and black_pieces['n'] > 0:
                evaluation -= 10  # Knight developed
            if board.piece_at(chess.C8) is None and black_pieces['b'] > 0:
                evaluation -= 10  # Bishop developed
            if board.piece_at(chess.F8) is None and black_pieces['b'] > 0:
                evaluation -= 10  # Bishop developed
            
            # Center control
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            for square in center_squares:
                # Count piece attacks on center
                attackers = board.attackers(chess.WHITE, square)
                evaluation += 5 * len(attackers)
                attackers = board.attackers(chess.BLACK, square)
                evaluation -= 5 * len(attackers)
        
        # Pawn structure evaluation
        evaluation += self.evaluate_pawn_structure(board, white_pawns_file, black_pawns_file)
        
        # King safety evaluation
        evaluation += self.evaluate_king_safety(board, is_endgame)
        
        # Mobility evaluation
        evaluation += self.evaluate_mobility(board)
        
        # Bishop pair bonus
        if white_pieces['B'] >= 2:
            evaluation += 30  # Bonus for having both bishops
        if black_pieces['b'] >= 2:
            evaluation -= 30
        
        # Rook on open file bonus
        evaluation += self.evaluate_rooks_on_open_files(board, white_pawns_file, black_pawns_file)
        
        # Endgame specific evaluations
        if is_endgame:
            evaluation += self.evaluate_endgame(board, white_pieces, black_pieces)
        
        # Return evaluation from white's perspective
        return evaluation
    
    def evaluate_pawn_structure(self, board, white_pawns_file, black_pawns_file):
        """Evaluate pawn structure"""
        score = 0
        
        # Isolated and doubled pawns
        for file_idx in range(8):
            # Check for white isolated pawns
            if white_pawns_file[file_idx] > 0:
                isolated = (file_idx == 0 or white_pawns_file[file_idx-1] == 0) and \
                           (file_idx == 7 or white_pawns_file[file_idx+1] == 0)
                if isolated:
                    score += self.pawn_structure_penalties['isolated'] * white_pawns_file[file_idx]
            
            # Check for black isolated pawns
            if black_pawns_file[file_idx] > 0:
                isolated = (file_idx == 0 or black_pawns_file[file_idx-1] == 0) and \
                           (file_idx == 7 or black_pawns_file[file_idx+1] == 0)
                if isolated:
                    score -= self.pawn_structure_penalties['isolated'] * black_pawns_file[file_idx]
            
            # Doubled pawns penalty
            if white_pawns_file[file_idx] > 1:
                score += self.pawn_structure_penalties['doubled'] * (white_pawns_file[file_idx] - 1)
            if black_pawns_file[file_idx] > 1:
                score -= self.pawn_structure_penalties['doubled'] * (black_pawns_file[file_idx] - 1)
        
        # Passed pawns
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                if piece.color == chess.WHITE:
                    passed = True
                    # Check if any black pawns can block this pawn
                    for f in range(max(0, file-1), min(8, file+2)):
                        for r in range(rank+1, 8):
                            s = chess.square(f, r)
                            p = board.piece_at(s)
                            if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                                passed = False
                                break
                        if not passed:
                            break
                    
                    if passed:
                        score += self.pawn_structure_penalties['passed_bonus'][rank]
                else:
                    passed = True
                    # Check if any white pawns can block this pawn
                    for f in range(max(0, file-1), min(8, file+2)):
                        for r in range(0, rank):
                            s = chess.square(f, r)
                            p = board.piece_at(s)
                            if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                                passed = False
                                break
                        if not passed:
                            break
                    
                    if passed:
                        score -= self.pawn_structure_penalties['passed_bonus'][7-rank]
        
        return score
    
    def evaluate_king_safety(self, board, is_endgame):
        """Evaluate king safety based on game phase"""
        score = 0
        
        # Get king positions
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if not is_endgame:
            # In midgame, kings should be castled and protected
            white_king_file = chess.square_file(white_king_square)
            black_king_file = chess.square_file(black_king_square)
            
            # Reward castling (king on g1/c1 or g8/c8)
            if white_king_file in [2, 6] and chess.square_rank(white_king_square) == 0:
                score += 50  # Castled king
            if black_king_file in [2, 6] and chess.square_rank(black_king_square) == 7:
                score -= 50  # Castled king
            
            # Penalize king in the center
            if white_king_file in [3, 4]:
                score -= 50
            if black_king_file in [3, 4]:
                score += 50
            
            # Penalize exposed king (attacks around king)
            for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
                try:
                    # White king safety
                    adjacent_square = white_king_square + offset
                    if 0 <= adjacent_square < 64:
                        attackers = board.attackers(chess.BLACK, adjacent_square)
                        score -= 10 * len(attackers)
                    
                    # Black king safety
                    adjacent_square = black_king_square + offset
                    if 0 <= adjacent_square < 64:
                        attackers = board.attackers(chess.WHITE, adjacent_square)
                        score += 10 * len(attackers)
                except ValueError:
                    pass
        else:
            # In endgame, king should be active and centralized
            white_king_file = chess.square_file(white_king_square)
            white_king_rank = chess.square_rank(white_king_square)
            black_king_file = chess.square_file(black_king_square)
            black_king_rank = chess.square_rank(black_king_square)
            
            white_file_distance = abs(white_king_file - 3.5)
            white_rank_distance = abs(white_king_rank - 3.5)
            black_file_distance = abs(black_king_file - 3.5)
            black_rank_distance = abs(black_king_rank - 3.5)
            
            white_distance = white_file_distance + white_rank_distance
            black_distance = black_file_distance + black_rank_distance
            
            score += (7 - white_distance) * 10
            score -= (7 - black_distance) * 10
        
        return score
    
    def evaluate_mobility(self, board):
        """Evaluate piece mobility"""
        original_turn = board.turn
        
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        board.turn = original_turn
        
        return (white_mobility - black_mobility) * 3
    
    def evaluate_rooks_on_open_files(self, board, white_pawns_file, black_pawns_file):
        """Bonus for rooks on open or semi-open files"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.ROOK:
                file_idx = chess.square_file(square)
                
                if white_pawns_file[file_idx] == 0 and black_pawns_file[file_idx] == 0:
                    if piece.color == chess.WHITE:
                        score += 20
                    else:
                        score -= 20
                elif (piece.color == chess.WHITE and white_pawns_file[file_idx] == 0) or \
                     (piece.color == chess.BLACK and black_pawns_file[file_idx] == 0):
                    if piece.color == chess.WHITE:
                        score += 10
                    else:
                        score -= 10
        
        return score
    
    def evaluate_endgame(self, board, white_pieces, black_pieces):
        """Special endgame evaluations"""
        score = 0
        
        # Pushing pawns forward in endgame
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.piece_type != chess.PAWN:
                continue
                
            rank = chess.square_rank(square)
            if piece.color == chess.WHITE:
                score += rank * 5
            else:
                score -= (7 - rank) * 5
        
        # Encourage trading when ahead in material
        material_difference = sum(white_pieces.values()) - sum(black_pieces.values())
        if material_difference > 3:
            score += 50
        elif material_difference < -3:
            score -= 50
        
        return score
    
    def get_piece_square_value(self, piece, square, total_material):
        """
        Get the positional value for a piece based on the piece-square tables.
        Adjust based on game phase (early/mid/endgame).
        """
        if piece.color == chess.BLACK:
            square = chess.square_mirror(square)
        
        is_endgame = total_material <= self.endgame_threshold
        
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[square]
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[square]
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[square]
        elif piece.piece_type == chess.ROOK:
            return self.rook_table[square]
        elif piece.piece_type == chess.QUEEN:
            return self.queen_table[square]
        elif piece.piece_type == chess.KING:
            if is_endgame:
                return self.king_end_table[square]
            else:
                return self.king_middle_table[square]
        
        return 0
    
    def choose_move(self, board):
        """
        Choose the best move using:
         1. An opening book (if available)
         2. Iterative deepening with alpha-beta pruning (via negamax)
        """
        board_fen = board.fen()
        if board_fen in self.opening_book:
            book_moves = self.opening_book[board_fen]
            selected_move = random.choice(book_moves)
            return chess.Move.from_uci(selected_move)
        
        start_time = time.time()
        best_move = None
        
        for current_depth in range(1, self.depth + 1):
            self.killer_moves = [[None for _ in range(2)] for _ in range(100)]
            score, move = self.negamax(board, current_depth, -math.inf, math.inf, 1, start_time)
            
            if time.time() - start_time > self.time_limit:
                break
                
            if move is not None:
                best_move = move
        
        if best_move is None and len(list(board.legal_moves)) > 0:
            best_move = random.choice(list(board.legal_moves))
            
        return best_move
    
    def negamax(self, board, depth, alpha, beta, color, start_time):
        """
        Negamax search with alpha-beta pruning and various optimizations.
        
        Returns:
            (score, best_move)
        """
        if time.time() - start_time > self.time_limit:
            return 0, None
            
        board_key = board.fen()
        alphaOrig = alpha
        if board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['score'], entry['best_move']
                elif entry['flag'] == 'LOWERBOUND':
                    alpha = max(alpha, entry['score'])
                elif entry['flag'] == 'UPPERBOUND':
                    beta = min(beta, entry['score'])
                
                if alpha >= beta:
                    return entry['score'], entry['best_move']
        
        if depth == 0 or board.is_game_over():
            if not board.is_game_over() and depth == 0:
                return self.quiescence_search(board, alpha, beta, color, start_time), None
            return color * self.evaluate(board), None
        
        moves = self.order_moves(board, depth)
        best_move = None
        max_score = -math.inf
        
        for move in moves:
            board.push(move)
            score, _ = self.negamax(board, depth - 1, -beta, -alpha, -color, start_time)
            score = -score
            board.pop()
            
            if time.time() - start_time > self.time_limit:
                return 0, None
            
            if score > max_score:
                max_score = score
                best_move = move
                
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move):
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                
                from_square = move.from_square
                to_square = move.to_square
                key = (from_square, to_square)
                if key not in self.history_table:
                    self.history_table[key] = 0
                self.history_table[key] += depth * depth
                
                break
        
        if best_move is not None:
            entry = {
                'score': max_score,
                'best_move': best_move,
                'depth': depth
            }
            
            if max_score <= alphaOrig:
                entry['flag'] = 'UPPERBOUND'
            elif max_score >= beta:
                entry['flag'] = 'LOWERBOUND'
            else:
                entry['flag'] = 'EXACT'
                
            self.transposition_table[board_key] = entry
        
        return max_score, best_move
    
    def quiescence_search(self, board, alpha, beta, color, start_time):
        """
        Quiescence search to evaluate capture sequences.
        """
        if time.time() - start_time > self.time_limit:
            return 0
        
        stand_pat = color * self.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        captures = self.order_captures(board, captures)
        
        for move in captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, -color, start_time)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
    
    def order_moves(self, board, depth):
        """
        Order moves for efficient pruning:
         1. PV move from transposition table
         2. Captures (by MVV-LVA)
         3. Killer moves
         4. History heuristic moves
         5. Others
        """
        legal_moves = list(board.legal_moves)
        scored_moves = []
        
        board_key = board.fen()
        tt_move = None
        if board_key in self.transposition_table:
            tt_move = self.transposition_table[board_key].get('best_move', None)
        
        for move in legal_moves:
            score = 0
            if tt_move and move == tt_move:
                score += 10000  # High bonus for transposition move
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                victim_value = self.piece_values[victim.piece_type] if victim else 0
                attacker_value = self.piece_values[attacker.piece_type] if attacker else 0
                score += (victim_value - attacker_value) * 10
            if move in self.killer_moves[depth]:
                score += 500
            key = (move.from_square, move.to_square)
            if key in self.history_table:
                score += self.history_table[key]
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def order_captures(self, board, moves):
        """
        Order capture moves using MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        """
        scored_moves = []
        for move in moves:
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_value = self.piece_values[victim.piece_type] if victim else 0
            attacker_value = self.piece_values[attacker.piece_type] if attacker else 0
            score = victim_value - attacker_value
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
