import sys
import chess
from ai import ChessAI  # Your advanced AI class

def uci_loop():
    board = chess.Board()
    engine = ChessAI(depth=4, time_limit=10)
    
    while True:
        line = sys.stdin.readline().strip()
        
        if line == "uci":
            print("id name MyAdvancedEngine")
            print("id author YourName")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("position"):
            # Handle both startpos and FEN input
            parts = line.split()
            if parts[1] == "startpos":
                board = chess.Board()
                if "moves" in parts:
                    moves_index = parts.index("moves") + 1
                    for move in parts[moves_index:]:
                        board.push(chess.Move.from_uci(move))
            elif parts[1] == "fen":
                # Assuming FEN is everything after "fen"
                fen = line.partition("fen")[2].strip()
                board = chess.Board(fen)
        elif line.startswith("go"):
            move = engine.choose_move(board)
            if move:
                print("bestmove", move.uci())
            else:
                print("bestmove 0000")
        elif line == "quit":
            break

if __name__ == "__main__":
    uci_loop()