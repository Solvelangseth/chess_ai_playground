import glob
import os
import torch
import chess.pgn
import chess.engine
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Replace with your Stockfish executable path

def get_stockfish_evaluation(fen, engine, cache):
    """
    Uses Stockfish to evaluate a FEN position.
    Returns a normalized evaluation between -1 and 1 and the best move in SAN.
    Caches results in 'cache' to avoid redundant evaluations.
    """
    if fen in cache:
        return cache[fen]
    
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=15))  # Adjust depth as needed
    score = info["score"].relative  # Relative centipawn score
    
    pv = info.get("pv")
    best_move = board.san(pv[0]) if pv and len(pv) > 0 else None
    
    if score.is_mate():
        evaluation = 1.0 if score.mate() > 0 else -1.0
    else:
        evaluation = max(min(score.score() / 1000.0, 1.0), -1.0)
    
    result = (evaluation, best_move)
    cache[fen] = result
    return result

def fen_to_tensor(fen):
    """
    Converts a FEN string into a 12x8x8 PyTorch tensor representation.
    """
    board = chess.Board(fen)
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # White pieces
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11  # Black pieces
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row, col = divmod(square, 8)
            tensor[piece_map[piece.symbol()], row, col] = 1
    return tensor

def extract_fens_with_evaluations(pgn_file, engine, cache):
    """
    Reads a PGN file, extracts FENs, evaluates them with Stockfish,
    and converts them into tensors.
    Returns a list of (tensor, evaluation, best move) tuples.
    """
    data = []
    with open(pgn_file, "r") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # No more games in the file
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                eval_score, best_move = get_stockfish_evaluation(fen, engine, cache)
                tensor = fen_to_tensor(fen)
                data.append((tensor, eval_score, best_move))
    return data

def process_single_file(pgn_file, output_dir, cache):
    """
    Processes a single PGN file:
      - Opens its own Stockfish engine instance.
      - Evaluates each position (with caching) and converts FENs to tensors.
      - Saves the processed data to the output directory.
    Returns the output filename.
    """
    # Create an engine instance in this process.
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        training_data = extract_fens_with_evaluations(pgn_file, engine, cache)
    
    base_filename = os.path.splitext(os.path.basename(pgn_file))[0]
    output_filename = os.path.join(output_dir, base_filename + ".pt")
    torch.save(training_data, output_filename)
    return output_filename

def process_files():
    DATA_DIR = "data"
    OUTPUT_DIR = "tensor_data"

    # Create the output folder if it doesn't exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all PGN files inside the 'data' folder.
    pgn_files = glob.glob(os.path.join(DATA_DIR, "*.pgn"))
    print(f"Found {len(pgn_files)} PGN file(s) in '{DATA_DIR}'.")

    # Create a shared cache for evaluations using a Manager.
    with Manager() as manager:
        evaluation_cache = manager.dict()

        # Use a ProcessPoolExecutor to process multiple files in parallel.
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_file, pgn_file, OUTPUT_DIR, evaluation_cache): pgn_file
                for pgn_file in pgn_files
            }
            for future in futures:
                try:
                    output_filename = future.result()
                    print(f"Saved training data to: {output_filename}")
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")

if __name__ == "__main__":
    process_files()
