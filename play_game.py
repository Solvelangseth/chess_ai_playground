import chess
import time
from ai import ChessAI
from ai2 import ChessAI2  # Assumes these classes exist in your ai modules

def play_game(ai_white, ai_black):
    """
    Plays a game between two AIs.
    Returns the final board, result string, and a list of move metrics.
    """
    board = chess.Board()
    move_metrics = []  # Each entry: (move_number, color, move_uci, time_taken)
    move_number = 0

    while not board.is_game_over():
        move_number += 1
        if board.turn == chess.WHITE:
            current_ai = ai_white
            color = "White"
        else:
            current_ai = ai_black
            color = "Black"
        
        start_time = time.time()
        move = current_ai.choose_move(board)
        move_time = time.time() - start_time

        if move is None:
            print(f"{color} AI did not return a move. Aborting game.")
            break

        board.push(move)
        move_metrics.append((move_number, color, move.uci(), move_time))
        print(f"Move {move_number}: {color} played {move.uci()} in {move_time:.3f} sec.")

    result = board.result()  # "1-0", "0-1", or "1/2-1/2"
    return board, result, move_metrics

def print_game_metrics(metrics):
    white_times = [t for _, color, _, t in metrics if color == "White"]
    black_times = [t for _, color, _, t in metrics if color == "Black"]
    total_moves = len(metrics)
    total_time_white = sum(white_times)
    total_time_black = sum(black_times)
    avg_time_white = total_time_white / len(white_times) if white_times else 0
    avg_time_black = total_time_black / len(black_times) if black_times else 0

    print("\n=== Game Metrics ===")
    print(f"Total moves: {total_moves}")
    print(f"White total time: {total_time_white:.3f} sec, average: {avg_time_white:.3f} sec per move")
    print(f"Black total time: {total_time_black:.3f} sec, average: {avg_time_black:.3f} sec per move")

def main():
    num_games = 10
    ai1_wins = 0
    ai2_wins = 0
    draws = 0

    # Create instances of your two AI classes.
    ai1 = ChessAI(depth=2, time_limit=5)   # ChessAI instance
    ai2 = ChessAI2(depth=4, time_limit = 10)                 # ChessAI2 instance

    for game_number in range(1, num_games + 1):
        print(f"\n=== Game {game_number} ===")
        # Switch sides each game:
        # Odd games: ai1 plays White, ai2 plays Black.
        # Even games: ai2 plays White, ai1 plays Black.
        if game_number % 2 == 1:
            ai_white = ai1
            ai_black = ai2
            print("White: ChessAI, Black: ChessAI2")
        else:
            ai_white = ai2
            ai_black = ai1
            print("White: ChessAI2, Black: ChessAI")
        
        final_board, result, metrics = play_game(ai_white, ai_black)
        print("\n=== Final Board Position ===")
        print(final_board)
        print("Game result:", result)
        
        # Determine the winner.
        # For odd games: "1-0" means ai1 wins; "0-1" means ai2 wins.
        # For even games: "1-0" means ai2 wins; "0-1" means ai1 wins.
        if result == "1-0":
            winner = "White"
        elif result == "0-1":
            winner = "Black"
        else:
            winner = "Draw"

        if game_number % 2 == 1:
            # Odd game: ai1 is White, ai2 is Black.
            if winner == "White":
                print("Winner: ChessAI")
                ai1_wins += 1
            elif winner == "Black":
                print("Winner: ChessAI2")
                ai2_wins += 1
            else:
                print("Game is a draw.")
                draws += 1
        else:
            # Even game: ai2 is White, ai1 is Black.
            if winner == "White":
                print("Winner: ChessAI2")
                ai2_wins += 1
            elif winner == "Black":
                print("Winner: ChessAI")
                ai1_wins += 1
            else:
                print("Game is a draw.")
                draws += 1

        print_game_metrics(metrics)
    
    print("\n=== Tournament Summary ===")
    print(f"Total games: {num_games}")
    print(f"ChessAI wins: {ai1_wins}")
    print(f"ChessAI2 wins: {ai2_wins}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    main()
