import chess
import random

def random_rollout(display=False):
    """
    Plays a completely random game of chess to completion.
    Returns the terminal state label z from White's perspective:
    +1 (White wins), -1 (Black wins), 0 (Draw)
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over():
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Pick a random move (our "policy" is completely random right now)
        move = random.choice(legal_moves)
        board.push(move)
        move_count += 1

        if display:
            print(f"\nMove {move_count}: {move}")
            print(board)

    # Game is over. Determine the outcome (the ground truth RL signal).
    outcome = board.outcome()

    if outcome.winner is chess.WHITE:
        z = 1
    elif outcome.winner is chess.BLACK:
        z = -1
    else:
        z = 0 # Draw (stalemate, repetition, 50-move rule, etc.)

    reason = outcome.termination.name
    return z, reason, board, move_count

if __name__ == "__main__":
    print("Tabula AI - Milestone 1: Random Rollout Engine")
    print("Starting a completely random game...\n")
    
    z, reason, final_board, total_moves = random_rollout(display=False)
    
    print(final_board)
    print("\n=== TERMINAL STATE REACHED ===")
    print(f"Total Moves Played:   {total_moves}")
    print(f"Termination Reason:   {reason}")
    print(f"Label (z) for White:  {z}")
    
    if z == 1:
        print("Result: White Wins!")
    elif z == -1:
        print("Result: Black Wins!")
    else:
        print("Result: Draw!")
