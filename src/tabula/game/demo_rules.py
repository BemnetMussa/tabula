import chess

print("=== TABULA AI: HOW THE ENGINE ENFORCES RULES ===")

print("\n--- 1. THE STARTING POSITION ---")
board = chess.Board()
print(board)
print(f"\nTotal legal moves for White right now: {board.legal_moves.count()}")
print("If you look at the list of legal moves, the engine only allows pawns (forward) and knights (L-shape). It knows bishops and rooks are blocked.")

print("\n--- 2. ENFORCING PINS (The Knight is Trapped) ---")
# I am setting up a custom board using FEN notation.
# The White Knight (N) is standing between the White King (K) and the Black Rook (r).
board_pinned = chess.Board("4r3/8/8/8/4N3/8/8/4K3 w - - 0 1")
print(board_pinned)
print("\nWhite to move. Can the White Knight jump away?")
print(f"List of legal moves: {[str(m) for m in board_pinned.legal_moves]}")
print("Result: NO. The engine knows moving the Knight would expose the King to check, so it's strictly illegal.")

print("\n--- 3. ATTEMPTING AN ILLEGAL MOVE ---")
print("What happens if we forcefully try to move the White King directly into the Rook's path (King to d1)?")
try:
    board_pinned.push_san("Kd1") # Trying to move King to d1
except chess.IllegalMoveError as e:
    print(f"🚨 ENGINE BLOCKED MOVE: {e} - You cannot move into check!")

print("\n--- 4. RECOGNIZING CHECKMATE ---")
# This is the board state for the famous 'Fool's Mate' (fastest possible checkmate)
board_mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
print(board_mate)
print(f"\nIs it a draw? {board_mate.is_stalemate()}")
print(f"Is it checkmate? {board_mate.is_checkmate()}")
print(f"Who won? {'Black' if board_mate.outcome().winner == chess.BLACK else 'White'}")
