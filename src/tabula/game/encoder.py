import numpy as np
import chess

def encode_board(board):
    """
    Converts a python-chess board into a 3D numpy array (13 x 8 x 8).
    This is the exact style AlphaZero uses to 'see' the board.
    
    - Channels 0-5: White pieces (P, N, B, R, Q, K)
    - Channels 6-11: Black pieces (p, n, b, r, q, k)
    - Channel 12: Player turn (all 1s if White, all 0s if Black)
    """
    # Initialize an empty 13 x 8 x 8 tensor with zeros
    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    # Mapping pieces to their respective layer numbers
    piece_to_layer = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Loop through all 64 squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get the row (rank) and column (file) of the piece
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Determine the base layer for the piece type
            layer = piece_to_layer[piece.piece_type]

            # If the piece is black, shift it to layers 6-11
            if piece.color == chess.BLACK:
                layer += 6

            # Mark the piece on the board with a 1.0
            tensor[layer][rank][file] = 1.0

    # Channel 12: Whose turn is it?
    # We fill the entire 8x8 plane with 1s if it's White's turn
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    # (If black, it remains 0.0 because of the np.zeros initialization)

    return tensor

if __name__ == "__main__":
    print("=== TABULA AI: THE EYES (State Encoder) ===\n")
    
    # Let's test the encoder on the starting position
    b = chess.Board()
    encoded = encode_board(b)
    
    print(f"Original board:\n{b}\n")
    print(f"Encoded tensor shape: {encoded.shape} (13 layers, 8 rows, 8 columns)\n")
    
    print("Here is Layer 0 (White Pawns ONLY):")
    # Tensors print upside down compared to chess boards usually, but you'll see the line of 1s!
    print(encoded[0])
    
    print("\nHere is Layer 6 (Black Pawns ONLY):")
    print(encoded[6])
    
    print("\nHere is Layer 12 (Whose turn is it? 1 = White, 0 = Black):")
    print(encoded[12])
