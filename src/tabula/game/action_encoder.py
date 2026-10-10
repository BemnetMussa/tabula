import chess
import numpy as np

class ActionEncoder:
    """
    Translates python-chess Moves into an integer index [0, 4671] for AlphaZero.
    AlphaZero represents moves as a stack of 73 planes of 8x8 grids (73 * 64 = 4672).
    """
    
    def __init__(self):
        # 8 compass directions for queen-like moves
        self.directions = [
            (0, 1),   # N
            (1, 1),   # NE
            (1, 0),   # E
            (1, -1),  # SE
            (0, -1),  # S
            (-1, -1), # SW
            (-1, 0),  # W
            (-1, 1)   # NW
        ]
        
        # 8 knight jump directions
        self.knight_jumps = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        
        # Promotion pieces (Queen is handled as a standard compass move)
        self.promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        
    def encode(self, move: chess.Move) -> int:
        """Converts a chess.Move into an index between 0 and 4671."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
        
        delta_rank = to_rank - from_rank
        delta_file = to_file - from_file
        
        # 1. Underpromotions (Planes 64-72)
        if move.promotion and move.promotion != chess.QUEEN:
            # delta_file is -1 (capture left), 0 (forward), or +1 (capture right)
            promo_idx = self.promotions.index(move.promotion)
            dir_idx = delta_file + 1 # maps -1, 0, 1 to 0, 1, 2
            plane = 64 + (promo_idx * 3) + dir_idx
            return plane * 64 + from_sq
            
        # 2. Knight Moves (Planes 56-63)
        for i, (df, dr) in enumerate(self.knight_jumps):
            if delta_file == df and delta_rank == dr:
                plane = 56 + i
                return plane * 64 + from_sq
                
        # 3. Queen-like moves (Planes 0-55)
        # Find the compass direction
        distance = max(abs(delta_rank), abs(delta_file))
        df_norm = delta_file // distance if distance != 0 else 0
        dr_norm = delta_rank // distance if distance != 0 else 0
        
        for i, (df, dr) in enumerate(self.directions):
            if df_norm == df and dr_norm == dr:
                # distance is 1-7. Plane is (direction * 7) + (distance - 1)
                plane = (i * 7) + (distance - 1)
                return plane * 64 + from_sq
                
        raise ValueError(f"Could not encode move {move}")

    def decode(self, index: int) -> chess.Move:
        """Converts an index [0, 4671] back into a chess.Move."""
        from_sq = index % 64
        plane = index // 64
        
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        
        # 1. Queen-like moves (0-55)
        if plane < 56:
            dir_idx = plane // 7
            distance = (plane % 7) + 1
            df, dr = self.directions[dir_idx]
            to_file = from_file + (df * distance)
            to_rank = from_rank + (dr * distance)
            to_sq = chess.square(to_file, to_rank)
            
            # Check for standard queen promotion (Pawn reaching back rank)
            promotion = None
            if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                # Technically we can't definitively know if it's a pawn without the board,
                # but if a move encodes to the back rank and isn't an underpromotion, 
                # python-chess requires the promotion flag to be set for pawns.
                # We will handle the pawn check logic dynamically in the MCTS if needed.
                pass 
                
            return chess.Move(from_sq, to_sq)
            
        # 2. Knight moves (56-63)
        elif plane < 64:
            jump_idx = plane - 56
            df, dr = self.knight_jumps[jump_idx]
            to_file = from_file + df
            to_rank = from_rank + dr
            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq)
            
        # 3. Underpromotions (64-72)
        else:
            promo_plane = plane - 64
            promo_idx = promo_plane // 3
            dir_idx = promo_plane % 3
            
            promotion = self.promotions[promo_idx]
            df = dir_idx - 1 # maps 0, 1, 2 back to -1, 0, 1
            
            # If white (from_rank 6), moving up. If black (from_rank 1), moving down.
            dr = 1 if from_rank == 6 else -1
            
            to_file = from_file + df
            to_rank = from_rank + dr
            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq, promotion=promotion)

if __name__ == "__main__":
    # Test the encoder
    encoder = ActionEncoder()
    test_move = chess.Move.from_uci("e2e4")
    idx = encoder.encode(test_move)
    decoded_move = encoder.decode(idx)
    print(f"Original: {test_move} -> Encoded Index: {idx} -> Decoded: {decoded_move}")
    
    knight_move = chess.Move.from_uci("g1f3")
    idx = encoder.encode(knight_move)
    print(f"Knight Move: {knight_move} -> Index: {idx} -> Decoded: {encoder.decode(idx)}")
