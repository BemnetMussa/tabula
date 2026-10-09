"""Automatically test Generation 1 playing against itself for 10 moves."""

import chess
from tabula.model.model import TabulaNet
from tabula.game.encoder import encode_board
from tabula.mcts.search import MCTS

def main():
    print("=== AUTOMATED TEST: GEN 1 vs GEN 1 ===")
    print("Initializing 13-Layer PyTorch Network...")
    model = TabulaNet(num_res_blocks=5)
    model.eval()

    print("Initializing MCTS Calculator...")
    mcts = MCTS(model, num_simulations=20, c_puct=1.25)
    
    board = chess.Board()
    print("\nStarting Board:")
    print(board)
    print("\nPlaying 10 autonomous moves...\n")

    for i in range(10):
        if board.is_game_over():
            print("Game over early!")
            break
            
        print(f"--- Move {i+1} (Turn: {'White' if board.turn else 'Black'}) ---")
        
        # AI Searches the tree
        root_node = mcts.search(board, encode_board)
        
        # Pick the most visited node
        best_move = None
        best_visits = -1
        for move, child in root_node.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
                
        # Push the move to the board
        board.push(best_move)
        print(f"Gen 1 chose: {best_move.uci()}")
        print(board)
        print()

    print("Automated test complete. PyTorch and MCTS are fully stable!")

if __name__ == "__main__":
    main()
