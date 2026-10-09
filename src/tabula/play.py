"""Play White against Generation 1 (PyTorch + MCTS) in the terminal."""

from __future__ import annotations

import random
import torch

from tabula.game import Game, IllegalMove, random_reply
from tabula.model.model import TabulaNet
from tabula.game.encoder import encode_board
from tabula.mcts.search import MCTS

def main() -> None:
    # Boot up the Brain (TabulaNet) and the Calculator (MCTS)
    print("Loading TabulaNet (Generation 1)...")
    model = TabulaNet(num_res_blocks=5)
    model.eval() # Set to evaluation mode
    
    # 50 simulations per turn so it responds fast in the terminal
    mcts = MCTS(model, num_simulations=50, c_puct=1.25)
    
    game = Game()
    print("You are White. Generation 1 (MCTS) is Black.")
    print("Moves: e2e4 or e4. Quit with q.")
    print(game.board)
    
    while game.result() is None:
        if game.board.turn:
            # Human's turn (White)
            typed = input("\nYour move: ").strip()
            if typed.lower() in {"q", "quit"}:
                print("stopped")
                return
            try:
                played = game.push(typed)
            except IllegalMove:
                print(f"illegal move: {typed}")
                continue
            print(f"you: {played}")
        else:
            # AI's turn (Black) - Use MCTS to find the best move
            print("Thinking...")
            root_node = mcts.search(game.board, encode_board)
            
            # AlphaZero picks the child node with the highest visit count
            best_move = None
            best_visits = -1
            for move, child in root_node.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move
                    
            # Push the move using the UCI string
            played = game.push(best_move.uci())
            print(f"generation 1: {played}")
            
        print(game.board)
        
    print(f"\n{game.result()} ({game.reason()})")

if __name__ == "__main__":
    main()
