import os
import time
import numpy as np
import torch
import chess

from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board
from tabula.game.action_encoder import ActionEncoder

class SelfPlayWorker:
    def __init__(self, model_path=None, num_simulations=40):
        print("Initializing TabulaNet for Self-Play...")
        self.model = TabulaNet(num_res_blocks=5)
        
        # If we had a trained model, we would load it here. 
        # For Gen 1, we just use the randomly initialized weights.
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.mcts = MCTS(self.model, num_simulations=num_simulations, c_puct=1.25)
        self.action_encoder = ActionEncoder()
        
    def play_game(self, render=False, delay=0.5):
        """Plays a single game of AlphaZero vs AlphaZero and returns the training examples."""
        board = chess.Board()
        examples = [] # Will store temporary tuples of (board_tensor, pi_tensor, player_turn)
        
        if not render:
            print("Starting new self-play game...")
        
        moves = 0
        while board.outcome(claim_draw=True) is None:
            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Tabula AI Self-Play (Move {moves}) ---")
                print(board.unicode())
                time.sleep(delay)

            # 1. Run MCTS to get the root node and visit counts
            root_node = self.mcts.search(board, encode_board)
            
            # 2. Extract the policy vector \pi (size 4672)
            pi = np.zeros(4672, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root_node.children.values())
            
            best_move = None
            best_visits = -1
            
            for move, child in root_node.children.items():
                prob = child.visit_count / total_visits
                idx = self.action_encoder.encode(move)
                pi[idx] = prob
                
                # For Version 1, we just greedily pick the most visited move
                # (In true AlphaZero, they add Dirichlet noise and pick probabilistically for the first 30 moves)
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move
            
            # 3. Store the State (s), Policy (\pi), and whose turn it is
            state_tensor = encode_board(board)
            examples.append([state_tensor, pi, board.turn])
            
            # 4. Push the move to the board
            board.push(best_move)
            moves += 1
            
            if not render and moves % 10 == 0:
                print(f"Move {moves}... (Eval: {root_node.q_value:.2f})")
                
            # Optional: Cap max moves to prevent infinite loops from totally random baby-AI weights
            if moves > 150:
                print("Hit 150 move limit (random weights dragging the game out). Capping as draw.")
                break
                
        # 5. Game Over - Calculate Rewards (z)
        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Tabula AI Self-Play (Game Over) ---")
            print(board.unicode())
            time.sleep(2.0)
            
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            reward = 0.0 # Draw (or artificially capped at 150 moves)
        elif outcome.winner == chess.WHITE:
            reward = 1.0
        else:
            reward = -1.0
            
        if not render:
            print(f"Game over after {moves} moves. Reward for White: {reward}")
        else:
            print(f"\nGame over after {moves} moves.")
            if reward == 1.0:
                print("Result: WHITE WINS")
            elif reward == -1.0:
                print("Result: BLACK WINS")
            else:
                print("Result: DRAW")
            time.sleep(2.0)
        
        # 6. Retroactively assign the final reward to all saved turns
        training_data = []
        for state_tensor, pi, player_turn in examples:
            # If it was White's turn, the reward is `reward`. If Black's turn, it's `-reward`.
            actual_reward = reward if player_turn == chess.WHITE else -reward
            training_data.append((state_tensor, pi, np.array([actual_reward], dtype=np.float32)))
            
        return training_data

def main():
    # We use fewer simulations for this test script so it runs fast in the terminal
    worker = SelfPlayWorker(num_simulations=40)
    
    print("--- GENERATING 1 TEST GAME ---")
    start_time = time.time()
    
    # Play 1 full game autonomously
    data = worker.play_game()
    
    # Save the (s, pi, z) dataset to disk for PyTorch to use later
    os.makedirs("data", exist_ok=True)
    filename = f"data/selfplay_test_{int(time.time())}.pt"
    
    torch.save(data, filename)
    
    elapsed = time.time() - start_time
    print(f"\nSaved {len(data)} board positions to {filename} in {elapsed:.1f} seconds.")
    print("The self-play pipeline is fully functional!")

if __name__ == "__main__":
    main()
