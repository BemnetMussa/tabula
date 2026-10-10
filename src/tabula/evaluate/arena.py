import os
import time
import chess
import torch
from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board

class Arena:
    def __init__(self, champion_path, challenger_path, num_simulations=40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Arena running on: {self.device}")
        
        # 1. Load the Champion
        self.champion = TabulaNet(num_res_blocks=5).to(self.device)
        if os.path.exists(champion_path):
            self.champion.load_state_dict(torch.load(champion_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: No champion found at {champion_path}. Using random weights.")
        self.champion.eval()
        
        # 2. Load the Challenger
        self.challenger = TabulaNet(num_res_blocks=5).to(self.device)
        if os.path.exists(challenger_path):
            self.challenger.load_state_dict(torch.load(challenger_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: No challenger found at {challenger_path}. Using random weights.")
        self.challenger.eval()
        
        # 3. Setup MCTS for both
        self.champ_mcts = MCTS(self.champion, num_simulations=num_simulations)
        self.chall_mcts = MCTS(self.challenger, num_simulations=num_simulations)
        
        self.champion_path = champion_path

    def play_game(self, challenger_is_white):
        """Plays a single game between Champion and Challenger."""
        board = chess.Board()
        moves = 0
        
        # Assign colors
        white_mcts = self.chall_mcts if challenger_is_white else self.champ_mcts
        black_mcts = self.champ_mcts if challenger_is_white else self.chall_mcts
        
        while board.outcome(claim_draw=True) is None:
            # Determine whose turn it is
            current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            
            # Search the tree
            root_node = current_mcts.search(board, encode_board)
            
            # Deterministic move selection (Argmax) for tournament play
            best_move = None
            best_visits = -1
            
            for move, child in root_node.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move
                    
            board.push(best_move)
            moves += 1
            
            # Cap moves to prevent infinite loops from bad networks
            if moves > 150:
                break
                
        # Determine outcome
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.5  # Draw
            
        # If challenger is white and white won, or challenger is black and black won -> Challenger gets 1
        if (challenger_is_white and outcome.winner == chess.WHITE) or \
           (not challenger_is_white and outcome.winner == chess.BLACK):
            return 1.0  # Challenger win
        else:
            return 0.0  # Champion win

    def evaluate(self, num_games=10, win_threshold=0.55):
        """
        Plays a tournament. If Challenger scores > win_threshold (e.g. 55%), 
        it becomes the new Champion.
        """
        print(f"\n--- TOURNAMENT: Challenger vs Champion ({num_games} games) ---")
        challenger_score = 0.0
        
        for i in range(num_games):
            start_time = time.time()
            # Alternate colors: Game 0 (Challenger=White), Game 1 (Challenger=Black)
            challenger_is_white = (i % 2 == 0)
            color_str = "White" if challenger_is_white else "Black"
            
            result = self.play_game(challenger_is_white)
            challenger_score += result
            
            elapsed = time.time() - start_time
            if result == 1.0:
                res_str = "Challenger WON"
            elif result == 0.0:
                res_str = "Champion WON"
            else:
                res_str = "DRAW"
                
            print(f"Game {i+1}/{num_games} | Challenger as {color_str} | Result: {res_str} | Time: {elapsed:.1f}s")
            
        win_rate = challenger_score / num_games
        print(f"\nFinal Score: Challenger {challenger_score}/{num_games} (Win Rate: {win_rate*100:.1f}%)")
        
        if win_rate >= win_threshold:
            print("GATING PASSED: Challenger is strictly better!")
            print(f"Promoting Challenger to new Champion at {self.champion_path}...")
            # Overwrite the champion weights with the challenger weights
            torch.save(self.challenger.state_dict(), self.champion_path)
            return True
        else:
            print("GATING FAILED: Challenger rejected. Keeping old Champion.")
            return False

if __name__ == "__main__":
    # Test the Arena
    champ_file = "models/tabula_v1_champion.pth"
    chall_file = "models/tabula_v1_latest.pth"
    
    # If there's no champion yet, the latest trained network automatically becomes it.
    if not os.path.exists(champ_file) and os.path.exists(chall_file):
        print("No champion found! Promoting the latest trained network to Champion automatically.")
        import shutil
        os.makedirs("models", exist_ok=True)
        shutil.copy(chall_file, champ_file)
        
    arena = Arena(champion_path=champ_file, challenger_path=chall_file, num_simulations=20)
    # Play a quick 4-game tournament for testing
    arena.evaluate(num_games=4, win_threshold=0.55)
