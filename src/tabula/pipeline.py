import os
import time
import torch

from tabula.selfplay.worker import SelfPlayWorker
from tabula.train.train import train
from tabula.evaluate.arena import Arena

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("==========================================")
    print("      TABULA AI AUTONOMOUS PIPELINE       ")
    print("==========================================")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    champ_path = "models/tabula_v1_champion.pth"
    chall_path = "models/tabula_v1_latest.pth"
    
    iteration = 1
    
    # We loop forever, generating data, training, and evaluating
    while True:
        clear_screen()
        print(f"======================================")
        print(f"       ITERATION {iteration}")
        print(f"======================================")
        
        # ---------------------------------------------------------
        # PHASE 1: SELF PLAY (Generate Data)
        # ---------------------------------------------------------
        print("\n[PHASE 1] Generating Self-Play Data...")
        # Use Champion to generate data. If no champion exists, worker uses random weights.
        worker = SelfPlayWorker(model_path=champ_path, num_simulations=40)
        
        # We generate a few games per iteration to keep it fast for laptops
        num_games = 5
        all_data = []
        
        for g in range(num_games):
            # Render the VERY FIRST game of every iteration so the user can watch the AI learn!
            render_game = (g == 0)
            
            data = worker.play_game(render=render_game, delay=0.5)
            all_data.extend(data)
            
            if not render_game:
                print(f"Completed game {g+1}/{num_games}")
                
        # Save the dataset
        os.makedirs("data", exist_ok=True)
        filename = f"data/selfplay_iter_{iteration}_{int(time.time())}.pt"
        torch.save(all_data, filename)
        print(f"Saved {len(all_data)} moves to {filename}")
        
        # ---------------------------------------------------------
        # PHASE 2: TRAIN (Update Neural Network)
        # ---------------------------------------------------------
        clear_screen()
        print(f"======================================")
        print(f"       ITERATION {iteration} - TRAINING")
        print(f"======================================")
        
        # Train function automatically loads all .pt files in data/
        # and saves the output to models/tabula_v1_latest.pth
        train()
        
        # ---------------------------------------------------------
        # PHASE 3: EVALUATE (Arena Gating)
        # ---------------------------------------------------------
        clear_screen()
        print(f"======================================")
        print(f"       ITERATION {iteration} - ARENA")
        print(f"======================================")
        
        # If there's no champion yet, skip arena and just promote the challenger
        if not os.path.exists(champ_path):
            print("No champion found. Promoting the first trained network to Champion!")
            import shutil
            shutil.copy(chall_path, champ_path)
        else:
            arena = Arena(champion_path=champ_path, challenger_path=chall_path, num_simulations=20)
            # Pit them against each other for 4 games
            arena.evaluate(num_games=4, win_threshold=0.55)
            
        print(f"\nIteration {iteration} complete. Sleeping 5 seconds before next cycle...")
        time.sleep(5)
        iteration += 1

if __name__ == "__main__":
    main()
