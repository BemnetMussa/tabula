import os
import time
import torch

from tabula.selfplay.worker import SelfPlayWorker
from tabula.train.train import train
from tabula.evaluate.arena import Arena
from tabula.watch.live import LiveBoard
from tabula.hardware.thermal import ThermalGuard

def get_next_iteration() -> int:
    """Auto-detect next iteration number from existing files in data/."""
    existing = []
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.startswith("selfplay_iter_") and f.endswith(".pt"):
                try:
                    parts = f.split("_")
                    existing.append(int(parts[2]))
                except (IndexError, ValueError):
                    pass
    return (max(existing) + 1) if existing else 1

def main():
    champ_path = "models/tabula_v1_champion.pth"
    chall_path = "models/tabula_v1_latest.pth"
    
    live = LiveBoard(enable_tui=True)
    thermal = ThermalGuard(warm_threshold=72.0, emergency_threshold=80.0, target_cool_temp=62.0, cooldown_sleep_sec=2.0)
    iteration = get_next_iteration()
    
    # We loop forever, generating data, training, and evaluating
    while True:
        # Pre-check thermals
        thermal.check_and_cool()
        
        # ---------------------------------------------------------
        # PHASE 1: SELF PLAY (Generate Data)
        # ---------------------------------------------------------
        print(f"\n[PHASE 1] Generating Self-Play Data (Iteration {iteration})...")
        # The latest network is the one whose value head is alive. The champion
        # stays the bar for the match and is not the player until it earns the games.
        latest_exists = os.path.exists(chall_path)
        play_path = chall_path if latest_exists else champ_path
        live.begin_round(iteration, latest_exists)
        worker = SelfPlayWorker(model_path=play_path, num_simulations=40)
        
        # We generate a few games per iteration to keep it fast for laptops
        num_games = 5
        all_data = []
        
        for g in range(num_games):
            thermal.check_and_cool()
            # The first game of each iteration is the one on the board.
            data = worker.play_game(live=live if g == 0 else None, hold=0.03 if g == 0 else 0.0)
            all_data.extend(data)
            print(f"Completed game {g+1}/{num_games}")
            thermal.rest(2.0) # 2s breather to dissipate heat between games
                
        # Save the dataset
        os.makedirs("data", exist_ok=True)
        filename = f"data/selfplay_iter_{iteration}_{int(time.time())}.pt"
        torch.save(all_data, filename)
        print(f"Saved {len(all_data)} moves to {filename}")
        
        # ---------------------------------------------------------
        # PHASE 2: TRAIN (Update Neural Network)
        # ---------------------------------------------------------
        thermal.check_and_cool()
        print(f"======================================")
        print(f"       ITERATION {iteration} - TRAINING")
        print(f"======================================")
        
        # Train function automatically loads all .pt files in data/
        # and saves the output to models/tabula_v1_latest.pth
        train(live=live)
        thermal.rest(3.0) # 3s breather after backpropagation
        
        # ---------------------------------------------------------
        # PHASE 3: EVALUATE (Arena Gating)
        # ---------------------------------------------------------
        thermal.check_and_cool()
        print(f"======================================")
        print(f"       ITERATION {iteration} - ARENA")
        print(f"======================================")
        
        # If there's no champion yet, skip arena and just promote the challenger
        if not os.path.exists(champ_path):
            print("No champion found. Promoting the first trained network to Champion!")
            import shutil
            shutil.copy(chall_path, champ_path)
            live.note_match(1.0, 0, True, first=True)
        else:
            arena = Arena(champion_path=champ_path, challenger_path=chall_path, num_simulations=20)
            # Ten games, so one repeated draw cannot decide the 55% bar.
            promoted = arena.evaluate(num_games=10, win_threshold=0.55, live=live, hold=0.02)
            live.note_match(
                arena.last_win_rate,
                arena.last_games,
                promoted,
                ch_wins=arena.challenger_wins,
                ch_losses=arena.champion_wins,
                draws=arena.draws
            )
            
        print(f"\nIteration {iteration} complete. Cooling down 10 seconds before next cycle...")
        thermal.rest(10.0)
        thermal.check_and_cool()
        iteration += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.stdout.write("\033[?25h\n\n[Tabula AI] Pipeline paused cleanly by user.\n")
        sys.stdout.flush()
