import os
import torch
import numpy as np

def main():
    print("=== AUTOMATED TEST: SELF-PLAY DATA PIPELINE ===")
    
    # 1. Find the latest generated .pt file in the data/ folder
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("ERROR: data/ directory not found.")
        return
        
    pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
    if not pt_files:
        print("ERROR: No .pt files found in data/. Run python -m tabula.selfplay.worker first.")
        return
        
    # Get the most recently created file
    latest_file = max(pt_files, key=lambda f: os.path.getctime(os.path.join(data_dir, f)))
    file_path = os.path.join(data_dir, latest_file)
    print(f"Loading latest self-play data: {file_path}")
    
    # 2. Load the dataset
    data = torch.load(file_path, weights_only=False)
    
    if len(data) == 0:
        print("ERROR: Dataset is empty.")
        return
        
    print(f"Successfully loaded {len(data)} training examples.")
    
    # 3. Verify the shapes and types of (s, pi, z)
    state, pi, z = data[0]
    
    print("\n--- Verifying Data Structure ---")
    
    # State (s) should be a numpy array of shape (13, 8, 8)
    assert isinstance(state, np.ndarray), f"Expected State to be numpy array, got {type(state)}"
    assert state.shape == (13, 8, 8), f"Expected State shape (13, 8, 8), got {state.shape}"
    print("[PASS] State (s) has correct shape (13, 8, 8) and type.")
    
    # Policy (\pi) should be a numpy array of shape (4672,)
    assert isinstance(pi, np.ndarray), f"Expected Policy to be numpy array, got {type(pi)}"
    assert pi.shape == (4672,), f"Expected Policy shape (4672,), got {pi.shape}"
    print("[PASS] Policy (pi) has correct shape (4672,) and type.")
    
    # Value (z) should be a numpy array of shape (1,)
    assert isinstance(z, np.ndarray), f"Expected Value to be numpy array, got {type(z)}"
    assert z.shape == (1,), f"Expected Value shape (1,), got {z.shape}"
    print("[PASS] Value (z) has correct shape (1,) and type.")
    
    # Verify probability distribution (pi should sum to ~1.0)
    pi_sum = np.sum(pi)
    assert np.isclose(pi_sum, 1.0, atol=1e-3), f"Expected Policy sum to be 1.0, got {pi_sum}"
    print(f"[PASS] Policy forms a valid probability distribution (sum = {pi_sum:.4f}).")
    
    print("\nAutomated pipeline test complete. The self-play data is perfectly formatted for PyTorch training!")

if __name__ == "__main__":
    main()
