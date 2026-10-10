import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tabula.model.model import TabulaNet

class SelfPlayDataset(Dataset):
    def __init__(self, data_dir):
        """
        Loads all self-play .pt files from the data directory.
        Each file contains a list of tuples: (state_tensor, pi_tensor, z_tensor).
        """
        self.examples = []
        pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        for f in pt_files:
            data = torch.load(f, weights_only=False)
            self.examples.extend(data)
            
        print(f"Loaded {len(self.examples)} examples from {len(pt_files)} files.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        s, pi, z = self.examples[idx]
        # Convert numpy arrays to PyTorch tensors
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32)
        )

def train():
    print("=== Tabula AI - AlphaZero Training Loop ===")
    
    # --- Configuration ---
    data_dir = "data"
    models_dir = "models"
    batch_size = 64  # Small batch to respect 4GB VRAM
    epochs = 10
    lr = 0.001
    weight_decay = 1e-4  # L2 regularization (c ||theta||^2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "tabula_v1_latest.pth")
    
    # 1. Load Data
    print("\n--- 1. Loading Self-Play Data ---")
    dataset = SelfPlayDataset(data_dir)
    if len(dataset) == 0:
        print("No training data found. Run self-play first (python -m tabula.selfplay.worker).")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # 2. Initialize Model
    print("\n--- 2. Initializing Neural Network ---")
    model = TabulaNet(num_res_blocks=5).to(device)
    
    if os.path.exists(model_path):
        print(f"Found existing weights. Loading from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("Starting with fresh randomly initialized weights.")
    
    # 3. Optimizer & Loss Functions
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # AlphaZero Joint Loss Function:
    # 1. Policy Loss: Cross Entropy between target probabilities (pi) and predicted logits (p).
    # 2. Value Loss: Mean Squared Error between actual game outcome (z) and predicted value (v).
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # 4. Training Loop
    print("\n--- 3. Starting Training Loop ---")
    model.train()
    
    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (states, pis, zs) in enumerate(dataloader):
            # Move data to GPU if available
            states, pis, zs = states.to(device), pis.to(device), zs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, values = model(states)
            
            # Compute losses
            loss_p = policy_loss_fn(policy_logits, pis)
            loss_v = value_loss_fn(values, zs)
            
            # Combine into joint loss
            total_loss = loss_p + loss_v
            
            # Backward pass & optimize
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += loss_p.item()
            epoch_value_loss += loss_v.item()
            epoch_total_loss += total_loss.item()
            
        # Calculate averages for logging
        avg_p_loss = epoch_policy_loss / len(dataloader)
        avg_v_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} [{elapsed:.1f}s] - Total Loss: {avg_total_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")
        
    # 5. Save the trained model
    print("\n--- 4. Saving Model ---")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Weights safely stored to {model_path}.")

if __name__ == "__main__":
    train()
