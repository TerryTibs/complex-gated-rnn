import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
CONFIG = {
    # Architecture
    "seq_len": 32,              # Context window size
    "embedding_dim": 64,        # Size of the latent vector space (Complex: 64+64)
    "recursion_depth": 3,       # Number of recursive processing steps per token
    "stochastic_prob": 0.9,     # Probability of executing a recursion step (Depth Dropout)
    
    # Training
    "learning_rate": 0.004,
    "epochs": 350,
    "batch_size": 1,
    "grad_clip": 1.0,           # Threshold to prevent exploding gradients
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Loss Weighting (Balancing Task vs. Structure)
    "lambda_prediction": 1.0,   # CrossEntropy (Task Accuracy)
    "lambda_regularization": 0.15, # Latent Stability (Robustness)
    "noise_injection": 0.01     # Magnitude of noise for robustness training
}

# ==========================================
# 2. Data Preparation
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

# Tokenizer
chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert text to tensor
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(CONFIG["device"])

# ==========================================
# 3. Neural Network Modules
# ==========================================

class ComplexLinear(nn.Module):
    """
    Implements a Linear Layer in the Complex domain.
    Logic: (r + qi) * (u + vi) = (ru - qv) + (rv + qu)i
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=False)
        self.fc_imag = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal Initialization is critical for Recursive/RNN stability
        # It ensures the singular values of the Jacobian are close to 1
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z):
        # z is a complex tensor
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

class RecursiveCell(nn.Module):
    """
    A recurrent processing unit applied to 'depth' rather than 'time'.
    """
    def __init__(self, dim):
        super().__init__()
        self.transform = ComplexLinear(dim, dim)

    def forward(self, z):
        # 1. Linear Transformation
        z_new = self.transform(z)
        
        # 2. Complex Activation (Bounded Tanh)
        z_act = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
        
        # 3. Normalization (Project to Unit Circle)
        # Prevents signal magnitude from exploding during recursion
        magnitude = torch.abs(z_act) + 1e-8
        z_act = z_act / magnitude
        
        # 4. Residual Connection
        return z + (0.5 * z_act)

class ComplexRecursiveRNN(nn.Module):
    """
    Main Architecture:
    1. Complex Embeddings (Magnitude + Phase)
    2. Dimension-Wise Gating (Input vs Memory)
    3. Stochastic Recursion (Depth Processing)
    4. Decoding
    """
    def __init__(self, vocab_size, dim, recursion_depth):
        super().__init__()
        self.dim = dim
        self.recursion_depth = recursion_depth
        
        # Embeddings
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        # Recursive Core
        self.recursive_cell = RecursiveCell(dim)
        
        # Dimension-Wise Gating
        # Allows specific dimensions to specialize in Perception vs Memory
        self.gate_param = nn.Parameter(torch.zeros(dim)) 
        
        # Decoder
        self.decoder = nn.Linear(dim * 2, vocab_size)

    def get_embeddings(self, input_ids):
        # Polar -> Cartesian
        r = self.emb_mag(input_ids)
        theta = self.emb_phase[input_ids]
        return torch.complex(r * torch.cos(theta), r * torch.sin(theta))

    def forward(self, input_ids, hidden_state=None, training=False):
        # 1. Embed
        input_z = self.get_embeddings(input_ids)
        
        # 2. Gate (Dimension-wise Sigmoid)
        # Shape: [1, 1, dim] broadcasting to batch
        gate = torch.sigmoid(self.gate_param).unsqueeze(0).unsqueeze(0)
        
        if hidden_state is None:
            z = input_z
        else:
            # Gated Recurrent Update
            z = ((1 - gate) * hidden_state) + (gate * input_z)

        # 3. Recursive Processing
        for _ in range(self.recursion_depth):
            # Stochastic Depth: Randomly skip steps during training for robustness
            if training and torch.rand(1).item() > CONFIG["stochastic_prob"]:
                continue
            z = self.recursive_cell(z)
            
        # 4. Decode
        features = torch.cat([z.real, z.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, z, gate

# ==========================================
# 4. Training Loop
# ==========================================

def train_model():
    model = ComplexRecursiveRNN(vocab_size, CONFIG["embedding_dim"], CONFIG["recursion_depth"])
    model.to(CONFIG["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"--- Starting Training on {CONFIG['device']} ---")
    
    for epoch in range(CONFIG["epochs"]):
        hidden_state = None
        epoch_loss = 0
        
        # Sliding window sequence training
        for i in range(len(data_tensor) - CONFIG["seq_len"] - 1):
            
            input_seq = data_tensor[i : i + CONFIG["seq_len"]].unsqueeze(0)
            target_seq = data_tensor[i+1 : i + CONFIG["seq_len"] + 1].unsqueeze(0)

            # --- Stability Check (Noise Injection) ---
            # Generate a perturbed version of the state without backprop
            # to measure how robust the model is to noise
            mask = torch.rand_like(input_seq.float()) < 0.15
            noisy_input = input_seq.clone()
            noisy_input[mask] = 0
            with torch.no_grad():
                _, z_perturbed, _ = model(noisy_input, hidden_state, training=False)
                # Add Gaussian noise
                noise = torch.randn_like(z_perturbed.real) + 1j * torch.randn_like(z_perturbed.imag)
                z_perturbed = z_perturbed + (CONFIG["noise_injection"] * noise)

            # --- Main Forward Pass ---
            logits, z_curr, gate_val = model(input_seq, hidden_state, training=True)

            # --- Loss Calculation ---
            
            # 1. Prediction Accuracy
            loss_pred = F.cross_entropy(logits.view(-1, vocab_size), target_seq.view(-1))
            
            # 2. Latent Regularization
            # A. Internal Coherence (Self-Similarity)
            z_flat = torch.cat([z_curr.real, z_curr.imag], dim=-1) # Shape: [Batch, Seq, Dim]
            z_norm = F.normalize(z_flat, p=2, dim=-1)
            
            # ** FIXED: Batched Matrix Transpose **
            # We want to multiply (Batch, Seq, Dim) by (Batch, Dim, Seq)
            # Use .transpose(-2, -1) instead of .T
            sim_matrix = torch.matmul(z_norm, z_norm.transpose(-2, -1))
            loss_coherence = 1.0 - sim_matrix.mean()
            
            # B. Robustness (Difference from perturbed state)
            diff = z_curr - z_perturbed
            loss_robustness = torch.mean((diff.real)**2 + (diff.imag)**2)
            
            total_reg = loss_coherence + loss_robustness
            
            # Combined Loss
            loss = (CONFIG["lambda_prediction"] * loss_pred) + \
                   (CONFIG["lambda_regularization"] * total_reg)

            # --- Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            
            # Truncate Gradient History
            hidden_state = z_curr.detach()
            epoch_loss += loss.item()

        if epoch % 50 == 0:
            avg_loss = epoch_loss / (len(data_tensor) - CONFIG["seq_len"])
            # Calculate mean gate opening
            gate_mean = gate_val.mean().item()
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Gate Avg: {gate_mean:.2f}")

    return model

# ==========================================
# 5. Inference & Visualization
# ==========================================

def generate_and_visualize(model, start_str="True"):
    model.eval()
    print("\n--- Generating Sequence & Latent Visualization ---")
    
    input_ids = torch.tensor([char_to_ix[c] for c in start_str], dtype=torch.long).to(CONFIG["device"]).unsqueeze(0)
    hidden_state = None
    output = start_str
    
    # Store latent states for visualization
    latent_trajectory = []
    
    with torch.no_grad():
        # Warm-up phase
        for t in range(input_ids.size(1) - 1):
            _, hidden_state, _ = model(input_ids[:, t].unsqueeze(1), hidden_state, training=False)
        
        current_input = input_ids[:, -1].unsqueeze(1)
        
        # Generation Loop
        for _ in range(200):
            logits, hidden_state, _ = model(current_input, hidden_state, training=False)
            
            # Capture latent state (Magnitude of complex vector)
            # We visualize the magnitude |z| to see activation intensity
            z_mag = torch.abs(hidden_state).squeeze().cpu().numpy()
            latent_trajectory.append(z_mag)
            
            # Sample next token
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_ix = torch.multinomial(probs, 1)
            
            output += ix_to_char[next_ix.item()]
            current_input = next_ix

    print(output)
    
    # --- Visualization ---
    latent_matrix = np.array(latent_trajectory).T # Shape: [Dim, Time]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(latent_matrix, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label='Latent Activation Magnitude |z|')
    plt.title(f"Latent State Trajectory ({CONFIG['embedding_dim']} Dimensions)")
    plt.xlabel("Time Steps (Generation)")
    plt.ylabel("Latent Dimensions")
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    trained_model = train_model()
    generate_and_visualize(trained_model)
