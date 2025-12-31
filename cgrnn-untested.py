import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
CONFIG = {
    # --- Architecture ---
    "seq_len": 32,              # Context window size
    "embedding_dim": 64,        # Size of the complex latent vector
    "max_recursion_depth": 6,   # Maximum thinking steps per token
    "p_keep": 0.9,              # Stochastic Depth probability (Drop-Depth)
    "lookahead_steps": 3,       # Imagination steps before output
    "memory_span": 10,          # How far back the model attends (Metacognition)
    
    # --- Hierarchical Abstraction ---
    "n_symbols": 32,            # Level 1: Discrete Symbols (Words of thought)
    "n_concepts": 8,            # Level 2: Abstract Concepts (Categories of thought)
    "commitment_cost": 0.25,    # VQ-VAE commitment factor
    "dynamic_threshold": 1.5,   # Distance threshold to create new symbols dynamically
    
    # --- Training ---
    "epochs": 350,
    "learning_rate": 0.003,
    "grad_clip": 1.0,           # Safety valve for gradients
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # --- Multi-Objective Loss Weights ---
    "w_prediction": 1.0,        # Task Accuracy
    "w_variance": 0.1,          # Prevent Latent Collapse
    "w_depth_consistency": 0.1, # Smoothness of thought
    "w_stability": 0.1,         # Robustness to noise
    "w_depth_usage": 0.05,      # Penalty for unnecessary thinking
    "w_symbolic": 0.1,          # Codebook alignment (Symbols)
    "w_hierarchy": 0.1,         # Codebook alignment (Concepts)
    
    # --- Robustness ---
    "noise_std": 0.05,          # Magnitude of noise injection
    "eps": 1e-8                 # Numerical stability epsilon
}

# ==========================================
# 2. Dataset Preparation
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

# Tokenization
chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Data Tensor
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(CONFIG["device"])

# ==========================================
# 3. Core Modules: Complex Arithmetic
# ==========================================
class ComplexLinear(nn.Module):
    """
    Linear layer operating in the Complex plane.
    Preserves phase relationships via rotation logic.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        # Orthogonal Initialization for stability
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z):
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

class RecursiveCell(nn.Module):
    """
    The Thinking Unit. Applies transformation and soft normalization.
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)

    def forward(self, z):
        z_new = self.linear(z)
        
        # Soft Normalization (Bounded Tanh)
        # Projects vectors towards the unit circle without hard clipping
        mag = torch.sqrt(z_new.real**2 + z_new.imag**2 + CONFIG["eps"])
        z_act = z_new / (1.0 + mag)
        z_act = torch.tanh(z_act.real) + 1j * torch.tanh(z_act.imag)
        
        return z_act

class ComplexAttention(nn.Module):
    """
    Metacognition: Allows the model to attend to its own past states.
    Scales attention based on 'confidence' from the symbolic layer.
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim*2, dim)
        self.key = nn.Linear(dim*2, dim)
        self.value = nn.Linear(dim*2, dim*2)
        self.scale = dim ** -0.5

    def forward(self, history, current, confidence=None):
        if not history: return current
        
        # Prepare Current Query
        h_curr_cat = torch.cat([current.real, current.imag], dim=-1)
        Q = self.query(h_curr_cat).unsqueeze(2) # [B, S, 1, D]

        # Prepare History Keys/Values
        hist_real = torch.stack([h.real for h in history], dim=2)
        hist_imag = torch.stack([h.imag for h in history], dim=2)
        hist_cat = torch.cat([hist_real, hist_imag], dim=-1) # [B, S, Hist, D*2]

        K = self.key(hist_cat)
        V = self.value(hist_cat)

        # Attention Scores
        scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        
        # Confidence Modulation: If confident, sharpen attention
        if confidence is not None:
            scores = scores * confidence.unsqueeze(-1).unsqueeze(-1)

        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted Sum
        context = torch.matmul(attn_weights, V).squeeze(2)
        
        # Residual Update (Mix Context into Current)
        half = context.shape[-1] // 2
        ctx_real, ctx_imag = context[..., :half], context[..., half:]
        return torch.complex(current.real + 0.1 * ctx_real, current.imag + 0.1 * ctx_imag)

# ==========================================
# 4. Hierarchical & Dynamic VQ
# ==========================================
class DynamicHierarchicalVQ(nn.Module):
    """
    Two-Level VQ-VAE with Dynamic Codebook Updates.
    Level 1: Continuous Thought -> Discrete Symbols
    Level 2: Discrete Symbols -> Abstract Concepts
    """
    def __init__(self, latent_dim, n_symbols, n_concepts):
        super().__init__()
        self.n_symbols = n_symbols
        self.n_concepts = n_concepts
        
        # Codebooks
        self.symbol_codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        self.concept_codebook = nn.Parameter(torch.randn(n_concepts, latent_dim*2))

    def quantize(self, z_flat, codebook):
        # Euclidean distance
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, codebook.t())
            
        min_indices = torch.argmin(d, dim=-1)
        min_dist = torch.min(d, dim=-1).values
        z_q = F.embedding(min_indices, codebook)
        
        # --- Dynamic Expansion ---
        # If input is very far from existing symbols, overwrite a random symbol
        if self.training and min_dist.mean() > CONFIG['dynamic_threshold']:
            idx_replace = torch.randint(0, codebook.size(0), (1,)).item()
            with torch.no_grad():
                codebook[idx_replace] = z_flat.mean(dim=[0,1]).detach()

        # VQ Losses
        loss_vq = F.mse_loss(z_q, z_flat.detach())      # Pull codebook to vector
        loss_commit = F.mse_loss(z_q.detach(), z_flat)  # Pull vector to codebook
        loss = loss_vq + CONFIG["commitment_cost"] * loss_commit
        
        # Straight-Through Estimator
        z_q = z_flat + (z_q - z_flat).detach()
        
        return z_q, loss, min_indices, min_dist

    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # Level 1: Latent -> Symbol
        z_sym, loss_sym, sym_indices, sym_dist = self.quantize(z_flat, self.symbol_codebook)
        
        # Level 2: Symbol -> Concept
        z_con, loss_con, con_indices, _ = self.quantize(z_sym, self.concept_codebook)
        
        # Confidence (Inverse of distance to symbol)
        confidence = 1.0 / (1.0 + sym_dist)
        
        # Convert back to Complex
        half = z_sym.shape[-1] // 2
        z_complex_sym = torch.complex(z_sym[..., :half], z_sym[..., half:])
        
        # One-hot encodings
        sym_probs = F.one_hot(sym_indices, num_classes=self.n_symbols).float()
        con_probs = F.one_hot(con_indices, num_classes=self.n_concepts).float()
        
        return z_complex_sym, sym_probs, con_probs, loss_sym, loss_con, sym_indices, con_indices, confidence

# ==========================================
# 5. The Ceiling Symbolic RNN (Master Model)
# ==========================================
class CeilingSymbolicRNN(nn.Module):
    def __init__(self, vocab_size, dim, max_depth):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        self.cell = RecursiveCell(dim)
        self.attn = ComplexAttention(dim)
        self.gate_param = nn.Parameter(torch.zeros(dim))
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        self.hierarchy = DynamicHierarchicalVQ(dim, CONFIG["n_symbols"], CONFIG["n_concepts"])

    def get_complex_embedding(self, input_ids):
        r = self.emb_mag(input_ids)
        theta = self.emb_phase[input_ids]
        return torch.complex(r*torch.cos(theta), r*torch.sin(theta))

    def forward(self, input_ids, hidden_state=None, past_states=None, collect_depth=False, training=False):
        # 1. Perception
        z_input = self.get_complex_embedding(input_ids)
        
        # 2. Gating
        gate = torch.sigmoid(self.gate_param).view(1, 1, -1)
        if hidden_state is None:
            z = z_input
        else:
            z = (1 - gate) * hidden_state + gate * z_input

        # Initialize State
        if past_states is None: past_states = []
        depth_states = []
        depth_used = 0
        total_sym_loss, total_con_loss = 0, 0
        
        # Viz placeholders
        last_sym_probs, last_con_probs = None, None
        sym_idx, con_idx = None, None
        confidence = None

        # 3. Recursive Thought Loop
        for _ in range(self.max_depth):
            # Stochastic Depth
            if training and torch.rand(1).item() > CONFIG["p_keep"]:
                if collect_depth: depth_states.append(z)
                continue
            
            # A. Think (Recurse)
            z = self.cell(z)
            
            # B. Reflect (Attend)
            z = self.attn(past_states, z, confidence)
            
            # C. Abstract (Hierarchical VQ)
            z_sym, sym_probs, con_probs, loss_sym, loss_con, s_idx, c_idx, conf = self.hierarchy(z)
            
            # Residual Integration: Mix continuous thought with discrete symbol
            z = 0.5 * z + 0.5 * z_sym
            
            # Accumulate info
            total_sym_loss += loss_sym
            total_con_loss += loss_con
            last_sym_probs, last_con_probs = sym_probs, con_probs
            sym_idx, con_idx = s_idx, c_idx
            confidence = conf
            
            # Update Memory
            past_states.append(z)
            if len(past_states) > CONFIG["memory_span"]:
                past_states.pop(0)
                
            depth_used += 1
            if collect_depth: depth_states.append(z)

        # 4. Lookahead (Imagination)
        z_final = z
        for _ in range(CONFIG["lookahead_steps"]):
            z_final = self.cell(z_final)
            
        # 5. Decode
        features = torch.cat([z_final.real, z_final.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, z, depth_states, depth_used, past_states, last_sym_probs, last_con_probs, total_sym_loss, total_con_loss, sym_idx, con_idx

# ==========================================
# 6. Loss Functions
# ==========================================
def variance_loss(z):
    z_flat = z.reshape(-1, z.size(-1))
    std = torch.sqrt(z_flat.var(0) + CONFIG["eps"])
    return torch.mean(F.relu(1.0 - std))

def depth_consistency_loss(states):
    if not states: return torch.tensor(0.0).to(CONFIG["device"])
    loss = 0
    for i in range(1, len(states)):
        diff = states[i] - states[i-1]
        loss += (diff.real**2 + diff.imag**2).mean()
    return loss / len(states)

def stability_loss(z_clean, z_perturbed):
    diff = z_clean - z_perturbed
    return (diff.real**2 + diff.imag**2).mean()

# ==========================================
# 7. Training Loop
# ==========================================
def train_model():
    model = CeilingSymbolicRNN(vocab_size, CONFIG["embedding_dim"], CONFIG["max_recursion_depth"])
    model.to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"--- Training on {CONFIG['device']} ---")
    
    for epoch in range(CONFIG["epochs"]):
        hidden_state = None
        past_states = []
        total_loss = 0
        
        for i in range(len(data_tensor) - CONFIG["seq_len"] - 1):
            input_seq = data_tensor[i : i + CONFIG["seq_len"]].unsqueeze(0)
            target_seq = data_tensor[i+1 : i + CONFIG["seq_len"] + 1].unsqueeze(0)

            # Clean Pass
            logits, z_curr, depth_states, d_used, past_states, sym_probs, con_probs, loss_sym, loss_con, _, _ = model(
                input_seq, hidden_state, past_states, collect_depth=True, training=True
            )
            past_states = [p.detach() for p in past_states]

            # Noisy Pass (Invariance)
            if hidden_state is not None:
                noise = torch.randn_like(hidden_state.real) + 1j * torch.randn_like(hidden_state.imag)
                h_noisy = hidden_state.detach() + (CONFIG["noise_std"] * noise)
            else:
                h_noisy = None
            
            past_copy = [p.clone() for p in past_states]
            _, z_perturbed, _, _, _, _, _, _, _, _, _ = model(
                input_seq, h_noisy, past_copy, collect_depth=False, training=False
            )

            # Losses
            loss_pred = F.cross_entropy(logits.view(-1, vocab_size), target_seq.view(-1))
            loss_var = variance_loss(torch.cat([z_curr.real, z_curr.imag], dim=-1))
            loss_depth = depth_consistency_loss(depth_states)
            loss_stab = stability_loss(z_curr, z_perturbed)
            loss_usage = 1.0 - (d_used / CONFIG["max_recursion_depth"])

            # Error-Aware Scaling
            scale = torch.clamp(loss_pred.detach(), 0.5, 2.0)
            
            loss_total = scale * (
                CONFIG["w_prediction"] * loss_pred +
                CONFIG["w_variance"] * loss_var +
                CONFIG["w_depth_consistency"] * loss_depth +
                CONFIG["w_stability"] * loss_stab +
                CONFIG["w_depth_usage"] * loss_usage +
                CONFIG["w_symbolic"] * loss_sym +
                CONFIG["w_hierarchy"] * loss_con
            )

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

            hidden_state = z_curr.detach()
            total_loss += loss_total.item()

        if epoch % 50 == 0:
            avg_loss = total_loss / (len(data_tensor) - CONFIG["seq_len"])
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | SymLoss: {loss_sym.item():.4f} | ConLoss: {loss_con.item():.4f}")

    return model

# ==========================================
# 8. Generation & Visualization
# ==========================================
def generate_text(model, start_str="True"):
    model.eval()
    input_ids = torch.tensor([char_to_ix[c] for c in start_str], dtype=torch.long).to(CONFIG["device"]).unsqueeze(0)
    hidden_state = None
    past_states = []
    output = start_str
    
    latent_hist = []
    symbol_hist = []
    concept_hist = []
    
    # Graph for Symbol -> Concept mapping
    sym_graph = nx.DiGraph()

    print("\n--- Generating Text & Building Graph ---")

    with torch.no_grad():
        # Warmup
        for t in range(input_ids.size(1) - 1):
            _, hidden_state, _, _, past_states, _, _, _, _, _, _ = model(
                input_ids[:, t].unsqueeze(1), hidden_state, past_states, training=False
            )
        current_input = input_ids[:, -1].unsqueeze(1)

        # Generation
        for _ in range(200):
            logits, hidden_state, _, _, past_states, sym_probs, con_probs, _, _, sym_idx, con_idx = model(
                current_input, hidden_state, past_states, training=False
            )
            
            if len(past_states) > CONFIG["memory_span"]:
                past_states = past_states[-CONFIG["memory_span"]:]
            
            # Store Data
            z_mag = torch.sqrt(hidden_state.real**2 + hidden_state.imag**2).squeeze().cpu().numpy()
            latent_hist.append(z_mag)
            symbol_hist.append(sym_probs.squeeze().cpu().numpy())
            concept_hist.append(con_probs.squeeze().cpu().numpy())
            
            # Update Graph
            s_node = f"S_{sym_idx.item()}"
            c_node = f"C_{con_idx.item()}"
            sym_graph.add_edge(s_node, c_node)
            
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_ix = torch.multinomial(probs, 1)
            output += ix_to_char[next_ix.item()]
            current_input = next_ix

    print(output)
    
    # --- Visualization ---
    
    # 1. Latent Dynamics
    plt.figure(figsize=(10, 3))
    plt.imshow(np.array(latent_hist).T, aspect='auto', cmap='magma')
    plt.title("Latent Dynamics"); plt.show()
    
    # 2. Symbolic Activation
    plt.figure(figsize=(10, 3))
    plt.imshow(np.array(symbol_hist).T, aspect='auto', cmap='viridis')
    plt.title("Symbolic Activation"); plt.show()
    
    # 3. Concept Activation
    plt.figure(figsize=(10, 3))
    plt.imshow(np.array(concept_hist).T, aspect='auto', cmap='plasma')
    plt.title("Concept Activation"); plt.show()
    
    # 4. Network Graph
    if len(sym_graph.nodes) > 0:
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(sym_graph, seed=42)
        color_map = []
        for node in sym_graph:
            if node.startswith('S'): color_map.append('skyblue')
            else: color_map.append('salmon')
        nx.draw(sym_graph, pos, node_color=color_map, with_labels=True, node_size=800, alpha=0.8, arrows=True)
        plt.title("Symbol-Concept Interaction Graph")
        plt.show()

# ==========================================
# 9. Execution
# ==========================================
if __name__ == "__main__":
    trained_model = train_model()
    generate_text(trained_model)
