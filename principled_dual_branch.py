import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


# ======================= Basic Modules =======================

class LayerNorm(nn.Module):
    """Layer normalization with optional learnable parameters"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class SiLU(nn.Module):
    """SiLU/Swish activation"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, ff_mult=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * ff_mult)
        self.fc2 = nn.Linear(hidden_size * ff_mult, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


# ======================= RG-LRU (from original) =======================

class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.constant_(self.W_a.bias, 1.0)
        nn.init.zeros_(self.W_o.bias)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        batch_size, seq_len, hidden_size = x.shape
        
        input_x = self.W_x(x)
        gate_a = torch.sigmoid(self.W_a(x))
        
        h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = gate_a[:, t, :] * h + (1 - gate_a[:, t, :]) * input_x[:, t, :]
            outputs.append(h)
        output = torch.stack(outputs, dim=1)
        
        output = self.W_o(output) * torch.sigmoid(self.W_o(x))
        output = self.dropout(output)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        return residual + output


class BidirectionalRGLRU(nn.Module):
    """Bidirectional RG-LRU"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.rglru_forward = RGLRU(hidden_size, dropout=0.0)
        self.rglru_backward = RGLRU(hidden_size, dropout=0.0)
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        forward_out = self.rglru_forward(x, mask)
        x_flipped = torch.flip(x, dims=[1])
        mask_flipped = torch.flip(mask, dims=[1]) if mask is not None else None
        backward_out = self.rglru_backward(x_flipped, mask_flipped)
        backward_out = torch.flip(backward_out, dims=[1])
        combined = torch.cat([forward_out, backward_out], dim=-1)
        output = self.fusion_proj(combined)
        output = self.norm(output)
        output = self.dropout(output)
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        return residual + output


# ======================= TRUE SPARSE MoE =======================

class Expert(nn.Module):
    """Single Expert Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    """
    True Sparse Mixture of Experts
    
    Key differences from dense MoE:
    1. Only top-K experts are activated per token (sparse)
    2. Load balancing loss to prevent expert collapse
    3. Auxiliary loss for router learning
    
    Reference: Switch Transformer, GShard
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2, 
                 dropout=0.1, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Router (gating network)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )
        
        # Output projection (optional)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # For analysis
        self.last_router_probs = None
        self.last_load_balance_loss = None
        
    def forward(self, x, return_aux_loss=True):
        """
        Args:
            x: (B, L, D) input
            return_aux_loss: whether to compute load balancing loss
        Returns:
            output: (B, L, D)
            aux_loss: load balancing loss (scalar)
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute router logits and probabilities
        router_logits = self.router(x)  # (B, L, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, num_experts)
        self.last_router_probs = router_probs.detach()
        
        # Select top-K experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (B, L, top_k)
        
        # Normalize top-k probabilities (so they sum to 1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute expert outputs - only for selected experts
        # Reshape for efficient computation
        x_flat = x.view(-1, dim)  # (B*L, D)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (B*L, top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)  # (B*L, top_k)
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens route to this expert
            expert_mask = (top_k_indices_flat == expert_idx).any(dim=-1)  # (B*L,)
            
            if expert_mask.sum() > 0:
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]  # (num_tokens, D)
                
                # Compute expert output
                expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, D)
                
                # Get the weight for this expert for these tokens
                expert_weights = torch.zeros(expert_mask.sum(), device=x.device)
                for k in range(self.top_k):
                    k_mask = top_k_indices_flat[expert_mask, k] == expert_idx
                    expert_weights[k_mask] = top_k_probs_flat[expert_mask, k][k_mask]
                
                # Weighted contribution
                output_flat[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        # Reshape back
        output = output_flat.view(batch_size, seq_len, dim)
        output = self.output_proj(output)
        
        # Compute load balancing loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if return_aux_loss and self.training:
            # Load balancing loss: encourage uniform distribution across experts
            # f_i = fraction of tokens routed to expert i
            # P_i = average router probability for expert i
            # loss = num_experts * sum(f_i * P_i)
            
            # Fraction of tokens routed to each expert (based on hard assignment)
            one_hot_indices = F.one_hot(top_k_indices, self.num_experts).float()  # (B, L, top_k, E)
            tokens_per_expert = one_hot_indices.sum(dim=[0, 1, 2]) / (batch_size * seq_len * self.top_k)  # (E,)
            
            # Average router probability for each expert
            mean_router_prob = router_probs.mean(dim=[0, 1])  # (E,)
            
            # Load balance loss
            aux_loss = self.num_experts * (tokens_per_expert * mean_router_prob).sum()
            self.last_load_balance_loss = aux_loss.item()
        
        return output, aux_loss * self.load_balance_weight
    
    def get_expert_usage_stats(self):
        """Return expert usage statistics for analysis"""
        if self.last_router_probs is None:
            return None
        
        # Average probability assigned to each expert
        mean_probs = self.last_router_probs.mean(dim=[0, 1])  # (num_experts,)
        return {
            f'expert_{i}_prob': mean_probs[i].item() 
            for i in range(self.num_experts)
        }


# ======================= SYMMETRIC DELTA LLM ADAPTER =======================

class SymmetricDeltaLLMAdapter(nn.Module):
    """
    Symmetric Delta LLM Adapter with True Sparse MoE
    
    Key design principles:
    1. Output delta (increment), not full features
    2. Use Sparse MoE for true expert specialization
    3. Maintain same gradient dynamics as sequence branch
    
    The adapter transforms LLM embeddings to delta signals that
    modify the input representation, rather than replacing it.
    """
    def __init__(self, llm_dim, hidden_size, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        
        # Determine intermediate dimension
        mid_dim = max(hidden_size * 2, llm_dim // 4) if llm_dim > 1024 else hidden_size
        
        # Initial projection (handles dimension mismatch)
        self.input_proj = nn.Sequential(
            nn.Linear(llm_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_size),
        )
        
        # Sparse MoE for specialization
        self.sparse_moe = SparseMoE(
            input_dim=hidden_size,
            hidden_dim=hidden_size * 2,
            output_dim=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        
        # Delta gate: controls how much delta to apply
        # Takes both LLM features and current hidden state
        self.delta_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.output_norm = LayerNorm(hidden_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.input_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, llm_emb, hidden, mask=None):
        """
        Args:
            llm_emb: (B, L, llm_dim) - LLM embeddings
            hidden: (B, L, hidden_size) - current hidden state (for gating)
            mask: (B, L) - padding mask
        Returns:
            delta: (B, L, hidden_size) - delta to add to hidden
            aux_loss: load balancing loss
        """
        # Project LLM embeddings to hidden_size
        projected = self.input_proj(llm_emb)  # (B, L, hidden_size)
        
        # Apply Sparse MoE
        moe_out, aux_loss = self.sparse_moe(projected)  # (B, L, hidden_size)
        
        # Compute delta gate (based on both LLM features and hidden state)
        gate_input = torch.cat([moe_out, hidden], dim=-1)
        gate = self.delta_gate(gate_input)  # (B, L, hidden_size)
        
        # Apply gate to get final delta
        delta = gate * moe_out  # (B, L, hidden_size)
        delta = self.output_norm(delta)
        
        if mask is not None:
            delta = delta * mask.unsqueeze(-1)
        
        return delta, aux_loss
    
    def get_moe_stats(self):
        """Get MoE usage statistics"""
        return self.sparse_moe.get_expert_usage_stats()


# ======================= PREDICTION-GUIDED ROUTER (PGR) =======================

class PredictionGuidedRouter(nn.Module):
    """
    Prediction-Guided Router using Thompson Sampling style approach
    
    Core idea: 
    - Treat branch selection as a Multi-Armed Bandit problem
    - Use prediction accuracy as reward signal
    - Balance exploration vs exploitation
    
    Implementation:
    - Maintain Beta distribution parameters (α, β) for each branch
    - Sample weights from Beta distribution during training (exploration)
    - Use posterior mean during inference (exploitation)
    - Update α, β based on prediction correctness
    
    Mathematical foundation:
    - Beta(α, β) is conjugate prior for Bernoulli likelihood
    - After observing success, update: α += reward
    - After observing failure, update: β += (1 - reward)
    - Thompson Sampling: sample from posterior for action selection
    
    Reference: Thompson Sampling for Contextual Bandits
    """
    def __init__(self, hidden_size, num_branches=2, prior_alpha=1.0, prior_beta=1.0,
                 context_dim=None, use_context=True, ema_decay=0.99):
        super().__init__()
        self.num_branches = num_branches
        self.use_context = use_context
        self.ema_decay = ema_decay
        
        # Beta distribution parameters (learnable priors)
        # α and β control the shape of the distribution
        # Higher α -> prefer this branch, higher β -> avoid this branch
        self.register_buffer('alpha', torch.ones(num_branches) * prior_alpha)
        self.register_buffer('beta', torch.ones(num_branches) * prior_beta)
        
        # EMA of rewards for each branch (for monitoring)
        self.register_buffer('reward_ema', torch.ones(num_branches) * 0.5)
        
        # Context-aware router (optional, can learn to adjust based on input)
        if use_context:
            context_dim = context_dim or hidden_size
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, num_branches),
            )
        else:
            self.context_encoder = None
        
        # For warmup: fixed equal weights before enough data
        self.register_buffer('forward_count', torch.tensor(0))
        self.warmup_forwards = 2000  # Use equal weights for first N forwards
        
        # Store last weights for analysis
        self.last_weights = None
        self.last_sampled_weights = None
        
    def forward(self, context=None, temperature=1.0):
        """
        Compute branch weights using Thompson Sampling
        
        Args:
            context: (B, L, D) optional context for context-aware routing
            temperature: controls exploration (higher = more exploration)
        
        Returns:
            weights: (B, L, num_branches) or (num_branches,) routing weights
        """
        self.forward_count += 1
        
        # During warmup, use equal weights
        if self.forward_count < self.warmup_forwards:
            base_weights = torch.ones(self.num_branches, device=self.alpha.device) / self.num_branches
        else:
            if self.training:
                # Thompson Sampling: sample from Beta posterior
                # More exploration during training
                dist = Beta(self.alpha, self.beta)
                sampled = dist.rsample()  # Reparameterized sampling for gradients
                
                # Apply temperature
                sampled = sampled ** (1.0 / temperature)
                
                # Normalize to sum to 1
                base_weights = sampled / (sampled.sum() + 1e-8)
                self.last_sampled_weights = sampled.detach()
            else:
                # Inference: use posterior mean (exploitation)
                # E[Beta(α, β)] = α / (α + β)
                mean = self.alpha / (self.alpha + self.beta + 1e-8)
                base_weights = mean / (mean.sum() + 1e-8)
        
        # Context-aware adjustment (if enabled)
        if self.use_context and context is not None and self.context_encoder is not None:
            # Pool context to get sequence representation
            if context.dim() == 3:
                context_pooled = context.mean(dim=1)  # (B, D)
            else:
                context_pooled = context
            
            # Compute context-dependent adjustment
            context_logits = self.context_encoder(context_pooled)  # (B, num_branches)
            context_weights = F.softmax(context_logits, dim=-1)
            
            # Combine base weights with context-dependent weights
            # base_weights: (num_branches,), context_weights: (B, num_branches)
            weights = 0.5 * base_weights.unsqueeze(0) + 0.5 * context_weights
        else:
            weights = base_weights
        
        self.last_weights = weights.detach() if isinstance(weights, torch.Tensor) else weights
        return weights
    
    def update_with_reward(self, rewards, branch_indices=None, hard_update=False, normalize_reward=True):
        """
        Update Beta distribution parameters based on rewards.
        
        Args:
            rewards: scalar or length-num_branches reward(s) in [0, 1]
            branch_indices: (B, num_branches) weights used for scalar reward
            hard_update: when rewards is vector, update only best branch
            normalize_reward: normalize vector rewards to relative scale
        """
        if not self.training:
            return

        # Vector rewards (branch-specific)
        reward_vec = None
        if isinstance(rewards, torch.Tensor) and rewards.numel() == self.num_branches:
            reward_vec = rewards.detach().float().view(-1)
        elif isinstance(rewards, (list, tuple)) and len(rewards) == self.num_branches:
            reward_vec = torch.tensor(rewards, device=self.alpha.device, dtype=self.alpha.dtype)

        if reward_vec is not None:
            reward_vec = reward_vec.clamp(0.0, 1.0)
            if normalize_reward:
                reward_sum = reward_vec.sum() + 1e-8
                reward_vec = reward_vec / reward_sum
            self.reward_ema = self.ema_decay * self.reward_ema + (1 - self.ema_decay) * reward_vec
            if hard_update:
                best_idx = int(torch.argmax(reward_vec).item())
                for i in range(self.num_branches):
                    if i == best_idx:
                        self.alpha[i] = self.alpha[i] + reward_vec[i] * 0.1
                        self.beta[i] = self.beta[i] + (1 - reward_vec[i]) * 0.1
            else:
                for i in range(self.num_branches):
                    self.alpha[i] = self.alpha[i] + reward_vec[i] * 0.1
                    self.beta[i] = self.beta[i] + (1 - reward_vec[i]) * 0.1
        else:
            # Scalar reward
            if isinstance(rewards, torch.Tensor):
                mean_reward = rewards.mean().item()
            else:
                mean_reward = rewards
            mean_reward = max(0.0, min(1.0, mean_reward))
            self.reward_ema = self.ema_decay * self.reward_ema + (1 - self.ema_decay) * mean_reward
            if branch_indices is not None and isinstance(branch_indices, torch.Tensor):
                if branch_indices.dim() == 2:
                    branch_weights = branch_indices.mean(dim=0)
                else:
                    branch_weights = branch_indices
                for i in range(self.num_branches):
                    weight = branch_weights[i].item()
                    self.alpha[i] = self.alpha[i] + weight * mean_reward * 0.1
                    self.beta[i] = self.beta[i] + weight * (1 - mean_reward) * 0.1
            else:
                for i in range(self.num_branches):
                    self.alpha[i] = self.alpha[i] + mean_reward * 0.05
                    self.beta[i] = self.beta[i] + (1 - mean_reward) * 0.05
        
        # Prevent parameters from growing too large (numerical stability)
        # Normalize to keep sum roughly constant
        total = (self.alpha + self.beta).mean()
        if total > 100:
            scale = 50 / total
            self.alpha = self.alpha * scale
            self.beta = self.beta * scale
    
    def get_stats(self):
        """Get router statistics for analysis"""
        with torch.no_grad():
            mean = self.alpha / (self.alpha + self.beta + 1e-8)
            var = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
            
        return {
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'posterior_mean': mean.tolist(),
            'posterior_var': var.tolist(),
            'reward_ema': self.reward_ema.tolist(),
            'forward_count': self.forward_count.item(),
        }


# ======================= PRINCIPLED DUAL-BRANCH BLOCK =======================

class PrincipledDualBranchBlock(nn.Module):
    """
    Principled Dual-Branch Block
    
    Key innovations:
    1. Symmetric Delta: Both branches output delta (not full features)
    2. Sparse MoE: LLM branch uses true sparse expert routing
    3. Prediction-Guided Router: MAB-style weight assignment
    
    Architecture:
        Input ─┬─> RG-LRU ─> delta_seq ─┐
               │                         │
               └─> LLM MoE ─> delta_sem ─┴─> PGR Weighted Fusion ─> FFN ─> Output
    """
    def __init__(self, hidden_size, llm_dim=3584, dropout=0.1, bidirectional=True,
                 num_experts=4, top_k=2, use_pgr=True, pgr_warmup=2000,
                 router_mode='pgr', use_pop_gate=False, pop_gate_strength=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_pgr = use_pgr
        self.router_mode = router_mode
        self.use_pop_gate = use_pop_gate
        self.pop_gate_strength = pop_gate_strength
        
        # ===== Branch 1: Sequence (RG-LRU) =====
        if bidirectional:
            self.seq_branch = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.seq_branch = RGLRU(hidden_size, dropout=dropout)
        
        # ===== Branch 2: Semantic (LLM with Sparse MoE) =====
        self.sem_branch = SymmetricDeltaLLMAdapter(
            llm_dim=llm_dim,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        
        # ===== Router =====
        if self.router_mode == 'pgr' and use_pgr:
            self.router = PredictionGuidedRouter(
                hidden_size=hidden_size,
                num_branches=2,
                use_context=True,
            )
            self.router.warmup_forwards = pgr_warmup
            self.learned_gate = None
        elif self.router_mode == 'learned':
            self.router = None
            self.learned_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 2),
            )
            self.pop_gate = nn.Sequential(
                nn.Linear(1, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 2),
            )
        else:
            # Fallback: learnable static weights
            self.router = None
            self.learned_gate = None
            self.pop_gate = None
            self.static_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # ===== Fusion and Output =====
        self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_norm = LayerNorm(hidden_size)
        self.ffn = PositionwiseFeedForward(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # For analysis
        self.last_weights = None
        self.last_seq_delta = None
        self.last_sem_delta = None
        self.last_aux_loss = None
        
    def forward(self, hidden, c, mask, llm_emb=None, force_branch=None, record_stats=True,
                item_popularity=None):
        """
        Args:
            hidden: (B, L, D) - input sequence embeddings
            c: unused (for API compatibility)
            mask: (B, L) - padding mask
            llm_emb: (B, L, llm_dim) - LLM embeddings
            force_branch: 'seq' | 'sem' | None
            record_stats: whether to update debug stats
        Returns:
            output: (B, L, D)
        """
        residual = hidden
        
        # ===== Branch 1: Sequence (outputs delta via residual design) =====
        seq_out = self.seq_branch(hidden, mask)
        seq_delta = seq_out - hidden  # Explicit delta
        if record_stats:
            self.last_seq_delta = seq_delta.detach()
        
        # ===== Branch 2: Semantic (outputs delta directly) =====
        if llm_emb is not None:
            sem_delta, aux_loss = self.sem_branch(llm_emb, hidden, mask)
            if record_stats:
                self.last_aux_loss = aux_loss
        else:
            sem_delta = torch.zeros_like(hidden)
            if record_stats:
                self.last_aux_loss = torch.tensor(0.0, device=hidden.device)
        if record_stats:
            self.last_sem_delta = sem_delta.detach()
        
        # ===== Compute Routing Weights =====
        if force_branch == 'seq':
            w_seq = torch.ones((hidden.size(0), 1, 1), device=hidden.device)
            w_sem = torch.zeros((hidden.size(0), 1, 1), device=hidden.device)
            weights = torch.cat([w_seq, w_sem], dim=-1)
        elif force_branch == 'sem':
            w_seq = torch.zeros((hidden.size(0), 1, 1), device=hidden.device)
            w_sem = torch.ones((hidden.size(0), 1, 1), device=hidden.device)
            weights = torch.cat([w_seq, w_sem], dim=-1)
        elif self.router_mode == 'learned' and self.learned_gate is not None:
            context = hidden.mean(dim=1)  # (B, D)
            gate_logits = self.learned_gate(context)  # (B, 2)
            if self.use_pop_gate and item_popularity is not None and self.pop_gate is not None:
                if item_popularity.dim() == 2:
                    if mask is not None:
                        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                        pop = (item_popularity * mask).sum(dim=1, keepdim=True) / denom
                    else:
                        pop = item_popularity.mean(dim=1, keepdim=True)
                else:
                    pop = item_popularity.view(-1, 1)
                pop = pop.clamp(0.0, 1.0)
                pop_bias = self.pop_gate(pop)  # (B, 2)
                gate_logits = gate_logits + self.pop_gate_strength * pop_bias
            weights = F.softmax(gate_logits, dim=-1).unsqueeze(1)  # (B, 1, 2)
            w_seq = weights[:, :, 0:1]
            w_sem = weights[:, :, 1:2]
        elif self.use_pgr and self.router is not None:
            # PGR: context-aware weights with Thompson Sampling
            weights = self.router(hidden)  # (B, 2) or (2,)
            
            if weights.dim() == 1:
                # Broadcast to (B, 1, 2) for element-wise multiplication
                weights = weights.unsqueeze(0).unsqueeze(0)
                weights = weights.expand(hidden.size(0), 1, -1)
            else:
                weights = weights.unsqueeze(1)  # (B, 1, 2)
            
            w_seq = weights[:, :, 0:1]  # (B, 1, 1)
            w_sem = weights[:, :, 1:2]
        else:
            # Static weights
            weights = F.softmax(self.static_weights, dim=0)
            w_seq = weights[0]
            w_sem = weights[1]
        
        if record_stats:
            self.last_weights = weights.detach() if isinstance(weights, torch.Tensor) else weights
        
        # ===== Fuse Deltas =====
        fused_delta = w_seq * seq_delta + w_sem * sem_delta
        
        # ===== Apply Fusion and FFN =====
        fused = self.fusion_proj(fused_delta)
        fused = self.dropout(fused)
        output = self.fusion_norm(residual + fused)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        # FFN
        output = self.ffn(output)
        
        return self.dropout(output)
    
    def get_aux_loss(self):
        """Get auxiliary loss (MoE load balancing)"""
        return self.last_aux_loss if self.last_aux_loss is not None else torch.tensor(0.0)


# ======================= PRINCIPLED DUAL-BRANCH ENCODER =======================

class PrincipledDualBranch_rep(nn.Module):
    """
    Principled Dual-Branch Encoder
    
    A theoretically grounded dual-branch architecture that addresses:
    1. Asymmetry in delta vs full-feature outputs
    2. Ineffective MoE (dense instead of sparse)
    3. Simple router without learning from feedback
    
    This encoder is designed to fully unleash the potential of both
    sequence modeling and LLM semantic information.
    """
    def __init__(self, args, item_num=None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = getattr(args, 'last', 2)
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.num_experts = getattr(args, 'num_experts', 4)
        self.top_k = getattr(args, 'top_k', 2)
        self.pgr_warmup = getattr(args, 'pgr_warmup_forwards', 2000)
        self.router_mode = getattr(args, 'router_mode', 'pgr')
        self.use_pop_gate = getattr(args, 'use_cold_start_gate', False)
        self.pop_gate_strength = getattr(args, 'pop_gate_strength', 1.0)
        
        # Stack of dual-branch blocks
        self.blocks = nn.ModuleList([
            PrincipledDualBranchBlock(
                hidden_size=self.hidden_size,
                llm_dim=self.llm_dim,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                num_experts=self.num_experts,
                top_k=self.top_k,
                use_pgr=True,
                pgr_warmup=self.pgr_warmup,
                router_mode=self.router_mode,
                use_pop_gate=self.use_pop_gate,
                pop_gate_strength=self.pop_gate_strength,
            )
            for _ in range(self.n_blocks)
        ])
        
        # Global output norm
        self.output_norm = LayerNorm(self.hidden_size)
        
        # For tracking
        self.supports_popularity = True
        self.supports_force_branch = True
        self.supports_record_stats = True
        self.supports_gate_weights = True
        self.item_num = item_num
        
    def forward(self, hidden, c, mask, llm_emb=None, item_popularity=None, force_branch=None, record_stats=True):
        """
        Args:
            hidden: (B, L, D) - item embeddings
            c: unused
            mask: (B, L) - padding mask
            llm_emb: (B, L, llm_dim) - LLM embeddings
            item_popularity: unused (for API compatibility)
            force_branch: 'seq' | 'sem' | None
            record_stats: whether to update debug stats
        Returns:
            hidden: (B, L, D) - final output
            encode: (B, L, D) - intermediate encoding
        """
        i = 0
        encode = None
        total_aux_loss = 0.0
        
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb,
                           force_branch=force_branch, record_stats=record_stats,
                           item_popularity=item_popularity)
            if record_stats:
                total_aux_loss = total_aux_loss + block.get_aux_loss()
            
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
        
        hidden = self.output_norm(hidden)
        
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1)
        
        # Store auxiliary loss for training
        if record_stats:
            self._last_aux_loss = total_aux_loss / self.n_blocks
            if len(self.blocks) > 0:
                self._last_gate_weights = self.blocks[-1].last_weights
        
        return hidden, encode

    def get_last_gate_weights(self):
        """Return last block's gate weights for supervision"""
        return getattr(self, '_last_gate_weights', None)
    
    def update_router_with_reward(self, reward, hard_update=False, normalize_reward=True):
        """
        Update PGR routers with prediction reward
        
        Call this after computing prediction accuracy in training loop.
        
        Args:
            reward: float or tensor, prediction accuracy (0-1)
            hard_update: update only best branch when using vector rewards
            normalize_reward: normalize vector rewards to relative scale
        """
        for block in self.blocks:
            if hasattr(block, 'router') and block.router is not None:
                # Get last used weights for this block
                last_weights = block.last_weights
                if last_weights is not None and last_weights.dim() > 1:
                    last_weights = last_weights.mean(dim=[0, 1])  # Average across batch and seq
                # Vector rewards (branch-specific) should not use last_weights
                if isinstance(reward, (list, tuple)) or (isinstance(reward, torch.Tensor) and reward.numel() == block.router.num_branches):
                    block.router.update_with_reward(reward, None, hard_update=hard_update, normalize_reward=normalize_reward)
                else:
                    block.router.update_with_reward(reward, last_weights, hard_update=hard_update, normalize_reward=normalize_reward)
    
    def get_aux_loss(self):
        """Get total auxiliary loss (MoE load balancing)"""
        return getattr(self, '_last_aux_loss', torch.tensor(0.0))
    
    def get_entropy_loss(self):
        """API compatibility"""
        return torch.tensor(0.0)
    
    def get_gate_stats(self):
        """Get detailed statistics for analysis"""
        stats = []
        
        for i, block in enumerate(self.blocks):
            block_stats = {'block': i}
            
            # Router weights
            if block.last_weights is not None:
                weights = block.last_weights
                if weights.dim() > 1:
                    weights = weights.mean(dim=[0, 1])
                block_stats['w_seq'] = weights[0].item() if hasattr(weights, '__getitem__') else weights
                block_stats['w_sem'] = weights[1].item() if hasattr(weights, '__getitem__') else 1 - weights
            
            # PGR stats
            if hasattr(block, 'router') and block.router is not None:
                pgr_stats = block.router.get_stats()
                block_stats['pgr'] = pgr_stats
            
            # MoE stats
            if hasattr(block.sem_branch, 'get_moe_stats'):
                moe_stats = block.sem_branch.get_moe_stats()
                if moe_stats:
                    block_stats['moe'] = moe_stats
            
            stats.append(block_stats)
        
        return stats
    
    def get_last_branch_outputs(self):
        """Get last block's branch outputs for analysis"""
        if len(self.blocks) > 0:
            last_block = self.blocks[-1]
            return last_block.last_seq_delta, last_block.last_sem_delta
        return None, None


# ======================= UTILITY FUNCTIONS =======================

def compute_prediction_reward(scores, labels, k=10):
    """
    Compute prediction accuracy as reward signal for PGR
    
    Args:
        scores: (B, num_items) prediction scores
        labels: (B,) or (B, 1) ground truth labels
    
    Returns:
        reward: scalar in [0, 1]
    """
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    
    # Top-K accuracy as reward
    _, topk_indices = torch.topk(scores, k, dim=-1)
    hits = (topk_indices == labels.unsqueeze(-1)).any(dim=-1).float()
    
    return hits.mean()


def create_principled_dual_branch_from_args(args):
    """Factory function to create encoder from args"""
    return PrincipledDualBranch_rep(args, getattr(args, 'item_num', None))
