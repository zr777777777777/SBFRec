import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= Mamba Import =======================

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("[INFO] mamba_ssm imported. Using real Mamba.")
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARNING] mamba_ssm not available. Using LSTM fallback.")


# ======================= RG-LRU (Real-Gated Linear Recurrent Unit) =======================

class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit from Griffin (DeepMind 2024).
    
    A simplified linear recurrent unit designed for efficiency:
    - Uses real-valued (not complex) recurrence
    - Gated mechanism similar to GRU but linear
    - O(n) complexity, hardware-friendly
    
    Recurrence: h_t = a_t * h_{t-1} + (1 - a_t) * (W_x * x_t)
    Output: y_t = h_t
    where a_t = sigmoid(W_a * x_t + b_a)
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input projection
        self.W_x = nn.Linear(hidden_size, hidden_size)
        
        # Recurrence gate (controls how much to forget)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        
        # Output gate (optional, for better expressiveness)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize gate bias to be slightly positive (start with more memory)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.constant_(self.W_a.bias, 1.0)  # Start with high recurrence (remember more)
        nn.init.zeros_(self.W_o.bias)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, H) input sequence
            mask: (B, L) padding mask
        Returns:
            (B, L, H) output sequence
        """
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute input and gate for all timesteps
        input_x = self.W_x(x)  # (B, L, H)
        gate_a = torch.sigmoid(self.W_a(x))  # (B, L, H) - recurrence gate
        
        # Linear recurrence (can be parallelized with parallel scan, but sequential here for clarity)
        h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # h_t = a_t * h_{t-1} + (1 - a_t) * input_t
            h = gate_a[:, t, :] * h + (1 - gate_a[:, t, :]) * input_x[:, t, :]
            outputs.append(h)
        
        output = torch.stack(outputs, dim=1)  # (B, L, H)
        
        # Output gate
        output = self.W_o(output) * torch.sigmoid(self.W_o(x))
        output = self.dropout(output)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
            
        return residual + output


class BidirectionalRGLRU(nn.Module):
    """
    Bidirectional RG-LRU: runs forward and backward, then fuses results.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Forward and backward RG-LRU
        self.rglru_forward = RGLRU(hidden_size, dropout=0.0)  # No dropout inside
        self.rglru_backward = RGLRU(hidden_size, dropout=0.0)
        
        # Fusion: concatenate forward + backward, project back
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        
        # Forward pass
        forward_out = self.rglru_forward(x, mask)
        
        # Backward pass: flip, process, flip back
        x_flipped = torch.flip(x, dims=[1])
        mask_flipped = torch.flip(mask, dims=[1]) if mask is not None else None
        backward_out = self.rglru_backward(x_flipped, mask_flipped)
        backward_out = torch.flip(backward_out, dims=[1])
        
        # Concatenate and fuse
        combined = torch.cat([forward_out, backward_out], dim=-1)  # (B, L, 2H)
        output = self.fusion_proj(combined)  # (B, L, H)
        output = self.norm(output)
        output = self.dropout(output)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
            
        return residual + output


# ======================= Enhanced RG-LRU with Conv1d (Griffin-style, PDF建议) =======================

class EnhancedRGLRU(nn.Module):
    """
    Enhanced RG-LRU with parallel Conv1d branch (inspired by Griffin, PDF建议).
    
    改进点：
    1. 增加前置Conv1d分支，捕获局部模式（弥补RNN对短距离学习的不足）
    2. 通过GLU门控融合Conv和RNN输出
    3. 更好的初始化策略
    
    参考：Griffin论文中的循环块设计
    """
    def __init__(self, hidden_size, kernel_size=3, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # RG-LRU 主分支
        self.W_x = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        
        # Conv1d 分支 (Griffin风格 - Depthwise Conv)
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=hidden_size  # Depthwise Conv，参数少效率高
        )
        self.conv_proj = nn.Linear(hidden_size, hidden_size)
        
        # GLU融合门控（让模型自动学习何时用RNN何时用Conv）
        self.glu_gate = nn.Linear(hidden_size, hidden_size)
        
        # 输出
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.constant_(self.W_a.bias, 1.0)  # 初始倾向于记忆
        nn.init.zeros_(self.W_o.bias)
        nn.init.xavier_uniform_(self.conv_proj.weight)
        nn.init.zeros_(self.conv_proj.bias)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        batch_size, seq_len, hidden_size = x.shape
        
        # RG-LRU 分支（全局时序）
        input_x = self.W_x(x)
        gate_a = torch.sigmoid(self.W_a(x))
        
        h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = gate_a[:, t, :] * h + (1 - gate_a[:, t, :]) * input_x[:, t, :]
            outputs.append(h)
        rnn_out = torch.stack(outputs, dim=1)
        
        # Conv1d 分支（局部模式）
        conv_in = x.transpose(1, 2)  # (B, H, L)
        conv_out = self.conv(conv_in).transpose(1, 2)  # (B, L, H)
        conv_out = self.conv_proj(conv_out)
        
        # GLU 门控融合（PDF建议：让RG-LRU分支内部也有局部处理能力）
        gate = torch.sigmoid(self.glu_gate(x))
        fused = gate * rnn_out + (1 - gate) * conv_out
        
        # 输出
        output = self.W_o(fused) * torch.sigmoid(self.W_o(x))
        output = self.dropout(output)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
            
        return residual + output


class BidirectionalEnhancedRGLRU(nn.Module):
    """双向增强版RG-LRU（带Conv1d分支）"""
    def __init__(self, hidden_size, kernel_size=3, dropout=0.1):
        super().__init__()
        self.rglru_forward = EnhancedRGLRU(hidden_size, kernel_size, dropout=0.0)
        self.rglru_backward = EnhancedRGLRU(hidden_size, kernel_size, dropout=0.0)
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
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
        return output



class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative position awareness.
    Applied directly to embeddings so RG-LRU can capture order relationships.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even hidden size."
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_cos_sin(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def apply_rotary(self, x):
        """
        Args:
            x: (B, L, D)
        """
        cos, sin = self._get_cos_sin(x.size(1), x.device, x.dtype)
        cos = cos.unsqueeze(0)  # (1, L, D)
        sin = sin.unsqueeze(0)  # (1, L, D)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_normal_(self.w_2.weight)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) 
                   for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class LocalAttention(nn.Module):
    """
    Local (Sliding Window) Attention for capturing local patterns.
    
    Key insight: Mamba is weak at local modeling in short sequences.
    Local attention with small window (e.g., 5) provides strong local context
    with O(n * window) complexity.
    
    Args:
        hidden_size: model dimension
        heads: number of attention heads
        window_size: size of attention window (default 5)
        dropout: dropout rate
    """
    def __init__(self, hidden_size, heads=4, window_size=5, dropout=0.1):
        super().__init__()
        assert hidden_size % heads == 0
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.window_size = window_size
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, D) input tensor
            mask: (B, L) padding mask
        Returns:
            (B, L, D) output with local attention applied
        """
        B, L, D = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        
        # Create local attention mask
        # Each position attends to [i - window//2, i + window//2]
        local_mask = self._create_local_mask(L, x.device)  # (L, L)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        
        # Apply local mask (positions outside window get -inf)
        attn_scores = attn_scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(padding_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)  # Handle all-masked rows
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)  # (B, H, L, d)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out
    
    def _create_local_mask(self, seq_len, device):
        """
        Create local attention mask where each position attends to 
        positions within window_size distance.
        
        Returns:
            (L, L) boolean mask, True = attend, False = mask out
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Calculate distance matrix
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (L, L)
        
        # Positions within window can attend (causal: only attend to past and current)
        # For recommendation, we want bidirectional local attention
        half_window = self.window_size // 2
        mask = dist <= half_window
        
        return mask


class LocalAttentionBlock(nn.Module):
    """Local Attention block with residual connection and FFN"""
    def __init__(self, hidden_size, heads=4, window_size=5, dropout=0.1):
        super().__init__()
        self.local_attn = LocalAttention(hidden_size, heads, window_size, dropout)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        x = self.local_attn(x, mask)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return residual + x


# ======================= Transformer Encoder =======================

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)]
        )

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for transformer in self.transformer_blocks:
            i += 1
            hidden = transformer.forward(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode


# ======================= Transformer + LLM Encoder =======================

class TransformerLLMBlock(nn.Module):
    """
    Transformer block with LLM semantic fusion
    
    Architecture:
        Input ─────> Self-Attention ─┐
                                     │
        LLM ─> Adapter ──────────────┼─> Two-way Gated Fusion ─> FFN ─> Output
    """
    def __init__(self, hidden_size, attn_heads=4, llm_dim=768, dropout=0.1, use_llm=True):
        super().__init__()
        self.use_llm = use_llm
        self.hidden_size = hidden_size
        
        # Self-attention
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.attn_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        
        # LLM adapter
        if use_llm:
            self.llm_adapter = LLMAdapter(llm_dim, hidden_size, dropout)
            # Two-way fusion: attention output + LLM
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.Sigmoid()
            )
        
        # Feed-forward
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        # Self-attention
        attn_out = self.attn_sublayer(
            hidden, 
            lambda x: self.attention(x, x, x, mask=mask)
        )
        
        if self.use_llm and llm_emb is not None:
            # LLM branch
            llm_out = self.llm_adapter(llm_emb, mask)
            
            # Gated fusion
            gate_input = torch.cat([hidden, attn_out - hidden, llm_out], dim=-1)
            gate = self.fusion_gate(gate_input)
            
            # Fuse: attn_delta + gated LLM
            fused = hidden + (attn_out - hidden) + gate * llm_out
        else:
            fused = attn_out
        
        # Feed-forward
        output = self.ff_sublayer(fused, self.feed_forward)
        return self.dropout(output)


class TransformerLLM_rep(nn.Module):
    """
    Transformer + LLM encoder
    
    对比实验用：验证是否比 RG-LRU+LLM 或 Mamba+LLM 更好
    
    理论优势：
    - Self-attention在短序列上建模能力强
    - 推荐序列通常<50，O(n²)不是瓶颈
    - SASRec/BERT4Rec等已验证Transformer在推荐上的有效性
    """
    def __init__(self, args):
        super(TransformerLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = getattr(args, 'num_heads', 4)
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.llm_dim = getattr(args, 'llm_dim', 768)
        self.use_llm = getattr(args, 'use_llm', True)
        
        self.blocks = nn.ModuleList([
            TransformerLLMBlock(
                self.hidden_size,
                attn_heads=self.heads,
                llm_dim=self.llm_dim,
                dropout=self.dropout,
                use_llm=self.use_llm
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb)
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
        return hidden, encode


# ======================= Mamba Encoder =======================

class MambaBlockWrapper(nn.Module):
    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            self.lstm_proj = nn.Linear(hidden_size, hidden_size)
            
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            x, _ = self.mamba(x)
            x = self.lstm_proj(x)
            
        x = self.dropout(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        return residual + x


class BidirectionalMambaWrapper(nn.Module):
    """
    Bidirectional Mamba: runs forward and backward Mamba, then fuses results.
    This is crucial for sequential recommendation where we have complete history.
    """
    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        if MAMBA_AVAILABLE:
            # Forward Mamba
            self.mamba_forward = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            # Backward Mamba
            self.mamba_backward = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback to bidirectional LSTM
            self.mamba_forward = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
        # Fusion: concatenate forward + backward, project back to hidden_size.
        # When Mamba is unavailable we fall back to a bidirectional LSTM that already
        # outputs `hidden_size`, so skip the extra doubling in that case.
        fusion_in = hidden_size * 2 if MAMBA_AVAILABLE else hidden_size
        self.fusion_proj = nn.Linear(fusion_in, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        
        if MAMBA_AVAILABLE:
            # Forward pass
            forward_out = self.mamba_forward(x)
            
            # Backward pass: flip, process, flip back
            x_flipped = torch.flip(x, dims=[1])
            backward_out = self.mamba_backward(x_flipped)
            backward_out = torch.flip(backward_out, dims=[1])
            
            # Concatenate and fuse
            combined = torch.cat([forward_out, backward_out], dim=-1)  # (B, L, 2H)
            x = self.fusion_proj(combined)  # (B, L, H)
        else:
            x, _ = self.mamba_forward(x)  # Already bidirectional
            x = self.fusion_proj(x)
            
        x = self.dropout(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        return residual + x


class MambaBlock_FF(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, bidirectional=True, d_state=16):
        super().__init__()
        if bidirectional:
            self.mamba = BidirectionalMambaWrapper(hidden_size, d_state=d_state, dropout=dropout)
        else:
            self.mamba = MambaBlockWrapper(hidden_size, d_state=d_state, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask):
        hidden = self.mamba(hidden, mask)
        hidden = self.ff_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Mamba_rep(nn.Module):
    def __init__(self, args):
        super(Mamba_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.d_state = getattr(args, 'd_state', 16)
        
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock_FF(self.hidden_size, self.dropout, 
                          bidirectional=self.bidirectional, d_state=self.d_state) 
             for _ in range(self.n_blocks)]
        )

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for mamba_block in self.mamba_blocks:
            i += 1
            hidden = mamba_block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
            
        return hidden, encode


# ======================= FFT Module =======================

class FFTBlock(nn.Module):
    """
    Enhanced FFT block with Learnable Frequency Filters (inspired by FMLP-Rec).
    
    Key improvements:
    1. Learnable frequency filter weights that adapt during training
    2. Low-pass filtering to suppress high-frequency noise
    3. IFFT to preserve position information
    
    The learnable filters act as data-driven band-pass filters, emphasizing
    meaningful periodic patterns and suppressing noise.
    """
    def __init__(self, hidden_size, seq_len=50, dropout=0.1, num_freq_bands=25, use_amp=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        # Use more frequency bands to preserve information
        self.num_freq_bands = min(num_freq_bands, seq_len // 2 + 1)
        self.use_amp = use_amp
        
        self.input_norm = LayerNorm(hidden_size)
        
        # ===== NEW: Learnable Frequency Filters (FMLP-Rec inspired) =====
        # Each frequency band gets a learnable weight
        # Lower frequencies initialized with higher weights (low-pass prior)
        self.freq_filter_real = nn.Parameter(torch.ones(self.num_freq_bands, hidden_size))
        self.freq_filter_imag = nn.Parameter(torch.ones(self.num_freq_bands, hidden_size))
        
        # Initialize with low-pass prior: lower frequencies get higher weights
        with torch.no_grad():
            for i in range(self.num_freq_bands):
                # Exponential decay: freq 0 gets weight 1.0, higher freqs get lower weights
                weight = math.exp(-0.1 * i)
                self.freq_filter_real.data[i] = weight
                self.freq_filter_imag.data[i] = weight
        
        # Frequency domain processing (after filtering)
        self.freq_real_fc = nn.Linear(hidden_size, hidden_size)
        self.freq_imag_fc = nn.Linear(hidden_size, hidden_size)
        
        # Separate processing for real and imaginary after transformation
        self.freq_real_out = nn.Linear(hidden_size, hidden_size)
        self.freq_imag_out = nn.Linear(hidden_size, hidden_size)
        
        # Output projection after IFFT
        self.output_fc = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        
        # Learnable scale for residual connection
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.freq_real_fc, self.freq_imag_fc, 
                       self.freq_real_out, self.freq_imag_out, self.output_fc]:
            nn.init.xavier_normal_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape
        residual = x
        
        # Input normalization
        x = self.input_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # FFT: time domain -> frequency domain
        x_freq = torch.fft.rfft(x, dim=1)  # [B, F, H] where F = seq_len//2 + 1
        
        # Keep only low frequency bands (filtering high-freq noise)
        x_freq_filtered = x_freq[:, :self.num_freq_bands, :]
        
        # Extract and clamp real/imaginary parts for stability
        real_part = torch.clamp(x_freq_filtered.real, -10, 10)
        imag_part = torch.clamp(x_freq_filtered.imag, -10, 10)
        
        # ===== Apply Learnable Frequency Filters =====
        # Element-wise multiplication with learnable filter weights
        # This allows the model to learn which frequency bands are important
        filter_real = torch.sigmoid(self.freq_filter_real)  # [num_freq, H], bounded [0,1]
        filter_imag = torch.sigmoid(self.freq_filter_imag)
        
        real_part = real_part * filter_real.unsqueeze(0)  # [B, F, H] * [1, F, H]
        imag_part = imag_part * filter_imag.unsqueeze(0)
        
        # Process real and imaginary parts
        real_features = F.gelu(self.freq_real_fc(real_part))
        imag_features = F.gelu(self.freq_imag_fc(imag_part))
        
        # Apply dropout in frequency domain
        real_features = self.dropout(real_features)
        imag_features = self.dropout(imag_features)
        
        # Project back
        real_out = self.freq_real_out(real_features)
        imag_out = self.freq_imag_out(imag_features)
        
        # Reconstruct complex tensor for IFFT
        # Pad back to full frequency length
        full_freq_len = seq_len // 2 + 1
        if self.num_freq_bands < full_freq_len:
            # Zero-pad high frequencies (low-pass filtering effect)
            real_padding = torch.zeros(batch_size, full_freq_len - self.num_freq_bands, hidden_size, 
                                       device=x.device, dtype=real_out.dtype)
            imag_padding = torch.zeros(batch_size, full_freq_len - self.num_freq_bands, hidden_size,
                                       device=x.device, dtype=imag_out.dtype)
            real_out = torch.cat([real_out, real_padding], dim=1)
            imag_out = torch.cat([imag_out, imag_padding], dim=1)
        
        # Create complex tensor
        freq_complex = torch.complex(real_out, imag_out)
        
        # IFFT: frequency domain -> time domain (preserves position info!)
        time_out = torch.fft.irfft(freq_complex, n=seq_len, dim=1)  # [B, L, H]
        
        # Output projection
        time_out = self.output_fc(time_out)
        time_out = self.dropout(time_out)
        
        # Residual connection with learnable scale
        scale = torch.sigmoid(self.scale)
        output = residual + scale * time_out
        output = self.output_norm(output)
        
        # Apply mask
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return output


# ======================= NEW: Enhanced LLM Adapter with MoE =======================

class LLMAdapter(nn.Module):
    """
    Enhanced Adapter with Mixture-of-Experts (MoE) for LLM embeddings.
    
    Inspired by KAR (Knowledge-Augmented Recommendation) and UniSRec.
    
    Key improvements:
    1. Mixture of Experts: Different experts specialize in different item types
       - Expert 1: Popular items (high interaction count)
       - Expert 2: Tail items (sparse interactions)  
       - Expert 3: Category-aware (semantic clustering)
    2. Progressive dimensionality reduction for large LLM embeddings
    3. Cross-attention with behavioral branch for better alignment
    
    This helps cold-start items benefit more from LLM semantics while
    popular items rely more on collaborative signals.
    """
    def __init__(self, llm_dim, hidden_size, dropout=0.1, num_experts=3):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Determine intermediate dimensions based on llm_dim
        if llm_dim > 1024:
            mid_dim = max(hidden_size * 2, llm_dim // 4)
        else:
            mid_dim = hidden_size
        
        # ===== Expert Networks (each specializes in different item types) =====
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if llm_dim > 1024:
                expert = nn.Sequential(
                    nn.Linear(llm_dim, mid_dim),
                    nn.GELU(),
                    nn.LayerNorm(mid_dim),
                    nn.Dropout(dropout),
                    nn.Linear(mid_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                expert = nn.Sequential(
                    nn.Linear(llm_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
            self.experts.append(expert)
        
        # ===== Gating Network: Decides which expert(s) to use per token =====
        # Takes LLM embedding and outputs expert weights
        self.gate_network = nn.Sequential(
            nn.Linear(llm_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_experts),
        )
        
        # ===== Final projection and normalization =====
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        
        # ===== Learnable importance gate (cold-start vs hot items) =====
        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training with large embeddings"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, llm_emb, mask=None):
        """
        Args:
            llm_emb: (B, L, llm_dim) - LLM embeddings for each item in sequence
            mask: (B, L) - 1 for valid items, 0 for padding
        Returns:
            output: (B, L, hidden_size) - projected and gated
        """
        batch_size, seq_len, _ = llm_emb.shape
        
        # Compute gating weights (which experts to use)
        gate_logits = self.gate_network(llm_emb)  # (B, L, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, L, num_experts)
        
        # Get each expert's output
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(llm_emb)  # (B, L, hidden_size)
            expert_outputs.append(expert_out)
        
        # Stack and apply gating: weighted sum of experts
        expert_stack = torch.stack(expert_outputs, dim=-1)  # (B, L, hidden_size, num_experts)
        gate_weights_expanded = gate_weights.unsqueeze(2)  # (B, L, 1, num_experts)
        
        # Weighted combination
        h = (expert_stack * gate_weights_expanded).sum(dim=-1)  # (B, L, hidden_size)
        
        # Final projection
        h = self.output_proj(h)
        h = self.dropout(h)
        
        # Apply importance gate (helps distinguish cold vs hot items)
        importance = self.importance_gate(h)
        h = h * importance
        
        # Apply mask
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        
        return self.output_norm(h)


# ======================= NEW: Three-way Gated Fusion =======================

class ThreeWayGatedFusion(nn.Module):
    """
    Enhanced Three-way Gated Fusion with Cross-Branch Attention.
    
    Key improvements (inspired by Related Work document):
    1. Instance-aware gating: Gates adapt per user/item, not just globally
    2. Cross-branch interaction: LLM semantics can guide FFT filtering
    3. Semantic-cooperative fusion: Branches interact before final fusion
    
    Architecture:
        Structural (RG-LRU/Mamba) ─┬─> Cross-Gating ─┬─> Adaptive Fusion
        Frequency (FFT) ───────────┤                 │
        Semantic (LLM) ────────────┴─────────────────┘
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # ===== Cross-Branch Gating: LLM guides other branches =====
        # Semantic-to-Structural gating
        self.semantic_to_struct_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Semantic-to-FFT gating (semantic content can emphasize certain patterns)
        self.semantic_to_fft_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # ===== Instance-aware Gating Network =====
        # Takes all three (cross-gated) inputs to compute final weights
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3)  # 3 gates
        )
        
        # ===== Sequence-level Context for better gating =====
        # Use sequence mean as global context for gating decisions
        self.context_fc = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize for balanced gates
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, struct_out, fft_out, llm_out, residual, mask=None):
        """
        Args:
            struct_out: (B, L, D) - structural/temporal features (RG-LRU/Mamba)
            fft_out: (B, L, D) - frequency features
            llm_out: (B, L, D) - semantic features
            residual: (B, L, D) - original input
            mask: (B, L)
        Returns:
            fused: (B, L, D)
            gates: (B, L, 3) - for analysis
        """
        # ===== Step 1: Cross-Branch Gating =====
        # LLM semantics guide structural branch
        struct_llm_concat = torch.cat([struct_out, llm_out], dim=-1)
        struct_gate = self.semantic_to_struct_gate(struct_llm_concat)
        struct_gated = struct_out * struct_gate  # (B, L, D)
        
        # LLM semantics guide FFT branch (semantic can emphasize relevant frequencies)
        fft_llm_concat = torch.cat([fft_out, llm_out], dim=-1)
        fft_gate = self.semantic_to_fft_gate(fft_llm_concat)
        fft_gated = fft_out * fft_gate  # (B, L, D)
        
        # ===== Step 2: Compute Sequence Context =====
        # Use masked mean as global context
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            seq_sum = (residual * mask_expanded).sum(dim=1)
            seq_count = mask_expanded.sum(dim=1).clamp(min=1)
            seq_context = seq_sum / seq_count  # (B, D)
        else:
            seq_context = residual.mean(dim=1)  # (B, D)
        
        seq_context = self.context_fc(seq_context)  # (B, D)
        
        # ===== Step 3: Instance-aware Gating =====
        # Concatenate all three (cross-gated) branches
        concat = torch.cat([struct_gated, fft_gated, llm_out], dim=-1)  # (B, L, 3D)
        
        # Add sequence context to each position for better global awareness
        concat_with_context = concat + seq_context.unsqueeze(1).expand(-1, concat.size(1), -1).repeat(1, 1, 3)
        
        # Compute gates with softmax (sum to 1)
        gate_logits = self.gate_fc(concat_with_context)  # (B, L, 3)
        gates = F.softmax(gate_logits, dim=-1)  # (B, L, 3)
        
        # ===== Step 4: Weighted Fusion =====
        g_struct = gates[:, :, 0:1]  # (B, L, 1)
        g_fft = gates[:, :, 1:2]
        g_llm = gates[:, :, 2:3]
        
        fused = g_struct * struct_gated + g_fft * fft_gated + g_llm * llm_out
        
        # Output projection + residual
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        out = self.output_norm(residual + fused)
        
        if mask is not None:
            out = out * mask.unsqueeze(-1)
            
        return out, gates


class TwoWayGatedFusion(nn.Module):
    """Enhanced two-way fusion with cross-gating (no LLM)"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # Cross-gating between structural and FFT
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Instance-aware gate
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, struct_out, fft_out, residual, mask=None):
        # Cross-gating: each branch gates the other
        concat_for_gate = torch.cat([struct_out, fft_out], dim=-1)
        cross_gate = self.cross_gate(concat_for_gate)
        
        struct_gated = struct_out * cross_gate
        fft_gated = fft_out * (1 - cross_gate)
        
        # Instance-aware fusion gate
        concat = torch.cat([struct_gated, fft_gated], dim=-1)
        gate = self.gate_fc(concat)
        
        fused = gate * struct_gated + (1 - gate) * fft_gated
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        out = self.output_norm(residual + fused)
        
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


# ======================= Mamba + FFT + LLM Block =======================

class MambaFFTLLMBlock(nn.Module):
    """
    Combined block with three branches: Mamba + FFT + LLM
    
    Architecture:
        Input ─┬─> Mamba ─┐
               │          │
               ├─> FFT ───┼─> Three-way Gated Fusion ─> FFN ─> Output
               │          │
        LLM ───┴──────────┘
    """
    def __init__(self, hidden_size, seq_len=50, llm_dim=768, dropout=0.1, use_llm=True, 
                 bidirectional=True, d_state=16):
        super().__init__()
        self.use_llm = use_llm
        
        # Mamba branch (time domain) - use bidirectional for better context
        if bidirectional:
            self.mamba = BidirectionalMambaWrapper(hidden_size, d_state=d_state, dropout=dropout)
        else:
            self.mamba = MambaBlockWrapper(hidden_size, d_state=d_state, dropout=dropout)
        
        # FFT branch (frequency domain)
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        
        # LLM adapter (semantic)
        if use_llm:
            self.llm_adapter = LLMAdapter(llm_dim, hidden_size, dropout)
            self.fusion = ThreeWayGatedFusion(hidden_size, dropout)
        else:
            self.fusion = TwoWayGatedFusion(hidden_size, dropout)
        
        # Feed-forward after fusion
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        """
        Args:
            hidden: (B, L, D) - ID embeddings
            c: condition (unused)
            mask: (B, L)
            llm_emb: (B, L, llm_dim) - LLM embeddings for items
        """
        # Mamba branch
        mamba_out = self.mamba(hidden, mask)
        mamba_delta = mamba_out - hidden
        
        # FFT branch
        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden
        
        if self.use_llm and llm_emb is not None:
            # LLM branch
            llm_out = self.llm_adapter(llm_emb, mask)
            
            # Three-way fusion
            fused, gates = self.fusion(mamba_delta, fft_delta, llm_out, hidden, mask)
        else:
            # Two-way fusion (fallback)
            fused = self.fusion(mamba_delta, fft_delta, hidden, mask)
        
        # Feed-forward
        fused = self.ff_sublayer(fused, self.feed_forward)
        
        return self.dropout(fused)


class MambaFFTLLM_rep(nn.Module):
    """Mamba + FFT + LLM encoder with Three-way Gated Fusion"""
    def __init__(self, args):
        super(MambaFFTLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.llm_dim = getattr(args, 'llm_dim', 768)
        self.use_llm = getattr(args, 'use_llm', True)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.d_state = getattr(args, 'd_state', 16)
        
        self.blocks = nn.ModuleList([
            MambaFFTLLMBlock(
                self.hidden_size, 
                seq_len=self.max_len, 
                llm_dim=self.llm_dim,
                dropout=self.dropout,
                use_llm=self.use_llm,
                bidirectional=self.bidirectional,
                d_state=self.d_state
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb)
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
            
        return hidden, encode


# ======================= Mamba + FFT (no LLM) for ablation =======================

class MambaFFTGatedBlock(nn.Module):
    """Two-way fusion block (Phase 2.1) with optional bidirectional Mamba"""
    def __init__(self, hidden_size, seq_len=50, dropout=0.1, bidirectional=True, d_state=16):
        super().__init__()
        # Use bidirectional Mamba for better context modeling
        if bidirectional:
            self.mamba = BidirectionalMambaWrapper(hidden_size, d_state=d_state, dropout=dropout)
        else:
            self.mamba = MambaBlockWrapper(hidden_size, d_state=d_state, dropout=dropout)
        
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        self.fusion = TwoWayGatedFusion(hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        mamba_out = self.mamba(hidden, mask)
        fft_out = self.fft(hidden, mask)
        mamba_delta = mamba_out - hidden
        fft_delta = fft_out - hidden
        fused = self.fusion(mamba_delta, fft_delta, hidden, mask)
        fused = self.ff_sublayer(fused, self.feed_forward)
        return self.dropout(fused)


class MambaFFT_rep(nn.Module):
    """Mamba + FFT encoder (no LLM)"""
    def __init__(self, args):
        super(MambaFFT_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.d_state = getattr(args, 'd_state', 16)
        
        self.blocks = nn.ModuleList([
            MambaFFTGatedBlock(
                self.hidden_size, 
                seq_len=self.max_len, 
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                d_state=self.d_state
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode


# ======================= RG-LRU + FFT =======================

class RGLRUFFTGatedBlock(nn.Module):
    """Two-way fusion block with RG-LRU instead of Mamba"""
    def __init__(self, hidden_size, seq_len=50, dropout=0.1, bidirectional=True):
        super().__init__()
        # Use bidirectional RG-LRU for better context modeling
        if bidirectional:
            self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.rglru = RGLRU(hidden_size, dropout=dropout)
        
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        self.fusion = TwoWayGatedFusion(hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        rglru_out = self.rglru(hidden, mask)
        fft_out = self.fft(hidden, mask)
        rglru_delta = rglru_out - hidden
        fft_delta = fft_out - hidden
        fused = self.fusion(rglru_delta, fft_delta, hidden, mask)
        fused = self.ff_sublayer(fused, self.feed_forward)
        return self.dropout(fused)


class RGLRUBlock(nn.Module):
    """Pure RG-LRU block (no FFT/LLM/attention)."""
    def __init__(self, hidden_size, dropout=0.1, bidirectional=True):
        super().__init__()
        if bidirectional:
            self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.rglru = RGLRU(hidden_size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        out = self.rglru(hidden, mask)
        out = self.ff_sublayer(out, self.feed_forward)
        return self.dropout(out)


class RGLRU_rep(nn.Module):
    """RG-LRU-only encoder (sequence baseline)."""
    def __init__(self, args):
        super(RGLRU_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.bidirectional = getattr(args, 'bidirectional', True)

        self.blocks = nn.ModuleList([
            RGLRUBlock(
                self.hidden_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode


class RGLRUFFT_rep(nn.Module):
    """RG-LRU + FFT encoder"""
    def __init__(self, args):
        super(RGLRUFFT_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.bidirectional = getattr(args, 'bidirectional', True)
        
        self.blocks = nn.ModuleList([
            RGLRUFFTGatedBlock(
                self.hidden_size, 
                seq_len=self.max_len, 
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode


# ======================= RG-LRU + FFT + LLM (NEW: Three-way Fusion) =======================

class RGLRUFFTLLMBlock(nn.Module):
    """
    Three-way fusion block: RG-LRU (time) + FFT (freq) + LLM (semantic)
    
    Architecture:
        Input ─┬─> RG-LRU (temporal patterns) ─┐
               │                                │
               ├─> FFT (frequency patterns) ────┼─> Three-way Gated Fusion ─> FFN ─> Output
               │                                │
        LLM ───┴────────────────────────────────┘
    """
    def __init__(self, hidden_size, seq_len=50, llm_dim=3584, dropout=0.1, 
                 use_llm=True, bidirectional=True):
        super().__init__()
        self.use_llm = use_llm
        
        # RG-LRU branch (temporal domain)
        if bidirectional:
            self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.rglru = RGLRU(hidden_size, dropout=dropout)
        
        # FFT branch (frequency domain)
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        
        # LLM adapter (semantic)
        if use_llm:
            self.llm_adapter = LLMAdapter(llm_dim, hidden_size, dropout)
            self.fusion = ThreeWayGatedFusion(hidden_size, dropout)
        else:
            self.fusion = TwoWayGatedFusion(hidden_size, dropout)
        
        # Feed-forward after fusion
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        """
        Args:
            hidden: (B, L, D) - ID embeddings
            c: condition (unused)
            mask: (B, L)
            llm_emb: (B, L, llm_dim) - LLM embeddings for items
        """
        # RG-LRU branch
        rglru_out = self.rglru(hidden, mask)
        rglru_delta = rglru_out - hidden
        
        # FFT branch
        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden
        
        if self.use_llm and llm_emb is not None:
            # LLM branch
            llm_out = self.llm_adapter(llm_emb, mask)
            
            # Three-way fusion
            fused, gates = self.fusion(rglru_delta, fft_delta, llm_out, hidden, mask)
        else:
            # Two-way fusion (fallback)
            fused = self.fusion(rglru_delta, fft_delta, hidden, mask)
        
        # Feed-forward
        fused = self.ff_sublayer(fused, self.feed_forward)
        
        return self.dropout(fused)


class RGLRUFFTLLM_rep(nn.Module):
    """RG-LRU + FFT + LLM encoder with Three-way Gated Fusion"""
    def __init__(self, args):
        super(RGLRUFFTLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.use_llm = getattr(args, 'use_llm', True)
        self.bidirectional = getattr(args, 'bidirectional', True)
        
        self.blocks = nn.ModuleList([
            RGLRUFFTLLMBlock(
                self.hidden_size, 
                seq_len=self.max_len, 
                llm_dim=self.llm_dim,
                dropout=self.dropout,
                use_llm=self.use_llm,
                bidirectional=self.bidirectional
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb)
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
            
        return hidden, encode


# ======================= Mamba + LocalAttn + FFT (New Architecture) =======================

class ThreeWayGatedFusionV2(nn.Module):
    """
    Three-way Gated Fusion for Mamba (global state) + LocalAttn (local) + FFT (freq).
    
    Uses softmax-normalized gates to distribute importance among three branches.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # Gate network: takes all three inputs, outputs 3 gate logits
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # 3 gates
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize for balanced gates
        self._init_weights()
    
    def _init_weights(self):
        for m in self.gate_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, mamba_out, local_out, fft_out, residual, mask=None):
        """
        Args:
            mamba_out: (B, L, D) - global state features from Mamba
            local_out: (B, L, D) - local features from LocalAttention
            fft_out: (B, L, D) - frequency features from FFT
            residual: (B, L, D) - original input
            mask: (B, L)
        Returns:
            fused: (B, L, D)
            gates: (B, L, 3) - for analysis
        """
        # Concatenate all three for gate computation
        concat = torch.cat([mamba_out, local_out, fft_out], dim=-1)  # (B, L, 3D)
        
        # Compute gates with softmax (sum to 1)
        gate_logits = self.gate_fc(concat)  # (B, L, 3)
        gates = F.softmax(gate_logits, dim=-1)  # (B, L, 3)
        
        # Three-way gated fusion
        g_mamba = gates[:, :, 0:1]  # (B, L, 1)
        g_local = gates[:, :, 1:2]
        g_fft = gates[:, :, 2:3]
        
        fused = g_mamba * mamba_out + g_local * local_out + g_fft * fft_out
        
        # Output projection + residual
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        out = self.output_norm(residual + fused)
        
        if mask is not None:
            out = out * mask.unsqueeze(-1)
            
        return out, gates


class FourWayGatedFusion(nn.Module):
    """
    Four-way Gated Fusion - 精简版
    
    只做一个核心改动：独立Sigmoid门控（而非Softmax竞争）
    移除了过度复杂的机制：跨分支交互、温度参数、基础权重
    
    理论：Softmax强制分支竞争(sum=1)，独立Sigmoid允许多分支同时发挥作用
    """
    def __init__(self, hidden_size, dropout=0.1, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax  # 控制是否用Softmax（用于消融实验）
        
        # 门控网络（与原版相同）
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 存储门控值用于分析
        self.last_gates = None
        
        self._init_weights()

    def _init_weights(self):
        for m in self.gate_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_entropy_loss(self):
        """计算门控熵损失（可选，通过entropy_weight=0关闭）"""
        if self.last_gates is None:
            return torch.tensor(0.0)
        
        gate_mean = self.last_gates.mean(dim=(0, 1))
        gate_mean = gate_mean.clamp(min=1e-8)
        entropy = -torch.sum(gate_mean * torch.log(gate_mean))
        max_entropy = math.log(4)
        return -(entropy / max_entropy)

    def forward(self, rglru_out, local_out, fft_out, llm_out, residual, mask=None):
        concat = torch.cat([rglru_out, local_out, fft_out, llm_out], dim=-1)
        gate_logits = self.gate_fc(concat)
        
        if self.use_softmax:
            # 原版：Softmax门控（分支竞争）
            gates = F.softmax(gate_logits, dim=-1)
        else:
            # 改进版：独立Sigmoid门控 + 归一化
            gates = torch.sigmoid(gate_logits)
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 存储用于分析
        self.last_gates = gates.detach()

        g_rglru = gates[:, :, 0:1]
        g_local = gates[:, :, 1:2]
        g_fft = gates[:, :, 2:3]
        g_llm = gates[:, :, 3:4]

        fused = g_rglru * rglru_out + g_local * local_out + g_fft * fft_out + g_llm * llm_out
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        out = self.output_norm(residual + fused)
        
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out, gates


class MambaLocalFFTBlock(nn.Module):
    """
    Three-way fusion block: Mamba (global state) + LocalAttn (local) + FFT (freq)
    
    Architecture:
        Input ─┬─> Mamba (global state tracking) ─┐
               │                                   │
               ├─> LocalAttn (local patterns) ────┼─> Three-way Gated Fusion ─> FFN ─> Output
               │                                   │
               └─> FFT (frequency patterns) ──────┘
    
    Motivation:
        - Mamba: Good at tracking state over sequences, but weak at local patterns in short seqs
        - LocalAttn: Strong local modeling with O(n*w) complexity
        - FFT: Captures periodic/frequency patterns globally
    """
    def __init__(self, hidden_size, seq_len=50, window_size=5, dropout=0.1, d_state=16):
        super().__init__()
        
        # Branch 1: Mamba (global state) - using unidirectional for efficiency
        self.mamba = MambaBlockWrapper(hidden_size, d_state=d_state, dropout=dropout)
        
        # Branch 2: Local Attention (local patterns)
        self.local_attn = LocalAttentionBlock(
            hidden_size, 
            heads=4, 
            window_size=window_size, 
            dropout=dropout
        )
        
        # Branch 3: FFT (frequency patterns)
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        
        # Three-way fusion
        self.fusion = ThreeWayGatedFusionV2(hidden_size, dropout)
        
        # Feed-forward after fusion
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # For gate analysis
        self.last_gates = None

    def forward(self, hidden, c, mask, llm_emb=None):
        """
        Args:
            hidden: (B, L, D) - input embeddings
            c: condition (unused, for API compatibility)
            mask: (B, L) - padding mask
            llm_emb: unused, for API compatibility
        Returns:
            (B, L, D) - output
        """
        # Branch 1: Mamba
        mamba_out = self.mamba(hidden, mask)
        mamba_delta = mamba_out - hidden
        
        # Branch 2: Local Attention
        local_out = self.local_attn(hidden, mask)
        local_delta = local_out - hidden
        
        # Branch 3: FFT
        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden
        
        # Three-way fusion
        fused, gates = self.fusion(mamba_delta, local_delta, fft_delta, hidden, mask)
        self.last_gates = gates  # Store for analysis
        
        # Feed-forward
        fused = self.ff_sublayer(fused, self.feed_forward)
        
        return self.dropout(fused)


class MambaLocalFFT_rep(nn.Module):
    """
    Mamba + LocalAttention + FFT encoder
    
    Story: "Enhancing Mamba for Short-Sequence Recommendation via 
            Local Attention and Frequency-Domain Fusion"
    """
    def __init__(self, args):
        super(MambaLocalFFT_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.window_size = getattr(args, 'window_size', 5)
        self.d_state = getattr(args, 'd_state', 16)
        
        self.blocks = nn.ModuleList([
            MambaLocalFFTBlock(
                self.hidden_size, 
                seq_len=self.max_len, 
                window_size=self.window_size,
                dropout=self.dropout,
                d_state=self.d_state
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode
    
    def get_gate_stats(self):
        """Get gate statistics from all blocks for analysis"""
        stats = []
        for i, block in enumerate(self.blocks):
            if block.last_gates is not None:
                gates = block.last_gates.detach()
                stats.append({
                    'block': i,
                    'mamba_gate': gates[:, :, 0].mean().item(),
                    'local_gate': gates[:, :, 1].mean().item(),
                    'fft_gate': gates[:, :, 2].mean().item(),
                })
        return stats


# ======================= RG-LRU + LocalAttn + FFT (New Architecture) =======================

class RGLRULocalFFTBlock(nn.Module):
    """
    Three-way fusion: RG-LRU (global trend) + LocalAttn (local precision) + FFT (periodicity)
    Maintains O(n * w) for local attention with window 8~16.
    """
    def __init__(self, hidden_size, seq_len=50, window_size=8, dropout=0.1, bidirectional=True):
        super().__init__()
        # Branch 1: RG-LRU (global/state)
        if bidirectional:
            self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.rglru = RGLRU(hidden_size, dropout=dropout)

        # Branch 2: Local Attention (local patterns)
        self.local_attn = LocalAttentionBlock(
            hidden_size,
            heads=4,
            window_size=window_size,
            dropout=dropout
        )

        # Branch 3: FFT (frequency patterns)
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)

        # Three-way fusion
        self.fusion = ThreeWayGatedFusionV2(hidden_size, dropout)

        # Feed-forward after fusion
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        rglru_out = self.rglru(hidden, mask)
        rglru_delta = rglru_out - hidden

        local_out = self.local_attn(hidden, mask)
        local_delta = local_out - hidden

        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden

        fused, _ = self.fusion(rglru_delta, local_delta, fft_delta, hidden, mask)
        fused = self.ff_sublayer(fused, self.feed_forward)
        return self.dropout(fused)


class RGLRULocalFFT_rep(nn.Module):
    """RG-LRU + LocalAttention + FFT encoder"""
    def __init__(self, args):
        super(RGLRULocalFFT_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.window_size = getattr(args, 'window_size', 8)
        self.bidirectional = getattr(args, 'bidirectional', True)

        self.blocks = nn.ModuleList([
            RGLRULocalFFTBlock(
                self.hidden_size,
                seq_len=self.max_len,
                window_size=self.window_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode


# ======================= RG-LRU + LocalAttn + FFT + LLM (New) =======================

class RGLRULocalFFTLLMBlock(nn.Module):
    """
    Four-branch fusion: RG-LRU (global) + LocalAttn (local) + FFT (periodic) + LLM (semantic)
    
    精简版：
    - 默认使用原版RG-LRU（use_enhanced_rglru=False）
    - 添加use_softmax参数控制门控类型（用于消融实验）
    """
    def __init__(self, hidden_size, seq_len=50, llm_dim=3584, window_size=8, dropout=0.1, 
                 bidirectional=True, use_llm=True, use_enhanced_rglru=False, use_softmax=True):
        super().__init__()
        self.use_llm = use_llm
        
        # Branch 1: RG-LRU（默认原版，可选增强版）
        if use_enhanced_rglru:
            if bidirectional:
                self.rglru = BidirectionalEnhancedRGLRU(hidden_size, kernel_size=3, dropout=dropout)
            else:
                self.rglru = EnhancedRGLRU(hidden_size, kernel_size=3, dropout=dropout)
        else:
            # 默认使用原版
            if bidirectional:
                self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
            else:
                self.rglru = RGLRU(hidden_size, dropout=dropout)

        # Branch 2: Local Attention
        self.local_attn = LocalAttentionBlock(
            hidden_size,
            heads=4,
            window_size=window_size,
            dropout=dropout
        )

        # Branch 3: FFT
        self.fft = FFTBlock(hidden_size, seq_len=seq_len, dropout=dropout)
        
        # Branch 4: LLM Adapter
        self.llm_adapter = LLMAdapter(llm_dim, hidden_size, dropout) if use_llm else None

        # Four-way Fusion（支持Softmax/Sigmoid切换）
        self.fusion = FourWayGatedFusion(hidden_size, dropout, use_softmax=use_softmax)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, c, mask, llm_emb=None):
        rglru_out = self.rglru(hidden, mask)
        rglru_delta = rglru_out - hidden

        local_out = self.local_attn(hidden, mask)
        local_delta = local_out - hidden

        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden

        if self.use_llm and llm_emb is not None:
            llm_out = self.llm_adapter(llm_emb, mask)
        else:
            llm_out = torch.zeros_like(hidden)

        fused, _ = self.fusion(rglru_delta, local_delta, fft_delta, llm_out, hidden, mask)
        fused = self.ff_sublayer(fused, self.feed_forward)
        return self.dropout(fused)
    
    def forward_with_branch_outputs(self, hidden, c, mask, llm_emb=None):
        """Forward pass that also returns individual branch outputs for cross-layer connections"""
        rglru_out = self.rglru(hidden, mask)
        rglru_delta = rglru_out - hidden

        local_out = self.local_attn(hidden, mask)
        local_delta = local_out - hidden

        fft_out = self.fft(hidden, mask)
        fft_delta = fft_out - hidden

        if self.use_llm and llm_emb is not None:
            llm_out = self.llm_adapter(llm_emb, mask)
        else:
            llm_out = torch.zeros_like(hidden)

        fused, gates = self.fusion(rglru_delta, local_delta, fft_delta, llm_out, hidden, mask)
        fused = self.ff_sublayer(fused, self.feed_forward)
        
        # Return fused output and individual branch deltas for cross-layer memory
        branch_outputs = {
            'rglru': rglru_delta,
            'local': local_delta,
            'fft': fft_delta,
            'llm': llm_out
        }
        return self.dropout(fused), branch_outputs, gates
    
    def get_entropy_loss(self):
        """获取本block的熵正则损失"""
        return self.fusion.compute_entropy_loss()


class RGLRULocalFFTLLM_rep(nn.Module):
    """RG-LRU + LocalAttention + FFT + LLM encoder"""
    def __init__(self, args):
        super(RGLRULocalFFTLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.window_size = getattr(args, 'window_size', 8)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.use_llm = getattr(args, 'use_llm', True)
        self.use_enhanced_rglru = getattr(args, 'use_enhanced_rglru', False)
        self.use_softmax = getattr(args, 'use_softmax', True)

        self.blocks = nn.ModuleList([
            RGLRULocalFFTLLMBlock(
                self.hidden_size,
                seq_len=self.max_len,
                llm_dim=self.llm_dim,
                window_size=self.window_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                use_llm=self.use_llm,
                use_enhanced_rglru=self.use_enhanced_rglru,
                use_softmax=self.use_softmax
            )
            for _ in range(self.n_blocks)
        ])

    def forward(self, hidden, c, mask, llm_emb=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb)
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode
    
    def get_gate_stats(self):
        """获取门控统计（用于分析各分支使用情况）"""
        stats = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.fusion, 'last_gates') and block.fusion.last_gates is not None:
                gates = block.fusion.last_gates
                stats.append({
                    'block': i,
                    'rglru': gates[:, :, 0].mean().item(),
                    'local': gates[:, :, 1].mean().item(),
                    'fft': gates[:, :, 2].mean().item(),
                    'llm': gates[:, :, 3].mean().item(),
                })
        return stats
    
    def get_entropy_loss(self):
        """获取所有block的平均熵正则损失"""
        total_loss = 0.0
        count = 0
        for block in self.blocks:
            loss = block.get_entropy_loss()
            if isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss
                count += 1
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0)


# ======================= Cross-Layer Enhanced Encoder =======================

class CrossLayerRGLRULocalFFTLLM_rep(nn.Module):
    """
    RG-LRU + LocalAttention + FFT + LLM encoder with Cross-Layer Connections
    
    极简版（减少过拟合风险）：
    1. DenseNet风格累积：各层输出简单累加，无门控
    2. 全局残差：输入直接加到输出，无门控
    3. 最少额外参数：只有两个标量参数
    """
    def __init__(self, args):
        super(CrossLayerRGLRULocalFFTLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.window_size = getattr(args, 'window_size', 8)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.use_llm = getattr(args, 'use_llm', True)
        self.use_enhanced_rglru = getattr(args, 'use_enhanced_rglru', False)
        self.use_softmax = getattr(args, 'use_softmax', True)

        # Main blocks
        self.blocks = nn.ModuleList([
            RGLRULocalFFTLLMBlock(
                self.hidden_size,
                seq_len=self.max_len,
                llm_dim=self.llm_dim,
                window_size=self.window_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                use_llm=self.use_llm,
                use_enhanced_rglru=self.use_enhanced_rglru,
                use_softmax=self.use_softmax
            )
            for _ in range(self.n_blocks)
        ])
        
        # 极简设计：只有两个可学习的标量参数
        self.global_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.accumulated_scale = nn.Parameter(torch.tensor(0.1))
        
        self.output_norm = LayerNorm(self.hidden_size)

    def forward(self, hidden, c, mask, llm_emb=None):
        input_hidden = hidden
        accumulated = torch.zeros_like(hidden)
        
        i = 0
        encode = None
        
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb)
            accumulated = accumulated + hidden / self.n_blocks
            
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
        
        final_hidden = hidden + self.global_residual_scale * input_hidden + self.accumulated_scale * accumulated
        final_hidden = self.output_norm(final_hidden)
        
        if mask is not None:
            final_hidden = final_hidden * mask.unsqueeze(-1)
        
        return final_hidden, encode
    
    def get_gate_stats(self):
        stats = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.fusion, 'last_gates') and block.fusion.last_gates is not None:
                gates = block.fusion.last_gates
                stats.append({
                    'block': i,
                    'rglru': gates[:, :, 0].mean().item(),
                    'local': gates[:, :, 1].mean().item(),
                    'fft': gates[:, :, 2].mean().item(),
                    'llm': gates[:, :, 3].mean().item(),
                })
        return stats
    
    def get_entropy_loss(self):
        total_loss = 0.0
        count = 0
        for block in self.blocks:
            loss = block.get_entropy_loss()
            if isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss
                count += 1
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0)


# ======================= Hierarchical Gated Fusion Encoder =======================

class HierarchicalGatedRGLRULocalFFTLLM_rep(nn.Module):
    """
    分层门控融合（Hierarchical Gated Fusion）
    
    核心思想（参考PDF）：
    - 第一级门控（层内）：每层内部的四分支Softmax/Sigmoid融合（保持现有）
    - 第二级门控（全局）：Meta-Gating决定每个分支跨所有层的整体贡献
    
    架构：
        Layer1: [RG-LRU, Local, FFT, LLM] → 层内融合 → out1
                     ↓ 收集各分支输出
        Layer2: [RG-LRU, Local, FFT, LLM] → 层内融合 → out2
                     ↓ 收集各分支输出
        ...
        LayerN: → outN
        
        Meta-Gating: 
            - 输入：各分支在所有层的累积输出 [rglru_acc, local_acc, fft_acc, llm_acc]
            - 输出：全局分支权重 [w_rglru, w_local, w_fft, w_llm]
        
        最终输出 = outN + Σ(w_branch * branch_accumulated)
    
    优势：
    1. 每个分支的"历史轨迹"都被考虑，不会被单层门控遗忘
    2. Meta-Gating学习每个分支的全局重要性
    3. 即使某分支在层内门控中权重低，仍可通过全局门控贡献
    """
    def __init__(self, args):
        super(HierarchicalGatedRGLRULocalFFTLLM_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.max_len = args.max_len
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.window_size = getattr(args, 'window_size', 8)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.use_llm = getattr(args, 'use_llm', True)
        self.use_enhanced_rglru = getattr(args, 'use_enhanced_rglru', False)
        self.use_softmax = getattr(args, 'use_softmax', True)

        # Main blocks
        self.blocks = nn.ModuleList([
            RGLRULocalFFTLLMBlock(
                self.hidden_size,
                seq_len=self.max_len,
                llm_dim=self.llm_dim,
                window_size=self.window_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                use_llm=self.use_llm,
                use_enhanced_rglru=self.use_enhanced_rglru,
                use_softmax=self.use_softmax
            )
            for _ in range(self.n_blocks)
        ])
        
        # ===== Meta-Gating Network =====
        # 输入：4个分支的累积表示（每个分支取最后位置的mean pooling）
        # 输出：4个分支的全局权重
        self.meta_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 4),
            # 使用Sigmoid让每个分支独立贡献（不竞争）
            nn.Sigmoid()
        )
        
        # 全局分支融合的输出投影
        self.branch_fusion_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 融合权重：控制全局分支融合对最终输出的影响
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        
        self.output_norm = LayerNorm(self.hidden_size)
        
        # 保存meta gate weights用于分析
        self.last_meta_weights = None

    def forward(self, hidden, c, mask, llm_emb=None):
        # 初始化各分支的累积输出
        branch_accumulated = {
            'rglru': torch.zeros_like(hidden),
            'local': torch.zeros_like(hidden),
            'fft': torch.zeros_like(hidden),
            'llm': torch.zeros_like(hidden)
        }
        
        i = 0
        encode = None
        
        for block in self.blocks:
            i += 1
            
            # Forward with branch outputs
            hidden, branch_outputs, _ = block.forward_with_branch_outputs(hidden, c, mask, llm_emb)
            
            # 累积各分支输出（简单平均）
            for key in branch_accumulated:
                branch_accumulated[key] = branch_accumulated[key] + branch_outputs[key] / self.n_blocks
            
            if i == (self.n_blocks - self.last):
                encode = hidden
        
        if encode is None:
            encode = hidden
        
        # ===== Meta-Gating: 计算全局分支权重 =====
        # 对累积的分支输出做mean pooling得到全局表示
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            mask_sum = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            
            rglru_global = (branch_accumulated['rglru'] * mask_expanded).sum(dim=1) / mask_sum.squeeze(1)
            local_global = (branch_accumulated['local'] * mask_expanded).sum(dim=1) / mask_sum.squeeze(1)
            fft_global = (branch_accumulated['fft'] * mask_expanded).sum(dim=1) / mask_sum.squeeze(1)
            llm_global = (branch_accumulated['llm'] * mask_expanded).sum(dim=1) / mask_sum.squeeze(1)
        else:
            rglru_global = branch_accumulated['rglru'].mean(dim=1)
            local_global = branch_accumulated['local'].mean(dim=1)
            fft_global = branch_accumulated['fft'].mean(dim=1)
            llm_global = branch_accumulated['llm'].mean(dim=1)
        
        # 拼接4个分支的全局表示
        meta_input = torch.cat([rglru_global, local_global, fft_global, llm_global], dim=-1)  # (B, 4H)
        
        # 计算全局分支权重
        meta_weights = self.meta_gate(meta_input)  # (B, 4)
        self.last_meta_weights = meta_weights.detach()
        
        # 扩展权重用于加权
        w_rglru = meta_weights[:, 0:1].unsqueeze(1)  # (B, 1, 1)
        w_local = meta_weights[:, 1:2].unsqueeze(1)
        w_fft = meta_weights[:, 2:3].unsqueeze(1)
        w_llm = meta_weights[:, 3:4].unsqueeze(1)
        
        # 全局分支融合
        global_branch_fusion = (
            w_rglru * branch_accumulated['rglru'] +
            w_local * branch_accumulated['local'] +
            w_fft * branch_accumulated['fft'] +
            w_llm * branch_accumulated['llm']
        )
        global_branch_fusion = self.branch_fusion_proj(global_branch_fusion)
        
        # 最终输出：层内融合结果 + 缩放的全局分支融合
        final_hidden = hidden + self.fusion_scale * global_branch_fusion
        final_hidden = self.output_norm(final_hidden)
        
        if mask is not None:
            final_hidden = final_hidden * mask.unsqueeze(-1)
        
        return final_hidden, encode
    
    def get_gate_stats(self):
        """获取门控统计（包括层内门控和Meta门控）"""
        stats = []
        
        # 层内门控统计
        for i, block in enumerate(self.blocks):
            if hasattr(block.fusion, 'last_gates') and block.fusion.last_gates is not None:
                gates = block.fusion.last_gates
                stats.append({
                    'block': i,
                    'rglru': gates[:, :, 0].mean().item(),
                    'local': gates[:, :, 1].mean().item(),
                    'fft': gates[:, :, 2].mean().item(),
                    'llm': gates[:, :, 3].mean().item(),
                })
        
        # Meta门控统计
        if self.last_meta_weights is not None:
            meta_stats = {
                'meta_rglru': self.last_meta_weights[:, 0].mean().item(),
                'meta_local': self.last_meta_weights[:, 1].mean().item(),
                'meta_fft': self.last_meta_weights[:, 2].mean().item(),
                'meta_llm': self.last_meta_weights[:, 3].mean().item(),
            }
            stats.append(meta_stats)
        
        return stats
    
    def get_entropy_loss(self):
        """获取所有block的平均熵正则损失"""
        total_loss = 0.0
        count = 0
        for block in self.blocks:
            loss = block.get_entropy_loss()
            if isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss
                count += 1
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0)


# ======================= Dual-Branch Encoder (RG-LRU + LLM) =======================
class ColdStartAdaptiveGate(nn.Module):
    """根据物品流行度自适应调整门控"""
    def __init__(self, hidden_size):
        super().__init__()
        self.popularity_encoder = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 2),
            nn.Tanh(),
        )

    def forward(self, item_popularity, base_gate_rglru, base_gate_llm):
        pop_input = item_popularity.unsqueeze(-1)
        bias = self.popularity_encoder(pop_input)

        rglru_bias = bias[..., 0:1]
        llm_bias = bias[..., 1:2]

        base_rglru = base_gate_rglru.clamp(0.01, 0.99)
        base_llm = base_gate_llm.clamp(0.01, 0.99)

        rglru_logit = torch.log(base_rglru) - torch.log1p(-base_rglru)
        llm_logit = torch.log(base_llm) - torch.log1p(-base_llm)

        adjusted_rglru = torch.sigmoid(rglru_logit + rglru_bias)
        adjusted_llm = torch.sigmoid(llm_logit + llm_bias - rglru_bias)

        return adjusted_rglru, adjusted_llm


class DualBranchBlock(nn.Module):
    """
    双分支融合：RG-LRU（时序）+ LLM（语义）
    
    简化设计：
    - 移除FFT和LocalAttn（根据gate stats它们贡献很小）
    - 只保留RG-LRU（全局时序）和LLM（语义）
    - 参数量减少约50%，降低过拟合风险
    """
    def __init__(self, hidden_size, llm_dim=3584, dropout=0.1,
                 bidirectional=True, use_cold_start_gate=False):
        super().__init__()
        self.use_cold_start_gate = use_cold_start_gate
        self.cold_start_gate = ColdStartAdaptiveGate(hidden_size) if use_cold_start_gate else None
        self.last_struct_out = None
        self.last_sem_out = None
        
        # Branch 1: RG-LRU（全局时序建模）
        if bidirectional:
            self.rglru = BidirectionalRGLRU(hidden_size, dropout=dropout)
        else:
            self.rglru = RGLRU(hidden_size, dropout=dropout)
        
        # Branch 2: LLM Adapter（语义建模）
        self.llm_adapter = LLMAdapter(llm_dim, hidden_size, dropout)
        
        # 简单的双分支门控（Sigmoid独立，不竞争）
        self.gate_rglru = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.gate_llm = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output layers
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.ff_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # 保存门控权重用于分析
        self.last_gates = None

    def forward(self, hidden, c, mask, llm_emb=None, item_popularity=None):
        residual = hidden
        
        # Branch 1: RG-LRU
        rglru_out = self.rglru(hidden, mask)
        rglru_delta = rglru_out - hidden
        self.last_struct_out = rglru_out
        
        # Branch 2: LLM
        if llm_emb is not None:
            llm_out = self.llm_adapter(llm_emb, mask)
        else:
            llm_out = torch.zeros_like(hidden)
        self.last_sem_out = llm_out if llm_emb is not None else None
        
        # 门控融合（基于当前hidden和分支输出）
        gate_input_rglru = torch.cat([hidden, rglru_delta], dim=-1)
        gate_input_llm = torch.cat([hidden, llm_out], dim=-1)

        g_rglru = self.gate_rglru(gate_input_rglru)
        g_llm = self.gate_llm(gate_input_llm)

        if self.use_cold_start_gate and item_popularity is not None:
            g_rglru, g_llm = self.cold_start_gate(item_popularity, g_rglru, g_llm)

        # 保存门控权重
        self.last_gates = torch.stack([g_rglru.mean(dim=-1), g_llm.mean(dim=-1)], dim=-1)

        # 融合
        fused = g_rglru * rglru_delta + g_llm * llm_out
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        fused = self.output_norm(residual + fused)
        
        if mask is not None:
            fused = fused * mask.unsqueeze(-1)
        
        # FFN
        fused = self.ff_sublayer(fused, self.feed_forward)
        
        return self.dropout(fused)


class DualBranch_rep(nn.Module):
    """
    双分支编码器：RG-LRU + LLM
    
    优势：
    - 参数量约为四分支的50%
    - 专注于有效的两个分支（时序+语义）
    - 减少过拟合风险
    """
    def __init__(self, args):
        super(DualBranch_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.llm_dim = getattr(args, 'llm_dim', 3584)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.use_cold_start_gate = getattr(args, 'use_cold_start_gate', False)
        self.supports_popularity = True

        self.blocks = nn.ModuleList([
            DualBranchBlock(
                self.hidden_size,
                llm_dim=self.llm_dim,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                use_cold_start_gate=self.use_cold_start_gate
            )
            for _ in range(self.n_blocks)
        ])
        self.last_struct_out = None
        self.last_sem_out = None

    def forward(self, hidden, c, mask, llm_emb=None, item_popularity=None):
        i = 0
        encode = None
        for block in self.blocks:
            i += 1
            hidden = block(hidden, c, mask, llm_emb, item_popularity=item_popularity)
            self.last_struct_out = block.last_struct_out
            self.last_sem_out = block.last_sem_out
            if i == (self.n_blocks - self.last):
                encode = hidden
        if encode is None:
            encode = hidden
        return hidden, encode
    
    def get_gate_stats(self):
        """获取门控统计"""
        stats = []
        for i, block in enumerate(self.blocks):
            if block.last_gates is not None:
                gates = block.last_gates
                stats.append({
                    'block': i,
                    'rglru': gates[:, :, 0].mean().item(),
                    'llm': gates[:, :, 1].mean().item(),
                })
        return stats

    def get_last_branch_outputs(self):
        return self.last_struct_out, self.last_sem_out
    
    def get_entropy_loss(self):
        return torch.tensor(0.0)


