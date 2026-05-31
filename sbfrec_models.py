import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from sbfrec_encoders import (
    DualBranch_rep,
    LayerNorm,
    MambaFFTLLM_rep,
    MambaFFT_rep,
    MambaLocalFFT_rep,
    Mamba_rep,
    RGLRUFFTLLM_rep,
    RGLRUFFT_rep,
    RGLRULocalFFTLLM_rep,
    RGLRULocalFFT_rep,
    RGLRU_rep,
    RotaryEmbedding,
    SiLU,
    TransformerLLM_rep,
    Transformer_rep,
)

try:
    from principled_dual_branch import PrincipledDualBranch_rep
    PRINCIPLED_DUAL_BRANCH_AVAILABLE = True
except ImportError:
    PRINCIPLED_DUAL_BRANCH_AVAILABLE = False
    print("[WARNING] principled_dual_branch not found. Place principled_dual_branch.py in the same directory.")

# ======================= FM_xstart =======================

class FM_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(FM_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        
        # ===== PDE Decoder 相关模块 =====
        self.decoder_type = getattr(args, 'decoder_type', 'ode')  # ode/pde_diffusion/pde_reaction_diffusion
        
        if self.decoder_type.startswith('pde'):
            print(f"[INFO] Using PDE decoder: {self.decoder_type}")
            
            # ===== 序列维度扩散（在L维度而非H维度） =====
            # 扩散系数（可学习）
            self.diffusion_coef = nn.Parameter(torch.tensor(0.1))
            
            # 序列维度拉普拉斯卷积：让相邻位置的表示互相影响
            # item_rep: (B, L, H) -> 转置为 (B, H, L) -> 卷积 -> 转回 (B, L, H)
            self.seq_lap_conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1,
                groups=hidden_size,  # depthwise：每个通道独立卷积
                bias=False
            )
            # 初始化为拉普拉斯核
            nn.init.constant_(self.seq_lap_conv.weight, 0)
            for i in range(hidden_size):
                self.seq_lap_conv.weight.data[i, 0, 0] = 1.0
                self.seq_lap_conv.weight.data[i, 0, 1] = -2.0
                self.seq_lap_conv.weight.data[i, 0, 2] = 1.0
            
            # ===== 门控融合：学习何时使用PDE修正 =====
            # 输入：当前状态x + ODE预测V + 序列上下文
            self.pde_gate = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()  # 逐维度门控
            )
            
            # ===== 序列级注意力聚合（用于反应项） =====
            self.seq_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=args.dropout,
                batch_first=True
            )
            
            if self.decoder_type == 'pde_reaction_diffusion':
                # 反应项网络（利用序列上下文）
                self.reaction_net = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),  # 输入：当前状态 + 序列上下文
                    nn.GELU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh()
                )
                
                # 边界条件投影：将历史序列信息注入
                self.boundary_proj = nn.Linear(hidden_size, hidden_size)
        
        # Select encoder based on args
        encoder_type = getattr(args, 'encoder_type', 'rglru_local_fft_llm')
        use_dual_branch = getattr(args, 'use_dual_branch', False)
        use_principled_dual_branch = getattr(args, 'use_principled_dual_branch', False)
        
        # Principled Dual-Branch (优先级最高)
        if use_principled_dual_branch:
            if PRINCIPLED_DUAL_BRANCH_AVAILABLE:
                print("[INFO] Using Principled Dual-Branch encoder (Symmetric Delta + Sparse MoE + Prediction-Guided Router)")
                self.att = PrincipledDualBranch_rep(args, args.item_num)
            else:
                raise ImportError("principled_dual_branch.py not found. Please place it in the same directory.")
        elif encoder_type == 'rglru':
            print("[INFO] Using RG-LRU encoder (Sequence-only baseline)")
            self.att = RGLRU_rep(args)
        elif encoder_type == 'rglru_fft_llm':
            print("[INFO] Using RG-LRU+FFT+LLM encoder (Three-way Fusion) - DEFAULT")
            self.att = RGLRUFFTLLM_rep(args)
        elif encoder_type == 'rglru_fft':
            print("[INFO] Using RG-LRU+FFT encoder (Two-way Fusion)")
            self.att = RGLRUFFT_rep(args)
        elif encoder_type == 'rglru_local_fft_llm':
            if use_dual_branch:
                print("[INFO] Using Dual-Branch encoder (RG-LRU + LLM only, ~50% params)")
                self.att = DualBranch_rep(args)
            else:
                print("[INFO] Using RG-LRU+LocalAttn+FFT+LLM encoder (Four-way Fusion, O(n*w))")
                self.att = RGLRULocalFFTLLM_rep(args)
        elif encoder_type == 'rglru_local_fft':
            print("[INFO] Using RG-LRU+LocalAttn+FFT encoder (Three-way Fusion, O(n*w))")
            self.att = RGLRULocalFFT_rep(args)
        elif encoder_type == 'mamba_local_fft':
            print("[INFO] Using Mamba+LocalAttn+FFT encoder (New Three-way Fusion)")
            self.att = MambaLocalFFT_rep(args)
        elif encoder_type == 'mamba_fft_llm':
            print("[INFO] Using Mamba+FFT+LLM encoder (Three-way Fusion)")
            self.att = MambaFFTLLM_rep(args)
        elif encoder_type == 'mamba_fft':
            print("[INFO] Using Mamba+FFT encoder (Two-way Fusion)")
            self.att = MambaFFT_rep(args)
        elif encoder_type == 'mamba':
            print(f"[INFO] Using Mamba encoder")
            self.att = Mamba_rep(args)
        elif encoder_type == 'transformer_llm':
            print("[INFO] Using Transformer+LLM encoder (Self-Attention + LLM Fusion)")
            self.att = TransformerLLM_rep(args)
        elif encoder_type == 'transformer':
            print("[INFO] Using Transformer encoder")
            self.att = Transformer_rep(args)
        else:
            print(f"[INFO] Unknown encoder_type '{encoder_type}', using Transformer")
            self.att = Transformer_rep(args)

        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_fm_rep = LayerNorm(self.hidden_size)

        self.item_num = args.item_num
        self.out_dims = [512, 2048]
        self.act_func = 'tanh'

        out_dims_temp = [self.hidden_size] + self.out_dims + [self.item_num]
        decoder_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            decoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == 'relu':
                decoder_modules.append(nn.ReLU())
            elif self.act_func == 'sigmoid':
                decoder_modules.append(nn.Sigmoid())
            elif self.act_func == 'tanh':
                decoder_modules.append(nn.Tanh())
            elif self.act_func == 'leaky_relu':
                decoder_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        decoder_modules.pop()
        self.decoder = nn.Sequential(*decoder_modules)
        
        self.xavier_normal_initialization(self.decoder)

    def xavier_normal_initialization(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq, llm_emb=None, item_popularity=None,
                force_branch=None, record_stats=True):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        
        lambda_uncertainty = torch.normal(mean=torch.full(rep_item.shape, self.lambda_uncertainty), std=torch.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)

        rep_item_New = rep_item + (lambda_uncertainty * (x_t + emb_t).unsqueeze(1))

        condition_cross = rep_item

        if item_popularity is not None and getattr(self.att, 'supports_popularity', False):
            if getattr(self.att, 'supports_force_branch', False) or getattr(self.att, 'supports_record_stats', False):
                rep_fm, encode = self.att(rep_item_New, condition_cross, mask_seq, llm_emb,
                                          item_popularity=item_popularity,
                                          force_branch=force_branch,
                                          record_stats=record_stats)
            else:
                rep_fm, encode = self.att(rep_item_New, condition_cross, mask_seq, llm_emb,
                                          item_popularity=item_popularity)
        else:
            if getattr(self.att, 'supports_force_branch', False) or getattr(self.att, 'supports_record_stats', False):
                rep_fm, encode = self.att(rep_item_New, condition_cross, mask_seq, llm_emb,
                                          force_branch=force_branch,
                                          record_stats=record_stats)
            else:
                rep_fm, encode = self.att(rep_item_New, condition_cross, mask_seq, llm_emb)

        rep_fm = self.norm_fm_rep(self.dropout(rep_fm))

        out = rep_fm[:, -1, :]

        encoded = encode[:, -1, :]

        decode = self.decoder(encoded)
        
        return out, decode


# ======================= FMRec =======================

class FMRec(nn.Module):
    def __init__(self, args):
        super(FMRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.xstart_model = FM_xstart(self.hidden_size, args)
        self.eps = args.eps
        self.sample_N = args.sample_N
        self.eps_reverse = args.eps_reverse
        self.m_logNorm = args.m_logNorm
        self.s_logNorm = args.s_logNorm
        self.s_modsamp = args.s_modsamp
        self.sampling_method = args.sampling_method

    def euler_sampler(self, item_rep, mask_seq, z0, llm_emb=None, item_popularity=None):
        with torch.no_grad():
            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape
            x = z0.to(device)
            dt = 1./self.sample_N
            eps = self.eps_reverse
            extra = (1 / self.eps_reverse) - 1

            for i in range(self.sample_N):
                num_t = i / self.sample_N * (self.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                x0_hat, _ = self.xstart_model(item_rep, x, t*extra, mask_seq, llm_emb,
                                              item_popularity=item_popularity)
                # Probability flow ODE: v_theta = (x0_hat - x) / (1 - t)
                denom = (1 - t).clamp(min=1e-5).unsqueeze(-1)
                v_theta = (x0_hat - x) / denom
                x = x.detach().clone() + v_theta * dt
            
            nfe = self.sample_N
        return x, nfe
    
    def pde_diffusion_sampler(self, item_rep, mask_seq, z0, llm_emb=None, item_popularity=None):
        """
        PDE扩散增强的Sampler（序列维度扩散 + 门控融合）
        
        改进：
        1. 在序列位置维度(L)而非embedding维度(H)做扩散
        2. 让相邻交互的信息传播到下一个item预测
        3. 门控机制决定何时使用PDE修正
        
        方程：dx/dt = v_theta(x,t) + gate * D * ∇²_seq(x)
        """
        with torch.no_grad():
            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape
            batch_size = shape[0]
            x = z0.to(device)
            dt = 1./self.sample_N
            eps = self.eps_reverse
            extra = (1 / self.eps_reverse) - 1
            
            # 获取PDE模块
            xm = self.xstart_model
            D = torch.abs(xm.diffusion_coef)
            
            # 序列上下文（用于门控和边界条件）
            # 使用attention聚合历史序列信息
            seq_context, _ = xm.seq_attention(
                item_rep[:, -1:, :],  # query: 最后位置
                item_rep,              # key: 整个序列
                item_rep,              # value: 整个序列
            )
            seq_context = seq_context.squeeze(1)  # (B, H)
            
            for i in range(self.sample_N):
                num_t = i / self.sample_N * (self.T - eps) + eps
                t = torch.ones(batch_size, device=device) * num_t
                
                # ===== 概率流 ODE 向量场 =====
                x0_hat, _ = xm(item_rep, x, t*extra, mask_seq, llm_emb,
                               item_popularity=item_popularity)
                denom = (1 - t).clamp(min=1e-5).unsqueeze(-1)
                v_theta = (x0_hat - x) / denom
                
                # ===== 序列维度PDE扩散（semi-implicit for diffusion）=====
                # Step 1: 显式推进向量场
                x_explicit = x + v_theta * dt
                
                # Step 2: 用新状态计算拉普拉斯并做半隐式扩散更新
                seq_with_x = torch.cat([item_rep, x_explicit.unsqueeze(1)], dim=1)  # (B, L+1, H)
                seq_transposed = seq_with_x.transpose(1, 2)  # (B, H, L+1)
                seq_lap = xm.seq_lap_conv(seq_transposed).transpose(1, 2)  # (B, L+1, H)
                diffusion_term = D * seq_lap[:, -1, :]  # (B, H)
                
                # 门控
                gate_input = torch.cat([x_explicit, v_theta, seq_context], dim=-1)
                gate = xm.pde_gate(gate_input)  # (B, H)
                
                # 半隐式缩放：x_new = x_explicit + dt * gate * diffusion / (1 + 2*dt*D)
                inv_denom = 1.0 / (1.0 + 2.0 * dt * D)
                diffusion_update = gate * diffusion_term * inv_denom
                x = x_explicit + diffusion_update * dt
            
            nfe = self.sample_N
        return x, nfe
    
    def pde_reaction_diffusion_sampler(self, item_rep, mask_seq, z0, llm_emb=None, item_popularity=None):
        """
        PDE反应-扩散增强的Sampler（序列维度 + 门控 + 边界条件）
        
        改进：
        1. 序列维度扩散：让相邻交互影响预测
        2. 门控融合：自适应PDE修正强度
        3. 边界条件：历史序列作为PDE边界约束
        4. 反应项：利用序列attention聚合上下文
        
        方程：dx/dt = v_theta(x,t) + gate * [D * ∇²_seq(x) + R(x, context) + boundary]
        """
        with torch.no_grad():
            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape
            batch_size = shape[0]
            x = z0.to(device)
            dt = 1./self.sample_N
            eps = self.eps_reverse
            extra = (1 / self.eps_reverse) - 1
            
            # 获取PDE模块
            xm = self.xstart_model
            D = torch.abs(xm.diffusion_coef)
            
            # 序列上下文（通过attention聚合历史信息）
            seq_context, attn_weights = xm.seq_attention(
                item_rep[:, -1:, :],  # query: 最后位置
                item_rep,              # key: 整个序列  
                item_rep,              # value: 整个序列
            )
            seq_context = seq_context.squeeze(1)  # (B, H)
            
            # 边界条件：来自最近几个交互的加权信息
            # 模拟PDE的Dirichlet边界条件
            boundary = xm.boundary_proj(item_rep[:, -1, :])  # 最后交互的投影
            
            for i in range(self.sample_N):
                num_t = i / self.sample_N * (self.T - eps) + eps
                t = torch.ones(batch_size, device=device) * num_t
                
                # ===== 概率流 ODE 向量场 =====
                x0_hat, _ = xm(item_rep, x, t*extra, mask_seq, llm_emb,
                               item_popularity=item_popularity)
                denom = (1 - t).clamp(min=1e-5).unsqueeze(-1)
                v_theta = (x0_hat - x) / denom
                
                # ===== 序列维度PDE扩散（semi-implicit for diffusion）=====
                # Step 1: 显式推进向量场
                x_explicit = x + v_theta * dt
                
                # Step 2: 用新状态计算拉普拉斯并做半隐式扩散更新
                seq_with_x = torch.cat([item_rep, x_explicit.unsqueeze(1)], dim=1)  # (B, L+1, H)
                seq_transposed = seq_with_x.transpose(1, 2)
                seq_lap = xm.seq_lap_conv(seq_transposed).transpose(1, 2)
                diffusion_term = D * seq_lap[:, -1, :]
                
                # ===== 反应项（利用序列上下文） =====
                reaction_input = torch.cat([x, seq_context], dim=-1)
                reaction_term = xm.reaction_net(reaction_input)
                
                # ===== 边界条件项 =====
                # 边界约束随时间衰减（早期强，让x接近历史；后期弱，允许探索）
                boundary_weight = 0.2 * (1 - i / self.sample_N)
                boundary_term = boundary_weight * (boundary - x)
                
                # ===== 门控融合 =====
                gate_input = torch.cat([x_explicit, v_theta, seq_context], dim=-1)
                gate = xm.pde_gate(gate_input)
                
                # ===== 反应项权重（随时间增加） =====
                reaction_weight = 0.1 + 0.1 * (i / self.sample_N)
                
                # ===== 组合更新 =====
                # PDE修正 = 扩散 + 反应 + 边界
                pde_correction = diffusion_term + reaction_weight * reaction_term + boundary_term
                
                # 最终更新：ODE + 门控后的PDE修正
                # 半隐式扩散缩放：x_new = x_explicit + dt * gate * pde_correction / (1 + 2*dt*D)
                inv_denom = 1.0 / (1.0 + 2.0 * dt * D)
                pde_update = gate * pde_correction * inv_denom
                x = x_explicit + pde_update * dt
                
                # 稳定化
                x = torch.clamp(x, -10, 10)
            
            nfe = self.sample_N
        return x, nfe

    def reverse_p_sample_rf(self, item_rep, z0, mask_seq, llm_emb=None, item_popularity=None):
        """根据decoder_type选择不同的sampler"""
        decoder_type = self.xstart_model.decoder_type
        if decoder_type == 'pde_diffusion':
            X_pred, nfe = self.pde_diffusion_sampler(item_rep, mask_seq, z0, llm_emb,
                                                     item_popularity=item_popularity)
        elif decoder_type == 'pde_reaction_diffusion':
            X_pred, nfe = self.pde_reaction_diffusion_sampler(item_rep, mask_seq, z0, llm_emb,
                                                              item_popularity=item_popularity)
        else:  # 默认ODE
            X_pred, nfe = self.euler_sampler(item_rep, mask_seq, z0, llm_emb,
                                             item_popularity=item_popularity)
        return X_pred 

    @property
    def T(self):
        return 1.

    def q_sample_rf(self, x_start, t, z0, mask=None):
        assert z0.shape == x_start.shape
        a_t = t
        b_t = 1 - t
        x_t = a_t * x_start + b_t * z0
        if mask is None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask==0, x_start, x_t)

    def Mode_sample_timestep(self, batch_size, s, device):
        u = torch.rand(batch_size, device=device)
        correction_term = s * (torch.cos((torch.pi / 2) * u)**2 - 1 + u)
        t = 1 - u - correction_term
        return t

    def forward(self, item_rep, item_tag, mask_seq, llm_emb=None, item_popularity=None,
                force_branch=None, record_stats=True, fixed_t_rf=None, fixed_z0=None):        
        noise = torch.randn_like(item_tag)
        z0 = noise
        batch_size = item_tag.shape[0]
        
        use_fixed = fixed_t_rf is not None and fixed_z0 is not None
        if use_fixed:
            t_rf = fixed_t_rf.view(-1).to(item_tag.device)
            z0 = fixed_z0.to(item_tag.device)
        else:
            if self.sampling_method == 'mode':
                t_rf = self.Mode_sample_timestep(batch_size, self.s_modsamp, item_tag.device) * (self.T - self.eps) + self.eps
            elif self.sampling_method == 'uniform':
                t_rf = torch.rand(item_tag.shape[0], device=item_tag.device) * (self.T - self.eps) + self.eps
            else:
                t_rf = self.Mode_sample_timestep(batch_size, self.s_modsamp, item_tag.device) * (self.T - self.eps) + self.eps

        t_rf_expand = t_rf.view(-1, 1).repeat(1, item_tag.shape[1])
        x_t = self.q_sample_rf(item_tag, t_rf_expand, z0=z0)
        extra = (1 / self.eps) - 1
        
        
        x_0, decode_out = self.xstart_model(item_rep, x_t, t_rf*extra, mask_seq, llm_emb,
                                            item_popularity=item_popularity,
                                            force_branch=force_branch,
                                            record_stats=record_stats)
        
        return x_0, decode_out, t_rf_expand, t_rf, z0
    
# ======================= Joint Model =======================

class Att_FM_model(nn.Module):
    def __init__(self, fm, args, item_popularity=None):
        super(Att_FM_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.use_rope = getattr(args, 'use_rope', False) and (args.hidden_size % 2 == 0)
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(args.hidden_size)
        else:
            self.rotary_emb = None
            if getattr(args, 'use_rope', False) and args.hidden_size % 2 != 0:
                print("[WARNING] RoPE disabled because hidden_size is not even.")
        self.fm = fm
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.mask_ratio = args.mask_ratio
        
        # LLM embeddings (loaded from file)
        self.use_llm = getattr(args, 'use_llm', False)
        if self.use_llm:
            # Support custom embedding file name
            llm_emb_file = getattr(args, 'llm_emb_file', None)
            if llm_emb_file:
                llm_emb_path = f'datasets/data/{args.dataset}/{llm_emb_file}'
            else:
                # Try Qwen embeddings first, fall back to default
                qwen_path = f'datasets/data/{args.dataset}/llm_embeddings_qwen.npy'
                default_path = f'datasets/data/{args.dataset}/llm_embeddings.npy'
                llm_emb_path = qwen_path if os.path.exists(qwen_path) else default_path
            
            # File existence already checked in main(), so this should always succeed
            llm_emb = np.load(llm_emb_path)
            
            # Auto-detect and validate LLM dimension
            actual_llm_dim = llm_emb.shape[1]
            expected_llm_dim = getattr(args, 'llm_dim', 3584)
            if actual_llm_dim != expected_llm_dim:
                print(f"[WARNING] LLM embedding dimension mismatch!")
                print(f"[WARNING] Expected: {expected_llm_dim}, Actual: {actual_llm_dim}")
                print(f"[INFO] Updating args.llm_dim to {actual_llm_dim}")
                args.llm_dim = actual_llm_dim
            
            # Register as buffer (not trainable); replace if already exists (e.g., hot reload)
            if 'llm_emb_table' in self._buffers:
                self._buffers['llm_emb_table'] = torch.from_numpy(llm_emb).float()
            else:
                self.register_buffer('llm_emb_table', torch.from_numpy(llm_emb).float())
            print(f"[INFO] Loaded LLM embeddings from: {llm_emb_path}")
            print(f"[INFO] LLM embeddings shape: {llm_emb.shape}")

            # LLM alignment (optional)
            self.llm_align_mode = getattr(args, 'llm_align_mode', 'infonce')
            self.llm_align_temp = float(getattr(args, 'llm_align_temp', 0.07))
            self.llm_align_weight = float(getattr(args, 'llm_align_weight', 0.0))
            self.llm_align_proj = nn.Sequential(
                nn.Linear(args.llm_dim, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
            )
        else:
            # Register empty buffer to avoid attribute errors
            if 'llm_emb_table' not in self._buffers:
                self.register_buffer('llm_emb_table', None)
            self.llm_align_mode = 'infonce'
            self.llm_align_temp = 0.07
            self.llm_align_weight = 0.0
            self.llm_align_proj = None

        # ===== 冷启动门控（可选）=====
        self.use_cold_start_gate = getattr(args, 'use_cold_start_gate', False)
        if self.use_cold_start_gate and not self.use_llm:
            print("[WARNING] Cold-start gate requires LLM embeddings; disabling.")
            self.use_cold_start_gate = False

        if self.use_cold_start_gate:
            if item_popularity is None:
                print("[WARNING] Cold-start gate enabled but item popularity missing; disabling.")
                self.use_cold_start_gate = False
            else:
                pop_tensor = torch.as_tensor(item_popularity, dtype=torch.float32)
                if 'item_popularity_table' in self._buffers:
                    self._buffers['item_popularity_table'] = pop_tensor
                else:
                    self.register_buffer('item_popularity_table', pop_tensor)

    def get_llm_emb(self, item_ids):
        """Lookup LLM embeddings for item IDs"""
        if self.llm_emb_table is None:
            return None
        # item_ids: (B, L), llm_emb_table: (num_items, llm_dim)
        return self.llm_emb_table[item_ids]  # (B, L, llm_dim)

    def get_item_popularity(self, item_ids):
        """Lookup normalized item popularity for IDs"""
        if not self.use_cold_start_gate:
            return None
        if not hasattr(self, 'item_popularity_table') or self.item_popularity_table is None:
            return None
        return self.item_popularity_table[item_ids]

    def get_llm_align_loss(self, target_ids):
        """Align LLM embeddings to item embedding space (optional)."""
        if (not self.use_llm or self.llm_emb_table is None
                or self.llm_align_weight <= 0 or self.llm_align_proj is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        llm_vec = self.llm_emb_table[target_ids]  # (B, llm_dim)
        llm_proj = self.llm_align_proj(llm_vec)   # (B, hidden)
        item_vec = self.item_embeddings(target_ids)  # (B, hidden)
        llm_proj = F.normalize(llm_proj, dim=-1)
        item_vec = F.normalize(item_vec, dim=-1)
        if self.llm_align_mode == 'cos':
            loss = 1.0 - (llm_proj * item_vec).sum(dim=-1).mean()
        else:
            logits = torch.matmul(llm_proj, item_vec.t()) / max(self.llm_align_temp, 1e-6)
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
        return loss

    def fm_pre(self, item_rep, tag_emb, mask_seq, llm_emb=None, item_popularity=None,
               force_branch=None, record_stats=True, fixed_t_rf=None, fixed_z0=None):
        x_0, decode_out, t_rf_expand, t, z0 = self.fm(
            item_rep, tag_emb, mask_seq, llm_emb, item_popularity=item_popularity,
            force_branch=force_branch, record_stats=record_stats,
            fixed_t_rf=fixed_t_rf, fixed_z0=fixed_z0
        )
        return x_0, decode_out, t_rf_expand, t, z0 

    def reverse(self, item_rep, z0, mask_seq, llm_emb=None, item_popularity=None):
        return self.fm.reverse_p_sample_rf(item_rep, z0, mask_seq, llm_emb,
                                           item_popularity=item_popularity)

    def loss_fm_ce(self, rep_fm, labels):
        scores = torch.matmul(rep_fm, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))
    
    def loss_fm_ranking(self, rep_fm, labels, ranking_loss_fn):
        """使用Ranking Loss（ListNet/ListMLE）"""
        scores = torch.matmul(rep_fm, self.item_embeddings.weight.t())
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        total_loss, ce_loss, ranking_loss = ranking_loss_fn(scores, labels)
        return total_loss, ce_loss, ranking_loss

    def fm_rep_pre(self, rep_fm):
        return torch.matmul(rep_fm, self.item_embeddings.weight.t())
    
    def loss_FM_mse(self, rep_fm, target_embeddings):
        return self.loss_mse(rep_fm, target_embeddings)

    def switch_Matrix(self, sequence, device):
        batch_size, seq_len = sequence.size()
        sparse_matrix = torch.zeros(batch_size, self.item_num, device=device)
        for i in range(batch_size):
            row_data = sequence[i]  
            non_zero_indices = row_data[row_data != 0]
            sparse_matrix[i, non_zero_indices] = 1
        return sparse_matrix

    def balanced_mse_loss(self, target, output, mask_ratio=1.0):
        num_ones = torch.sum(target == 1).item()
        num_zeros = torch.sum(target == 0).item()
        num_selected_zeros = int(min(num_zeros, num_ones * mask_ratio))
        zero_positions = (target == 0).nonzero(as_tuple=True)
        one_positions = (target == 1).nonzero(as_tuple=True)
        zero_rows, zero_cols = zero_positions
        selected_zero_indices = torch.randint(0, num_zeros, (num_selected_zeros,))
        selected_zero_positions = (zero_rows[selected_zero_indices], zero_cols[selected_zero_indices])
        mask = torch.zeros_like(target)
        mask[selected_zero_positions] = 1
        mask[one_positions] = 1
        masked_target = target * mask
        masked_output = output * mask
        return self.loss_mse(masked_output, masked_target)

    def forward(self, sequence, tag, forward_mse_time=0, train_flag=True,
                force_branch=None, record_stats=True, fixed_t_rf=None, fixed_z0=None): 
        batch_size, seq_len = sequence.shape
        
        # Item embeddings
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        
        position_ids = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        if self.use_rope and self.rotary_emb is not None:
            # RoPE for relative positions; keeps O(n) and benefits RG-LRU temporal modeling
            item_embeddings = self.rotary_emb.apply_rotary(item_embeddings)
        else:
            item_embeddings = item_embeddings + position_embeddings
        item_embeddings = self.LayerNorm(item_embeddings)
        mask_seq = (sequence > 0).float()
        
        # Get LLM embeddings
        llm_emb = self.get_llm_emb(sequence) if self.use_llm else None
        item_popularity = self.get_item_popularity(sequence)
        
        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))
            rep_fm, decode_out, t_rf_expand, t_rf, z0 = self.fm_pre(
                item_embeddings, tag_emb, mask_seq, llm_emb, item_popularity=item_popularity,
                force_branch=force_branch, record_stats=record_stats,
                fixed_t_rf=fixed_t_rf, fixed_z0=fixed_z0
            )
            seq_Matrix = self.switch_Matrix(sequence, device=sequence.device)
            loss_mse = self.balanced_mse_loss(seq_Matrix, decode_out, self.mask_ratio)
            scores = loss_mse
            loss_FM_mse = self.loss_FM_mse(rep_fm, tag_emb)
            
            item_rep_dis = loss_FM_mse
        else:
            z0 = torch.randn_like(item_embeddings[:,-1,:])
            rep_fm = self.reverse(item_embeddings, z0, mask_seq, llm_emb,
                                  item_popularity=item_popularity)
            t_rf_expand, t_rf, item_rep_dis = None, None, None
            scores = None

        return scores, rep_fm, t_rf_expand, t_rf, item_rep_dis, z0
    
    def get_entropy_loss(self):
        """获取熵正则损失（鼓励均衡使用各分支）"""
        # 检查encoder是否支持熵损失
        if hasattr(self.fm.xstart_model, 'att'):
            att = self.fm.xstart_model.att
            if hasattr(att, 'get_entropy_loss'):
                return att.get_entropy_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_aux_loss(self):
        """获取MoE负载均衡辅助损失"""
        if hasattr(self.fm.xstart_model, 'att'):
            att = self.fm.xstart_model.att
            if hasattr(att, 'get_aux_loss'):
                return att.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_gate_stats(self):
        """获取门控统计信息（用于分析）"""
        if hasattr(self.fm.xstart_model, 'att'):
            att = self.fm.xstart_model.att
            if hasattr(att, 'get_gate_stats'):
                return att.get_gate_stats()
        return []

    def get_gate_weights(self):
        """获取最后一层门控权重（用于门控监督）"""
        if hasattr(self.fm.xstart_model, 'att'):
            att = self.fm.xstart_model.att
            if hasattr(att, 'get_last_gate_weights'):
                return att.get_last_gate_weights()
        return None


def create_model_FM(args):
    return FMRec(args)

