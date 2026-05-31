#!/usr/bin/env bash
# Reproduction commands for SBFRec (KDD'26).
# Configuration follows the "Implementation Details" section of the paper:
#   B=4 dual-branch blocks, d=128, dropout=0.1, RoPE on, bidirectional RG-LRU,
#   E=4 sparse-MoE experts with top-2 routing, GTE-Base (d_llm=768) semantics,
#   PAAF router (lambda_pop=1.0), ListNet ranking loss (w=0.1, tau=0.07),
#   dual-branch aux losses (w=0.1, tau_align=0.07, tau_gate=1.0),
#   FM loss weights alpha=0.2, beta=0.4, 30 Euler steps,
#   batch=512, lr=1e-3, early stopping (>=3 metrics, patience=2).
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. Put datasets/data/<DATASET>/dataset.pkl in place (see datasets/README.md).
#   3. Generate GTE-Base embeddings once per dataset (Section "GTE-Base" below).

set -euo pipefail

DEVICE="${DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# Step 0 — Generate GTE-Base semantic embeddings (d_llm = 768)
# Paper uses frozen GTE-Base, L2-normalized; output -> llm_embeddings.npy
# ---------------------------------------------------------------------------
# for ds in amazon_beauty amazon_toys ml-100k yelp; do
#   python generate_embeddings.py --dataset "${ds}" --model gte-base
# done

# ---------------------------------------------------------------------------
# Shared SBFRec main-config (paper Implementation Details).
# Add or override flags after this block as needed.
# ---------------------------------------------------------------------------
SBFREC_MAIN_ARGS=(
  --encoder_type rglru_local_fft_llm
  --use_principled_dual_branch true
  --router_mode learned
  --pop_gate_strength 1.0
  --num_experts 4
  --top_k 2
  --num_blocks 4
  --hidden_size 128
  --dropout 0.1
  --emb_dropout 0.3
  --use_rope true
  --bidirectional true
  --llm_dim 768
  --batch_size 512
  --lr 1e-3
  --sample_N 30
  --ranking_loss listnet
  --ranking_weight 0.1
  --llm_align_weight 0.1
  --llm_align_mode infonce
  --llm_align_temp 0.07
  --gate_supervision_weight 0.1
  --gate_supervision_temp 1.0
  --moe_aux_weight 0.1
  --Loss_Alpha 0.2
  --Loss_Beta 0.4
  --metric_ks 5 10 20
  --min_improved_metrics 3
  --patience 2
  --device "${DEVICE}"
)

# ---------------------------------------------------------------------------
# Step 1 — Reproduce the four main-table results.
# ---------------------------------------------------------------------------
python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}"
# python main.py --dataset amazon_toys  "${SBFREC_MAIN_ARGS[@]}"
# python main.py --dataset ml-100k      "${SBFREC_MAIN_ARGS[@]}"
# python main.py --dataset yelp         "${SBFREC_MAIN_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 2 — Ablations from Table "ablation_beauty_toys".
# Run on Beauty / Toys; flip exactly one knob at a time.
# ---------------------------------------------------------------------------
# w/o LLM  -> drop semantic branch (downgrade encoder to non-LLM variant)
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" \
#   --encoder_type rglru_local_fft \
#   --use_principled_dual_branch false

# w/o MoE  -> single MLP adapter instead of sparse MoE (1 expert, top-1)
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" \
#   --num_experts 1 --top_k 1

# w/o lambda_rank  -> plain CE, no ListNet term
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" \
#   --ranking_loss none --ranking_weight 0.0

# w/o L_dual  -> turn off all dual-branch auxiliary losses
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" \
#   --llm_align_weight 0.0 \
#   --gate_supervision_weight 0.0 \
#   --moe_aux_weight 0.0

# ---------------------------------------------------------------------------
# Step 3 — Fusion-gate comparison (Table "gate_compare").
# Sweep `--router_mode` while keeping everything else fixed.
# ---------------------------------------------------------------------------
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" --router_mode static
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" --router_mode learned --pop_gate_strength 0.0  # branch-cond, no popularity
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" --router_mode pgr                              # prediction-guided router

# ---------------------------------------------------------------------------
# Step 4 — Depth sensitivity (Fig. "depth_sensitivity"): B in {1,2,3,4,5}.
# ---------------------------------------------------------------------------
# for B in 1 2 3 4 5; do
#   python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" --num_blocks "${B}"
# done

# ---------------------------------------------------------------------------
# Step 5 — Evaluation-only mode for a saved checkpoint.
# ---------------------------------------------------------------------------
# python main.py --dataset amazon_beauty "${SBFREC_MAIN_ARGS[@]}" \
#   --load_checkpoint checkpoints/<run>/best.pt \
#   --eval_only true

# ---------------------------------------------------------------------------
# Step 6 — CPU smoke test (verifies the pipeline only; ignores paper hparams).
# ---------------------------------------------------------------------------
# python main.py --dataset amazon_beauty \
#   --device cpu \
#   --encoder_type rglru_local_fft \
#   --batch_size 64 --hidden_size 32 --num_blocks 2 \
#   --epochs 2 --eval_interval 1 --use_aug false
