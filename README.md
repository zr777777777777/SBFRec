# SBFRec

This is code for SBFRec: Semantic-Behavioral Fusion with Trajectory Smoothing for Generative Sequential Recommendation.

## Project layout

| File | Purpose |
| --- | --- |
| [main.py](main.py) | Training / evaluation entry point — CLI args, data loading, model build, train loop |
| [sbfrec_data.py](sbfrec_data.py) | `Data_Train` / `Data_Val` / `Data_Test` + DA4Rec-style augmentation |
| [sbfrec_encoders.py](sbfrec_encoders.py) | Encoder zoo: Transformer, Mamba(+FFT), RG-LRU(+FFT, +Local), LLM variants |
| [sbfrec_models.py](sbfrec_models.py) | `Att_FM_model`, `create_model_FM` — wraps encoders into the FM recommender |
| [sbfrec_losses.py](sbfrec_losses.py) | `RankingLoss` (ListNet / ListMLE) |
| [sbfrec_training.py](sbfrec_training.py) | `model_train`, `hrs_and_ndcgs_k`, early-stopping logic |
| [sbfrec_utils.py](sbfrec_utils.py) | Seeding, item-popularity, JSON run logger |
| [principled_dual_branch.py](principled_dual_branch.py) | Symmetric Delta + Sparse MoE + Prediction-Guided Router |
| [generate_embeddings.py](generate_embeddings.py) | Offline script to build `llm_embeddings_*.npy` from item text |

## Installation

```bash
pip install -r requirements.txt
```

`mamba-ssm` is commented out in `requirements.txt` because it requires CUDA and a matching toolchain. If you want the real Mamba kernel (encoder types `mamba`, `mamba_fft`, `mamba_fft_llm`, `mamba_local_fft`), uncomment those lines. Otherwise the code automatically falls back to an LSTM stub.

## Data layout

The trainer expects a pickle at `datasets/data/<DATASET>/dataset.pkl` with keys `train`, `val`, `test`, `smap` (leave-one-out split). For LLM encoders, also drop `llm_embeddings_qwen.npy` (or `llm_embeddings.npy`) alongside it.

Full schema, 5-core preprocessing recipe, and embedding generation instructions: [datasets/README.md](datasets/README.md).

## Quick start

See [run_example.sh](run_example.sh) for a copy-paste-ready set of 8 common commands (baseline, LLM embedding generation, full method, ranking loss, fine-tune, eval-only, CPU smoke test).

Train the default RG-LRU + Local-FFT + LLM encoder on Amazon Beauty:

```bash
python main.py --dataset amazon_beauty --encoder_type rglru_local_fft_llm
```

Train without LLM embeddings (no `.npy` file needed):

```bash
python main.py --dataset amazon_beauty --encoder_type rglru_local_fft
```

Use the principled dual-branch encoder with MoE + PGR routing:

```bash
python main.py --dataset amazon_beauty \
  --encoder_type rglru_local_fft_llm \
  --use_principled_dual_branch true \
  --router_mode pgr \
  --num_experts 4 --top_k 2
```

## Evaluation-only mode

Load a checkpoint and run val + test without training:

```bash
python main.py --dataset amazon_beauty \
  --load_checkpoint checkpoints/<run>/best.pt \
  --eval_only true
```

## Important CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--encoder_type` | `rglru_local_fft_llm` | One of the 11 supported encoders (see [main.py](main.py)) |
| `--use_principled_dual_branch` | `false` | Enables Symmetric Delta + Sparse MoE branch |
| `--router_mode` | `learned` | `learned` / `pgr` / `static` |
| `--ranking_loss` | `none` | `none` / `listnet` / `listmle` |
| `--decoder_type` | `ode` | `ode` / `pde_diffusion` / `pde_reaction_diffusion` |
| `--use_aug` | `true` | DA4Rec crop/mask/reorder augmentation |
| `--use_rope` | `true` | RoPE relative position encoding |
| `--metric_ks` | `5 10 20` | HR@k / NDCG@k report cut-offs |
| `--patience` | `2` | Early-stopping patience (set internally to 2) |

Run `python main.py --help` for the full list.

## Outputs

- Per-epoch text logs: `log/<dataset>/<timestamp>_<encoder>.log`
- Structured JSON event log: `log/<dataset>/<dataset>_run_*.json`
- Checkpoints: `checkpoints/<run>/best.pt` (when `--save_model true`)
