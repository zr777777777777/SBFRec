import argparse
import logging
import os
import pickle
import time

import numpy as np
import torch

from sbfrec_data import Data_Test, Data_Train, Data_Val
from sbfrec_models import Att_FM_model, create_model_FM
from sbfrec_training import hrs_and_ndcgs_k, model_train
from sbfrec_utils import (
    RunJsonLogger,
    _next_run_json_path,
    compute_item_popularity,
    fix_random_seed_as,
)


def main(args, logger, run_logger=None):
    fix_random_seed_as(args.random_seed)

    path_data = 'datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    args.item_num = len(data_raw['smap']) + 1
    args.user_num = len(data_raw['train'].items()) + 1

    item_popularity = None
    use_cold_start_gate = getattr(args, 'use_cold_start_gate', False)
    if use_cold_start_gate:
        item_popularity = compute_item_popularity(
            data_raw['train'],
            args.item_num,
            smooth=1.0,
            norm='max'
        )
        print("[INFO] Computed item popularity for cold-start gate (norm=max, smooth=1.0)")

    # Set use_llm based on encoder_type AND check file existence BEFORE model creation
    # Support both RGLRU and Mamba variants with LLM
    llm_encoder_types = ['rglru_fft_llm', 'mamba_fft_llm', 'rglru_local_fft_llm', 'transformer_llm']
    args.use_llm = (args.encoder_type in llm_encoder_types)

    if args.use_llm:
        # Check multiple possible embedding file paths
        llm_emb_file = getattr(args, 'llm_emb_file', None)
        if llm_emb_file:
            possible_paths = [f'datasets/data/{args.dataset}/{llm_emb_file}']
        else:
            possible_paths = [
                f'datasets/data/{args.dataset}/llm_embeddings_qwen.npy',  # Qwen embeddings
                f'datasets/data/{args.dataset}/llm_embeddings.npy',       # Default
            ]

        llm_emb_found = False
        llm_emb_path = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[INFO] Found LLM embeddings at {path}")
                llm_emb_found = True
                llm_emb_path = path
                break

        if not llm_emb_found:
            print(f"[WARNING] LLM embeddings not found!")
            print(f"[WARNING] Searched paths: {possible_paths}")
            print("[WARNING] Please generate embeddings using one of these methods:")
            print(f"  1. python generate_qwen_embeddings.py --dataset {args.dataset}  (recommended)")
            print("[WARNING] Falling back to rglru_fft (without LLM)")
            args.use_llm = False
            # Downgrade to corresponding no-LLM version
            if args.encoder_type == 'rglru_fft_llm':
                args.encoder_type = 'rglru_fft'
            elif args.encoder_type == 'mamba_fft_llm':
                args.encoder_type = 'mamba_fft'
            elif args.encoder_type == 'rglru_local_fft_llm':
                args.encoder_type = 'rglru_local_fft'
            elif args.encoder_type == 'transformer_llm':
                args.encoder_type = 'transformer'
        else:
            # Preload to detect dimension BEFORE model creation to avoid shape mismatch
            llm_emb_preview = np.load(llm_emb_path, mmap_mode='r')
            actual_llm_dim = llm_emb_preview.shape[1]
            if actual_llm_dim != getattr(args, 'llm_dim', actual_llm_dim):
                print(f"[INFO] Adjusting llm_dim to match embedding file: {actual_llm_dim}")
                args.llm_dim = int(actual_llm_dim)

    print(f"Dataset: {args.dataset}")
    print(f"Items: {args.item_num}, Users: {args.user_num}")
    print(f"Encoder type: {args.encoder_type}")
    print(f"Use LLM: {args.use_llm}")
    if run_logger:
        run_logger.log("dataset_loaded", {
            "dataset": args.dataset,
            "items": args.item_num,
            "users": args.user_num,
            "encoder_type": args.encoder_type,
            "use_llm": args.use_llm,
        })

    if getattr(args, 'use_cold_start_gate', False):
        print("Improvements: ColdStartGate")
    else:
        print("Improvements: None (baseline)")

    # Principled 双分支：默认启用门控/MoE 平衡损失
    if getattr(args, 'use_principled_dual_branch', False):
        if getattr(args, 'entropy_weight', 0.0) == 0.0:
            args.entropy_weight = 0.1
            print("[INFO] Principled dual-branch: enabling balance loss (entropy_weight=0.1)")

    # 严格早停策略：固定阈值与回退机制
    args.min_improved_metrics = 3
    args.patience = 2
    print("[INFO] Early stopping: require >=3 metrics improve; patience=2; rollback on fail")

    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    FM_rec = create_model_FM(args)
    rec_fm_joint_model = Att_FM_model(FM_rec, args, item_popularity=item_popularity)

    total_params = sum(p.numel() for p in rec_fm_joint_model.parameters())
    print(f"Total parameters: {total_params:,}")
    if run_logger:
        run_logger.log("model_initialized", {
            "total_params": int(total_params),
        })

    # ===== 加载预训练模型（NEW）=====
    load_checkpoint = getattr(args, 'load_checkpoint', None)
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"[INFO] Loading checkpoint: {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint, map_location=args.device)

        # 支持两种格式：纯state_dict或完整checkpoint
        if 'model_state_dict' in checkpoint:
            rec_fm_joint_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded model from checkpoint")
            if 'test_metrics' in checkpoint:
                print(f"[INFO] Previous test metrics: {checkpoint['test_metrics']}")
            if run_logger:
                run_logger.log("checkpoint_loaded", {
                    "path": load_checkpoint,
                    "test_metrics": checkpoint.get('test_metrics'),
                })
        else:
            # 兼容旧格式（纯state_dict）
            rec_fm_joint_model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded model (legacy format)")
            if run_logger:
                run_logger.log("checkpoint_loaded", {
                    "path": load_checkpoint,
                })

    # ===== 仅评估模式（NEW）=====
    eval_only = getattr(args, 'eval_only', False)
    if eval_only:
        print("[INFO] Evaluation only mode")
        device = args.device
        rec_fm_joint_model = rec_fm_joint_model.to(device)
        rec_fm_joint_model.eval()
        if run_logger:
            run_logger.log("eval_only_start")

        metric_ks = args.metric_ks

        with torch.no_grad():
            # Validation
            val_metrics = {f'HR@{k}': [] for k in metric_ks}
            val_metrics.update({f'NDCG@{k}': [] for k in metric_ks})

            for val_batch in val_data_loader:
                val_batch = [x.to(device) for x in val_batch]
                _, rep_fm, _, _, _, _ = rec_fm_joint_model(val_batch[0], val_batch[1], train_flag=False)
                scores = rec_fm_joint_model.fm_rep_pre(rep_fm)
                metrics = hrs_and_ndcgs_k(scores, val_batch[1], metric_ks)
                for k, v in metrics.items():
                    val_metrics[k].append(v)

            print("\nValidation Results:")
            for k in val_metrics:
                print(f"  {k}: {np.mean(val_metrics[k])*100:.4f}")

            # Test
            test_metrics = {f'HR@{k}': [] for k in metric_ks}
            test_metrics.update({f'NDCG@{k}': [] for k in metric_ks})

            for test_batch in test_data_loader:
                test_batch = [x.to(device) for x in test_batch]
                _, rep_fm, _, _, _, _ = rec_fm_joint_model(test_batch[0], test_batch[1], train_flag=False)
                scores = rec_fm_joint_model.fm_rep_pre(rep_fm)
                metrics = hrs_and_ndcgs_k(scores, test_batch[1], metric_ks)
                for k, v in metrics.items():
                    test_metrics[k].append(v)

            print("\nTest Results:")
            test_results = {}
            for k in test_metrics:
                val = np.mean(test_metrics[k]) * 100
                test_results[k] = round(val, 4)
                print(f"  {k}: {val:.4f}")

        if run_logger:
            val_metrics_mean = {k: round(float(np.mean(v)) * 100, 4) for k, v in val_metrics.items()}
            run_logger.log("eval_only_results", {
                "val_metrics": val_metrics_mean,
                "test_metrics": test_results,
            })
            run_logger.log("val_metrics", {
                "epoch": "eval_only",
                "metrics": val_metrics_mean,
            })
            run_logger.log("test_metrics", {
                "epoch": "eval_only",
                "metrics": test_results,
            })

        return rec_fm_joint_model, test_results

    best_model, test_results = model_train(
        tra_data_loader,
        val_data_loader,
        test_data_loader,
        rec_fm_joint_model,
        args,
        logger,
        run_logger=run_logger,
    )

    return best_model, test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name')
    parser.add_argument('--log_file', default='log/', help='log dir path')
    parser.add_argument('--random_seed', type=int, default=1997)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_dropout', type=float, default=0.3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--min_improved_metrics', type=int, default=3,
                        help='Minimum number of improved metrics to update model (default: 3)')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--lambda_uncertainty', type=float, default=0.001)
    parser.add_argument('--eval_interval', type=int, default=20)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--sample_N', type=int, default=30)
    parser.add_argument('--eps_reverse', type=float, default=0.001)
    parser.add_argument('--m_logNorm', type=float, default=1.0)
    parser.add_argument('--s_logNorm', type=float, default=0.6)
    parser.add_argument('--s_modsamp', type=float, default=1.0)
    parser.add_argument('--last', type=int, default=2)
    parser.add_argument('--mask_ratio', type=float, default=1.0)
    parser.add_argument('--sampling_method', type=str, default='mode')
    parser.add_argument('--Loss_Alpha', type=float, default=0.2)
    parser.add_argument('--Loss_Beta', type=float, default=0.4)
    # Encoder type (RGLRU-based encoders are default, no Mamba dependency)
    parser.add_argument('--encoder_type', type=str, default='rglru_local_fft_llm',
                        choices=['transformer', 'transformer_llm',
                                 'mamba', 'mamba_fft', 'mamba_fft_llm', 'mamba_local_fft',
                                 'rglru', 'rglru_fft', 'rglru_fft_llm', 'rglru_local_fft', 'rglru_local_fft_llm'],
                        help='Encoder type: transformer_llm | mamba_fft_llm | rglru_local_fft_llm (default)')

    # Mamba settings
    parser.add_argument('--bidirectional', type=lambda x: x.lower() == 'true', default=True,
                        help='Use bidirectional Mamba (default: True)')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')

    # Local Attention settings
    parser.add_argument('--window_size', type=int, default=8, help='Local attention window size (suggested 8-16)')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention heads (default: 4)')

    # LLM settings (supports Qwen2.5-7B-Instruct and other models)
    parser.add_argument('--llm_dim', type=int, default=3584,
                        help='LLM embedding dimension (3584 for Qwen2.5-7B, 2048 for Qwen2.5-3B, 384 for MiniLM)')
    parser.add_argument('--llm_emb_file', type=str, default=None,
                        help='Custom LLM embedding file name (default: llm_embeddings.npy)')

    # DA4Rec-style data augmentation
    parser.add_argument('--use_aug', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable DA4Rec-style crop/mask/reorder augmentation for training')
    parser.add_argument('--aug_crop_prob', type=float, default=0.2, help='Probability of random crop')
    parser.add_argument('--aug_mask_prob', type=float, default=0.15, help='Probability of masking a token')
    parser.add_argument('--aug_reorder_prob', type=float, default=0.2, help='Probability of local reorder')
    parser.add_argument('--aug_reorder_max_span', type=int, default=4, help='Max span size for reorder window')

    # Positional encoding
    parser.add_argument('--use_rope', type=lambda x: x.lower() == 'true', default=True,
                        help='Use RoPE relative position encoding instead of purely learned absolute embeddings')

    # 熵正则损失权重（设为0关闭）
    parser.add_argument('--entropy_weight', type=float, default=0.0,
                        help='Weight for entropy regularization loss (0 to disable)')
    parser.add_argument('--moe_aux_weight', type=float, default=1.0,
                        help='Weight for MoE load-balance auxiliary loss (default: 1.0)')
    parser.add_argument('--llm_align_weight', type=float, default=0.1,
                        help='Weight for LLM alignment loss (default: 0.1)')
    parser.add_argument('--llm_align_mode', type=str, default='infonce',
                        choices=['infonce', 'cos'],
                        help='LLM alignment loss type (default: infonce)')
    parser.add_argument('--llm_align_temp', type=float, default=0.07,
                        help='Temperature for LLM alignment (default: 0.07)')

    # 双分支编码器（RG-LRU + LLM only）
    parser.add_argument('--use_dual_branch', type=lambda x: x.lower() == 'true', default=False,
                        help='Use Dual-Branch encoder (RG-LRU + LLM only), ~50%% params of four-branch')

    # Ranking Loss（ListNet/ListMLE）- 优化top-K排序
    parser.add_argument('--ranking_loss', type=str, default='none', choices=['none', 'listnet', 'listmle'],
                        help='Ranking loss type: none (default CE), listnet, or listmle')
    parser.add_argument('--ranking_weight', type=float, default=0.1,
                        help='Weight for ranking loss component (default: 0.1)')

    # Decoder类型：ODE vs PDE
    parser.add_argument('--decoder_type', type=str, default='ode',
                        choices=['ode', 'pde_diffusion', 'pde_reaction_diffusion'],
                        help='Decoder type: ode (default), pde_diffusion, or pde_reaction_diffusion')

    # 冷启动自适应门控（logit-bias 版）
    parser.add_argument('--use_cold_start_gate', type=lambda x: x.lower() == 'true', default=False,
                        help='Use cold-start adaptive gating (logit-bias) for dual-branch encoder')
    # ===== Principled Dual-Branch Encoder =====
    parser.add_argument('--use_principled_dual_branch', type=lambda x: x.lower() == 'true', default=False,
                        help='Use Principled Dual-Branch encoder (Symmetric Delta + Sparse MoE + Prediction-Guided Router)')
    parser.add_argument('--router_mode', type=str, default='learned',
                        choices=['learned', 'pgr', 'static'],
                        help='Branch router mode: learned (default), pgr, or static')
    parser.add_argument('--pop_gate_strength', type=float, default=1.0,
                        help='Popularity gate strength for learned router (default: 1.0)')
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of experts in Sparse MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of experts to activate per token (default: 2)')
    parser.add_argument('--pgr_warmup_forwards', type=int, default=2000,
                        help='PGR warmup steps (fixed 0.5/0.5 weights before routing), default 2000')
    parser.add_argument('--pgr_update_interval', type=int, default=1,
                        help='Update PGR every N batches (default: 1)')
    parser.add_argument('--pgr_update_start_steps', type=int, default=0,
                        help='Delay PGR updates until this global step (default: 0)')
    parser.add_argument('--pgr_update_mode', type=str, default='branch_reward',
                        choices=['branch_reward', 'hard', 'global'],
                        help='PGR update mode: branch_reward (default), hard, or global')
    parser.add_argument('--pgr_reward_type', type=str, default='loss',
                        choices=['loss', 'hr'],
                        help='PGR reward type for branch updates: loss (default) or hr')
    parser.add_argument('--pgr_loss_temp', type=float, default=1.0,
                        help='Temperature for loss-based PGR reward (default: 1.0)')
    parser.add_argument('--pgr_reward_normalize', type=lambda x: x.lower() == 'true', default=True,
                        help='Normalize branch rewards to relative scale (default: True)')
    parser.add_argument('--gate_supervision_weight', type=float, default=0.1,
                        help='Weight for learned-gate supervision loss (default: 0.1)')
    parser.add_argument('--gate_supervision_temp', type=float, default=1.0,
                        help='Temperature for learned-gate supervision target (default: 1.0)')
    parser.add_argument('--gate_supervision_interval', type=int, default=1,
                        help='Compute gate supervision every N steps (default: 1)')
    parser.add_argument('--pgr_reward_k', type=int, default=0,
                        help='Top-K for PGR reward (0=auto use max metric_ks)')

    # ===== 模型保存和加载 =====
    parser.add_argument('--save_model', type=lambda x: x.lower() == 'true', default=True,
                        help='Save best model checkpoint (default: True)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints (default: checkpoints)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (for fine-tuning or evaluation)')
    parser.add_argument('--eval_only', type=lambda x: x.lower() == 'true', default=False,
                        help='Only evaluate model, no training (requires --load_checkpoint)')

    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        os.makedirs(args.log_file)
    if not os.path.exists(args.log_file + args.dataset):
        os.makedirs(args.log_file + args.dataset)

    run_json_path = _next_run_json_path(args.log_file + args.dataset, args.dataset)
    run_logger = RunJsonLogger(run_json_path, args)
    run_logger.log("run_start")

    print(args)

    log_filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + f'_{args.encoder_type}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=args.log_file + args.dataset + '/' + log_filename,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')
    logger = logging.getLogger(__name__)
    logger.info(args)
    run_logger.log("log_file", {"path": args.log_file + args.dataset + '/' + log_filename})

    try:
        main(args, logger, run_logger=run_logger)
        run_logger.log("run_end", {"status": "success"})
    except Exception as exc:
        run_logger.log("run_end", {"status": "error", "error": repr(exc)})
        raise
