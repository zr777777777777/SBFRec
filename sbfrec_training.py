import copy
import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sbfrec_losses import RankingLoss

try:
    from principled_dual_branch import compute_prediction_reward
except ImportError:
    compute_prediction_reward = None

# ======================= Training =======================

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    return [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    return (hit/log2).sum(dim=-1)


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().cpu(), scores.clone().detach().cpu(), ks)
    hr = cal_hr(labels.clone().detach().cpu(), scores.clone().detach().cpu(), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics[f'HR@{k}'] = hr_temp
        metrics[f'NDCG@{k}'] = ndcg_temp
    return metrics


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger, run_logger=None):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    Loss_Alpha = args.Loss_Alpha
    Loss_Beta = args.Loss_Beta
    
    # 熵正则损失权重（默认0=关闭，与test.py行为一致）
    entropy_weight = getattr(args, 'entropy_weight', 0.0)
    # MoE负载均衡辅助损失权重
    moe_aux_weight = getattr(args, 'moe_aux_weight', 1.0)
    # LLM对齐损失权重
    llm_align_weight = float(getattr(args, 'llm_align_weight', 0.0))
    
    # Ranking Loss 初始化
    ranking_loss_type = getattr(args, 'ranking_loss', 'none')  # none/listnet/listmle
    ranking_weight = getattr(args, 'ranking_weight', 0.1)
    ranking_loss_fn = None
    if ranking_loss_type != 'none':
        ranking_loss_fn = RankingLoss(
            ranking_loss_type,  # loss_type
            ranking_weight,     # ranking_weight
            1.0,                # temperature
            20                  # top_k
        ).to(device)
        print(f"[INFO] Using {ranking_loss_type.upper()} ranking loss with weight={ranking_weight}")
    
    model_joint = model_joint.to(device)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    pgr_update_interval = int(getattr(args, 'pgr_update_interval', 1))
    pgr_update_start_steps = int(getattr(args, 'pgr_update_start_steps', 0))
    router_mode = getattr(args, 'router_mode', 'pgr')
    pgr_update_mode = getattr(args, 'pgr_update_mode', 'branch_reward')
    pgr_reward_type = getattr(args, 'pgr_reward_type', 'loss')
    pgr_loss_temp = float(getattr(args, 'pgr_loss_temp', 1.0))
    pgr_reward_normalize = getattr(args, 'pgr_reward_normalize', True)
    gate_supervision_weight = float(getattr(args, 'gate_supervision_weight', 0.0))
    gate_supervision_temp = float(getattr(args, 'gate_supervision_temp', 1.0))
    gate_supervision_interval = max(1, int(getattr(args, 'gate_supervision_interval', 1)))
    pgr_reward_k = int(getattr(args, 'pgr_reward_k', 0))
    if pgr_reward_k <= 0:
        pgr_reward_k = max(metric_ks) if metric_ks else 10
    # Match test.py: disable AMP for consistent FP32 behavior
    use_amp = False
    
    best_metrics_dict = {f'Best_HR@{k}': 0.0 for k in metric_ks}
    best_metrics_dict.update({f'Best_NDCG@{k}': 0.0 for k in metric_ks})
    best_epoch = {f'Best_epoch_HR@{k}': -1 for k in metric_ks}
    best_epoch.update({f'Best_epoch_NDCG@{k}': -1 for k in metric_ks})

    bad_count = 0
    best_model_state = None
    best_optimizer_state = None
    best_scheduler_state = None
    
    # 模型选择策略：至少min_improved_metrics个指标提升才更新
    min_improved = getattr(args, 'min_improved_metrics', 3)
    best_primary_epoch = -1
    print(f"[INFO] Model selection: update when >= {min_improved} metrics improve")
    if run_logger:
        run_logger.log("train_start", {
            "epochs": epochs,
            "metric_ks": list(metric_ks),
            "ranking_loss": ranking_loss_type,
            "ranking_weight": ranking_weight,
        })

    global_step = 0
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
        epoch_losses = []
        epoch_entropy_losses = []
        epoch_aux_losses = []
        epoch_gate_losses = []
        epoch_llm_align_losses = []
        flag_update = 0
        
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            # FP32 training to match test.py baseline
            loss_mse, fm_rep, weights, t, loss_FM_mse, z0 = model_joint(train_batch[0], train_batch[1], train_flag=True)
            labels_flat = train_batch[1]
            if labels_flat.dim() > 1:
                labels_flat = labels_flat.squeeze(-1)
            
            # 选择损失函数：Ranking Loss 或 CrossEntropy
            if ranking_loss_fn is not None:
                loss_fm_total, loss_fm_ce, loss_fm_ranking = model_joint.loss_fm_ranking(fm_rep, labels_flat, ranking_loss_fn)
                loss_fm_value = loss_fm_total
            else:
                loss_fm_value = model_joint.loss_fm_ce(fm_rep, labels_flat)
            
            loss_FM = loss_FM_mse
            
            # 主损失
            loss_all = loss_FM + Loss_Alpha * loss_fm_value + Loss_Beta * loss_mse
            
            # 熵正则损失（当entropy_weight > 0时启用）
            if entropy_weight > 0:
                entropy_loss = model_joint.get_entropy_loss()
                if isinstance(entropy_loss, torch.Tensor) and entropy_loss.requires_grad:
                    loss_all = loss_all + entropy_weight * entropy_loss
                    epoch_entropy_losses.append(entropy_loss.item())

            # MoE负载均衡辅助损失（当moe_aux_weight > 0时启用）
            if moe_aux_weight > 0:
                aux_loss = model_joint.get_aux_loss()
                if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                    loss_all = loss_all + moe_aux_weight * aux_loss
                    epoch_aux_losses.append(aux_loss.item())

            # LLM对齐损失（可选）
            if llm_align_weight > 0:
                align_loss = model_joint.get_llm_align_loss(labels_flat)
                if isinstance(align_loss, torch.Tensor) and align_loss.requires_grad:
                    loss_all = loss_all + llm_align_weight * align_loss
                    epoch_llm_align_losses.append(align_loss.item())

            # 门控监督（learned router）：用分支loss构造目标权重
            if (router_mode == 'learned' and gate_supervision_weight > 0
                    and global_step % gate_supervision_interval == 0):
                with torch.no_grad():
                    _, rep_fm_seq, _, _, _, _ = model_joint(
                        train_batch[0], train_batch[1], train_flag=True,
                        force_branch='seq', record_stats=False,
                        fixed_t_rf=t, fixed_z0=z0
                    )
                    _, rep_fm_sem, _, _, _, _ = model_joint(
                        train_batch[0], train_batch[1], train_flag=True,
                        force_branch='sem', record_stats=False,
                        fixed_t_rf=t, fixed_z0=z0
                    )
                    scores_seq = model_joint.fm_rep_pre(rep_fm_seq)
                    scores_sem = model_joint.fm_rep_pre(rep_fm_sem)
                    loss_seq = F.cross_entropy(scores_seq, labels_flat, reduction='none')
                    loss_sem = F.cross_entropy(scores_sem, labels_flat, reduction='none')
                    loss_vec = torch.stack([loss_seq, loss_sem], dim=-1)  # (B, 2)
                    inv_temp = 1.0 / max(gate_supervision_temp, 1e-6)
                    gate_target = torch.softmax(-loss_vec * inv_temp, dim=-1)  # (B, 2)
                gate_weights = model_joint.get_gate_weights()
                if isinstance(gate_weights, torch.Tensor):
                    if gate_weights.dim() == 3:
                        gate_weights = gate_weights.squeeze(1)  # (B, 2)
                    gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
                    gate_loss = F.kl_div(torch.log(gate_weights + 1e-8), gate_target, reduction='batchmean')
                    loss_all = loss_all + gate_supervision_weight * gate_loss
                    epoch_gate_losses.append(gate_loss.item())
            
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model_joint.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss_all.detach().item())

            # ===== PGR reward 更新 =====
            if (router_mode == 'pgr' and compute_prediction_reward is not None and pgr_update_interval > 0
                    and global_step >= pgr_update_start_steps
                    and global_step % pgr_update_interval == 0):
                att = getattr(model_joint.fm.xstart_model, 'att', None)
                if att is not None and hasattr(att, 'update_router_with_reward'):
                    with torch.no_grad():
                        if pgr_update_mode in ['branch_reward', 'hard']:
                            _, rep_fm_seq, _, _, _, _ = model_joint(
                                train_batch[0], train_batch[1], train_flag=True,
                                force_branch='seq', record_stats=False,
                                fixed_t_rf=t, fixed_z0=z0
                            )
                            _, rep_fm_sem, _, _, _, _ = model_joint(
                                train_batch[0], train_batch[1], train_flag=True,
                                force_branch='sem', record_stats=False,
                                fixed_t_rf=t, fixed_z0=z0
                            )
                            if pgr_reward_type == 'loss':
                                if ranking_loss_fn is not None:
                                    loss_seq, _, _ = model_joint.loss_fm_ranking(rep_fm_seq, labels_flat, ranking_loss_fn)
                                    loss_sem, _, _ = model_joint.loss_fm_ranking(rep_fm_sem, labels_flat, ranking_loss_fn)
                                else:
                                    loss_seq = model_joint.loss_fm_ce(rep_fm_seq, labels_flat)
                                    loss_sem = model_joint.loss_fm_ce(rep_fm_sem, labels_flat)
                                loss_vec = torch.stack([loss_seq, loss_sem])
                                inv_temp = 1.0 / max(pgr_loss_temp, 1e-6)
                                reward_vec = torch.softmax(-loss_vec * inv_temp, dim=0)
                                att.update_router_with_reward(
                                    reward_vec,
                                    hard_update=(pgr_update_mode == 'hard'),
                                    normalize_reward=False
                                )
                            else:
                                scores_seq = model_joint.fm_rep_pre(rep_fm_seq)
                                scores_sem = model_joint.fm_rep_pre(rep_fm_sem)
                                reward_seq = compute_prediction_reward(scores_seq, labels_flat, k=pgr_reward_k)
                                reward_sem = compute_prediction_reward(scores_sem, labels_flat, k=pgr_reward_k)
                                att.update_router_with_reward(
                                    [reward_seq, reward_sem],
                                    hard_update=(pgr_update_mode == 'hard'),
                                    normalize_reward=pgr_reward_normalize
                                )
                        else:
                            scores_reward = model_joint.fm_rep_pre(fm_rep)
                            reward = compute_prediction_reward(scores_reward, labels_flat, k=pgr_reward_k)
                            att.update_router_with_reward(reward, normalize_reward=pgr_reward_normalize)
            global_step += 1

            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))

        average_loss = sum(epoch_losses) / len(epoch_losses)
        print("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))
        logger.info("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))
        
        avg_entropy = None
        gate_stats = None

        # 打印熵损失、MoE辅助损失和门控监督损失
        if epoch_entropy_losses:
            avg_entropy = sum(epoch_entropy_losses) / len(epoch_entropy_losses)
            print(f"  Avg entropy loss: {avg_entropy:.4f}")
            logger.info(f"Avg entropy loss: {avg_entropy:.4f}")
        if epoch_aux_losses:
            avg_aux = sum(epoch_aux_losses) / len(epoch_aux_losses)
            print(f"  Avg MoE aux loss: {avg_aux:.4f}")
            logger.info(f"Avg MoE aux loss: {avg_aux:.4f}")
        if epoch_gate_losses:
            avg_gate = sum(epoch_gate_losses) / len(epoch_gate_losses)
            print(f"  Avg gate supervision loss: {avg_gate:.4f}")
            logger.info(f"Avg gate supervision loss: {avg_gate:.4f}")
        if epoch_llm_align_losses:
            avg_align = sum(epoch_llm_align_losses) / len(epoch_llm_align_losses)
            print(f"  Avg LLM align loss: {avg_align:.4f}")
            logger.info(f"Avg LLM align loss: {avg_align:.4f}")
        
        # 每隔几个epoch打印门控统计（用于分析各分支使用情况）
        if epoch_temp % 10 == 0:
            gate_stats = model_joint.get_gate_stats()
            if gate_stats:
                print(f"  Gate stats: {gate_stats}")
                logger.info(f"Gate stats: {gate_stats}")
        
        if run_logger:
            epoch_event = {
                "epoch": epoch_temp,
                "avg_loss": float(round(average_loss, 6)),
            }
            if avg_entropy is not None:
                epoch_event["avg_entropy_loss"] = float(round(avg_entropy, 6))
            if gate_stats:
                epoch_event["gate_stats"] = gate_stats
            run_logger.log("epoch_end", epoch_event)
            train_metrics = {
                "epoch": epoch_temp,
                "metrics": {
                    "loss": float(round(average_loss, 6)),
                },
            }
            if avg_entropy is not None:
                train_metrics["metrics"]["avg_entropy_loss"] = float(round(avg_entropy, 6))
            run_logger.log("train_metrics", train_metrics)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.6g}")
        logger.info(f"LR: {current_lr:.6g}")
        lr_scheduler.step()

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            
            with torch.no_grad():
                metrics_dict = {f'HR@{k}': [] for k in metric_ks}
                metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    _, rep_fm, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                    scores_rec_fm = model_joint.fm_rep_pre(rep_fm)
                    metrics = hrs_and_ndcgs_k(scores_rec_fm, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            
            # 计算当前epoch的平均指标
            current_metrics = {}
            for key_temp, values_temp in metrics_dict.items():
                current_metrics[key_temp] = round(np.mean(values_temp) * 100, 4)
            
            # 统计有多少指标提升（不立即更新best，严格按阈值）
            improved_count = 0
            improved_metrics = []
            for key_temp, values_mean in current_metrics.items():
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    improved_count += 1
                    improved_metrics.append(key_temp)
            
            # ===== 模型选择策略（严格）=====
            min_improved = getattr(args, 'min_improved_metrics', 3)  # 必须达到阈值才更新
            
            if improved_count >= min_improved:
                # 更新最佳模型（并同步所有指标，确保best与模型一致）
                best_primary_epoch = epoch_temp
                flag_update = 1
                bad_count = 0
                best_model_state = copy.deepcopy(model_joint.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
                best_metrics_dict = {f'Best_{k}': v for k, v in current_metrics.items()}
                best_epoch = {f'Best_epoch_{k}': epoch_temp for k in current_metrics.keys()}
                print(f"  [Best Model Updated] {improved_count}/{len(current_metrics)} metrics improved: {improved_metrics}")
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
            else:
                flag_update = 0
                bad_count += 1
                if improved_count > 0:
                    print(f"  [Not Updated] Only {improved_count}/{len(current_metrics)} metrics improved (need >= {min_improved})")
                # 回退到最佳模型
                if best_model_state is not None:
                    model_joint.load_state_dict(best_model_state)
                    if best_optimizer_state is not None:
                        optimizer.load_state_dict(best_optimizer_state)
                    if best_scheduler_state is not None:
                        lr_scheduler.load_state_dict(best_scheduler_state)
                    print(f"  [Rollback] Reverted to best model at epoch {best_primary_epoch}")
            
            if run_logger:
                run_logger.log("eval_metrics", {
                    "epoch": epoch_temp,
                    "current_metrics": current_metrics,
                    "improved_count": improved_count,
                    "improved_metrics": improved_metrics,
                    "best_metrics": best_metrics_dict,
                    "best_epoch": best_epoch,
                    "bad_count": bad_count,
                })
                run_logger.log("val_metrics", {
                    "epoch": epoch_temp,
                    "metrics": current_metrics,
                })
                run_logger.log("eval_interval_metrics", {
                    "epoch": epoch_temp,
                    "eval_interval": args.eval_interval,
                    "metrics": current_metrics,
                })
                
            if bad_count >= args.patience:
                print(f"Early stopping at epoch {epoch_temp} (no model update for {args.patience} evals)")
                if run_logger:
                    run_logger.log("early_stop", {
                        "epoch": epoch_temp,
                        "bad_count": bad_count,
                        "patience": args.patience,
                    })
                break

    print(f"\n[Summary] Best model from epoch {best_primary_epoch}")
    print(f"  Final best metrics: {best_metrics_dict}")
    logger.info(f"Best model from epoch {best_primary_epoch}")
    if run_logger:
        run_logger.log("best_summary", {
            "best_primary_epoch": best_primary_epoch,
            "best_metrics": best_metrics_dict,
            "best_epoch": best_epoch,
        })
        
    if args.eval_interval > epochs or best_model_state is None:
        best_model_state = copy.deepcopy(model_joint.state_dict())
    
    # Test evaluation
    # 加载最佳模型
    model_joint.load_state_dict(best_model_state)
    model_joint.eval()
    
    with torch.no_grad():
        test_metrics_dict = {f'HR@{k}': [] for k in metric_ks}
        test_metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})
        test_metrics_dict_mean = {}
        
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            _, rep_fm, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_fm = model_joint.fm_rep_pre(rep_fm)
            metrics = hrs_and_ndcgs_k(scores_rec_fm, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        test_metrics_dict_mean[key_temp] = round(np.mean(values_temp) * 100, 4)

    print('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info('Test: ' + str(test_metrics_dict_mean))

    print('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info('Best: ' + str(best_metrics_dict))
    logger.info('Best Epoch: ' + str(best_epoch))
    if run_logger:
        run_logger.log("test_results", {
            "test_metrics": test_metrics_dict_mean,
        })
        run_logger.log("test_metrics", {
            "metrics": test_metrics_dict_mean,
        })
    
    # ===== 保存模型（NEW）=====
    save_model = getattr(args, 'save_model', True)
    if save_model:
        save_dir = getattr(args, 'save_dir', 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成模型文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        model_name = f"{args.dataset}_{args.encoder_type}"
        
        # 添加关键配置到文件名
        if getattr(args, 'ranking_loss', 'none') != 'none':
            model_name += f"_{args.ranking_loss}"
        
        model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")
        
        # 保存完整的checkpoint（包括args和metrics）
        checkpoint = {
            'model_state_dict': best_model_state,
            'args': vars(args),
            'best_metrics': best_metrics_dict,
            'best_epoch': best_epoch,
            'test_metrics': test_metrics_dict_mean,
            'best_model_epoch': best_primary_epoch,
        }
        torch.save(checkpoint, model_path)
        print(f"[INFO] Model saved to: {model_path}")
        print(f"[INFO] Model from epoch {best_primary_epoch}")
        logger.info(f"Model saved to: {model_path}")
        
        # 同时保存一个latest链接（方便快速加载最新模型）
        latest_path = os.path.join(save_dir, f"{args.dataset}_latest.pth")
        torch.save(checkpoint, latest_path)
        print(f"[INFO] Latest model link: {latest_path}")
        if run_logger:
            run_logger.log("model_saved", {
                "model_path": model_path,
                "latest_path": latest_path,
                "best_model_epoch": best_primary_epoch,
            })

    return model_joint, test_metrics_dict_mean
