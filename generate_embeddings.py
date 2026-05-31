import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# ======================= 模型配置 =======================

MODEL_CONFIGS = {
    # E5 系列 (Microsoft) - 需要加 "passage: " 前缀
    'e5-small': {
        'name': 'intfloat/e5-small-v2',
        'dim': 384,
        'prefix': 'passage: ',
        'type': 'e5'
    },
    'e5-base': {
        'name': 'intfloat/e5-base-v2',
        'dim': 768,
        'prefix': 'passage: ',
        'type': 'e5'
    },
    'e5-large': {
        'name': 'intfloat/e5-large-v2',
        'dim': 1024,
        'prefix': 'passage: ',
        'type': 'e5'
    },
    
    # BGE 系列 (BAAI/智源)
    'bge-small': {
        'name': 'BAAI/bge-small-en-v1.5',
        'dim': 384,
        'prefix': '',
        'type': 'bge'
    },
    'bge-base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dim': 768,
        'prefix': '',
        'type': 'bge'
    },
    'bge-large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'prefix': '',
        'type': 'bge'
    },
    
    # GTE 系列 (Alibaba) - 对短文本效果好
    'gte-small': {
        'name': 'thenlper/gte-small',
        'dim': 384,
        'prefix': '',
        'type': 'gte'
    },
    'gte-base': {
        'name': 'thenlper/gte-base',
        'dim': 768,
        'prefix': '',
        'type': 'gte'
    },
    'gte-large': {
        'name': 'thenlper/gte-large',
        'dim': 1024,
        'prefix': '',
        'type': 'gte'
    },
    
    # Sentence-BERT 系列
    'minilm': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'prefix': '',
        'type': 'sbert'
    },
    'mpnet': {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'dim': 768,
        'prefix': '',
        'type': 'sbert'
    },
    
    # 自定义模型
    'custom': {
        'name': None,
        'dim': None,
        'prefix': '',
        'type': 'custom'
    }
}


def load_dataset(dataset_name):
    """加载数据集并解析 item 文本"""
    path_data = f'datasets/data/{dataset_name}/dataset.pkl'
    
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"Dataset not found: {path_data}")
    
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    
    print(f"[INFO] Dataset keys: {list(data_raw.keys())}")
    
    # 获取 smap
    smap = data_raw['smap']
    num_items = len(smap) + 1  # +1 for padding (index 0)
    
    print(f"[INFO] Number of items: {num_items - 1} (+ 1 padding)")
    
    # 构建 内部ID -> 文本 的映射
    id_to_text = {}
    
    # 优先级1: item_text 字段
    if 'item_text' in data_raw:
        item_text_dict = data_raw['item_text']
        print(f"[INFO] Found 'item_text' field with {len(item_text_dict)} entries")
        
        for orig_id, internal_id in smap.items():
            if orig_id in item_text_dict:
                id_to_text[internal_id] = str(item_text_dict[orig_id])
            elif internal_id in item_text_dict:
                id_to_text[internal_id] = str(item_text_dict[internal_id])
            else:
                id_to_text[internal_id] = f'item_{internal_id}'
    
    # 优先级2: smap 的 key 本身是文本
    else:
        first_key = next(iter(smap.keys()))
        if isinstance(first_key, str) and len(first_key) > 3:
            print(f"[INFO] Using smap keys as item text")
            for orig_id, internal_id in smap.items():
                id_to_text[internal_id] = f"Product {orig_id}"
        else:
            # 检查其他可能的文本字段
            for key in ['item_meta', 'meta', 'item_info', 'item_name']:
                if key in data_raw:
                    meta = data_raw[key]
                    print(f"[INFO] Found '{key}' field")
                    for orig_id, internal_id in smap.items():
                        if orig_id in meta:
                            id_to_text[internal_id] = str(meta[orig_id])
                        else:
                            id_to_text[internal_id] = f'item_{internal_id}'
                    break
    
    # 填充缺失的 item
    for i in range(1, num_items):
        if i not in id_to_text:
            id_to_text[i] = f'item_{i}'
    
    # 转换为列表 (index 0 留空，稍后用零向量)
    # 注意: item_texts[0] 不会被编码，我们单独处理
    item_texts = [''] + [id_to_text[i] for i in range(1, num_items)]
    
    # 统计
    valid_texts = sum(1 for t in item_texts[1:] if not t.startswith('item_'))
    placeholder_count = len(item_texts) - 1 - valid_texts
    
    print(f"[INFO] Items with real text: {valid_texts}/{num_items-1}")
    if placeholder_count > 0:
        print(f"[WARN] Items with placeholder: {placeholder_count}")
    
    # 打印样本
    print(f"\n[INFO] Sample items:")
    for i in range(1, min(6, num_items)):
        text = item_texts[i][:80] + '...' if len(item_texts[i]) > 80 else item_texts[i]
        print(f"  [{i}]: {text}")
    
    return item_texts, num_items, data_raw


def check_text_quality(item_texts):
    """检查文本质量"""
    print(f"\n[文本质量检查]")
    
    lengths = [len(t) for t in item_texts[1:]]  # 跳过 index 0
    
    print(f"  文本长度: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # 统计问题文本
    too_short = sum(1 for l in lengths if l < 10)
    placeholder = sum(1 for t in item_texts[1:] if t.startswith('item_'))
    
    if too_short > 0:
        print(f"  ⚠️  太短 (<10 字符): {too_short} 个")
    if placeholder > 0:
        print(f"  ⚠️  占位符文本: {placeholder} 个")
    
    # 检查重复
    unique_texts = len(set(item_texts[1:]))
    duplicates = len(item_texts) - 1 - unique_texts
    if duplicates > 0:
        print(f"  ⚠️  重复文本: {duplicates} 个")
    
    return {
        'min_len': min(lengths),
        'mean_len': np.mean(lengths),
        'too_short': too_short,
        'placeholder': placeholder,
        'duplicates': duplicates
    }


def generate_embeddings(item_texts, model_key, custom_model_name=None, batch_size=32, device='cuda'):
    """生成 embeddings"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] Please install sentence-transformers:")
        print("  pip install sentence-transformers")
        return None
    
    # 获取模型配置
    if model_key == 'custom':
        if custom_model_name is None:
            raise ValueError("Must specify --model_name for custom model")
        config = {
            'name': custom_model_name,
            'dim': None,
            'prefix': '',
            'type': 'custom'
        }
    else:
        config = MODEL_CONFIGS.get(model_key)
        if config is None:
            print(f"[ERROR] Unknown model: {model_key}")
            print(f"[INFO] Available models: {list(MODEL_CONFIGS.keys())}")
            return None
    
    model_name = config['name']
    prefix = config['prefix']
    
    print(f"\n[INFO] Loading model: {model_name}")
    print(f"[INFO] Model type: {config['type']}")
    
    # 加载模型
    model = SentenceTransformer(model_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {embed_dim}")
    
    # 只编码真实文本 (跳过 index 0)
    texts_to_encode = item_texts[1:]
    
    # 添加前缀 (E5 模型需要)
    if prefix:
        print(f"[INFO] Adding prefix: '{prefix}'")
        texts_to_encode = [prefix + t for t in texts_to_encode]
    
    # 编码
    print(f"\n[INFO] Encoding {len(texts_to_encode)} items (batch_size={batch_size})...")
    
    item_embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize
    )
    
    # 构建完整 embeddings: index 0 为零向量
    embeddings = np.zeros((len(item_texts), embed_dim), dtype=np.float32)
    embeddings[1:] = item_embeddings
    
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Index 0 (PAD): zero vector")
    
    return embeddings, embed_dim


def analyze_embeddings(embeddings, item_texts, sample_size=1000):
    """分析 embedding 质量"""
    print(f"\n{'='*50}")
    print(f"[质量诊断]")
    print(f"{'='*50}")
    
    # 只分析真实 item (跳过 index 0)
    real_embeddings = embeddings[1:]
    n_items = len(real_embeddings)
    
    # 1. 基础统计
    norms = np.linalg.norm(real_embeddings, axis=1)
    print(f"\n  [Norm 统计]")
    print(f"    min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    
    zero_norm_count = np.sum(norms < 1e-6)
    if zero_norm_count > 0:
        print(f"    ⚠️  零向量: {zero_norm_count} 个")
    
    # 2. 相似度分布分析
    print(f"\n  [相似度分布] (采样 {min(sample_size, n_items)} 个)")
    
    n_sample = min(sample_size, n_items)
    indices = np.random.choice(n_items, size=n_sample, replace=False)
    sampled = real_embeddings[indices]
    
    # 计算两两余弦相似度 (已经 L2 normalized，直接点积)
    sim_matrix = sampled @ sampled.T
    
    # 取上三角 (不含对角线)
    upper_tri_indices = np.triu_indices(n_sample, k=1)
    similarities = sim_matrix[upper_tri_indices]
    
    sim_mean = similarities.mean()
    sim_std = similarities.std()
    sim_min = similarities.min()
    sim_max = similarities.max()
    
    print(f"    min={sim_min:.4f}, max={sim_max:.4f}")
    print(f"    mean={sim_mean:.4f}, std={sim_std:.4f}")
    
    # 分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(similarities, percentiles)
    print(f"    分位数: ", end='')
    print(', '.join([f"P{p}={v:.3f}" for p, v in zip(percentiles, pct_values)]))
    
    # 3. 预警
    print(f"\n  [诊断结果]")
    
    warnings = []
    
    if sim_mean > 0.85:
        warnings.append(f"❌ 严重: 平均相似度过高 ({sim_mean:.3f} > 0.85)，embedding 几乎无区分度!")
    elif sim_mean > 0.75:
        warnings.append(f"⚠️  警告: 平均相似度偏高 ({sim_mean:.3f} > 0.75)，可能影响推荐效果")
    else:
        print(f"    ✓ 平均相似度正常 ({sim_mean:.3f})")
    
    if sim_std < 0.05:
        warnings.append(f"⚠️  警告: 相似度方差过小 ({sim_std:.3f} < 0.05)，embedding 缺乏多样性")
    else:
        print(f"    ✓ 相似度方差正常 ({sim_std:.3f})")
    
    if sim_max > 0.99:
        # 检查有多少对接近重复
        near_duplicate = np.sum(similarities > 0.99)
        warnings.append(f"⚠️  警告: 存在 {near_duplicate} 对近似重复 (sim > 0.99)")
    
    for w in warnings:
        print(f"    {w}")
    
    # 4. 找出最相似和最不相似的 pair
    print(f"\n  [样例对比]")
    
    # 最相似
    flat_idx = np.argmax(similarities)
    i, j = upper_tri_indices[0][flat_idx], upper_tri_indices[1][flat_idx]
    real_i, real_j = indices[i] + 1, indices[j] + 1  # +1 因为跳过了 index 0
    print(f"    最相似 (sim={sim_max:.4f}):")
    print(f"      [{real_i}]: {item_texts[real_i][:60]}...")
    print(f"      [{real_j}]: {item_texts[real_j][:60]}...")
    
    # 最不相似
    flat_idx = np.argmin(similarities)
    i, j = upper_tri_indices[0][flat_idx], upper_tri_indices[1][flat_idx]
    real_i, real_j = indices[i] + 1, indices[j] + 1
    print(f"    最不相似 (sim={sim_min:.4f}):")
    print(f"      [{real_i}]: {item_texts[real_i][:60]}...")
    print(f"      [{real_j}]: {item_texts[real_j][:60]}...")
    
    return {
        'sim_mean': sim_mean,
        'sim_std': sim_std,
        'sim_min': sim_min,
        'sim_max': sim_max,
        'warnings': warnings
    }


def save_embeddings(embeddings, dataset_name, model_key, embed_dim, quality_stats):
    """保存 embeddings"""
    output_dir = f'datasets/data/{dataset_name}'
    
    # 保存主文件
    main_path = os.path.join(output_dir, 'llm_embeddings.npy')
    np.save(main_path, embeddings)
    print(f"\n[INFO] Saved: {main_path}")
    
    # 保存带模型名的备份
    backup_path = os.path.join(output_dir, f'llm_embeddings_{model_key}.npy')
    np.save(backup_path, embeddings)
    print(f"[INFO] Saved: {backup_path}")
    
    # 保存元数据
    meta = {
        'model': model_key,
        'dim': embed_dim,
        'num_items': len(embeddings),
        'normalized': True,
        'pad_is_zero': True,
        'quality': quality_stats
    }
    meta_path = os.path.join(output_dir, 'llm_embeddings_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"[INFO] Saved: {meta_path}")
    
    return main_path


def main():
    parser = argparse.ArgumentParser(description='Generate item embeddings with quality analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, default='e5-base',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Embedding model to use')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom model name (for --model custom)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--sample_size', type=int, default=1000, 
                        help='Sample size for similarity analysis')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print("\n可用的 Embedding 模型:")
        print("=" * 70)
        print(f"  {'Key':<15} | {'Dim':<5} | {'Model Name'}")
        print("-" * 70)
        for key, config in MODEL_CONFIGS.items():
            if key != 'custom':
                print(f"  {key:<15} | {config['dim']:<5} | {config['name']}")
        print("\n推荐:")
        print("  - 短文本 (商品标题): gte-base, bge-base")
        print("  - 长文本 (商品描述): e5-base, e5-large")
        print("\n使用示例:")
        print("  python generate_embeddings_v2.py --dataset amazon_beauty --model gte-base")
        return
    
    print("=" * 60)
    print("Item Embedding Generator v2")
    print("=" * 60)
    
    # 1. 加载数据
    item_texts, num_items, data_raw = load_dataset(args.dataset)
    
    # 2. 文本质量检查
    text_stats = check_text_quality(item_texts)
    
    # 3. 生成 embeddings
    result = generate_embeddings(
        item_texts,
        args.model,
        args.model_name,
        args.batch_size,
        args.device
    )
    
    if result is None:
        return
    
    embeddings, embed_dim = result
    
    # 4. 质量分析
    quality_stats = analyze_embeddings(embeddings, item_texts, args.sample_size)
    
    # 5. 保存
    save_embeddings(embeddings, args.dataset, args.model, embed_dim, quality_stats)
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"  Embedding 维度: {embed_dim}")
    print(f"  平均相似度: {quality_stats['sim_mean']:.4f}")
    if quality_stats['warnings']:
        print(f"  ⚠️  存在 {len(quality_stats['warnings'])} 个警告，请检查上方诊断结果")
    print("=" * 60)


if __name__ == '__main__':
    main()
