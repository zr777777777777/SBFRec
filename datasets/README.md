# Datasets

This directory hosts the sequential-recommendation datasets consumed by [main.py](../main.py) and [generate_embeddings.py](../generate_embeddings.py).

## Directory layout

```
datasets/
└── data/
    └── <DATASET>/
        ├── dataset.pkl                # required: interaction sequences + smap
        ├── llm_embeddings_qwen.npy    # optional: LLM item embeddings (for *_llm encoders)
        └── llm_embeddings.npy         # optional fallback name
```

Replace `<DATASET>` with the name you pass to `--dataset` (e.g. `amazon_beauty`, `ml-1m`, `yelp`). [main.py:24](../main.py#L24) loads `datasets/data/<DATASET>/dataset.pkl` directly — the path is hard-coded relative to the CWD, so always launch training from the project root.

## `dataset.pkl` schema

A pickle of a single Python `dict` with the following keys:

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `train` | `Dict[user_id, List[item_id]]` | yes | Each user's training interaction sequence, chronologically ordered. Item IDs are remapped internal integers (≥ 1, since 0 is reserved for padding). |
| `val` | `Dict[user_id, List[item_id]]` | yes | One held-out next item per user — `List[item_id]` is usually length 1. |
| `test` | `Dict[user_id, List[item_id]]` | yes | The final held-out next item per user (length 1). |
| `smap` | `Dict[original_id, internal_id]` | yes | Maps the raw item ID (e.g. ASIN) to the contiguous internal ID used in `train`/`val`/`test`. `len(smap) + 1` becomes `args.item_num`. |
| `item_text` | `Dict[id, str]` | optional | Per-item text (title / description / metadata). Required only if you want to run `*_llm` encoders. The keys can be either original or internal IDs — [generate_embeddings.py:115-126](../generate_embeddings.py#L115-L126) handles both. |

### Splitting convention

The code follows the standard leave-one-out protocol used by BERT4Rec / SASRec:

```
[i1, i2, i3, ..., i_{n-2}]  → train
[i_{n-1}]                   → val
[i_n]                       → test
```

Inside [sbfrec_data.py:74-78](../sbfrec_data.py#L74-L78) the trainer further explodes each `train` sequence into one example per prefix (`seq[:2], seq[:3], ..., seq[:n]`), using the last element as the label.

### Padding & ID conventions

- Item ID `0` is reserved for padding. All real items must have IDs `≥ 1`.
- Sequences are right-aligned and zero-padded to `args.max_len` (default 50) on the left.
- User IDs can be any hashable type (usually `int`); they are only used as dict keys.

## Building `dataset.pkl` from raw interactions

There is no built-in preprocessing script — `dataset.pkl` is treated as an input artifact. The standard recipe (matches the public RecSys community convention):

1. **Filter** out users with `< 5` interactions and items with `< 5` interactions (5-core filter), iterating until stable.
2. **Sort** each user's interactions chronologically by timestamp.
3. **Build `smap`** by assigning each surviving item a contiguous internal ID starting at `1`.
4. **Leave-one-out split** — last item → `test`, second-to-last → `val`, the rest → `train`.
5. **Pickle** the resulting dict.

A minimal reference implementation:

```python
import pickle
from collections import defaultdict

# raw_interactions: List[(user, item, timestamp)] and optional item_text_map: {raw_item_id: str}

def build_dataset(raw_interactions, item_text_map=None, min_count=5):
    # 5-core filter (iterative)
    while True:
        u_cnt = defaultdict(int); i_cnt = defaultdict(int)
        for u, i, _ in raw_interactions:
            u_cnt[u] += 1; i_cnt[i] += 1
        keep = [(u, i, t) for u, i, t in raw_interactions
                if u_cnt[u] >= min_count and i_cnt[i] >= min_count]
        if len(keep) == len(raw_interactions):
            break
        raw_interactions = keep

    # group + sort
    by_user = defaultdict(list)
    for u, i, t in raw_interactions:
        by_user[u].append((t, i))
    for u in by_user:
        by_user[u].sort()

    # smap (internal id starts at 1, 0 = pad)
    items = sorted({i for seq in by_user.values() for _, i in seq})
    smap = {raw_id: idx + 1 for idx, raw_id in enumerate(items)}

    train, val, test = {}, {}, {}
    for u, seq in by_user.items():
        ids = [smap[i] for _, i in seq]
        if len(ids) < 3:
            continue  # need >=1 train, 1 val, 1 test
        train[u] = ids[:-2]
        val[u]   = [ids[-2]]
        test[u]  = [ids[-1]]

    out = {'train': train, 'val': val, 'test': test, 'smap': smap}
    if item_text_map is not None:
        out['item_text'] = {raw_id: item_text_map[raw_id]
                            for raw_id in smap if raw_id in item_text_map}
    return out

with open('datasets/data/my_dataset/dataset.pkl', 'wb') as f:
    pickle.dump(build_dataset(raw_interactions, item_text_map), f)
```

## LLM embeddings (optional)

The `*_llm` encoder variants (`rglru_fft_llm`, `mamba_fft_llm`, `rglru_local_fft_llm`, `transformer_llm`) require a `.npy` file with semantic item embeddings.

Run from the project root:

```bash
python generate_embeddings.py --dataset <DATASET> --model qwen-7b
```

Supported `--model` keys (see [generate_embeddings.py:9-91](../generate_embeddings.py#L9-L91)): `e5-small/base/large`, `bge-small/base/large`, `gte-small/base/large`, `minilm`, `mpnet`, or `custom` (pass `--model_name <hf_id>`).

Output: a `float32` array of shape `(item_num, llm_dim)`, L2-normalized, with row 0 set to the zero vector (padding). [main.py:84-89](../main.py#L84-L89) auto-detects `llm_dim` from the file, so you don't need to match `--llm_dim` to the model manually — but you can override it.

Naming convention (searched in this order — [main.py:53-56](../main.py#L53-L56)):

1. The file passed via `--llm_emb_file` (under the dataset dir).
2. `llm_embeddings_qwen.npy`
3. `llm_embeddings.npy`

If none of these are found, the trainer falls back to the non-LLM variant of the chosen encoder (e.g. `rglru_local_fft_llm` → `rglru_local_fft`) and prints a warning.

## Verifying a dataset

```python
import pickle
with open('datasets/data/<DATASET>/dataset.pkl', 'rb') as f:
    d = pickle.load(f)
print('users:', len(d['train']))
print('items:', len(d['smap']))
print('sample:', next(iter(d['train'].items())))
assert all(min(seq) >= 1 for seq in d['train'].values()), "item id 0 reserved for padding"
```

If this passes and `len(val) == len(test) == len(train)`, the dataset is ready to plug into `python main.py --dataset <DATASET>`.
