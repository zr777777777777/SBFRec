import datetime
import json
import os
import random
import shlex
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

def _next_run_json_path(base_dir, dataset_name):
    os.makedirs(base_dir, exist_ok=True)
    prefix = f"{dataset_name}"
    suffix = ".json"
    max_idx = 0
    for name in os.listdir(base_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        tail = name[len(prefix):-len(suffix)]
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    return os.path.join(base_dir, f"{dataset_name}{max_idx + 1}.json")


class RunJsonLogger:
    def __init__(self, path, args):
        self.path = path
        self.data = {
            "dataset": args.dataset,
            "start_time": datetime.datetime.now().isoformat(),
            "command": {
                "argv": list(sys.argv),
                "executable": sys.executable,
                "command_line": self._format_command(),
            },
            "args": self._normalize_args(args),
            "events": [],
        }
        self._write()

    def _format_command(self):
        return " ".join(shlex.quote(arg) for arg in [sys.executable] + sys.argv)

    def _normalize_args(self, args):
        return dict(vars(args))

    def log(self, event_type, data=None):
        event = {
            "time": datetime.datetime.now().isoformat(),
            "type": event_type,
        }
        if data is not None:
            event["data"] = data
        self.data["events"].append(event)
        self._write()

    def _write(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, ensure_ascii=True, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def compute_item_popularity(u2seq, item_num, smooth=1.0, norm='log'):
    counts = np.zeros(item_num, dtype=np.float32)
    for seq in u2seq.values():
        for item_id in seq:
            if item_id > 0:
                counts[item_id] += 1.0
    
    if norm == 'log':
        counts = counts + smooth
        log_counts = np.log(counts)
        denom = log_counts.max() - log_counts.min()
        if denom > 0:
            counts = (log_counts - log_counts.min()) / (denom + 1e-8)
        else:
            counts = np.zeros_like(log_counts)
    else:
        max_count = counts.max()
        if max_count > 0:
            counts = counts / max_count
    
    counts[0] = 0.0
    return counts
