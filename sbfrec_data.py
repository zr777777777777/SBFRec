import random

import torch
import torch.utils.data as data_utils

class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len, args):
        self.id2seq = id2seq
        self.max_len = max_len
        # DA4Rec-style lightweight augmentations (crop/mask/reorder)
        self.use_aug = getattr(args, 'use_aug', True)
        self.aug_crop_prob = getattr(args, 'aug_crop_prob', 0.2)
        self.aug_mask_prob = getattr(args, 'aug_mask_prob', 0.15)
        self.aug_reorder_prob = getattr(args, 'aug_reorder_prob', 0.2)
        self.aug_reorder_max_span = getattr(args, 'aug_reorder_max_span', 4)

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1]]
        tokens = seq[:-1]
        if self.use_aug:
            tokens = self._augment(tokens)
        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _augment(self, tokens):
        """Apply lightweight DA4Rec augmentations on-the-fly."""
        if len(tokens) <= 1:
            return tokens

        # 1) Random crop: keep a random tail segment (keeps target untouched)
        if random.random() < self.aug_crop_prob:
            keep_len = random.randint(max(1, len(tokens) // 2), len(tokens))
            tokens = tokens[-keep_len:]

        # 2) Local reorder: shuffle a small window to add order noise
        if len(tokens) > 2 and random.random() < self.aug_reorder_prob:
            span = min(self.aug_reorder_max_span, len(tokens))
            start = random.randint(0, len(tokens) - span)
            window = tokens[start:start + span]
            random.shuffle(window)
            tokens = tokens[:start] + window + tokens[start + span:]

        # 3) Mask: drop some tokens to simulate missing interactions
        if self.aug_mask_prob > 0:
            tokens = [
                0 if (tok != 0 and random.random() < self.aug_mask_prob) else tok
                for tok in tokens
            ]

        return tokens

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.args = args
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        for user_temp, seq_temp in self.u2seq.items():
            for star in range(len(seq_temp)-1):
                self.id_seq[idx] = seq_temp[:star+2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len, self.args)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
