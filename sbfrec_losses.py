import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= ListNet/ListMLE Loss Functions =======================

class ListNetLoss(nn.Module):
    """
    ListNet Loss: 优化排序分布而非单点预测
    
    核心思想：
    - 将预测分数和真实标签都转换为概率分布
    - 最小化两个分布之间的交叉熵
    - 比CrossEntropy更关注相对顺序
    
    参考: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach"
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, scores, labels, mask=None):
        """
        Args:
            scores: (B, num_items) 预测分数
            labels: (B,) 真实标签（正样本的item index）
            mask: 可选，用于过滤
        Returns:
            loss: 标量
        """
        batch_size, num_items = scores.shape
        
        # 创建真实标签的one-hot分布
        # labels是正样本的index，我们希望它的概率最高
        true_dist = torch.zeros_like(scores)
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # 将预测分数转换为概率分布
        pred_dist = F.softmax(scores / self.temperature, dim=-1)
        
        # 计算交叉熵: -sum(true_dist * log(pred_dist))
        # 只在true_dist=1的位置计算（即正样本）
        loss = -torch.sum(true_dist * torch.log(pred_dist + 1e-10), dim=-1)
        
        return loss.mean()


class ListMLELoss(nn.Module):
    """
    ListMLE Loss: 最大化正确排序的似然
    
    核心思想：
    - 给定ground truth排序，计算预测分数产生该排序的概率
    - 最大化这个概率（最小化负对数似然）
    - 考虑完整的排序而非单个位置
    
    简化版本：只考虑top-k个位置
    
    参考: Xia et al., "Listwise Approach to Learning to Rank"
    """
    def __init__(self, top_k=20):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, scores, labels, mask=None):
        """
        Args:
            scores: (B, num_items) 预测分数
            labels: (B,) 真实标签（正样本的item index）
        Returns:
            loss: 标量
        """
        batch_size, num_items = scores.shape
        device = scores.device
        
        # 获取预测的top-k items
        _, pred_indices = torch.topk(scores, min(self.top_k, num_items), dim=-1)  # (B, top_k)
        
        # 计算ListMLE loss
        # 对于每个位置i，计算P(item_i | 剩余items) = exp(s_i) / sum(exp(s_j) for j >= i)
        
        # 获取top-k位置的分数
        topk_scores = torch.gather(scores, 1, pred_indices)  # (B, top_k)
        
        # 计算累积logsumexp（从后往前）
        # loss = -sum(s_i - logsumexp(s_j for j >= i))
        loss = 0.0
        for i in range(min(self.top_k, num_items)):
            remaining_scores = topk_scores[:, i:]  # (B, top_k - i)
            logsumexp = torch.logsumexp(remaining_scores, dim=-1)  # (B,)
            loss = loss - (topk_scores[:, i] - logsumexp)
        
        # 额外惩罚：如果真实标签不在top-k中
        labels_expanded = labels.unsqueeze(1)  # (B, 1)
        in_topk = (pred_indices == labels_expanded).any(dim=1).float()  # (B,)
        
        # 对不在top-k的样本增加额外损失
        true_scores = torch.gather(scores, 1, labels_expanded).squeeze(1)  # (B,)
        max_topk_scores = topk_scores[:, 0]  # (B,)
        margin_loss = F.relu(max_topk_scores - true_scores + 1.0) * (1 - in_topk)
        
        total_loss = loss.mean() / self.top_k + margin_loss.mean()
        
        return total_loss


class RankingLoss(nn.Module):
    """
    组合损失：CrossEntropy + ListNet/ListMLE
    
    目的：
    - CE优化top-1准确率
    - ListNet/ListMLE优化top-K排序质量
    - 组合使用平衡precision和coverage
    """
    def __init__(self, *args, **kwargs):
        """
        Fully robust to positional/keyword calls.
        Positional after self: loss_type, ranking_weight, temperature, top_k.
        Keyword overrides take precedence.
        """
        super().__init__()
        
        # Defaults
        loss_type = 'listnet'
        ranking_weight = 0.1
        temperature = 1.0
        top_k = 20
        
        # Positional overrides (after self)
        if len(args) > 0:
            loss_type = args[0]
        if len(args) > 1:
            ranking_weight = args[1]
        if len(args) > 2:
            temperature = args[2]
        if len(args) > 3:
            top_k = args[3]
        
        # Keyword overrides
        if 'loss_type' in kwargs:
            loss_type = kwargs['loss_type']
        if 'ranking_weight' in kwargs:
            ranking_weight = kwargs['ranking_weight']
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
        if 'top_k' in kwargs:
            top_k = kwargs['top_k']

        self.loss_type = loss_type
        self.ranking_weight = ranking_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
        if loss_type == 'listnet':
            self.ranking_loss = ListNetLoss(temperature=temperature)
        elif loss_type == 'listmle':
            self.ranking_loss = ListMLELoss(top_k=top_k)
        else:
            self.ranking_loss = None
    
    def forward(self, scores, labels):
        """
        Args:
            scores: (B, num_items)
            labels: (B,) 或 (B, 1)
        """
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        # CrossEntropy loss
        ce = self.ce_loss(scores, labels)
        
        # Ranking loss
        if self.ranking_loss is not None and self.ranking_weight > 0:
            ranking = self.ranking_loss(scores, labels)
            total = ce + self.ranking_weight * ranking
        else:
            total = ce
            ranking = torch.tensor(0.0)
        
        return total, ce, ranking


