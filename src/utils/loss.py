import torch
import torch.nn.functional as F

def mse_loss(scores, labels):
    """Pointwise MSE loss."""
    return F.mse_loss(scores, labels.float())

def bpr_loss(user_emb, pos_emb, neg_emb):
    """Pairwise BPR loss."""
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores + 1e-8)))

def listwise_loss(scores, labels):
    """
    ListNet-style listwise loss:
    - scores: [batch_size, num_items]
    - labels: [batch_size, num_items]
    """
    # softmax over labels and predictions
    label_prob = F.softmax(labels, dim=1)
    pred_prob = F.log_softmax(scores, dim=1)
    return -torch.mean(torch.sum(label_prob * pred_prob, dim=1))
