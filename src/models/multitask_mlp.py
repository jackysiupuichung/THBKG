import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskClinicalMLP(nn.Module):
    """
    Multi-Task MLP Decoder for Clinical Trial Phase Prediction.
    
    Predicts the probability of achieving each of 4 max clinical trial phases/outcomes:
    - pos
    - unmet
    - adv
    - op
    
    Inputs: Concatenated node embeddings [h_u || h_v]
    Outputs: 4 independent probability scores in [0, 1]
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task head: 4 outputs
        # We output LOGITS. Sigmoid is applied in loss (BCEWithLogits) or during inference.
        self.head = nn.Linear(hidden_dim // 2, 4)
        
    def forward(self, disease_emb, target_emb):
        """
        Args:
            disease_emb: [batch_size, dim]
            target_emb: [batch_size, dim]
            
        Returns:
            Dictionary of logits for each task {'pos', 'unmet', 'adv', 'op'}
        """
        # Concatenate embeddings
        x = torch.cat([disease_emb, target_emb], dim=-1)
        
        # Shared encoder
        feat = self.net(x)
        
        # Multi-task prediction (logits)
        logits = self.head(feat)
        
        return {
            'pos': logits[:, 0],
            'unmet': logits[:, 1],
            'adv': logits[:, 2],
            'op': logits[:, 3]
        }

class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss (BCEWithLogits or Huber).
    """
    def __init__(self, weights=None, use_huber=False):
        super().__init__()
        self.weights = weights if weights else {'pos': 1.0, 'unmet': 1.0, 'adv': 1.0, 'op': 1.0}
        self.use_huber = use_huber
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict of logits
            targets: Dict of float targets
            
        Returns:
            total_loss, dict_of_task_losses
        """
        total_loss = 0
        task_losses = {}
        
        for task in ['pos', 'unmet', 'adv', 'op']:
            pred = predictions[task]
            target = targets[task]
            
            if self.use_huber:
                # For Huber, we need probabilities (sigmoid applied)
                # target should be 0.0 or 1.0
                loss = F.huber_loss(torch.sigmoid(pred), target)
            else:
                # Standard BCE (expects logits)
                loss = F.binary_cross_entropy_with_logits(pred, target)
                
            w_loss = loss * self.weights.get(task, 1.0)
            total_loss += w_loss
            task_losses[task] = loss.item() # Log raw loss
            
        return total_loss, task_losses

def compute_task_weights(labels_df, inverse_freq=True):
    """
    Compute task weights based on label frequencies.
    """
    tasks = ['y_pos', 'y_unmet', 'y_adv', 'y_op']
    weights = {}
    
    total = len(labels_df)
    
    for task in tasks:
        task_key = task.replace('y_', '')
        
        if inverse_freq:
            pos_count = labels_df[task].sum()
            # Simple inverse frequency
            if pos_count > 0:
                w = total / (2 * pos_count) # Heuristic for balancing
            else:
                w = 1.0
            weights[task_key] = w
        else:
            weights[task_key] = 1.0
            
    return weights
