import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AttentionProbe(nn.Module):
    """
    Probe that learns to attend to specific tokens in the sequence 
    before making a classification decision.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        
        # Learnable attention vector (query)
        # We compute scores = x @ attn_vector
        self.attn_vector = nn.Parameter(torch.randn(input_dim) * 0.02)
        
        # Classifier on top of aggregated embedding
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Binary classification logits
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] (1 for valid, 0 for padding)
            
        Returns:
            logits: [batch, 1]
            weights: [batch, seq_len] (Attention weights)
        """
        # 1. Compute attention scores
        # x: [B, S, D], vector: [D] -> scores: [B, S]
        
        # Ensure inputs match weight dtype (e.g. if x is Half and weights are Float)
        if x.dtype != self.attn_vector.dtype:
            x = x.to(self.attn_vector.dtype)
            
        scores = torch.matmul(x, self.attn_vector)
        
        # 2. Masking
        if mask is not None:
            # Set padded positions to -inf
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax to get weights
        weights = F.softmax(scores, dim=1) # [B, S]
        
        # 4. Aggregate
        # weights: [B, S] -> [B, 1, S]
        # x: [B, S, D]
        # bmm: [B, 1, S] @ [B, S, D] -> [B, 1, D]
        aggregated = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # [B, D]
        
        # 5. Classify
        logits = self.classifier(aggregated)
        
        return logits, weights
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits, _ = self.forward(x, mask)
        return torch.sigmoid(logits) > 0.5
