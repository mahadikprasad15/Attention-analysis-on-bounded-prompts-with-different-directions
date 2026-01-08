import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

class LinearProbe:
    """Linear probe for binary classification"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.direction = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit the probe
        
        Returns:
            Training accuracy
        """
        self.classifier.fit(X, y)
        self.direction = torch.tensor(
            self.classifier.coef_[0], 
            dtype=torch.float32
        )
        self.is_fitted = True
        return self.classifier.score(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions"""
        if not self.is_fitted:
            raise ValueError("Probe not fitted yet!")
        return self.classifier.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on test set"""
        if not self.is_fitted:
            raise ValueError("Probe not fitted yet!")
        return self.classifier.score(X, y)
    
    def get_continuous_score(self, activation: torch.Tensor) -> float:
        """
        Get continuous score (projection onto direction)
        
        Args:
            activation: [hidden_dim] tensor
        
        Returns:
            Scalar score
        """
        if not self.is_fitted:
            raise ValueError("Probe not fitted yet!")
        return (activation @ self.direction).item()
