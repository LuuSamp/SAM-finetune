from abc import ABC, abstractmethod
import torch

class Loss(ABC):
    @abstractmethod
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Call 

        Args:
            inputs (torch.Tensor): _description_
            targets (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        pass

class FocalLoss(Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()
    
class BCELoss(Loss):
    def __init__(self):
        pass

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        return bce_loss