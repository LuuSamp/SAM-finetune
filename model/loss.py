"""
Custom loss functions for SAM fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Calculate Dice Loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            torch.Tensor: Dice loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha (float): Weighting factor for rare classes [0,1]
        gamma (float): Focusing parameter (>=0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Calculate Focal Loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Get probabilities
        p_t = torch.exp(-bce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss (Dice Loss & Focal Loss).
    
    Args:
        dice_weight (float): Weight for Dice loss component
        focal_weight (float): Weight for Focal loss component
        focal_alpha (float): Alpha parameter for Focal loss
        focal_gamma (float): Gamma parameter for Focal loss
    """
    
    def __init__(self, dice_weight: float = 20/21, focal_weight: float = 1/21,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            torch.Tensor: Combined loss value
        """
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal
