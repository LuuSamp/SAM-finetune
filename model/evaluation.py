"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred_mask (numpy.ndarray): Predicted mask
        gt_mask (numpy.ndarray): Ground truth mask
        threshold (float): Threshold for binarization
        
    Returns:
        float: IoU score
    """
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    union = ((pred_binary + gt_binary) > 0).sum()
    return intersection / union if union > 0 else 1.0


def calculate_dice(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Dice Coefficient.
    
    Args:
        pred_mask (numpy.ndarray): Predicted mask
        gt_mask (numpy.ndarray): Ground truth mask
        threshold (float): Threshold for binarization
        
    Returns:
        float: Dice score
    """
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    return (2 * intersection) / total if total > 0 else 1.0


def plot_training_curves(history, save_path):
    """
    Plot training and validation curves.
    
    Args:
        history (dict): Training history
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['val_iou'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['val_dice'], 'm-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Dice Coefficient', fontsize=12)
    axes[2].set_title('Validation Dice', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()


def visualize_predictions(model, test_loader, save_path, n_samples=6):
    """
    Visualize model predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        save_path (str): Path to save the visualization
        n_samples (int): Number of samples to visualize
    """
    model.sam.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= n_samples:
                break
            
            images = batch['image'].to(model.device)
            masks_gt = batch['mask'].to(model.device)
            point_coords = batch['point_coords'].to(model.device)
            point_labels = batch['point_labels'].to(model.device)
            
            image_embedding = model.sam.image_encoder(images[0:1])
            sparse_embeddings, dense_embeddings = model.sam.prompt_encoder(
                points=(point_coords[0:1], point_labels[0:1]), 
                boxes=None, 
                masks=None
            )
            low_res_masks, _ = model.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            upscaled_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(model.config.IMAGE_SIZE, model.config.IMAGE_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            pred_mask = torch.sigmoid(upscaled_masks[0, 0]).cpu().numpy()
            gt_mask = masks_gt[0, 0].cpu().numpy()
            
            image = images[0].cpu().permute(1, 2, 0).numpy()
            image = image * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            image = np.clip(image / 255.0, 0, 1)
            
            iou = calculate_iou(pred_mask, gt_mask)
            dice = calculate_dice(pred_mask, gt_mask)
            
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Original Image", fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(gt_mask, cmap='gray')
            axes[idx, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title(f"Prediction\nIoU: {iou:.3f} | Dice: {dice:.3f}", 
                                  fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions visualization saved to: {save_path}")
    plt.show()


def save_results(config, history, test_metrics, output_dir):
    """
    Save training results to JSON file.
    
    Args:
        config: Configuration object
        history (dict): Training history
        test_metrics (dict): Test set metrics
        output_dir (str): Output directory
    """
    results = {
        'config': {
            'model_type': config.MODEL_TYPE,
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
        },
        'history': history,
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'iou': float(test_metrics['iou']),
            'dice': float(test_metrics['dice'])
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_dir}/results.json")