"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2
import random


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
    # Convert history values to native Python types (numpy types are not JSON serializable)
    history_serializable = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_iou': [float(x) for x in history['val_iou']],
        'val_dice': [float(x) for x in history['val_dice']]
    }
    
    results = {
        'config': {
            'model_type': config.MODEL_TYPE,
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
        },
        'history': history_serializable,
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'iou': float(test_metrics['iou']),
            'dice': float(test_metrics['dice'])
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_dir}/results.json")


def compare_models(model_original, model_finetuned, test_loader, show_example=True):
    """
    Compare original and fine-tuned models on test set.
    
    Args:
        model_original: SAMFineTuner instance with original (pre-trained) model
        model_finetuned: SAMFineTuner instance with fine-tuned model
        test_loader: Test data loader
        show_example: Whether to show a random example prediction
    """
    print("=" * 60)
    print("MODEL COMPARISON: Original vs Fine-tuned")
    print("=" * 60)
    
    # Evaluate both models
    print("\nEvaluating Original Model...")
    orig_loss, orig_iou, orig_dice = model_original.validate(test_loader)
    
    print("\nEvaluating Fine-tuned Model...")
    ft_loss, ft_iou, ft_dice = model_finetuned.validate(test_loader)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Loss':<15} {orig_loss:<15.4f} {ft_loss:<15.4f} {ft_loss - orig_loss:>+14.4f}")
    print(f"{'IoU':<15} {orig_iou:<15.4f} {ft_iou:<15.4f} {ft_iou - orig_iou:>+14.4f}")
    print(f"{'Dice':<15} {orig_dice:<15.4f} {ft_dice:<15.4f} {ft_dice - orig_dice:>+14.4f}")
    print("=" * 60)
    
    # Show random example
    if show_example:
        print("\nRandom Example Prediction:")
        print("-" * 60)
        
        # Get a random batch
        test_list = list(test_loader)
        random_batch = random.choice(test_list)
        
        # Get predictions from both models
        with torch.no_grad():
            model_original.sam.eval()
            model_finetuned.sam.eval()
            
            images = random_batch['image'].to(model_original.device)
            masks_gt = random_batch['mask'].to(model_original.device)
            point_coords = random_batch['point_coords'].to(model_original.device)
            point_labels = random_batch['point_labels'].to(model_original.device)
            box_coords = random_batch.get('box_coords', None)
            if box_coords is not None:
                box_coords = box_coords.to(model_original.device)
            
            # Original model prediction
            orig_image_embedding = model_original.sam.image_encoder(images[0:1])
            
            # Determine prompt strategy for original model
            if model_original.config.USE_BOTH_PROMPTS and model_original.config.USE_BOX_PROMPTS and box_coords is not None:
                orig_points_input = point_coords[0:1]
                orig_labels_input = point_labels[0:1]
                orig_boxes_input = box_coords[0:1]
            elif model_original.config.USE_BOX_PROMPTS and box_coords is not None:
                orig_boxes_input = box_coords[0:1]
                orig_points_input = None
                orig_labels_input = None
            else:
                orig_points_input = point_coords[0:1]
                orig_labels_input = point_labels[0:1]
                orig_boxes_input = None
            
            orig_sparse_embeddings, orig_dense_embeddings = model_original.sam.prompt_encoder(
                points=(orig_points_input, orig_labels_input) if orig_points_input is not None else None,
                boxes=orig_boxes_input,
                masks=None
            )
            
            orig_low_res_masks, _ = model_original.sam.mask_decoder(
                image_embeddings=orig_image_embedding,
                image_pe=model_original.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=orig_sparse_embeddings,
                dense_prompt_embeddings=orig_dense_embeddings,
                multimask_output=False
            )
            
            orig_upscaled_masks = torch.nn.functional.interpolate(
                orig_low_res_masks,
                size=(model_original.config.IMAGE_SIZE, model_original.config.IMAGE_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            orig_pred_mask = torch.sigmoid(orig_upscaled_masks[0, 0]).cpu().numpy()
            
            # Fine-tuned model prediction
            ft_image_embedding = model_finetuned.sam.image_encoder(images[0:1])
            
            # Determine prompt strategy for fine-tuned model
            if model_finetuned.config.USE_BOTH_PROMPTS and model_finetuned.config.USE_BOX_PROMPTS and box_coords is not None:
                ft_points_input = point_coords[0:1]
                ft_labels_input = point_labels[0:1]
                ft_boxes_input = box_coords[0:1]
            elif model_finetuned.config.USE_BOX_PROMPTS and box_coords is not None:
                ft_boxes_input = box_coords[0:1]
                ft_points_input = None
                ft_labels_input = None
            else:
                ft_points_input = point_coords[0:1]
                ft_labels_input = point_labels[0:1]
                ft_boxes_input = None
            
            ft_sparse_embeddings, ft_dense_embeddings = model_finetuned.sam.prompt_encoder(
                points=(ft_points_input, ft_labels_input) if ft_points_input is not None else None,
                boxes=ft_boxes_input,
                masks=None
            )
            
            ft_low_res_masks, _ = model_finetuned.sam.mask_decoder(
                image_embeddings=ft_image_embedding,
                image_pe=model_finetuned.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=ft_sparse_embeddings,
                dense_prompt_embeddings=ft_dense_embeddings,
                multimask_output=False
            )
            
            ft_upscaled_masks = torch.nn.functional.interpolate(
                ft_low_res_masks,
                size=(model_finetuned.config.IMAGE_SIZE, model_finetuned.config.IMAGE_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            ft_pred_mask = torch.sigmoid(ft_upscaled_masks[0, 0]).cpu().numpy()
            gt_mask = masks_gt[0, 0].cpu().numpy()
            
            # Denormalize image
            image = images[0].cpu().permute(1, 2, 0).numpy()
            image = image * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            image = np.clip(image / 255.0, 0, 1)
            
            # Calculate metrics
            orig_iou_ex = calculate_iou(orig_pred_mask, gt_mask)
            orig_dice_ex = calculate_dice(orig_pred_mask, gt_mask)
            ft_iou_ex = calculate_iou(ft_pred_mask, gt_mask)
            ft_dice_ex = calculate_dice(ft_pred_mask, gt_mask)
            
            # Display results
            print(f"Original - IoU: {orig_iou_ex:.4f} | Dice: {orig_dice_ex:.4f}")
            print(f"Fine-tuned - IoU: {ft_iou_ex:.4f} | Dice: {ft_dice_ex:.4f}")
            print(f"Improvement - IoU: {ft_iou_ex - orig_iou_ex:+.4f} | Dice: {ft_dice_ex - orig_dice_ex:+.4f}")
            
            # Show visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(orig_pred_mask, cmap='gray')
            axes[2].set_title(f"Original Model\nIoU: {orig_iou_ex:.3f} | Dice: {orig_dice_ex:.3f}", 
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(ft_pred_mask, cmap='gray')
            axes[3].set_title(f"Fine-tuned Model\nIoU: {ft_iou_ex:.3f} | Dice: {ft_dice_ex:.3f}", 
                            fontsize=12, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    return {
        'original': {'loss': orig_loss, 'iou': orig_iou, 'dice': orig_dice},
        'finetuned': {'loss': ft_loss, 'iou': ft_iou, 'dice': ft_dice}
    }