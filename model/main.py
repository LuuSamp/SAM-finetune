"""
Main script for SAM fine-tuning on Kvasir-SEG dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataloaders import prepare_data_splits, create_data_loaders
from sam_model import SAMFineTuner
from evaluation import plot_training_curves, visualize_predictions, save_results


def main():
    """
    Main training pipeline for SAM fine-tuning.
    """
    print("=" * 60)
    print("SAM FINE-TUNING - POLYP SEGMENTATION (KVASIR-SEG)")
    print("=" * 60)
    
    # Initialize configuration (this will setup the environment)
    config = Config()
    config.print_config()
    
    # Prepare data splits
    splits = prepare_data_splits(config.DATA_DIR, config.TRAIN_SPLIT, config.VAL_SPLIT)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, splits)
    
    # Initialize and train model
    trainer = SAMFineTuner(config)
    history = trainer.fit(train_loader, val_loader)
    
    # Evaluate on test set
    print("Final Evaluation on Test Set:")
    test_loss, test_iou, test_dice = trainer.validate(test_loader)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test IoU: {test_iou:.4f}")
    print(f"  Test Dice: {test_dice:.4f}")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_training_curves(history, f"{config.OUTPUT_DIR}/training_curves.png")
    visualize_predictions(trainer, test_loader, f"{config.OUTPUT_DIR}/predictions.png", n_samples=6)
    
    # Save results
    test_metrics = {
        'loss': test_loss,
        'iou': test_iou,
        'dice': test_dice
    }
    save_results(config, history, test_metrics, config.OUTPUT_DIR)
    
    print(f"Training completed!")
    print(f"Results saved in: {config.OUTPUT_DIR}/")
    print(f"  - best_model.pth")
    print(f"  - training_curves.png")
    print(f"  - predictions.png")
    print(f"  - results.json")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()   