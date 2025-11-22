"""
Configuration settings for SAM fine-tuning on Kvasir-SEG dataset.
"""

import torch
from pathlib import Path
from .setup import setup_complete_environment


class Config:
    """Configuration class for SAM fine-tuning."""
    
    def __init__(self):
        """Initialize configuration and setup environment."""
        # Setup environment and get paths
        self.DATA_DIR, self.CHECKPOINT_PATH = setup_complete_environment()
        
        # Output directory
        self.OUTPUT_DIR = "./outputs"
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Model settings
        self.MODEL_TYPE = "vit_b"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Image processing
        self.IMAGE_SIZE = 1024  # SAM requires 1024x1024 images
        
        # Training parameters
        self.BATCH_SIZE = 4
        self.NUM_EPOCHS = 15
        self.LEARNING_RATE = 1e-5
        self.WEIGHT_DECAY = 1e-4
        
        # Data splits
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        
        # Freeze settings
        self.FREEZE_IMAGE_ENCODER = True
        self.FREEZE_PROMPT_ENCODER = True
        
        # Augmentation
        self.USE_AUGMENTATION = True
        
        # Prompt settings
        self.USE_BOX_PROMPTS = True  # Enable bounding box prompts
        self.USE_BOTH_PROMPTS = True  # Use both points and boxes simultaneously
        self.PROMPT_MIX_RATIO = 0.5  # Ratio of boxes vs points when USE_BOTH_PROMPTS=False (0.0 = only points, 1.0 = only boxes)
        
        # Checkpoint settings
        self.CHECKPOINT_FREQUENCY = 5  # Save checkpoint every N epochs
    
    def print_config(self):
        """Print configuration settings."""
        print("Configuration Settings:")
        print(f"  Device: {self.DEVICE}")
        print(f"  Image Size: {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"  Batch Size: {self.BATCH_SIZE}")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Train/Val/Test Split: {self.TRAIN_SPLIT}/{self.VAL_SPLIT}/{self.TEST_SPLIT}")
        print(f"  Prompt Strategy: {'Both (points + boxes)' if self.USE_BOTH_PROMPTS and self.USE_BOX_PROMPTS else ('Boxes only' if self.USE_BOX_PROMPTS else 'Points only')}")
        if not self.USE_BOTH_PROMPTS:
            print(f"  Prompt Mix Ratio: {self.PROMPT_MIX_RATIO}")
        print(f"  Data Directory: {self.DATA_DIR}")
        print(f"  Checkpoint Path: {self.CHECKPOINT_PATH}")