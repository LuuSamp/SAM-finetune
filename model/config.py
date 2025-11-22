import torch
import os

# Enable memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Dataset settings
DATA_ROOT = './data/oxford_pets'
BATCH_SIZE = 1  
IMAGE_SIZE = 1024  
NUM_WORKERS = 0  

# Model settings - Use smallest model
MODEL_TYPE = 'vit_b'  # vit_b is smallest (vit_h would use 3x more memory)
SAM_CHECKPOINT = './sam_vit_b_01ec64.pth'

# LoRA settings 
LORA_RANK = 4  
LORA_ALPHA = 1

# Training settings
NUM_EPOCHS = 5  
LEARNING_RATE = 5e-5  
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True
USE_AMP = False  
GRADIENT_ACCUMULATION_STEPS = 4  

# Checkpoint settings
SAVE_DIR = './checkpoints'

print("="*70)
print(" LOW VRAM Configuration Loaded")
print("="*70)
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_TYPE}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"  Mixed Precision: {USE_AMP}")
print("="*70)
