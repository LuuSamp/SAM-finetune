import torch
import os
import sys
from pathlib import Path

from config import *
from dataloaders import create_dataloaders
from sam_model import train_sam_lora

import urllib.request
from tqdm import tqdm


def download_sam_checkpoint(model_type='vit_b', checkpoint_dir='./'):
    """Download SAM checkpoint if not present."""
    checkpoint_urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    }
    
    checkpoint_names = {
        'vit_b': 'sam_vit_b_01ec64.pth',
    }
    
    checkpoint_path = Path(checkpoint_dir) / checkpoint_names[model_type]
    
    if checkpoint_path.exists():
        print(f" SAM checkpoint found: {checkpoint_path}")
        return str(checkpoint_path)
    
    print(f"Downloading SAM {model_type} checkpoint (~375MB)...")
    
    try:
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                desc=f'sam_{model_type}') as t:
            urllib.request.urlretrieve(
                checkpoint_urls[model_type], 
                filename=checkpoint_path,
                reporthook=t.update_to
            )
        
        print(f"✓ Downloaded: {checkpoint_path}")
        return str(checkpoint_path)
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print(f"\nManual download: {checkpoint_urls[model_type]}")
        sys.exit(1)


def main():
    """Main training function for low VRAM."""
    print("="*70)
    print(" SAM LoRA Fine-tuning - LOW VRAM MODE")
    print("="*70)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("  No GPU detected!")
        print("  SAM training requires GPU (even with optimizations)")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.2f} GB")
    print()
    
    if gpu_memory < 3.5:
        print("   WARNING: Less than 3.5GB VRAM detected")
        print("   Training may still fail. Consider:")
        print("   1. Close all other programs")
        print("   2. Reduce dataset size")
        print("   3. Use cloud GPU (Google Colab)")
        print()
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            return
    
    # Download SAM
    print("Checking SAM checkpoint...")
    sam_checkpoint = download_sam_checkpoint(MODEL_TYPE)
    print()
    
    # Create dataloaders
    print("Preparing dataset...")
    train_dataloader, val_dataloader = create_dataloaders(
        root_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        download=True
    )
    print()
    
    # Display config
    print("="*70)
    print("LOW VRAM Training Configuration:")
    print("="*70)
    print(f"  GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}x")
    print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  LoRA Rank: {LORA_RANK} (reduced)")
    print(f"  Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"  Mixed Precision: {USE_AMP}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("="*70)
    print()
    print("  Expected time per epoch: 30-60 minutes")
    print("  Expected VRAM usage: 3.2-3.5 GB")
    print()
    
    response = input("Start training? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return
    
    print()
    
    # Train
    try:
        model = train_sam_lora(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            sam_checkpoint=sam_checkpoint,
            model_type=MODEL_TYPE,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=DEVICE,
            use_checkpointing=USE_GRADIENT_CHECKPOINTING,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            save_dir=SAVE_DIR
        )
        
        print()
        print("="*70)
        print("✓ Training completed!")
        print(f"✓ Model saved: {SAVE_DIR}/best_model.pth")
        print("="*70)
        
    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "="*70)
        print(" CUDA OUT OF MEMORY")
        print("="*70)
        print("\nYour GPU does not have enough VRAM for SAM training.")
        print("\nOptions:")
        print("  1. Use Google Colab (free GPU with 15GB VRAM)")
        print("     https://colab.research.google.com")
        print("  2. Use a smaller dataset (reduce number of images)")
        print("  3. Try a cloud GPU service (vast.ai, runpod.io)")
        print("\nSAM is a very large model and needs at least 6GB VRAM")
        print("even with all optimizations enabled.")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()