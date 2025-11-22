import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc

from segment_anything import sam_model_registry
from tqdm import tqdm
import numpy as np
import os

from config import *
from loss import *


class LoRALayer(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: int = 1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRASAM(nn.Module):
    def __init__(self, sam_model, lora_rank: int = 4, lora_alpha: int = 1, 
                 use_checkpointing: bool = False):
        super().__init__()
        self.sam = sam_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_checkpointing = use_checkpointing
        
        # Freeze ALL SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False
        
        # Apply gradient checkpointing if enabled
        if use_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Apply LoRA
        self._inject_lora_to_attention()
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        print("  Enabling gradient checkpointing...")
        for block in self.sam.image_encoder.blocks:
            block.use_checkpoint = True
    
    def _inject_lora_to_attention(self):
        self.lora_layers = nn.ModuleDict()
        
        for block_idx, block in enumerate(self.sam.image_encoder.blocks):
            attn = block.attn
            dim = attn.qkv.in_features
            
            self.lora_layers[f'block{block_idx}_qkv'] = LoRALayer(
                dim, dim * 3, self.lora_rank, self.lora_alpha
            )
            self.lora_layers[f'block{block_idx}_proj'] = LoRALayer(
                dim, dim, self.lora_rank, self.lora_alpha
            )
    
    def forward(self, images, target_size=(256, 256)):
        with torch.enable_grad():
            self._apply_lora_hooks()
            image_embeddings = self.sam.image_encoder(images)
            self._remove_hooks()
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        masks = F.interpolate(
            low_res_masks,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return masks, iou_predictions
    
    def _apply_lora_hooks(self):
        self.hooks = []
        
        for block_idx, block in enumerate(self.sam.image_encoder.blocks):
            def qkv_hook(module, input, output, block_idx=block_idx):
                lora_layer = self.lora_layers[f'block{block_idx}_qkv']
                return output + lora_layer(input[0])
            
            def proj_hook(module, input, output, block_idx=block_idx):
                lora_layer = self.lora_layers[f'block{block_idx}_proj']
                return output + lora_layer(input[0])
            
            h1 = block.attn.qkv.register_forward_hook(qkv_hook)
            h2 = block.attn.proj.register_forward_hook(proj_hook)
            self.hooks.extend([h1, h2])
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def calculate_iou(predictions, targets, threshold: float = 0.5):
    predictions = (torch.sigmoid(predictions) > threshold).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_epoch(model, dataloader, optimizer, criterion, device, 
                gradient_accumulation_steps=1):
    """Memory-optimized training with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions, _ = model(images, target_size=masks.shape[-2:])
        loss = criterion(predictions, masks)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Cleanup memory after optimizer step
            if batch_idx % 10 == 0:
                cleanup_memory()
        
        # Metrics
        with torch.no_grad():
            iou = calculate_iou(predictions.detach(), masks.detach())
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'iou': f'{iou:.4f}',
            'mem': f'{torch.cuda.memory_allocated() / 1e9:.2f}GB'
        })
        
        # Clean up batch
        del images, masks, predictions, loss
    
    # Final step if there are remaining gradients
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    cleanup_memory()
    
    return avg_loss, avg_iou


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        predictions, _ = model(images, target_size=masks.shape[-2:])
        loss = criterion(predictions, masks)
        
        iou = calculate_iou(predictions, masks)
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })
        
        # Clean up
        del images, masks, predictions, loss
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    cleanup_memory()
    
    return avg_loss, avg_iou


def train_sam_lora(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    sam_checkpoint: str,
    model_type: str = 'vit_b',
    lora_rank: int = 4,
    lora_alpha: int = 1,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    use_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    save_dir: str = './checkpoints'
):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Clear memory before starting
    cleanup_memory()
    
    # Load SAM
    print(f'Loading SAM ({model_type})...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    
    # Apply LoRA with checkpointing
    print(f'Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...')
    model = LoRASAM(sam, lora_rank=lora_rank, lora_alpha=lora_alpha,
                    use_checkpointing=use_checkpointing)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')
    
    # Print memory stats
    if torch.cuda.is_available():
        print(f'GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        print(f'GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Training
    best_val_iou = 0.0
    
    print('\nStarting training...')
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_dataloader, optimizer, criterion, device,
            gradient_accumulation_steps
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_dataloader, criterion, device
        )
        
        # Update lr
        scheduler.step()
        
        # Results
        print(f'\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            
            # Save only LoRA weights to save disk space
            lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
            torch.save({
                'epoch': epoch,
                'lora_state_dict': lora_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
            }, checkpoint_path)
            print(f'Best model saved! (IoU: {val_iou:.4f})')
        
        cleanup_memory()
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best validation IoU: {best_val_iou:.4f}')
    print(f'{"="*60}')
    
    return model


# Usage
# if __name__ == '__main__':
#     from dataset_loader import create_dataloaders
    
#     train_dataloader, val_dataloader = create_dataloaders(
#         batch_size=BATCH_SIZE,
#         image_size=IMAGE_SIZE,
#         num_workers=NUM_WORKERS
#     )
    
#     model = train_sam_lora(
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         sam_checkpoint=SAM_CHECKPOINT,
#         model_type=MODEL_TYPE,
#         lora_rank=LORA_RANK,
#         lora_alpha=LORA_ALPHA,
#         num_epochs=NUM_EPOCHS,
#         learning_rate=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#         device=DEVICE,
#         use_checkpointing=USE_GRADIENT_CHECKPOINTING,
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         save_dir=SAVE_DIR
#     )