"""
SAM (Segment Anything Model) fine-tuning implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from segment_anything import sam_model_registry


class SAMFineTuner:
    """
    Fine-tuner for SAM model.
    
    Args:
        config: Configuration object
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        print(f"Loading SAM {config.MODEL_TYPE}...")
        self.sam = sam_model_registry[config.MODEL_TYPE](checkpoint=config.CHECKPOINT_PATH)
        self.sam.to(self.device)
        
        self._freeze_layers()
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.sam.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_iou': [], 
            'val_dice': []
        }
    
    def _freeze_layers(self):
        """Freeze specified layers according to configuration."""
        if self.config.FREEZE_IMAGE_ENCODER:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        if self.config.FREEZE_PROMPT_ENCODER:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
        
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.sam.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks_gt = batch['mask'].to(self.device)
            point_coords = batch['point_coords'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            
            batch_size = images.shape[0]
            self.optimizer.zero_grad()
            
            batch_loss = 0
            for i in range(batch_size):
                with torch.no_grad():
                    image_embedding = self.sam.image_encoder(images[i:i+1])
                
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(point_coords[i:i+1], point_labels[i:i+1]),
                    boxes=None,
                    masks=None
                )
                
                low_res_masks, _ = self.sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                
                upscaled_masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                    mode='bilinear',
                    align_corners=False
                )
                
                loss = self.criterion(upscaled_masks, masks_gt[i:i+1])
                batch_loss += loss
            
            avg_loss = batch_loss / batch_size
            avg_loss.backward()
            self.optimizer.step()
            
            total_loss += avg_loss.item()
            pbar.set_postfix({'loss': f'{avg_loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            tuple: (average_loss, average_iou, average_dice)
        """
        self.sam.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        n_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks_gt = batch['mask'].to(self.device)
                point_coords = batch['point_coords'].to(self.device)
                point_labels = batch['point_labels'].to(self.device)
                
                batch_size = images.shape[0]
                
                for i in range(batch_size):
                    image_embedding = self.sam.image_encoder(images[i:i+1])
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=(point_coords[i:i+1], point_labels[i:i+1]),
                        boxes=None,
                        masks=None
                    )
                    
                    low_res_masks, _ = self.sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    
                    upscaled_masks = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    loss = self.criterion(upscaled_masks, masks_gt[i:i+1])
                    total_loss += loss.item()
                    
                    pred_mask = torch.sigmoid(upscaled_masks[0, 0]).cpu().numpy()
                    gt_mask = masks_gt[i, 0].cpu().numpy()
                    
                    from evaluation import calculate_iou, calculate_dice
                    total_iou += calculate_iou(pred_mask, gt_mask)
                    total_dice += calculate_dice(pred_mask, gt_mask)
                    n_samples += 1
        
        return (total_loss / n_samples, total_iou / n_samples, total_dice / n_samples)
    
    def fit(self, train_loader, val_loader):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            dict: Training history
        """
        print("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_iou, val_dice = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_dice'].append(val_dice)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.sam.state_dict(), f"{self.config.OUTPUT_DIR}/best_model.pth")
                print("  Model saved!")
            print()
        
        return self.history