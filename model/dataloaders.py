"""
Data loading and preprocessing utilities for Kvasir-SEG dataset.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import albumentations as A


class SAMDataset(Dataset):
    """
    Dataset class with proper preprocessing for SAM model.
    
    Args:
        image_paths (list): List of paths to input images
        mask_paths (list): List of paths to mask images
        transform (callable, optional): Albumentations transform
        target_size (int): Target image size (default: 1024)
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, target_size=1024):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_image(self, image):
        """
        Preprocess image to be compatible with SAM requirements.
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            tuple: (preprocessed_image, (original_height, original_width))
        """
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        image = image.astype(np.float32)
        image = (image - self.pixel_mean) / self.pixel_std
        
        return image, (new_h, new_w)
    
    def preprocess_mask(self, mask, target_h, target_w):
        """
        Preprocess mask to match image dimensions.
        
        Args:
            mask (numpy.ndarray): Input mask
            target_h (int): Target height
            target_w (int): Target width
            
        Returns:
            numpy.ndarray: Preprocessed mask
        """
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        pad_h = self.target_size - target_h
        pad_w = self.target_size - target_w
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        return mask
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            dict: Contains image, mask, point coordinates, and labels
        """
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image, (new_h, new_w) = self.preprocess_image(image)
        mask = self.preprocess_mask(mask, new_h, new_w)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            center = coords.mean(axis=0).astype(int)
            point_coords = np.array([[center[1], center[0]]])
            point_labels = np.array([1])
        else:
            point_coords = np.array([[self.target_size//2, self.target_size//2]])
            point_labels = np.array([0])
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'point_coords': torch.from_numpy(point_coords).float(),
            'point_labels': torch.from_numpy(point_labels),
            'original_size': (new_h, new_w)
        }


def get_transforms(is_train=True):
    """
    Get data augmentation transforms.
    
    Args:
        is_train (bool): Whether to return training transforms
        
    Returns:
        albumentations.Compose: Composition of transforms
    """
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ])
    return None


def prepare_data_splits(data_dir, train_split=0.7, val_split=0.15):
    """
    Prepare train/validation/test splits.
    
    Args:
        data_dir (str): Path to data directory
        train_split (float): Proportion of training data
        val_split (float): Proportion of validation data
        
    Returns:
        dict: Dictionary containing splits for train, val, and test
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    mask_dir = data_path / "masks"
    
    # Check if directories exist
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    valid_pairs = []
    for img in image_files:
        mask_path = mask_dir / f"{img.stem}.png"
        if not mask_path.exists():
            # Try with different extensions
            mask_path_jpg = mask_dir / f"{img.stem}.jpg"
            if mask_path_jpg.exists():
                mask_path = mask_path_jpg
        
        if mask_path.exists():
            valid_pairs.append((img, mask_path))
    
    if len(valid_pairs) == 0:
        raise ValueError(f"No valid image-mask pairs found in {data_dir}")
    
    print(f"Dataset Statistics:")
    print(f"  Total valid images: {len(valid_pairs)}")
    
    images, masks = zip(*valid_pairs)
    
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        images, masks, test_size=(1-train_split), random_state=42
    )
    
    val_size = val_split / (val_split + (1 - train_split - val_split))
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks, test_size=(1-val_size), random_state=42
    )
    
    print(f"  Training: {len(train_imgs)}")
    print(f"  Validation: {len(val_imgs)}")
    print(f"  Test: {len(test_imgs)}")
    
    return {
        'train': (train_imgs, train_msks),
        'val': (val_imgs, val_msks),
        'test': (test_imgs, test_msks)
    }


def create_data_loaders(config, splits):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        splits (dict): Data splits from prepare_data_splits
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_images, train_masks = splits['train']
    val_images, val_masks = splits['val']
    test_images, test_masks = splits['test']
    
    train_dataset = SAMDataset(
        train_images, train_masks,
        get_transforms(is_train=True) if config.USE_AUGMENTATION else None,
        target_size=config.IMAGE_SIZE
    )
    val_dataset = SAMDataset(val_images, val_masks, target_size=config.IMAGE_SIZE)
    test_dataset = SAMDataset(test_images, test_masks, target_size=config.IMAGE_SIZE)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader