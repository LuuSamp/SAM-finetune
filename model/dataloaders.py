import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset for segmentation.
    Automatically downloads and prepares the dataset.
    """
    
    IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    def __init__(self, root_dir='./data/oxford_pets', split='train', 
                 image_size=1024, download=True, train_split=0.8):
        """
        Args:
            root_dir: Root directory to store dataset
            split: 'train' or 'val'
            image_size: Size to resize images (SAM REQUIRES 1024x1024)
            download: Whether to download if not present
            train_split: Proportion of data for training
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'annotations' / 'trimaps'
        
        # Download if needed
        if download and not self._check_exists():
            self.download()
        
        # Get file lists
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Split into train/val
        np.random.seed(42)
        indices = np.random.permutation(len(self.image_files))
        split_idx = int(len(indices) * train_split)
        
        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        
        self.image_files = [self.image_files[i] for i in indices]
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
        
        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def _check_exists(self):
        """Check if dataset is already downloaded."""
        return (self.images_dir.exists() and 
                self.masks_dir.exists() and 
                len(list(self.images_dir.glob('*.jpg'))) > 0)
    
    def download(self):
        """Download and extract the dataset."""
        print("Downloading Oxford-IIIT Pet Dataset...")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Download images
        images_tar = self.root_dir / 'images.tar.gz'
        if not images_tar.exists():
            print("Downloading images...")
            self._download_with_progress(self.IMAGES_URL, images_tar)
            print("Extracting images...")
            with tarfile.open(images_tar, 'r:gz') as tar:
                tar.extractall(self.root_dir)
            images_tar.unlink()  
        
        # Download annotations
        annotations_tar = self.root_dir / 'annotations.tar.gz'
        if not annotations_tar.exists():
            print("Downloading annotations...")
            self._download_with_progress(self.ANNOTATIONS_URL, annotations_tar)
            print("Extracting annotations...")
            with tarfile.open(annotations_tar, 'r:gz') as tar:
                tar.extractall(self.root_dir)
            annotations_tar.unlink()  
        
        print("Dataset downloaded and extracted successfully!")
    
    def _download_with_progress(self, url, destination):
        """Download file with progress bar."""
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, 
                                miniters=1, desc=destination.name) as t:
            urllib.request.urlretrieve(url, filename=destination, 
                                      reporthook=t.update_to)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_name = image_path.stem + '.png'
        mask_path = self.masks_dir / mask_name
        
        if not mask_path.exists():
            # If mask doesn't exist, create empty mask
            mask = Image.new('L', image.size, 0)
        else:
            mask = Image.open(mask_path)
        
        # Convert trimap to binary (1: foreground, 2: boundary, 3: background)
        # We'll use 1 and 2 as foreground (pet), 3 as background
        mask_np = np.array(mask)
        binary_mask = np.where(mask_np >= 2, 0, 1).astype(np.uint8)
        mask = Image.fromarray(binary_mask)
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask


def create_dataloaders(root_dir='./data/oxford_pets', 
                      batch_size=4, 
                      image_size=1024,
                      num_workers=2,
                      download=True):
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory for dataset
        batch_size: Batch size
        image_size: Image size (SAM expects 1024x1024)
        num_workers: Number of workers for dataloader
        download: Whether to download dataset if not present
    
    Returns:
        train_dataloader, val_dataloader
    """
    
    # Create datasets
    train_dataset = OxfordPetDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size,
        download=download
    )
    
    val_dataset = OxfordPetDataset(
        root_dir=root_dir,
        split='val',
        image_size=image_size,
        download=False  
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    
    return train_dataloader, val_dataloader


# Test the dataset
if __name__ == '__main__':
    print("Testing Oxford Pet Dataset...")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir='./data/oxford_pets',
        batch_size=2,
        image_size=256,  # Smaller for testing
        num_workers=0,
        download=True
    )
    
    # Test loading a batch
    print("\nLoading test batch...")
    images, masks = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")  # Should be [batch_size, 3, 256, 256]
    print(f"  Masks: {masks.shape}")    # Should be [batch_size, 1, 256, 256]
    print(f"\nImage range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mask unique values: {masks.unique().tolist()}")
    
    print("\nDataset loaded successfully!")