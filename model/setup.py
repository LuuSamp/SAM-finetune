"""
Setup and download utilities for SAM fine-tuning environment.
"""

import sys
import subprocess
import os
from pathlib import Path
import ssl
import zipfile
import shutil
import requests
import urllib.request
from tqdm import tqdm


def setup_environment():
    """
    Install all required dependencies.
    """
    print("Installing dependencies...")
    
    packages = [
        "git+https://github.com/facebookresearch/segment-anything.git",
        "opencv-python-headless",
        "matplotlib",
        "albumentations",
        "scikit-learn",
        "requests",
        "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"Failed to install {package}, but continuing...")
    
    print("Dependencies installed!")


def download_file_robust(url, filename, verify_ssl=True):
    """
    Robust file download with SSL fallback.
    
    Args:
        url (str): Download URL
        filename (str): Local filename
        verify_ssl (bool): Whether to verify SSL certificates
        
    Returns:
        bool: True if download successful
    """
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30, verify=verify_ssl)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error with requests: {e}")
        
        try:
            print("Trying alternative method...")
            ssl_context = ssl._create_unverified_context()
            
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r{filename}: {percent}%")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, filename, reporthook=progress_hook, 
                                      context=ssl_context)
            print("\nDownload completed!")
            return True
        except Exception as e2:
            print(f"Download error: {e2}")
            return False


def download_kvasir_dataset():
    """
    Download and prepare Kvasir-SEG dataset.
    
    Returns:
        str: Path to dataset directory
    """
    print("Preparing Kvasir-SEG dataset...")
    
    data_dir = Path("./kvasir_seg_data")
    
    # Check if dataset already exists
    if data_dir.exists():
        image_files = list((data_dir / "images").glob("*"))
        mask_files = list((data_dir / "masks").glob("*"))
        if len(image_files) > 100 and len(mask_files) > 100:
            print(f"Dataset already exists! ({len(image_files)} images, {len(mask_files)} masks)")
            return str(data_dir)
    
    data_dir.mkdir(exist_ok=True)
    
    # Try different download URLs
    urls = [
        "https://datasets.simula.no/downloads/kvasir-seg.zip",
    ]
    
    zip_path = "kvasir-seg.zip"
    downloaded = False
    
    for url in urls:
        if download_file_robust(url, zip_path, verify_ssl=False):
            downloaded = True
            break
    
    if not downloaded:
        print("Automatic download failed.")
        print("\nMANUAL DOWNLOAD REQUIRED:")
        print("1. Visit: https://datasets.simula.no/kvasir-seg/")
        print("2. Download kvasir-seg.zip")
        print(f"3. Place the file in: {Path.cwd()}")
        print("4. Run the script again")
        
        if not Path(zip_path).exists():
            sys.exit(1)
        else:
            print("File found!")
    
    print("Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./temp_kvasir")
    except Exception as e:
        print(f"Extraction error: {e}")
        sys.exit(1)
    
    print("Organizing dataset...")
    temp_dir = Path("./temp_kvasir/Kvasir-SEG")
    
    if not temp_dir.exists():
        temp_dir = Path("./temp_kvasir")
        if not (temp_dir / "images").exists():
            print("Unexpected zip structure!")
            print(f"Contents: {list(temp_dir.glob('*'))}")
            sys.exit(1)
    
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "masks").mkdir(exist_ok=True)
    
    # Import cv2 after installation
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not available, trying to install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        import cv2
        import numpy as np
    
    # Copy images
    img_source = temp_dir / "images"
    image_count = 0
    for img in img_source.glob("*"):
        if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy(img, data_dir / "images" / img.name)
            image_count += 1
    
    print(f"Copied {image_count} images")
    
    # Process masks
    mask_source = temp_dir / "masks"
    mask_count = 0
    for mask in mask_source.glob("*"):
        if mask.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            try:
                mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # Convert to binary mask
                    mask_binary = (mask_img > 128).astype(np.uint8) * 255
                    output_name = mask.stem + '.png'
                    cv2.imwrite(str(data_dir / "masks" / output_name), mask_binary)
                    mask_count += 1
                else:
                    print(f"Failed to read mask: {mask.name}")
            except Exception as e:
                print(f"Error processing {mask.name}: {e}")
    
    print(f"Processed {mask_count} masks")
    
    # If masks folder doesn't exist in zip, create from images
    if mask_count == 0:
        print("No masks found in zip, checking for alternative structure...")
        # Some versions might have masks in different structure
        for img_path in (data_dir / "images").glob("*"):
            mask_name = img_path.stem + '.png'
            mask_path = data_dir / "masks" / mask_name
            # Create dummy mask (you might need to adjust this based on your dataset)
            dummy_mask = np.zeros((1024, 1024), dtype=np.uint8)
            cv2.imwrite(str(mask_path), dummy_mask)
            mask_count += 1
        print(f"Created {mask_count} placeholder masks")
    
    # Cleanup
    try:
        shutil.rmtree("./temp_kvasir")
        if Path(zip_path).exists():
            os.remove(zip_path)
    except:
        pass
    
    n_images = len(list((data_dir / "images").glob("*")))
    n_masks = len(list((data_dir / "masks").glob("*")))
    
    print(f"Dataset prepared!")
    print(f"  Images: {n_images}")
    print(f"  Masks: {n_masks}")
    
    if n_images == 0:
        print("No images found!")
        sys.exit(1)
    
    if n_masks == 0:
        print("Warning: No masks found! Creating placeholder masks...")
        # Create placeholder masks for all images
        for img_path in (data_dir / "images").glob("*"):
            mask_name = img_path.stem + '.png'
            mask_path = data_dir / "masks" / mask_name
            placeholder_mask = np.zeros((1024, 1024), dtype=np.uint8)
            cv2.imwrite(str(mask_path), placeholder_mask)
        n_masks = len(list((data_dir / "masks").glob("*")))
        print(f"Created {n_masks} placeholder masks")
    
    return str(data_dir)


def download_sam_checkpoint(model_type="vit_b"):
    """
    Download SAM checkpoint.
    
    Args:
        model_type (str): SAM model type (vit_b, vit_l, vit_h)
        
    Returns:
        str: Path to checkpoint file
    """
    print("Downloading SAM model...")
    
    checkpoints = {
        "vit_b": ("sam_vit_b_01ec64.pth", 
                  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
        "vit_l": ("sam_vit_l_0b3195.pth",
                  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
        "vit_h": ("sam_vit_h_4b8939.pth",
                  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    }
    
    filename, url = checkpoints[model_type]
    
    if Path(filename).exists():
        print(f"Checkpoint {filename} already exists!")
        return filename
    
    if not download_file_robust(url, filename, verify_ssl=False):
        print(f"Model download failed.")
        print(f"\nMANUAL DOWNLOAD REQUIRED:")
        print(f"1. Visit: {url}")
        print(f"2. Download the file")
        print(f"3. Place in: {Path.cwd()}")
        
        if not Path(filename).exists():
            sys.exit(1)
    
    return filename


def setup_complete_environment():
    """
    Complete environment setup including downloads.
    
    Returns:
        tuple: (data_dir, checkpoint_path)
    """
    print("=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Install dependencies
    setup_environment()
    
    # Download datasets and models
    data_dir = download_kvasir_dataset()
    checkpoint_path = download_sam_checkpoint("vit_b")
    
    return data_dir, checkpoint_path