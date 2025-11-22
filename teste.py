"""
SAM Fine-Tuning para Segmentação de Pólipos (Kvasir-SEG)
=========================================================
Versão CORRIGIDA - Resolve erro de dimensão de tensores
"""

# ============================================================================
# PARTE 1: INSTALAÇÃO E SETUP
# ============================================================================

import sys
import subprocess
import os
from pathlib import Path
import ssl

def setup_environment():
    """Instala todas as dependências necessárias."""
    print("🔧 Instalando dependências...")
    
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
            pass
    
    print("✅ Dependências instaladas!")

# Executar setup
setup_environment()

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import zipfile
import shutil
from sklearn.model_selection import train_test_split
import requests
import urllib.request

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    print("⚠️ Execute novamente após instalação!")
    sys.exit(1)

import albumentations as A

# ============================================================================
# PARTE 2: DOWNLOAD ROBUSTO
# ============================================================================

def download_file_robust(url, filename, verify_ssl=True):
    """Download robusto com fallback para SSL."""
    print(f"Baixando {filename}...")
    
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
        print(f"Erro com requests: {e}")
        
        try:
            print("Tentando método alternativo...")
            ssl_context = ssl._create_unverified_context()
            
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r{filename}: {percent}%")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, filename, reporthook=progress_hook, 
                                      context=ssl_context)
            print("\n✅ Download concluído!")
            return True
        except Exception as e2:
            print(f"Erro no download: {e2}")
            return False

def download_kvasir_dataset():
    """Baixa e prepara o dataset Kvasir-SEG."""
    print("\n📦 Preparando dataset Kvasir-SEG...")
    
    data_dir = Path("./kvasir_seg_data")
    
    if data_dir.exists() and len(list((data_dir / "images").glob("*"))) > 100:
        print(f"✅ Dataset já existe! ({len(list((data_dir / 'images').glob('*')))} imagens)")
        return str(data_dir)
    
    data_dir.mkdir(exist_ok=True)
    
    urls = [
        "https://datasets.simula.no/downloads/kvasir-seg.zip",
        "https://www.kaggle.com/datasets/debeshjha1/kvasirseg/download"
    ]
    
    zip_path = "kvasir-seg.zip"
    downloaded = False
    
    for url in urls:
        if download_file_robust(url, zip_path, verify_ssl=False):
            downloaded = True
            break
    
    if not downloaded:
        print("\n⚠️ Download automático falhou.")
        print("\n📥 DOWNLOAD MANUAL:")
        print("1. Acesse: https://datasets.simula.no/kvasir-seg/")
        print("2. Baixe o arquivo kvasir-seg.zip")
        print(f"3. Coloque o arquivo em: {Path.cwd()}")
        print("4. Execute o script novamente")
        
        if not Path(zip_path).exists():
            sys.exit(1)
        else:
            print("✅ Arquivo encontrado!")
    
    print("\nExtraindo arquivos...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./temp_kvasir")
    except Exception as e:
        print(f"Erro ao extrair: {e}")
        sys.exit(1)
    
    print("Organizando dataset...")
    temp_dir = Path("./temp_kvasir/Kvasir-SEG")
    
    if not temp_dir.exists():
        temp_dir = Path("./temp_kvasir")
        if not (temp_dir / "images").exists():
            print("⚠️ Estrutura do zip inesperada!")
            print(f"Conteúdo: {list(temp_dir.glob('*'))}")
            sys.exit(1)
    
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "masks").mkdir(exist_ok=True)
    
    img_source = temp_dir / "images"
    for img in img_source.glob("*"):
        if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy(img, data_dir / "images" / img.name)
    
    mask_source = temp_dir / "masks"
    for mask in mask_source.glob("*"):
        if mask.suffix.lower() in ['.jpg', '.png']:
            try:
                mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_binary = (mask_img > 128).astype(np.uint8) * 255
                    output_name = mask.stem + '.png'
                    cv2.imwrite(str(data_dir / "masks" / output_name), mask_binary)
            except Exception as e:
                print(f"Erro ao processar {mask.name}: {e}")
    
    try:
        shutil.rmtree("./temp_kvasir")
        if Path(zip_path).exists():
            os.remove(zip_path)
    except:
        pass
    
    n_images = len(list((data_dir / "images").glob("*")))
    n_masks = len(list((data_dir / "masks").glob("*")))
    
    print(f"✅ Dataset preparado!")
    print(f"   Imagens: {n_images}")
    print(f"   Máscaras: {n_masks}")
    
    if n_images == 0:
        print("⚠️ Nenhuma imagem encontrada!")
        sys.exit(1)
    
    return str(data_dir)

def download_sam_checkpoint(model_type="vit_b"):
    """Baixa checkpoint do SAM."""
    print("\n🤖 Baixando modelo SAM...")
    
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
        print(f"✅ Checkpoint {filename} já existe!")
        return filename
    
    if not download_file_robust(url, filename, verify_ssl=False):
        print(f"\n⚠️ Download do modelo falhou.")
        print(f"\n📥 DOWNLOAD MANUAL:")
        print(f"1. Acesse: {url}")
        print(f"2. Baixe o arquivo")
        print(f"3. Coloque em: {Path.cwd()}")
        
        if not Path(filename).exists():
            sys.exit(1)
    
    return filename

# Executar downloads
print("\n" + "="*60)
print("PREPARAÇÃO DO AMBIENTE")
print("="*60)

DATA_DIR = download_kvasir_dataset()
CHECKPOINT_PATH = download_sam_checkpoint("vit_b")

# ============================================================================
# PARTE 3: CONFIGURAÇÕES (CORRIGIDO)
# ============================================================================

class Config:
    DATA_DIR = DATA_DIR
    CHECKPOINT_PATH = CHECKPOINT_PATH
    OUTPUT_DIR = "./outputs"
    
    MODEL_TYPE = "vit_b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CORREÇÃO CRÍTICA: SAM requer imagens 1024x1024
    IMAGE_SIZE = 1024  # Tamanho esperado pelo SAM
    
    BATCH_SIZE = 2  # Reduzido devido ao tamanho maior
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    FREEZE_IMAGE_ENCODER = True
    FREEZE_PROMPT_ENCODER = True
    
    USE_AUGMENTATION = True

Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

print(f"\n⚙️ Configurações:")
print(f"   Device: {Config.DEVICE}")
print(f"   Image Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
print(f"   Batch Size: {Config.BATCH_SIZE}")
print(f"   Épocas: {Config.NUM_EPOCHS}")

# ============================================================================
# PARTE 4: DATASET (CORRIGIDO)
# ============================================================================

class SAMDataset(Dataset):
    """Dataset com pré-processamento correto para SAM."""
    
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
        """Pré-processamento compatível com SAM."""
        # Redimensionar mantendo proporção
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding para 1024x1024
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        # Normalização do SAM
        image = image.astype(np.float32)
        image = (image - self.pixel_mean) / self.pixel_std
        
        return image, (new_h, new_w)
    
    def preprocess_mask(self, mask, target_h, target_w):
        """Pré-processamento de máscara."""
        # Redimensionar
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Padding
        pad_h = self.target_size - target_h
        pad_w = self.target_size - target_w
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        
        return mask
    
    def __getitem__(self, idx):
        # Carregar imagem e máscara
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)
        
        # Augmentation (antes do redimensionamento)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Pré-processar imagem
        image, (new_h, new_w) = self.preprocess_image(image)
        
        # Pré-processar máscara
        mask = self.preprocess_mask(mask, new_h, new_w)
        
        # Converter para tensores
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Calcular ponto de prompt no centro do pólipo
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            center = coords.mean(axis=0).astype(int)
            point_coords = np.array([[center[1], center[0]]])  # (x, y)
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
    """Data augmentation."""
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
    """Prepara splits de dados."""
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    mask_dir = data_path / "masks"
    
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    valid_pairs = []
    for img in image_files:
        mask_path = mask_dir / f"{img.stem}.png"
        if mask_path.exists():
            valid_pairs.append((img, mask_path))
    
    print(f"\n📊 Dataset:")
    print(f"   Total de imagens válidas: {len(valid_pairs)}")
    
    images, masks = zip(*valid_pairs)
    
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        images, masks, test_size=(1-train_split), random_state=42
    )
    
    val_size = val_split / (val_split + (1 - train_split - val_split))
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks, test_size=(1-val_size), random_state=42
    )
    
    print(f"   Treino: {len(train_imgs)}")
    print(f"   Validação: {len(val_imgs)}")
    print(f"   Teste: {len(test_imgs)}")
    
    return {
        'train': (train_imgs, train_msks),
        'val': (val_imgs, val_msks),
        'test': (test_imgs, test_msks)
    }

# ============================================================================
# PARTE 5: MÉTRICAS
# ============================================================================

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """Calcula IoU (Intersection over Union)."""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    union = ((pred_binary + gt_binary) > 0).sum()
    return intersection / union if union > 0 else 1.0

def calculate_dice(pred_mask, gt_mask, threshold=0.5):
    """Calcula Dice Coefficient."""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    return (2 * intersection) / total if total > 0 else 1.0

# ============================================================================
# PARTE 6: MODELO (CORRIGIDO)
# ============================================================================

class SAMFineTuner:
    """Fine-tuner para SAM com correção de dimensões."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        print(f"\n🔧 Carregando SAM {config.MODEL_TYPE}...")
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
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
    
    def _freeze_layers(self):
        """Congela camadas especificadas."""
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
        print(f"   Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def train_epoch(self, dataloader):
        """Treina uma época."""
        self.sam.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Treinando", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks_gt = batch['mask'].to(self.device)
            point_coords = batch['point_coords'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            
            batch_size = images.shape[0]
            
            self.optimizer.zero_grad()
            
            # Processar cada imagem individualmente para evitar problemas de batch
            batch_loss = 0
            for i in range(batch_size):
                # Forward pass para uma imagem
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
                
                # Upscale para tamanho original
                upscaled_masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                    mode='bilinear',
                    align_corners=False
                )
                
                loss = self.criterion(upscaled_masks, masks_gt[i:i+1])
                batch_loss += loss
            
            # Backpropagation
            avg_loss = batch_loss / batch_size
            avg_loss.backward()
            self.optimizer.step()
            
            total_loss += avg_loss.item()
            pbar.set_postfix({'loss': f'{avg_loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Valida o modelo."""
        self.sam.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        n_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validando", leave=False)
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks_gt = batch['mask'].to(self.device)
                point_coords = batch['point_coords'].to(self.device)
                point_labels = batch['point_labels'].to(self.device)
                
                batch_size = images.shape[0]
                
                # Processar cada imagem individualmente
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
                    
                    total_iou += calculate_iou(pred_mask, gt_mask)
                    total_dice += calculate_dice(pred_mask, gt_mask)
                    n_samples += 1
        
        return (total_loss / n_samples, total_iou / n_samples, total_dice / n_samples)
    
    def fit(self, train_loader, val_loader):
        """Treina o modelo."""
        print("\n🚀 Iniciando treinamento...\n")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"Época {epoch+1}/{self.config.NUM_EPOCHS}")
            
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
                print("  ✅ Modelo salvo!")
            print()
        
        return self.history

# ============================================================================
# PARTE 7: VISUALIZAÇÃO
# ============================================================================

def plot_training_curves(history, save_path):
    """Plota curvas de treinamento."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['val_iou'], 'g-', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['val_dice'], 'm-', linewidth=2)
    axes[2].set_xlabel('Época', fontsize=12)
    axes[2].set_ylabel('Dice Coefficient', fontsize=12)
    axes[2].set_title('Validation Dice', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")
    plt.show()

def visualize_predictions(model, test_loader, save_path, n_samples=6):
    """Visualiza predições."""
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
            
            # Processar uma imagem por vez
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
            
            # Desnormalizar imagem
            image = images[0].cpu().permute(1, 2, 0).numpy()
            image = image * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            image = np.clip(image / 255.0, 0, 1)
            
            iou = calculate_iou(pred_mask, gt_mask)
            dice = calculate_dice(pred_mask, gt_mask)
            
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Imagem Original", fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(gt_mask, cmap='gray')
            axes[idx, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title(f"Predição\nIoU: {iou:.3f} | Dice: {dice:.3f}", 
                                  fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Predições salvas em: {save_path}")
    plt.show()

# ============================================================================
# PARTE 8: SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*60)
    print("SAM FINE-TUNING - SEGMENTAÇÃO DE PÓLIPOS (KVASIR-SEG)")
    print("="*60)
    
    # Preparar dados
    splits = prepare_data_splits(Config.DATA_DIR, Config.TRAIN_SPLIT, Config.VAL_SPLIT)
    
    train_images, train_masks = splits['train']
    val_images, val_masks = splits['val']
    test_images, test_masks = splits['test']
    
    # Datasets
    train_dataset = SAMDataset(
        train_images, train_masks,
        get_transforms(is_train=True) if Config.USE_AUGMENTATION else None,
        target_size=Config.IMAGE_SIZE
    )
    val_dataset = SAMDataset(val_images, val_masks, target_size=Config.IMAGE_SIZE)
    test_dataset = SAMDataset(test_images, test_masks, target_size=Config.IMAGE_SIZE)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # 0 para Windows, 2+ para Linux
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    # Treinar
    trainer = SAMFineTuner(Config)
    history = trainer.fit(train_loader, val_loader)
    
    # Avaliar no teste
    print("\n📊 Avaliação Final no Conjunto de Teste:")
    test_loss, test_iou, test_dice = trainer.validate(test_loader)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test IoU: {test_iou:.4f}")
    print(f"   Test Dice: {test_dice:.4f}")
    
    # Visualizações
    print("\n📈 Gerando visualizações...")
    plot_training_curves(history, f"{Config.OUTPUT_DIR}/training_curves.png")
    visualize_predictions(trainer, test_loader, f"{Config.OUTPUT_DIR}/predictions.png", n_samples=6)
    
    # Salvar métricas
    results = {
        'config': {
            'model_type': Config.MODEL_TYPE,
            'image_size': Config.IMAGE_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
        },
        'history': history,
        'test_metrics': {
            'loss': float(test_loss),
            'iou': float(test_iou),
            'dice': float(test_dice)
        }
    }
    
    with open(f"{Config.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Treinamento concluído!")
    print(f"📁 Resultados salvos em: {Config.OUTPUT_DIR}/")
    print(f"   - best_model.pth")
    print(f"   - training_curves.png")
    print(f"   - predictions.png")
    print(f"   - results.json")
    
    return trainer, history

# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    trainer, history = main()