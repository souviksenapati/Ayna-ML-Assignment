import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ## 2. Custom Dataset Class (Updated)

class PolygonColorDataset(Dataset):
    def __init__(self, json_path, input_dir, output_dir, color_to_idx, input_transform=None, output_transform=None):
        """
        Dataset for polygon coloring task.
        Accepts a global color map to ensure color index consistency.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        # Use the global, pre-defined color mapping passed in
        self.color_to_idx = color_to_idx
        self.colors = sorted(color_to_idx.keys())
        self.num_colors = len(self.color_to_idx)
        
        print(f"Loaded {len(self.data)} samples for {os.path.basename(os.path.dirname(json_path))}.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_path = os.path.join(self.input_dir, item['input_polygon'])
        input_image = Image.open(input_path).convert('RGB')
        
        output_path = os.path.join(self.output_dir, item['output_image'])
        output_image = Image.open(output_path).convert('RGB')
        
        color = item['colour']
        color_idx = self.color_to_idx[color]
        
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.output_transform:
            output_image = self.output_transform(output_image)
        
        return {
            'input_image': input_image,
            'output_image': output_image,
            'color_idx': torch.tensor(color_idx, dtype=torch.long),
            'color_name': color
        }

# ## 3. UNet Model Implementation (No change needed)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, n_colors=5, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels, self.n_classes, self.n_colors, self.bilinear = n_channels, n_classes, n_colors, bilinear
        self.color_embedding = nn.Embedding(n_colors, 64)
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3 = Down(64, 128), Down(128, 256), Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.color_proj = nn.Linear(64, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x, color_idx):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        color_emb = self.color_embedding(color_idx)
        color_proj = self.color_proj(color_emb)
        B, C, H, W = x5.shape
        color_proj = color_proj.view(B, C, 1, 1).expand(-1, -1, H, W)
        x5 = x5 + color_proj
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

# ## 4. Data Preparation and Augmentation (Updated)

# --- Create a global, unified color map ---
train_json_path = "/kaggle/input/aynadataset/dataset/dataset/training/data.json"
val_json_path = "/kaggle/input/aynadataset/dataset/dataset/validation/data.json"

with open(train_json_path, 'r') as f:
    train_data = json.load(f)
with open(val_json_path, 'r') as f:
    val_data = json.load(f)

# Get all unique colors from both training and validation sets
all_colors = set([item['colour'] for item in train_data] + [item['colour'] for item in val_data])
sorted_colors = sorted(list(all_colors))

# This is the single, authoritative color map for the entire project
global_color_to_idx = {color: idx for idx, color in enumerate(sorted_colors)}
num_total_colors = len(global_color_to_idx)

print("--- Global Color Map ---")
print(f"Total unique colors found: {num_total_colors}")
print(global_color_to_idx)
print("------------------------")


# Transform for input images: includes augmentation
train_input_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Jitter is safe on the input outline
    transforms.ToTensor(),
])

# Transform for output/target images: NO augmentation
output_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load datasets, passing the GLOBAL color map to both
train_dataset = PolygonColorDataset(
    train_json_path,
    "/kaggle/input/aynadataset/dataset/dataset/training/inputs",
    "/kaggle/input/aynadataset/dataset/dataset/training/outputs",
    color_to_idx=global_color_to_idx, # Pass global map
    input_transform=train_input_transform,
    output_transform=output_transform
)

val_dataset = PolygonColorDataset(
    val_json_path,
    "/kaggle/input/aynadataset/dataset/dataset/validation/inputs",
    "/kaggle/input/aynadataset/dataset/dataset/validation/outputs",
    color_to_idx=global_color_to_idx, # Pass global map
    input_transform=output_transform,
    output_transform=output_transform
)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ## 5. Visualization Functions (No change needed)
# ... The rest of your code from Section 5 onwards is correct and does not need to be changed ...
# ... I will include it here for completeness ...

def visualize_batch(dataset, num_samples=4):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    for i in range(num_samples):
        sample = dataset[i]
        input_img, output_img = sample['input_image'].permute(1, 2, 0).numpy(), sample['output_image'].permute(1, 2, 0).numpy()
        axes[0, i].imshow(input_img); axes[0, i].set_title(f'Input: {sample["color_name"]}'); axes[0, i].axis('off')
        axes[1, i].imshow(output_img); axes[1, i].set_title('Target Output'); axes[1, i].axis('off')
    plt.tight_layout(); plt.show()

def visualize_predictions(model, dataset, device, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 12))
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            input_img, color_idx = sample['input_image'].unsqueeze(0).to(device), sample['color_idx'].unsqueeze(0).to(device)
            pred = model(input_img, color_idx)
            input_np, target_np, pred_np = sample['input_image'].permute(1, 2, 0).numpy(), sample['output_image'].permute(1, 2, 0).numpy(), pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(input_np); axes[0, i].set_title(f'Input: {sample["color_name"]}'); axes[0, i].axis('off')
            axes[1, i].imshow(target_np); axes[1, i].set_title('Target'); axes[1, i].axis('off')
            axes[2, i].imshow(pred_np); axes[2, i].set_title('Prediction'); axes[2, i].axis('off')
    plt.tight_layout(); plt.show()

print("Sample training data:")
visualize_batch(train_dataset, num_samples=4)

# ## 6. Training Setup
# --- Model Initialization ---
model = ConditionalUNet(n_channels=3, n_classes=3, n_colors=num_total_colors, bilinear=True).to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters for {num_total_colors} colors.")

# --- Hyperparameters ---
num_epochs = 200
learning_rate = 5e-4
weight_decay = 1e-5

# --- Optimizer and Loss ---
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# --- W&B INTEGRATION ---
# Login to wandb with API key
wandb.login(key="b1609041d5e61c6167815e24c1346912b8bf8c1f")

# Initialize wandb
wandb.init(
    project="ayna-colored-polygon",
    name="unet-run-v11-final",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "architecture": "Conditional UNet",
        "dataset": "Polygon Coloring",
        "optimizer": "Adam",
        "loss_function": "L1Loss",
        "weight_decay": weight_decay,
    }
)


# Loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ## 7. Training Loop

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train(); running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_images, output_images, color_indices = batch['input_image'].to(device), batch['output_image'].to(device), batch['color_idx'].to(device)
        optimizer.zero_grad()
        predictions = model(input_images, color_indices)
        loss = criterion(predictions, output_images)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval(); running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            input_images, output_images, color_indices = batch['input_image'].to(device), batch['output_image'].to(device), batch['color_idx'].to(device)
            predictions = model(input_images, color_indices)
            loss = criterion(predictions, output_images)
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    return running_loss / len(val_loader)

# Training loop
num_epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []

# Optional: clean up old model before starting a new run
if os.path.exists('best_model.pth'):
    os.remove('best_model.pth')
    print("Removed previous best_model.pth to ensure a fresh start.")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the global color map with the model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'color_to_idx': global_color_to_idx,
            'model_config': {
                'n_channels': 3,
                'n_classes': 3,
                'n_colors': num_total_colors,
                'bilinear': True
            }
        }, 'best_model.pth')
        print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    if (epoch + 1) % 10 == 0:
        print("Sample predictions:")
        visualize_predictions(model, val_dataset, device, num_samples=4)

# ## 8. Training Results Visualization

plt.figure(figsize=(12, 4))

# Plot training curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot learning rate
plt.subplot(1, 2, 2)
lr_history = []
for epoch in range(num_epochs):
    if epoch < 5:
        lr_history.append(1e-3)
    elif epoch < 10:
        lr_history.append(5e-4)
    elif epoch < 20:
        lr_history.append(2.5e-4)
    else:
        lr_history.append(1.25e-4)

plt.plot(lr_history)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.show()

# ## 9. Model Evaluation and Testing

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model loaded from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")

# ==================== COMPREHENSIVE MODEL TESTING ====================

def calculate_metrics(model, dataset, device, num_samples=None):
    """Calculate comprehensive metrics on dataset"""
    model.eval()
    
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    color_accuracies = {}
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Calculating metrics"):
            sample = dataset[i]
            
            # Prepare input
            input_img = sample['input_image'].unsqueeze(0).to(device)
            color_idx = sample['color_idx'].unsqueeze(0).to(device)
            target_img = sample['output_image'].unsqueeze(0).to(device)
            color_name = sample['color_name']
            
            # Get prediction
            pred_img = model(input_img, color_idx)
            
            # Calculate MSE
            mse = F.mse_loss(pred_img, target_img).item()
            total_mse += mse
            
            # Calculate MAE
            mae = F.l1_loss(pred_img, target_img).item()
            total_mae += mae
            
            # Calculate PSNR
            psnr = 20 * torch.log10(1.0 / (torch.sqrt(F.mse_loss(pred_img, target_img)) + 1e-8))
            total_psnr += psnr.item()
            
            # Track per-color performance
            if color_name not in color_accuracies:
                color_accuracies[color_name] = []
            color_accuracies[color_name].append(mse)
    
    # Calculate averages
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_psnr = total_psnr / num_samples
    
    # Calculate per-color averages
    for color in color_accuracies:
        color_accuracies[color] = np.mean(color_accuracies[color])
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'psnr': avg_psnr,
        'per_color_mse': color_accuracies
    }

# ==================== TEST ON VALIDATION SET ====================
print("Testing on VALIDATION SET:")
print("=" * 50)

val_metrics = calculate_metrics(model, val_dataset, device)
print(f"Validation MSE: {val_metrics['mse']:.6f}")
print(f"Validation MAE: {val_metrics['mae']:.6f}")
print(f"Validation PSNR: {val_metrics['psnr']:.2f} dB")
print("\nPer-color MSE on validation:")
for color, mse in val_metrics['per_color_mse'].items():
    print(f"  {color}: {mse:.6f}")

# ==================== TEST ON TRAINING SET (subset) ====================
print("\nTesting on TRAINING SET (subset):")
print("=" * 50)

train_metrics = calculate_metrics(model, train_dataset, device, num_samples=100)
print(f"Training MSE (100 samples): {train_metrics['mse']:.6f}")
print(f"Training MAE (100 samples): {train_metrics['mae']:.6f}")
print(f"Training PSNR (100 samples): {train_metrics['psnr']:.2f} dB")

# ==================== VISUAL TESTING ====================
print("\nVISUAL TESTING:")
print("=" * 50)

# Test on validation samples
print("Validation set predictions:")
# --- CORRECTED CODE ---
# Ensure we don't try to show more samples than available
num_to_visualize = min(8, len(val_dataset)) 
visualize_predictions(model, val_dataset, device, num_samples=num_to_visualize)

# Test same polygon with different colors
def test_color_consistency():
    """Test how well model handles same polygon with different colors"""
    print("Testing color consistency (same polygon, different colors):")
    
    # Find samples with same polygon shape
    polygon_groups = {}
    for i, sample in enumerate(val_dataset.data):
        polygon_name = sample['input_polygon']
        if polygon_name not in polygon_groups:
            polygon_groups[polygon_name] = []
        polygon_groups[polygon_name].append(i)
    
    # Find a polygon that appears with multiple colors
    for polygon_name, indices in polygon_groups.items():
        if len(indices) >= 2:
            print(f"Testing polygon: {polygon_name}")
            
            fig, axes = plt.subplots(3, len(indices), figsize=(4*len(indices), 12))
            if len(indices) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, idx in enumerate(indices):
                sample = val_dataset[idx]
                
                # Prepare input
                input_img = sample['input_image'].unsqueeze(0).to(device)
                color_idx = sample['color_idx'].unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    pred = model(input_img, color_idx)
                
                # Convert to numpy for visualization
                input_np = sample['input_image'].permute(1, 2, 0).numpy()
                target_np = sample['output_image'].permute(1, 2, 0).numpy()
                pred_np = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
                
                # Plot
                axes[0, i].imshow(input_np)
                axes[0, i].set_title(f'Input\n{sample["color_name"]}')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(target_np)
                axes[1, i].set_title('Target')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(pred_np)
                axes[2, i].set_title('Prediction')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            break  # Only show first polygon group with multiple colors

test_color_consistency()

# ==================== FAILURE CASE ANALYSIS ====================
print("FAILURE CASE ANALYSIS:")
print("=" * 50)

def analyze_worst_predictions(model, dataset, device, num_worst=5):
    """Analyze the worst performing predictions"""
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Prepare input
            input_img = sample['input_image'].unsqueeze(0).to(device)
            color_idx = sample['color_idx'].unsqueeze(0).to(device)
            target_img = sample['output_image'].unsqueeze(0).to(device)
            
            # Get prediction
            pred_img = model(input_img, color_idx)
            
            # Calculate error
            mse = F.mse_loss(pred_img, target_img).item()
            errors.append((i, mse, sample['color_name']))
    
    # Sort by error (highest first)
    errors.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Analyzing {num_worst} worst predictions:")
    fig, axes = plt.subplots(3, num_worst, figsize=(4*num_worst, 12))
    if num_worst == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_worst):
        idx, error, color_name = errors[i]
        sample = dataset[idx]
        
        # Prepare input and get prediction
        input_img = sample['input_image'].unsqueeze(0).to(device)
        color_idx = sample['color_idx'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(input_img, color_idx)
        
        # Convert to numpy
        input_np = sample['input_image'].permute(1, 2, 0).numpy()
        target_np = sample['output_image'].permute(1, 2, 0).numpy()
        pred_np = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Plot
        axes[0, i].imshow(input_np)
        axes[0, i].set_title(f'Input: {color_name}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_np)
        axes[1, i].set_title('Target')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(pred_np)
        axes[2, i].set_title(f'Pred (MSE: {error:.4f})')
        axes[2, i].axis('off')
    
    plt.suptitle('Worst Performing Predictions')
    plt.tight_layout()
    plt.show()
    
    return errors[:num_worst]

worst_cases = analyze_worst_predictions(model, val_dataset, device, num_worst=5)

# ==================== BEST CASE ANALYSIS ====================
print("BEST CASE ANALYSIS:")
print("=" * 50)

def analyze_best_predictions(model, dataset, device, num_best=5):
    """Analyze the best performing predictions"""
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Prepare input
            input_img = sample['input_image'].unsqueeze(0).to(device)
            color_idx = sample['color_idx'].unsqueeze(0).to(device)
            target_img = sample['output_image'].unsqueeze(0).to(device)
            
            # Get prediction
            pred_img = model(input_img, color_idx)
            
            # Calculate error
            mse = F.mse_loss(pred_img, target_img).item()
            errors.append((i, mse, sample['color_name']))
    
    # Sort by error (lowest first)
    errors.sort(key=lambda x: x[1])
    
    print(f"Analyzing {num_best} best predictions:")
    fig, axes = plt.subplots(3, num_best, figsize=(4*num_best, 12))
    if num_best == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_best):
        idx, error, color_name = errors[i]
        sample = dataset[idx]
        
        # Prepare input and get prediction
        input_img = sample['input_image'].unsqueeze(0).to(device)
        color_idx = sample['color_idx'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(input_img, color_idx)
        
        # Convert to numpy
        input_np = sample['input_image'].permute(1, 2, 0).numpy()
        target_np = sample['output_image'].permute(1, 2, 0).numpy()
        pred_np = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Plot
        axes[0, i].imshow(input_np)
        axes[0, i].set_title(f'Input: {color_name}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_np)
        axes[1, i].set_title('Target')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(pred_np)
        axes[2, i].set_title(f'Pred (MSE: {error:.4f})')
        axes[2, i].axis('off')
    
    plt.suptitle('Best Performing Predictions')
    plt.tight_layout()
    plt.show()

best_cases = analyze_best_predictions(model, val_dataset, device, num_best=5)

# ## 10. Interactive Testing - Color Your Own Polygon!

def inference(model, input_image_path, color_name, device, color_to_idx, transform):
    """
    Perform inference on a single image
    
    Args:
        model: Trained model
        input_image_path: Path to input polygon image
        color_name: Name of desired color
        device: Device to run inference on
        color_to_idx: Color name to index mapping
        transform: Image transformation
    
    Returns:
        Generated colored polygon image
    """
    model.eval()
    
    # Load and preprocess image
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Get color index
    if color_name not in color_to_idx:
        print(f"Error: Color '{color_name}' not found. Available colors: {list(color_to_idx.keys())}")
        return None, input_image
    
    color_idx = torch.tensor([color_to_idx[color_name]], dtype=torch.long).to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(input_tensor, color_idx)
    
    # Convert to PIL image
    pred_np = prediction.squeeze(0).cpu().permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0, 1)
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8))
    
    return pred_image, input_image

# ======================== INTERACTIVE TESTING BLOCK ========================
print("🎨 INTERACTIVE POLYGON COLORING 🎨")
print("=" * 60)

# Get available colors from training dataset
available_colors = list(train_dataset.color_to_idx.keys())
print(f"Available colors: {available_colors}")
print(f"Total colors: {len(available_colors)}")

def color_polygon(input_image_path, desired_color):
    """
    Main function to color a polygon - THIS IS WHERE YOU TEST YOUR INPUT!
    
    Args:
        input_image_path (str): Path to your input polygon image
        desired_color (str): Color name (e.g., 'blue', 'red', 'yellow', etc.)
    
    Returns:
        Displays the input and colored output side by side
    """
    
    print(f"\n🔄 Processing: {input_image_path} → {desired_color}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(input_image_path):
        print(f"❌ Error: File '{input_image_path}' not found!")
        return
    
    # Check if color is available
    if desired_color not in available_colors:
        print(f"❌ Error: Color '{desired_color}' not available!")
        print(f"Available colors: {available_colors}")
        return
    
    try:
        # Get prediction
        colored_image, original_image = inference(
            model, 
            input_image_path, 
            desired_color, 
            device, 
            train_dataset.color_to_idx, 
            transform
        )
        
        if colored_image is None:
            return
        
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Input image
        axes[0].imshow(original_image)
        axes[0].set_title('Input Polygon', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Colored output
        axes[1].imshow(colored_image)
        axes[1].set_title(f'Colored Output: {desired_color.capitalize()}', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Successfully colored polygon with {desired_color}!")
        
        # Optionally save the result
        output_path = f"colored_{desired_color}_{os.path.basename(input_image_path)}"
        colored_image.save(output_path)
        print(f"💾 Result saved as: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")

# ======================== USAGE EXAMPLES ========================

print("\n📝 USAGE INSTRUCTIONS:")
print("1. Upload your polygon image to Kaggle")
print("2. Use the color_polygon() function with your image path and desired color")
print("3. Example usage:")
print("   color_polygon('/kaggle/input/your-image/polygon.png', 'blue')")
print("\n" + "="*60)

# Example 1: Test with a validation image
print("\n🧪 EXAMPLE 1: Testing with validation data")
sample_idx = 0
sample = val_dataset[0]

# Save sample as temporary file for demonstration
temp_input = sample['input_image'].permute(1, 2, 0).numpy()
temp_input = (temp_input * 255).astype(np.uint8)
temp_input_pil = Image.fromarray(temp_input)
temp_input_pil.save('sample_polygon.png')

# Test the function
color_polygon('sample_polygon.png', 'blue')

# Example 2: Test multiple colors on same shape
print("\n🧪 EXAMPLE 2: Same polygon with different colors")
colors_to_test = available_colors[:3]  # Test first 3 colors

fig, axes = plt.subplots(2, len(colors_to_test), figsize=(15, 8))
if len(colors_to_test) == 1:
    axes = axes.reshape(-1, 1)

for i, color in enumerate(colors_to_test):
    colored_image, original_image = inference(
        model, 'sample_polygon.png', color, device, train_dataset.color_to_idx, transform
    )
    
    axes[0, i].imshow(original_image)
    axes[0, i].set_title('Input')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(colored_image)
    axes[1, i].set_title(f'{color.capitalize()}')
    axes[1, i].axis('off')

plt.suptitle('Same Polygon with Different Colors', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Clean up temporary file
os.remove('sample_polygon.png')

print("\n" + "="*60)
print("🎯 YOUR TURN!")
print("Upload your polygon image and use:")
print("color_polygon('path/to/your/image.png', 'desired_color')")
print("="*60)

# ======================== BATCH TESTING FUNCTION ========================

def test_multiple_images(image_folder_path, colors_to_test=None):
    """
    Test multiple images with multiple colors
    
    Args:
        image_folder_path (str): Path to folder containing polygon images
        colors_to_test (list): List of colors to test (default: all available)
    """
    if colors_to_test is None:
        colors_to_test = available_colors
    
    if not os.path.exists(image_folder_path):
        print(f"❌ Folder not found: {image_folder_path}")
        return
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_folder_path) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"❌ No image files found in {image_folder_path}")
        return
    
    print(f"Found {len(image_files)} images. Testing with {len(colors_to_test)} colors...")
    
    for img_file in image_files[:3]:  # Test first 3 images
        img_path = os.path.join(image_folder_path, img_file)
        print(f"\n📸 Testing: {img_file}")
        
        # Test with first color only for display
        color_polygon(img_path, colors_to_test[0])

# Example usage (uncomment and modify path as needed):
# test_multiple_images('/kaggle/input/your-test-images/', ['blue', 'red', 'yellow'])

# ## 11. Save Final Model and Artifacts

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'color_to_idx': train_dataset.color_to_idx,
    'colors': train_dataset.colors,
    'num_colors': train_dataset.num_colors,
    'model_config': {
        'n_channels': 3,
        'n_classes': 3,
        'n_colors': train_dataset.num_colors,
        'bilinear': True
    }
}, 'final_model.pth')

print("Model saved successfully!")

# Log final metrics to wandb
wandb.log({
    "final_train_loss": train_losses[-1],
    "final_val_loss": val_losses[-1],
    "best_val_loss": best_val_loss,
    "total_epochs": num_epochs
})

# Close wandb run
wandb.finish()

print("Training completed successfully!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")

# ## 12. Model Loading Function for Future Use

def load_trained_model(model_path, device):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
    
    Returns:
        Loaded model and metadata
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same configuration
    model = ConditionalUNet(
        n_channels=checkpoint['model_config']['n_channels'],
        n_classes=checkpoint['model_config']['n_classes'],
        n_colors=checkpoint['model_config']['n_colors'],
        bilinear=checkpoint['model_config']['bilinear']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['color_to_idx'], checkpoint['colors']

# Example usage:
# model, color_to_idx, colors = load_trained_model('final_model.pth', device)
# pred_image, input_image = inference(model, 'path/to/polygon.png', 'blue', device, color_to_idx, transform)