import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from model import UNetSR
from dataset import get_dataloaders

def ssim_loss(img1, img2, window_size=11, size_average=True):
    import torch.nn.functional as F
    def _fspecial_gauss_1d(window_size, sigma):
        coords = torch.arange(window_size, dtype=img1.dtype)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1, 1)
    
    def _fspecial_gauss_2d(window_size, channel, sigma):
        kernel_x = _fspecial_gauss_1d(window_size, sigma)
        kernel_y = _fspecial_gauss_1d(window_size, sigma)
        kernel_2d = torch.mm(kernel_y, kernel_x.t())
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
        return kernel
    
    def _ssim(X, Y, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(X, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(Y, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(X * X, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(Y * Y, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(X * Y, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
        
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    channel = img1.size(1)
    window = _fspecial_gauss_2d(window_size, channel, 1.5)
    window = window.to(img1.device)
    
    return 1 - _ssim(img1, img2, window, window_size, channel, size_average)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for degraded, gt in tqdm(loader, desc='Train', leave=False):
        degraded = degraded.to(device)
        gt = gt.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(degraded)
            l1 = nn.L1Loss()(outputs, gt)
            ssim = ssim_loss(outputs, gt)
            loss = 0.8 * l1 + 0.2 * ssim  # weighted sum as per plan
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * degraded.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for degraded, gt in tqdm(loader, desc='Val', leave=False):
            degraded = degraded.to(device)
            gt = gt.to(device)
            with autocast():
                outputs = model(degraded)
                l1 = nn.L1Loss()(outputs, gt)
                ssim = ssim_loss(outputs, gt)
                loss = 0.8 * l1 + 0.2 * ssim
            running_loss += loss.item() * degraded.size(0)
    val_loss = running_loss / len(loader.dataset)
    return val_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = UNetSR().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # we'll combine with SSIM in training loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    
    train_loader, val_loader = get_dataloaders(batch_size=16, num_workers=2)
    
    best_val_loss = float('inf')
    epochs = 100
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}/{epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print('  -> Saved best model')
        
        # Also save last epoch
        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    
    print('Training completed.')

if __name__ == '__main__':
    main()
