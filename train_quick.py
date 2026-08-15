import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from model import UNetSR
from dataset import DegradedRestorationDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Full dataset
    train_degraded = os.path.join('data', 'train', 'degraded')
    train_gt = os.path.join('data', 'train', 'ground_truth')
    full_dataset = DegradedRestorationDataset(train_degraded, train_gt, augment=False)
    
    # Take first 200 samples for quick training
    indices = list(range(200))
    subset = Subset(full_dataset, indices)
    train_loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0)
    
    model = UNetSR().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    save_dir = 'checkpoints_quick'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for degraded, gt in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            degraded = degraded.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            outputs = model(degraded)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * degraded.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}')
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    print('Training completed.')

if __name__ == '__main__':
    main()
