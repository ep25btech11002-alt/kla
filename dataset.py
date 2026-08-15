import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class DegradedRestorationDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir, split='train', augment=False):
        """
        degraded_dir: directory with degraded .npy files
        gt_dir: directory with ground truth .npy files
        split: 'train' or 'val' (used for deterministic augmentation)
        augment: whether to apply data augmentation
        """
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.augment = augment
        
        # Get list of filenames (assumed matching)
        self.filenames = sorted([f for f in os.listdir(degraded_dir) if f.endswith('.npy')])
        
        # Define augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90, expand=False, fill=0),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # Load degraded
        degraded_path = os.path.join(self.degraded_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)
        
        degraded = np.load(degraded_path).astype(np.float32)  # shape (H, W)
        gt = np.load(gt_path).astype(np.float32)
        
        # Normalize degraded to [0,1] using its own min and max
        d_min = degraded.min()
        d_max = degraded.max()
        if d_max - d_min > 1e-8:
            degraded_norm = (degraded - d_min) / (d_max - d_min)
        else:
            degraded_norm = degraded  # constant image
        
        # GT is already in [0,1] as seen earlier, but ensure
        gt_norm = np.clip(gt, 0.0, 1.0)
        
        # Add channel dimension
        degraded_norm = degraded_norm[np.newaxis, ...]  # (1, H, W)
        gt_norm = gt_norm[np.newaxis, ...]
        
        # Convert to torch tensors
        degraded_tensor = torch.from_numpy(degraded_norm)
        gt_tensor = torch.from_numpy(gt_norm)
        
        # Apply augmentation (same seed for both to keep consistency)
        if self.augment and self.augment_transform is not None:
            # torchvision transforms work on PIL Images or torch tensors (C,H,W)
            # We'll convert to tensor, apply transforms, but need to ensure same random state.
            # Simpler: apply random flips and rotations manually using torch.
            if random.random() > 0.5:
                degraded_tensor = torch.flip(degraded_tensor, dims=[2])  # horizontal flip
                gt_tensor = torch.flip(gt_tensor, dims=[2])
            if random.random() > 0.5:
                degraded_tensor = torch.flip(degraded_tensor, dims=[1])  # vertical flip
                gt_tensor = torch.flip(gt_tensor, dims=[1])
            # Rotation by 90 degrees multiples
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                degraded_tensor = torch.rot90(degraded_tensor, k, dims=[1, 2])
                gt_tensor = torch.rot90(gt_tensor, k, dims=[1, 2])
        
        return degraded_tensor, gt_tensor

def get_dataloaders(batch_size=8, num_workers=2):
    train_degraded = os.path.join('data', 'train', 'degraded')
    train_gt = os.path.join('data', 'train', 'ground_truth')
    val_degraded = os.path.join('data', 'val', 'degraded')
    val_gt = os.path.join('data', 'val', 'ground_truth')
    
    train_dataset = DegradedRestorationDataset(train_degraded, train_gt, augment=True)
    val_dataset = DegradedRestorationDataset(val_degraded, val_gt, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

if __name__ == '__main__':
    # Quick test
    train_loader, val_loader = get_dataloaders(batch_size=4)
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    for degraded, gt in train_loader:
        print('Degraded shape:', degraded.shape)
        print('GT shape:', gt.shape)
        print('Degraded min/max:', degraded.min().item(), degraded.max().item())
        print('GT min/max:', gt.min().item(), gt.max().item())
        break
