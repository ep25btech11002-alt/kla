import os
import time
import numpy as np
import torch
from model import UNetSR
import argparse

def load_model(model_path, device):
    model = UNetSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_npy(filepath):
    # Load .npy file
    img = np.load(filepath).astype(np.float32)  # shape (H, W)
    # Normalize to [0,1] using min and max of the image
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val > 1e-8:
        img_norm = (img - min_val) / (max_val - min_val)
    else:
        img_norm = img
    # Add channel dimension
    img_norm = img_norm[np.newaxis, ...]  # (1, H, W)
    return torch.from_numpy(img_norm), min_val, max_val

def postprocess_npy(tensor, original_min, original_max):
    # tensor is (1, 1, H, W) after model output (batch, channel, H, W)
    img = tensor.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
    # Scale back to original range
    img = img * (original_max - original_min) + original_min
    return img

def main():
    parser = argparse.ArgumentParser(description='Restore degraded images using trained model.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing degraded .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save restored .npy files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pth)')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process (for quick testing)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = load_model(args.model_path, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of .npy files
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    files.sort()
    if args.num_images is not None:
        files = files[:args.num_images]
    print(f'Found {len(files)} images to process.')
    
    times = []
    for fname in files:
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        
        # Preprocess
        input_tensor, orig_min, orig_max = preprocess_npy(input_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dimension -> (1, 1, H, W)
        
        # Inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output_tensor = model(input_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        times.append(end - start)
        
        # Postprocess
        restored_img = postprocess_npy(output_tensor, orig_min, orig_max)
        
        # Save
        np.save(output_path, restored_img.astype(np.float32))
        
        if (len(times) % 10) == 0:
            print(f'Processed {len(times)}/{len(files)} - Avg time per image: {np.mean(times):.3f}s')
    
    avg_time = np.mean(times)
    print(f'Finished processing {len(files)} images.')
    print(f'Average inference time per image: {avg_time:.3f} seconds')
    print(f'Total time: {sum(times):.3f} seconds')

if __name__ == '__main__':
    main()
