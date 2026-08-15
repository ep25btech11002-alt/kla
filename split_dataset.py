import os
import shutil
import random

src_gt = os.path.join('data', 'train', 'train', 'GT')
src_nl = os.path.join('data', 'train', 'train', 'NoisyLR')
dst_train_gt = os.path.join('data', 'train', 'ground_truth')
dst_train_nl = os.path.join('data', 'train', 'degraded')
dst_val_gt = os.path.join('data', 'val', 'ground_truth')
dst_val_nl = os.path.join('data', 'val', 'degraded')

for d in [dst_train_gt, dst_train_nl, dst_val_gt, dst_val_nl]:
    os.makedirs(d, exist_ok=True)

files = sorted([f for f in os.listdir(src_gt) if f.endswith('.npy')])
random.seed(42)
random.shuffle(files)
split_idx = int(0.9 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def copy_set(file_list, dst_gt_dir, dst_nl_dir):
    for f in file_list:
        shutil.copy(os.path.join(src_gt, f), os.path.join(dst_gt_dir, f))
        shutil.copy(os.path.join(src_nl, f), os.path.join(dst_nl_dir, f))

print(f'Total files: {len(files)}')
print(f'Train: {len(train_files)}')
print(f'Val: {len(val_files)}')
copy_set(train_files, dst_train_gt, dst_train_nl)
copy_set(val_files, dst_val_gt, dst_val_nl)
print('Done.')
