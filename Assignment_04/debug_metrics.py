import os, numpy as np
from PIL import Image

# Find the actual rendered directories
base = '/root/autodl-tmp/data/lego/output/official_fix'
for split in ['train', 'test']:
    p = os.path.join(base, split)
    if os.path.exists(p):
        print(f'{split}: {os.listdir(p)}')

# Check the original image format
orig = Image.open('/root/autodl-tmp/data/lego/images/r_0.png')
print(f'\nOriginal r_0.png: size={orig.size}, mode={orig.mode}')

# Check pixel values at a few locations
orig_arr = np.array(orig)
print(f'Original shape: {orig_arr.shape}')
print(f'Original min/max per channel: R:{orig_arr[:,:,0].min()}-{orig_arr[:,:,0].max()}, G:{orig_arr[:,:,1].min()}-{orig_arr[:,:,1].max()}, B:{orig_arr[:,:,2].min()}-{orig_arr[:,:,2].max()}')
if orig_arr.shape[2] == 4:
    print(f'Alpha min/max: {orig_arr[:,:,3].min()}-{orig_arr[:,:,3].max()}')
