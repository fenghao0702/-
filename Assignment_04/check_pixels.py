import os, numpy as np
from PIL import Image

base = '/root/autodl-tmp/data/lego/output/official_3dgs/test/ours_5000'
gt_dir = os.path.join(base, 'gt')
pred_dir = os.path.join(base, 'renders')

gt_files = sorted(os.listdir(gt_dir))
pred_files = sorted(os.listdir(pred_dir))

print("First 5 GT files:", gt_files[:5])
print("First 5 Pred files:", pred_files[:5])
print("Files match:", gt_files == pred_files)

# Compare one pair more thoroughly — check if images are actually from same view
f = gt_files[0]
gt = np.array(Image.open(os.path.join(gt_dir, f))).astype(np.float32)
pred = np.array(Image.open(os.path.join(pred_dir, f))).astype(np.float32)

# Count non-zero pixels
gt_nonzero = (gt.sum(axis=2) > 10).sum()
pred_nonzero = (pred.sum(axis=2) > 10).sum()
print(f"\n{f}: GT nonzero pixels={gt_nonzero}, Pred nonzero pixels={pred_nonzero}")
print(f"GT corner (10:15,10:15):\n{gt[10:15,10:15,0]}")
print(f"Pred corner (10:15,10:15):\n{pred[10:15,10:15,0]}")
