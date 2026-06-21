import os, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

for split in ['train', 'test']:
    base = f'/root/autodl-tmp/data/lego/output/official_rgb/{split}/ours_7000'
    if not os.path.exists(base):
        print(f'{split}: not found')
        continue
    gt_dir = os.path.join(base, 'gt')
    pred_dir = os.path.join(base, 'renders')
    files = sorted(os.listdir(gt_dir))

    psnrs, ssims = [], []
    for f in files:
        gt = np.array(Image.open(os.path.join(gt_dir, f))).astype(np.float32)
        pred = np.array(Image.open(os.path.join(pred_dir, f))).astype(np.float32)
        mse = np.mean((gt - pred) ** 2)
        psnrs.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0)
        ssims.append(ssim(gt, pred, channel_axis=2, data_range=255))

    print(f'{split}: PSNR={np.mean(psnrs):.2f} dB, SSIM={np.mean(ssims):.4f}, views={len(files)}')
