import lpips, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np

loss_fn = lpips.LPIPS(net='alex').cuda()
transform = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5],[0.5])])

real_dir = Path('./data/ISIC2016/images')
fake_dir = Path('./generated_results/version_4/images')

scores = []
for r, f in zip(sorted(real_dir.glob('*.jpg')), sorted(fake_dir.glob('*.png'))):
    img_r = transform(Image.open(r).convert('RGB')).unsqueeze(0).cuda()
    img_f = transform(Image.open(f).convert('RGB')).unsqueeze(0).cuda()
    scores.append(loss_fn(img_r, img_f).item())

print(f'LPIPS: {np.mean(scores):.4f} ± {np.std(scores):.4f}')