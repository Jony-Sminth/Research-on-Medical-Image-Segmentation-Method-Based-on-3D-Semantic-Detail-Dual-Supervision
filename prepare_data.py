import os
import json

# ========== 配置区 ==========
IMAGE_DIR = './data/ISIC2016/images'
MASK_DIR  = './data/ISIC2016/masks'
OUTPUT    = './data/prompt.json'
PROMPT    = 'a dermoscopy image of skin lesion, high quality, medical imaging'
# ============================

images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

lines = []
for img_file in images:
    img_id = img_file.replace('.jpg', '')
    mask_file = img_id + '_Segmentation.png'
    mask_path = os.path.join(MASK_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f'Warning: mask not found for {img_file}, skipping.')
        continue

    entry = {
        "source": os.path.join(MASK_DIR,  mask_file),
        "target": os.path.join(IMAGE_DIR, img_file),
        "prompt": PROMPT
    }
    lines.append(entry)

with open(OUTPUT, 'w') as f:
    for entry in lines:
        f.write(json.dumps(entry) + '\n')

print(f'Done. Total {len(lines)} entries written to {OUTPUT}')
