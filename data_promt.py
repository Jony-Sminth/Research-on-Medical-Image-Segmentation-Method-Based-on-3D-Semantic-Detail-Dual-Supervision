import os, json

image_dir = "./data/ISIC2018/images"
mask_dir = "./data/ISIC2018/masks"

with open("./data/ISIC2018/prompt.json", "w") as f:
    for fname in sorted(os.listdir(image_dir)):
        stem = os.path.splitext(fname)[0]
        img_path = f"./data/ISIC2018/images/{fname}"
        mask_fname = f"{stem}_segmentation.png"
        mask_path = f"./data/ISIC2018/masks/{mask_fname}"
        if os.path.exists(mask_path):
            record = {
                "source": mask_path,
                "target": img_path,
                "prompt": "a dermoscopy image of skin lesion"
            }
            f.write(json.dumps(record) + "\n")