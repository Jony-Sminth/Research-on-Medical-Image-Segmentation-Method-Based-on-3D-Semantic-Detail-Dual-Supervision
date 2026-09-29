# paper2/rename_gen_to_real.py
import os, json, shutil, argparse

def load_prompt_names(path):
    """按顺序读出 prompt.json 里每行的真实图文件名"""
    names = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 兼容多种字段名
            src = (item.get('jpg') or item.get('target')
                   or item.get('source') or item.get('image'))
            if src is None:
                raise KeyError(f'找不到图片字段，keys={list(item.keys())}')
            names.append(os.path.basename(src))
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt_json', required=True)
    ap.add_argument('--gen_src',     required=True,
                    help='inference2.py 输出目录（含 000000_k0.png ...）')
    ap.add_argument('--gen_dst',     required=True,
                    help='目标目录（会存成 ISIC_xxxxxxx.jpg）')
    ap.add_argument('--only_split',  default=None,
                    help='可选：只复制 split 里出现的文件名')
    args = ap.parse_args()

    prompt_names = load_prompt_names(args.prompt_json)
    print(f'prompt.json: {len(prompt_names)} 条')

    gen_src = os.path.expanduser(args.gen_src)
    gen_dst = os.path.expanduser(args.gen_dst)
    os.makedirs(gen_dst, exist_ok=True)

    # 读 inference2.py 生成的 _k0.png，按数字排序，保证 idx 顺序
    gen_files = sorted(
        [f for f in os.listdir(gen_src) if f.endswith('_k0.png')],
        key=lambda x: int(x.split('_')[0])
    )
    print(f'生成图 _k0.png: {len(gen_files)} 张')

    only = None
    if args.only_split:
        with open(os.path.expanduser(args.only_split)) as f:
            only = set(l.strip() for l in f if l.strip())
        print(f'只复制 split 中的 {len(only)} 个文件')

    n = min(len(prompt_names), len(gen_files))
    copied = 0
    for i in range(n):
        real_name = prompt_names[i]
        if only is not None and real_name not in only:
            continue
        src = os.path.join(gen_src, gen_files[i])
        dst = os.path.join(gen_dst, real_name)
        shutil.copy2(src, dst)
        copied += 1

    print(f'✅ 复制 {copied} 张 -> {gen_dst}')
    print(f'   示例: {prompt_names[0]} <- {gen_files[0]}')

if __name__ == '__main__':
    main()