import os
import json
import shutil
import argparse

def load_prompt_json(path):
    names = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            names.append(os.path.basename(item["target"]))
    return names

def load_split(path):
    with open(os.path.expanduser(path)) as f:
        return set(l.strip() for l in f if l.strip())

def build_mapping(prompt_names, gen_src):
    mapping = {}
    gen_files = sorted(os.listdir(os.path.expanduser(gen_src)))
    for i, gen_file in enumerate(gen_files):
        if i >= len(prompt_names):
            break
        mapping[prompt_names[i]] = os.path.join(os.path.expanduser(gen_src), gen_file)
    return mapping

def copy_with_rename(mapping, split_files, dst_dir):
    dst_dir = os.path.expanduser(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    needed = set()
    for s in split_files:
        needed |= load_split(s)
    found, missing = 0, []
    for real_name in needed:
        if real_name in mapping:
            src = mapping[real_name]
            dst = os.path.join(dst_dir, real_name)
            shutil.copy2(src, dst)
            found += 1
        else:
            missing.append(real_name)
    print(f"✓ 复制完成：{found} 张 → {dst_dir}")
    if missing:
        print(f"⚠️  以下 {len(missing)} 张未找到对应：")
        for m in missing[:10]:
            print(f"   {m}")
    return found, missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_json", required=True)
    parser.add_argument("--gen_src",     required=True)
    parser.add_argument("--gen_dst",     required=True)
    parser.add_argument("--split_40",    required=True)
    parser.add_argument("--split_100",   required=True)
    args = parser.parse_args()

    print("1. 读取 prompt.json ...")
    prompt_names = load_prompt_json(args.prompt_json)
    print(f"   共 {len(prompt_names)} 条记录")

    print("2. 扫描生成图目录 ...")
    gen_src = os.path.expanduser(args.gen_src)
    gen_count = len(os.listdir(gen_src))
    print(f"   共 {gen_count} 张生成图")

    print("3. 构建映射关系 ...")
    mapping = build_mapping(prompt_names, gen_src)
    print(f"   映射建立完成：{len(mapping)} 条")

    print("4. 按 split 过滤并复制 ...")
    found, missing = copy_with_rename(
        mapping,
        split_files=[args.split_40, args.split_100],
        dst_dir=args.gen_dst,
    )

    print()
    if not missing:
        print("✅ 全部完成，无缺失。可以直接跑 E-30~E-33。")
    else:
        print(f"❌ 有 {len(missing)} 张缺失，请检查 prompt.json 与生成图是否来自同一次训练。")

if __name__ == "__main__":
    main()