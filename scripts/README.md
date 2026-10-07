# 项目脚本集

论文实验、可视化、数据处理的辅助脚本。

## 一、核心脚本（论文最终用）

### 可视化
| 脚本 | 作用 | 输出 |
|---|---|---|
| `select_good_samples.py` | 自动筛选"E-B 稳定略优"的样本 | 终端输出 SAMPLE_IDX |
| `visualize_generation.py` | 生成质量对比（mask + 3 方法） | `fig_A_*.png` / `fig_B_*.png` |
| `visualize_segmentation_v2.py` | 分割预测（4 列含错误图） | `fig_C_segmentation_v2.png` |

### 批量运行
| 脚本 | 作用 |
|---|---|
| `run_E23B_fix.sh` | E-23B 重跑（ISIC2016 N=40） |
| `run_N40.sh` | N=40 多种子验证（10 个种子） |
| `run_2018_downstream.sh` | ISIC2018 下游分割 |
| `run_2018_seeds.sh` | ISIC2018 N=40 多种子 |
| `run_all_EB2018.sh` | ISIC2018 E-B 全 step 生成 |

## 二、辅助脚本

| 脚本 | 作用 |
|---|---|
| `visualize_segmentation.py` | 分割预测 v1（3 列，无错误图） |
| `check_seeds_2018.sh` | 实时查看 ISIC2018 多种子进度 |
| `check_down_2018.sh` | 实时查看 ISIC2018 下游分割进度 |

## 三、废弃脚本（保留参考）

| 脚本 | 说明 |
|---|---|
| `run_seed_sweep.sh` | 早期版本，有 `gen_ours` symlink bug，被 `run_E23B_fix.sh` 替代 |
| `visualize_boundary_zoom.py` | 边界放大图，效果不理想，未用于论文 |

## 四、用法示例

```bash
# 论文图 C（分割预测）
python scripts/select_good_samples.py     # 输出推荐的 SAMPLE_IDX
# 改 visualize_segmentation_v2.py 里的 SAMPLE_IDX
python scripts/visualize_segmentation_v2.py

# 论文图 A/B（生成质量）
python scripts/visualize_generation.py
五、依赖
torch、monai

Pillow、matplotlib、numpy

scipy

六、路径说明
脚本里路径：

/mnt/f/240493014/paper2/：项目输出根目录

/home/pc/SiameseDiffusion_outputs/：WSL 训练输出

/home/pc/isic2016_downstream/、/home/pc/isic2018_downstream/：下游数据

/mnt/f/seg_experiments/checkpoints/：下游分割 ckpt

路径变了需相应修改。

版本：v1
日期：2026-10-07
