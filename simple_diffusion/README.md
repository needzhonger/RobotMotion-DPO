# 人体到机器人动作的 x0-Diffusion

这是一个简洁的条件 Diffusion 基线。输入是每帧 69 维人体参数，输出是每帧 36 维机器人动作：

本目录与仓库中的其他实现相互独立，默认从同级仓库根目录下的 `../data/` 读取数据。

- 条件：`input_pose_body` + `input_root_orient` + `input_trans`
- 目标：`output_root_pos` + `output_root_rot` + `output_dof_pos`

训练目标是直接预测干净动作 `x_0`，而不是预测噪声 `epsilon`：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
loss = MSE(model(x_t, t, condition), x_0)
```

数据按完整 NPZ 文件，以固定随机种子划分为 80%/10%/10%。归一化统计量只由训练集计算，划分和统计量都保存在 checkpoint 中，测试时会严格复用。

## 训练

```bash
conda activate gvhmr
cd simple_diffusion
python scripts/train.py
```

### 使用 Weights & Biases 观察训练曲线

首次使用时登录 W&B，然后在训练命令中加入 `--wandb`：

```bash
wandb login
python scripts/train.py \
  --wandb --wandb-project physmodpo --wandb-name x0-diffusion-baseline
```

训练过程中会记录 batch/epoch 训练损失、验证损失、最佳验证损失、学习率和每个 epoch 的耗时。常用参数：

- `--wandb-project`：项目名称，默认 `physmodpo`
- `--wandb-entity`：W&B 用户名或团队名称
- `--wandb-name`：本次实验名称
- `--wandb-log-interval`：batch 损失记录间隔，默认每 10 个 batch
- `--wandb-mode offline`：无网络时记录到本地，之后可使用 `wandb sync` 上传

不传 `--wandb` 时不会导入或初始化 W&B，也不会影响原有训练流程。

快速检查代码是否能跑通：

```bash
python scripts/train.py --epochs 1 --hidden-dim 64 --num-blocks 2 \
  --batch-size 8 --num-workers 0 --output-dir outputs_smoke
```

## 测试和导出

```bash
python scripts/test.py --checkpoint outputs/best.pt \
  --output-dir predictions --sampling-steps 20
```

测试脚本使用确定性 DDIM 加速采样，输出保持原目录结构，并保存为与机器人目标字段兼容的 NPZ。`--sampling-steps` 越大通常越精细，但速度越慢，最大值等于训练时的 `--diffusion-steps`。

主要文件：

- `models/temporal_denoiser.py`：时序卷积去噪网络
- `models/diffusion.py`：前向加噪、x0 训练损失与 DDIM 采样
- `datasets/motion_dataset.py`：文件级划分、四元数处理、归一化和窗口切分
- `scripts/train.py`：训练和验证入口
- `scripts/test.py`：测试、指标计算和 NPZ 导出入口
