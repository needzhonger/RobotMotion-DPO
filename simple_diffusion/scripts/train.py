#!/usr/bin/env python3
"""训练直接预测 x_0 的条件 Diffusion。"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import MotionWindowDataset, Normalizer, split_files
from models import GaussianDiffusion, TemporalDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练人体到机器人动作的 x0-Diffusion")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 记录")
    parser.add_argument("--wandb-project", default="physmodpo", help="W&B 项目名称")
    parser.add_argument("--wandb-entity", default=None, help="W&B 团队或用户名")
    parser.add_argument("--wandb-name", default=None, help="本次运行名称")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default="online", help="W&B 运行模式"
    )
    parser.add_argument(
        "--wandb-log-interval", type=int, default=10, help="每多少个训练 batch 记录一次损失"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(args: argparse.Namespace, dataset_info: dict):
    """按需初始化 W&B；未启用时返回 None，不引入额外依赖。"""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "已指定 --wandb，但当前环境未安装 wandb。请执行: pip install wandb"
        ) from error

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("wandb")
    }
    config.update(dataset_info)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=config,
        dir=str(args.output_dir),
    )
    # batch 曲线与 epoch 曲线分别使用各自的横轴。
    run.define_metric("train/global_step")
    run.define_metric("train/batch_loss", step_metric="train/global_step")
    run.define_metric("epoch")
    run.define_metric("train/epoch_loss", step_metric="epoch")
    run.define_metric("val/loss", step_metric="epoch")
    run.define_metric("val/best_loss", step_metric="epoch")
    run.define_metric("train/learning_rate", step_metric="epoch")
    run.define_metric("time/epoch_seconds", step_metric="epoch")
    return run


@torch.no_grad()
def evaluate(model: GaussianDiffusion, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        condition = batch["condition"].to(device)
        motion = batch["motion"].to(device)
        mask = batch["mask"].to(device)
        total_loss += model.training_loss(motion, condition, mask).item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    if args.stride > args.window_size:
        raise ValueError("stride 不能大于 window_size，否则测试序列拼接时会出现空隙")
    if args.wandb_log_interval <= 0:
        raise ValueError("wandb_log_interval 必须为正整数")
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_splits = split_files(args.data_dir, args.val_ratio, args.test_ratio, args.seed)
    normalizer = Normalizer.fit(file_splits["train"])
    train_dataset = MotionWindowDataset(
        file_splits["train"], normalizer, args.window_size, args.stride
    )
    val_dataset = MotionWindowDataset(file_splits["val"], normalizer, args.window_size, args.stride)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model_config = {
        "condition_dim": 69,
        "motion_dim": 36,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "time_dim": args.hidden_dim,
    }
    model = GaussianDiffusion(
        TemporalDenoiser(**model_config), num_steps=args.diffusion_steps
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val_loss = float("inf")
    wandb_run = init_wandb(
        args,
        {
            "num_train_files": len(file_splits["train"]),
            "num_val_files": len(file_splits["val"]),
            "num_test_files": len(file_splits["test"]),
            "num_train_windows": len(train_dataset),
            "num_val_windows": len(val_dataset),
        },
    )
    global_step = 0

    print(
        f"设备: {device} | 文件 train/val/test: "
        f"{len(file_splits['train'])}/{len(file_splits['val'])}/{len(file_splits['test'])} | "
        f"训练窗口: {len(train_dataset)}"
    )
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            condition = batch["condition"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            loss = model.training_loss(motion, condition, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
            if wandb_run is not None and global_step % args.wandb_log_interval == 0:
                wandb_run.log(
                    {"train/global_step": global_step, "train/batch_loss": loss.item()}
                )

        train_loss /= max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config,
            "diffusion_steps": args.diffusion_steps,
            "window_size": args.window_size,
            "stride": args.stride,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "normalizer": normalizer.state_dict(),
            "file_splits": {
                name: [str(path.relative_to(args.data_dir)) for path in paths]
                for name, paths in file_splits.items()
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        epoch_seconds = time.time() - start_time
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "time/epoch_seconds": epoch_seconds,
                }
            )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | {epoch_seconds:.1f}s"
        )

    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳权重: {args.output_dir / 'best.pt'}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_checkpoint"] = str(args.output_dir / "best.pt")
        wandb_run.finish()


if __name__ == "__main__":
    main()
