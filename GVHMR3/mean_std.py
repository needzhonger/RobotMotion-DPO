import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
)


def quat_xyzw_to_wxyz(q):
    """(..., 4) xyzw -> wxyz"""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def quat_wxyz_to_xyzw(q):
    """(..., 4) wxyz -> xyzw"""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def az_to_ay_rotmat():
    """Z-up -> Y-up 旋转矩阵: (x,y,z)_new = (x,z,-y)_old"""
    return torch.tensor(
        [[1.0, 0.0,  0.0],
         [0.0, 0.0,  1.0],
         [0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )


def apply_az_to_ay_vec(v):
    """v: (..., 3) Z-up -> Y-up"""
    T = az_to_ay_rotmat()
    return v @ T.T


def apply_az_to_ay_quat_xyzw(q_xyzw):
    """q_xyzw: (..., 4) xyzw, Z-up -> Y-up"""
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    R = quaternion_to_matrix(q_wxyz)           # (..., 3, 3)
    T = az_to_ay_rotmat()
    R_new = torch.matmul(T, R)                 # 左乘：旋转世界轴
    q_wxyz_new = matrix_to_quaternion(R_new)
    return quat_wxyz_to_xyzw(q_wxyz_new)


@torch.no_grad()
def generate_g1_stats():
    """
    扫描 G1 训练 NPZ，对每帧先做 az→ay 坐标变换（与 dataset 一致），
    再组装 299D 向量:
        [joint_pos(29) | body_pos_w_flat(90) | body_quat_w_6d_flat(180)]
    计算 mean/std 并保存到 data/g1_stats.npz (keys: 'mean', 'std').
    """
    np.random.seed(0)
    torch.manual_seed(0)

    g1_dir = Path("g1_paired/train/g1")
    files = sorted(g1_dir.glob("*.npz"))
    if len(files) == 0:
        print("❌ 没有找到 G1 数据文件")
        return

    # 在线统计 (避免内存溢出)
    running_sum = None
    running_sumsq = None
    total_frames = 0
    skipped = 0

    for f in tqdm(files, desc="扫描 G1 文件"):
        try:
            data = np.load(f)
        except Exception as e:
            print(f"⚠️ 跳过 {f.name}: {e}")
            skipped += 1
            continue

        if "joint_pos" not in data or "body_pos_w" not in data or "body_quat_w" not in data:
            print(f"⚠️ 跳过 {f.name}: 缺少必需字段")
            skipped += 1
            continue

        joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32)     # (T, 29)
        body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32)   # (T, 30, 3)
        body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32) # (T, 30, 4) xyzw

        T = joint_pos.shape[0]
        if T < 2:
            skipped += 1
            continue

        # ---- 与 dataset 保持一致：先做 az->ay 坐标系变换 ----
        body_pos_w  = apply_az_to_ay_vec(body_pos_w)
        body_quat_w = apply_az_to_ay_quat_xyzw(body_quat_w)

        # xyzw → wxyz → 6D rotation
        quat_wxyz = quat_xyzw_to_wxyz(body_quat_w)              # (T, 30, 4)
        quat_6d = matrix_to_rotation_6d(
            quaternion_to_matrix(quat_wxyz)
        )                                                         # (T, 30, 6)

        # 拼接: [joint_pos(29) | body_pos_w(90) | body_quat_6d(180)] = 299D
        vec = torch.cat([
            joint_pos,                         # (T, 29)
            body_pos_w.reshape(T, -1),         # (T, 90)
            quat_6d.reshape(T, -1),            # (T, 180)
        ], dim=-1).double()                    # (T, 299)

        if running_sum is None:
            running_sum = vec.sum(dim=0)
            running_sumsq = (vec ** 2).sum(dim=0)
        else:
            running_sum += vec.sum(dim=0)
            running_sumsq += (vec ** 2).sum(dim=0)
        total_frames += T

    if total_frames == 0:
        print("❌ 没有有效数据")
        return

    mean = (running_sum / total_frames).float().numpy()
    var = (running_sumsq / total_frames - (running_sum / total_frames) ** 2).clamp(min=0).float()
    std = var.sqrt().clamp(min=1e-6).numpy()

    save_path = Path("data/g1_stats.npz")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, mean=mean, std=std)
    print(f"\n✅ 保存完成: {save_path}")
    print(f"   mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"   总帧数: {total_frames}, 处理文件: {len(files) - skipped}, 跳过: {skipped}")


if __name__ == "__main__":
    generate_g1_stats()