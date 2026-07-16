# ============================================================
# 文件: hmr4d/dataset/pure_motion/g1_amass.py
# 作用: 加载 G1-AMASS 配对数据（双文件夹结构）
#      - AMASS: 你的 npz 是 SMPL-X/markers 格式:
#          trans/root_orient/pose_body/pose_hand/pose_jaw/pose_eye/poses/betas...
#      - G1: 你的 npz:
#          joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w (+fps)
#      - 数据集会把 G1 的所有字段作为监督 g1_target 返回，模型可按同名输出
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

from .base_dataset import BaseDataset
from .cam_traj_utils import CameraAugmentorV11, StaticCameraV11
from .utils import interpolate_smpl_params
from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam, get_R_c2gv
from hmr4d.utils.geo.hmr_cam import create_camera_sensor, get_bbx_xys, perspective_projection
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from canonicalize_motion import (
    canonicalize_g1_first_frame_bones_seed,
    canonicalize_smpl_first_frame_bones_seed,
)


# ------------------------------------------------------------------
# SLERP helpers
# ------------------------------------------------------------------

def _slerp_quat(q0, q1, alpha):
    """SLERP between two quaternion tensors (wxyz). alpha in [0,1].
    q0, q1: (..., 4) wxyz.  alpha: scalar or (...,1).
    Returns: (..., 4) normalised wxyz quaternion.
    """
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)          # (..., 1)
    q1 = torch.where(dot < 0, -q1, q1)             # shortest path
    dot = dot.abs().clamp(0.0, 1.0)

    theta = torch.acos(dot)                         # (..., 1)
    sin_t = theta.sin()

    # linear fallback when theta ≈ 0
    w0 = torch.where(sin_t > 1e-6,
                     torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-10),
                     torch.ones_like(sin_t) * (1.0 - alpha))
    w1 = torch.where(sin_t > 1e-6,
                     torch.sin(alpha * theta) / (sin_t + 1e-10),
                     torch.ones_like(sin_t) * alpha)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def slerp_sequence_quat(q_seq, tgt_len):
    """Resample a quaternion sequence to tgt_len using SLERP.

    Args:
        q_seq: (T, *shape, 4) wxyz quaternions
        tgt_len: int
    Returns:
        (tgt_len, *shape, 4) wxyz quaternions
    """
    T = q_seq.shape[0]
    if T == tgt_len:
        return F.normalize(q_seq, dim=-1)
    if T == 1:
        return F.normalize(q_seq, dim=-1).expand(tgt_len, *q_seq.shape[1:]).clone()

    extra = q_seq.shape[1:]                             # (*shape, 4)
    n_extra = len(extra)

    t_idx = torch.linspace(0, T - 1, tgt_len, device=q_seq.device, dtype=q_seq.dtype)
    i0 = t_idx.floor().long().clamp(0, T - 2)          # (tgt_len,)
    i1 = (i0 + 1).clamp(0, T - 1)
    alpha = (t_idx - i0.to(dtype=q_seq.dtype)).reshape(tgt_len, *([1] * n_extra))  # broadcastable

    q0 = q_seq[i0]   # (tgt_len, *shape, 4)
    q1 = q_seq[i1]
    return _slerp_quat(q0, q1, alpha)


# ==========================================================
# 小工具：npz key 兼容
# ==========================================================
def npz_get(npz_obj, candidates, required=True, file_for_log=""):
    """
    从 npz 里按候选 key 取第一个存在的.
    """
    for k in candidates:
        if k in npz_obj.files:
            return npz_obj[k]
    if required:
        raise KeyError(
            f"Missing keys {candidates} in npz: {file_for_log}\n"
            f"Available keys: {list(npz_obj.files)}"
        )
    return None


# ==========================================================
# 插值工具
# ==========================================================
def interpolate_tensor(x, tgt_len):
    """对 1D 序列特征进行线性插值: (T, C) -> (tgt_len, C)"""
    x = x.unsqueeze(0).transpose(1, 2)  # (1, C, T)
    x_interp = F.interpolate(x, size=tgt_len, mode="linear", align_corners=True)
    return x_interp.transpose(1, 2).squeeze(0)  # (tgt_len, C)


def interpolate_seq(x, tgt_len):
    """
    通用线性插值：x shape = (T, *D)
    flatten 后插值再 reshape
    """
    T = x.shape[0]
    x_flat = x.reshape(T, -1)
    x_interp = interpolate_tensor(x_flat, tgt_len)
    return x_interp.reshape(tgt_len, *x.shape[1:])


def retime_speed_ratio(src_len, tgt_len, src_fps=None, tgt_fps=None):
    """Velocity multiplier for retiming a clip from src_len frames to tgt_len frames.

    Interpolation uses align_corners=True, so the exact duration ratio is based on
    frame intervals, not frame count. If fps is omitted, source and target are
    assumed to be sampled at the same fps. Values are multiplied by this ratio:
    - src_len > tgt_len: clip is sped up, velocities get larger
    - src_len < tgt_len: clip is slowed down, velocities get smaller
    """
    if src_len <= 1 or tgt_len <= 1:
        return 1.0
    src_fps = 1.0 if src_fps is None else float(src_fps)
    tgt_fps = src_fps if tgt_fps is None else float(tgt_fps)
    src_duration = float(src_len - 1) / src_fps
    tgt_duration = float(tgt_len - 1) / tgt_fps
    if tgt_duration <= 0:
        return 1.0
    return src_duration / tgt_duration


def quat_xyzw_to_wxyz(q):
    """输入 (...,4) 且为 xyzw，输出 wxyz（pytorch3d 需要 wxyz）"""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def quat_wxyz_to_xyzw(q):
    """输入 (...,4) wxyz，输出 xyzw"""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


# ==========================================================
# （可选）Z-up -> Y-up 坐标系旋转（批量）
# 说明：你原来只对 SMPL 根做 az->ay。这里对 G1 的 world quantities 也做同样旋转。
# 如果你确认 G1 本身已经是目标坐标系，则把这段在 _load_data 里注释掉即可。
# ==========================================================
def az_to_ay_rotmat(device, dtype):
    # Applied via `v @ T.T`: (x, y, z)_zup → (x, -z, y)_yup.
    # i.e. z-up gravity -z becomes y-up gravity -y.
    return torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def apply_az_to_ay_on_vec(v):
    """v: (..., 3)"""
    T = az_to_ay_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_az_to_ay_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4)  wxyz format (as stored in NPZ and g1_target)"""
    R = quaternion_to_matrix(q_wxyz)              # (...,3,3)
    T = az_to_ay_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = torch.matmul(T, R)                    # world-axis rotation: left-multiply
    return matrix_to_quaternion(R_new)            # back to wxyz


def apply_az_to_ay_on_axis_angle(global_orient_aa):
    T = az_to_ay_rotmat(global_orient_aa.device, global_orient_aa.dtype)
    return matrix_to_axis_angle(T @ axis_angle_to_matrix(global_orient_aa))


# ==========================================================
# 把 AMASS canonical (SMPL body local: x=left, y=up, z=fwd) 旋到
# G1     canonical (URDF root local: x=fwd, y=left, z=up)
#   x_new = z_old   (forward)
#   y_new = x_old   (left)
#   z_new = y_old   (up)
# det = +1, 右手系。首帧规范化之后左乘到 transl/global_orient,
# AMASS 输出就变成 "z-up, 面向 +x", 与 G1 同坐标系。
# 注意: 应用之后 global_orient[0] 不再是零向量, 而是 R_post 对应的 axis-angle
#       (≈ (1.209, 1.209, 1.209), 即绕 [1,1,1]/√3 转 120°). 这是物理正确的.
# ==========================================================
R_SMPL_LOCAL_TO_G1_LOCAL = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)

# ------------------------------------------------------------------
# 地面高度计算: T-pose 近似
# 脚踝 7=l_ankle 8=r_ankle, 趾尖 10=l_foot 11=r_foot
# ------------------------------------------------------------------
_SMPLX_FOOT_JOINT_IDS = (7, 8, 10, 11)
_SMPLX_FOOT_CACHE: dict = {}


def get_smplx_foot_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    """返回 (J_template_feet (4,3), J_shapedirs_feet (4,3,B))，对应四个脚部关节。"""
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_FOOT_CACHE:
        return _SMPLX_FOOT_CACHE[key]
    data = np.load(smplx_path, allow_pickle=True)
    v_template  = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)
    shapedirs   = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)
    rows = J_regressor[list(_SMPLX_FOOT_JOINT_IDS)]               # (4, V)
    J_template_feet  = rows @ v_template                            # (4, 3)
    J_shapedirs_feet = torch.einsum("jv,vcb->jcb", rows, shapedirs) # (4, 3, B)
    _SMPLX_FOOT_CACHE[key] = (J_template_feet, J_shapedirs_feet)
    return _SMPLX_FOOT_CACHE[key]


def compute_smpl_floor_height(betas, J_template_pelvis, J_shapedirs_pelvis,
                               J_template_feet, J_shapedirs_feet):
    """T-pose 近似下的地面高度 (SMPL y-up local 坐标系, 单位 m).

    系统统一 y-up: y 轴即竖直方向.
    pelvis 被放在 world 原点 (y=0), 脚在 y<0 的位置 (T-pose).
    floor_height = -(脚的 y) = pelvis_y_smpl - min(foot_y_smpl)
    返回正数, 即 pelvis 关节距地面的高度.
    """
    pelvis_y = float((J_template_pelvis + J_shapedirs_pelvis @ betas)[1])
    feet_pos  = J_template_feet + torch.einsum("jcb,b->jc", J_shapedirs_feet, betas)
    min_foot_y = float(feet_pos[:, 1].min())
    return pelvis_y - min_foot_y               # > 0




def apply_smpl_to_g1_axes_on_smpl(global_orient_aa, transl, pelvis_offset_smpl=None):
    """把世界坐标系从 SMPL y-up 旋转到 G1 z-up (R_post 左乘所有世界量).

    关键约定: SMPL FK 满足  pelvis_world(f) = transl(f) + J[0]_rest(betas),
    其中 J[0]_rest 在 SMPL native local frame (y-up), 不受 global_orient 影响.

    所以单纯 transl @ R_post.T 不能让 pelvis_world 跟随 R_post 旋转:
        pelvis_world_new = transl_new + J[0]_rest = R_post@transl + J[0]_rest
        我们想要:         R_post @ pelvis_world_old = R_post@transl + R_post@J[0]_rest
        差额 = (R_post - I) @ J[0]_rest, 必须补在 transl 上.

    若 pelvis_offset_smpl 给定 (y-up SMPL local 下 J[0]_rest), 函数会自动补偿,
    保证 pelvis_world 整段轨迹按 R_post 旋转 → 首帧 pelvis_world 仍在原点
    (前提是输入已经 canonicalize 到 pelvis 在原点).
    """
    R_post = R_SMPL_LOCAL_TO_G1_LOCAL.to(device=transl.device, dtype=transl.dtype)
    R_orient = axis_angle_to_matrix(global_orient_aa)
    R_orient_new = R_post.unsqueeze(0) @ R_orient
    global_orient_aa_new = matrix_to_axis_angle(R_orient_new)
    if pelvis_offset_smpl is not None:
        delta = (R_post @ pelvis_offset_smpl) - pelvis_offset_smpl   # (3,)
        transl_new = transl @ R_post.T + delta.unsqueeze(0)          # (F, 3)
    else:
        transl_new = transl @ R_post.T
    return global_orient_aa_new, transl_new, None


# ==========================================================
# 加载 SMPL-X pelvis 行（J_template[0] 与 J_shapedirs[0]），用于把 SMPL pelvis 关节
# 真正放到原点（而不是只把 transl 参数放到原点）。
#   pelvis_world(f) = transl(f) + J_pelvis(betas)
# 其中 J_pelvis(betas) = J_template[0] + J_shapedirs[0] @ betas，与 global_orient 无关。
# ==========================================================
_SMPLX_PELVIS_CACHE = {}


def get_smplx_pelvis_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    """返回 (J_template_pelvis (3,), J_shapedirs_pelvis (3, num_betas))。整库缓存。"""
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_PELVIS_CACHE:
        return _SMPLX_PELVIS_CACHE[key]

    data = np.load(smplx_path, allow_pickle=True)
    v_template = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)                       # (V, 3)
    shapedirs  = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)      # (V, 3, B)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)                     # (J, V)

    pelvis_row = J_regressor[0]                                                  # (V,)
    J_template_pelvis  = pelvis_row @ v_template                                 # (3,)
    J_shapedirs_pelvis = torch.einsum("v, vcb -> cb", pelvis_row, shapedirs)     # (3, B)

    _SMPLX_PELVIS_CACHE[key] = (J_template_pelvis, J_shapedirs_pelvis)
    return _SMPLX_PELVIS_CACHE[key]


def compute_smpl_pelvis_offset(betas, J_template_pelvis, J_shapedirs_pelvis):
    """betas: (..., B). 返回 (..., 3) pelvis 在 canonical 模型系下的位置。"""
    return J_template_pelvis + torch.einsum("...b, cb -> ...c", betas, J_shapedirs_pelvis)


# ==========================================================
# 首帧归一化：把第 0 帧 root 平移到原点，旋转整段使第 0 帧 root 朝向 = identity
#
# 对 SMPL：pelvis_world(f) = transl(f) + J_pelvis(betas)，global_orient 不影响 pelvis 位置。
# 把世界旋转 R0^T 后，再平移使 pelvis_world(0) = 0，最终：
#     global_orient_new(f) = R0^T @ global_orient(f)
#     transl_new(f)        = R0^T @ (transl(f) - transl(0)) - J_pelvis(betas)
# 验证：pelvis_world_new(0) = transl_new(0) + J_pelvis = -J_pelvis + J_pelvis = 0 ✓
#       global_orient_new(0) = I ✓
#
# 对 G1：body_pos_w / body_quat_w 是物理位姿（不存在 J_canonical 概念），直接绕首帧 root
# 应用同一刚体变换即可，最终 body_pos_w[0, root]=0、body_quat_w[0, root]=identity。
# 这两套归一化合起来 → SMPL pelvis 关节 与 G1 root link 在第 0 帧都在原点、同朝向。
# ==========================================================
def _first_frame_inv_rot(R0, yaw_only=False, up_axis=1, fwd_axis=2):
    """给定首帧旋转 R0 (3,3)，返回要左乘到全帧上的逆旋转 R0_inv (3,3)。

    - yaw_only=False: 旧行为，返回 R0^T（去掉整个首帧旋转，含 pitch/roll）。
    - yaw_only=True : 只去掉绕竖直轴 (up_axis) 的 yaw(航向)，保留 pitch/roll，
                      从而首帧后重力方向不被转歪。航向定义与 hmr4d.utils.matrix.calc_heading
                      一致：取 R0 的 forward 列 (fwd_axis)，在水平面用 atan2 求航向角。
        up_axis=1 (y-up): heading = atan2(fwd[0], fwd[2])
        up_axis=2 (z-up): heading = atan2(fwd[1], fwd[0])
        up_axis=0 (x-up): heading = atan2(fwd[2], fwd[1])
      R0_inv = axis_angle_to_matrix(-heading * e_up)（绕 up 轴转 -heading）。
    """
    if not yaw_only:
        return R0.transpose(-1, -2)
    fwd = R0[:, fwd_axis]                                  # (3,) 世界系 forward
    if up_axis == 1:
        heading = torch.atan2(fwd[0], fwd[2])
    elif up_axis == 2:
        heading = torch.atan2(fwd[1], fwd[0])
    else:  # up_axis == 0
        heading = torch.atan2(fwd[2], fwd[1])
    aa = torch.zeros(3, device=R0.device, dtype=R0.dtype)
    aa[up_axis] = -heading
    return axis_angle_to_matrix(aa)                       # (3, 3) 纯绕 up 轴旋转


def canonicalize_smpl_first_frame(global_orient_aa, transl, pelvis_offset=None,
                                  yaw_only=False, up_axis=1, fwd_axis=2):
    """全帧应用 R0^T（或仅去 yaw 的 R0_inv）。
    Args:
        global_orient_aa: (F, 3) 轴角
        transl:          (F, 3)
        pelvis_offset:   (3,) 可选，= J_pelvis(betas)。给定时让第 0 帧 pelvis 关节落到原点；
                         不给则退化为"让第 0 帧 transl=0"的旧口径。
        yaw_only:        True 时只去首帧航向(绕 up_axis 的 yaw)，保留 pitch/roll，重力不被转歪。
        up_axis/fwd_axis: 竖直轴 / forward 轴索引。SMPL 在 y-up 系: up=1(y), fwd=2(+z)。
    """
    R = axis_angle_to_matrix(global_orient_aa)            # (F, 3, 3)
    R0_inv = _first_frame_inv_rot(R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)  # (3, 3)
    R_new = R0_inv @ R                                    # (F, 3, 3)
    transl_new = (transl - transl[0:1]) @ R0_inv.T        # (F, 3) — 此时 transl_new[0] = 0

    if pelvis_offset is not None:
        # 此时 pelvis_world[0] = 0 + pelvis_offset = pelvis_offset，再减去使其归零。
        transl_new = transl_new - pelvis_offset
    return matrix_to_axis_angle(R_new), transl_new


def canonicalize_g1_first_frame(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    """对 G1 全字段应用 R0^T（或仅去 yaw 的 R0_inv，绕原点），使第 0 帧 root_body 位置=0。
    g1 fields:
        body_pos_w:     (F, N, 3)
        body_quat_w:    (F, N, 4) wxyz
        body_lin_vel_w: (F, N, 3)
        body_ang_vel_w: (F, N, 3)
    其它字段（joint_pos/joint_vel/fps）属于关节空间，不受世界刚体变换影响。

    yaw_only=False: 旧行为，首帧 root 姿态归 identity。
    yaw_only=True : 只去首帧航向(绕 up_axis 的 yaw)，保留 pitch/roll，重力不被转歪；
                    此时**不再**把首帧 root quat 钉成 identity（否则会抹掉保留的 tilt）。
    up_axis/fwd_axis: 竖直轴 / forward 轴。G1 在 z-up 系: up=2(z), fwd=0(URDF forward=+x)。
    """
    body_quat_w   = g1["body_quat_w"]                                  # (F, N, 4) wxyz
    body_pos_w    = g1["body_pos_w"]                                   # (F, N, 3)

    R = quaternion_to_matrix(body_quat_w)                              # (F, N, 3, 3)
    R0_inv = _first_frame_inv_rot(R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)  # (3, 3)
    center = body_pos_w[0, root_body_id].clone()                       # (3,)

    R_new = R0_inv @ R                                                 # (F, N, 3, 3)
    q_new = matrix_to_quaternion(R_new)                                # (F, N, 4) wxyz, ambiguous ±q
    # 规范双覆盖：让 real(w) 部分 ≥ 0，得到确定符号
    sign = torch.where(q_new[..., 0:1] < 0, -torch.ones_like(q_new[..., 0:1]), torch.ones_like(q_new[..., 0:1]))
    q_new = q_new * sign
    if not yaw_only:
        # 显式钉死首帧 root 为 identity，避免 1e-7 量级的浮点噪声。
        # yaw_only 下首帧 root 仍带 pitch/roll，不能钉成 identity。
        q_new[0, root_body_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype)

    g1["body_quat_w"]    = q_new                                       # (F, N, 4) wxyz
    g1["body_pos_w"]     = (body_pos_w - center) @ R0_inv.T            # (F, N, 3)
    g1["body_lin_vel_w"] = g1["body_lin_vel_w"] @ R0_inv.T             # (F, N, 3)
    g1["body_ang_vel_w"] = g1["body_ang_vel_w"] @ R0_inv.T             # (F, N, 3)
    # 显式钉死首帧 root 位置为 0
    g1["body_pos_w"][0, root_body_id] = 0.0
    return g1


# ==========================================================
# G1 插值（全字段时间重采样）
# joint_pos / body_pos_w → 线性插值；velocities → 线性插值后按重定时比例缩放
# body_quat_w → SLERP（球面线性插值）
# ==========================================================
def interpolate_g1_params(g1_data, tgt_len, speed_ratio=1.0, fps_out=None):
    """G1 全字段时间重采样.

    Args:
        speed_ratio: 重定时比 (src_duration / tgt_duration). joint_vel / ang_vel
            原本是 m/s 量纲, 拉伸/压缩时间后用 speed_ratio 缩放保持 m/s.
        fps_out: 重采样后输出的 fps. 给定则 body_lin_vel_w / joint_vel 改成
            "从对应 pos 差分重算" (确保 vel 与 pos 严格一致, 单位 m/s 或 rad/s).
            None 时退回老逻辑.
    """
    speed_ratio = float(speed_ratio)

    # --- joint DOFs and positions: linear ---
    joint_pos = interpolate_seq(g1_data["joint_pos"], tgt_len)

    # global translation: linear
    body_pos_w = interpolate_seq(g1_data["body_pos_w"], tgt_len)

    # global orientation: already wxyz in NPZ → SLERP → keep as wxyz
    quat_wxyz = g1_data["body_quat_w"]                       # (T, N, 4) wxyz
    body_quat_w_i = slerp_sequence_quat(quat_wxyz, tgt_len)  # (tgt_len, N, 4) wxyz

    # Velocities: 优先从 interpolate 后的 pos 差分重算, 保证 vel 跟 pos 严格一致.
    # joint_vel / body_lin_vel_w 都用 finite diff. body_ang_vel_w 仍走 npz × speed_ratio
    # (因为 ang_vel 需要从 quat 差分换算, 实现成本高且当前训练无人 consume).
    if fps_out is not None and tgt_len >= 2:
        # joint_vel: rad/s
        jp_diff = joint_pos[1:] - joint_pos[:-1]                         # (tgt_len-1, 29)
        jp_diff = torch.cat([jp_diff, jp_diff[-1:].clone()], dim=0)      # (tgt_len, 29)
        joint_vel = jp_diff * float(fps_out)
        # body_lin_vel_w: m/s
        diff = body_pos_w[1:] - body_pos_w[:-1]
        diff = torch.cat([diff, diff[-1:].clone()], dim=0)
        body_lin_vel_w = diff * float(fps_out)
    else:
        joint_vel = interpolate_seq(g1_data["joint_vel"], tgt_len) * speed_ratio
        body_lin_vel_w = interpolate_seq(g1_data["body_lin_vel_w"], tgt_len) * speed_ratio
    body_ang_vel_w = interpolate_seq(g1_data["body_ang_vel_w"], tgt_len) * speed_ratio

    return {
        "fps": g1_data.get("fps", None),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w_i,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


# ==========================================================
# 核心数据集类
# ==========================================================
class G1AmassDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        amass_subdir="amass",
        g1_subdir="g1",
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=False,  # True: 竖直方向平移使脚踩地 (y=0), pelvis/root y > 0
        # 首帧归一化只去 yaw(航向), 保留 pitch/roll, 使重力方向严格保持 -y, 与 gravity-view
        # 编码自洽 (与 G1AmassDualPthDataset 同口径). False 时退回旧的"去整个 R0"行为.
        yaw_only_canon=True,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ):
        self.data_dir = Path(data_dir)
        self.amass_dir = self.data_dir / amass_subdir
        self.g1_dir = self.data_dir / g1_subdir
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.dataset_name = "G1_AMASS"
        self.betas_dim = betas_dim
        self.root_body_id = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust = floor_adjust
        self.yaw_only_canon = bool(yaw_only_canon)
        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        # G1 npz 是固定 betas 物理仿真, 关闭父类的 SMPL betas 加噪 (std=0.1) 防止
        # SMPL pelvis offset 抖动 ~3cm 跟固定 G1 root 错位.
        self._skip_betas_aug = True

        super().__init__(cam_augmentation, limit_size)

    # =====================================================
    # 扫描配对文件
    # =====================================================
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] 扫描 AMASS: {self.amass_dir}")
        Log.info(f"[{self.dataset_name}] 扫描 G1:    {self.g1_dir}")

        amass_files = {f.name for f in self.amass_dir.glob("*.npz")}
        g1_files = {f.name for f in self.g1_dir.glob("*.npz")}

        paired_names = sorted(amass_files & g1_files)

        amass_only = amass_files - g1_files
        g1_only = g1_files - amass_files
        if amass_only:
            Log.warning(f"[{self.dataset_name}] {len(amass_only)} 个 AMASS 文件无配对: {list(amass_only)[:5]}...")
        if g1_only:
            Log.warning(f"[{self.dataset_name}] {len(g1_only)} 个 G1 文件无配对: {list(g1_only)[:5]}...")

        self.paired_files = [
            {"amass": self.amass_dir / name, "g1": self.g1_dir / name, "name": Path(name).stem}
            for name in paired_names
        ]
        Log.info(f"[{self.dataset_name}] 成功配对 {len(self.paired_files)} 个动作文件。")

    # =====================================================
    # 切分长动作序列为短片段
    # =====================================================
    def _get_idx2meta(self):
        # 同原版 GVHMR AmassDataset：每条序列按 max(L // motion_frames, 1) 次重复，
        # 起点固定为 0，_load_data 内做整段随机变速截取，G1 / AMASS 永远同步切片。
        self.idx2meta = []
        skipped = 0

        amass_fps_missing = 0
        for pair in self.paired_files:
            with np.load(pair["amass"], allow_pickle=False) as amass_data, np.load(pair["g1"]) as g1_data:
                amass_len = npz_get(amass_data, ["trans", "transl"], file_for_log=str(pair["amass"])).shape[0]
                amass_fps_val = npz_get(
                    amass_data,
                    ["mocap_framerate", "mocap_frame_rate", "fps"],
                    required=False,
                    file_for_log=str(pair["amass"]),
                )
                if amass_fps_val is None:
                    amass_fps = 120.0
                    amass_fps_missing += 1
                else:
                    amass_fps = float(amass_fps_val)

                g1_len = npz_get(g1_data, ["joint_pos"], file_for_log=str(pair["g1"])).shape[0]
                g1_fps = float(g1_data["fps"]) if "fps" in g1_data.files else 50.0

            fps_ratio = float(amass_fps) / float(g1_fps)
            amass_len_in_g1_frames = int(amass_len / fps_ratio)
            length = min(amass_len_in_g1_frames, g1_len)
            if length < 25:
                skipped += 1
                continue

            num_samples = max(length // self.motion_frames, 1)
            meta = {
                "amass_file": pair["amass"],
                "g1_file":    pair["g1"],
                "usable_len": length,        # G1 帧数（基准）
                "fps_ratio":  fps_ratio,     # amass_fps / g1_fps
                "name":       pair["name"],
            }
            self.idx2meta.extend([meta] * num_samples)

        Log.info(f"[{self.dataset_name}] 切分完毕: {len(self.idx2meta)} 个片段, 跳过 {skipped} 个过短动作。")
        if amass_fps_missing > 0:
            Log.warning(
                f"[{self.dataset_name}] {amass_fps_missing}/{len(self.paired_files)} 个 AMASS 文件缺少 "
                f"mocap_framerate/mocap_frame_rate/fps 字段, 已回退到 120fps。"
                f"如果实际帧率不是 120, fps_ratio 会算错, 训练时 AMASS 与 G1 时序对不齐。"
            )

    # =====================================================
    # 读取数据 + 随机变速截取 + 插值
    # =====================================================
    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]     # G1 帧数（基准）
        fps_ratio  = meta["fps_ratio"]      # amass_fps / g1_fps
        tgt_len    = self.motion_frames

        # 速度增强：物理时长范围对齐原版 GVHMR。
        # raw_subset_len 在 G1 帧空间，除以 fps_ratio 才能得到与原版相同的物理时长。
        lo = max(2,      int(float(tgt_len) / self.l_factor / fps_ratio))
        hi = max(lo + 1, int(float(tgt_len) * self.l_factor / fps_ratio))
        raw_subset_len = np.random.randint(lo, hi)
        if raw_subset_len <= usable_len:
            start_g1 = np.random.randint(0, usable_len - raw_subset_len + 1)
            end_g1   = start_g1 + raw_subset_len
        else:
            start_g1, end_g1 = 0, usable_len

        # AMASS 帧索引：按 fps 比值同步换算 → G1 / AMASS 切到同一物理时段
        start_amass = int(round(start_g1 * fps_ratio))
        end_amass   = int(round(end_g1   * fps_ratio))
        if end_amass - start_amass < 2:
            end_amass = start_amass + 2

        # 兼容旧代码变量名（G1 用 start/end，AMASS 用 start_amass/end_amass）
        start = start_g1
        end   = end_g1

        with np.load(meta["amass_file"], allow_pickle=False) as amass_raw, np.load(meta["g1_file"]) as g1_raw:
            # -------------------------
            # AMASS（你的格式）
            # -------------------------
            # 必须存在：trans/root_orient/pose_body/betas（或用 poses 切片也行）
            # AMASS 用 start_amass/end_amass（已按 fps 比值换算）
            trans = npz_get(amass_raw, ["trans", "transl"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            root_orient = npz_get(amass_raw, ["root_orient", "global_orient"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            pose_body = npz_get(amass_raw, ["pose_body", "body_pose"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            betas = npz_get(amass_raw, ["betas"], file_for_log=str(meta["amass_file"]))

            smpl_data = {
                # 统一成你工程里常用的 key
                "transl": torch.tensor(trans, dtype=torch.float32),
                "global_orient": torch.tensor(root_orient, dtype=torch.float32),
                "body_pose": torch.tensor(pose_body, dtype=torch.float32),
                "betas": torch.tensor(betas, dtype=torch.float32),
            }

            # betas: (16,) -> (T,16)
            actual_len = end_amass - start_amass
            if smpl_data["betas"].ndim == 1:
                smpl_data["betas"] = smpl_data["betas"][: self.betas_dim]
                smpl_data["betas"] = smpl_data["betas"].unsqueeze(0).expand(actual_len, -1)
            else:
                smpl_data["betas"] = smpl_data["betas"][:actual_len, : self.betas_dim]

            # -------------------------
            # G1（你的格式）
            # -------------------------
            required_g1_keys = [
                "joint_pos", "joint_vel",
                "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w",
            ]
            missing = [k for k in required_g1_keys if k not in g1_raw.files]
            if missing:
                raise KeyError(
                    f"[G1 npz 格式不对] 文件: {meta['g1_file']}\n"
                    f"缺少 keys: {missing}\n"
                    f"实际 keys: {list(g1_raw.files)}\n"
                    f"（这通常意味着：你配对到了错误的 npz，或者 g1 文件还没按预期导出）"
                )

            g1_data = {
                "fps": torch.tensor(float(g1_raw["fps"]), dtype=torch.float32) if "fps" in g1_raw.files else None,
                "joint_pos": torch.tensor(g1_raw["joint_pos"][start:end], dtype=torch.float32),
                "joint_vel": torch.tensor(g1_raw["joint_vel"][start:end], dtype=torch.float32),
                "body_pos_w": torch.tensor(g1_raw["body_pos_w"][start:end], dtype=torch.float32),
                "body_quat_w": torch.tensor(g1_raw["body_quat_w"][start:end], dtype=torch.float32),  # wxyz
                "body_lin_vel_w": torch.tensor(g1_raw["body_lin_vel_w"][start:end], dtype=torch.float32),
                "body_ang_vel_w": torch.tensor(g1_raw["body_ang_vel_w"][start:end], dtype=torch.float32),
            }

        # 插值到固定帧数
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
        g1_interpolated = interpolate_g1_params(g1_data, tgt_len)

        # AMASS 坐标系转换: 原始 AMASS npz 是 z-up, 转到 y-up (与 dualpth filtered_amass 同系).
        smpl_interpolated["global_orient"], smpl_interpolated["transl"], _ = get_tgtcoord_rootparam(
            smpl_interpolated["global_orient"],
            smpl_interpolated["transl"],
            tsf="az->ay",
        )

        # ---- 对齐链路与 G1AmassDualPthDataset 完全一致 (refined 口径) ----
        # G1 npz 是 z-up (IsaacGym/IsaacLab 默认). 顺序:
        #   1. 先在 z-up native 系下 canonicalize G1 first frame (只去 yaw, 保留 pitch/roll),
        #      R0_inv 绕 z 轴, 把 G1 root 朝向校到 URDF forward=+x. 必须在 az_to_ay 之前,
        #      否则 y-up 下 R0_inv 会含 "URDF up(z)→world up(y)" 的 90° pitch 被错误一并撤销.
        #   2. az_to_ay 把 G1 整段 z-up → y-up.
        #   3. R_y(-90°) 把 G1 forward +x → +z, 与 SMPL canonical forward 重合 (保持 +y 不变).
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(
                g1_interpolated, root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon, up_axis=2, fwd_axis=0,   # G1 z-up, URDF forward=+x
            )

        if self.apply_az_to_ay_g1:
            g1_interpolated["body_pos_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

            g1_interpolated["body_pos_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = _apply_ry_neg90_on_quat_wxyz(g1_interpolated["body_quat_w"])

        # SMPL 首帧归一化 (pelvis 落原点 + 只去 yaw): SMPL y-up, forward=+z.
        # 这样 SMPL pelvis 与 G1 root 在第 0 帧都在原点、同朝向, 重力方向严格保持 -y.
        if self.canonicalize_first_frame:
            # pelvis 模式：减掉 J_pelvis(betas)，让 SMPL pelvis 关节真正落到原点。
            # transl 模式：pelvis_offset=None，旧口径（让 transl 落到原点）。
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
            )
        else:
            # 退路：保持旧行为——只把 G1 root 平移到 SMPL transl[0]。
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        # 地面高度校正: 把脚踩到 y=0, pelvis/root 竖直坐标 (y) > 0
        if self.floor_adjust and self.canonicalize_first_frame:
            # G1: 第 0 帧最低 body link 的 y 就是地面
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            # SMPL: 用 frame 0 实际 FK 找最低关节 (T-pose 估算对非站立 pose 不准)
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        # 返回：SMPL 给父类处理；G1 全字段给模型监督（在 _process_data 里移到 g1_target）
        return {
            "data_name": "g1_paired",
            # human
            "body_pose": smpl_interpolated["body_pose"],
            "betas": smpl_interpolated["betas"],
            "global_orient": smpl_interpolated["global_orient"],
            "transl": smpl_interpolated["transl"],

            # g1 (all)
            "g1_fps": g1_interpolated["fps"] if g1_interpolated["fps"] is not None else torch.tensor(0.0),
            "g1_joint_pos": g1_interpolated["joint_pos"],
            "g1_joint_vel": g1_interpolated["joint_vel"],
            "g1_body_pos_w": g1_interpolated["body_pos_w"],
            "g1_body_quat_w": g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    def _process_data(self, data, idx):
        # Pop 全部 G1 字段，避免父类相机增强影响
        _ = data.pop("g1_fps")
        g1_joint_pos = data.pop("g1_joint_pos")            # (T, 29)
        g1_joint_vel = data.pop("g1_joint_vel")             # (T, 29)
        g1_body_pos_w = data.pop("g1_body_pos_w")           # (T, 30, 3)
        g1_body_quat_w = data.pop("g1_body_quat_w")         # (T, 30, 4) wxyz
        g1_body_lin_vel_w = data.pop("g1_body_lin_vel_w")   # (T, 30, 3)
        g1_body_ang_vel_w = data.pop("g1_body_ang_vel_w")   # (T, 30, 3)

        g1_target = {
            "g1_joint_pos": g1_joint_pos,       # (T, 29)
            "g1_joint_vel": g1_joint_vel,       # (T, 29)
            "g1_body_pos_w": g1_body_pos_w,     # (T, 30, 3)
            "g1_body_quat_w": g1_body_quat_w,   # (T, 30, 4) wxyz
            "g1_body_lin_vel_w": g1_body_lin_vel_w,  # (T, 30, 3)
            "g1_body_ang_vel_w": g1_body_ang_vel_w,  # (T, 30, 3)
        }

        return_data = super()._process_data(data, idx)
        # G1 专属相机: 拷贝 AMASS 的 T_w2c (已 repeat_to_max_len), 把相机位置整体缩放 0.9
        # (绝对值缩放, 非 delta 缩放). R / K 保持不变.
        #   - 每一帧 t_g1[f] = 0.9 * t_amass[f]
        # static 相机 (40%) 因 t 是常量, 也整体乘 0.9 → G1 相机离原点更近
        g1_T_w2c = return_data["T_w2c"].clone()
        assert g1_T_w2c.shape[0] == return_data["T_w2c"].shape[0]
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target

        # repeat_to_max_len 所有 g1_target 字段, 跟 T_w2c/R_c2gv/cam_angvel 一样规整.
        # 当前 motion_frames == length 时是 no-op, 但若日后引入可变长度采样不会爆.
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)

        # NOTE: 不要把 dataset 端的 bbx_xys 替换成 G1 投影!
        # 设计是: AMASS 相机 (T_w2c) 配 SMPL synth bbox 给 cliff_cam / obs 用,
        #         G1 相机 (g1_T_w2c) 配 robot_bbx_xys 给 j2d / transl_c loss 用.
        # 让 gvhmr_pl 自己跑 SMPL synth bbox 路径 (mask[bbx_xys]=False 保持 placeholder).

        return return_data


# ==========================================================
# 基于 GVHMR 预处理 .pth 的 AMASS 配对版本
#   - AMASS 数据来自 smplxpose_v2.pth (已下采样到 30fps，pose 为 (F,66))
#   - G1 数据仍是原始 npz (50fps)
#   - fps_ratio = 30 / g1_fps (典型 = 0.6)
# ==========================================================
import re


def _normalize_underscores(s):
    """合并连续下划线，处理 G1 retarget 工具与 .pth 命名差异。"""
    return re.sub(r"_+", "_", s)


def _candidates_from_pth_key(k):
    """
    .pth key 格式: inputs/smplx_amass/<flavor>/<Subset>/[<Subset_or_Sub2>/]<rest>.npz
    G1 文件名: <Subset>_<rest_with_underscores>.npz 或 <Sub2>_<rest>.npz
    生成两种候选，匹配时双下划线合并为单。
    """
    parts = k.split("/")
    cands = set()
    cands.add("_".join(parts[3:]))      # 含 parts[3]: 例如 BMLrub_rub001_xxx
    if len(parts) > 4:
        cands.add("_".join(parts[4:]))  # 跳过 parts[3]: 例如 ACCAD_Female1Running_xxx
    return {_normalize_underscores(c) for c in cands}


class G1AmassPthDataset(BaseDataset):
    """
    与 G1AmassDataset 接口一致，但 AMASS 端从 GVHMR 预处理好的 .pth 读取。

    .pth schema (每条 entry):
        pose:  (F, 66) float32  = [root_orient(3), body_pose(63)]
        trans: (F, 3) float32
        beta:  (10,)   float32
        model / gender / file_name (string, 不用)
    .pth 已下采样到 30fps，没有 mocap_framerate 字段。
    """

    def __init__(
        self,
        data_dir,
        pth_path,
        g1_subdir="g1",
        amass_fps=30.0,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        skip_moyo=True,
        skip_substrings=("weiz",),
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=False,
        # ratio_lo/ratio_hi: per-axis range for k=G1/AMASS LSQ slope (x, y, z)
        # 默认 z 轴不约束 (G1 vertical bobbing 与 AMASS 关联弱, 容易误拒)
        ratio_lo=(0.7, 0.7, -float("inf")),
        ratio_hi=(1.05, 1.05,  float("inf")),
        ratio_thr=0.1,                # 同时用作 per-frame 与 per-seq 阈值 (m)
        ratio_n_retry=3,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ):
        self.data_dir = Path(data_dir)
        self.g1_dir = self.data_dir / g1_subdir
        self.pth_path = Path(pth_path)
        self.amass_fps = float(amass_fps)
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.dataset_name = "G1_AMASS_PTH"
        self.betas_dim = betas_dim
        self.root_body_id = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.skip_moyo = skip_moyo
        # 子串黑名单 (大小写不敏感), 任一命中就跳过该 pth key.
        # 默认含 'weiz' (Weizmann); skip_moyo=True 时另把 'moyo_smplxn' 加进去.
        _sk = [str(s) for s in (skip_substrings or ())]
        if skip_moyo and not any("moyo" in s.lower() for s in _sk):
            _sk.append("moyo_smplxn")
        self.skip_substrings = tuple(_sk)
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust = floor_adjust
        # Root translation ratio filter (k = G1 / AMASS, per axis LSQ slope).
        # 三轴都要在各自 [ratio_lo[axis], ratio_hi[axis]] 内才接受这条 slice; 否则重新采样
        # 同条 motion 重试 ratio_n_retry 次仍失败 → 跳到下一个 idx.
        # AMASS 在某轴 max|disp| < ratio_thr 视为该轴近似静止, 自动放行该轴.
        # 坐标轴: post-canonicalize 是 G1 z-up local, 所以 z 是垂直轴 (G1 bobbing<AMASS).
        ratio_lo = tuple(float(v) for v in ratio_lo)
        ratio_hi = tuple(float(v) for v in ratio_hi)
        assert len(ratio_lo) == 3 and len(ratio_hi) == 3, "ratio_lo/hi 必须是 3 元组 (x,y,z)"
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.ratio_thr = float(ratio_thr)
        self.ratio_n_retry = int(ratio_n_retry)
        assert self.ratio_n_retry >= 1, "ratio_n_retry 至少为 1, 否则 fallback 没有 last_data"
        self._ratio_stats = {"accept": 0, "retry": 0, "skip": 0}
        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        # G1 npz 是固定 betas 物理仿真, 关闭父类的 SMPL betas 加噪 (std=0.1) 防止
        # SMPL pelvis offset 抖动 ~3cm 跟固定 G1 root 错位.
        self._skip_betas_aug = True

        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] 加载 .pth: {self.pth_path}")
        tic = Log.time()
        self.motion_files = torch.load(self.pth_path, weights_only=False)
        Log.info(
            f"[{self.dataset_name}] .pth 总条目: {len(self.motion_files)}, "
            f"耗时 {Log.time() - tic:.1f}s"
        )

        # 建立: 规范化 G1 文件名 → .pth key
        self._g1name_to_pth_key = {}
        ambiguous = 0
        skipped = 0
        for k in self.motion_files.keys():
            if any(s.lower() in k.lower() for s in self.skip_substrings):
                skipped += 1
                continue
            for c in _candidates_from_pth_key(k):
                if c in self._g1name_to_pth_key:
                    ambiguous += 1
                else:
                    self._g1name_to_pth_key[c] = k
        Log.info(
            f"[{self.dataset_name}] 候选索引: {len(self._g1name_to_pth_key)}, "
            f"歧义跳过: {ambiguous}, 黑名单跳过: {skipped} "
            f"(skip_substrings={self.skip_substrings})"
        )

        # 扫描 G1 文件
        Log.info(f"[{self.dataset_name}] 扫描 G1: {self.g1_dir}")
        g1_files = sorted(self.g1_dir.glob("*.npz"))

        self.paired_files = []
        unmatched = []
        for g1_path in g1_files:
            norm = _normalize_underscores(g1_path.name)
            pth_key = self._g1name_to_pth_key.get(norm)
            if pth_key is None:
                unmatched.append(g1_path.name)
                continue
            self.paired_files.append(
                {"amass_key": pth_key, "g1": g1_path, "name": g1_path.stem}
            )

        Log.info(
            f"[{self.dataset_name}] 配对成功 {len(self.paired_files)}/{len(g1_files)}"
        )
        if unmatched:
            Log.warning(
                f"[{self.dataset_name}] {len(unmatched)} 个 G1 文件无 .pth 匹配, "
                f"前 3 个: {unmatched[:3]}"
            )

    def _get_idx2meta(self):
        # 时间基准：amass_fps（30fps，跟 .pth 同步）。
        # G1 npz 源 fps（典型 50）→ 等效 30fps 帧数 = round(g1_len_src × 30 / g1_fps)。
        # _load_data 里把 G1 整体从 50fps resample 到 30fps，再跟 AMASS 同步切片。
        # 这样 sampling 与原版 GVHMR AmassDataset 完全对齐（都在 30fps 空间）。
        self.idx2meta = []
        skipped = 0

        # 缓存 G1 npz 的 (joint_pos.shape[0], fps) 到 disk, 避免每次启动都打开 7k+ 文件.
        # cache key 含 g1_dir 路径与文件 mtime, 防止文件改了用脏缓存.
        import json, hashlib
        cache_dir = Path(".cache_g1_meta")
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.md5(str(self.g1_dir.resolve()).encode()).hexdigest()[:12]
        cache_file = cache_dir / f"g1_meta_{cache_key}.json"
        g1_meta_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    g1_meta_cache = json.load(f)
            except Exception:
                g1_meta_cache = {}

        cache_dirty = False
        for pair in self.paired_files:
            entry = self.motion_files[pair["amass_key"]]
            amass_len = entry["pose"].shape[0]   # 30fps 帧数

            g1_path_str = str(pair["g1"])
            mtime = pair["g1"].stat().st_mtime
            ck = f"{g1_path_str}::{mtime}"
            if ck in g1_meta_cache:
                g1_len_src, g1_fps_src = g1_meta_cache[ck]
                g1_fps_src = float(g1_fps_src)
            else:
                with np.load(pair["g1"]) as g1_data:
                    g1_len_src = int(g1_data["joint_pos"].shape[0])
                    g1_fps_src = float(g1_data["fps"]) if "fps" in g1_data.files else 50.0
                g1_meta_cache[ck] = [g1_len_src, g1_fps_src]
                cache_dirty = True

            # G1 在 30fps 下的等效帧数（先在 _load_data 真正 resample，这里只算长度）
            g1_len_30 = max(2, int(round(g1_len_src * float(self.amass_fps) / g1_fps_src)))
            length = min(amass_len, g1_len_30)   # 两边都在 30fps，取较短
            if length < 25:
                skipped += 1
                continue

            num_samples = max(length // self.motion_frames, 1)
            meta = {
                "amass_key":  pair["amass_key"],
                "g1_file":    pair["g1"],
                "usable_len": length,        # 30fps 帧数
                "g1_fps_src": g1_fps_src,    # 留给 _load_data 做 resample 用
                "name":       pair["name"],
            }
            self.idx2meta.extend([meta] * num_samples)

        if cache_dirty:
            try:
                with open(cache_file, "w") as f:
                    json.dump(g1_meta_cache, f)
            except Exception:
                pass

        Log.info(
            f"[{self.dataset_name}] 切分完毕: {len(self.idx2meta)} 个片段, "
            f"跳过 {skipped} 个过短动作。"
        )

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]      # 30fps 帧数（time base，跟 AMASS 同步）
        g1_fps_src = meta["g1_fps_src"]      # G1 npz 源 fps（典型 50）
        tgt_len    = self.motion_frames

        # 速度增强：raw_subset_len 现在跟原版 GVHMR amass.py 完全一致（30fps 空间）。
        # 物理时长 [tgt_len / l_factor / 30, tgt_len * l_factor / 30) 秒。
        # 用 hi = max(lo+1, ...)+1 让上界 inclusive 且 l_factor=1.0 时不崩.
        lo = max(2, int(float(tgt_len) / self.l_factor))
        hi = max(lo + 1, int(float(tgt_len) * self.l_factor) + 1)
        raw_subset_len = np.random.randint(lo, hi)
        if raw_subset_len <= usable_len:
            start = np.random.randint(0, usable_len - raw_subset_len + 1)
            end   = start + raw_subset_len
        else:
            start, end = 0, usable_len

        # ========== AMASS from .pth (already at 30fps) ==========
        entry = self.motion_files[meta["amass_key"]]
        pose_full = entry["pose"][start:end]   # (F, 66) torch.float32
        trans = entry["trans"][start:end]      # (F, 3)  torch.float32
        beta = entry["beta"]                   # (10,)   torch.float32

        actual_len = pose_full.shape[0]
        smpl_data = {
            "transl": trans.float(),
            "global_orient": pose_full[:, :3].float(),
            "body_pose": pose_full[:, 3:].float(),
            "betas": beta[: self.betas_dim].unsqueeze(0).expand(actual_len, -1).float(),
        }

        # ========== G1 from npz: 先整体 resample 到 30fps，再 slice ==========
        with np.load(meta["g1_file"]) as g1_raw:
            required_g1_keys = [
                "joint_pos", "joint_vel",
                "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w",
            ]
            missing = [k for k in required_g1_keys if k not in g1_raw.files]
            if missing:
                raise KeyError(
                    f"[G1 npz 格式不对] 文件: {meta['g1_file']}\n缺少 keys: {missing}"
                )
            g1_full_src = {
                "fps": torch.tensor(g1_fps_src, dtype=torch.float32),
                "joint_pos":      torch.tensor(g1_raw["joint_pos"], dtype=torch.float32),
                "joint_vel":      torch.tensor(g1_raw["joint_vel"], dtype=torch.float32),
                "body_pos_w":     torch.tensor(g1_raw["body_pos_w"], dtype=torch.float32),
                "body_quat_w":    torch.tensor(g1_raw["body_quat_w"], dtype=torch.float32),
                "body_lin_vel_w": torch.tensor(g1_raw["body_lin_vel_w"], dtype=torch.float32),
                "body_ang_vel_w": torch.tensor(g1_raw["body_ang_vel_w"], dtype=torch.float32),
            }

        # Step 1: G1 全文件 50fps → 30fps（speed_ratio=1.0 保 lin_vel/ang_vel m/s 量纲）
        # 关键: 强制 G1 30fps 长度等于 amass_len, 避免 int(round(...)) 与 _get_idx2meta
        # 的 round 不一致, 造成 G1 比 AMASS 少 1 帧导致后续切片错位.
        amass_len = entry["pose"].shape[0]
        g1_len_30 = amass_len
        g1_at_30fps = interpolate_g1_params(g1_full_src, g1_len_30, speed_ratio=1.0)

        # Step 2: 在 30fps 空间用同一 [start, end] 切 slice，跟 AMASS 严格对齐
        # 由于 g1_at_30fps 长度 = amass_len, [start,end] 一定合法.
        assert g1_at_30fps["joint_pos"].shape[0] == amass_len, \
            f"G1 resample length {g1_at_30fps['joint_pos'].shape[0]} != AMASS {amass_len}"
        end_g1 = end
        g1_slice = {
            "fps":            g1_at_30fps["fps"],
            "joint_pos":      g1_at_30fps["joint_pos"][start:end_g1],
            "joint_vel":      g1_at_30fps["joint_vel"][start:end_g1],
            "body_pos_w":     g1_at_30fps["body_pos_w"][start:end_g1],
            "body_quat_w":    g1_at_30fps["body_quat_w"][start:end_g1],
            "body_lin_vel_w": g1_at_30fps["body_lin_vel_w"][start:end_g1],
            "body_ang_vel_w": g1_at_30fps["body_ang_vel_w"][start:end_g1],
        }

        # Step 3: 两边都从 30fps slice 插值到 motion_frames（这一步跟原版 GVHMR 一致）
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
        # Speed augmentation: 把 src_len_30 帧 (30fps) 拉伸/压缩到 tgt_len=120 帧, 仍按 30fps
        # 播放. 即模型看到 4 秒动作但物理上原 clip 可能 ((src_len_30-1)/29) 秒.
        # 播放速度 = src_phys_dur / tgt_phys_dur = (src_len_30-1)/(tgt_len-1) = speed_ratio_aug
        # vel 量纲 m/s, 重定时后 m/s 值 *= speed_ratio_aug (压缩时间 → 速度变大).
        src_len_30 = g1_slice["joint_pos"].shape[0]
        speed_ratio_aug = (src_len_30 - 1) / max(tgt_len - 1, 1)
        # body_lin_vel_w 从插值后的 body_pos_w 用 fps_out=30 (输出播放 fps) 差分得 m/s.
        # 这等价于先得到 m/frame@30fps 再乘 30, 跟"npz_vel * speed_ratio_aug" 物理上一致.
        fps_out = float(self.amass_fps)
        g1_interpolated = interpolate_g1_params(
            g1_slice, tgt_len, speed_ratio=speed_ratio_aug, fps_out=fps_out
        )

        smpl_interpolated["global_orient"], smpl_interpolated["transl"], _ = get_tgtcoord_rootparam(
            smpl_interpolated["global_orient"],
            smpl_interpolated["transl"],
            tsf="az->ay",
        )

        # 顺序: 先在 z-up native 系下 canonicalize G1 (R0_inv 把 G1 root 朝向校到
        # URDF identity = forward +x), 再 az_to_ay 把整段转到 y-up. 这样可以避免在
        # y-up 系下 canonicalize 时 R0_inv 包含 "URDF up (z) → world up (y)" 的 90° pitch
        # 分量被错误地一并撤销.
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(g1_interpolated, root_body_id=self.root_body_id)

        if self.apply_az_to_ay_g1:
            # az_to_ay 是 world axis (z-up → y-up) 的常数旋转, 对所有 body 一致应用.
            # 应用后 root quat[0,0] 不再是 identity, 而是 R_az_to_ay 对应的 quat — 这是
            # *物理上正确的* 表示 (G1 URDF local 是 z-up, 在 y-up world 下 root 必须带这个 90° tilt
            # 才站得起来). SMPL global_orient[0]=(0,0,0) 是 SMPL local (y-up) 在 y-up world 下的
            # identity, 与 G1 URDF root frame 物理上不可能相等. encode/decode/FK 全程一致用 R_w2c @ R_root
            # 处理这两套不同 local frame, 不需要让 quat[0]=identity.
            g1_interpolated["body_pos_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

        if self.canonicalize_first_frame:
            # pelvis 模式：减掉 J_pelvis(betas)，让 SMPL pelvis 关节真正落到原点。
            # transl 模式：pelvis_offset=None，旧口径（让 transl 落到原点）。
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset
            )

            # NOTE: SMPL/AMASS 保持在 y-up native (gravity = -y). G1 已在上面先
            # canonicalize 再 az_to_ay, 也落在同样的 y-up nominal 系.
        else:
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        # 地面高度校正: 把脚踩到 y=0, pelvis/root 竖直坐标 (y) > 0
        if self.floor_adjust and self.canonicalize_first_frame:
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            # SMPL: 用 frame 0 实际 FK 找最低关节 (T-pose 估算对非站立 pose 不准).
            # 用 self.smplx_lite (24 关节) 比 self.smplx (127 关节) 快.
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]   # (24, 3) world
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        return {
            "data_name": "g1_paired",
            "body_pose": smpl_interpolated["body_pose"],
            "betas": smpl_interpolated["betas"],
            "global_orient": smpl_interpolated["global_orient"],
            "transl": smpl_interpolated["transl"],
            "g1_fps": g1_interpolated["fps"] if g1_interpolated["fps"] is not None else torch.tensor(0.0),
            "g1_joint_pos": g1_interpolated["joint_pos"],
            "g1_joint_vel": g1_interpolated["joint_vel"],
            "g1_body_pos_w": g1_interpolated["body_pos_w"],
            "g1_body_quat_w": g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    def _process_data(self, data, idx):
        # 与 G1AmassDataset._process_data 同逻辑: 先 pop G1 字段，避免父类相机增强污染
        _ = data.pop("g1_fps")
        g1_target = {
            "g1_joint_pos":      data.pop("g1_joint_pos"),
            "g1_joint_vel":      data.pop("g1_joint_vel"),
            "g1_body_pos_w":     data.pop("g1_body_pos_w"),
            "g1_body_quat_w":    data.pop("g1_body_quat_w"),
            "g1_body_lin_vel_w": data.pop("g1_body_lin_vel_w"),
            "g1_body_ang_vel_w": data.pop("g1_body_ang_vel_w"),
        }
        return_data = super()._process_data(data, idx)
        # G1 专属相机: 拷贝 AMASS 的 T_w2c (已 repeat_to_max_len), 把相机位置整体缩放 0.9
        # (绝对值缩放, 非 delta 缩放). R / K 保持不变.
        #   - 每一帧 t_g1[f] = 0.9 * t_amass[f]
        # static 相机 (40%) 因 t 是常量, 也整体乘 0.9 → G1 相机离原点更近
        g1_T_w2c = return_data["T_w2c"].clone()
        assert g1_T_w2c.shape[0] == return_data["T_w2c"].shape[0]
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target

        # repeat_to_max_len 所有 g1_target 字段, 跟 T_w2c/R_c2gv/cam_angvel 一样规整.
        # 当前 motion_frames == length 时是 no-op, 但若日后引入可变长度采样不会爆.
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)

        # NOTE: 不要把 dataset 端的 bbx_xys 替换成 G1 投影!
        # 设计是: AMASS 相机 (T_w2c) 配 SMPL synth bbox 给 cliff_cam / obs 用,
        #         G1 相机 (g1_T_w2c) 配 robot_bbx_xys 给 j2d / transl_c loss 用.
        # 让 gvhmr_pl 自己跑 SMPL synth bbox 路径 (mask[bbx_xys]=False 保持 placeholder).

        return return_data

    # ----------------------------------------------------------
    # Root translation ratio filter
    # ----------------------------------------------------------
    def _root_axis_ratios(self, data):
        """
        Per-axis LSQ slope k where G1_delta[a] ≈ k * AMASS_delta[a].

        delta 取的是 frame f 相对 frame 0 的 pelvis / root 位移, AMASS 用 transl
        (与 pelvis joint 只差一个常数 pelvis_offset, 不影响 delta).

        Filter:
          - per-seq:   max|amass_delta[axis]| <= ratio_thr  → 该轴静止, 返回 None (放行)
          - per-frame: 仅 |amass_delta[axis,t]| > ratio_thr 的帧参与 LSQ (分母稳定)
          - 不足 2 帧通过 per-frame 阈值时, 也视为静止 (返回 None).

        Returns:
            k_per_axis: list[float | None]  长度 3
        """
        transl = data["transl"]                                # (F, 3) AMASS pelvis-mode transl
        g1_pos = data["g1_body_pos_w"][:, self.root_body_id]   # (F, 3) G1 root link
        a_delta = (transl     - transl[0:1]).detach().cpu().numpy()
        g_delta = (g1_pos     - g1_pos[0:1]).detach().cpu().numpy()
        out = []
        for ax in range(3):
            a = a_delta[:, ax]
            g = g_delta[:, ax]
            # per-seq 静止判定
            if float(np.abs(a).max()) <= self.ratio_thr:
                out.append(None)
                continue
            # per-frame mask: 只用 |amass| > thr 的帧做 LSQ
            mask = np.abs(a) > self.ratio_thr
            if int(mask.sum()) < 2:
                out.append(None)
                continue
            am, gm = a[mask], g[mask]
            denom = float((am * am).sum())
            if denom < 1e-12:
                out.append(None)
                continue
            out.append(float((gm * am).sum() / denom))
        return out

    def _ratio_pass(self, data):
        ks = self._root_axis_ratios(data)
        for ax, k in enumerate(ks):
            if k is None:
                continue
            if not (self.ratio_lo[ax] <= k <= self.ratio_hi[ax]):
                return False, ks
        return True, ks

    def __getitem__(self, idx):
        # Ratio filter: 同一 motion 的 ratio 是该动作物理性质, 跨 slice 不变.
        # 不再做 retry-resample (旧逻辑只是碰运气, 削弱 filter), 直接顺延到下一个 idx.
        # 整体最多顺延 len(idx2meta) 次, 兜底以避免死循环.
        start_idx = idx
        max_skip  = len(self.idx2meta)
        skip_n    = 0
        last_data = None
        last_idx  = idx
        while skip_n < max_skip:
            data = self._load_data(idx)
            last_data = data
            last_idx  = idx
            ok, _ks = self._ratio_pass(data)
            if ok:
                self._ratio_stats["accept"] += 1
                return self._process_data(data, idx)
            # 不合规: 跳到下一个 idx (同一 motion 的其它 slice 大概率也不合规, 直接换 motion)
            self._ratio_stats["skip"] += 1
            skip_n += 1
            idx = (idx + 1) % len(self.idx2meta)
            if idx == start_idx:
                break
        # 全数据集都不过 → 用 last_data / last_idx (一致的对应关系) 兜底返回
        Log.warning(
            f"[{self.dataset_name}] root ratio filter 找不到合规 sample, "
            f"start_idx={start_idx}, 兜底用 idx={last_idx} 的最后一次采样"
        )
        return self._process_data(last_data, last_idx)


# ==========================================================
# G1AmassDualPthDataset: AMASS + G1 都从 .pth 读
# ----------------------------------------------------------
# 数据源 (默认):
#   AMASS: filtered_amass.pth   {pose(F,66) AA / trans(F,3) / beta(10)}  y-up, 30fps
#   G1:    filtered_g1.pth      {root_pos(F,3) / root_rot(F,4) xyzw / dof_pos(F,29) MJC}
#                                **y-up**, 30fps (跟 AMASS 同坐标系, 不是 z-up!)
# 同 key 一一配对 (8013/8013), 长度严格相等, 都已 30fps → 不重采样.
# 内部加载时:
#   - G1 root_rot xyzw → wxyz (统一全仓库 wxyz)
#   - G1 dof_pos MJC → BYD   (统一旧 npz / 模型 BYD 契约)
#   - G1 root y-up → z-up    (ay_to_az; 这样下游的 canonicalize_g1 + az_to_ay 链路
#                              和 AMASS_G1_GMR_aligned.pth 一致, 保证机器人站立姿态)
#   - G1 没有 body_pos_w / quat / velocities, 在 _load_data 内用 Humanoid_Batch 跑 FK
#     拿到 30 link 的 pos/rot (URDF-DFS 序, z-up), 再 URDF-DFS → BYD body 序 permute
#   - velocities 用 finite-difference 算 (末帧复制).
# train/val/test 在公共 key 上做固定 seed 90/5/5 随机切分.
# 相机 G1: g1_T_w2c[:, :3, 3] = 0.9 * T_w2c[:, :3, 3] (绝对值缩放, 同旧契约新口径).
# ==========================================================

import sys as _sys


# 内联 dataset.py 里的小常量, 避免模块导入时序问题 (dataset.py 在仓库根)
_BYD_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]
_MJC_JOINT_NAMES = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
    'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
    'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
    'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]
# MJC→BYD perm: 用法 dof_mjc[..., _MJC_TO_BYD] = dof_byd.
#   _MJC_TO_BYD[byd_i] = mjc_index_of_byd_i.  对每个 BYD 槽, 取其在 MJC 中的下标.
#   等价于 dataset.py:65 的 mujoco_joint_to_byd_joint.
_MJC_TO_BYD = [_MJC_JOINT_NAMES.index(b[:-6]) for b in _BYD_JOINT_NAMES]
# BYD→MJC perm: 用法 dof_byd[..., _BYD_TO_MJC] = dof_mjc.
#   _BYD_TO_MJC[mjc_i] = byd_index_of_mjc_i.  对每个 MJC 槽, 取其在 BYD 中的下标.
#   等价于 dataset.py:64 的 byd_joint_to_mujoco_joint.
# 注意: _MJC_TO_BYD 不是 involution, 所以这俩 perm 必须分别维护, 不能复用一份.
_BYD_TO_MJC = [_BYD_JOINT_NAMES.index(n + '_joint') for n in _MJC_JOINT_NAMES]
# G5 rotation axis (MJC 序, 与 dataset.py:13 完全一致)
_G5_AXIS_MJC = torch.tensor([
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 0, 1], [1, 0, 0], [0, 1, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
], dtype=torch.float32)   # (29, 3)


def _quat_xyzw_to_wxyz(q):
    """(...,4) xyzw → (...,4) wxyz."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


# ----------------------------------------------------------------------
# ay_to_az: 把 y-up world 量旋到 z-up world (inverse of az_to_ay = R_x(90°)).
#   ay_to_az = R_x(-90°) = [[1, 0,  0],
#                            [0, 0, -1],
#                            [0, 1,  0]]
# 用法: 当 G1 数据源已经是 y-up (例如 filtered_g1.pth) 时, 在 load 时先用
# ay_to_az 转回 z-up, 然后跑既有的 (canonicalize_g1 z-up → az_to_ay → R_y(-90°))
# 链路, 保证 G1 root 朝向规范化的同时机器人保持站立 (URDF z-up 直立约定不被破坏).
# ----------------------------------------------------------------------
def ay_to_az_rotmat(device, dtype):
    return torch.tensor(
        [[1.0, 0.0,  0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0]],
        device=device, dtype=dtype,
    )


def apply_ay_to_az_on_vec(v):
    """v: (..., 3) y-up → z-up.  Applied via `v @ T.T`: (x, y, z)_yup → (x, z, -y)_zup."""
    T = ay_to_az_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_ay_to_az_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4) y-up world. Left-multiply by ay_to_az rotmat."""
    R = quaternion_to_matrix(q_wxyz)
    T = ay_to_az_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = T @ R
    return matrix_to_quaternion(R_new)


# ----------------------------------------------------------------------
# 朝向对齐: az_to_ay 之后 G1 forward = URDF +x (y-up world).
# SMPL canonical forward = SMPL local +z (y-up world).
# 两者差 90°. 在 G1 一侧再左乘 R_y(-90°) (绕 +y 轴顺时针 -90°)
# 把 G1 +x → +z, 与 SMPL forward 重合.
#   R_y(-90°) = [[ 0, 0, -1],
#                [ 0, 1,  0],
#                [ 1, 0,  0]]
# 该旋转保持 +y (vertical) 不变, 仅做 horizontal yaw.
# ----------------------------------------------------------------------
def _ry_neg90_rotmat(device, dtype):
    return torch.tensor(
        [[0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0],
         [1.0, 0.0,  0.0]],
        device=device, dtype=dtype,
    )


def _apply_ry_neg90_on_vec(v):
    """v: (..., 3) y-up world → R_y(-90°) @ v.  (x, y, z) → (-z, y, x)."""
    R = _ry_neg90_rotmat(v.device, v.dtype)
    return v @ R.T


def _apply_ry_neg90_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4). Left-multiply world quaternion by R_y(-90°)."""
    R = quaternion_to_matrix(q_wxyz)
    Ry = _ry_neg90_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = Ry @ R
    return matrix_to_quaternion(R_new)


def _byd_to_mjc(dof_byd):
    """(..., 29) BYD → (..., 29) MJC.

    必须用 _BYD_TO_MJC (BYD→MJC 反向 perm), 不能用 _MJC_TO_BYD!
    _MJC_TO_BYD 不是 involution: 比如 BYD[1]=right_hip_pitch, MJC[6]=right_hip_pitch,
    所以 _MJC_TO_BYD[1]=6; 但 BYD[6]=left_hip_yaw 不是 RHP, 用 _MJC_TO_BYD 反向会读到错位的值.
    """
    return dof_byd[..., _BYD_TO_MJC]


def _dof_to_pose_aa(dof_mjc, root_rot_wxyz, g5_axis_mjc):
    """与 dataset.py:dof_to_pose_aa_torch 等价, 内联避免依赖仓库根的 dataset.py.
    Args:
        dof_mjc:        (..., 29)
        root_rot_wxyz:  (..., 4) wxyz
        g5_axis_mjc:    (29, 3)
    Returns:
        pose_aa: (..., 30, 3) axis-angle, idx 0 是 root
    """
    leading = dof_mjc.shape[:-1]                                  # e.g. (B, F)
    dof_aa = g5_axis_mjc.to(dof_mjc.device, dof_mjc.dtype) * dof_mjc[..., None]  # (..., 29, 3)
    root_aa = matrix_to_axis_angle(quaternion_to_matrix(root_rot_wxyz))           # (..., 3)
    pose_aa = torch.cat([root_aa[..., None, :], dof_aa], dim=-2)  # (..., 30, 3)
    return pose_aa


# 全局 lazy: URDF-DFS → BYD body 序 (30,), pelvis 已对齐 idx 0
_URDF_DFS_TO_BYD_BODY_CACHE = None
def _get_urdf_dfs_to_byd_body():
    global _URDF_DFS_TO_BYD_BODY_CACHE
    if _URDF_DFS_TO_BYD_BODY_CACHE is None:
        from hmr4d.model.gvhmr.utils.endecoder import _G1_URDF_DFS_TO_BYD_BODY
        _URDF_DFS_TO_BYD_BODY_CACHE = list(_G1_URDF_DFS_TO_BYD_BODY)
    return _URDF_DFS_TO_BYD_BODY_CACHE


_HUMANOID_BATCH_CACHE = {}
def _get_humanoid_batch(device):
    """缓存 Humanoid_Batch 实例 (按 device). MJCF 路径相对仓库根."""
    key = str(device)
    if key not in _HUMANOID_BATCH_CACHE:
        from motion_lib.torch_h1_humanoid_batch import Humanoid_Batch
        _HUMANOID_BATCH_CACHE[key] = Humanoid_Batch(
            extend_hand=False, extend_head=False,
            mjcf_file='unitree_description/mjcf/g1.xml',
            device=device,
        )
    return _HUMANOID_BATCH_CACHE[key]


def fk_g1_pth_slice(dof_byd, root_pos_z, root_rot_wxyz_z, device=None):
    """对一条 G1 slice 跑 FK, 输出 BYD body 序的 body_pos_w / body_quat_w (z-up).

    Args:
        dof_byd:        (F, 29) BYD 序, torch.float32
        root_pos_z:     (F, 3)
        root_rot_wxyz_z:(F, 4) wxyz, z-up world
        device:         FK 运算 device. 默认用 dof_byd.device.
    Returns:
        body_pos_w_z:  (F, 30, 3) BYD body 序, z-up
        body_quat_w_z: (F, 30, 4) wxyz, BYD body 序, z-up
    """
    if device is None:
        device = dof_byd.device
    dof_byd        = dof_byd.to(device).float()
    root_pos_z     = root_pos_z.to(device).float()
    root_rot_wxyz_z = root_rot_wxyz_z.to(device).float()

    dof_mjc  = _byd_to_mjc(dof_byd)                              # (F, 29)
    pose_aa  = _dof_to_pose_aa(dof_mjc[None], root_rot_wxyz_z[None], _G5_AXIS_MJC)  # (1,F,30,3)
    from pytorch3d.transforms import axis_angle_to_quaternion as _aa2q
    pose_quat = _aa2q(pose_aa)                                    # (1,F,30,4) wxyz
    pose_mat  = quaternion_to_matrix(pose_quat)                   # (1,F,30,3,3)

    humanoid = _get_humanoid_batch(device)
    pos_dfs, rot_dfs = humanoid.forward_kinematics_batch(
        pose_mat[:, :, 1:], pose_mat[:, :, 0:1], root_pos_z[None]
    )
    # pos_dfs: (1, F, 30, 3), rot_dfs: (1, F, 30, 3, 3), URDF-DFS 序, z-up
    perm = torch.tensor(_get_urdf_dfs_to_byd_body(), dtype=torch.long, device=device)
    body_pos_w_z  = pos_dfs[0][:, perm]                           # (F, 30, 3)
    body_quat_w_z = matrix_to_quaternion(rot_dfs[0])[:, perm]     # (F, 30, 4) wxyz
    return body_pos_w_z, body_quat_w_z


def _finite_diff_lin_vel(x, fps):
    """x: (T, ...) → (T, ...) finite-diff velocity, 末帧复制."""
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    diff = (x[1:] - x[:-1]) * float(fps)
    return torch.cat([diff, diff[-1:].clone()], dim=0)


def _finite_diff_ang_vel(q_wxyz, fps):
    """q_wxyz: (T, N, 4) → (T, N, 3) angular velocity in world frame, 末帧复制.

    angvel[t] = aa( R[t+1] @ R[t]^T ) * fps,  where R = quat_to_mat(q).
    """
    if q_wxyz.shape[0] < 2:
        return torch.zeros(q_wxyz.shape[0], q_wxyz.shape[1], 3,
                           device=q_wxyz.device, dtype=q_wxyz.dtype)
    R = quaternion_to_matrix(q_wxyz)                              # (T, N, 3, 3)
    dR = R[1:] @ R[:-1].transpose(-1, -2)                         # (T-1, N, 3, 3)
    aa = matrix_to_axis_angle(dR)                                  # (T-1, N, 3)
    ang = aa * float(fps)
    return torch.cat([ang, ang[-1:].clone()], dim=0)


# ==========================================================
class G1AmassDualPthDataset(BaseDataset):
    """与 G1AmassPthDataset 接口完全一致, 但 AMASS + G1 都从 .pth 读, 且不重采样.

    .pth 已经做过 AMASS↔G1 对齐: 每个 common key 长度严格相等, 都是 30fps.
    G1 没有 body_pos_w/quat/velocities, 在 _load_data 内 FK + 差分得到.
    """

    def __init__(
        self,
        amass_pth_path="filtered_amass.pth",
        g1_pth_path="filtered_g1.pth",
        amass_fps=30.0,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        # 输入 G1 是否本身就是 y-up (filtered_g1.pth=True; AMASS_G1_GMR_aligned.pth=False).
        # True 时 _load_data 会先用 ay_to_az 把 G1 转回 z-up, 再走既有 canonicalize+az_to_ay 链.
        g1_input_is_yup=True,
        # filtered_*.pth 默认已经做完上游过滤, skip_substrings 留空; 用户可在 YAML 里覆盖
        skip_substrings=(),
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=True,
        # 首帧归一化只去 yaw(航向), 保留 pitch/roll, 使重力方向严格保持 -y, 与 gravity-view
        # 编码自洽. False 时退回旧的"去整个 R0"行为(首帧倾斜姿态会把重力轴转歪).
        yaw_only_canon=True,
        # train/val/test 公共 key 随机切分
        split="train",
        split_seed=42,
        split_ratios=(0.9, 0.05, 0.05),
        # Root translation ratio filter (沿用旧版, 默认 z 不约束)
        ratio_lo=(0.7, 0.7, -float("inf")),
        ratio_hi=(1.05, 1.05,  float("inf")),
        ratio_thr=0.1,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
        # 推理用: 整段一次 forward (而非 motion_frames=120 切窗). val/test 生效, train 强制 False.
        # 训练完的 ckpt 不需要重训 — relative_transformer 用 RoPE + sliding-window attn mask
        # (max_len=motion_frames) 天然支持 L>max_len, 见 relative_transformer.py:161-170.
        # 推荐配合 cam_augmentation="static_v11" 用, 避免长序列 push-away 把相机推得太远 OOD.
        full_sequence=False,
        # 推理用: 不加载 G1 .pth, 只跑 AMASS → G1 推理. val/test 生效, train 强制 False.
        # 推理路径 (pipeline.forward(train=False)) 只读 g1_target.g1_body_pos_w[:, [0], 0]
        # 当 root anchor, 缺失时 pipeline 自动 fallback 到 (0, default_pelvis_height, 0) y-up.
        # PredictionWriter callback 应同时设 pl_module.skip_val_loss=True, 否则 val_loss
        # 一支 (train=True) 会因 encode_g1 找不到 g1_joint_pos 等字段而崩.
        amass_only=False,
        # 推理用: 从指定 split 里随机抽 K 条做小批量评测 (不动训练侧 idx2meta 顺序).
        # subset_split: "train"|"val"|"test"|None — 从哪个 split 抽 (None 维持当前 self.split 行为).
        # subset_n:     int|None — 抽几条 (None 不抽).
        # subset_seed:  int — 抽样种子, 跟 split_seed 独立, 方便手工换 seed 看不同 20 条样本.
        # 用法: 想跑 train 集随机 20 条评测时, 配 split="test" (走 val/test deterministic 路径)
        #       + subset_split="train" + subset_n=20.
        subset_split=None,
        subset_n=None,
        subset_seed=42,
    ):
        self.amass_pth_path = Path(amass_pth_path)
        self.g1_pth_path    = Path(g1_pth_path)
        self.amass_fps      = float(amass_fps)
        self.motion_frames  = motion_frames
        self.l_factor       = l_factor
        self.dataset_name   = "G1_AMASS_DUAL_PTH"
        self.betas_dim      = betas_dim
        self.root_body_id   = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.g1_input_is_yup   = bool(g1_input_is_yup)
        self.skip_substrings = tuple(str(s) for s in (skip_substrings or ()))
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust   = floor_adjust
        self.yaw_only_canon = bool(yaw_only_canon)
        assert split in ("train", "val", "test"), split
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios 必须和=1.0"
        self.split          = split
        self.split_seed     = int(split_seed)
        self.split_ratios   = tuple(float(r) for r in split_ratios)

        ratio_lo = tuple(float(v) for v in ratio_lo)
        ratio_hi = tuple(float(v) for v in ratio_hi)
        assert len(ratio_lo) == 3 and len(ratio_hi) == 3
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.ratio_thr = float(ratio_thr)
        self._ratio_stats = {"accept": 0, "skip": 0}

        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        # 跟旧 G1AmassPthDataset 一致, 固定 betas → 关 betas 加噪
        self._skip_betas_aug = True

        # full_sequence 只在非 train 路径生效, 防止意外把训练改成全长 (会爆显存).
        self.full_sequence = bool(full_sequence) and split != "train"
        if bool(full_sequence) and split == "train":
            Log.warning(f"[{self.dataset_name}] full_sequence=True 在 train split 被忽略 "
                        f"(只 val/test 生效, 防止全长喂训练爆显存).")
        # amass_only 同理只在非 train 路径生效 (train 必须有 G1 GT 算 loss).
        self.amass_only = bool(amass_only) and split != "train"
        if bool(amass_only) and split == "train":
            Log.warning(f"[{self.dataset_name}] amass_only=True 在 train split 被忽略 "
                        f"(训练必须有 G1 GT 算 loss).")
        # subset 抽样: 不在 train 上做 (训练有自己的 ratio_filter / shuffle 逻辑).
        if subset_split is not None or subset_n is not None:
            assert split != "train", "subset_split / subset_n 仅 val/test 生效, 防止干扰训练."
            if subset_split is not None:
                assert subset_split in ("train", "val", "test"), f"bad subset_split={subset_split!r}"
        self.subset_split = subset_split
        self.subset_n     = int(subset_n) if subset_n is not None else None
        self.subset_seed  = int(subset_seed)

        super().__init__(cam_augmentation, limit_size)

    # ------------------------------------------------------------------
    # 加载 + 公共 key 90/5/5 随机切分
    # ------------------------------------------------------------------
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] 加载 AMASS pth: {self.amass_pth_path}")
        self.amass_files = torch.load(self.amass_pth_path, weights_only=False, map_location="cpu")
        if self.amass_only:
            Log.info(f"[{self.dataset_name}] amass_only=True, 跳过加载 G1 .pth")
            self.g1_files = None
            common = sorted(self.amass_files.keys())
        else:
            Log.info(f"[{self.dataset_name}] 加载 G1   pth: {self.g1_pth_path}")
            self.g1_files = torch.load(self.g1_pth_path, weights_only=False, map_location="cpu")
            common = sorted(set(self.amass_files.keys()) & set(self.g1_files.keys()))
        before = len(common)
        if self.skip_substrings:
            common = [k for k in common
                      if not any(s.lower() in k.lower() for s in self.skip_substrings)]
        Log.info(f"[{self.dataset_name}] 候选 key {before} → 过滤 skip_substrings 后 {len(common)} "
                 f"(skip_substrings={self.skip_substrings})")

        # 固定 seed 90/5/5 split
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(common))
        n_total = len(common)
        n_train = int(self.split_ratios[0] * n_total)
        n_val   = int(self.split_ratios[1] * n_total)
        if self.split_ratios[2] <= 0:
            n_val = n_total - n_train
        if self.split == "train":
            idx_sel = perm[:n_train]
        elif self.split == "val":
            idx_sel = perm[n_train:n_train + n_val]
        else:
            idx_sel = perm[n_train + n_val:]
        self.split_keys = [common[i] for i in idx_sel]
        Log.info(f"[{self.dataset_name}] split={self.split} 取 {len(self.split_keys)}/{n_total} 条 key "
                 f"(ratios={self.split_ratios}, seed={self.split_seed})")

        # subset 抽样 (推理评测用): 想用 train 集随机 K 条做 eval, 配 subset_split="train" + subset_n=K.
        # 注意 subset_split 跟 self.split 是两件事 — split 控制 _load_data 走 train/val/test 路径,
        # subset_split 只是改用哪个 split 的 key 池.
        if self.subset_split is not None:
            src_idx = {
                "train": perm[:n_train],
                "val":   perm[n_train:n_train + n_val],
                "test":  perm[n_train + n_val:],
            }[self.subset_split]
            pool = [common[i] for i in src_idx]
            Log.info(f"[{self.dataset_name}] subset_split={self.subset_split}, pool size={len(pool)}")
            if self.subset_n is not None and self.subset_n < len(pool):
                rng2 = np.random.default_rng(self.subset_seed)
                pick = rng2.choice(len(pool), size=self.subset_n, replace=False)
                pick.sort()   # 排序后跨 run 可复现, 也方便 idx2meta 顺序稳定
                self.split_keys = [pool[i] for i in pick]
            else:
                self.split_keys = pool
            Log.info(f"[{self.dataset_name}] subset_n={self.subset_n}, 最终取 {len(self.split_keys)} 条")

    def _get_idx2meta(self):
        self.idx2meta = []
        skipped = 0
        for k in self.split_keys:
            length = int(self.amass_files[k]["pose"].shape[0])
            if not self.amass_only:
                # G1 长度也是同一值 (.pth 已配对), 但 sanity-check 一下
                g1_len = int(self.g1_files[k]["root_pos"].shape[0])
                if g1_len != length:
                    Log.warning(f"[{self.dataset_name}] key {k!r}: AMASS len {length} ≠ G1 len {g1_len}, 跳过")
                    skipped += 1
                    continue
            if length < 25:
                skipped += 1
                continue
            if self.full_sequence:
                # 整段一次进网络 (RoPE + sliding-window attn 接管), 不再切窗.
                self.idx2meta.append({
                    "key": k, "usable_len": length, "name": Path(k).stem,
                    "seg_id": 0, "num_seg": 1,
                })
            else:
                num_samples = max(length // self.motion_frames, 1)
                # seg_id 给 val/test 用作 deterministic 切片下标 (start = seg_id * motion_frames).
                # train 路径不读 seg_id, 不影响.
                for seg_id in range(num_samples):
                    self.idx2meta.append({
                        "key": k, "usable_len": length, "name": Path(k).stem,
                        "seg_id": seg_id, "num_seg": num_samples,
                    })
        Log.info(f"[{self.dataset_name}] 切分完毕: {len(self.idx2meta)} 个片段, "
                 f"跳过 {skipped} 条 (full_sequence={self.full_sequence})")

    # ------------------------------------------------------------------
    # _load_data: 切片 → 插值 → FK → 差分 → canonicalize → floor_adjust
    # ------------------------------------------------------------------
    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]
        tgt_len    = self.motion_frames

        if self.split == "train":
            # 速度增强 (照搬旧逻辑): 随机速度 + 随机起点
            lo = max(2, int(float(tgt_len) / self.l_factor))
            hi = max(lo + 1, int(float(tgt_len) * self.l_factor) + 1)
            raw_subset_len = np.random.randint(lo, hi)
            if raw_subset_len <= usable_len:
                start = np.random.randint(0, usable_len - raw_subset_len + 1)
                end   = start + raw_subset_len
            else:
                start, end = 0, usable_len
        else:
            # val/test deterministic: 不做速度抖动, 按 seg_id 取不重叠的 tgt_len 片段.
            # 同 idx 跨 epoch / 跨 run 完全一致, val/loss 可复现.
            if self.full_sequence:
                # 整段一次进网络 (推理用), 不切窗. tgt_len 重写为整段长度,
                # 下游 interpolate_smpl_params(src_len == tgt_len) 是 no-op.
                start = 0
                end = usable_len
                tgt_len = usable_len
            else:
                seg_id = int(meta.get("seg_id", 0))
                raw_subset_len = min(tgt_len, usable_len)
                # 可选: meta["start"] 显式指定起始帧 (用于可视化"以蹲下为起始"的片段);
                # 不给则按 seg_id 走 deterministic 切片.
                if meta.get("start", None) is not None:
                    start = int(meta["start"])
                else:
                    start = seg_id * tgt_len
                if start + raw_subset_len > usable_len:
                    start = max(0, usable_len - raw_subset_len)
                end = start + raw_subset_len

        # -------------------- AMASS slice (已 y-up) --------------------
        ea = self.amass_files[meta["key"]]
        pose_full = ea["pose"][start:end].float()                  # (F, 66)
        trans     = ea["trans"][start:end].float()                 # (F, 3)
        beta      = ea["beta"].float()                             # (10,)
        actual_len = pose_full.shape[0]
        smpl_data = {
            "transl":        trans,
            "global_orient": pose_full[:, :3],
            "body_pose":     pose_full[:, 3:],
            "betas":         beta[: self.betas_dim].unsqueeze(0).expand(actual_len, -1).float(),
        }

        # -------------------- amass_only: 直接走 SMPL-only 短路 --------------------
        if self.amass_only:
            smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
            if self.canonicalize_first_frame:
                if self.align_target == "pelvis":
                    pelvis_offset = compute_smpl_pelvis_offset(
                        smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                    )
                else:
                    pelvis_offset = None
                smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                    smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                    yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
                )
            if self.floor_adjust and self.canonicalize_first_frame:
                with torch.no_grad():
                    j24 = self.smplx_lite(
                        smpl_interpolated["body_pose"][:1],
                        smpl_interpolated["betas"][:1],
                        smpl_interpolated["global_orient"][:1],
                        smpl_interpolated["transl"][:1],
                    )[0]   # (24, 3) world y-up
                smpl_floor_y = float(-j24[:, 1].min())
                smpl_interpolated["transl"][:, 1] += smpl_floor_y
            # 不返回任何 g1_* 字段. pipeline.forward(train=False) 走 (0, 0.793, 0) fallback.
            return {
                "data_name":     "g1_paired_dualpth",
                "body_pose":     smpl_interpolated["body_pose"],
                "betas":         smpl_interpolated["betas"],
                "global_orient": smpl_interpolated["global_orient"],
                "transl":        smpl_interpolated["transl"],
            }

        # -------------------- G1 slice (默认 y-up, 转回 z-up) --------------------
        eg = self.g1_files[meta["key"]]
        # dof_pos: MJC → BYD; root_rot: xyzw → wxyz
        dof_pos_mjc = torch.as_tensor(eg["dof_pos"][start:end], dtype=torch.float32)  # (F, 29)
        dof_pos_byd = dof_pos_mjc[..., _MJC_TO_BYD]                                   # (F, 29) BYD
        root_pos_raw     = torch.as_tensor(eg["root_pos"][start:end], dtype=torch.float32) # (F, 3)
        root_rot_xyzw_raw = torch.as_tensor(eg["root_rot"][start:end], dtype=torch.float32)# (F, 4) xyzw
        root_rot_wxyz_raw = _quat_xyzw_to_wxyz(root_rot_xyzw_raw)                          # (F, 4) wxyz

        if self.g1_input_is_yup:
            # filtered_g1.pth 是 y-up world. 先转回 z-up, 这样下游 FK 拿到 z-up 输入,
            # canonicalize_g1_first_frame 在 z-up 下零化 root 后机器人仍站立 (URDF up = +z).
            root_pos_z      = apply_ay_to_az_on_vec(root_pos_raw)
            root_rot_wxyz_z = apply_ay_to_az_on_quat_wxyz(root_rot_wxyz_raw)
        else:
            # AMASS_G1_GMR_aligned.pth: 已经是 z-up
            root_pos_z      = root_pos_raw
            root_rot_wxyz_z = root_rot_wxyz_raw

        # -------------------- SMPL 插值到 motion_frames --------------------
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)

        # -------------------- G1 插值到 motion_frames (位姿三件套, vel 后面差分) --------------------
        dof_pos_byd_i = interpolate_seq(dof_pos_byd, tgt_len)            # (tgt_len, 29)
        root_pos_z_i  = interpolate_seq(root_pos_z,  tgt_len)            # (tgt_len, 3)
        root_rot_wxyz_z_i = slerp_sequence_quat(root_rot_wxyz_z, tgt_len)# (tgt_len, 4) wxyz, z-up

        # -------------------- FK → body_pos_w / body_quat_w (z-up, BYD body 序) --------------------
        with torch.no_grad():
            body_pos_w_z, body_quat_w_z = fk_g1_pth_slice(
                dof_pos_byd_i, root_pos_z_i, root_rot_wxyz_z_i, device=dof_pos_byd_i.device
            )

        # -------------------- 速度: finite-difference @ 30fps --------------------
        fps_out = float(self.amass_fps)
        joint_vel_byd_i   = _finite_diff_lin_vel(dof_pos_byd_i, fps_out)             # (tgt_len, 29)
        body_lin_vel_w_z  = _finite_diff_lin_vel(body_pos_w_z, fps_out)              # (tgt_len, 30, 3)
        body_ang_vel_w_z  = _finite_diff_ang_vel(body_quat_w_z, fps_out)             # (tgt_len, 30, 3)

        g1_interpolated = {
            "fps":            torch.tensor(fps_out, dtype=torch.float32),
            "joint_pos":      dof_pos_byd_i,
            "joint_vel":      joint_vel_byd_i,
            "body_pos_w":     body_pos_w_z,
            "body_quat_w":    body_quat_w_z,
            "body_lin_vel_w": body_lin_vel_w_z,
            "body_ang_vel_w": body_ang_vel_w_z,
        }

        # NOTE: 新 AMASS 已 y-up, 不做 get_tgtcoord_rootparam(tsf="az->ay")

        # -------------------- canonicalize G1 first frame (在 z-up local) --------------------
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(
                g1_interpolated, root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon, up_axis=2, fwd_axis=0,   # G1 z-up, URDF forward=+x
            )

        # -------------------- G1 世界量 z-up → y-up --------------------
        if self.apply_az_to_ay_g1:
            g1_interpolated["body_pos_w"]     = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"]    = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

            # 朝向对齐: G1 URDF forward=+x (y-up world), SMPL canonical forward=+z (y-up world).
            # 在 G1 一侧再左乘 R_y(-90°) 把 G1 +x → +z, 与 SMPL forward 重合.
            # 该旋转保持 +y (vertical) 不变, 不影响脚踩地高度.
            g1_interpolated["body_pos_w"]     = _apply_ry_neg90_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"]    = _apply_ry_neg90_on_quat_wxyz(g1_interpolated["body_quat_w"])

        # -------------------- canonicalize SMPL first frame (pelvis 落原点) --------------------
        if self.canonicalize_first_frame:
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
            )
        else:
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        # -------------------- floor_adjust: 脚踩到 y=0 --------------------
        if self.floor_adjust and self.canonicalize_first_frame:
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]   # (24, 3) world y-up
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        return {
            "data_name":         "g1_paired_dualpth",
            "body_pose":         smpl_interpolated["body_pose"],
            "betas":             smpl_interpolated["betas"],
            "global_orient":     smpl_interpolated["global_orient"],
            "transl":            smpl_interpolated["transl"],
            "g1_fps":            g1_interpolated["fps"],
            "g1_joint_pos":      g1_interpolated["joint_pos"],
            "g1_joint_vel":      g1_interpolated["joint_vel"],
            "g1_body_pos_w":     g1_interpolated["body_pos_w"],
            "g1_body_quat_w":    g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    # ------------------------------------------------------------------
    # _process_data: 抄旧 G1AmassPthDataset._process_data, camera 改成绝对 0.9
    # ------------------------------------------------------------------
    def _process_data(self, data, idx):
        # amass_only: data 只有 SMPL 字段, 没有 g1_*. 直接进 BaseDataset._process_data,
        # 出来后不构造 g1_target — pipeline.forward(train=False) 走 fallback.
        if self.amass_only:
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            self._static_cam_seed = int(seed)
            np_state    = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                return_data = super()._process_data(data, idx)
            finally:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)
            return return_data

        _ = data.pop("g1_fps")
        g1_target = {
            "g1_joint_pos":      data.pop("g1_joint_pos"),
            "g1_joint_vel":      data.pop("g1_joint_vel"),
            "g1_body_pos_w":     data.pop("g1_body_pos_w"),
            "g1_body_quat_w":    data.pop("g1_body_quat_w"),
            "g1_body_lin_vel_w": data.pop("g1_body_lin_vel_w"),
            "g1_body_ang_vel_w": data.pop("g1_body_ang_vel_w"),
        }
        if self.split == "train":
            return_data = super()._process_data(data, idx)
        else:
            # val/test deterministic camera: 全程 np.random.*,
            # 按 (split_seed, idx) 种子摇一次, 同 idx 跨 epoch / 跨 run 拿到同一条相机轨迹.
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            # static_v11 路径不读全局 np.random (它在 StaticCameraV11.__call__ 内部 reseed),
            # 我们把这个 seed 通过 _static_cam_seed 传进去, 保证每 idx 相机不同但可复现.
            self._static_cam_seed = int(seed)
            np_state    = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                return_data = super()._process_data(data, idx)
            finally:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)

        # G1 相机 = AMASS 相机, 平移整体乘 0.9 (绝对值缩放, 同旧契约新口径)
        g1_T_w2c = return_data["T_w2c"].clone()
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target

        # repeat_to_max_len 所有 g1_target 字段
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)
        return return_data

    # ------------------------------------------------------------------
    # ratio filter (沿用旧版)
    # ------------------------------------------------------------------
    def _root_axis_ratios(self, data):
        transl = data["transl"]
        g1_pos = data["g1_body_pos_w"][:, self.root_body_id]
        a_delta = (transl - transl[0:1]).detach().cpu().numpy()
        g_delta = (g1_pos - g1_pos[0:1]).detach().cpu().numpy()
        out = []
        for ax in range(3):
            a = a_delta[:, ax]; g = g_delta[:, ax]
            if float(np.abs(a).max()) <= self.ratio_thr:
                out.append(None); continue
            mask = np.abs(a) > self.ratio_thr
            if int(mask.sum()) < 2:
                out.append(None); continue
            am, gm = a[mask], g[mask]
            denom = float((am * am).sum())
            if denom < 1e-12:
                out.append(None); continue
            out.append(float((gm * am).sum() / denom))
        return out

    def _ratio_pass(self, data):
        ks = self._root_axis_ratios(data)
        for ax, k in enumerate(ks):
            if k is None: continue
            if not (self.ratio_lo[ax] <= k <= self.ratio_hi[ax]):
                return False, ks
        return True, ks

    def __getitem__(self, idx):
        # val/test: ratio filter 不能"跳到下一个 idx", 否则
        # (a) 某些 idx 被评估多次, 某些永不被评估; (b) 评估集实际大小 != 命名值.
        # → val/test 直接走 deterministic 路径, 不做 filter.
        if self.split != "train":
            return self._process_data(self._load_data(idx), idx)

        start_idx = idx
        max_skip  = len(self.idx2meta)
        skip_n    = 0
        last_data = None
        last_idx  = idx
        while skip_n < max_skip:
            data = self._load_data(idx)
            last_data = data
            last_idx  = idx
            ok, _ks = self._ratio_pass(data)
            if ok:
                self._ratio_stats["accept"] += 1
                return self._process_data(data, idx)
            self._ratio_stats["skip"] += 1
            skip_n += 1
            idx = (idx + 1) % len(self.idx2meta)
            if idx == start_idx:
                break
        Log.warning(f"[{self.dataset_name}] root ratio filter 全失败, "
                    f"start_idx={start_idx}, 用 idx={last_idx} 兜底")
        return self._process_data(last_data, last_idx)


BONES_SEED_SOMA_OBS14 = [6, 7, 9, 15, 10, 16, 11, 17, 20, 24, 21, 25, 22, 26]


def _load_joblib(path):
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "BonesSeedG1SomaDataset requires joblib. Activate the project env "
            "before loading this dataset."
        ) from e
    return joblib.load(path)


def _as_float_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32).clone()


class BonesSeedG1SomaDataset(Dataset):
    """Bones Seed paired SOMA/G1 dataset using the shared pkl files."""

    def __init__(
        self,
        data_dir="/mnt/ddn/shared/datasets/humanoid/bones_seed_g1_soma_pair",
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        root_body_id=0,
        canonicalize_first_frame=True,
        yaw_only_canon=True,
        floor_adjust=True,
        split="train",
        split_seed=42,
        split_ratios=(0.9, 0.05, 0.05),
        dof_pos_order="mjc",
        soma_obs_indices=tuple(BONES_SEED_SOMA_OBS14),
        full_sequence=False,
        subset_split=None,
        subset_n=None,
        subset_seed=42,
    ):
        assert split in ("train", "val", "test"), split
        assert dof_pos_order in ("mjc", "byd"), dof_pos_order
        assert abs(sum(split_ratios) - 1.0) < 1e-6
        self.data_dir = Path(data_dir)
        self.motion_frames = int(motion_frames)
        self.l_factor = float(l_factor)
        self.cam_augmentation = cam_augmentation
        self.limit_size = limit_size
        self.root_body_id = int(root_body_id)
        self.canonicalize_first_frame = bool(canonicalize_first_frame)
        self.yaw_only_canon = bool(yaw_only_canon)
        self.floor_adjust = bool(floor_adjust)
        self.split = split
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(x) for x in split_ratios)
        self.dof_pos_order = dof_pos_order
        self.soma_obs_indices = tuple(int(i) for i in soma_obs_indices)
        self.full_sequence = bool(full_sequence) and split != "train"
        self.subset_split = subset_split
        self.subset_n = int(subset_n) if subset_n is not None else None
        self.subset_seed = int(subset_seed)
        self.dataset_name = "BONES_SEED_G1_SOMA"
        self._cached_pkl_path = None
        self._cached_pkl_data = None

        self._load_dataset()
        self._get_idx2meta()

    def __len__(self):
        if self.limit_size is not None:
            return min(int(self.limit_size), len(self.idx2meta))
        return len(self.idx2meta)

    def _load_dataset(self):
        pkl_files = sorted(self.data_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found under {self.data_dir}")

        self.records = {}
        for p in pkl_files:
            data = _load_joblib(p)
            for seq_name, payload in data.items():
                key = f"{p.stem}/{seq_name}"
                if key in self.records:
                    raise ValueError(f"Duplicate Bones Seed key: {key}")
                length = int(np.asarray(payload["g1"]["dof_pos"]).shape[0])
                self.records[key] = {"file": str(p), "seq_name": seq_name, "length": length}
            del data

        keys = sorted(self.records)
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(keys))
        n_total = len(keys)
        n_train = int(self.split_ratios[0] * n_total)
        n_val = int(self.split_ratios[1] * n_total)
        splits = {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }
        split_for_keys = self.subset_split or self.split
        assert split_for_keys in ("train", "val", "test")
        selected = [keys[i] for i in splits[split_for_keys]]
        if self.subset_n is not None and self.subset_n < len(selected):
            rng2 = np.random.default_rng(self.subset_seed)
            pick = rng2.choice(len(selected), size=self.subset_n, replace=False)
            pick.sort()
            selected = [selected[i] for i in pick]
        self.split_keys = selected
        Log.info(
            f"[{self.dataset_name}] files={len(pkl_files)} seqs={len(keys)} "
            f"split={self.split} selected={len(self.split_keys)}"
        )

    def _seq_len(self, key):
        return int(self.records[key]["length"])

    def _load_record_payload(self, key):
        rec = self.records[key]
        pkl_path = rec["file"]
        if self._cached_pkl_path != pkl_path:
            self._cached_pkl_data = _load_joblib(pkl_path)
            self._cached_pkl_path = pkl_path
        return self._cached_pkl_data[rec["seq_name"]]

    def _get_idx2meta(self):
        self.idx2meta = []
        for key in self.split_keys:
            length = self._seq_len(key)
            if length < 25:
                continue
            if self.full_sequence:
                self.idx2meta.append({"key": key, "usable_len": length, "seg_id": 0, "num_seg": 1})
            else:
                n_seg = max(length // self.motion_frames, 1)
                for seg_id in range(n_seg):
                    self.idx2meta.append({"key": key, "usable_len": length, "seg_id": seg_id, "num_seg": n_seg})
        Log.info(f"[{self.dataset_name}] idx2meta={len(self.idx2meta)} full_sequence={self.full_sequence}")

    def _slice_bounds(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]
        tgt_len = self.motion_frames
        if self.split == "train":
            lo = max(2, int(tgt_len / self.l_factor))
            hi = max(lo + 1, int(tgt_len * self.l_factor) + 1)
            raw_len = np.random.randint(lo, hi)
            if raw_len <= usable_len:
                start = np.random.randint(0, usable_len - raw_len + 1)
                end = start + raw_len
            else:
                start, end = 0, usable_len
        else:
            if self.full_sequence:
                start, end, tgt_len = 0, usable_len, usable_len
            else:
                raw_len = min(tgt_len, usable_len)
                start = int(meta.get("seg_id", 0)) * tgt_len
                if start + raw_len > usable_len:
                    start = max(0, usable_len - raw_len)
                end = start + raw_len
        return start, end, tgt_len

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        rec = self._load_record_payload(meta["key"])
        g1_raw = rec["g1"]
        soma_raw = rec["soma"]
        start, end, tgt_len = self._slice_bounds(idx)

        fps = float(g1_raw.get("fps", soma_raw.get("fps", 30.0)))
        dof_pos = _as_float_tensor(g1_raw["dof_pos"][start:end])
        if self.dof_pos_order == "mjc":
            dof_pos = dof_pos[..., _MJC_TO_BYD]
        root_pos = _as_float_tensor(g1_raw["root_pos"][start:end])
        root_rot = _as_float_tensor(g1_raw["root_rot"][start:end])
        body_pos = _as_float_tensor(g1_raw["body_pos_w_full"][start:end])
        body_quat = _as_float_tensor(g1_raw["body_quat_w_full"][start:end])

        soma_go = _as_float_tensor(soma_raw["global_orient"][start:end])
        soma_joints = _as_float_tensor(soma_raw["joints"][start:end])

        dof_pos = interpolate_seq(dof_pos, tgt_len)
        root_pos = interpolate_seq(root_pos, tgt_len)
        root_rot = slerp_sequence_quat(root_rot, tgt_len)
        body_pos = interpolate_seq(body_pos, tgt_len)
        body_quat = slerp_sequence_quat(body_quat, tgt_len)
        soma_go_q = slerp_sequence_quat(matrix_to_quaternion(axis_angle_to_matrix(soma_go)), tgt_len)
        soma_go = matrix_to_axis_angle(quaternion_to_matrix(soma_go_q))
        soma_joints = interpolate_seq(soma_joints, tgt_len)

        g1 = {
            "fps": torch.tensor(fps, dtype=torch.float32),
            "joint_pos": dof_pos,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "body_pos_w_full": body_pos,
            "body_quat_w_full": body_quat,
        }
        if self.canonicalize_first_frame:
            g1 = canonicalize_g1_first_frame_bones_seed(
                g1,
                root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon,
                up_axis=2,
                fwd_axis=0,
            )
            soma_go, soma_joints = canonicalize_smpl_first_frame_bones_seed(
                soma_go,
                soma_joints,
                yaw_only=self.yaw_only_canon,
                up_axis=2,
                fwd_axis=1,
            )
        else:
            g1["body_pos_w"] = body_pos
            g1["body_quat_w"] = body_quat

        soma_go = apply_az_to_ay_on_axis_angle(soma_go)
        soma_joints = apply_az_to_ay_on_vec(soma_joints)

        g1["body_pos_w"] = apply_az_to_ay_on_vec(g1["body_pos_w"])
        g1["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1["body_quat_w"])
        g1["body_pos_w"] = _apply_ry_neg90_on_vec(g1["body_pos_w"])
        g1["body_quat_w"] = _apply_ry_neg90_on_quat_wxyz(g1["body_quat_w"])
        g1["root_pos"] = g1["body_pos_w"][:, self.root_body_id].clone()
        g1["root_rot"] = g1["body_quat_w"][:, self.root_body_id].clone()

        if self.floor_adjust:
            floor_y = float(min(g1["body_pos_w"][0, :, 1].min(), soma_joints[0, :, 1].min()))
            g1["body_pos_w"][:, :, 1] -= floor_y
            soma_joints[:, :, 1] -= floor_y
            g1["root_pos"] = g1["body_pos_w"][:, self.root_body_id].clone()

        g1["joint_vel"] = _finite_diff_lin_vel(g1["joint_pos"], fps)
        g1["body_lin_vel_w"] = _finite_diff_lin_vel(g1["body_pos_w"], fps)
        g1["body_ang_vel_w"] = _finite_diff_ang_vel(g1["body_quat_w"], fps)

        return {
            "data_name": "bones_seed_g1_soma",
            "key": meta["key"],
            "fps": fps,
            "soma_global_orient": soma_go,
            "soma_joints_w": soma_joints,
            "soma_obs_joints_w": soma_joints[:, self.soma_obs_indices],
            "g1_joint_pos": g1["joint_pos"],
            "g1_joint_vel": g1["joint_vel"],
            "g1_body_pos_w": g1["body_pos_w"],
            "g1_body_quat_w": g1["body_quat_w"],
            "g1_body_lin_vel_w": g1["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1["body_ang_vel_w"],
        }

    def _make_camera(self, soma_joints_w, idx):
        length = soma_joints_w.shape[0]
        _, _, K_fullimg = create_camera_sensor(1000, 1000, 43.3)
        if self.cam_augmentation == "v11":
            augmentor = CameraAugmentorV11()
            return augmentor(soma_joints_w, length), K_fullimg
        if self.cam_augmentation == "static_v11":
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            augmentor = StaticCameraV11(seed=seed)
            return augmentor(soma_joints_w, length), K_fullimg
        raise NotImplementedError(f"Unknown cam_augmentation={self.cam_augmentation!r}")

    def _process_data(self, data, idx):
        length = int(data["soma_joints_w"].shape[0])
        T_w2c, K = self._make_camera(data["soma_joints_w"], idx)
        K_fullimg = K.repeat(length, 1, 1)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], torch.tensor([0, -1, 0], dtype=torch.float32))
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])

        g1_target = {
            "g1_joint_pos": data["g1_joint_pos"],
            "g1_joint_vel": data["g1_joint_vel"],
            "g1_body_pos_w": data["g1_body_pos_w"],
            "g1_body_quat_w": data["g1_body_quat_w"],
            "g1_body_lin_vel_w": data["g1_body_lin_vel_w"],
            "g1_body_ang_vel_w": data["g1_body_ang_vel_w"],
        }
        g1_T_w2c = T_w2c.clone()
        g1_T_w2c[..., :3, 3] = 0.9 * T_w2c[..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c

        zeros_body = torch.zeros(length, 63, dtype=torch.float32)
        zeros_betas = torch.zeros(length, 10, dtype=torch.float32)
        smpl_params = {
            "body_pose": zeros_body,
            "betas": zeros_betas,
            "global_orient": data["soma_global_orient"],
            "transl": data["soma_joints_w"][:, 0],
        }

        out = {
            "meta": {"data_name": data["data_name"], "idx": idx, "key": data["key"], "T_w2c": T_w2c},
            "length": length,
            "smpl_params_c": smpl_params,
            "smpl_params_w": smpl_params,
            "T_w2c": T_w2c,
            "R_c2gv": R_c2gv,
            "gravity_vec": torch.tensor([0, -1, 0], dtype=torch.float32),
            "bbx_xys": torch.zeros((length, 3), dtype=torch.float32),
            "K_fullimg": K_fullimg,
            "f_imgseq": torch.zeros((length, 1024), dtype=torch.float32),
            "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
            "cam_angvel": cam_angvel,
            "soma_joints_w": data["soma_joints_w"],
            "soma_obs_joints_w": data["soma_obs_joints_w"],
            "g1_target": g1_target,
            "mask": {
                "valid": get_valid_mask(length, length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
            },
        }

        max_len = length
        out["smpl_params_c"] = repeat_to_max_len_dict(out["smpl_params_c"], max_len)
        out["smpl_params_w"] = repeat_to_max_len_dict(out["smpl_params_w"], max_len)
        for k in ("T_w2c", "R_c2gv", "K_fullimg", "cam_angvel", "soma_joints_w", "soma_obs_joints_w"):
            out[k] = repeat_to_max_len(out[k], max_len)
        for k, v in list(g1_target.items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                g1_target[k] = repeat_to_max_len(v, max_len)
        return out

    def __getitem__(self, idx):
        return self._process_data(self._load_data(idx), idx)


# ==========================================================
# 配置注册
# ==========================================================
# 注意: 不在此注册 train_datasets/test_datasets，
# 因为 YAML 文件 (configs/train_datasets/train.yaml, configs/test_datasets/val.yaml)
# 已经提供了正确的嵌套 dict 结构 {amass: {_target_: ..., data_dir: ...}}。
# 如果同时用 builds() 注册扁平 schema，Hydra 合并时会冲突。
