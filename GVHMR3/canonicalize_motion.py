# ============================================================
# canonicalize_motion.py
# 从 hmr4d/dataset/pure_motion/g1_amass.py 提取的运动规范化工具函数
#
# 包含内容:
#   - 坐标系转换: az_to_ay, ay_to_az, R_y(-90°)
#   - SMPLX pelvis/foot buffer 加载
#   - _first_frame_inv_rot  (核心辅助)
#   - canonicalize_smpl_first_frame
#   - canonicalize_g1_first_frame
#   - apply_smpl_to_g1_axes_on_smpl
#
# 完整规范化链路 (G1AmassDualPthDataset):
#   1. canonicalize_g1_first_frame(yaw_only=True, z-up: up=2, fwd=0)
#   2. apply_az_to_ay_on_vec / apply_az_to_ay_on_quat_wxyz  (G1 z-up → y-up)
#   3. _apply_ry_neg90_on_vec / _apply_ry_neg90_on_quat_wxyz (G1 fwd +x → +z)
#   4. canonicalize_smpl_first_frame(yaw_only=True, y-up: up=1, fwd=2, pelvis_offset)
#   5. floor_adjust (可选): SMPL 脚踩地 y=0
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np

from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)


# ==========================================================
# Z-up → Y-up 坐标系旋转
# Applied via `v @ T.T`: (x, y, z)_zup → (x, -z, y)_yup.
# ==========================================================

def az_to_ay_rotmat(device, dtype):
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


# ==========================================================
# Y-up → Z-up 坐标系旋转 (az_to_ay 的逆)
# ay_to_az = R_x(-90°): (x, y, z)_yup → (x, z, -y)_zup  via  v @ T.T
# 用途: G1 数据源已是 y-up (如 filtered_g1.pth) 时先转回 z-up,
#       再走 (canonicalize_g1 z-up → az_to_ay → R_y(-90°)) 链路。
# ==========================================================

def ay_to_az_rotmat(device, dtype):
    return torch.tensor(
        [[1.0, 0.0,  0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0]],
        device=device, dtype=dtype,
    )


def apply_ay_to_az_on_vec(v):
    """v: (..., 3) y-up → z-up."""
    T = ay_to_az_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_ay_to_az_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4) y-up world. Left-multiply by ay_to_az rotmat."""
    R = quaternion_to_matrix(q_wxyz)
    T = ay_to_az_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = T @ R
    return matrix_to_quaternion(R_new)


# ==========================================================
# 朝向对齐: az_to_ay 之后 G1 forward = URDF +x (y-up world).
# SMPL canonical forward = SMPL local +z (y-up world). 两者差 90°.
# 在 G1 一侧左乘 R_y(-90°) 把 G1 +x → +z，与 SMPL forward 重合。
#   R_y(-90°) = [[ 0, 0, -1],
#                [ 0, 1,  0],
#                [ 1, 0,  0]]
# 该旋转保持 +y (vertical) 不变，仅做水平 yaw。
# ==========================================================

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


# ==========================================================
# SMPL local → G1 local 坐标系变换矩阵
# AMASS canonical (SMPL body local: x=left, y=up, z=fwd) →
# G1     canonical (URDF root local: x=fwd, y=left, z=up)
#   x_new = z_old (forward)
#   y_new = x_old (left)
#   z_new = y_old (up)
# ==========================================================

R_SMPL_LOCAL_TO_G1_LOCAL = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


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
# SMPL-X pelvis 行加载（用于让 pelvis 关节真正落到原点）
#   pelvis_world(f) = transl(f) + J_pelvis(betas)
#   J_pelvis(betas) = J_template[0] + J_shapedirs[0] @ betas
#   与 global_orient 无关
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
    v_template  = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)                   # (V, 3)
    shapedirs   = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)  # (V, 3, B)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)                  # (J, V)

    pelvis_row = J_regressor[0]                                              # (V,)
    J_template_pelvis  = pelvis_row @ v_template                             # (3,)
    J_shapedirs_pelvis = torch.einsum("v, vcb -> cb", pelvis_row, shapedirs) # (3, B)

    _SMPLX_PELVIS_CACHE[key] = (J_template_pelvis, J_shapedirs_pelvis)
    return _SMPLX_PELVIS_CACHE[key]


def compute_smpl_pelvis_offset(betas, J_template_pelvis, J_shapedirs_pelvis):
    """betas: (..., B). 返回 (..., 3) pelvis 在 canonical 模型系下的位置。"""
    return J_template_pelvis + torch.einsum("...b, cb -> ...c", betas, J_shapedirs_pelvis)


# ==========================================================
# SMPL-X foot buffer 加载（用于 floor_adjust: 脚踩地 y=0）
# 脚踝 7=l_ankle 8=r_ankle, 趾尖 10=l_foot 11=r_foot
# ==========================================================

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
    rows = J_regressor[list(_SMPLX_FOOT_JOINT_IDS)]                # (4, V)
    J_template_feet  = rows @ v_template                             # (4, 3)
    J_shapedirs_feet = torch.einsum("jv,vcb->jcb", rows, shapedirs) # (4, 3, B)
    _SMPLX_FOOT_CACHE[key] = (J_template_feet, J_shapedirs_feet)
    return _SMPLX_FOOT_CACHE[key]


def compute_smpl_floor_height(betas, J_template_pelvis, J_shapedirs_pelvis,
                               J_template_feet, J_shapedirs_feet):
    """T-pose 近似下的地面高度 (SMPL y-up local 坐标系, 单位 m).

    系统统一 y-up: y 轴即竖直方向.
    pelvis 被放在 world 原点 (y=0), 脚在 y<0 的位置 (T-pose).
    floor_height = pelvis_y_smpl - min(foot_y_smpl)
    返回正数, 即 pelvis 关节距地面的高度.
    """
    pelvis_y = float((J_template_pelvis + J_shapedirs_pelvis @ betas)[1])
    feet_pos  = J_template_feet + torch.einsum("jcb,b->jc", J_shapedirs_feet, betas)
    min_foot_y = float(feet_pos[:, 1].min())
    return pelvis_y - min_foot_y               # > 0


# ==========================================================
# 首帧归一化核心辅助
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
    return axis_angle_to_matrix(aa)                        # (3, 3) 纯绕 up 轴旋转


# ==========================================================
# canonicalize_smpl_first_frame
# ==========================================================

def canonicalize_smpl_first_frame(global_orient_aa, transl, pelvis_offset=None,
                                  yaw_only=False, up_axis=1, fwd_axis=2):
    """全帧应用 R0^T（或仅去 yaw 的 R0_inv），使第 0 帧 pelvis 落原点。

    Args:
        global_orient_aa: (F, 3) 轴角
        transl:           (F, 3)
        pelvis_offset:    (3,) 可选，= J_pelvis(betas)。给定时让第 0 帧 pelvis 关节落到原点；
                          不给则退化为"让第 0 帧 transl=0"的旧口径。
        yaw_only:         True 时只去首帧航向(绕 up_axis 的 yaw)，保留 pitch/roll，重力不被转歪。
        up_axis/fwd_axis: 竖直轴 / forward 轴索引。
                          SMPL 在 y-up 系: up=1(y), fwd=2(+z)。

    Returns:
        global_orient_aa_new: (F, 3)
        transl_new:           (F, 3)

    典型调用 (G1AmassDualPthDataset, y-up, yaw-only):
        smpl["global_orient"], smpl["transl"] = canonicalize_smpl_first_frame(
            smpl["global_orient"], smpl["transl"],
            pelvis_offset=pelvis_offset,
            yaw_only=True, up_axis=1, fwd_axis=2,
        )
    """
    R = axis_angle_to_matrix(global_orient_aa)                              # (F, 3, 3)
    R0_inv = _first_frame_inv_rot(
        R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )                                                                        # (3, 3)
    R_new      = R0_inv @ R                                                  # (F, 3, 3)
    transl_new = (transl - transl[0:1]) @ R0_inv.T                          # (F, 3)
    #   transl_new[0] = 0 after centering

    if pelvis_offset is not None:
        # pelvis_world[0] = transl_new[0] + pelvis_offset = pelvis_offset；再减去归零。
        transl_new = transl_new - pelvis_offset
    return matrix_to_axis_angle(R_new), transl_new


def canonicalize_smpl_first_frame_bones_seed(global_orient_aa, joints,
                                             yaw_only=False, up_axis=2, fwd_axis=1):
    """Bones Seed/SOMA variant: center the first root joint and normalize heading."""
    R = axis_angle_to_matrix(global_orient_aa)
    R0_inv = _first_frame_inv_rot(
        R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )
    R_new = R0_inv @ R
    joints_new = (joints - joints[[0], [0]]) @ R0_inv.T[None]
    return matrix_to_axis_angle(R_new), joints_new


# ==========================================================
# canonicalize_g1_first_frame
# ==========================================================

def canonicalize_g1_first_frame(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    """对 G1 全字段应用 R0^T（或仅去 yaw 的 R0_inv，绕原点），使第 0 帧 root_body 位置=0。

    g1 fields（in-place 修改，同时返回 g1）:
        body_pos_w:     (F, N, 3)
        body_quat_w:    (F, N, 4) wxyz
        body_lin_vel_w: (F, N, 3)
        body_ang_vel_w: (F, N, 3)
    其它字段（joint_pos/joint_vel/fps）属于关节空间，不受世界刚体变换影响。

    yaw_only=False: 旧行为，首帧 root 姿态归 identity（全量去除 pitch/roll/yaw）。
    yaw_only=True : 只去首帧航向(绕 up_axis 的 yaw)，保留 pitch/roll，重力不被转歪；
                    此时首帧 root quat 不会被钉成 identity（否则会抹掉保留的 tilt）。
    up_axis/fwd_axis: G1 在 z-up 系: up=2(z), fwd=0(URDF forward=+x)。

    典型调用 (G1AmassDualPthDataset, z-up native, yaw-only, 在 az_to_ay 之前):
        g1 = canonicalize_g1_first_frame(
            g1, root_body_id=0,
            yaw_only=True, up_axis=2, fwd_axis=0,
        )
    """
    body_quat_w = g1["body_quat_w"]                                         # (F, N, 4) wxyz
    body_pos_w  = g1["body_pos_w"]                                          # (F, N, 3)

    R = quaternion_to_matrix(body_quat_w)                                   # (F, N, 3, 3)
    R0_inv = _first_frame_inv_rot(
        R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )                                                                        # (3, 3)
    center = body_pos_w[0, root_body_id].clone()                            # (3,)

    R_new = R0_inv @ R                                                      # (F, N, 3, 3)
    q_new = matrix_to_quaternion(R_new)                                     # (F, N, 4) wxyz, ambiguous ±q

    # 规范双覆盖：令 w ≥ 0，得到确定符号
    sign = torch.where(
        q_new[..., 0:1] < 0,
        -torch.ones_like(q_new[..., 0:1]),
         torch.ones_like(q_new[..., 0:1]),
    )
    q_new = q_new * sign

    if not yaw_only:
        # 显式钉死首帧 root 为 identity，避免 1e-7 浮点噪声。
        # yaw_only 下首帧 root 仍带 pitch/roll，不能钉成 identity。
        q_new[0, root_body_id] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype
        )

    g1["body_quat_w"]    = q_new                                            # (F, N, 4) wxyz
    g1["body_pos_w"]     = (body_pos_w - center) @ R0_inv.T                 # (F, N, 3)
    g1["body_lin_vel_w"] = g1["body_lin_vel_w"] @ R0_inv.T                  # (F, N, 3)
    g1["body_ang_vel_w"] = g1["body_ang_vel_w"] @ R0_inv.T                  # (F, N, 3)
    # 显式钉死首帧 root 位置为 0
    g1["body_pos_w"][0, root_body_id] = 0.0
    return g1


def canonicalize_g1_first_frame_bones_seed(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    """Bones Seed/G1 variant using FK full-body fields from the paired pkl files."""
    body_quat_w = g1["body_quat_w_full"]
    body_pos_w = g1["body_pos_w_full"]

    R = quaternion_to_matrix(body_quat_w)
    R0_inv = _first_frame_inv_rot(
        R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )
    center = body_pos_w[0, root_body_id].clone()

    R_new = R0_inv @ R
    q_new = matrix_to_quaternion(R_new)
    sign = torch.where(
        q_new[..., 0:1] < 0,
        -torch.ones_like(q_new[..., 0:1]),
        torch.ones_like(q_new[..., 0:1]),
    )
    q_new = q_new * sign

    if not yaw_only:
        q_new[0, root_body_id] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype
        )

    g1["body_quat_w"] = q_new
    g1["body_pos_w"] = (body_pos_w - center) @ R0_inv.T
    g1["root_rot"] = g1["body_quat_w"][:, root_body_id].clone()
    g1["root_pos"] = g1["body_pos_w"][:, root_body_id].clone()
    return g1
