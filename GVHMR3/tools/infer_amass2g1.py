"""
推理脚本: 给定一个 AMASS npz 文件，使用训练好的模型输出 G1 机器人数据。

用法:
    conda run --no-capture-output -n gvhmr3 python tools/infer_amass2g1.py \
        --amass_npz  path/to/your_amass.npz \
        --ckpt       outputs/g1_paired_amass/g1_motion_v1/checkpoints/e499-s001000.ckpt \
        --output     outputs/infer_g1/result.npz

说明:
    - amass_npz: AMASS 格式 npz (含 trans/transl, root_orient/global_orient, pose_body/body_pose, betas)
    - ckpt:      训练的 checkpoint 路径
    - output:    推理结果保存路径 (npz 格式)，包含:
                   joint_pos     (T, 29)  每个关节 1 维 scalar dof, BYD 顺序 (frame-agnostic)
                   root_pos_w    (T, 3)   pelvis 世界系坐标 (z-up, G1 URDF 原生)
                   root_quat_w   (T, 4)   pelvis 世界系四元数 (wxyz, z-up)
                 注: 网络内部在 y-up world 系工作 (与训练 dataset apply_az_to_ay_g1 一致),
                     这里在保存前对 root_pos_w / root_quat_w 做 y-up → z-up 反变换,
                     所以 npz 直接拿去喂 IsaacGym/IsaacLab/Mujoco 等 z-up 平台。
    - --window:  滑窗长度，默认 120（与训练一致）
    - --device:  cuda / cpu
    - --mjcf:    G1 MJCF 路径，需与训练一致，默认 unitree_description/mjcf/g1.xml
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

# ---------- project imports ----------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam
from hmr4d.utils.geo.hmr_cam import (
    create_camera_sensor,
    perspective_projection,
    normalize_kp2d,
    safely_render_x3d_K,
    get_bbx_xys,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.geo.hmr_global import get_c_rootparam
from hmr4d.utils.net_utils import get_valid_mask

from hmr4d.dataset.pure_motion.cam_traj_utils import CameraAugmentorV11
from hmr4d.dataset.pure_motion.utils import interpolate_smpl_params  # 训练侧的版本 (aa→R6D 插值)
from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears


# ====================================================================
# 工具函数
# ====================================================================

def npz_get(npz_obj, candidates, required=True, path=""):
    for k in candidates:
        if k in npz_obj.files:
            return npz_obj[k]
    if required:
        raise KeyError(f"Missing keys {candidates} in {path}, available: {list(npz_obj.files)}")
    return None


# NOTE: 之前推理脚本自己实现的 interpolate_smpl_params 在 body_pose 上做 AA 直接线性插值,
#       与训练侧 (hmr4d/dataset/pure_motion/utils.py) 走 aa→R6D→linear→aa 的做法不同;
#       在 src!=tgt fps 的重采样里会给出与训练 distribution 不一致的中间姿态.
#       现已删除本地实现, 改 import 训练侧那份, 见文件顶部 import.


# ====================================================================
# 从 AMASS npz 生成模型所需的 batch 输入
# ====================================================================

def _set_seed(seed=42):
    """让 np.random 决定的相机轨迹可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def reduce_singular_chains(joint_pos_byd):
    """把网络输出落在 axis-angle 奇异点 (θ ≈ ±π) 的 3 关节链
    重新做 Euler 分解,得到等价的小角度。

    encoder 用 [0, θ, 0] 编码每个 joint, 再用 r6d→matrix→matrix_to_axis_angle[..., 1]
    解出 θ。在 θ ≈ ±π 附近, axis 选择有歧义,网络若把 rotmat 落在该奇异点,会同时给出
    pitch=π / roll=π / yaw=π。在物理 FK 里:
        R_total = R_y(π) @ R_x(π) @ R_z(π) = I  (累计 0 旋转)
    所以 fk_direct / 小球可视化看着对; 但 Blender 的 bone-local Y 复合不一定抵消。

    本函数按各链的物理轴顺序累计 R, 再用同顺序 Euler 分解, 把 (π, π, π) 这类等价于
    单位旋转的组合还原成 (0, 0, 0); 累积旋转**严格不变**,仅每关节角度被拉回小区间。
    """
    from scipy.spatial.transform import Rotation as Rot

    # 每条 3-关节链: (BYD 索引按 kinematic-chain 顺序, 物理轴 Euler order)
    # URDF 链路: parent → ... → child;axis 取 G5_AXIS 的物理轴。
    chains = [
        # 髋: pitch (Y) → roll (X) → yaw (Z)
        ([0, 3, 6],  "YXZ"),  # left_hip
        ([1, 4, 7],  "YXZ"),  # right_hip
        # 腰: yaw (Z) → roll (X) → pitch (Y)
        ([2, 5, 8],  "ZXY"),  # waist
        # 肩: pitch (Y) → roll (X) → yaw (Z)
        ([11, 15, 19], "YXZ"),  # left_shoulder
        ([12, 16, 20], "YXZ"),  # right_shoulder
        # 腕: roll (X) → pitch (Y) → yaw (Z)
        ([23, 25, 27], "XYZ"),  # left_wrist
        ([24, 26, 28], "XYZ"),  # right_wrist
    ]
    axis_map = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}

    out = joint_pos_byd.copy()
    for indices, order in chains:
        i1, i2, i3 = indices
        a1, a2, a3 = order  # e.g., 'Y', 'X', 'Z'
        rv1 = axis_map[a1][None, :] * out[:, i1, None]
        rv2 = axis_map[a2][None, :] * out[:, i2, None]
        rv3 = axis_map[a3][None, :] * out[:, i3, None]
        R_total = Rot.from_rotvec(rv1) * Rot.from_rotvec(rv2) * Rot.from_rotvec(rv3)
        new_angles = R_total.as_euler(order, degrees=False)  # (T, 3)
        out[:, i1] = new_angles[:, 0]
        out[:, i2] = new_angles[:, 1]
        out[:, i3] = new_angles[:, 2]
    return out


def make_batch_from_amass(amass_path, window=120, device="cuda", tgt_fps=30.0, seed=42,
                          input_axis_up="z"):
    """
    读取 AMASS npz → 插值到 tgt_fps → 整段一次性合成相机 + 算 2D obs，
    再按窗口切片返回。多窗口间共用同一组 T_w2c，避免每窗随机相机的不一致。

    Returns:
        prep: dict 含 (在 device 上) 的 obs/bbx_xys/cam_angvel/T_w2c/K_fullimg 全长 tensor + chunks_meta
        total_len: int, 重采样后的帧数
    """
    _set_seed(seed)
    amass_path = str(amass_path)
    with np.load(amass_path) as data:
        trans = npz_get(data, ["trans", "transl"], path=amass_path)
        root_orient = npz_get(data, ["root_orient", "global_orient"], path=amass_path)
        pose_body = npz_get(data, ["pose_body", "body_pose"], path=amass_path)
        betas = npz_get(data, ["betas"], path=amass_path)
        src_fps_val = npz_get(
            data,
            ["mocap_framerate", "mocap_frame_rate", "fps"],
            required=False,
            path=amass_path,
        )

    src_len = trans.shape[0]
    if src_fps_val is None:
        src_fps = 120.0  # AMASS 默认 120fps（与 g1_amass.py:297 同口径）
        print(f"[Warn] {amass_path} 未找到 mocap_framerate，假定 {src_fps} fps")
    else:
        src_fps = float(src_fps_val)

    smpl_data = {
        "transl": torch.tensor(trans, dtype=torch.float32),
        "global_orient": torch.tensor(root_orient, dtype=torch.float32),
        "body_pose": torch.tensor(pose_body, dtype=torch.float32),
        "betas": torch.tensor(betas, dtype=torch.float32),
    }
    # betas 先按源帧数对齐（之后跟其他字段一起重采样）
    if smpl_data["betas"].ndim == 1:
        smpl_data["betas"] = smpl_data["betas"][:10].unsqueeze(0).expand(src_len, -1).contiguous()
    else:
        smpl_data["betas"] = smpl_data["betas"][:src_len, :10]

    # ----- 插值到 tgt_fps（30fps）-----
    if abs(src_fps - tgt_fps) > 1e-3:
        tgt_len = max(2, int(round(src_len * tgt_fps / src_fps)))
        print(f"[Info] AMASS 文件: {amass_path}")
        print(f"       重采样: {src_len} 帧 @ {src_fps:g}fps → {tgt_len} 帧 @ {tgt_fps:g}fps")
        smpl_data = interpolate_smpl_params(smpl_data, tgt_len)
        total_len = tgt_len
    else:
        print(f"[Info] AMASS 文件: {amass_path}, 总帧数: {src_len} @ {src_fps:g}fps (无需重采样)")
        total_len = src_len

    # 坐标系转换: az -> ay (仅当输入是 z-up 时).
    # 训练 dataset (g1_amass.py:1788) 读 filtered_amass.pth 时已是 y-up, 不再做这一步.
    # 如果用户从 filtered_amass.pth 反 dump npz 喂进来, 必须传 --input_axis_up=y 以免双转.
    assert input_axis_up in ("z", "y"), f"input_axis_up 必须是 z|y, got {input_axis_up!r}"
    if input_axis_up == "z":
        smpl_data["global_orient"], smpl_data["transl"], _ = get_tgtcoord_rootparam(
            smpl_data["global_orient"],
            smpl_data["transl"],
            tsf="az->ay",
        )
    else:
        print("[Info] input_axis_up=y, 跳过 az->ay 转换 (输入视为已 y-up)")

    # 移到 device 上加速 FK
    smpl_data = {k: v.to(device) for k, v in smpl_data.items()}

    # SMPL 模型 (需要做 FK 拿到 3D 关节点来生成合成相机和 2D 投影)
    smplx = make_smplx("supermotion_v437coco17").to(device)
    smplx_lite = make_smplx("supermotion_smpl24").to(device)

    L_full = total_len

    # ===== 整段一次性 FK 拿世界系 j3d（用于合成相机）=====
    # NOTE: 必须逐帧 FK, 不能子采样.
    # 训练侧 base_dataset.py:75-95 也是逐帧:
    #   "不再每 10 帧采一次 (旧的 N=10 让 cam_augmentor 看到的 j3d 阶梯化, 手脚远端
    #    在窗口内的突出动作会被漏掉, push-away/FoV 检查给出错的 z_trg)"
    # 推理这里若再子采样, CameraAugmentorV11 看到的轨迹与训练分布不一致 → 模型 OOD.
    with torch.no_grad():
        w_j3d_full = smplx_lite(
            smpl_data["body_pose"],
            smpl_data["betas"],
            smpl_data["global_orient"],
            None,
        )                                                                      # (L, J, 3)
    w_j3d_full = w_j3d_full + smpl_data["transl"][:L_full, None]              # 加根平移

    # ===== 整段单一相机轨迹（关键！避免每窗随机相机不一致）=====
    width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)
    K_fullimg = K_fullimg.to(device)
    cam_augmentor = CameraAugmentorV11()
    T_w2c_full = cam_augmentor(w_j3d_full.cpu(), L_full).to(device)            # 内部强制 cpu 路径; (L, 4, 4)

    # ===== 整段算相机系 SMPL 参数 → 2D obs / bbx =====
    offset = smplx.get_skeleton(smpl_data["betas"][0])[0]
    go_c_full, transl_c_full = get_c_rootparam(
        smpl_data["global_orient"], smpl_data["transl"], T_w2c_full, offset
    )
    with torch.no_grad():
        gt_verts437_full, gt_j3d_full = smplx(
            body_pose=smpl_data["body_pose"],
            betas=smpl_data["betas"],
            global_orient=go_c_full,
            transl=transl_c_full,
        )

    K_expand_full = K_fullimg.unsqueeze(0).expand(L_full, -1, -1)              # (L, 3, 3)
    i_x2d_full   = safely_render_x3d_K(gt_verts437_full.unsqueeze(0), K_expand_full.unsqueeze(0), thr=0.3)
    bbx_xys_full = get_bbx_xys(i_x2d_full, do_augment=False)[0]                # (L, 3)
    obs_i_j2d_full = perspective_projection(gt_j3d_full.unsqueeze(0), K_expand_full.unsqueeze(0))[0]  # (L, 17, 2)
    # 与训练一致 (gvhmr_pl._build_synthetic_obs do_augment=False 路径):
    #   1) j2d_visible_mask = (gt_j3d[..., 2] >= 0.3)
    #   2) obs_kp2d 第 3 列存 visible_mask
    #   3) obs = normalize_kp2d(obs_kp2d, bbx_xys)
    #   4) obs[~visible_mask] = 0  ←  整条 (x, y, conf) 在 z<0.3 处置 0
    # 漏掉 (4) 的话: perspective_projection 在 z 接近 0 时给出极大的 x/z,
    # normalize_kp2d 后仍是大数, 训练分布永远不出现 → 网络 OOD.
    j2d_visible_mask = gt_j3d_full[..., 2] >= 0.3                                # (L, 17)
    conf_full      = j2d_visible_mask.float().unsqueeze(-1)                      # (L, 17, 1)
    obs_kp2d_full  = torch.cat([obs_i_j2d_full, conf_full], dim=-1)
    obs_full       = normalize_kp2d(obs_kp2d_full.unsqueeze(0), bbx_xys_full.unsqueeze(0))[0]         # (L, 17, 3)
    obs_full[~j2d_visible_mask] = 0                                              # 与训练 step (4) 对齐
    obs_full       = select_coco17_no_nose_ears(obs_full)                        # (L, 14, 3)
    cam_angvel_full = compute_cam_angvel(T_w2c_full[:, :3, :3])                # (L, 6)

    # ===== 滑窗切分：1 帧重叠用于跨窗 transl_0 衔接 =====
    if L_full <= window:
        chunks_meta = [(0, L_full)]
    else:
        starts = [0]
        while starts[-1] + window < L_full:
            starts.append(starts[-1] + window - 1)              # overlap by 1 frame
        chunks_meta = [(s, min(window, L_full - s)) for s in starts]

    print(f"[Info] 共 {len(chunks_meta)} 个滑窗 (window={window}, overlap=1 frame)")

    prep = {
        "obs_full":        obs_full,
        "bbx_xys_full":    bbx_xys_full,
        "K_fullimg":       K_fullimg,
        "cam_angvel_full": cam_angvel_full,
        "T_w2c_full":      T_w2c_full,
        "chunks_meta":     chunks_meta,
        "window":          window,
        "total_len":       L_full,
        "obs_kp2d_pixel":  obs_kp2d_full,
    }
    return prep, L_full


# ====================================================================
# 加载模型
# ====================================================================

def load_pipeline(ckpt_path, exp="gvhmr/g1_dualpth", device="cuda", overrides=None):
    """用 hydra compose 读训练 yaml → instantiate Pipeline → 加载 ckpt 权重.

    比手写 args 更稳: 训练 yaml 加 / 改任何 weight / static_conf / robot 字段都会
    自动同步, 不会再因为推理脚本写死的默认值与训练偏离而静默错位.
    """
    from hydra import initialize_config_module, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr

    register_store_gvhmr()
    # GlobalHydra 单例: 同进程二次调用会抛 "already initialized"
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    print(f"[Info] 用 hydra compose 读 exp={exp} 的训练 yaml")
    overrides_list = [f"exp={exp}"] + list(overrides or [])
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="train", overrides=overrides_list)

    # Pipeline.__init__ 内部完成: endecoder + setup_g1_fk + g1_pred_stats +
    # pred_cam_stats + cam_angvel 归一化 buffer 全部按 yaml 配置注册到位.
    pipeline = instantiate(cfg.pipeline, _recursive_=False)

    # ---- 加载 ckpt 权重 ----
    print(f"[Info] 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["state_dict"]
    # 去掉 "pipeline." 前缀
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("pipeline."):
            new_sd[k[len("pipeline."):]] = v
        else:
            new_sd[k] = v

    missing, unexpected = pipeline.load_state_dict(new_sd, strict=False)
    # 关键校验：denoiser3d 任何 weight 进了 missing → 网络拿的是随机初始化，输出必然垃圾。
    denoiser_missing = [k for k in missing if k.startswith("denoiser3d.")]
    denoiser_unexpected = [k for k in unexpected if k.startswith("denoiser3d.")]
    if denoiser_missing or denoiser_unexpected:
        print(f"[ERROR] denoiser3d state_dict mismatch:")
        if denoiser_missing:
            print(f"  Missing ({len(denoiser_missing)}): {denoiser_missing[:5]}{'...' if len(denoiser_missing) > 5 else ''}")
        if denoiser_unexpected:
            print(f"  Unexpected ({len(denoiser_unexpected)}): {denoiser_unexpected[:5]}{'...' if len(denoiser_unexpected) > 5 else ''}")
        raise RuntimeError("denoiser3d 权重未正确加载。检查 output_dim / pred_cam_dim / static_conf_dim / imgseq_dim 与训练 yaml 是否一致。")
    if missing:
        print(f"[Warn] Missing non-denoiser keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected non-denoiser keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    pipeline.eval()
    pipeline = pipeline.to(device)
    print(f"[Info] Pipeline 加载完毕, device={device}")
    return pipeline


# ====================================================================
# 推理 & 合并结果
# ====================================================================

@torch.no_grad()
def infer(pipeline, prep, total_len, device="cuda", singular_rewrite="none"):
    """顺序推理 + 跨窗 transl/rotation 联合衔接.

    策略:
      - 单段: 当 total_len <= window, 一发推理直接拿 [:total_len].
      - 多段: 1 帧重叠的非重叠切窗.
        (a) 把 chunk N-1 最后一帧的预测 root_pos 塞进 g1_target.g1_body_pos_w,
            让 pipeline 的 rollout_local_transl_vel 把 transl_0 锚到该位置.
        (b) pipeline 不接受 rotation anchor (推理路径只读 g1_body_pos_w),
            所以 chunk N 输出后 post-hoc 旋转: 计算 delta_R = R_prev_last @ R_curr_first.T,
            把整段 quat 左乘 delta_R, 把 position 相对 frame-0 的位移用 delta_R 转向,
            保证 chunk N frame-0 严格等于 chunk N-1 frame-last (位置+朝向都连续).
        (c) 写入时跳过 chunk N>0 的 frame 0 (与上窗重叠).

    rotation alignment 是关键 — 没有它每 119 帧朝向会"跳一下" (网络对新窗独立预测 R_0).
    """
    obs_full        = prep["obs_full"]
    bbx_xys_full    = prep["bbx_xys_full"]
    K_fullimg       = prep["K_fullimg"]
    cam_angvel_full = prep["cam_angvel_full"]
    T_w2c_full      = prep["T_w2c_full"]
    chunks_meta     = prep["chunks_meta"]
    window          = prep["window"]

    joint_pos_out = torch.zeros(total_len, 29, device=device)
    root_pos_out  = torch.zeros(total_len, 3,  device=device)
    root_quat_out = torch.zeros(total_len, 4,  device=device)   # wxyz

    def pad_window(x, pad_len):
        if pad_len <= 0:
            return x
        last = x[-1:].expand(pad_len, *x.shape[1:])
        return torch.cat([x, last], dim=0)

    prev_last_pos  = None  # (3,)  上窗最后一帧的预测 root_pos
    prev_last_quat = None  # (4,)  上窗最后一帧的预测 root_quat (wxyz, post-alignment 后的)

    for chunk_idx, (start, actual_len) in enumerate(chunks_meta):
        end = start + actual_len
        pad_len = window - actual_len
        L = window

        obs       = pad_window(obs_full[start:end],        pad_len)
        bbx_xys   = pad_window(bbx_xys_full[start:end],    pad_len)
        cam_angvel= pad_window(cam_angvel_full[start:end], pad_len)
        T_w2c     = pad_window(T_w2c_full[start:end],      pad_len)

        batch = {
            "length":    torch.tensor([actual_len], device=device, dtype=torch.long),
            "obs":       obs.unsqueeze(0),
            "bbx_xys":   bbx_xys.unsqueeze(0),
            "K_fullimg": K_fullimg.unsqueeze(0).expand(L, -1, -1).unsqueeze(0),
            "cam_angvel":cam_angvel.unsqueeze(0),
            "f_imgseq":  torch.zeros(1, L, 1024, device=device),  # network imgseq_dim=0 时被忽略
            "T_w2c":     T_w2c.unsqueeze(0),
        }

        # 跨窗 transl 锚: 把 prev 最后一帧的世界系 root_pos 塞进 g1_target,
        # pipeline 读 g1_target["g1_body_pos_w"][:, [0], 0] 作 root_pos_0.
        if prev_last_pos is not None:
            g1_body_pos_w = torch.zeros(1, 1, 1, 3, device=device, dtype=prev_last_pos.dtype)
            g1_body_pos_w[0, 0, 0] = prev_last_pos
            batch["g1_target"] = {"g1_body_pos_w": g1_body_pos_w}

        outputs = pipeline.forward(batch, train=False)

        pred_jp = outputs["pred_g1_joint_pos"][0, :actual_len]   # (actual_len, 29)
        pred_rp = outputs["pred_root_pos"][0, :actual_len]       # (actual_len, 3)
        pred_rq = outputs["pred_root_quat_w"][0, :actual_len]    # (actual_len, 4) wxyz

        # --- chunk N>0: post-hoc rotation alignment ---
        # delta_R 使 chunk N frame-0 朝向严格等于 chunk N-1 frame-last 朝向.
        # 同时把位移 (pred_rp - pred_rp[0]) 用 delta_R 转向, 保证轨迹方向也对齐.
        if prev_last_quat is not None:
            R_prev_last  = quaternion_to_matrix(prev_last_quat)             # (3, 3)
            R_curr_first = quaternion_to_matrix(pred_rq[0])                 # (3, 3)
            delta_R      = R_prev_last @ R_curr_first.transpose(-1, -2)     # (3, 3)

            # 位置: 相对 chunk frame-0 的位移用 delta_R 旋转后回到 prev_last_pos
            disp     = pred_rp - pred_rp[0:1]                               # (L, 3)
            pred_rp  = pred_rp[0:1] + disp @ delta_R.transpose(-1, -2)      # delta_R @ disp

            # 朝向: 整段左乘 delta_R
            R_curr_all = quaternion_to_matrix(pred_rq)                      # (L, 3, 3)
            R_aligned  = delta_R.unsqueeze(0) @ R_curr_all                  # (L, 3, 3)
            pred_rq    = matrix_to_quaternion(R_aligned)                    # (L, 4) wxyz

        # 写入: chunk 0 全写; chunk N>0 跳过 frame 0 (与上窗最后一帧重叠).
        if chunk_idx == 0:
            joint_pos_out[start:end]  = pred_jp
            root_pos_out [start:end]  = pred_rp
            root_quat_out[start:end]  = pred_rq
        else:
            joint_pos_out[start + 1:end]  = pred_jp[1:]
            root_pos_out [start + 1:end]  = pred_rp[1:]
            root_quat_out[start + 1:end]  = pred_rq[1:]

        # 更新衔接信号 (用对齐后的值, 这样下一窗的 alignment 是累积一致的)
        prev_last_pos  = pred_rp[actual_len - 1].detach()
        prev_last_quat = pred_rq[actual_len - 1].detach()

    # ---- 旋转 y-up (训练系) → URDF z-up world ----
    # 训练 g1_amass.py:1810-1815 对 G1 GT 做了两步:
    #   Step 1: az_to_ay      v_y = v_z @ T_az2ay.T          R_y = T_az2ay @ R_z
    #   Step 2: Ry(-90°)      v_yr = v_y @ Ry_neg90.T        R_yr = Ry_neg90 @ R_y
    # 网络输出 pred_root_pos / pred_root_quat_w 都在 (az_to_ay + Ry(-90°)) 之后的"旋转 y-up"系.
    # 逆向回 z-up URDF:
    #   Step A (撤 Ry(-90°)):  v_y = v_yr @ Ry_neg90          R_y = Ry_neg90.T @ R_yr
    #   Step B (撤 az_to_ay):  v_z = v_y @ T_az2ay            R_z = T_az2ay.T @ R_y
    # 合并: v_z = v_yr @ (Ry_neg90 @ T_az2ay),  R_z = (Ry_neg90 @ T_az2ay).T @ R_yr.
    #
    #   T_az2ay   = [[1,0,0],[0,0,1],[0,-1,0]]
    #   Ry_neg90  = [[0,0,-1],[0,1,0],[1,0,0]]
    #   M_pos     = Ry_neg90 @ T_az2ay = [[0,1,0],[0,0,1],[1,0,0]]   (3-cycle x→z, y→x, z→y)
    # joint_pos 是 scalar dof, frame-invariant, 不需要变换.
    M_pos = torch.tensor(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]],
        device=device, dtype=root_pos_out.dtype,
    )
    root_pos_out = root_pos_out @ M_pos                                              # 位置: yr → z
    R_yr = quaternion_to_matrix(root_quat_out)                                       # (T, 3, 3)
    R_zup = torch.matmul(M_pos.transpose(-1, -2), R_yr)                              # R_z = M_pos.T @ R_yr
    root_quat_out = matrix_to_quaternion(R_zup)                                      # (T, 4) wxyz, z-up

    # 单位化 quaternion（每帧由网络独立产出，平时已接近单位长度，这里收个尾）
    root_quat_out = F.normalize(root_quat_out, dim=-1)

    # joint_pos 后处理:
    #   none    -- 直接保存网络原始输出. 用于 Mujoco / IsaacLab 这种 per-joint PD
    #              控制 + URDF 关节限位的场合 (改写 joint_pos 会改变 reference 轨迹和
    #              joint vel, 还可能把单关节角度推到 URDF 限位外).
    #   blender -- 把 3-joint chain (hip/waist/shoulder/wrist) 上 (π,π,π) 这类奇异
    #              组合用 Euler 重新分解, 累计旋转 (FK) 不变, 每关节角度落到小区间.
    #              仅适合 Blender bone-local 可视化, 不要用作物理仿真 reference.
    assert singular_rewrite in ("none", "blender"), singular_rewrite
    joint_pos_np = joint_pos_out.cpu().numpy()
    if singular_rewrite == "blender":
        joint_pos_np = reduce_singular_chains(joint_pos_np)

    return {
        "joint_pos":   joint_pos_np,                   # (T, 29) BYD scalar dofs (frame-invariant)
        "root_pos_w":  root_pos_out.cpu().numpy(),     # (T, 3)  z-up world
        "root_quat_w": root_quat_out.cpu().numpy(),    # (T, 4)  wxyz, z-up world
    }


# ====================================================================
# main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="AMASS → G1 推理")
    parser.add_argument("--amass_npz", type=str, required=True, help="输入 AMASS npz 文件路径")
    parser.add_argument("--ckpt", type=str,
                        default="outputs/g1_paired_amass/g1_motion_v1/checkpoints/e499-s070000.ckpt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="输出 npz 路径 (默认: outputs/infer_g1/<输入文件名>.npz)")
    parser.add_argument("--window", type=int, default=120, help="滑窗长度 (与训练一致)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument("--exp", type=str, default="gvhmr/g1_dualpth",
                        help="训练 exp yaml 名 (相对 hmr4d/configs/exp/), Pipeline 按此 instantiate")
    parser.add_argument("--input_axis_up", type=str, default="z", choices=["z", "y"],
                        help="输入 AMASS 的 up 轴: 原 AMASS npz 是 z; 从 filtered_amass.pth 反 dump 是 y")
    parser.add_argument("--tgt_fps", type=float, default=30.0,
                        help="把输入 AMASS 重采样到该帧率（与训练对齐，默认 30）")
    parser.add_argument("--seed", type=int, default=42,
                        help="合成相机轨迹随机种子（保证可复现）")
    parser.add_argument("--singular_rewrite", type=str, default="none", choices=["none", "blender"],
                        help="joint_pos 后处理: none=原始 (给 Mujoco/IsaacLab); blender=3-chain "
                             "Euler 再分解 (仅 Blender 可视化用, 会改变 per-joint 轨迹)")
    args = parser.parse_args()

    # 输出路径
    if args.output is None:
        out_dir = Path("outputs/infer_g1")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / (Path(args.amass_npz).stem + "_g1.npz"))
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 1) 加载模型 (hydra compose 读 exp yaml → instantiate Pipeline → 加载 ckpt)
    pipeline = load_pipeline(args.ckpt, exp=args.exp, device=args.device)

    # 2) 准备输入 (整段一次相机 + 逐帧 FK, 再切窗口)
    prep, total_len = make_batch_from_amass(
        args.amass_npz, window=args.window, device=args.device,
        tgt_fps=args.tgt_fps, seed=args.seed, input_axis_up=args.input_axis_up,
    )

    # 3) 推理 (顺序 + 跨窗 transl/rotation 联合衔接)
    results = infer(pipeline, prep, total_len, device=args.device,
                    singular_rewrite=args.singular_rewrite)

    # 4) 保存
    np.savez(
        args.output,
        joint_pos=results["joint_pos"],          # (T, 29) BYD
        root_pos_w=results["root_pos_w"],        # (T, 3)  z-up world (G1 URDF 原生)
        root_quat_w=results["root_quat_w"],      # (T, 4)  wxyz, z-up world
    )
    print(f"[Done] 结果已保存到: {args.output}")
    print(f"  - joint_pos:   {results['joint_pos'].shape}   (BYD scalar dofs)")
    print(f"  - root_pos_w:  {results['root_pos_w'].shape}   (z-up world)")
    print(f"  - root_quat_w: {results['root_quat_w'].shape}   (wxyz, z-up world)")


if __name__ == "__main__":
    main()
