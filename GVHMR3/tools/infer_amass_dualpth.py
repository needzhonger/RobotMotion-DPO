"""按训练代码 (G1AmassDualPthDataset, split=train, 全增强) 取样本 → 跑模型推 pred →
出 (1) 2x2 视频: 人2D | GT机器人2D | pred机器人2D | 3D骨架叠加视角,
   (2) bundle npz (z-up, 给 blender_skeleton_dualpth.py 画关节小球).

数据增强 (与训练一致):
  - 数据集级: 速度/起点增强 + CameraAugmentorV11 相机位置/速度增强 (split=train 自带).
    train 路径不自重 seed → 本脚本在每次 ds[idx] 前 seed, 保证可复现.
  - 训练态 obs 噪声 + bbox 抖动: 复刻 gvhmr_pl._build_synthetic_obs(do_augment=True).
    默认开 (--obs_augment 1); 设 0 切到 validation 干净 obs. 相机增强始终在.

模型 forward 走 train=False (推理 rollout 拿 pred). pred / GT / 人 都在同一 canonical
y-up 世界 (pelvis 近原点, 脚踩地 y=0), 视频/blend 都不做并排偏移、不做 yaw 对齐.

用法 (repo 根, gvhmr3 env):
    conda run -n gvhmr3 python tools/infer_viz_dualpth.py \
        --ckpt outputs/g1_amass_dualpth/g1_dualpth_v3/checkpoints/e199-s027400.ckpt \
        --idx 0 1 2 3 --split train --seed 42 --out outputs/infer_viz_dualpth \
        --blend 1 --blender /home/eerrr/.local/bin/blender
"""
import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _REPO_ROOT / "tools"
for _p in (str(_REPO_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch

from hmr4d.dataset.pure_motion.g1_amass import G1AmassDualPthDataset, apply_ay_to_az_on_vec
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.hmr_cam import (
    safely_render_x3d_K,
    get_bbx_xys,
    perspective_projection,
    normalize_kp2d,
)
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_modify_hands_legs,
)
from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears

# 复用 2D 渲染 helper (project/draw/bbox/skeleton 定义)
from render_dual_camera_dualpth import (
    project_points,
    draw_skeleton,
    draw_bbox_xys,
    label,
    compute_amass_bbx_xys,
    compute_g1_bbx_xys,
    SMPL22_BONES,
    G1_BONES,
)
# 模型加载 (hydra compose exp yaml → instantiate Pipeline → load ckpt → eval)
from infer_amass2g1 import load_pipeline


# 颜色 (BGR)
HUMAN_COLOR = (210, 100, 30)    # 蓝
HUMAN_PT = (255, 140, 40)
GT_COLOR = (60, 180, 75)        # 绿
GT_PT = (90, 220, 110)
PRED_COLOR = (30, 110, 230)     # 橙红
PRED_PT = (50, 150, 255)
BBX_COLOR = (0, 200, 255)       # 黄
OBS_COLOR = (200, 50, 200)      # 品红: 模型真正看到的带噪 obs
OBS_PT = (255, 80, 255)
OBS_BBX_COLOR = (200, 50, 200)  # 模型实际用的 bbox (可能带抖动增强)

# COCO17 骨架 (同 hmr4d/utils/vis/cv2_utils.draw_coco17_skeleton)
COCO17_BONES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


# ======================================================================
# 复刻 gvhmr_pl._build_synthetic_obs (host 端, 不实例化整个 LightningModule)
# ======================================================================
def build_synthetic_obs(smplx_v437, smpl_params_c, K_fullimg, do_augment, do_bbox_augment):
    """smpl_params_c: dict (B,L,*), K_fullimg: (B,L,3,3).
    返回 obs (B,L,17,3), bbx_xys (B,L,3),
        obs_kp2d (B,L,17,3) 全图坐标带噪 2D 关键点 [u,v,vis] (模型输入的未归一化版),
        noisy_j3d (B,L,17,3) 相机系带噪 3D 关节 (do_augment=0 时即干净 gt_j3d)."""
    with torch.no_grad():
        gt_verts437, gt_j3d = smplx_v437(**smpl_params_c)

    i_x2d = safely_render_x3d_K(gt_verts437, K_fullimg, thr=0.3)
    bbx_xys = get_bbx_xys(i_x2d, do_augment=do_bbox_augment)

    if do_augment:
        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2]).to(gt_j3d)
        noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
        obs_i_j2d = perspective_projection(noisy_j3d, K_fullimg)
        vis = get_visible_mask(gt_j3d.shape[:2]).to(gt_j3d.device)
        vis[noisy_j3d[..., 2] < 0.3] = False
        legs_inv = get_invisible_legs_mask(gt_j3d.shape[:2]).to(gt_j3d.device)
        vis[legs_inv] = False
    else:
        noisy_j3d = gt_j3d
        obs_i_j2d = perspective_projection(gt_j3d, K_fullimg)
        vis = gt_j3d[..., 2] >= 0.3

    obs_kp2d = torch.cat([obs_i_j2d, vis[:, :, :, None].float()], dim=-1)
    obs = normalize_kp2d(obs_kp2d, bbx_xys)
    obs[~vis] = 0
    obs = select_coco17_no_nose_ears(obs)
    return obs, bbx_xys, obs_kp2d, noisy_j3d


# ======================================================================
# B=1 batch 构造 (复刻 collate_fn: tensor unsqueeze, dict 递归, length→tensor)
# ======================================================================
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def batchify(x):
    if torch.is_tensor(x):
        return x.unsqueeze(0)
    if isinstance(x, bool):
        return torch.tensor([x])          # 复刻 collate: scalar mask flag → (B,) 张量
    if isinstance(x, dict):
        return {k: batchify(v) for k, v in x.items()}
    return x  # 其余 python 标量 (int/str, 仅 meta 用) 原样保留


# ======================================================================
# 3D 骨架正交视角投影
# ======================================================================
def ortho_project(pts_w, azim_deg, elev_deg, scale, img_wh, center_w):
    """pts_w: (N,3) y-up world. 返回 (N,2) 像素 + (N,) 全 True 的 in_front.
    视角: 先平移 -center, 再 Ry(azim) 后 Rx(elev); 屏幕 u=右(+x_rot), v=上(+y_rot)."""
    W, H = img_wh
    p = np.asarray(pts_w, dtype=np.float64) - np.asarray(center_w, dtype=np.float64)[None]
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    pr = p @ Ry.T @ Rx.T
    u = W * 0.5 + scale * pr[:, 0]
    v = H * 0.5 - scale * pr[:, 1]
    uv = np.stack([u, v], axis=-1)
    return uv, np.ones(len(uv), dtype=bool)


def draw_floor_grid(img, azim, elev, scale, img_wh, center_w, n=6, step=0.5, color=(200, 200, 200)):
    """在 y=0 平面画网格线 (世界 x-z 方向), 给 3D 视角地面参考."""
    rng = np.arange(-n, n + 1) * step
    lines = []
    for x in rng:
        lines.append(np.array([[x, 0, -n * step], [x, 0, n * step]]))
    for z in rng:
        lines.append(np.array([[-n * step, 0, z], [n * step, 0, z]]))
    for seg in lines:
        uv, _ = ortho_project(seg, azim, elev, scale, img_wh, center_w)
        p1 = tuple(np.round(uv[0]).astype(int))
        p2 = tuple(np.round(uv[1]).astype(int))
        cv2.line(img, p1, p2, color, 1, lineType=cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--exp", default="gvhmr/g1_dualpth")
    p.add_argument("--amass_pth", default="filtered_amass.pth")
    p.add_argument("--g1_pth", default="filtered_g1.pth")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--idx", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--motion_frames", type=int, default=120)
    p.add_argument("--obs_augment", type=int, default=1,
                   help="1=训练态 obs 噪声+bbox 抖动 (do_augment=True); 0=validation 干净 obs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="outputs/infer_viz_dualpth")
    p.add_argument("--azim", type=float, default=35.0, help="3D 视角方位角(度)")
    p.add_argument("--elev", type=float, default=12.0, help="3D 视角俯仰角(度)")
    p.add_argument("--blend", type=int, default=0, help="1=自动 spawn blender 出 .blend")
    p.add_argument("--blender", default="/home/eerrr/.local/bin/blender")
    p.add_argument("--render_blend", type=int, default=0, help="blend 时也渲 mp4")
    args = p.parse_args()

    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据集 (与 train_dualpth.yaml 完全一致的 kwargs) ----
    ds = G1AmassDualPthDataset(
        amass_pth_path=args.amass_pth, g1_pth_path=args.g1_pth,
        g1_input_is_yup=True, amass_fps=30.0, motion_frames=args.motion_frames,
        cam_augmentation="v11", canonicalize_first_frame=True, align_target="pelvis",
        floor_adjust=True, yaw_only_canon=True, apply_az_to_ay_g1=True,
        split=args.split, split_seed=42, split_ratios=(0.9, 0.05, 0.05),
        ratio_lo=(0.7, 0.7, -float("inf")), ratio_hi=(1.05, 1.05, float("inf")),
        ratio_thr=0.1, skip_substrings=(),
    )
    print(f"[data] split={args.split}: {len(ds)} 个片段")

    # ---- 模型 ----
    pipeline = load_pipeline(args.ckpt, exp=args.exp, device=device)

    # ---- SMPLX 模型 ----
    smplx_v437 = make_smplx("supermotion_v437coco17").eval().to(device)  # obs/bbox 用
    smplx_full = make_smplx("supermotion").eval().to(device)             # 人 j22 用

    blend_jobs = []
    for idx in args.idx:
        if idx < 0 or idx >= len(ds.idx2meta):
            print(f"[skip] idx={idx} 越界 (dataset size={len(ds.idx2meta)})")
            continue

        # train 路径不自重 seed → 这里 seed 保证增强可复现
        np.random.seed(args.seed + idx)
        torch.manual_seed(args.seed + idx)

        sample = ds[idx]
        actual_idx = int(sample["meta"]["idx"])
        name = ds.idx2meta[actual_idx]["name"]
        print(f"\n[{idx}] (actual idx={actual_idx}) {name}")

        # B=1 batch (cpu→device), 合成 obs/bbx_xys
        batch = batchify(sample)
        batch = to_device(batch, device)
        batch["length"] = torch.tensor([int(sample["length"])], device=device)
        smpl_c = batch["smpl_params_c"]
        obs, bbx_xys, obs_kp2d, noisy_j3d_c = build_synthetic_obs(
            smplx_v437, smpl_c, batch["K_fullimg"],
            do_augment=bool(args.obs_augment), do_bbox_augment=bool(args.obs_augment),
        )

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": bbx_xys,
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
            "g1_target": batch.get("g1_target", {}),
            "T_w2c": batch["T_w2c"],
        }
        with torch.no_grad():
            outputs = pipeline.forward(batch_, train=False, postproc=False)
        pred_g1_pos_w = outputs["pred_g1_body_pos_w"][0].float().cpu()   # (F,30,3) y-up

        # ---- loss (train=True 路径, 与 validation val-loss 一致, null_condition=False) ----
        batch_loss = {**batch, "obs": obs, "bbx_xys": bbx_xys}
        with torch.no_grad():
            loss_out = pipeline.forward(batch_loss, train=True, null_condition=False)
        weights = getattr(pipeline, "weights", {})

        def _lv(k):
            v = loss_out.get(k, None)
            return float(v.item()) if torch.is_tensor(v) else None

        loss_rows = [  # (显示名, key, weight key)
            ("simple_loss",      "simple_loss",       None),
            ("j3d_loss(cr_j3d)", "cr_j3d_loss",       "cr_j3d"),
            ("j2d_loss",         "j2d_loss",          "j2d"),
            ("transl_w_loss",    "transl_w_loss",     "transl_w"),
            ("transl_c_loss",    "transl_c_loss",     "transl_c"),
            ("static_conf_loss", "static_conf_loss",  "static_conf_bce"),
        ]
        txt_path = out_dir / f"{idx:05d}_{name[:55]}.loss.txt"
        lines = [
            f"name           : {name}",
            f"requested_idx  : {idx}",
            f"actual_idx     : {actual_idx}",
            f"split          : {args.split}",
            f"seed           : {args.seed + idx}",
            f"obs_augment    : {bool(args.obs_augment)} (do_augment for synthetic obs/bbox)",
            f"frames         : {int(sample['length'])}",
            f"ckpt           : {args.ckpt}",
            "",
            f"{'loss':18s} {'value':>14s} {'weight':>8s} {'weighted':>14s}",
            "-" * 58,
        ]
        for disp, k, wk in loss_rows:
            v = _lv(k)
            w = float(weights.get(wk, 1.0)) if wk is not None else 1.0
            if v is None:
                lines.append(f"{disp:18s} {'n/a':>14s} {w:>8.1f} {'n/a':>14s}")
            else:
                lines.append(f"{disp:18s} {v:>14.6f} {w:>8.1f} {v * w:>14.6f}")
        total = _lv("loss")
        lines.append("-" * 58)
        lines.append(f"{'TOTAL loss':18s} {'':>14s} {'':>8s} "
                     f"{(f'{total:.6f}' if total is not None else 'n/a'):>14s}")
        txt_path.write_text("\n".join(lines) + "\n")
        print(f"  -> loss   {txt_path}")
        print("     " + "  ".join(
            f"{d}={_lv(k):.4f}" if _lv(k) is not None else f"{d}=n/a" for d, k, _ in loss_rows))

        # ---- 人 j22 (y-up world) ----
        # "supermotion" 模型按 (F,*) 2D 输入 (F 当 batch), 与 render_dual_camera 一致.
        spw = {k: v.float().to(device) for k, v in sample["smpl_params_w"].items()}
        with torch.no_grad():
            so = smplx_full(
                global_orient=spw["global_orient"],
                body_pose=spw["body_pose"],
                betas=spw["betas"],
                transl=spw["transl"],
            )
        human_j22_w = so.joints[:, :22].float().cpu()                     # (F,22,3) y-up

        # ---- 取相机 / GT (cpu) ----
        T_amass = sample["T_w2c"].float()                                 # (F,4,4)
        T_g1 = sample["g1_target"]["g1_T_w2c"].float()                    # (F,4,4)
        K = sample["K_fullimg"].float()                                   # (F,3,3)
        g1_gt_pos_w = sample["g1_target"]["g1_body_pos_w"].float()        # (F,30,3) y-up
        smpl_c_cpu = {k: v.float() for k, v in sample["smpl_params_c"].items()}

        F_len = T_amass.shape[0]
        W, H = 1000, 1000

        # ---- 模型真正看到的带噪 obs (2D 全图坐标 + vis, 以及反投影回世界系的 3D) ----
        obs_uv = obs_kp2d[0, :, :, :2].float().cpu().numpy()             # (F,17,2)
        obs_vis = obs_kp2d[0, :, :, 2].float().cpu().numpy() > 0.5       # (F,17)
        model_bbx = bbx_xys[0].float().cpu().numpy()                     # (F,3) 模型实际用的 bbox
        noisy_c = noisy_j3d_c[0].float().cpu()                           # (F,17,3) 相机系
        R_w2c, t_w2c = T_amass[:, :3, :3], T_amass[:, :3, 3]
        # p_w = R^T (p_c - t)
        obs_j3d_w = torch.einsum("fji,fnj->fni", R_w2c, noisy_c - t_w2c[:, None])

        # ---- 2D 投影 ----
        h_uv, h_front, _ = project_points(human_j22_w, T_amass, K)
        gt_uv, gt_front, _ = project_points(g1_gt_pos_w, T_g1, K)
        pr_uv, pr_front, _ = project_points(pred_g1_pos_w, T_g1, K)

        # ---- bbox (训练级) ----
        smplx_v437_cpu = make_smplx("supermotion_v437coco17").eval()
        amass_bbx = compute_amass_bbx_xys(smpl_c_cpu, K, smplx_v437_cpu).cpu().numpy()
        gt_bbx = compute_g1_bbx_xys(g1_gt_pos_w, T_g1, K).cpu().numpy()
        pr_bbx = compute_g1_bbx_xys(pred_g1_pos_w, T_g1, K).cpu().numpy()

        # ---- 3D 视角参数 (固定, 所有帧一致) ----
        all_pts = np.concatenate([
            human_j22_w.numpy().reshape(-1, 3),
            g1_gt_pos_w.numpy().reshape(-1, 3),
            pred_g1_pos_w.numpy().reshape(-1, 3),
        ], axis=0)
        center_w = all_pts.mean(axis=0)
        center_w[1] = all_pts[:, 1].min() + 0.9   # 视角中心抬到身体中段
        extent = np.percentile(np.abs(all_pts - center_w[None]), 98) * 2 + 1e-6
        scale3d = 0.42 * min(W, H) / extent

        # ---- 写视频 ----
        out_mp4 = out_dir / f"{idx:05d}_{name[:55]}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        sep = 4
        writer = cv2.VideoWriter(str(out_mp4), fourcc, args.fps, (W * 2 + sep, H * 2 + sep))
        for f in range(F_len):
            # TL 人 2D: GT(蓝) + 模型真正看到的带噪 obs(品红, vis=False 的关节不画)
            tl = np.full((H, W, 3), 245, np.uint8)
            draw_skeleton(tl, h_uv[f], h_front[f], SMPL22_BONES, HUMAN_COLOR, HUMAN_PT, 5, 3)
            draw_skeleton(tl, obs_uv[f], obs_vis[f], COCO17_BONES, OBS_COLOR, OBS_PT, 4, 2)
            draw_bbox_xys(tl, amass_bbx[f], BBX_COLOR, 2)
            draw_bbox_xys(tl, model_bbx[f], OBS_BBX_COLOR, 2)
            label(tl, f"Human (T_w2c)  f{f:>3}/{F_len}  bbx={amass_bbx[f,2]:.0f}", (15, 35))
            label(tl, "blue=GT  magenta=noisy obs (model input)", (15, 65))
            # TR GT 2D
            tr = np.full((H, W, 3), 245, np.uint8)
            draw_skeleton(tr, gt_uv[f], gt_front[f], G1_BONES, GT_COLOR, GT_PT, 5, 3)
            draw_bbox_xys(tr, gt_bbx[f], BBX_COLOR, 2)
            label(tr, f"G1 GT (g1_T_w2c)  f{f:>3}/{F_len}  bbx={gt_bbx[f,2]:.0f}", (15, 35))
            # BL pred 2D
            bl = np.full((H, W, 3), 245, np.uint8)
            draw_skeleton(bl, pr_uv[f], pr_front[f], G1_BONES, PRED_COLOR, PRED_PT, 5, 3)
            draw_bbox_xys(bl, pr_bbx[f], BBX_COLOR, 2)
            label(bl, f"G1 PRED (g1_T_w2c)  f{f:>3}/{F_len}  bbx={pr_bbx[f,2]:.0f}", (15, 35))
            # BR 3D 叠加
            br = np.full((H, W, 3), 245, np.uint8)
            draw_floor_grid(br, args.azim, args.elev, scale3d, (W, H), center_w)
            for pts, bones, col, pt, vis_mask in (
                (human_j22_w[f].numpy(), SMPL22_BONES, HUMAN_COLOR, HUMAN_PT, None),
                (obs_j3d_w[f].numpy(), COCO17_BONES, OBS_COLOR, OBS_PT, obs_vis[f]),
                (g1_gt_pos_w[f].numpy(), G1_BONES, GT_COLOR, GT_PT, None),
                (pred_g1_pos_w[f].numpy(), G1_BONES, PRED_COLOR, PRED_PT, None),
            ):
                uv, front = ortho_project(pts, args.azim, args.elev, scale3d, (W, H), center_w)
                if vis_mask is not None:
                    front = front & vis_mask
                draw_skeleton(br, uv, front, bones, col, pt, 5, 3)
            label(br, f"3D overlay (true world pos)  f{f:>3}/{F_len}", (15, 35))
            label(br, "human=blue  obs=magenta  GT=green  pred=orange", (15, 65))

            top = np.concatenate([tl, np.full((H, sep, 3), 80, np.uint8), tr], axis=1)
            bot = np.concatenate([bl, np.full((H, sep, 3), 80, np.uint8), br], axis=1)
            grid = np.concatenate([top, np.full((sep, W * 2 + sep, 3), 80, np.uint8), bot], axis=0)
            writer.write(grid)
        writer.release()
        print(f"  -> video {out_mp4}")

        # ---- bundle npz (y-up → z-up, 给 blender 关节小球) ----
        human_z = apply_ay_to_az_on_vec(human_j22_w).numpy().astype(np.float32)
        gt_z = apply_ay_to_az_on_vec(g1_gt_pos_w).numpy().astype(np.float32)
        pred_z = apply_ay_to_az_on_vec(pred_g1_pos_w).numpy().astype(np.float32)
        obs_z = apply_ay_to_az_on_vec(obs_j3d_w).numpy().astype(np.float32)
        bundle = out_dir / f"{idx:05d}_{name[:55]}.bundle.npz"
        np.savez(bundle,
                 human_j22_w=human_z, g1_gt_pos_w=gt_z, g1_pred_pos_w=pred_z,
                 human_obs_j17_w=obs_z, human_obs_vis=obs_vis,
                 fps=np.float32(args.fps), name=str(name))
        print(f"  -> bundle {bundle}")
        blend_jobs.append((bundle, out_dir / f"{idx:05d}_{name[:55]}.blend",
                           out_dir / f"{idx:05d}_{name[:55]}.blend.mp4"))

    # ---- 可选: spawn blender 出 .blend ----
    if args.blend:
        for bundle, out_blend, out_blend_mp4 in blend_jobs:
            cmd = [args.blender, "--background", "--python",
                   str(_TOOLS / "blender_skeleton_dualpth.py"), "--",
                   "--bundle", str(bundle.resolve()), "--save", str(out_blend.resolve()),
                   "--fps", str(args.fps)]
            if args.render_blend:
                cmd += ["--render", str(out_blend_mp4.resolve())]
            print(f"\n[blender] {out_blend.name}")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"  [error] rc={ret.returncode}")
                print(ret.stdout[-1500:]); print(ret.stderr[-1000:])
            else:
                for line in ret.stdout.split("\n"):
                    if any(t in line for t in ("[ok]", "[saved]", "[rendered]", "rror")):
                        print("  " + line)

    print(f"\n完成. 输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
