"""
Video/cache -> G1 robot motion inference.

This script follows the original GVHMR demo preprocessing path:
  video -> tracker -> VitPose -> ViT image features -> visual odometry
and feeds the same observations into the trained G1 pipeline, then saves
robot actions in the same format as tools/infer_amass2g1.py.

For BEDLAM2 train-set smoke tests, the same inference path can read the
precomputed training caches instead of running preprocessors:
  bbox/camera + VitPose + HMR2/ViT image features -> G1 pipeline.

Example:
    conda run --no-capture-output -n gvhmr python tools/infer_video2g1.py \
        --video inputs/demo/dance_3.mp4 \
        --ckpt outputs/g1_dualpth_bedlam2/g1_dualpth_bedlam2_v1_4gpu/checkpoints/e499-s258500.ckpt \
        --output outputs/infer_g1_video/dance_3_g1.npz
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = Path("/mnt/ddn/tianyi/apps/blender/blender")
DEFAULT_BASE_BLEND = REPO_ROOT / "vis_smplx.blend"
DEFAULT_G1_BLEND = REPO_ROOT / "assets" / "load_g1_animation_fast.blend"
DEFAULT_GROUND_BLEND = REPO_ROOT / "ground.blend"


def _argv_after_double_dash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Video -> G1 robot motion inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Input video path, using original GVHMR demo preprocess")
    src.add_argument(
        "--bedlam2_idx",
        type=int,
        help="Use a BEDLAM2 train-set sample with precomputed bbox/VitPose/HMR2 features",
    )
    src.add_argument(
        "--bedlam2_key",
        type=str,
        help="Use a specific BEDLAM2 key with precomputed bbox/VitPose/HMR2 features",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/g1_dualpth_bedlam2/g1_dualpth_bedlam2_v1_4gpu/checkpoints/e499-s258500.ckpt",
        help="Trained G1 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output npz path. Default: outputs/infer_g1_video/<video_stem>_g1.npz",
    )
    parser.add_argument("--output_root", type=str, default=None, help="Preprocess cache root, default: outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="Skip visual odometry")
    parser.add_argument("--use_dpvo", action="store_true", help="Use DPVO instead of SimpleVO")
    parser.add_argument("--f_mm", type=int, default=None, help="Full-frame focal length in mm")
    parser.add_argument("--verbose", action="store_true", help="Save preprocessing overlays")
    parser.add_argument("--window", type=int, default=120, help="Sliding window length")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument(
        "--exp",
        type=str,
        default="gvhmr/g1_dualpth_bedlam2",
        help="Training exp yaml name under hmr4d/configs/exp/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch/np/python RNG. Diffusion sampling consumes torch RNG.",
    )
    parser.add_argument(
        "--singular_rewrite",
        type=str,
        default="none",
        choices=["none", "blender"],
        help="joint_pos postprocess. none=physics/reference, blender=Euler rewrite for visualization only",
    )
    parser.add_argument(
        "--save_hmr4d",
        action="store_true",
        help="Also run original GVHMR HMR4D prediction and save hmr4d_results.pt for inspection/rendering.",
    )
    parser.add_argument(
        "--hmr4d_ckpt",
        type=str,
        default=None,
        help="Optional original GVHMR ckpt. Defaults to hmr4d/configs/demo.yaml ckpt_path.",
    )
    parser.add_argument("--bedlam2_search", type=int, default=256, help="Search window for a cached BEDLAM2 sample")
    parser.add_argument("--bedlam2_split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--render_mp4", default=None, help="Optional G1+ground mp4 render path")
    parser.add_argument("--render_limit_frames", type=int, default=240)
    parser.add_argument(
        "--render_camera",
        choices=("auto", "video", "fixed"),
        default="auto",
        help="Blender render camera. auto/video uses saved video camera when available; fixed uses an orbit view.",
    )
    parser.add_argument(
        "--render_camera_trans_scale",
        type=float,
        default=0.9,
        help="Scale video-camera translation for G1 render, matching BEDLAM2 g1_T_w2c construction.",
    )
    parser.add_argument(
        "--camera_basis",
        choices=("model", "loader", "identity", "loader_only"),
        default="identity",
        help="Legacy z-up render camera basis. canonical_yup render space uses identity, matching training.",
    )
    parser.add_argument(
        "--render_space",
        choices=("canonical_yup", "zup"),
        default="canonical_yup",
        help="Blender render coordinate space. canonical_yup matches BEDLAM2 training/debug camera setup.",
    )
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument("--base_blend", default=str(DEFAULT_BASE_BLEND))
    parser.add_argument("--g1_blend", default=str(DEFAULT_G1_BLEND))
    parser.add_argument("--ground_blend", default=str(DEFAULT_GROUND_BLEND))
    parser.add_argument(
        "--ground_rot_x",
        type=float,
        default=0.0,
        help="Rotation in degrees applied to appended ground.blend around Blender X axis.",
    )
    parser.add_argument(
        "--mirror_axis",
        choices=("none", "x", "y", "z"),
        default="y",
        help="Visual-only mirror for Blender render. Default y mirrors body left/right without flipping up.",
    )
    parser.add_argument("--blender_mode", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def build_demo_cfg(args):
    """Reuse original demo.py config construction without duplicating its Hydra wiring."""
    from tools.demo.demo import parse_args_to_cfg as _demo_parse_args_to_cfg

    argv = [
        "tools/demo/demo.py",
        "--video",
        args.video,
    ]
    if args.output_root is not None:
        argv += ["--output_root", args.output_root]
    if args.static_cam:
        argv.append("--static_cam")
    if args.use_dpvo:
        argv.append("--use_dpvo")
    if args.f_mm is not None:
        argv += ["--f_mm", str(args.f_mm)]
    if args.verbose:
        argv.append("--verbose")

    old_argv = sys.argv
    try:
        sys.argv = argv
        cfg = _demo_parse_args_to_cfg()
    finally:
        sys.argv = old_argv
    if args.hmr4d_ckpt is not None:
        cfg.ckpt_path = args.hmr4d_ckpt
    return cfg


def _to_numpy_f32(x):
    import torch

    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _identity_T_w2c(length):
    T = np.repeat(np.eye(4, dtype=np.float32)[None], int(length), axis=0)
    return T


def add_video_camera_data(data, cfg):
    """Store full video-camera matrices for Blender render, without changing model inputs."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from tools.demo.demo import get_video_lwh

    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        T_w2c = _identity_T_w2c(length)
    else:
        traj = torch.load(cfg.paths.slam, map_location="cpu")
        if cfg.use_dpvo:
            traj_quat = torch.as_tensor(traj[:, [6, 3, 4, 5]], dtype=torch.float32)
            R_w2c = quaternion_to_matrix(traj_quat).mT.detach().cpu().numpy().astype(np.float32)
            T_w2c = _identity_T_w2c(length)
            T_w2c[:, :3, :3] = R_w2c
            if traj.shape[-1] >= 3:
                T_w2c[:, :3, 3] = np.asarray(traj[:, :3], dtype=np.float32)
        else:
            traj_np = np.asarray(traj, dtype=np.float32)
            if traj_np.ndim == 3 and traj_np.shape[1:] == (4, 4):
                T_w2c = traj_np.copy()
            elif traj_np.ndim == 3 and traj_np.shape[1:] == (3, 4):
                T_w2c = _identity_T_w2c(length)
                T_w2c[:, :3, :] = traj_np
            else:
                T_w2c = _identity_T_w2c(length)
                T_w2c[:, :3, :3] = traj_np[:, :3, :3]

    data["T_w2c"] = T_w2c[:length].astype(np.float32)
    data["image_wh"] = np.asarray([width, height], dtype=np.int32)


def make_prep_from_video_data(data, window, device):
    """Convert original GVHMR demo data dict into G1 pipeline inference tensors."""
    import torch

    from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    length = int(data["length"].item() if torch.is_tensor(data["length"]) else data["length"])

    obs = select_coco17_no_nose_ears(normalize_kp2d(data["kp2d"], data["bbx_xys"]))
    prep = {
        # Keep full-video tensors on CPU. Long videos can have large 1024-D ViT
        # features, so each sliding window is moved to GPU only when needed.
        "obs_full": obs.to(dtype=torch.float32).cpu(),
        "bbx_xys_full": data["bbx_xys"].to(dtype=torch.float32).cpu(),
        "K_fullimg_full": data["K_fullimg"].to(dtype=torch.float32).cpu(),
        "cam_angvel_full": data["cam_angvel"].to(dtype=torch.float32).cpu(),
        "f_imgseq_full": data["f_imgseq"].to(dtype=torch.float32).cpu(),
        "window": int(window),
        "total_len": length,
    }

    if length <= window:
        chunks_meta = [(0, length)]
    else:
        starts = [0]
        while starts[-1] + window < length:
            starts.append(starts[-1] + window - 1)
        chunks_meta = [(s, min(window, length - s)) for s in starts]
    prep["chunks_meta"] = chunks_meta
    print(f"[Info] Video length={length}, {len(chunks_meta)} chunks (window={window}, overlap=1)")
    return prep, length


def load_bedlam2_cached_data(args):
    import torch

    from hmr4d.dataset.bedlam.bedlam import Bedlam2G1Dataset
    from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    ds = Bedlam2G1Dataset(split=args.bedlam2_split, split_ratios=(0.95, 0.05, 0.0))
    if args.bedlam2_key is not None:
        matches = [i for i, m in enumerate(ds.idx2meta) if m.get("key") == args.bedlam2_key]
        if not matches:
            raise KeyError(f"BEDLAM2 key not found in {args.bedlam2_split} split: {args.bedlam2_key}")
        start_idx = matches[0]
    else:
        start_idx = int(args.bedlam2_idx)

    for off in range(int(args.bedlam2_search)):
        idx = (start_idx + off) % len(ds)
        sample = ds[idx]
        if bool(sample["mask"]["vitpose"]) and bool(sample["mask"]["f_imgseq"]):
            break
    else:
        raise RuntimeError(
            f"No BEDLAM2 sample with both VitPose and HMR2/img features in {args.bedlam2_search} tries"
        )

    length = int(sample["length"])
    meta = sample["meta"]
    print(f"[BEDLAM2] idx={idx} length={length} key={meta.get('key')}")
    print(f"[BEDLAM2] cached vitpose={sample['mask']['vitpose']} hmr2/imgfeat={sample['mask']['f_imgseq']}")
    print(f"[BEDLAM2] source mp4={meta.get('mp4_path')}")

    obs = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"]))
    obs[~sample["mask"]["valid"]] = 0

    data = {
        "length": torch.tensor(length),
        "obs": obs,
        "bbx_xys": sample["bbx_xys"],
        "K_fullimg": sample["K_fullimg"],
        "T_w2c": sample.get("T_w2c", None),
        "cam_angvel": sample["cam_angvel"],
        "f_imgseq": sample["f_imgseq"],
        "source_name": f"bedlam2_idx{idx}",
        "source_video": str(meta.get("mp4_path", "")),
        "preprocess_dir": "",
        "fps": float(sample.get("fps", 30.0)),
        "image_wh": _to_numpy_f32(sample.get("img_wh", None)),
        "bedlam2_meta": {
            "idx": idx,
            "key": str(meta.get("key", "")),
            "mp4_path": str(meta.get("mp4_path", "")),
            "has_vitpose": bool(sample["mask"]["vitpose"]),
            "has_imgfeat": bool(sample["mask"]["f_imgseq"]),
        },
    }
    return data


def make_prep_from_obs_data(data, window, device):
    import torch

    length = int(data["length"].item() if torch.is_tensor(data["length"]) else data["length"])
    prep = {
        "obs_full": data["obs"].to(dtype=torch.float32).cpu(),
        "bbx_xys_full": data["bbx_xys"].to(dtype=torch.float32).cpu(),
        "K_fullimg_full": data["K_fullimg"].to(dtype=torch.float32).cpu(),
        "cam_angvel_full": data["cam_angvel"].to(dtype=torch.float32).cpu(),
        "f_imgseq_full": data["f_imgseq"].to(dtype=torch.float32).cpu(),
        "window": int(window),
        "total_len": length,
    }

    if length <= window:
        chunks_meta = [(0, length)]
    else:
        starts = [0]
        while starts[-1] + window < length:
            starts.append(starts[-1] + window - 1)
        chunks_meta = [(s, min(window, length - s)) for s in starts]
    prep["chunks_meta"] = chunks_meta
    print(f"[Info] sequence length={length}, {len(chunks_meta)} chunks (window={window}, overlap=1)")
    return prep, length


def infer_video(pipeline, prep, total_len, device="cuda", singular_rewrite="none"):
    import torch
    import torch.nn.functional as F
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    from tools.infer_amass2g1 import reduce_singular_chains

    obs_full = prep["obs_full"]
    bbx_xys_full = prep["bbx_xys_full"]
    K_fullimg_full = prep["K_fullimg_full"]
    cam_angvel_full = prep["cam_angvel_full"]
    f_imgseq_full = prep["f_imgseq_full"]
    chunks_meta = prep["chunks_meta"]
    window = prep["window"]

    joint_pos_out = torch.zeros(total_len, 29, device=device)
    root_pos_out = torch.zeros(total_len, 3, device=device)
    root_quat_out = torch.zeros(total_len, 4, device=device)
    body_pos_out = None

    def pad_window(x, pad_len):
        if pad_len <= 0:
            return x
        return torch.cat([x, x[-1:].expand(pad_len, *x.shape[1:])], dim=0)

    prev_last_pos = None
    prev_last_quat = None

    for chunk_idx, (start, actual_len) in enumerate(chunks_meta):
        end = start + actual_len
        pad_len = window - actual_len
        L = window

        obs = pad_window(obs_full[start:end], pad_len)
        bbx_xys = pad_window(bbx_xys_full[start:end], pad_len)
        K_fullimg = pad_window(K_fullimg_full[start:end], pad_len)
        cam_angvel = pad_window(cam_angvel_full[start:end], pad_len)
        f_imgseq = pad_window(f_imgseq_full[start:end], pad_len)

        obs = obs.to(device)
        bbx_xys = bbx_xys.to(device)
        K_fullimg = K_fullimg.to(device)
        cam_angvel = cam_angvel.to(device)
        f_imgseq = f_imgseq.to(device)

        batch = {
            "length": torch.tensor([actual_len], device=device, dtype=torch.long),
            "obs": obs.unsqueeze(0),
            "bbx_xys": bbx_xys.unsqueeze(0),
            "K_fullimg": K_fullimg.unsqueeze(0),
            "cam_angvel": cam_angvel.unsqueeze(0),
            "f_imgseq": f_imgseq.unsqueeze(0),
        }

        if prev_last_pos is not None:
            g1_body_pos_w = torch.zeros(1, 1, 1, 3, device=device, dtype=prev_last_pos.dtype)
            g1_body_pos_w[0, 0, 0] = prev_last_pos
            batch["g1_target"] = {"g1_body_pos_w": g1_body_pos_w}

        outputs = pipeline.forward(batch, train=False)
        pred_jp = outputs["pred_g1_joint_pos"][0, :actual_len]
        pred_rp = outputs["pred_root_pos"][0, :actual_len]
        pred_rq = outputs["pred_root_quat_w"][0, :actual_len]
        pred_body = outputs.get("pred_g1_body_pos_w")
        if pred_body is not None:
            pred_body = pred_body[0, :actual_len]
            if body_pos_out is None:
                body_pos_out = torch.zeros(total_len, pred_body.shape[1], 3, device=device)

        if prev_last_quat is not None:
            R_prev_last = quaternion_to_matrix(prev_last_quat)
            R_curr_first = quaternion_to_matrix(pred_rq[0])
            delta_R = R_prev_last @ R_curr_first.transpose(-1, -2)

            disp = pred_rp - pred_rp[0:1]
            pred_rp = pred_rp[0:1] + disp @ delta_R.transpose(-1, -2)

            R_curr_all = quaternion_to_matrix(pred_rq)
            pred_rq = matrix_to_quaternion(delta_R.unsqueeze(0) @ R_curr_all)

        if chunk_idx == 0:
            write_slice = slice(start, end)
            src_slice = slice(0, actual_len)
        else:
            write_slice = slice(start + 1, end)
            src_slice = slice(1, actual_len)
        joint_pos_out[write_slice] = pred_jp[src_slice]
        root_pos_out[write_slice] = pred_rp[src_slice]
        root_quat_out[write_slice] = pred_rq[src_slice]
        if body_pos_out is not None and pred_body is not None:
            body_pos_out[write_slice] = pred_body[src_slice]

        prev_last_pos = pred_rp[actual_len - 1].detach()
        prev_last_quat = pred_rq[actual_len - 1].detach()

    # Match infer_amass2g1.py output convention: convert internal y-up frame to G1 z-up URDF world.
    M_pos = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        device=device,
        dtype=root_pos_out.dtype,
    )
    root_pos_out = root_pos_out @ M_pos
    body_pos_z = body_pos_out @ M_pos if body_pos_out is not None else None
    R_yup = quaternion_to_matrix(root_quat_out)
    root_quat_out = matrix_to_quaternion(M_pos.transpose(-1, -2) @ R_yup)
    root_quat_out = F.normalize(root_quat_out, dim=-1)

    assert singular_rewrite in ("none", "blender"), singular_rewrite
    joint_pos_np = joint_pos_out.cpu().numpy()
    if singular_rewrite == "blender":
        joint_pos_np = reduce_singular_chains(joint_pos_np)

    results = {
        "joint_pos": joint_pos_np,
        "root_pos_w": root_pos_out.cpu().numpy(),
        "root_quat_w": root_quat_out.cpu().numpy(),
    }
    if body_pos_z is not None:
        results["body_pos_w"] = body_pos_z.cpu().numpy()
    return results


def maybe_save_hmr4d_results(cfg, data):
    if Path(cfg.paths.hmr4d_results).exists():
        print(f"[Info] HMR4D results already exist: {cfg.paths.hmr4d_results}")
        return
    import hydra

    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu

    print("[Info] Running original GVHMR HMR4D prediction for inspection")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    pred = model.predict(data, static_cam=cfg.static_cam)
    torch.save(detach_to_cpu(pred), cfg.paths.hmr4d_results)
    print(f"[Info] Saved HMR4D results: {cfg.paths.hmr4d_results}")


def render_g1_with_ground(npz_path, render_path, args):
    render_path = Path(render_path)
    save_blend = render_path.with_suffix(".blend")
    render_path.parent.mkdir(parents=True, exist_ok=True)
    save_blend.parent.mkdir(parents=True, exist_ok=True)

    blender = Path(args.blender)
    base_blend = Path(args.base_blend)
    if not blender.exists():
        raise FileNotFoundError(f"Blender not found: {blender}")
    if not base_blend.exists():
        raise FileNotFoundError(f"base blend not found: {base_blend}")

    cmd = [
        str(blender),
        str(base_blend),
        "--background",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--blender_mode",
        "--npz",
        str(npz_path),
        "--render",
        str(render_path),
        "--save_blend",
        str(save_blend),
        "--g1_blend",
        str(args.g1_blend),
        "--ground_blend",
        str(args.ground_blend),
        "--ground_rot_x",
        str(args.ground_rot_x),
        "--limit_frames",
        str(args.render_limit_frames),
        "--render_camera",
        str(args.render_camera),
        "--camera_trans_scale",
        str(args.render_camera_trans_scale),
        "--camera_basis",
        str(args.camera_basis),
        "--render_space",
        str(args.render_space),
        "--mirror_axis",
        str(args.mirror_axis),
    ]
    print("[Blender]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _append_blend_objects(blend_path):
    import bpy

    before = set(bpy.data.objects.keys())
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.objects = [n for n in data_from.objects]
        data_to.collections = []
    new_objs = []
    coll = bpy.context.scene.collection
    for obj in data_to.objects:
        if obj is None:
            continue
        if obj.name not in coll.objects:
            coll.objects.link(obj)
        new_objs.append(obj)
    for name in set(bpy.data.objects.keys()) - before:
        obj = bpy.data.objects[name]
        if obj not in new_objs:
            new_objs.append(obj)
            if obj.name not in coll.objects:
                coll.objects.link(obj)
    return new_objs


def _find_g1_armature(objs):
    for obj in objs:
        if obj.type == "ARMATURE" and obj.name.lower().startswith("g1"):
            return obj
    for obj in objs:
        if obj.type == "ARMATURE":
            return obj
    return None


def _objects_bound_to_armature(armature, objs):
    bound = {armature}
    for obj in objs:
        if obj is armature:
            continue
        if obj.type == "MESH":
            bound.add(obj)
            for mod in obj.modifiers:
                if mod.type == "ARMATURE":
                    mod.object = armature
        elif obj.parent is armature:
            bound.add(obj)
    return bound


def _object_world_points(objs):
    import bpy

    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for obj in objs:
        if obj.type != "MESH":
            continue
        ev = obj.evaluated_get(deps)
        me = ev.to_mesh()
        mw = obj.matrix_world
        for v in me.vertices:
            w = mw @ v.co
            pts.append((w.x, w.y, w.z))
        ev.to_mesh_clear()
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _rotmat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = np.zeros(R.shape[:-2] + (4,), dtype=np.float64)
    good = trace > 0.0
    if np.any(good):
        s = np.sqrt(trace[good] + 1.0) * 2.0
        q[good, 0] = 0.25 * s
        q[good, 1] = (R[good, 2, 1] - R[good, 1, 2]) / s
        q[good, 2] = (R[good, 0, 2] - R[good, 2, 0]) / s
        q[good, 3] = (R[good, 1, 0] - R[good, 0, 1]) / s
    for idx in zip(*np.where(~good)):
        m = R[idx]
        diag = np.array([m[0, 0], m[1, 1], m[2, 2]])
        i = int(np.argmax(diag))
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = np.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2.0
        qi = np.zeros(4, dtype=np.float64)
        qi[0] = (m[k, j] - m[j, k]) / s
        qi[1 + i] = 0.25 * s
        qi[1 + j] = (m[j, i] + m[i, j]) / s
        qi[1 + k] = (m[k, i] + m[i, k]) / s
        q[idx] = qi
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def _quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _zup_npz_to_canonical_yup(root_pos_z, root_quat_z, body_pos_z=None):
    # infer_video saved p_z = p_y @ M and R_z = M.T @ R_y.
    # Invert this for the training/debug canonical y-up render path.
    M = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    root_pos_y = np.asarray(root_pos_z, dtype=np.float64) @ M.T
    R_z = _quat_wxyz_to_rotmat(root_quat_z)
    R_y = M @ R_z
    root_quat_y = _rotmat_to_quat_wxyz(R_y)
    body_pos_y = None if body_pos_z is None else np.asarray(body_pos_z, dtype=np.float64) @ M.T
    return root_pos_y, root_quat_y, body_pos_y


def _canonical_yup_to_loader_inputs(root_pos_y, root_quat_y):
    # blend.load_g1_animation_fast maps loader input points as
    #   p_blender = [x, z, -y] = p_loader @ L.
    # To make the final rendered G1 live in canonical y-up coordinates,
    # feed p_loader = p_yup @ L.T and the corresponding basis-converted root
    # orientation. This mirrors the training camera world instead of trying to
    # rotate the camera around a z-up render after the fact.
    L = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    S = L.T
    root_pos_loader = np.asarray(root_pos_y, dtype=np.float64) @ L.T
    R_y = _quat_wxyz_to_rotmat(root_quat_y)
    R_loader = S.T @ R_y @ S
    root_quat_loader = _rotmat_to_quat_wxyz(R_loader)
    return root_pos_loader, root_quat_loader


def _set_visible(objs, visible):
    for obj in objs:
        obj.hide_viewport = not visible
        obj.hide_render = not visible


def _mirror_objects(objs, axis, pivot):
    import bpy
    import mathutils

    if axis == "none":
        return None
    scale_by_axis = {
        "x": (-1.0, 1.0, 1.0),
        "y": (1.0, -1.0, 1.0),
        "z": (1.0, 1.0, -1.0),
    }
    mirror_root = bpy.data.objects.new(f"video2g1_mirror_{axis}", None)
    bpy.context.scene.collection.objects.link(mirror_root)
    mirror_root.location = mathutils.Vector(tuple(pivot))
    mirror_root.scale = scale_by_axis[axis]
    for obj in list(objs):
        if obj.parent is None:
            obj.parent = mirror_root
            obj.matrix_parent_inverse = mirror_root.matrix_world.inverted()
    return mirror_root


def _setup_camera_from_K(cam_data, K, width, height):
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = float(width)
    cam_data.lens = fx
    cam_data.shift_x = -(cx - width * 0.5) / max(float(width), 1.0)
    cam_data.shift_y = (cy - height * 0.5) / max(float(width), 1.0)
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0


def _camera_matrix_world_from_cv(T_w2c, world_to_cv_basis):
    import mathutils

    T_cv = np.asarray(T_w2c, dtype=np.float64)
    R_cv = T_cv[:3, :3]
    t_cv = T_cv[:3, 3]
    cv_to_blender_cam = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    basis = np.asarray(world_to_cv_basis, dtype=np.float64)

    R_w2cam_bl = cv_to_blender_cam @ R_cv @ basis
    t_w2cam_bl = cv_to_blender_cam @ t_cv
    R_cam2w = R_w2cam_bl.T
    loc = -R_cam2w @ t_w2cam_bl
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_cam2w
    mat[:3, 3] = loc
    return mathutils.Matrix(mat.tolist())


def _camera_basis_matrix(mode):
    yrot_to_saved_zup = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    saved_zup_to_loader_blender = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    if mode == "model":
        # Use the model/train world basis. This maps Blender z=0 ground to
        # GVHMR y=0 ground, so the rendered floor stays horizontal.
        return yrot_to_saved_zup
    if mode == "loader":
        return yrot_to_saved_zup @ saved_zup_to_loader_blender
    if mode == "identity":
        return np.eye(3, dtype=np.float64)
    if mode == "loader_only":
        return saved_zup_to_loader_blender
    raise ValueError(f"unknown camera_basis: {mode}")


def _setup_video_camera(scene, nz, T, camera_trans_scale, camera_basis="identity"):
    import bpy

    if "T_w2c" not in nz.files or "K_fullimg" not in nz.files:
        return None

    T_w2c = np.asarray(nz["T_w2c"], dtype=np.float64)[:T].copy()
    K = np.asarray(nz["K_fullimg"], dtype=np.float64)
    if K.ndim == 2:
        K = np.repeat(K[None], T, axis=0)
    K = K[:T]
    if T_w2c.shape[0] < T or K.shape[0] < T:
        return None

    if "image_wh" in nz.files:
        wh = np.asarray(nz["image_wh"]).reshape(-1)
        width = max(2, int(round(float(wh[0]))))
        height = max(2, int(round(float(wh[1]))))
    else:
        width = max(2, int(round(float(K[0, 0, 2]) * 2.0)))
        height = max(2, int(round(float(K[0, 1, 2]) * 2.0)))
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    T_w2c[:, :3, 3] *= float(camera_trans_scale)
    world_to_cv_basis = _camera_basis_matrix(camera_basis)

    cam_data = bpy.data.cameras.new("video_camera_g1")
    cam = bpy.data.objects.new("video_camera_g1", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    for f in range(T):
        scene.frame_set(f + 1)
        _setup_camera_from_K(cam.data, K[f], width, height)
        cam.matrix_world = _camera_matrix_world_from_cv(T_w2c[f], world_to_cv_basis)
        cam.keyframe_insert(data_path="location", frame=f + 1)
        cam.keyframe_insert(data_path="rotation_euler", frame=f + 1)
        cam.data.keyframe_insert(data_path="lens", frame=f + 1)
        cam.data.keyframe_insert(data_path="shift_x", frame=f + 1)
        cam.data.keyframe_insert(data_path="shift_y", frame=f + 1)
    return cam


def _setup_fixed_camera(scene, center, radius):
    import bpy
    import mathutils

    az = np.radians(45.0)
    elev = np.radians(16.0)
    dist = max(5.0, radius * 1.8)
    cam_loc = center + np.array([
        dist * np.cos(elev) * np.cos(az),
        -dist * np.sin(elev),
        dist * np.cos(elev) * np.sin(az),
    ])
    bpy.ops.object.camera_add(location=tuple(cam_loc))
    cam = bpy.context.object
    direction = mathutils.Vector(tuple(center)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(radius * 2.1, 2.2)
    scene.camera = cam
    return cam


def blender_main(argv):
    import bpy
    import mathutils

    p = argparse.ArgumentParser()
    p.add_argument("--blender_mode", action="store_true")
    p.add_argument("--npz", required=True)
    p.add_argument("--render", required=True)
    p.add_argument("--save_blend", required=True)
    p.add_argument("--g1_blend", required=True)
    p.add_argument("--ground_blend", default=str(DEFAULT_GROUND_BLEND))
    p.add_argument("--ground_rot_x", type=float, default=0.0)
    p.add_argument("--limit_frames", type=int, default=240)
    p.add_argument("--render_camera", choices=("auto", "video", "fixed"), default="auto")
    p.add_argument("--camera_trans_scale", type=float, default=0.9)
    p.add_argument("--camera_basis", choices=("model", "loader", "identity", "loader_only"), default="model")
    p.add_argument("--render_space", choices=("canonical_yup", "zup"), default="canonical_yup")
    p.add_argument("--mirror_axis", choices=("none", "x", "y", "z"), default="y")
    args = p.parse_args(argv)

    nz = np.load(args.npz, allow_pickle=True)
    root_pos = np.asarray(nz["root_pos_w"], dtype=np.float64)
    root_quat_wxyz = np.asarray(nz["root_quat_w"], dtype=np.float64)
    dof_byd = np.asarray(nz["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(nz["body_pos_w"], dtype=np.float64) if "body_pos_w" in nz.files else None
    fps = int(round(float(nz["fps"]))) if "fps" in nz.files else 30
    T = int(min(root_pos.shape[0], dof_byd.shape[0], max(1, args.limit_frames)))
    root_pos = root_pos[:T]
    root_quat_wxyz = root_quat_wxyz[:T]
    dof_byd = dof_byd[:T]
    if body_pos is not None:
        body_pos = body_pos[:T]

    if args.render_space == "canonical_yup":
        root_pos_render, root_quat_render, body_pos_render = _zup_npz_to_canonical_yup(
            root_pos, root_quat_wxyz, body_pos
        )
        root_pos_loader, root_quat_loader = _canonical_yup_to_loader_inputs(root_pos_render, root_quat_render)
        camera_basis = "identity"
        default_ground_rot_x = 90.0
        up_axis = 1
    else:
        root_pos_render, root_quat_render, body_pos_render = root_pos, root_quat_wxyz, body_pos
        root_pos_loader, root_quat_loader = root_pos_render, root_quat_render
        camera_basis = args.camera_basis
        default_ground_rot_x = 0.0
        up_axis = 2
    ground_rot_x = float(args.ground_rot_x)
    if abs(ground_rot_x) < 1e-12 and args.render_space == "canonical_yup":
        ground_rot_x = default_ground_rot_x
    body_pts = body_pos_render.reshape(-1, 3) if body_pos_render is not None else root_pos_render

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = T
    scene.frame_step = 1
    scene.frame_set(1)
    scene.render.fps = fps
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.025, 0.025, 0.03)

    sys.path.insert(0, str(REPO_ROOT))
    import types

    sys.modules.setdefault("joblib", types.ModuleType("joblib"))
    if "scipy" not in sys.modules:
        scipy_mod = types.ModuleType("scipy")
        spatial_mod = types.ModuleType("scipy.spatial")
        transform_mod = types.ModuleType("scipy.spatial.transform")

        class _UnusedRotation:
            pass

        transform_mod.Rotation = _UnusedRotation
        spatial_mod.transform = transform_mod
        scipy_mod.spatial = spatial_mod
        sys.modules["scipy"] = scipy_mod
        sys.modules["scipy.spatial"] = spatial_mod
        sys.modules["scipy.spatial.transform"] = transform_mod
    from blend import byd_joint_to_mujoco_joint, load_g1_animation_fast

    g1_objs = _append_blend_objects(Path(args.g1_blend))
    g1_arm = _find_g1_armature(g1_objs)
    if g1_arm is None:
        raise SystemExit(f"[error] no G1 armature in {args.g1_blend}")
    bound_objs = _objects_bound_to_armature(g1_arm, g1_objs)

    load_g1_animation_fast(
        g1_arm,
        root_pos=root_pos_loader,
        root_rot=_wxyz_to_xyzw(root_quat_loader),
        dof_pos=dof_byd[:, byd_joint_to_mujoco_joint],
    )

    lo = body_pts.min(0)
    hi = body_pts.max(0)
    center = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo)) * 0.65, 1.5)
    floor_level = float(min(lo[up_axis], 0.0) - 0.015)

    ground_blend = Path(args.ground_blend)
    if not ground_blend.exists():
        raise SystemExit(f"[error] ground blend not found: {ground_blend}")
    ground_objs = {
        o
        for o in _append_blend_objects(ground_blend)
        if o.type in ("MESH", "CURVE", "SURFACE", "FONT", "EMPTY")
    }
    if not ground_objs:
        raise SystemExit(f"[error] no visible ground objects in {ground_blend}")

    ground_root = bpy.data.objects.new("ground_root", None)
    bpy.context.scene.collection.objects.link(ground_root)
    ground_root.rotation_euler = (np.radians(ground_rot_x), 0.0, 0.0)
    for obj in ground_objs:
        if obj.parent is None:
            obj.parent = ground_root
            obj.matrix_parent_inverse = ground_root.matrix_world.inverted()
    bpy.context.view_layer.update()
    pts = _object_world_points(ground_objs)
    g_center = (pts.min(0) + pts.max(0)) * 0.5 if pts.size else np.zeros(3)
    g_min_up = float(pts[:, up_axis].min()) if pts.size else 0.0
    loc = center - g_center
    loc[up_axis] = floor_level - g_min_up
    ground_root.location += mathutils.Vector(tuple(float(x) for x in loc))
    ground_objs.add(ground_root)

    mirror_root = _mirror_objects(bound_objs, args.mirror_axis, center)
    if mirror_root is not None:
        bound_objs.add(mirror_root)

    cam = None
    if args.render_camera in ("auto", "video"):
        cam = _setup_video_camera(scene, nz, T, args.camera_trans_scale, camera_basis)
        if cam is None and args.render_camera == "video":
            raise SystemExit("[error] --render_camera video requested, but npz has no complete T_w2c/K_fullimg/image_wh")
    if cam is None:
        cam = _setup_fixed_camera(scene, center, radius)
    print(
        f"[camera] {cam.name} mode={'video' if cam.name.startswith('video_camera') else 'fixed'} "
        f"space={args.render_space} basis={camera_basis} ground_rot_x={ground_rot_x}"
    )

    bpy.ops.object.light_add(type="SUN", location=(0, -3, 5))
    bpy.context.object.data.energy = 2.0
    bpy.ops.object.light_add(type="AREA", location=tuple(center + np.array([0.0, -2.0, 3.0])))
    area = bpy.context.object
    area.data.energy = 450.0
    area.data.size = 5.0

    visible = list(bound_objs | ground_objs)
    _set_visible([o for o in bpy.data.objects if o.type in ("MESH", "ARMATURE", "EMPTY")], False)
    _set_visible(visible, True)

    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend)))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(Path(args.render))
    bpy.ops.render.render(animation=True)


def main():
    if "--blender_mode" in sys.argv:
        blender_main(_argv_after_double_dash())
        return

    import torch
    from tools.demo.demo import load_data_dict, run_preprocess
    from tools.infer_amass2g1 import load_pipeline

    args = parse_args()
    _set_seed(args.seed)

    if args.video is not None:
        source_name = Path(args.video).stem
    elif args.bedlam2_key is not None:
        source_name = "bedlam2_" + Path(args.bedlam2_key).stem
    else:
        source_name = f"bedlam2_idx{args.bedlam2_idx}"

    if args.output is None:
        out_dir = Path("outputs/infer_g1_video")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{source_name}_g1.npz")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        cfg = build_demo_cfg(args)
        run_preprocess(cfg)
        data = load_data_dict(cfg)
        add_video_camera_data(data, cfg)
        data["source_name"] = Path(args.video).stem
        data["source_video"] = str(args.video)
        data["preprocess_dir"] = str(cfg.preprocess_dir)
        data["fps"] = 30.0

        if args.save_hmr4d:
            maybe_save_hmr4d_results(cfg, data)

        prep, total_len = make_prep_from_video_data(data, window=args.window, device=args.device)
    else:
        data = load_bedlam2_cached_data(args)
        prep, total_len = make_prep_from_obs_data(data, window=args.window, device=args.device)

    pipeline = load_pipeline(args.ckpt, exp=args.exp, device=args.device)
    with torch.no_grad():
        results = infer_video(
            pipeline,
            prep,
            total_len,
            device=args.device,
            singular_rewrite=args.singular_rewrite,
        )

    np.savez(
        args.output,
        joint_pos=results["joint_pos"],
        root_pos_w=results["root_pos_w"],
        root_quat_w=results["root_quat_w"],
        **({"body_pos_w": results["body_pos_w"]} if "body_pos_w" in results else {}),
        K_fullimg=_to_numpy_f32(data.get("K_fullimg", None)),
        **({"T_w2c": _to_numpy_f32(data["T_w2c"])} if data.get("T_w2c", None) is not None else {}),
        **({"image_wh": np.asarray(data["image_wh"], dtype=np.int32)} if data.get("image_wh", None) is not None else {}),
        source_video=str(data.get("source_video", "")),
        preprocess_dir=str(data.get("preprocess_dir", "")),
        ckpt=str(args.ckpt),
        exp=str(args.exp),
        fps=np.array(float(data.get("fps", 30.0)), dtype=np.float32),
        source_name=str(data.get("source_name", source_name)),
    )
    if "bedlam2_meta" in data:
        Path(args.output).with_suffix(".json").write_text(
            json.dumps(data["bedlam2_meta"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[Done] Saved: {args.output}")
    print(f"  - joint_pos:   {results['joint_pos'].shape} (BYD scalar dofs)")
    print(f"  - root_pos_w:  {results['root_pos_w'].shape} (z-up world)")
    print(f"  - root_quat_w: {results['root_quat_w'].shape} (wxyz, z-up world)")
    if args.render_mp4 is not None:
        render_g1_with_ground(args.output, args.render_mp4, args)
        print(f"[Done] Rendered: {args.render_mp4}")


if __name__ == "__main__":
    main()
