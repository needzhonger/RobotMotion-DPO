"""BEDLAM cached observations -> G1 robot motion inference.

This mirrors the BEDLAM2 cached/video inference path, but points the configurable
dataset stores at filtered BEDLAM by default. It does not run YOLO, ViTPose, or
HMR2; bbox/camera, ViTPose keypoints, and HMR2 image features are loaded from
precomputed pth files.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_DIR = "/mnt/ddn/tianyi/PHUMA/data/filtered_bedlam"
DEFAULT_BBOX_PTH = "/mnt/ddn/tianyi/PHUMA/data/filtered_bedlam_bbox/filtered_bedlam_bboxes.pth"
DEFAULT_VITPOSE_PTH = "/mnt/ddn/tianyi/PHUMA/data/filtered_bedlam_vitpose/filtered_bedlam_vitpose.pth"
DEFAULT_IMGFEAT_PTH = "/mnt/ddn/tianyi/PHUMA/data/filtered_bedlam_imgfeat/filtered_bedlam_imgfeat.pth"
DEFAULT_CKPT = "outputs/g1_dualpth_bedlam2/g1_dualpth_bedlam2_v1_4gpu/checkpoints/e499-s258500.ckpt"


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy_f32(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Precomputed BEDLAM cache -> G1 robot inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--idx", type=int, help="Index into sorted common BEDLAM keys")
    src.add_argument("--key", type=str, help="Exact BEDLAM key")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--bbox_pth_path", default=DEFAULT_BBOX_PTH)
    p.add_argument("--vitpose_pth_path", default=DEFAULT_VITPOSE_PTH)
    p.add_argument("--imgfeat_pth_path", default=DEFAULT_IMGFEAT_PTH)
    p.add_argument("--smplx_pth_name", default="filtered_bedlam_smplx.pth")
    p.add_argument("--g1_pth_name", default="filtered_bedlam_g1.pth")
    p.add_argument("--output", required=True)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--exp", default="gvhmr/g1_dualpth_bedlam2")
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--singular_rewrite", choices=("none", "blender"), default="none")
    p.add_argument("--dof_pos_order", choices=("mjc", "byd"), default="mjc")
    p.add_argument("--g1_input_is_yup", action="store_true")
    p.add_argument("--projection_name", default="bedlam")
    return p.parse_args()


def choose_key(args):
    data_dir = Path(args.data_dir)
    smplx = _torch_load_cpu(data_dir / args.smplx_pth_name)
    g1 = _torch_load_cpu(data_dir / args.g1_pth_name)
    bbox = _torch_load_cpu(args.bbox_pth_path)
    common = sorted(set(smplx) & set(g1) & set(bbox))
    if not common:
        raise RuntimeError("no common BEDLAM keys in smplx/g1/bbox stores")
    if args.key is not None:
        if args.key not in common:
            raise KeyError(f"BEDLAM key not found in common stores: {args.key}")
        return args.key, common.index(args.key), len(common), bbox[args.key]
    idx = int(args.idx) % len(common)
    return common[idx], idx, len(common), bbox[common[idx]]


def load_full_sequence_sample(args, key):
    from hmr4d.dataset.bedlam.bedlam import Bedlam2G1Dataset

    ds = Bedlam2G1Dataset(
        data_dir=args.data_dir,
        bbox_pth_path=args.bbox_pth_path,
        vitpose_pth_path=args.vitpose_pth_path,
        imgfeat_pth_path=args.imgfeat_pth_path,
        smplx_pth_name=args.smplx_pth_name,
        g1_pth_name=args.g1_pth_name,
        motion_frames=args.window,
        min_motion_frames=1,
        dof_pos_order=args.dof_pos_order,
        g1_input_is_yup=args.g1_input_is_yup,
        split="val",
        split_ratios=(0.0, 1.0, 0.0),
        full_sequence=True,
    )
    matches = [i for i, m in enumerate(ds.idx2meta) if m.get("key") == key]
    if not matches:
        raise KeyError(f"BEDLAM key not found in full-sequence dataset: {key}")
    sample = ds[matches[0]]
    if not bool(sample["mask"]["vitpose"]):
        raise RuntimeError(f"BEDLAM key has no cached ViTPose: {key}")
    if not bool(sample["mask"]["f_imgseq"]):
        raise RuntimeError(f"BEDLAM key has no cached HMR2/imgfeat: {key}")
    return matches[0], sample


def make_obs_data(sample, key, sorted_idx, bbox_raw):
    from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    length = int(sample["length"])
    obs = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"]))
    obs[~sample["mask"]["valid"]] = 0
    meta = sample["meta"]
    return {
        "length": torch.tensor(length),
        "obs": obs,
        "bbx_xys": sample["bbx_xys"],
        "K_fullimg": sample["K_fullimg"],
        "T_w2c": sample.get("T_w2c", None),
        "cam_angvel": sample["cam_angvel"],
        "f_imgseq": sample["f_imgseq"],
        "source_name": f"bedlam_idx{sorted_idx}",
        "source_video": str(meta.get("mp4_path", bbox_raw.get("mp4_path", ""))),
        "preprocess_dir": "",
        "fps": float(bbox_raw.get("fps", 30.0)),
        "image_wh": _to_numpy_f32(bbox_raw.get("img_wh", None)),
        "meta": {
            "dataset": "bedlam",
            "idx": int(sorted_idx),
            "key": str(key),
            "length": length,
            "mp4_path": str(meta.get("mp4_path", bbox_raw.get("mp4_path", ""))),
            "has_vitpose": bool(sample["mask"]["vitpose"]),
            "has_imgfeat": bool(sample["mask"]["f_imgseq"]),
        },
    }


def main():
    from tools.infer_amass2g1 import load_pipeline
    from tools.infer_video2g1 import infer_video, make_prep_from_obs_data

    args = parse_args()
    _set_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    key, sorted_idx, n_common, bbox_raw = choose_key(args)
    dataset_idx, sample = load_full_sequence_sample(args, key)
    data = make_obs_data(sample, key, sorted_idx, bbox_raw)

    print(f"[BEDLAM] sorted_idx={sorted_idx}/{n_common} dataset_idx={dataset_idx}")
    print(f"[BEDLAM] key={key}")
    print(f"[BEDLAM] length={int(data['length'])} vitpose={data['meta']['has_vitpose']} imgfeat={data['meta']['has_imgfeat']}")
    print(f"[BEDLAM] source mp4={data['source_video']}")

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
        out_path,
        joint_pos=results["joint_pos"],
        root_pos_w=results["root_pos_w"],
        root_quat_w=results["root_quat_w"],
        **({"body_pos_w": results["body_pos_w"]} if "body_pos_w" in results else {}),
        K_fullimg=_to_numpy_f32(data.get("K_fullimg", None)),
        **({"T_w2c": _to_numpy_f32(data["T_w2c"])} if data.get("T_w2c", None) is not None else {}),
        **({"image_wh": np.asarray(data["image_wh"], dtype=np.int32)} if data.get("image_wh", None) is not None else {}),
        source_video=str(data.get("source_video", "")),
        preprocess_dir="",
        ckpt=str(args.ckpt),
        exp=str(args.exp),
        fps=np.array(float(data.get("fps", 30.0)), dtype=np.float32),
        source_name=str(data.get("source_name", f"bedlam_idx{sorted_idx}")),
    )
    out_path.with_suffix(".json").write_text(
        json.dumps({**data["meta"], "output": str(out_path)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Done] Saved: {out_path}")
    print(f"  - joint_pos:   {results['joint_pos'].shape} (BYD scalar dofs)")
    print(f"  - root_pos_w:  {results['root_pos_w'].shape} (z-up world)")
    print(f"  - root_quat_w: {results['root_quat_w'].shape} (wxyz, z-up world)")


if __name__ == "__main__":
    main()
