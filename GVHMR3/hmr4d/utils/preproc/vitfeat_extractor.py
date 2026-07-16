import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.cuda()
        batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size]

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pth")
    torch.save(data, tmp)
    os.replace(tmp, path)


def _make_shard_path(out_pth, num_shards, shard_idx):
    out_pth = Path(out_pth)
    return out_pth.with_name(f"{out_pth.stem}.shard{shard_idx:02d}-of-{num_shards:02d}{out_pth.suffix}")


def _validate_shard_args(num_shards, shard_idx):
    num_shards = int(num_shards)
    shard_idx = int(shard_idx)
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")
    return num_shards, shard_idx


def _select_keys_for_shard(keys, start=0, limit=0, num_shards=1, shard_idx=0):
    if start:
        keys = keys[int(start) :]
    if limit:
        keys = keys[: int(limit)]
    num_shards, shard_idx = _validate_shard_args(num_shards, shard_idx)
    if num_shards > 1:
        keys = keys[shard_idx::num_shards]
    return keys


def merge_bedlam2_imgfeat_shards(out_pth, num_shards, allow_missing_shards=False):
    out_pth = Path(out_pth)
    num_shards, _ = _validate_shard_args(num_shards, 0)
    merged = {}
    merged_errors = {}
    for shard_idx in range(num_shards):
        shard_pth = _make_shard_path(out_pth, num_shards, shard_idx)
        if not shard_pth.exists():
            if allow_missing_shards:
                print(f"[BEDLAM2 HMR2Feat][merge] missing shard: {shard_pth}")
                continue
            raise FileNotFoundError(shard_pth)
        shard_data = _torch_load_cpu(shard_pth)
        overlap = set(merged).intersection(shard_data)
        if overlap:
            raise RuntimeError(f"duplicate keys while merging {shard_pth}: {len(overlap)}")
        merged.update(shard_data)
        print(f"[BEDLAM2 HMR2Feat][merge] {shard_pth}: {len(shard_data)} clips")

        shard_err_pth = shard_pth.with_suffix(".errors.pth")
        if shard_err_pth.exists():
            merged_errors.update(_torch_load_cpu(shard_err_pth))

    _save_atomic(out_pth, merged)
    if merged_errors:
        err_pth = out_pth.with_suffix(".errors.pth")
        _save_atomic(err_pth, merged_errors)
        print(f"[BEDLAM2 HMR2Feat][merge] saved {len(merged_errors)} errors to {err_pth}")
    print(f"[BEDLAM2 HMR2Feat][merge] saved {len(merged)} clips to {out_pth}")
    return merged


def extract_bedlam2_imgfeat(
    bbox_pth,
    out_pth,
    start=0,
    limit=0,
    save_every=20,
    img_ds=0.5,
    num_shards=1,
    shard_idx=0,
):
    """Run original GVHMR/HMR2 image feature extraction on filtered BEDLAM2 videos."""
    bbox_pth = Path(bbox_pth)
    out_pth = Path(out_pth)
    bbox_data = _torch_load_cpu(bbox_pth)
    keys = _select_keys_for_shard(
        list(bbox_data.keys()), start=start, limit=limit, num_shards=num_shards, shard_idx=shard_idx
    )
    if int(num_shards) > 1:
        print(f"[BEDLAM2 HMR2Feat] shard {int(shard_idx) + 1}/{int(num_shards)}: {len(keys)} clips -> {out_pth}")

    results = {}
    if out_pth.exists():
        results = _torch_load_cpu(out_pth)
        print(f"[BEDLAM2 HMR2Feat] resume from {out_pth}: {len(results)} done")

    extractor = Extractor(tqdm_leave=False)
    processed = 0
    errors = {}
    for key in tqdm(keys, desc="BEDLAM2 HMR2Feat", dynamic_ncols=True):
        if key in results:
            continue
        rec = bbox_data[key]
        try:
            video_path = rec.get("mp4_path")
            if not video_path:
                raise KeyError(f"missing mp4_path for {key}")
            bbx_xys = rec["bbx_xys"].float()
            valid = rec.get("valid")
            if valid is not None:
                valid = valid.bool()
                bbx_xys = bbx_xys.clone()
                if (~valid).any() and valid.any():
                    bbx_xys[~valid] = bbx_xys[valid][0]
            features = extractor.extract_video_features(str(video_path), bbx_xys, img_ds=img_ds)
            results[key] = {
                "features": features.float().cpu(),
                "start_end": (0, int(features.shape[0])),
                "mp4_path": str(video_path),
            }
            processed += 1
            if save_every > 0 and processed % int(save_every) == 0:
                _save_atomic(out_pth, results)
        except Exception as exc:
            errors[key] = repr(exc)
            print(f"[BEDLAM2 HMR2Feat][error] {key}: {exc}")

    _save_atomic(out_pth, results)
    if errors:
        err_path = out_pth.with_suffix(".errors.pth")
        _save_atomic(err_path, errors)
        print(f"[BEDLAM2 HMR2Feat] saved errors to {err_path}")
    print(f"[BEDLAM2 HMR2Feat] saved {len(results)} clips to {out_pth}")
    return results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedlam2_bbox_pth", type=Path, default=None)
    parser.add_argument("--out_pth", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--img_ds", type=float, default=0.5)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--allow_missing_shards", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    out_pth = args.out_pth or Path(
        "/mnt/ddn/tianyi/PHUMA/data/filtered_bedlam2_imgfeat/filtered_bedlam2_imgfeat.pth"
    )
    _validate_shard_args(args.num_shards, args.shard_idx)
    if args.merge_shards:
        merge_bedlam2_imgfeat_shards(
            out_pth,
            num_shards=args.num_shards,
            allow_missing_shards=args.allow_missing_shards,
        )
        return
    if args.bedlam2_bbox_pth is None:
        raise SystemExit("Pass --bedlam2_bbox_pth to run BEDLAM2 HMR2 image feature extraction.")
    run_out_pth = _make_shard_path(out_pth, args.num_shards, args.shard_idx) if args.num_shards > 1 else out_pth
    extract_bedlam2_imgfeat(
        args.bedlam2_bbox_pth,
        run_out_pth,
        start=args.start,
        limit=args.limit,
        save_every=args.save_every,
        img_ds=args.img_ds,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )


if __name__ == "__main__":
    main()
