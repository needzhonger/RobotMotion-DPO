from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo.hmr_global import get_R_c2gv
from hmr4d.utils.geo_transform import compute_cam_angvel, transform_mat
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict

from .cam_traj_utils import CameraAugmentorV11, StaticCameraV11
from .g1_amass import _MJC_TO_BYD, _apply_ry_neg90_on_quat_wxyz, _apply_ry_neg90_on_vec

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


BONES_SEED_SOMA_OBS14 = [6, 7, 9, 15, 10, 16, 11, 17, 20, 24, 21, 25, 22, 26]


def _load_joblib(path):
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "BonesSeedG1SomaDataset requires joblib. Activate the project env "
            "(for example: conda activate gvhmr) before loading this dataset."
        ) from e
    return joblib.load(path)


def _as_float_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32).clone()


def _finite_diff_lin_vel(x, fps):
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    diff = (x[1:] - x[:-1]) * float(fps)
    return torch.cat([diff, diff[-1:].clone()], dim=0)


def _finite_diff_ang_vel(q_wxyz, fps):
    if q_wxyz.shape[0] < 2:
        return torch.zeros(*q_wxyz.shape[:-1], 3, device=q_wxyz.device, dtype=q_wxyz.dtype)
    R = quaternion_to_matrix(F.normalize(q_wxyz, dim=-1))
    dR = R[1:] @ R[:-1].transpose(-1, -2)
    aa = matrix_to_axis_angle(dR) * float(fps)
    return torch.cat([aa, aa[-1:].clone()], dim=0)


def _slerp_quat(q_seq, tgt_len):
    q_seq = F.normalize(q_seq, dim=-1)
    T = q_seq.shape[0]
    if T == tgt_len:
        return q_seq
    if T == 1:
        return q_seq.expand(tgt_len, *q_seq.shape[1:]).clone()

    extra = q_seq.shape[1:]
    t_idx = torch.linspace(0, T - 1, tgt_len, device=q_seq.device, dtype=q_seq.dtype)
    i0 = t_idx.floor().long().clamp(0, T - 2)
    i1 = (i0 + 1).clamp(0, T - 1)
    alpha = (t_idx - i0.to(dtype=q_seq.dtype)).reshape(tgt_len, *([1] * len(extra)))

    q0 = q_seq[i0]
    q1 = q_seq[i1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sin_t = theta.sin()
    w0 = torch.where(sin_t > 1e-6, torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-10), 1.0 - alpha)
    w1 = torch.where(sin_t > 1e-6, torch.sin(alpha * theta) / (sin_t + 1e-10), alpha)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def _interp_seq(x, tgt_len):
    if x.shape[0] == tgt_len:
        return x
    T = x.shape[0]
    x_flat = x.reshape(T, -1).transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x_flat, size=tgt_len, mode="linear", align_corners=True)
    return y.squeeze(0).transpose(0, 1).reshape(tgt_len, *x.shape[1:])


def az_to_ay_rotmat(device, dtype):
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def apply_az_to_ay_on_vec(v):
    T = az_to_ay_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_az_to_ay_on_quat_wxyz(q_wxyz):
    T = az_to_ay_rotmat(q_wxyz.device, q_wxyz.dtype)
    return matrix_to_quaternion(T @ quaternion_to_matrix(F.normalize(q_wxyz, dim=-1)))


def apply_az_to_ay_on_axis_angle(global_orient_aa):
    T = az_to_ay_rotmat(global_orient_aa.device, global_orient_aa.dtype)
    return matrix_to_axis_angle(T @ axis_angle_to_matrix(global_orient_aa))


def _first_frame_inv_rot(R0, yaw_only=False, up_axis=2, fwd_axis=0):
    if not yaw_only:
        return R0.transpose(-1, -2)
    fwd = R0[:, fwd_axis]
    if up_axis == 1:
        heading = torch.atan2(fwd[0], fwd[2])
    elif up_axis == 2:
        heading = torch.atan2(fwd[1], fwd[0])
    else:
        heading = torch.atan2(fwd[2], fwd[1])
    aa = torch.zeros(3, device=R0.device, dtype=R0.dtype)
    aa[up_axis] = -heading
    return axis_angle_to_matrix(aa)


def canonicalize_smpl_first_frame_bones_seed(global_orient_aa, joints, yaw_only=True, up_axis=2, fwd_axis=1):
    R = axis_angle_to_matrix(global_orient_aa)
    R0_inv = _first_frame_inv_rot(R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)
    R_new = R0_inv @ R
    root0 = joints[0, 0].clone()
    joints_new = (joints - root0) @ R0_inv.T
    joints_new[0, 0] = 0.0
    return matrix_to_axis_angle(R_new), joints_new


def canonicalize_g1_first_frame_bones_seed(g1, root_body_id=0, yaw_only=True, up_axis=2, fwd_axis=0):
    body_quat_w = F.normalize(g1["body_quat_w_full"], dim=-1)
    body_pos_w = g1["body_pos_w_full"]
    R = quaternion_to_matrix(body_quat_w)
    R0_inv = _first_frame_inv_rot(R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)
    center = body_pos_w[0, root_body_id].clone()

    R_new = R0_inv @ R
    q_new = matrix_to_quaternion(R_new)
    q_new = torch.where(q_new[..., :1] < 0, -q_new, q_new)
    if not yaw_only:
        q_new[0, root_body_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype)

    g1["body_quat_w"] = q_new
    g1["body_pos_w"] = (body_pos_w - center) @ R0_inv.T
    g1["body_pos_w"][0, root_body_id] = 0.0
    g1["root_rot"] = g1["body_quat_w"][:, root_body_id].clone()
    g1["root_pos"] = g1["body_pos_w"][:, root_body_id].clone()
    return g1


class BonesSeedG1SomaDataset(Dataset):
    """Bones Seed paired SOMA/G1 dataset.

    The shared data directory is read-only. This dataset keeps all generated
    tensors in memory and writes nothing outside the project output path.
    """

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
                self.records[key] = {"file": str(p), "seq_name": seq_name, "payload": payload}

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
        return int(np.asarray(self.records[key]["payload"]["g1"]["dof_pos"]).shape[0])

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
        rec = self.records[meta["key"]]["payload"]
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

        dof_pos = _interp_seq(dof_pos, tgt_len)
        root_pos = _interp_seq(root_pos, tgt_len)
        root_rot = _slerp_quat(root_rot, tgt_len)
        body_pos = _interp_seq(body_pos, tgt_len)
        body_quat = _slerp_quat(body_quat, tgt_len)
        soma_go_q = _slerp_quat(matrix_to_quaternion(axis_angle_to_matrix(soma_go)), tgt_len)
        soma_go = matrix_to_axis_angle(quaternion_to_matrix(soma_go_q))
        soma_joints = _interp_seq(soma_joints, tgt_len)

        g1 = {
            "fps": torch.tensor(fps, dtype=torch.float32),
            "joint_pos": dof_pos,
            "body_pos_w_full": body_pos,
            "body_quat_w_full": root_rot.new_tensor(body_quat),
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
            g1["root_pos"] = root_pos
            g1["root_rot"] = root_rot

        # z-up -> y-up. Then rotate G1 by Ry(-90) so G1 +x forward aligns to
        # human +z forward, matching the AMASS/G1 training convention.
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
        width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)
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

        # Minimal SMPL placeholders. They keep downstream debug code batchable;
        # Bones Seed observations come from soma_obs_joints_w, not these tensors.
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
