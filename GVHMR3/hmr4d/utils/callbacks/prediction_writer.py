"""Test-time PredictionWriter callback for the G1 paired AMASS pipeline.

设计目标 — 跑 `task=test` 时:
  - 在 `validation_step` 处理完每个样本后, 把 pipeline 推理结果反变换到 z-up URDF
    原生帧, 落盘成 npz, schema 与 tools/infer_amass2g1.py 输出一致.
  - 关掉 validation_step 内的 val_loss forward (节省一次 train=True forward, 同时
    AMASS-only 模式下没有 GT G1 也不会崩, 见 gvhmr_pl.py:315 ~ self.skip_val_loss).
  - 不再单独跑一次 forward (复用 validation_step return 的 outputs, 见 gvhmr_pl.py:438).

落盘字段:
  joint_pos    (T, 29)   BYD scalar dofs (frame-invariant)
  root_pos_w   (T, 3)    z-up URDF world (M_pos = Ry_neg90 @ T_az2ay 撤回)
  root_quat_w  (T, 4)    wxyz, z-up URDF world
  fps          ()        训练 amass_fps (默认 30)
  T_w2c        (4, 4)    static_v11 下整段一台相机, 存 frame-0 即可
  K_fullimg    (3, 3)    相机内参
  key          (str)     唯一 pth key (= meta["key"]), viz 精确匹配 AMASS/GT-G1 用
                         (老 npz 无此字段, viz 端会回退 stem 模糊匹配)

文件名: dataset.idx2meta[batch_idx]['name'] + seg_id. dataset 直接从 datamodule.testsets[dl_idx] 拿
(`trainer.test_dataloaders` 是 CombinedLoader, 没有 `.dataset` 属性, 走 datamodule 才稳).
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.hmr_cam import perspective_projection
from hmr4d.utils.pylogger import Log


# (apply_az_to_ay_g1 + Ry_neg90) 的合成 (g1_amass.py:1810-1822). 推理输出在 "rotated y-up" 系.
# 反过来到 z-up URDF native:  v_z = v_yr @ M_pos
# M_pos = Ry_neg90 @ T_az2ay = [[0,1,0],[0,0,1],[1,0,0]]   (3-cycle x→z, y→x, z→y)
_M_POS_BUF = torch.tensor(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0]]
)


def _yup_to_zup_pos(pos_y: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotated-y-up → z-up URDF native."""
    M = _M_POS_BUF.to(pos_y.device, pos_y.dtype)
    return pos_y @ M


def _yup_to_zup_quat_wxyz(quat_y: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz, rotated-y-up → z-up URDF native wxyz."""
    M = _M_POS_BUF.to(quat_y.device, quat_y.dtype)
    R_y = quaternion_to_matrix(quat_y)
    R_z = M.transpose(-1, -2) @ R_y          # R_z = M.T @ R_yr
    return matrix_to_quaternion(R_z)


def _resolve_test_dataset(trainer, dataloader_idx=0):
    """多路径回退抓 test dataset, 兼容不同 Lightning 版本 / 调用约定.

    背景:
      - mocap_trainX_testY.test_dataloader() 返回 `CombinedLoader(loaders, mode='sequential')`
        (mocap_trainX_testY.py:148). PL 2.x 下 CombinedLoader 没有 `.dataset`,
        但有 `.iterables` / `.flattened` (验证: hasattr check pass).
      - train.py:103 调 `trainer.test(model, datamodule.test_dataloader())`, 没把 datamodule
        本身传给 trainer, 所以 `trainer.datamodule` 会是 None.
      - 唯一稳的路径: `trainer.test_dataloaders` → CombinedLoader → `.iterables[i].dataset`.

    多个备选都试一遍, 哪个能拿到带 idx2meta 的 dataset 就用哪个.
    """
    candidates = []

    # Path A: trainer.test_dataloaders 直接是 list / 单个 DataLoader / CombinedLoader
    td = getattr(trainer, "test_dataloaders", None)
    if td is not None:
        # 单个 DataLoader: 直接拿 .dataset
        candidates.append(lambda: td.dataset)
        # 单个 CombinedLoader (PL 2.x): .iterables[i].dataset
        candidates.append(lambda: td.iterables[dataloader_idx].dataset)
        candidates.append(lambda: td.flattened[dataloader_idx].dataset)
        # 多个 DataLoader 的 list (PL 1.x)
        candidates.append(lambda: td[dataloader_idx].dataset)

    # Path B: 借 datamodule.testsets (我们自己在 mocap_trainX_testY.py:96 setattr 的 list)
    dm = getattr(trainer, "datamodule", None)
    if dm is not None and hasattr(dm, "testsets"):
        candidates.append(lambda: dm.testsets[dataloader_idx])

    from torch.utils.data import Subset as _Subset

    def _has_meta(d):
        """d 本身或其 Subset.dataset 有 idx2meta 即可."""
        if d is None:
            return False
        if hasattr(d, "idx2meta"):
            return True
        if isinstance(d, _Subset) and hasattr(d.dataset, "idx2meta"):
            return True
        return False

    for fn in candidates:
        try:
            ds = fn()
        except (AttributeError, TypeError, IndexError, KeyError):
            continue
        if _has_meta(ds):
            return ds
    return None


class G1PredictionWriter(pl.Callback):
    """每个 test sample 落一个 npz, 直接复用 validation_step 返回的 outputs."""

    def __init__(
        self,
        output_dir: str,
        filename_template: str = "{name}_seg{seg_id:03d}.npz",
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename_template = filename_template
        self._written = 0
        self._test_dataset = None

    def on_test_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Log.info(f"[G1PredictionWriter] dump dir: {self.output_dir}")
        # 关掉 validation_step 的 val_loss forward — test 时只算推理产物,
        # 省一次 train=True forward, 也让 AMASS-only 模式 (无 GT G1) 不崩.
        # 见 gvhmr_pl.py:315 `if not getattr(self, "skip_val_loss", False):`.
        pl_module.skip_val_loss = True
        Log.info(f"[G1PredictionWriter] pl_module.skip_val_loss=True (跳过 val_loss forward)")
        self._written = 0
        # 提前缓存 test dataset, 不每 batch 都重新解析 (CombinedLoader 跨版本属性容易踩坑).
        self._test_dataset = _resolve_test_dataset(trainer, dataloader_idx=0)
        if self._test_dataset is None:
            Log.warning("[G1PredictionWriter] 无法定位 test dataset, 文件名会退化为 sample{batch_idx}.npz")
        else:
            from torch.utils.data import Subset as _Subset
            _inner = self._test_dataset.dataset if isinstance(self._test_dataset, _Subset) else self._test_dataset
            n = len(getattr(_inner, "idx2meta", []))
            Log.info(f"[G1PredictionWriter] resolved test dataset: {type(self._test_dataset).__name__} (idx2meta={n})")

    def on_test_end(self, trainer, pl_module):
        # 还原, 不污染其他 trainer.test() 调用.
        pl_module.skip_val_loss = False
        Log.info(f"[G1PredictionWriter] wrote {self._written} npz file(s) to {self.output_dir}")

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """consume validation_step return value (gvhmr_pl.py:438) — 不再单独跑 forward."""
        if outputs is None:
            Log.warning("[G1PredictionWriter] on_test_batch_end 收到 outputs=None, 跳过 dump")
            return
        # validation_step 已经为这几个 key 取 [0] 把 batch 维 drop 掉了 (gvhmr_pl.py:306-308):
        #   pred_g1_joint_pos / pred_g1_body_pos_w / pred_joints_w / pred_root_pos
        # 但 pred_root_quat_w 不在那个 list 里, 仍是 (B, L, 4), 这里手动 [0].
        L = int(batch["length"][0])
        joint_pos   = outputs["pred_g1_joint_pos"][:L].float().cpu()      # (L, 29)
        root_pos_y  = outputs["pred_root_pos"][:L].float().cpu()          # (L, 3)
        prq         = outputs["pred_root_quat_w"]                          # (B, L, 4) wxyz
        root_quat_y = prq[0, :L].float().cpu() if prq.dim() == 3 else prq[:L].float().cpu()

        # rotated-y-up → URDF z-up
        root_pos_z  = _yup_to_zup_pos(root_pos_y)
        root_quat_z = _yup_to_zup_quat_wxyz(root_quat_y)

        # ---- 2D 输入观测 (COCO17, pixel coords) — 给 viz 可视化网络输入用 ----
        # 重新跑一次 _build_synthetic_obs 拿 gt_j3d (camera frame), 再 perspective_projection.
        # 多一次 SMPLX forward, full_sequence 长 L 慢但只在 dump 时跑一次.
        try:
            synth = pl_module._build_synthetic_obs(batch, do_augment=False, do_bbox_augment=False)
            gt_j3d_cam = synth["gt_j3d"]                                                # (1, L, 17, 3)
            kp2d_pixel = perspective_projection(gt_j3d_cam, batch["K_fullimg"])         # (1, L, 17, 2)
            kp2d_pixel_l = kp2d_pixel[0, :L].float().cpu().numpy().astype(np.float32)
            kp2d_visible = (gt_j3d_cam[0, :L, :, 2] >= 0.3).float().cpu().numpy().astype(np.float32)  # (L, 17)
        except Exception as e:
            Log.warning(f"[G1PredictionWriter] dump obs_kp2d_pixel 失败: {e}")
            kp2d_pixel_l = np.zeros((L, 17, 2), dtype=np.float32)
            kp2d_visible = np.zeros((L, 17), dtype=np.float32)
        # 网络在 1000×1000 K_fullimg (create_camera_sensor(1000, 1000, 43.3)) 下投影, 图像尺寸固定 1000.
        image_wh = np.array([1000, 1000], dtype=np.int32)

        # ---- 文件名 ----
        # 优先用 on_test_start 缓存的 dataset; 兜底再现场解析一次.
        ds = self._test_dataset
        if ds is None or not hasattr(ds, "idx2meta"):
            ds = _resolve_test_dataset(trainer, dataloader_idx=dataloader_idx)
        # key_str = 抽取动作时配好的唯一 pth key (AMASS/GT-G1 同一个), 写进 npz 供 viz 精确匹配.
        # 缺 meta 时回退用文件名 name (viz 端再退化为 stem 模糊匹配).
        key_str = ""
        # 支持 Subset 包装 (data.test_subset_indices): 从 ds.indices[batch_idx] 取真实 idx.
        from torch.utils.data import Subset as _Subset
        _meta_ds, _meta_idx = ds, batch_idx
        if isinstance(ds, _Subset) and batch_idx < len(ds.indices):
            _meta_ds = ds.dataset
            _meta_idx = ds.indices[batch_idx]
        if _meta_ds is not None and hasattr(_meta_ds, "idx2meta") and 0 <= _meta_idx < len(_meta_ds.idx2meta):
            meta = _meta_ds.idx2meta[_meta_idx]
            name = meta.get("name", f"sample{batch_idx:06d}")
            seg_id = int(meta.get("seg_id", 0))
            key_str = str(meta.get("key", name))
            try:
                fname = self.filename_template.format(name=name, seg_id=seg_id)
            except (KeyError, IndexError):
                fname = f"{name}_seg{seg_id:03d}.npz"
        else:
            fname = f"sample{batch_idx:06d}.npz"

        # ---- fps + camera (T_w2c 第 0 帧 + K), 让下游知道帧率 + 几何 ----
        fps = float(getattr(ds, "amass_fps", 30.0)) if ds is not None else 30.0
        T_w2c = (batch["T_w2c"][0, 0].float().cpu().numpy().astype(np.float32)
                 if "T_w2c" in batch else np.eye(4, dtype=np.float32))
        K_fullimg = batch["K_fullimg"][0, 0].float().cpu().numpy().astype(np.float32)

        np.savez(
            self.output_dir / fname,
            joint_pos       = joint_pos.numpy().astype(np.float32),
            root_pos_w      = root_pos_z.numpy().astype(np.float32),
            root_quat_w     = root_quat_z.numpy().astype(np.float32),
            fps             = np.asarray(fps, dtype=np.float32),
            T_w2c           = T_w2c,
            K_fullimg       = K_fullimg,
            obs_kp2d_pixel  = kp2d_pixel_l,    # (L, 17, 2)  COCO17 网络输入像素坐标
            obs_kp2d_visible= kp2d_visible,    # (L, 17)     可见性 (cam-frame depth ≥ 0.3)
            image_wh        = image_wh,        # (2,)        投影时的图像 W,H
            key             = key_str,         # str         唯一 pth key, viz 精确匹配用
        )
        self._written += 1


# ==========================================================
# Hydra store registration
# ==========================================================
group_name = "callbacks/prediction_writer"
base = builds(
    G1PredictionWriter,
    output_dir="${output_dir}/predictions/",
    populate_full_signature=True,
)
MainStore.store(name="g1_dualpth", node=base, group=group_name)
