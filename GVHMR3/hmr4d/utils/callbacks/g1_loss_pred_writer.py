"""G1PredictionWriter + 逐 window loss 记录.

在 G1PredictionWriter (落 pred npz) 基础上, 对每个 test window 额外跑一次 train forward
(与 gvhmr_pl.validation_step 的 val_loss 路径完全一致: 同样的 synthetic obs + train=True +
null_condition=False), 把 outputs["loss"] (总) 与各分量 (*_loss) 记到 loss_table.csv.

用途: 全训练集窗口化扫一遍 loss, 供 tools/select_loss_windows.py 抽 40 随机 + 40 高 loss.
loss 与 pred npz 在同一次 forward / 同一 (split_seed, idx) 确定性样本上算, 故与 hepta 渲染一致.
"""

from pathlib import Path

import numpy as np
import torch

from hmr4d.configs import MainStore, builds
from hmr4d.utils.callbacks.prediction_writer import G1PredictionWriter
from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears
from hmr4d.utils.pylogger import Log


class G1LossPredWriter(G1PredictionWriter):
    def __init__(
        self,
        output_dir: str,
        filename_template: str = "{name}_seg{seg_id:03d}.npz",
        loss_table: str = "loss_table.csv",
    ):
        super().__init__(output_dir, filename_template)
        self.loss_table = loss_table
        self._rows = []

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)  # 设 skip_val_loss=True + 缓存 test dataset
        self._rows = []
        # 防撞名 + 标 idx: PredictionWriter 用 Path(key).stem 当文件名, 不同 key 可能同 stem
        # (如 KIT/10 与 KIT/5 下都有 RightTurn03_stageii) → npz 互相覆盖. 这里在 name 开头
        # 加上该 window 的全局 idx (= idx2meta 顺序 = test 时 batch_idx, 0..N-1), 使文件名
        # 以 idx 开头且全局唯一. name 只用于文件名, 不参与数据加载 (_load_data 只用
        # key/seg_id/usable_len), 故可安全改写.
        ds = self._test_dataset
        if ds is not None and hasattr(ds, "idx2meta"):
            ndig = max(5, len(str(len(ds.idx2meta) - 1)))
            for i, meta in enumerate(ds.idx2meta):
                stem = str(meta.get("name", ""))
                meta["name"] = f"{i:0{ndig}d}_{stem}"

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1) 先复用父类落 pred npz (消费 validation_step return 的 inference outputs)
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        # 2) 复刻 validation_step (gvhmr_pl.py:275-325) 的 obs 构造 + train forward 取 loss
        loss_out = {}
        try:
            if pl_module._needs_synthetic_eval_obs(batch):
                synth = pl_module._build_synthetic_obs(batch, do_augment=False, do_bbox_augment=False)
                obs = synth["obs"]
                bbx_xys = synth["bbx_xys"]
            else:
                from hmr4d.utils.geo.hmr_cam import normalize_kp2d
                obs = select_coco17_no_nose_ears(normalize_kp2d(batch["kp2d"], batch["bbx_xys"]))
                bbx_xys = batch["bbx_xys"]
            if "mask" in batch:
                mv = batch["mask"]["valid"] if isinstance(batch["mask"], dict) else batch["mask"]
                obs[0, ~mv[0]] = 0
            batch_loss = {**batch, "obs": obs, "bbx_xys": bbx_xys}
            loss_out = pl_module.pipeline.forward(batch_loss, train=True, null_condition=False)
        except Exception as e:
            Log.warning(f"[G1LossPredWriter] loss forward 失败 batch={batch_idx}: {e}")

        # 3) meta (name/key/seg_id) — 与父类落名同口径
        ds = self._test_dataset
        name = f"sample{batch_idx:06d}"
        key = ""
        seg_id = 0
        if ds is not None and hasattr(ds, "idx2meta") and 0 <= batch_idx < len(ds.idx2meta):
            meta = ds.idx2meta[batch_idx]
            name = meta.get("name", name)
            seg_id = int(meta.get("seg_id", 0))
            key = str(meta.get("key", name))
        try:
            fname = self.filename_template.format(name=name, seg_id=seg_id)
        except (KeyError, IndexError):
            fname = f"{name}_seg{seg_id:03d}.npz"

        row = {"idx": batch_idx, "file": fname, "key": key, "seg_id": seg_id}
        for k, v in loss_out.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0 and (k == "loss" or k.endswith("_loss")):
                row[k] = float(v.item())
        self._rows.append(row)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        if not self._rows:
            Log.warning("[G1LossPredWriter] 没有 loss 行可写")
            return
        loss_keys = sorted({k for r in self._rows for k in r if k == "loss" or k.endswith("_loss")})
        # 总 loss 'loss' 放最前, 其余分量字母序
        ordered = (["loss"] if "loss" in loss_keys else []) + [k for k in loss_keys if k != "loss"]
        cols = ["idx", "file", "key", "seg_id"] + ordered
        out = self.output_dir / self.loss_table
        with open(out, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in self._rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        Log.info(f"[G1LossPredWriter] wrote loss table ({len(self._rows)} rows) → {out}")


# ==========================================================
# Hydra store registration
# ==========================================================
group_name = "callbacks/prediction_writer"
base = builds(
    G1LossPredWriter,
    output_dir="${output_dir}/predictions/",
    populate_full_signature=True,
)
MainStore.store(name="g1_dualpth_loss", node=base, group=group_name)
