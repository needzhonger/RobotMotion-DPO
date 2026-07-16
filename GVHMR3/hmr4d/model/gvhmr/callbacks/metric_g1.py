"""G1 robot metric callback for GVHMR1 diffusion pipeline.

与 metric_emdb.py / metric_rich.py 同结构, 但评估 G1 机器人 FK 关节位置而非 SMPL 顶点.

指标说明:
  cr_mpjpe  (mm) : 每帧 root-aligned MPJPE, 对 30 个 body 平均.
  wa2_mpjpe (mm) : WA2 alignment MPJPE (first-frame align), 同 EMDB-2 全局轨迹指标.
  rte         (%) : relative trajectory error = |err| / total_disp × 100, 同 WHAM/EMDB-2.
  jitter       (:) : 运动抖动 (加速度二阶差分 / 10), 同 eval_utils.compute_jitter.
  dof_mae  (deg) : 29 个关节标量 DOF 的平均绝对误差.

multi-sample 扩散测试 (n_samples > 1):
  扩散模型每次推理采样不同, 跑 N 次取 cr_mpjpe 最小的样本 (best-of-N).
  需要 pl_module.pipeline.diffusion_enabled == True.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path

from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import (
    compute_global_metrics,
    as_np_array,
    first_align_joints,
    compute_jpe,
    compute_rte,
    compute_jitter,
)
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from hmr4d.model.gvhmr.utils.obs_joints import select_coco17_no_nose_ears


# ───────────────────────────── helpers ──────────────────────────────────────

def _mask_seq(x, mask):
    """Apply bool mask (L,) to tensor (L, ...) → (valid_L, ...)."""
    return x[mask]


def _cr_mpjpe_seq(pred_pos, gt_pos, mask):
    """Root-aligned MPJPE per frame.
    Args:
        pred_pos / gt_pos: (L, J, 3)
        mask: (L,) bool
    Returns:
        (valid_L,) numpy float32, unit mm
    """
    pred = _mask_seq(pred_pos, mask).float().cpu()  # (V, J, 3)
    gt   = _mask_seq(gt_pos,   mask).float().cpu()
    pred_cr = pred - pred[:, :1]   # root at origin
    gt_cr   = gt   - gt[:,   :1]
    diff = (pred_cr - gt_cr).norm(dim=-1)  # (V, J)
    return (diff.mean(dim=-1).numpy() * 1000).astype(np.float32)  # mm


def _dof_mae_seq(pred_dof, gt_dof, mask):
    """Mean absolute DOF error per frame.
    Args:
        pred_dof / gt_dof: (L, 29) radians
        mask: (L,) bool
    Returns:
        (valid_L,) numpy float32, unit degrees
    """
    pred = _mask_seq(pred_dof, mask).float().cpu()
    gt   = _mask_seq(gt_dof,   mask).float().cpu()
    diff_rad = (pred - gt).abs().mean(dim=-1)   # (V,)
    return (diff_rad.numpy() * (180.0 / np.pi)).astype(np.float32)  # degrees


# ────────────────────────────── callback ────────────────────────────────────

class MetricG1(pl.Callback):
    def __init__(self, n_samples: int = 1):
        """
        Args:
            n_samples: number of diffusion forward passes per sequence.
                       n_samples=1 → standard single-sample eval (works for both
                       regression and diffusion).
                       n_samples>1 → best-of-N diffusion eval: run N times, keep
                       the sample with minimum cr_mpjpe vs GT.
        """
        super().__init__()
        self.n_samples = max(1, int(n_samples))

        self.metric_aggregator = {
            "cr_mpjpe":  {},   # mm
            "wa2_mpjpe": {},   # mm  (WA2, first-frame xy align)
            "rte":       {},   # cm
            "jitter":    {},   # mm/s²
            "dof_mae":   {},   # degrees
        }

        # 与原版 callbacks 保持一致: val/test/predict 行为相同
        self.on_test_batch_end       = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end       = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ─────────────────────── per-batch computation ──────────────────────────

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        assert batch["B"] == 1, "MetricG1: batch size must be 1 during evaluation."

        if outputs is None:
            return

        # 只处理 G1 paired 数据 (有 g1_target), 兼容混合 dataloader
        if "g1_target" not in batch or "g1_body_pos_w" not in batch["g1_target"]:
            return

        vid       = batch.get("meta", [{"vid": f"seq{batch_idx:06d}"}])[0].get("vid", f"seq{batch_idx:06d}")
        L         = int(batch["length"][0])
        mask_full = batch["mask"]["valid"][0] if isinstance(batch["mask"], dict) else batch["mask"][0]  # (L_pad,)
        mask      = mask_full[:L]                                                                         # (L,)

        gt_body_pos = batch["g1_target"]["g1_body_pos_w"][0, :L]   # (L, 30, 3)  y-up world
        gt_joint    = batch["g1_target"]["g1_joint_pos"][0, :L]    # (L, 29) scalar DOFs

        # ── 得到预测: 单样本 (直接用 validation_step 结果) ──────────────────
        pred_body_pos = _get_pred_body_pos(outputs, L)   # (L, 30, 3) or None
        pred_joint    = _get_pred_joint_pos(outputs, L)  # (L, 29)    or None

        if pred_body_pos is None:
            Log.warning(f"[MetricG1] vid={vid}: missing pred_g1_body_pos_w in outputs, skip.")
            return

        # ── best-of-N 扩散采样 ─────────────────────────────────────────────
        if self.n_samples > 1 and getattr(getattr(pl_module, "pipeline", None), "diffusion_enabled", False):
            pred_body_pos, pred_joint = self._best_of_n(
                pl_module, batch, L, mask, gt_body_pos,
                pred_body_pos, pred_joint,
            )

        # ── 指标计算 ───────────────────────────────────────────────────────
        cr_mpjpe = _cr_mpjpe_seq(pred_body_pos, gt_body_pos, mask)
        self.metric_aggregator["cr_mpjpe"][vid] = cr_mpjpe

        # WA2 + RTE + Jitter: 沿用 compute_global_metrics 框架
        # (把 30-body positions 当 joints 传入; WA2 = first-frame XZ align)
        p_pos = _mask_seq(pred_body_pos, mask).float().cpu()   # (V, 30, 3)
        g_pos = _mask_seq(gt_body_pos,   mask).float().cpu()
        V = p_pos.shape[0]

        wa2_pos = first_align_joints(g_pos, p_pos)             # (V, 30, 3)
        wa2_mpjpe = compute_jpe(g_pos, wa2_pos) * 1000.0       # mm
        self.metric_aggregator["wa2_mpjpe"][vid] = wa2_mpjpe.astype(np.float32)

        # rte: 相对轨迹误差 (%). compute_rte 除以 total_disp → 无量纲, ×100 = %.
        # 至少需要 2 帧且轨迹非零; 否则跳过 (不插入 aggregator, 最终 concat 会略过).
        total_disp = (g_pos[1:, 0] - g_pos[:-1, 0]).norm(dim=-1).sum().item() if V >= 2 else 0.0
        if V >= 2 and total_disp > 1e-6:
            rte = compute_rte(g_pos[:, 0], p_pos[:, 0]) * 100.0
            self.metric_aggregator["rte"][vid] = np.atleast_1d(rte).astype(np.float32)

        # jitter: compute_jitter 需要 ≥ 4 帧
        if V >= 4:
            jitter = compute_jitter(p_pos, fps=30)
            self.metric_aggregator["jitter"][vid] = np.atleast_1d(jitter).astype(np.float32)

        if pred_joint is not None:
            dof_mae = _dof_mae_seq(pred_joint, gt_joint, mask)
            self.metric_aggregator["dof_mae"][vid] = dof_mae

    # ─────────────────────── best-of-N helper ───────────────────────────────

    @torch.no_grad()
    def _best_of_n(self, pl_module, batch, L, mask, gt_body_pos,
                   best_body_pos, best_joint_pos):
        """多跑 (n_samples-1) 次 pipeline.forward(train=False), 保留 cr_mpjpe 最小的.

        第一个样本 (outputs 里的那个) 已由 validation_step 算好, 这里补齐剩余 N-1 次.
        """
        def _cr_mpjpe_scalar(pred, gt, mask_):
            pred_m = _mask_seq(pred, mask_).float()
            gt_m   = _mask_seq(gt,   mask_).float()
            return (pred_m - pred_m[:, :1] - gt_m + gt_m[:, :1]).norm(dim=-1).mean().item()

        best_err = _cr_mpjpe_scalar(best_body_pos, gt_body_pos, mask)

        # 重建 batch_ (与 gvhmr_pl.validation_step 完全一致)
        batch_ = _rebuild_batch_for_inference(pl_module, batch)
        if batch_ is None:
            return best_body_pos, best_joint_pos

        pipeline = pl_module.pipeline
        for _ in range(self.n_samples - 1):
            try:
                out = pipeline.forward(batch_, train=False, postproc=False)
            except Exception as e:
                Log.warning(f"[MetricG1] best-of-N extra forward failed: {e}")
                continue

            cand_body_pos = _get_pred_body_pos(out, L)
            if cand_body_pos is None:
                continue

            # 只保留比当前最优更好的
            err = _cr_mpjpe_scalar(cand_body_pos, gt_body_pos, mask)
            if err < best_err:
                best_err      = err
                best_body_pos = cand_body_pos
                best_joint_pos = _get_pred_joint_pos(out, L)

        return best_body_pos, best_joint_pos

    # ─────────────────────── epoch summary ──────────────────────────────────

    def on_predict_epoch_end(self, trainer, pl_module):
        local_rank = trainer.local_rank
        monitor_metric = "cr_mpjpe"

        # 跨卡汇聚
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):
            gathered = all_gather(self.metric_aggregator)
        for k in metric_keys:
            for d in gathered:
                self.metric_aggregator[k].update(d[k])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # 每条序列监控指标排序 (最差在前)
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if mm_per_seq and local_rank == 0:
            sorted_mm = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm)
            Log.info(
                f"monitored metric {monitor_metric} per sequence\n"
                + "\n".join([f"{m:6.1f} : {s}" for s, m in sorted_mm[:n_worst]])
                + "\n------"
            )

        # 全局平均
        metrics_avg = {
            k: np.concatenate(list(v.values())).mean()
            for k, v in self.metric_aggregator.items()
            if len(v) > 0
        }
        if local_rank == 0:
            n_str = f" (best-of-{self.n_samples})" if self.n_samples > 1 else ""
            Log.info(
                f"[Metrics] G1{n_str}:\n"
                + "\n".join(f"  {k}: {v:.2f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # 记录到 logger
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_G1/{k}": v}, step=cur_epoch)

        # 重置
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


# ─────────────────────── batch reconstruction helper ────────────────────────

def _rebuild_batch_for_inference(pl_module, batch):
    """重建供 pipeline.forward(train=False) 用的 batch_, 与 gvhmr_pl.validation_step 一致.

    仅在 best-of-N 多样本推理时调用. Returns None on failure.
    """
    try:
        needs_synth = pl_module._needs_synthetic_eval_obs(batch)
        if needs_synth:
            synth  = pl_module._build_synthetic_obs(batch, do_augment=False, do_bbox_augment=False)
            obs    = synth["obs"]
            bbx    = synth["bbx_xys"]
        else:
            obs = select_coco17_no_nose_ears(normalize_kp2d(batch["kp2d"], batch["bbx_xys"]))
            bbx = batch["bbx_xys"]

        if "mask" in batch:
            mask_valid = batch["mask"]["valid"] if isinstance(batch["mask"], dict) else batch["mask"]
            obs[0, ~mask_valid[0]] = 0

        batch_ = {
            "length":    batch["length"],
            "obs":       obs,
            "bbx_xys":   bbx,
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq":  batch["f_imgseq"],
            "g1_target": batch.get("g1_target", {}),
        }
        if "mask" in batch:
            batch_["mask"] = batch["mask"]
        if "T_w2c" in batch:
            batch_["T_w2c"] = batch["T_w2c"]
        return batch_
    except Exception as e:
        Log.warning(f"[MetricG1] _rebuild_batch_for_inference failed: {e}")
        return None


# ─────────────────────── output extraction helpers ──────────────────────────

def _get_pred_body_pos(outputs, L):
    """Extract (L, 30, 3) body positions from validation_step outputs.

    validation_step 已对 pred_g1_body_pos_w / pred_joints_w 做了 [0] batch-strip;
    对 best-of-N 额外 forward 的 out 则是原始 (B, L, 30, 3) 带 batch 维.
    """
    for key in ("pred_g1_body_pos_w", "pred_joints_w"):
        v = outputs.get(key)
        if v is None:
            continue
        if v.dim() == 3:       # (L, 30, 3) — already stripped
            return v[:L]
        elif v.dim() == 4:     # (B, L, 30, 3) — extra forward
            return v[0, :L]
    return None


def _get_pred_joint_pos(outputs, L):
    """Extract (L, 29) scalar DOFs. Returns None if absent."""
    v = outputs.get("pred_g1_joint_pos")
    if v is None:
        return None
    if v.dim() == 2:   # (L, 29) stripped
        return v[:L]
    elif v.dim() == 3: # (B, L, 29) extra forward
        return v[0, :L]
    return None


# ─────────────────────── Hydra store registration ───────────────────────────

node_g1        = builds(MetricG1, n_samples=1)
node_g1_best5  = builds(MetricG1, n_samples=5)

MainStore.store(name="metric_g1",       node=node_g1,       group="callbacks", package="callbacks.metric_g1")
MainStore.store(name="metric_g1_best5", node=node_g1_best5, group="callbacks", package="callbacks.metric_g1_best5")
