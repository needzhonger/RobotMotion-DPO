import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import gaussian_smooth

from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from hmr4d.model.gvhmr.utils import stats_compose

from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    get_a_pred_cam,
    get_bbx_xys,
    perspective_projection,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_tgtcoord_rootparam,
)


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        # Log.info(self.denoiser3d)

        # Condition embedder (GEM-X style: condition embedding moved OUT of the
        # per-step denoiser so f_cond is built once & reused across DDIM steps,
        # and a separate unconditional f_uncond can be built for CFG). Sized from
        # the network's *_dim attrs. Used by both diffusion and regression paths.
        from hmr4d.network.gvhmr.relative_transformer import ConditionEmbedder
        self.cond_embedder = ConditionEmbedder(
            latent_dim=self.denoiser3d.latent_dim,
            cliffcam_dim=self.denoiser3d.cliffcam_dim,
            cam_angvel_dim=self.denoiser3d.cam_angvel_dim,
            imgseq_dim=self.denoiser3d.imgseq_dim,
            obs_num_joints=getattr(self.denoiser3d, "obs_num_joints", 14),
            dropout=self.denoiser3d.dropout,
        )

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)

        # G1 FK engine — required for cr_j3d / fk_j3d / j2d / transl_c losses.
        # Without this call all FK-dependent losses silently fall back to GT-vs-GT.
        mjcf_file = getattr(getattr(self.args, "robot", None), "mjcf_file", None)
        if mjcf_file is not None:
            self.endecoder.setup_g1_fk(mjcf_file)
            Log.info(f"[Pipeline] G1 FK initialised from {mjcf_file}")
        else:
            Log.warning("[Pipeline] args.robot.mjcf_file not set; FK losses will be skipped.")

        # Load G1 pred_x stats (189-D GVHMR format).
        # Falls back to no normalisation when stats_file is absent.
        stats_file = getattr(self.args, "stats_file", None)
        if stats_file is not None:
            try:
                import numpy as _np
                _d = _np.load(stats_file)
                if "pred_mean" in _d and "pred_std" in _d:
                    from hmr4d.model.gvhmr.utils.endecoder import G1_PRED_X_DIM
                    _mean = torch.tensor(_d["pred_mean"]).float()
                    _std  = torch.tensor(_d["pred_std"]).float().clamp(min=1e-6)
                    assert _mean.shape[-1] == G1_PRED_X_DIM, (
                        f"pred_mean dim {_mean.shape[-1]} != G1_PRED_X_DIM {G1_PRED_X_DIM}. "
                        "Regenerate stats with tools/make_g1_pred_stats.py"
                    )
                    self.endecoder.set_g1_pred_stats(_mean, _std)
                    Log.info(f"[Pipeline] Loaded G1 pred_x stats (189-D) from {stats_file}")
                else:
                    # legacy 299-D stats – still register for decode_g1 compatibility
                    self.endecoder.compute_g1_stats_from_npz(stats_file)
                    Log.info(f"[Pipeline] Loaded legacy G1 stats (299-D) from {stats_file}")
            except Exception as e:
                Log.warning(f"[Pipeline] Could not load G1 stats from {stats_file}: {e}. "
                         "Running without normalisation.")
        else:
            Log.warning("[Pipeline] args.stats_file not set – G1 pred_x will NOT be normalised.")

        # Sanity-check: network output_dim must be 189
        from hmr4d.model.gvhmr.utils.endecoder import G1_PRED_X_DIM
        if hasattr(self.denoiser3d, "output_dim"):
            net_dim = int(self.denoiser3d.output_dim)
            if net_dim != G1_PRED_X_DIM:
                raise ValueError(
                    f"[Pipeline] Denoiser output_dim ({net_dim}) != G1_PRED_X_DIM "
                    f"({G1_PRED_X_DIM}). Set network.output_dim={G1_PRED_X_DIM} in your exp yaml."
                )

        # ===== Diffusion (optional, config-gated) ===== #
        # 当 args.diffusion 存在时, Pipeline 变成 x0-预测的条件扩散模型:
        #   训练: 把 target_x (encode_g1 出的 189-D x0) 加噪到随机步 t, 网络去噪预测 x0;
        #         simple_loss 与所有 FK loss 照旧作用在预测的 x0 上 (与回归版同形).
        #   推理: DDIM 迭代采样得到 x0, 再走原 decode + FK 路径.
        # 不设置时退回原回归路径 (denoiser3d 直接回归 pred_x), 旧 ckpt/实验零改动可用.
        diff_cfg = getattr(self.args, "diffusion", None)
        self.diffusion_enabled = diff_cfg is not None
        if self.diffusion_enabled:
            from hmr4d.utils.diffusion.model_util import create_gaussian_diffusion
            from hmr4d.utils.diffusion.resample import create_named_schedule_sampler

            self.diffusion_cfg = diff_cfg
            self.train_diffusion = create_gaussian_diffusion(diff_cfg, training=True)
            self.test_diffusion = create_gaussian_diffusion(diff_cfg, training=False)
            self.schedule_sampler = create_named_schedule_sampler(
                getattr(diff_cfg, "schedule_sampler_type", "uniform"), self.train_diffusion
            )
            self.ddim_eta = float(getattr(diff_cfg, "ddim_eta", 0.0))
            # CFG: train with sample-level obs-drop (prob cfg_uncond_prob) to teach
            # the unconditional branch; at inference guide with guidance_param
            # (scale=1.0 -> CFG off). ∅ = drop obs (the 2D-pose condition).
            self.cfg_scale = float(getattr(diff_cfg, "guidance_param", 1.0))
            self.cfg_uncond_prob = float(getattr(diff_cfg, "cfg_uncond_prob", 0.1))
            # xt_dim 必须 == 189 (网络拼接噪声 motion 用)
            if hasattr(self.denoiser3d, "xt_dim") and int(self.denoiser3d.xt_dim) != G1_PRED_X_DIM:
                raise ValueError(
                    f"[Pipeline] Denoiser xt_dim ({int(self.denoiser3d.xt_dim)}) != "
                    f"G1_PRED_X_DIM ({G1_PRED_X_DIM}). Set network.xt_dim={G1_PRED_X_DIM}."
                )
            Log.info(
                f"[Pipeline] Diffusion ENABLED: train_steps={self.train_diffusion.num_timesteps}, "
                f"test_steps={self.test_diffusion.num_timesteps}, sampler="
                f"{getattr(diff_cfg, 'schedule_sampler_type', 'uniform')}, ddim_eta={self.ddim_eta}"
            )
        else:
            Log.info("[Pipeline] Diffusion disabled -> regression path.")

        pred_cam_stats_file = getattr(self.args, "pred_cam_stats_file", None)
        if pred_cam_stats_file is not None and hasattr(self.denoiser3d, "set_pred_cam_stats"):
            # 加载失败必须 raise, 不能静默回退 —— 否则会用 network 里注册的 *人类* 默认
            # pred_cam stats ([1.0606,-0.0027,0.2702]), 机器人 pred_cam 尺度全错且无声.
            try:
                pred_cam_stats = np.load(pred_cam_stats_file)
                self.denoiser3d.set_pred_cam_stats(pred_cam_stats["mean"], pred_cam_stats["std"])
                Log.info(f"[Pipeline] Loaded robot pred_cam stats from {pred_cam_stats_file}")
            except Exception as e:
                raise RuntimeError(
                    f"[Pipeline] Failed to load pred_cam stats from {pred_cam_stats_file}: {e}. "
                    f"拒绝静默回退到人类默认 pred_cam stats. 请用 tools/make_robot_pred_cam_stats.py "
                    f"重新生成, 或把 args.pred_cam_stats_file 置 null 显式表示不归一化."
                ) from e
        elif hasattr(self.denoiser3d, "set_pred_cam_stats"):
            Log.warning(
                "[Pipeline] args.pred_cam_stats_file 未设置 —— pred_cam 头将使用 network 里"
                "注册的人类默认 stats, 机器人尺度可能不对. 建议生成并配置 g1_pred_cam_stats.npz."
            )

        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)

    # ========== Training ========== #

    def forward(self, inputs, train=False, postproc=False, static_cam=False, null_condition=None):
        """
        Args:
            train: bool — if True, run loss-computation path; if False, run inference.
            null_condition: bool or None — whether to randomly null-mask conditions
                (10% per frame, original GVHMR augmentation). When None (default),
                follows `train` (legacy behaviour). Set to False during val-loss
                computation so the loss is deterministic.
        """
        length = inputs["length"]  # (B,)
        if null_condition is None:
            null_condition = train

        # ------------------------------------------------------------------
        # 1. 条件特征
        # ------------------------------------------------------------------
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])

        f_cam_angvel = inputs.get("cam_angvel", None)
        if f_cam_angvel is not None and getattr(self.args, "normalize_cam_angvel", False):
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std.clamp(min=1e-8)

        f_imgseq = inputs.get("f_imgseq", None)
        f_imgseq_mask = None
        if f_imgseq is not None and "mask" in inputs and isinstance(inputs["mask"], dict):
            f_imgseq_mask = inputs["mask"].get("f_imgseq", None)

        f_condition = {
            "obs": inputs["obs"],
            "f_cliffcam": cliff_cam,
            "f_cam_angvel": f_cam_angvel,
            "f_imgseq": f_imgseq,
            "f_imgseq_mask": f_imgseq_mask,
        }

        # CFG: 训练时整段 drop obs 的样本 (cfg_uncond_prob) 学无条件分支 (∅ = 去掉 2D 姿态).
        # 先于 per-frame null aug 算出 drop_obs, 并把这些样本从 per-frame aug 中跳过
        # (skip_mask), 使无条件分支语义纯净 —— 只缺 obs, cliffcam/cam_angvel 保持干净 (Q1).
        B0 = length.shape[0]
        drop_obs = None
        if self.diffusion_enabled and train and getattr(self, "cfg_uncond_prob", 0.0) > 0:
            drop_obs = torch.rand(B0, device=length.device) < self.cfg_uncond_prob

        if null_condition:
            f_condition = self._randomly_set_null_condition(
                f_condition, uncond_prob=0.1, skip_mask=drop_obs
            )

        # ------------------------------------------------------------------
        # 2. 网络前向 (pred_x: B,L,189 归一化 G1-GVHMR 格式)
        # ------------------------------------------------------------------
        # target_x (x0) 在扩散训练分支里提前算好, 供 q_sample 与 simple_loss 复用;
        # 其余分支保持 None, simple_loss 处再 encode. t_weights: 重要性采样权重 (uniform 恒 1).
        # FK 几何 loss 不按 timestep 门控; 所有 diffusion step 都参与 FK / j2d / transl 监督.
        target_x = None
        t_weights = None
        if self.diffusion_enabled and train:
            # --- 扩散训练: 加噪 target_x 到随机步 t, 网络去噪预测 x0 ---
            f_cond = self.cond_embedder(**f_condition, drop_obs=drop_obs)  # (B,L,latent)
            target_x = self.endecoder.encode_g1(inputs)          # (B, L, 189) x_start
            t, t_weights = self.schedule_sampler.sample(target_x.shape[0], target_x.device)
            noise = torch.randn_like(target_x)
            x_t = self.train_diffusion.q_sample(target_x, t, noise=noise)
            model_output = self.denoiser3d(
                f_cond=f_cond,
                length=length,
                xt=x_t,
                timesteps=self.train_diffusion._scale_timesteps(t),
            )
        elif self.diffusion_enabled and not train:
            # --- 扩散推理: DDIM 迭代采样得到 x0 (内部建 f_cond / f_uncond 做 CFG) ---
            model_output = self._ddim_sample(length, f_condition)
        else:
            # --- 回归路径 (无扩散), 与原版完全一致 (只是 embedding 搬出去了) ---
            f_cond = self.cond_embedder(**f_condition, drop_obs=drop_obs)
            model_output = self.denoiser3d(f_cond=f_cond, length=length)
        pred_x = model_output["pred_x"]   # (B, L, 189)

        # ------------------------------------------------------------------
        # 3. 推理模式
        # ------------------------------------------------------------------
        if not train:
            decode_dict = self.endecoder.decode_g1_new(pred_x)
            out = {"model_output": model_output, "decode_dict": decode_dict}
            out["static_conf_logits"] = model_output.get("static_conf_logits")

            # FK to recover body positions if engine is available
            if hasattr(self.endecoder, "_g1_humanoid"):
                gt = inputs.get("g1_target", {})
                root_pos_0 = gt.get("g1_body_pos_w", None)
                if root_pos_0 is not None:
                    root_pos_0 = root_pos_0[:, [0], 0]  # (B,1,3)
                else:
                    # Demo / 真实视频推理时无 GT. floor_adjust 后训练用的起点是
                    # (0, ~0.79, 0), 这里 fallback 用 default_pelvis_height (y-up)
                    # 防止 pelvis 被"埋在地下". 可在 yaml args.robot.default_pelvis_height 覆盖.
                    pelvis_h = float(getattr(
                        getattr(self.args, "robot", None), "default_pelvis_height", 0.793
                    ))
                    B_pred = decode_dict["local_transl_vel"].shape[0]
                    device = decode_dict["local_transl_vel"].device
                    root_pos_0 = torch.tensor(
                        [[0.0, pelvis_h, 0.0]], device=device
                    ).expand(B_pred, 1, 3)

                # Recover world-frame root orientation using GV + cam_angvel,
                # identical to original GVHMR's inference path.
                # Under precision=16-mixed, decode_dict tensors are Half;
                # cast everything to fp32 before the rotation math.
                go_c_rotmat_f32      = decode_dict["global_orient_rotmat"].float()
                local_transl_vel_f32 = decode_dict["local_transl_vel"].float()
                root_pos_0_f32       = root_pos_0.float()

                go_c_aa_f32  = matrix_to_axis_angle(go_c_rotmat_f32)
                go_gv_aa_f32 = matrix_to_axis_angle(
                    rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float())
                )
                params_w = get_smpl_params_w_Rt_v2(
                    global_orient_gv  = go_gv_aa_f32,
                    local_transl_vel  = local_transl_vel_f32,
                    global_orient_c   = go_c_aa_f32,
                    cam_angvel        = inputs["cam_angvel"].float(),
                )
                # get_smpl_params_w_Rt_v2 outputs y-up (ay, SMPL convention).
                # 整套系统已统一 y-up, 不再左乘 R_post 转 z-up.
                pred_root_R_w    = axis_angle_to_matrix(params_w["global_orient"])  # (B,L,3,3) y-up
                pred_root_aa_w   = params_w["global_orient"]                        # (B,L,3)   y-up
                pred_root_quat_w = matrix_to_quaternion(pred_root_R_w)              # (B,L,4)   wxyz

                # Translation rollout in y-up using pred_root_aa_w,
                # anchored at root_pos_0 (y-up) from g1_target.
                pred_root_pos = rollout_local_transl_vel(
                    local_transl_vel_f32, pred_root_aa_w, root_pos_0_f32
                )
                # Recover scalar θ in BYD order from [0,θ,0]-encoded rotmat (mirrors dataset.py).
                # 与 training-path (~line 295) 同样用解析提取, 与训练时保持完全一致.
                R_bp_inf = decode_dict["body_pose_rotmat"].float()                 # (B,L,29,3,3)
                pred_dof_byd = torch.atan2(R_bp_inf[..., 0, 2], R_bp_inf[..., 0, 0])  # (B,L,29)
                pred_joints_w, _ = self.endecoder.fk_g1(
                    pred_dof_byd,
                    pred_root_pos,
                    pred_root_quat_w,
                    world_up='y',
                )
                out["pred_joints_w"] = pred_joints_w
                out["pred_root_pos"] = pred_root_pos
                # Task 3: expose G1-specific outputs so val_step / metric callbacks
                # can consume them without re-deriving from FK internals.
                # NOTE: per-body quaternions (30 bodies × 4) would need extra FK
                # work; skip until a metric actually needs them.
                out["pred_g1_joint_pos"]  = pred_dof_byd     # (B,L,29) scalar DOF (BYD order)
                out["pred_g1_body_pos_w"] = pred_joints_w    # (B,L,30,3)
                out["pred_root_quat_w"]   = pred_root_quat_w # (B,L,4) wxyz, world frame
            return out

        # ------------------------------------------------------------------
        # 4. 训练模式
        # ------------------------------------------------------------------
        outputs = {"model_output": model_output}
        # t_weights: diffusion timestep 重要性加权; uniform sampler 下恒 1.
        outputs["t_weights"] = t_weights
        total_loss = 0
        mask = inputs["mask"]["valid"]   # (B, L)

        gt = inputs["g1_target"]
        B, L = mask.shape

        # --- 4a. Simple loss: MSE on 189-D normalised pred_x ---
        # 扩散训练时 target_x 已在 section 2 算好 (即 x0); 回归时这里 encode.
        # x0-预测扩散下, 该 MSE 就是 diffusion 的 x_start 重建 loss (与回归版同形).
        # 扩散重建项 *不* 门控 (所有 t 都监督 x0); 只做 t_weights 加权 (Q2).
        if target_x is None:
            target_x = self.endecoder.encode_g1(inputs)  # (B, L, 189)
        simple_loss_el = F.mse_loss(pred_x, target_x, reduction="none")
        simple_loss, simple_loss_nan_count = _reduce_loss(
            simple_loss_el, mask[..., None], t_weights=t_weights, fk_gate=None
        )
        total_loss += simple_loss
        outputs["simple_loss"] = simple_loss
        outputs["simple_loss_nan_count"] = simple_loss_nan_count

        # --- 4b. Decode predictions & prepare FK inputs ---
        # 不做 FK timestep 门控: 所有样本的 FK 几何链都保留梯度.
        decode_dict = self.endecoder.decode_g1_new(pred_x)
        outputs["decode_dict"] = decode_dict

        # GT body positions and orientations (WORLD frame)
        gt_root_wxyz  = gt["g1_body_quat_w"][:, :, 0]      # (B,L,4) wxyz
        gt_root_R_w   = quaternion_to_matrix(gt_root_wxyz)  # (B,L,3,3) world
        gt_root_pos   = gt["g1_body_pos_w"][:, :, 0]        # (B,L,3)

        # GT joint rotation matrices (from [0,θ,0] encoding, same convention as encode_g1)
        gt_bp_r6d    = self.endecoder._dof_to_r6d(gt["g1_joint_pos"])         # (B,L,174)
        gt_bp_rotmat = rotation_6d_to_matrix(gt_bp_r6d.reshape(B, L, 29, 6))  # (B,L,29,3,3)

        # Recover world-frame root orientation using global_orient_gv + cam_angvel,
        # IDENTICAL to inference path (lines 167-191). Previously the training
        # path used the shortcut `R_w = R_w2c.T @ R_c`, which gives the model a
        # perfect-T_w2c crutch — `global_orient_gv_r6d` never enters the
        # supervision graph and is therefore unusable at inference time when
        # T_w2c is not available. Unifying both paths forces gv to be learned.
        # bf16-mixed: cast to fp32 first to avoid acos NaN at singular rotations.
        go_c_rotmat_f32      = decode_dict["global_orient_rotmat"].float()
        local_transl_vel_f32 = decode_dict["local_transl_vel"].float()
        go_c_aa_f32  = matrix_to_axis_angle(go_c_rotmat_f32)
        go_gv_aa_f32 = matrix_to_axis_angle(
            rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float())
        )
        params_w = get_smpl_params_w_Rt_v2(
            global_orient_gv = go_gv_aa_f32,
            local_transl_vel = local_transl_vel_f32,
            global_orient_c  = go_c_aa_f32,
            cam_angvel       = inputs["cam_angvel"].float(),
        )
        pred_root_aa_w = params_w["global_orient"]                        # (B,L,3)   y-up
        pred_root_R_w  = axis_angle_to_matrix(pred_root_aa_w)             # (B,L,3,3) y-up

        # Predicted root position (rollout in world frame from GT start)
        pred_root_pos = rollout_local_transl_vel(
            local_transl_vel_f32, pred_root_aa_w, gt_root_pos[:, [0]].float()
        )  # (B,L,3)
        outputs["pred_root_pos"] = pred_root_pos

        # --- 4c. FK-based 3D joint loss ---
        if hasattr(self.endecoder, "_g1_humanoid"):
            # GT: scalar joint_pos already BYD order; gt_root_wxyz already wxyz quat
            gt_joints_w, _ = self.endecoder.fk_g1(gt["g1_joint_pos"], gt_root_pos, gt_root_wxyz, world_up='y')

            # Pred: recover scalar θ (BYD) from [0,θ,0]-encoded rotmat; rotmat → wxyz quat.
            # 注: body_pose r6d 用 Y-axis-only 编码 (encode_g1: aa[...,1]=θ), body_pose_rotmat
            # 一定是 Ry(θ) 形式. matrix_to_axis_angle 在 identity 处反向 1/sin(θ)→∞,
            # 即使下面 finite-mask 屏蔽前向 NaN, 反传 0×∞ 仍 NaN, 污染整网 grad
            # (tools/diag_grad_isolate_loss.py: cr_j3d 开启 → 39M NaN grad). 改用解析提取.
            R_bp = decode_dict["body_pose_rotmat"].float()                # (B,L,29,3,3)
            pred_dof_byd = torch.atan2(R_bp[..., 0, 2], R_bp[..., 0, 0])  # (B,L,29)
            pred_root_quat_w = matrix_to_quaternion(pred_root_R_w)
            pred_joints_w, _ = self.endecoder.fk_g1(pred_dof_byd, pred_root_pos, pred_root_quat_w, world_up='y')

            # (已删除 NaN/Inf 兜底: 不再用 GT 替换坏关节, pred_joints_w 原样保留)
            outputs["gt_joints_w"]   = gt_joints_w    # (B,L,30,3)
            outputs["pred_joints_w"] = pred_joints_w

            # --- incam pose for j2d/transl_c (ORIGINAL GVHMR semantics) ---
            # 原版 j2d 用的相机系绝对关节 = FK(global_orient=相机系直接解码朝向,
            # body_pose) + transl=compute_transl_full_cam(pred_cam). 这里复刻:
            #   1) 相机系直接解码朝向 root_R_c (decode_dict, encode_g1 中 R_c=R_w2c@R_w)
            #   2) 经 R_c2w 还原成 world 朝向 (shortcut, 仅供 j2d 这一相机系 loss 用,
            #      不进 world 输出/transl_w 的 gv 学习), FK 出 root-relative 世界关节
            #   3) 用 R_w2c_g1 旋回相机系 → 净相机系 root 朝向严格 == root_R_c.
            # 平移在 j2d block 里加 pred_cam (需 robot_bbx_xys, 在 _prepare 后才有).
            R_bp_inc        = decode_dict["body_pose_rotmat"].float()       # (B,L,29,3,3)
            incam_dof_byd   = torch.atan2(R_bp_inc[..., 0, 2], R_bp_inc[..., 0, 0])
            root_R_c_direct = decode_dict["global_orient_rotmat"].float()   # (B,L,3,3) 相机系
            R_w2c_g1        = inputs["g1_target"]["g1_T_w2c"][..., :3, :3].float()
            root_R_w_short  = R_w2c_g1.transpose(-1, -2) @ root_R_c_direct  # (B,L,3,3) world
            root_quat_w_sh  = matrix_to_quaternion(root_R_w_short)
            zeros_root      = torch.zeros_like(pred_root_pos)
            incam_joints_w, _ = self.endecoder.fk_g1(
                incam_dof_byd, zeros_root, root_quat_w_sh, world_up='y'
            )                                                               # (B,L,30,3)
            incam_joints_w_rr = incam_joints_w - incam_joints_w[:, :, :1]   # root-relative
            incam_pred_c_rootrel = torch.einsum(
                "blij,blnj->blni", R_w2c_g1, incam_joints_w_rr
            )                                                               # (B,L,30,3) 相机系, root 在原点
            # (已删除 NaN/Inf 兜底: 不再用 GT 替换坏关节, incam_pred_c_rootrel 原样保留)
            outputs["incam_pred_c_rootrel"] = incam_pred_c_rootrel

            g1_loss, g1_dict = compute_extra_g1_loss(inputs, outputs, self)
            total_loss += g1_loss
            outputs.update(g1_dict)

        # --- 4d. Incam / global losses (position-based, using FK body_pos_w) ---
        if hasattr(self.endecoder, "_g1_humanoid"):
            pred_body_pos_w = outputs["pred_joints_w"]   # (B,L,30,3)
            gt_body_pos_w   = outputs["gt_joints_w"]
        else:
            # Fallback: use GT body positions when FK is not available
            pred_body_pos_w = gt["g1_body_pos_w"]
            gt_body_pos_w   = gt["g1_body_pos_w"]

        outputs.update(self._prepare_robot_outputs_from_fk(
            inputs, pred_body_pos_w, gt_body_pos_w
        ))

        for extra_func in [compute_extra_incam_loss, compute_extra_global_loss]:
            extra_loss, extra_loss_dict = extra_func(inputs, outputs, self)
            total_loss += extra_loss
            outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss
        return outputs

    def _prepare_robot_outputs_from_fk(self, inputs, pred_body_pos_w, gt_body_pos_w):
        """Build camera-frame tensors from world-frame FK joint positions.

        使用 g1_T_w2c (初始位置与 AMASS 相机相同, 后续帧位移为 AMASS 的 0.9×) 投影,
        与 G1 训练相机完全一致.  g1_T_w2c.R == T_w2c.R (旋转相同),
        只有 translation 差 0.9× delta, 因此 encode_g1 中的旋转编码不受影响.
        """
        root_body_id = getattr(getattr(self.args, "robot", None), "root_body_id", 0)

        g1_T_w2c = inputs["g1_target"]["g1_T_w2c"]   # (B,L,4,4) G1 camera
        pred_body_pos_c = transform_points_w2c(pred_body_pos_w, g1_T_w2c)
        gt_body_pos_c   = transform_points_w2c(gt_body_pos_w,   g1_T_w2c)

        # (已删除 NaN/Inf 兜底: 不再用 GT 替换坏关节, pred_body_pos_c 原样保留)

        # robot_bbx 不喂给网络 (网络只见人体 bbx via cliff_cam), 它只用于 transl_c/j2d 的
        # GT 目标. 因此对它做 do_augment 等于往 pred_cam 目标里注入网络观测不到的随机噪声
        # (与原版"对网络可见的输入 bbx 做增强"语义相反) → 关掉, 让目标是输入的确定函数.
        robot_bbx_xys = compute_robot_bbx_xys(
            gt_body_pos_c, inputs["K_fullimg"], do_augment=False,
        )

        return {
            "pred_body_pos_c": pred_body_pos_c,
            "gt_body_pos_c":   gt_body_pos_c,
            "pred_root_pos_w": pred_body_pos_w[:, :, root_body_id],
            "gt_root_pos_w":   gt_body_pos_w[:,  :, root_body_id],
            "pred_root_pos_c": pred_body_pos_c[:, :, root_body_id],
            "gt_root_pos_c":   gt_body_pos_c[:,  :, root_body_id],
            "robot_bbx_xys":   robot_bbx_xys,
        }

    # kept for backward compat
    def _prepare_robot_outputs(self, inputs, pred_g1):
        gt = inputs["g1_target"]
        return self._prepare_robot_outputs_from_fk(
            inputs, pred_g1["body_pos_w"], gt["g1_body_pos_w"]
        )

    def _randomly_set_null_condition(self, f_condition, uncond_prob=0.1, skip_mask=None):
        """Per-frame 10% null aug. skip_mask: (B,) bool — 这些样本整体跳过 aug
        (用于 CFG 的 drop_obs 样本, 保持其无条件分支语义纯净, 只缺 obs)."""
        keys = list(f_condition.keys())
        for k in keys:
            if f_condition[k] is None:
                continue
            if k.endswith("_mask"):
                continue
            f_condition[k] = f_condition[k].clone()
            # NOTE: device= 必须传, 否则 mask 在 CPU、tensor 在 CUDA, 隐式 H2D 拷贝/索引报错.
            mask = torch.rand(f_condition[k].shape[:2], device=f_condition[k].device) < uncond_prob
            if skip_mask is not None:
                mask[skip_mask] = False  # CFG uncond 样本不参与 per-frame aug
            f_condition[k][mask] = 0.0
            if k == "f_imgseq" and f_condition.get("f_imgseq_mask", None) is not None:
                img_mask = f_condition["f_imgseq_mask"]
                if img_mask.ndim == 1:
                    img_mask = img_mask[:, None].expand(mask.shape[0], mask.shape[1]).clone()
                else:
                    img_mask = img_mask.clone()
                img_mask[mask] = False
                f_condition["f_imgseq_mask"] = img_mask
        return f_condition

    def _ddim_sample(self, length, f_condition):
        """扩散推理: 从纯高斯噪声出发, DDIM 迭代去噪得到 x0 (B, L, 189).

        条件 embedding (f_cond) 在循环外 *只算一次*, 通过闭包复用于全部 DDIM 步 (GEM-X
        式), 不放进 model_kwargs —— 避免 SpacedDiffusion 的 _WrappedModel 把它当 timestep
        重映射. gaussian_diffusion.p_mean_variance 从网络返回 dict 取 pred_x_start, 并把
        pred_cam / static_conf_logits 等 aux 透传到结果 dict.

        CFG (classifier-free guidance): 当 cfg_scale != 1.0 时, 每步额外跑一次无条件分支
        (f_uncond = drop obs), 在 x0 空间外推:
            x0_guided = x0_uncond + scale * (x0_cond - x0_uncond)
        x0-预测模型下, 对 x0 线性外推 == 对 eps 外推 (二者在给定 x_t,t 时仿射相关), 与标准
        eps-CFG 等价.
        """
        from hmr4d.model.gvhmr.utils.endecoder import G1_PRED_X_DIM

        obs = f_condition["obs"]
        B, L = obs.shape[0], obs.shape[1]
        shape = (B, L, G1_PRED_X_DIM)

        # 条件 embedding 只算一次, 全程复用
        f_cond = self.cond_embedder(**f_condition, drop_obs=None)
        scale = float(getattr(self, "cfg_scale", 1.0))
        f_uncond = None
        if scale != 1.0:
            drop_all = torch.ones(B, dtype=torch.bool, device=obs.device)  # ∅ = drop obs
            f_uncond = self.cond_embedder(**f_condition, drop_obs=drop_all)

        def _cfg(a_c, a_u):
            # CFG 外推; 任一分支为 None 则退回条件分支.
            if a_c is None or a_u is None:
                return a_c
            return a_u + scale * (a_c - a_u)

        def model_fn(x, ts, **kwargs):
            out_c = self.denoiser3d(f_cond=f_cond, length=length, xt=x, timesteps=ts)
            if f_uncond is None:
                return out_c
            out_u = self.denoiser3d(f_cond=f_uncond, length=length, xt=x, timesteps=ts)
            out_c = dict(out_c)
            # Q4: pred_x_start / pred_cam / static_conf_logits 全部 CFG 外推 (都依赖条件).
            guided = _cfg(out_c["pred_x_start"], out_u["pred_x_start"])
            out_c["pred_x_start"] = guided
            out_c["pred_x"] = guided
            out_c["pred_cam"] = _cfg(out_c.get("pred_cam"), out_u.get("pred_cam"))
            out_c["static_conf_logits"] = _cfg(
                out_c.get("static_conf_logits"), out_u.get("static_conf_logits")
            )
            return out_c

        # PL 的测试流程会把全局 CUDA RNG 重置到固定状态，导致每次 DDIM 初始噪声
        # 都一样，seed= 参数失效。解决方案：用 CPU RNG（pl.seed_everything 能正确
        # 控制它）拉出一个种子，给本地 CUDA Generator，完全绕开全局 CUDA RNG。
        # 每次调用 _ddim_sample 都从 CPU RNG 消费一个值，多 chunk 时噪声自然不同。
        _cpu_noise_seed = torch.randint(0, 2**31, (1,)).item()  # advance CPU RNG
        _gen = torch.Generator(device=obs.device).manual_seed(_cpu_noise_seed)
        init_noise = torch.randn(*shape, device=obs.device, generator=_gen)

        out = self.test_diffusion.ddim_sample_loop_with_aux(
            model_fn,
            shape,
            noise=init_noise,            # 用 CPU-seed 生成的噪声, 不受 CUDA RNG 干扰
            clip_denoised=False,         # 189-D 已归一化, 非 [-1,1] 图像, 不能裁剪
            eta=self.ddim_eta,
            model_kwargs={"y": {}},      # ddim_sample 会往 model_kwargs["y"] 写中间量
            device=obs.device,
            progress=False,
        )
        return {
            "pred_x": out["pred_xstart"],
            "pred_x_start": out["pred_xstart"],
            "pred_cam": out.get("pred_cam"),
            "static_conf_logits": out.get("static_conf_logits"),
            "pred_context": out.get("pred_context"),
        }


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2], device=f_condition[k].device) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def _reduce_loss(loss_el, mask_bcast, t_weights=None, fk_gate=None):
    """统一的 loss 归约: 返回 (标量 loss, nan_count).

    - nan_count: loss_el 里 NaN/Inf 元素总数 (含被 fk_gate 关掉的样本), 纯测量、始终统计,
      记录到 wandb 供可见性 (Q3 要求门控后仍统计 NaN).
    - fk_gate: (B,) bool, True=保留该样本的 FK 几何 loss. 大噪声步 (False) 的样本用
      torch.where 把 loss 置 0 *并* 从分母剔除 —— 避免其 NaN 经 NaN*0 污染梯度 (这是 *门控*,
      非 pred←GT 兜底; 被保留样本里的真实 NaN 仍原样暴露进 loss).
    - t_weights: (B,) 重要性采样权重, 先按样本算 masked-mean 再乘权重 (Q2). uniform 下恒 1.
    None/None 时退回原始全局 masked-mean, 与回归版完全一致.
    """
    nan_count = (~torch.isfinite(loss_el)).sum().float()

    if fk_gate is None and t_weights is None:
        return (loss_el * mask_bcast).mean(), nan_count

    B = loss_el.shape[0]
    loss_el2 = loss_el
    if fk_gate is not None:
        keep = fk_gate.view(B, *([1] * (loss_el.dim() - 1)))
        # 门控样本的 loss 置 0 (不是换 GT), 使其 NaN 不经 0×NaN 反传污染.
        loss_el2 = torch.where(keep.expand_as(loss_el), loss_el, torch.zeros_like(loss_el))

    w = mask_bcast.expand_as(loss_el2).float()
    per_sample = (loss_el2 * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1)  # (B,)

    sample_w = t_weights if t_weights is not None else loss_el.new_ones(B)
    if fk_gate is not None:
        sample_w = sample_w * fk_gate.float()  # 门控样本权重 0 (其 per_sample 已为有限值 0)
    return (per_sample * sample_w).mean(), nan_count


def compute_extra_incam_loss(inputs, outputs, ppl):
    weights = ppl.weights

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")
    model_output = outputs["model_output"]
    pred_body_pos_c = outputs["pred_body_pos_c"]
    gt_body_pos_c = outputs["gt_body_pos_c"]
    pred_root_pos_c = outputs["pred_root_pos_c"]
    gt_root_pos_c = outputs["gt_root_pos_c"]
    robot_bbx_xys = outputs["robot_bbx_xys"]
    mask_reproj = ~inputs["mask"].get("spv_incam_only", torch.zeros_like(mask, dtype=torch.bool))

    # 相机系 root-aligned 关节差分. 原版 cr_j3d 与 j2d 共用同一份 pred_c_j3d
    # (相机系直接解码朝向 fk_v2). 这里 pred 也用 incam_pred_c_rootrel (直接相机朝向,
    # forward 里算好) 与 j2d 保持一致; 没有时回退 world-FK 的 pred_body_pos_c.
    # GT 用实际 GT 相机系关节 (gt_body_pos_c), 与原版 gt_c_j3d 对齐.
    incam_rootrel = outputs.get("incam_pred_c_rootrel")
    if incam_rootrel is not None:
        pred_cr_j3d = incam_rootrel - incam_rootrel[:, :, :1]
    else:
        pred_cr_j3d = pred_body_pos_c - pred_body_pos_c[:, :, :1]
    gt_cr_j3d   = gt_body_pos_c   - gt_body_pos_c[:, :, :1]

    if weights.get("cr_j3d", 0.0) > 0:
        cr_j3d_loss_el = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss, cr_j3d_loss_nan_count = _reduce_loss(
            cr_j3d_loss_el, mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += cr_j3d_loss * weights["cr_j3d"]
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss
        extra_loss_dict["cr_j3d_loss_nan_count"] = cr_j3d_loss_nan_count

    if weights.get("transl_c", 0.0) > 0 and model_output.get("pred_cam") is not None:
        pred_cam = model_output["pred_cam"]
        gt_pred_cam = get_a_pred_cam(gt_root_pos_c, robot_bbx_xys, inputs["K_fullimg"])
        gt_pred_cam[gt_pred_cam.isinf()] = -1

        gt_body_z_min = gt_body_pos_c[..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_body_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (robot_bbx_xys[..., 2] > 0)
        )[..., None]
        transl_c_loss_el = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
        transl_c_loss, transl_c_loss_nan_count = _reduce_loss(
            transl_c_loss_el, mask[..., None] * valid_mask, t_weights, fk_gate
        )
        extra_loss += transl_c_loss * weights["transl_c"]
        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss_dict["transl_c_loss_nan_count"] = transl_c_loss_nan_count

    if weights.get("j2d", 0.0) > 0:
        reproj_z_thr = 0.3

        # 原版语义: pred 相机系绝对关节 = 直接解码相机朝向的 root-relative 姿态
        # (forward 里 outputs["incam_pred_c_rootrel"] 算好) + pred_cam 反解的相机系平移.
        # 这样 j2d 梯度流入 pred_cam(经 compute_transl_full_cam)+ body_pose + 相机系
        # global_orient, 与原版 fk_v2(**pred_smpl_params_incam) 完全一致.
        # get_a_pred_cam(transl_c) 与 compute_transl_full_cam 互逆, 二者构成闭环.
        incam_rootrel = outputs.get("incam_pred_c_rootrel")
        pred_cam = model_output.get("pred_cam")
        if incam_rootrel is not None and pred_cam is not None:
            transl_c_cam = compute_transl_full_cam(pred_cam, robot_bbx_xys, inputs["K_fullimg"])
            pred_body_pos_c_j2d = incam_rootrel + transl_c_cam[:, :, None]   # (B,L,30,3)
        else:
            # fallback (无 FK / 无 pred_cam): 退回 world-FK 投影
            pred_body_pos_c_j2d = pred_body_pos_c

        # z 幅度下限 (回退到初始 GVHMR 行为): |z| <= reproj_z_thr 的点整体置为 +reproj_z_thr,
        # 不保留符号. clone 避免就地修改 outputs 中共享的 pred/gt_body_pos_c.
        pred_body_pos_c_j2d = pred_body_pos_c_j2d.clone()
        gt_body_pos_c_j2d = gt_body_pos_c.clone()
        pred_body_pos_c_j2d[pred_body_pos_c_j2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr
        gt_body_pos_c_j2d[gt_body_pos_c_j2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_body_pos_c_j2d, robot_bbx_xys, inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_body_pos_c_j2d, robot_bbx_xys, inputs["K_fullimg"])
        # 和之前的 gvhmr1 一致: 对投影做 clamp(-2, 3) 限幅 (原版算法行为, 非 NaN 兜底;
        # clamp(NaN)=NaN 不会隐藏 NaN). 已删除此前的 nan_to_num(投影) 保护.
        pred_j2d_01 = pred_j2d_01.clamp(-2.0, 3.0)
        gt_j2d_01   = gt_j2d_01.clamp(-2.0, 3.0)

        valid_mask = (
            (gt_body_pos_c_j2d[..., 2] > reproj_z_thr)
            * (pred_body_pos_c_j2d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False
        j2d_loss_el = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        j2d_loss, j2d_loss_nan_count = _reduce_loss(
            j2d_loss_el, mask[..., None, None] * valid_mask, t_weights, fk_gate
        )
        extra_loss += j2d_loss * weights["j2d"]
        extra_loss_dict["j2d_loss"] = j2d_loss
        extra_loss_dict["j2d_loss_nan_count"] = j2d_loss_nan_count

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")
    gt = inputs["g1_target"]
    gt_root_pos_w = outputs["gt_root_pos_w"]

    if weights.get("transl_w", 0.0) > 0:
        # 原版 GVHMR 语义: rollout 用 *GT 世界朝向* + GT 起点, 预测量只有 local_transl_vel
        # (原版 compute_extra_global_loss: rollout(local_transl_vel, gt_global_orient_w,
        #  gt_transl_w[:, [0]])). Gradient 只流到 local_transl_vel, 朝向借 GT 当参考,
        # 避免训练初期 gv/cam_angvel 没学好时位移监督被错误朝向带偏.
        gt_root_wxyz = gt["g1_body_quat_w"][:, :, 0]                              # (B,L,4) wxyz
        gt_root_aa_w = matrix_to_axis_angle(quaternion_to_matrix(gt_root_wxyz).float())  # (B,L,3) world
        local_transl_vel = outputs["decode_dict"]["local_transl_vel"].float()
        pred_transl_w = rollout_local_transl_vel(
            local_transl_vel, gt_root_aa_w, gt_root_pos_w[:, [0]].float()
        )
        transl_w_loss_el = F.l1_loss(pred_transl_w, gt_root_pos_w, reduction="none")
        transl_w_loss, transl_w_loss_nan_count = _reduce_loss(
            transl_w_loss_el, mask[..., None], t_weights, fk_gate
        )
        extra_loss += transl_w_loss * weights["transl_w"]
        extra_loss_dict["transl_w_loss"] = transl_w_loss
        extra_loss_dict["transl_w_loss_nan_count"] = transl_w_loss_nan_count

    if weights.get("static_conf_bce", 0.0) > 0 and outputs["model_output"].get("static_conf_logits") is not None:
        # `vel_thr` is the original-GVHMR HuMoR threshold (0.15 m/s). Per-body
        # threshold = vel_thr × vel_thr_mul[i]; default mul = 1.0 for every body.
        # Both `g1_body_lin_vel_w` and the threshold are in m/s, so the rule is
        # invariant to motion_frames / speed-augmentation playback fps.
        vel_thr = args.static_conf.vel_thr
        body_ids = getattr(args.static_conf, "body_ids", None)
        if body_ids is None:
            body_ids = list(range(gt["g1_body_pos_w"].shape[2]))
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]
        if pred_static_conf_logits.shape[-1] != len(body_ids):
            pred_static_conf_logits = pred_static_conf_logits[:, :, body_ids]

        # per-body threshold tensor, shape (J,)
        vel_thr_mul = getattr(args.static_conf, "vel_thr_mul", None)
        device = pred_static_conf_logits.device
        if vel_thr_mul is None:
            vel_thr_per_body = torch.full(
                (len(body_ids),), float(vel_thr), device=device, dtype=torch.float32,
            )
        else:
            assert len(vel_thr_mul) == len(body_ids), (
                f"vel_thr_mul ({len(vel_thr_mul)}) must match body_ids ({len(body_ids)})"
            )
            vel_thr_per_body = (
                torch.tensor(list(vel_thr_mul), device=device, dtype=torch.float32) * float(vel_thr)
            )

        if "g1_body_lin_vel_w" in gt:
            # m/s: |v| < per-body threshold. NOTE: dataset 内 _finite_diff_lin_vel
            # 末帧速度是倒数第 2 帧的复制 → 末帧 label 不可信, 用 vel_mask 屏蔽.
            speed = gt["g1_body_lin_vel_w"][:, :, body_ids].norm(dim=-1)            # (B, L, J)
            static_gt = speed < vel_thr_per_body                                     # broadcast (J,)
        else:
            # Fallback (no lin_vel field): position differences → 转换为 m/s 再
            # 比阈值, 与主路径单位一致 (注意: pos diff = m/frame, × fps = m/s).
            fps = float(getattr(args, "fps", 30.0))
            gt_disp = gt["g1_body_pos_w"][:, 1:, body_ids] - gt["g1_body_pos_w"][:, :-1, body_ids]
            speed_prev = gt_disp.norm(dim=-1) * fps                                  # (B, L-1, J) m/s
            static_prev = speed_prev < vel_thr_per_body
            # 末帧用倒数第 2 帧复制 (与 _finite_diff_lin_vel 一致), 仍由 vel_mask 屏蔽.
            static_gt = torch.cat([static_prev, static_prev[:, -1:]], dim=1)
        static_gt = static_gt.float()

        # vel_mask: 屏蔽末帧 (其速度 GT 是上一帧复制, label 不可靠).
        vel_mask = mask.clone()
        vel_mask[:, -1] = False
        static_conf_loss_el = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        # static_conf 不是 FK 派生 (网络 head 直出), 不门控; 只做 t_weights 加权.
        static_conf_loss, static_conf_loss_nan_count = _reduce_loss(
            static_conf_loss_el, vel_mask[..., None], t_weights, fk_gate=None
        )
        extra_loss += static_conf_loss * weights["static_conf_bce"]
        extra_loss_dict["static_conf_loss"] = static_conf_loss
        extra_loss_dict["static_conf_loss_nan_count"] = static_conf_loss_nan_count

    return extra_loss, extra_loss_dict


def compute_extra_g1_loss(inputs, outputs, ppl):
    """FK-based 3D joint position loss for G1 robot.

    Requires:
        outputs["pred_joints_w"]: (B, L, 30, 3)  predicted world-frame joint positions
        outputs["gt_joints_w"]:   (B, L, 30, 3)  GT world-frame joint positions
    """
    weights = ppl.weights
    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]   # (B, L)
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")

    pred_j = outputs["pred_joints_w"]   # (B, L, 30, 3)
    gt_j   = outputs["gt_joints_w"]

    # Root-aligned FK joint loss.
    if weights.get("fk_j3d", 0.0) > 0:
        pred_cr = pred_j - pred_j[:, :, :1]
        gt_cr   = gt_j   - gt_j[:,   :, :1]
        fk_loss_el = F.mse_loss(pred_cr, gt_cr, reduction="none")
        fk_loss, fk_j3d_loss_nan_count = _reduce_loss(fk_loss_el, mask[..., None, None], t_weights, fk_gate)
        extra_loss += fk_loss * weights["fk_j3d"]
        extra_loss_dict["fk_j3d_loss"] = fk_loss
        extra_loss_dict["fk_j3d_loss_nan_count"] = fk_j3d_loss_nan_count

    # Absolute world-frame FK joint loss (root translation also supervised)
    if weights.get("fk_j3d_abs", 0.0) > 0:
        fk_abs_loss_el = F.mse_loss(pred_j, gt_j, reduction="none")
        fk_abs_loss, fk_j3d_abs_loss_nan_count = _reduce_loss(fk_abs_loss_el, mask[..., None, None], t_weights, fk_gate)
        extra_loss += fk_abs_loss * weights["fk_j3d_abs"]
        extra_loss_dict["fk_j3d_abs_loss"] = fk_abs_loss
        extra_loss_dict["fk_j3d_abs_loss_nan_count"] = fk_j3d_abs_loss_nan_count

    return extra_loss, extra_loss_dict


def transform_points_w2c(points_w, T_w2c):
    """Transform points from world coordinates to camera coordinates.

    Args:
        points_w: (B, L, N, 3)
        T_w2c: (B, L, 4, 4)
    Returns:
        (B, L, N, 3)
    """
    R_w2c = T_w2c[..., :3, :3]
    t_w2c = T_w2c[..., :3, 3]
    return torch.einsum("blij,blnj->blni", R_w2c, points_w) + t_w2c[:, :, None, :]


def compute_robot_bbx_xys(body_pos_c, K_fullimg, reproj_z_thr=0.3, do_augment=False):
    """Compute G1 bbox in camera frame.

    do_augment: True 时给 bbox 加随机 scale (1.05~1.35) 和平移 (±0.8 × bbx_size),
    与原版 GVHMR (gvhmr_pl._build_synthetic_obs) 对 AMASS bbox 的做法一致, 训练时打开,
    val/inference 时关掉.
    """
    body_pos_c_safe = body_pos_c.clone()
    body_pos_c_safe[..., 2].clamp_(min=reproj_z_thr)
    robot_i2d = perspective_projection(body_pos_c_safe, K_fullimg)
    return get_bbx_xys(robot_i2d, do_augment=do_augment)


@autocast("cuda", enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        # 用 torch.where 替代 advanced-index in-place: 后者在 require_grad 的中间
        # 张量上是 in-place 修改, 可能破坏 autograd 图 / 触发 leaf 警告.
        # 判别用 ||R - I||_inf < tol (近单位阵处 ‖R-I‖∞ ≈ 旋转角 θ, 阈值口径与原
        # axis-angle.norm()<1e-5 一致), 避免 matrix_to_axis_angle 在近单位阵处的奇异
        # —— 该算子在本文件其他处被刻意回避 (见 :296-299, :700).
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        is_I = (R - eye).abs().amax(dim=(-1, -2)) < 1e-5              # (..., )
        return torch.where(is_I[..., None, None], eye, R)

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = vec1_xyz / vec1_xyz.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = vec2_xyz / vec2_xyz.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # 用 atan2(|cross|, dot) 代替 acos(clamp(dot, -1, 1)): vec1≈vec2 时 (训练初期
    # cam_angvel 接近 identity, dot≈1) acos 反向 1/sqrt(1-1)=∞, 后续 inf*0=NaN
    # 污染整网 grad (tools/diag_anomaly.py 定位到这一行). atan2 处处可微.
    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)                                # (B,L,3)
    _cross_norm = aa_tp1_to_t.norm(dim=-1, keepdim=True)                           # (B,L,1) ≥0
    _dot        = (vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True)                  # (B,L,1)
    aa_tp1_to_t_angle = torch.atan2(_cross_norm, _dot)                             # (B,L,1)
    aa_tp1_to_t = aa_tp1_to_t / _cross_norm.clamp(min=1e-6) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt
