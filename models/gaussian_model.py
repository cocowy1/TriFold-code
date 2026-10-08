#!/usr/bin/env python3

import torch
import torch.nn as nn

from models.archs.gs_decoder import DETRGaussianDecoder
from models.archs.encoders.conv_pointnet import ConvPointnet
from models.archs.encoders.gs_vae import GaussianVAE


class GsModel(nn.Module):
    """
    新版 GsModel:
    1) 用 ConvPointnet 编码得到 triplane features + vm vectors
    2) 持有 DETRGaussianDecoder
    """

    def __init__(self, specs, point2gs=True):
        super().__init__()
        self.specs = specs
        self.point2gs = point2gs

        encoder_specs = self.specs["GSModelSpecs"]
        self.hidden_dim = encoder_specs["hidden_dim"]
        self.latent_dim = encoder_specs["latent_dim"]
        self.pn_hidden = encoder_specs.get("pn_hidden_dim", self.latent_dim)

        # -------------------------------------------------
        # Encoder: ConvPointnet
        # -------------------------------------------------
        self.pointnet = ConvPointnet(
            c_dim=self.latent_dim,
            dim=self.specs.get("input_dim", 59),
            hidden_dim=self.pn_hidden,
            plane_resolution=encoder_specs.get("plane_resolution", 256),
            qformer_query=encoder_specs.get("qformer_query", 4),
            qformer_heads=encoder_specs.get("qformer_heads", 8),
            qformer_layers=encoder_specs.get("qformer_layers", 3),
            qformer_num_freqs=encoder_specs.get("qformer_num_freqs", 8),
            qformer_ca_stages=encoder_specs.get("qformer_ca_stages", 2),
            plane_aggregator=encoder_specs.get("plane_aggregator", "qformer"),
            use_plane_unet=encoder_specs.get("use_plane_unet", False),
            plane_resnet_hidden=encoder_specs.get("plane_resnet_hidden", 128),
            plane_resnet_blocks=encoder_specs.get("plane_resnet_blocks", 2),
            plane_resnet_stem=encoder_specs.get("plane_resnet_stem", 7),
            use_cross_plane_sparse_qformer=encoder_specs.get("use_cross_plane_sparse_qformer", True),
            cross_plane_sparse_qformer_heads=encoder_specs.get("cross_plane_sparse_qformer_heads", 8),
            cross_plane_sparse_qformer_layers=encoder_specs.get("cross_plane_sparse_qformer_layers", 1),
            cross_plane_sparse_qformer_ff_mult=encoder_specs.get("cross_plane_sparse_qformer_ff_mult", 4),
            cross_plane_sparse_qformer_chunk_size=encoder_specs.get("cross_plane_sparse_qformer_chunk_size", 8000),
            plane_active_eps=encoder_specs.get("plane_active_eps", 1e-5),
            # ⭐ NEW
            trunk_type=encoder_specs.get("trunk_type", "pointnet"),
            sparse_conv_specs=encoder_specs.get("sparse_conv_specs", None),  
            ptv3_specs=encoder_specs.get("ptv3_specs", None),
                                                  
        )
        vae_cfg = self.specs.get("VAESpecs", {})
        self.use_vae = vae_cfg.get("enabled", False)
        if self.use_vae:
            self.vae = GaussianVAE(
                qformer_query=self.pointnet.qformer_query,
                latent_dim=self.latent_dim,
                plane_resolution=encoder_specs.get("plane_resolution", 256),
                plane_mode=vae_cfg.get("plane_mode", "coupled"),

                z_channels=vae_cfg.get("z_channels", 32),
                base_ch=vae_cfg.get("base_ch", 128),
                ch_mult=tuple(vae_cfg.get("ch_mult", [1, 2, 2, 4])),
                num_res_per_stage=vae_cfg.get("num_res_per_stage", 2),
                dropout=vae_cfg.get("dropout", 0.0),

                latent_scale=vae_cfg.get("latent_scale", 1.0),

                plane_l1_weight=vae_cfg.get("plane_l1_weight", 1.0),
                plane_cosine_weight=vae_cfg.get("plane_cosine_weight", 0.0),
                plane_stat_weight=vae_cfg.get("plane_stat_weight", 0.0),
                plane_cos_norm_eps=vae_cfg.get("plane_cos_norm_eps", 0),
                plane_active_l1_weight=vae_cfg.get("plane_active_l1_weight", 1.0),
                plane_inactive_l1_weight=vae_cfg.get("plane_inactive_l1_weight", 0.1),
                rec_mu_monitor_every=vae_cfg.get("rec_mu_monitor_every", 10),
                input_downsample_factor=vae_cfg.get("input_downsample_factor", 1),
                learned_input_downsample=vae_cfg.get("learned_input_downsample", False),

                transformer_d_model=vae_cfg.get("transformer_d_model", 512),
                transformer_depth=vae_cfg.get("transformer_depth", 4),
                transformer_heads=vae_cfg.get("transformer_heads", 8),
                transformer_mlp_mult=vae_cfg.get("transformer_mlp_mult", 4.0),
                transformer_downsample=vae_cfg.get("transformer_downsample", 2),
                transformer_patch_size=vae_cfg.get("transformer_patch_size", 2),
                transformer_use_rope=vae_cfg.get("transformer_use_rope", True),
                transformer_rope_freq=tuple(vae_cfg.get("transformer_rope_freq", [1.0, 10000.0])),
                transformer_attn_mode=vae_cfg.get("transformer_attn_mode", "full"),
                transformer_latent_full_res=vae_cfg.get("transformer_latent_full_res", True),
                transformer_up_res_blocks=vae_cfg.get("transformer_up_res_blocks", 1),
                attention_impl=vae_cfg.get("attention_impl", None),

                use_checkpoint=vae_cfg.get("use_checkpoint", False),
            )
            
        # -------------------------------------------------
        # DETR Gaussian Decoder
        # 注意: context_dim 要和 plane feature channel 对齐
        # ConvPointnet 的 c_dim = latent_dim，所以这里默认用 latent_dim
        # -------------------------------------------------
        decoder_cfg = self.specs.get("DETRDecoderSpecs", {})

        decoder_specs = {
            "hidden_dim": self.hidden_dim,
            "context_dim": self.latent_dim,
            "voxel_reso": self.specs.get("voxel_reso", 256),
            "padding": self.specs.get("voxel_padding", 0.1),
            "sh_dim": self.specs.get("sh_dim", 48),

            "num_queries": decoder_cfg.get("num_queries", 24),
            "num_layers": decoder_cfg.get("num_layers", 6),
            "num_heads": decoder_cfg.get("num_heads", 8),
            "ca_kv_heads": decoder_cfg.get("ca_kv_heads", 8),

            "ff_mult": decoder_cfg.get("ff_mult", 4),
            "dropout": decoder_cfg.get("dropout", 0.0),

            "context_num_tokens_per_plane": decoder_cfg.get(
                "context_num_tokens_per_plane",
                self.pointnet.qformer_query,
            ),

            "use_patch_context": decoder_cfg.get("use_patch_context", False),
            "context_patch_radius": decoder_cfg.get("context_patch_radius", 0),
            "context_rel_num_freqs": decoder_cfg.get("context_rel_num_freqs", 6),

            "use_query_resample_refine": decoder_cfg.get("use_query_resample_refine", True),
            "query_resample_fuse_mode": decoder_cfg.get("query_resample_fuse_mode", "sum"),
            "query_resample_detach_xyz": decoder_cfg.get("query_resample_detach_xyz", True),
            "refine_scale": decoder_cfg.get("refine_scale", True),
            "refine_rotation": decoder_cfg.get("refine_rotation", True),
            "hybrid_inter_voxel_scales": decoder_cfg.get(
                "hybrid_inter_voxel_scales",
                None,
            ),
            "occ_context_fuse_mode": decoder_cfg.get(
                "occ_context_fuse_mode",
                "sum",
            ),

            "predict_opacity": decoder_cfg.get("predict_opacity", True),
            "predict_objectness": decoder_cfg.get("predict_objectness", True),

            "use_dn": decoder_cfg.get("use_dn", False),

            "decoder_type": decoder_cfg.get("decoder_type", "detr"),

            # sparse mode
            "sparse_num_layers": decoder_cfg.get("sparse_num_layers", 6),
            "sparse_window_size": decoder_cfg.get("sparse_window_size", 3),
            "sparse_coord_stride": decoder_cfg.get("sparse_coord_stride", 1),
            "sparse_use_shifted_windows": decoder_cfg.get("sparse_use_shifted_windows", True),
            "sparse_use_rel_pe": decoder_cfg.get("sparse_use_rel_pe", True),
            "sparse_rel_pe_num_freqs": decoder_cfg.get("sparse_rel_pe_num_freqs", 4),

            # hybrid mode
            "hybrid_num_layers": decoder_cfg.get("hybrid_num_layers", 6),
            "hybrid_window_size": decoder_cfg.get("hybrid_window_size", 3),
            "hybrid_coord_stride": decoder_cfg.get("hybrid_coord_stride", 1),
            "hybrid_use_shifted_windows": decoder_cfg.get("hybrid_use_shifted_windows", True),
            "hybrid_use_rel_pe": decoder_cfg.get("hybrid_use_rel_pe", True),
            "hybrid_rel_pe_num_freqs": decoder_cfg.get("hybrid_rel_pe_num_freqs", 4),
            "hybrid_ff_mult": decoder_cfg.get("hybrid_ff_mult", 4),
            "hybrid_use_ffn": decoder_cfg.get("hybrid_use_ffn", True),
            "hybrid_zero_init_intra_out": decoder_cfg.get("hybrid_zero_init_intra_out", True),

            # QueryHeads shape trunk
            "shape_trunk_layers": decoder_cfg.get("shape_trunk_layers", 2),
            "shape_trunk_mult": decoder_cfg.get("shape_trunk_mult", 2),

            # QueryHeads prediction heads
            "head_layers": decoder_cfg.get("head_layers", 2),
            "head_hidden_mult": decoder_cfg.get("head_hidden_mult", 3),
        }


        self.gs_decoder = DETRGaussianDecoder(**decoder_specs)

    # =====================================================
    # 1) 编码：pc -> plane_dict, vm_dict
    # =====================================================
    def encode_planes(
        self,
        pc,
        sample_latent: bool = False,
        return_vae_aux: bool = False,
        step: int = None,
        noise_scale=None,
        decoder_clean_latent: torch.Tensor = None,
        decoder_diffusion_latent: torch.Tensor = None,
    ):
        """
        pc: [B, N, 59]  输入高斯属性 / 点云属性

        step: 当前全局训练步数（保留参数以兼容未来扩展）。

        Returns:
            plane_dict = {"xz": ..., "xy": ..., "yz": ...}
            vm_dict    = {"xz": ..., "xy": ..., "yz": ...}
        """
        plane_xz, plane_xy, plane_yz = self.pointnet.get_plane_features(pc)
        plane_dict = {
            "xz": plane_xz,
            "xy": plane_xy,
            "yz": plane_yz,
        }

        # q_stats = self.pointnet.qtoken_heterogeneity_stats(
        # {
        #     "xz": plane_xz.detach(),
        #     "xy": plane_xy.detach(),
        #     "yz": plane_yz.detach(),
        # },
        # active_masks = self.pointnet._cached_plane_active_masks,
        # max_grids=8192,
        # )
        # print(q_stats)

        vm_dict = self.pointnet._cached_vm_vectors
        if not self.use_vae:
            return (plane_dict, vm_dict) if not return_vae_aux else (plane_dict, vm_dict, None)

        # Pass triplane active masks from ConvPointnet to VAE for mask-aware L1
        active_masks = self.pointnet._cached_plane_active_masks
        if active_masks is not None:
            active_masks = {k: v.detach() for k, v in active_masks.items()}

        vae_out = self.vae(
            plane_dict,
            sample=sample_latent,
            step=step,
            return_plane_loss=self.training,
            noise_scale=noise_scale,
            active_masks=active_masks,
            decoder_noise_tau=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_noise_tau", 0.0
                )
            ),
            decoder_noise_prob=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_noise_prob", 1.0
                )
            ),
            decoder_plane_noise_tau=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_plane_noise_tau", 0.0
                )
            ),
            decoder_plane_noise_prob=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_plane_noise_prob", 1.0
                )
            ),
            decoder_clean_latent=decoder_clean_latent,
            decoder_diffusion_latent=decoder_diffusion_latent,
            decoder_residual_prob=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_residual_prob", 0.0
                )
            ),
            decoder_residual_scale_min=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_residual_scale_min", 0.0
                )
            ),
            decoder_residual_scale_max=float(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_residual_scale_max", 1.0
                )
            ),
            decoder_residual_clean_prob=(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_residual_clean_prob", None
                )
            ),
            decoder_residual_full_prob=(
                self.specs.get("VAESpecs", {}).get(
                    "decoder_residual_full_prob", None
                )
            ),
        )
        
        plane_dict_rec = vae_out["plane_dict_rec"]
        # VM: 训练期仍用 encoder 真实的 vm_dict 做监督锚点;
        # 推理期 (diffusion 采样) 用 VAE 预测的 vm 替代.
        vm_dict_used = vm_dict
        # if self.training:
        #     vm_dict_used = vm_dict
        # else:
        #     vm_dict_used = vae_out.get("vm_dict_pred", vm_dict)

        if return_vae_aux:
            return plane_dict_rec, vm_dict_used, vae_out
        return plane_dict_rec, vm_dict_used

    # =====================================================
    # 2) 已知 voxel 的解码接口（训练最常用）
    # =====================================================
    def decode_voxels(
        self,
        plane_dict,
        vm_dict,
        voxel_centers: torch.Tensor,
        voxel_batch: torch.Tensor,
        gt_per_voxel: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
        skip_enhance: bool = False,
    ):
        """
        对指定 voxels 做解码。
        训练时 CombinedModel 通常会直接调这个接口 / 或直接调 self.gs_decoder。
        """
        return self.gs_decoder(
            plane_dict=plane_dict,
            vm_dict=vm_dict,
            voxel_centers=voxel_centers,
            voxel_batch=voxel_batch,
            gt_per_voxel=gt_per_voxel,
            valid_mask=valid_mask,
            skip_enhance=skip_enhance,
        )

    # =====================================================
    # 3) 端到端 forward：先 encode，再 decode_given_voxels
    # =====================================================
    def forward(
        self,
        pc: torch.Tensor,
        voxel_centers: torch.Tensor,
        voxel_batch: torch.Tensor,
        gt_per_voxel: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
        skip_enhance: bool = False,
        step: int = None,
    ):
        """
        适合"训练时已知 voxel_centers / voxel_batch"的情形。

        step: 当前全局训练步数（保留参数以兼容未来扩展）。
        """
        if self.use_vae and self.training:
            plane_dict, vm_dict, vae_out = self.encode_planes(
                pc, sample_latent=True, return_vae_aux=True, step=step,
            )
        else:
            plane_dict, vm_dict = self.encode_planes(pc, step=step)
            vae_out = None

        out = self.decode_voxels(
            plane_dict=plane_dict,
            vm_dict=vm_dict,
            voxel_centers=voxel_centers,
            voxel_batch=voxel_batch,
            gt_per_voxel=gt_per_voxel,
            valid_mask=valid_mask,
            skip_enhance=skip_enhance,
        )

        if vae_out is not None:
            out["vae_out"] = vae_out
            out["vae_kl_loss"] = vae_out["kl_loss"]
            if "rec_loss" in vae_out:
                out["vae_rec_loss"] = vae_out["rec_loss"]
            if "plane_l1" in vae_out:
                out["vae_plane_l1"] = vae_out["plane_l1"]
            if "plane_cosine" in vae_out:
                out["vae_plane_cosine"] = vae_out["plane_cosine"]
            if "plane_stat" in vae_out:
                out["vae_plane_stat"] = vae_out["plane_stat"]

        return out
