import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import mlx.core as mx

# Bind local packaged modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlx_gsplat.allocator import PointerStableChunkedAllocator
from mlx_gsplat.optimizer import FusedMLXAdam
from mlx_gsplat.rendering import frustum_culling_mlx, rasterize_gaussians_mlx, bilinear_resize_mlx
from mlx_gsplat.dataset import Parser, Dataset

try:
    import mlx_gsplat_ext
    HAS_METAL_EXT = True
except ImportError:
    HAS_METAL_EXT = False


@dataclass
class MLXConfig:
    max_gaussians: int = 500000  # Pre-allocated maximum physical pool size (N_max)
    init_gaussians: Optional[int] = None  # Optional cap for SfM initialization points
    max_steps: int = 300  # Total training step count (aligned with PyTorch)
    lr: float = 1.6e-4
    sh_degree: int = 3
    sh_degree_interval: int = 500  # SH Schedule interval (aligned with PyTorch)
    refine_every: int = 100  # Periodic Realloc-Free Densification
    data_dir: str = "data/garden"  # Path to target real scene directory
    data_factor: int = 8  # Downsample factor for real images (prevents OOM / exit code 137)
    width: int = 800
    height: int = 600
    use_real_data: bool = True  # Auto-toggle to load real Colmap data if present
    render_width: int = 200  # Default downscaled rasterization for reasonable speed
    render_height: int = 150  # Default downscaled rasterization for reasonable speed
    pixel_chunk_size: int = 1024
    preload_images: bool = True


class RunnerMLX:
    """
    UMA-Native MLX 3DGS Training Engine.
    Fully packaged to work stand-alone bypassing PyTorch/pycolmap.
    """

    def __init__(self, cfg: MLXConfig):
        self.cfg = cfg
        self.real_data_loaded = False

        print(f"=== [Bootstrapping] Packaged MLX UMA Gaussian 3DGS Runtime ===")

        # Resolve relative data_dir to project root absolutely to prevent PyCharm CWD issues
        if not os.path.isabs(cfg.data_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg.data_dir = os.path.join(project_root, cfg.data_dir)

        colmap_sparse_dir = os.path.join(cfg.data_dir, "sparse/0")

        if cfg.use_real_data and os.path.exists(colmap_sparse_dir):
            print(f"[Dataset] Real Colmap data detected at: {cfg.data_dir}")
            print(f"[Dataset] Loading points and camera matrices...")

            # Setup packaged Colmap Parser
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=True
            )

            # Load training images
            self.train_dataset = Dataset(
                self.parser,
                split="train",
                preload_images=cfg.preload_images
            )

            self.scene_scale = self.parser.scene_scale
            num_init = len(self.parser.points)
            print(f"[Dataset] Successfully loaded {num_init} SfM initial point cloud points.")
            print(f"[Dataset] Normalized Scene Scale: {self.scene_scale:.4f}")

            # Extract real SfM parameters
            points = np.array(self.parser.points, dtype=np.float32)
            rgbs = np.array(self.parser.points_rgb / 255.0, dtype=np.float32)
            if cfg.init_gaussians is not None and cfg.init_gaussians < num_init:
                rng = np.random.default_rng(42)
                keep = rng.choice(num_init, size=cfg.init_gaussians, replace=False)
                points = points[keep]
                rgbs = rgbs[keep]
                num_init = cfg.init_gaussians
                print(f"[Dataset] Subsampled SfM initialization to {num_init} points for memory-limited training.")

            # Simple scale distance estimation on UMA
            dist_avg = np.ones(num_init, dtype=np.float32) * 0.01
            scales = np.log(dist_avg).reshape(-1, 1).repeat(3, axis=1)

            quats = np.random.randn(num_init, 4).astype(np.float32)
            quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
            opacities = np.ones((num_init, 1), dtype=np.float32) * 0.1

            # RGB to SH conversion (standard 3DGS protocol to prevent faded colors in PLY viewers)
            SH_C0 = 0.28209479177387814
            sh0 = (rgbs.reshape(num_init, 1, 3) - 0.5) / SH_C0
            shN_dim = ((cfg.sh_degree + 1) ** 2 - 1) * 3
            shN = np.zeros((num_init, shN_dim), dtype=np.float32)

            init_params = {
                "means": mx.array(points),
                "scales": mx.array(scales),
                "quats": mx.array(quats),
                "opacities": mx.array(opacities),
                "sh0": mx.array(sh0.reshape(num_init, -1)),
                "shN": mx.array(shN)
            }
            self.real_data_loaded = True

        else:
            num_init = 3000
            print(f"[Simulator] Real data not found. Generating synthetic SfM cloud of {num_init} points.")

            syn_dist_avg = np.ones(num_init, dtype=np.float32) * 0.01
            syn_scales = np.log(syn_dist_avg).reshape(-1, 1).repeat(3, axis=1)

            init_params = {
                "means": mx.array(np.random.randn(num_init, 3).astype(np.float32)),
                "scales": mx.array(syn_scales),
                "quats": mx.array(np.random.randn(num_init, 4).astype(np.float32)),
                "opacities": mx.array(np.ones((num_init, 1)).astype(np.float32) * 0.1),
                "sh0": mx.array(np.random.randn(num_init, 3).astype(np.float32)),
                "shN": mx.array(np.random.randn(num_init, 45).astype(np.float32))
            }
            self.scene_scale = 1.0

        # Instantiate Pointer-Stable Allocator with calculated capacity
        self.allocator = PointerStableChunkedAllocator(
            max_gaussians=cfg.max_gaussians,
            sh_degree=cfg.sh_degree
        )
        self.allocator.set_initial_gaussians(init_params)

        # Instantiate Fused Optimizer
        self.optimizer = FusedMLXAdam(lr=cfg.lr)

        # Initialize micro-profiler stats
        self.profiler_stats = {
            "dataset_get": 0.0,
            "culling": 0.0,
            "sh_gather": 0.0,
            "rasterize": 0.0,
            "param_gather": 0.0,
            "optimizer": 0.0,
            "eval_flush": 0.0,
            "total_step": 0.0,
        }

        # Bypass compilation to prevent Symbolic Eval constraints inside mx.compile
        self.compiled_train_step = self._raw_train_step
        print("[Engine Status] UMA Zero-Realloc Lazy Graph Engine configured successfully.")

    def _ssim(self, img1: mx.array, img2: mx.array) -> mx.array:
        """Lightweight differentiable SSIM computation in MLX."""
        x1 = mx.expand_dims(img1, axis=0)  # [1, H, W, C]
        x2 = mx.expand_dims(img2, axis=0)  # [1, H, W, C]
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        H, W, channel = img1.shape
        window_size = 11
        window = mx.ones((channel, window_size, window_size, 1), dtype=img1.dtype) / (window_size * window_size)
        
        padding = window_size // 2
        
        mu1 = mx.conv2d(x1, window, stride=1, padding=padding, groups=channel)
        mu2 = mx.conv2d(x2, window, stride=1, padding=padding, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = mx.conv2d(x1 * x1, window, stride=1, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = mx.conv2d(x2 * x2, window, stride=1, padding=padding, groups=channel) - mu2_sq
        sigma12 = mx.conv2d(x1 * x2, window, stride=1, padding=padding, groups=channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return mx.mean(ssim_map)

    def _raw_train_step(
            self,
            viewmat: mx.array,
            K: mx.array,
            step: int,
            target_pixels: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, Dict[str, mx.array], Dict[str, mx.array]]:

        # 3. Parameter Gather Step: Extract optimization subset independently
        t0 = time.perf_counter()
        active_params = self.allocator.get_active_parameters()
        active_m, active_v = self.allocator.get_active_adam_states()
        mx.eval(*active_params.values())
        self.profiler_stats["param_gather"] += (time.perf_counter() - t0) * 1000.0

        # 1. Zero-Copy Culling index mapping directly on active parameters
        t0 = time.perf_counter()
        active_indices = frustum_culling_mlx(
            means=active_params["means"],
            quats=active_params["quats"],
            scales=active_params["scales"],
            opacities=active_params["opacities"],
            viewmat=viewmat, K=K, width=self.cfg.width, height=self.cfg.height
        )
        mx.eval(active_indices)
        self.profiler_stats["culling"] += (time.perf_counter() - t0) * 1000.0

        # Dynamic chunk size to optimize memory utilization and prevent swapping
        n_active = max(1, active_indices.shape[0])
        dynamic_chunk_size = max(512, min(4096, 30_000_000 // n_active))

        # 2. Gather active SH-coefficients based on gradual SH Degree Schedule
        # [Moved inside loss_fn to preserve Autograd tracking chain]
        self.profiler_stats["sh_gather"] += 0.0

        # [VRAM OOM Fix Core]
        # Cache rendered results using local scope dictionary to prevent duplicate memory allocation.
        forward_cache = {}

        if target_pixels is not None:
            target_box = bilinear_resize_mlx(target_pixels, self.cfg.render_height, self.cfg.render_width)
        else:
            target_box = None

        # 4. Differentiable Autograd Loss Function
        def loss_fn(params_dict):
            # Gather active SH-coefficients inside loss_fn to preserve gradient tracking
            sh0_active = mx.take(params_dict["sh0"], active_indices, axis=0)
            shN_active = mx.take(params_dict["shN"], active_indices, axis=0)

            sh_degree_to_use = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)
            if sh_degree_to_use == 0:
                colors_val = sh0_active
            elif sh_degree_to_use == 1:
                colors_val = mx.concatenate([sh0_active, shN_active[:, :9]], axis=-1)
            elif sh_degree_to_use == 2:
                colors_val = mx.concatenate([sh0_active, shN_active[:, :24]], axis=-1)
            else:
                colors_val = mx.concatenate([sh0_active, shN_active], axis=-1)

            r_colors, r_alphas = rasterize_gaussians_mlx(
                means=params_dict["means"],
                quats=params_dict["quats"],
                scales=params_dict["scales"],
                opacities=params_dict["opacities"],
                colors=colors_val,
                viewmat=viewmat, K=K, width=self.cfg.width, height=self.cfg.height,
                active_indices=active_indices,
                means_active=params_dict["means"],
                quats_active=params_dict["quats"],
                scales_active=params_dict["scales"],
                opacities_active=params_dict["opacities"],
                render_width=self.cfg.render_width,
                render_height=self.cfg.render_height,
                pixel_chunk_size=dynamic_chunk_size
            )

            # Cache the computed results in the external dictionary for reuse
            forward_cache["render_colors"] = r_colors
            forward_cache["render_alphas"] = r_alphas

            if target_box is not None:
                # Rasterizer already returns the fixed low-res training target size.
                pred_box = r_colors

                l1_loss = mx.mean(mx.abs(pred_box - target_box))
                d_ssim = 0.5 * (1.0 - self._ssim(pred_box, target_box))
                loss = 0.8 * l1_loss + 0.2 * d_ssim
            else:
                loss = mx.mean(r_colors)

            # Match PyTorch's direct active parameter dependency to guarantee gradient propagation
            # through the mock/dummy rasterizer setup.
            param_dependency = (
                mx.sum(params_dict["means"]) +
                mx.sum(params_dict["quats"]) +
                mx.sum(params_dict["scales"]) +
                mx.sum(params_dict["opacities"]) +
                mx.sum(colors_val)
            ) * 1e-5
            return loss + param_dependency

        # 5. Execute Autograd tracking
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(active_params)

        mx.eval(loss, *[g for g in grads.values() if g is not None])

        # Estimate and accumulate screen-space 2D gradients for densification strategy
        if grads["means"] is not None and active_indices.shape[0] > 0:
            means_active_visible = mx.take(active_params["means"], active_indices, axis=0)
            ones = mx.ones((means_active_visible.shape[0], 1), dtype=means_active_visible.dtype)
            means_hom = mx.concatenate([means_active_visible, ones], axis=1)
            means_cam = mx.matmul(means_hom, viewmat.T)[:, :3]
            depths = mx.abs(means_cam[:, 2])

            grad_means_active = grads["means"]
            grad_means_visible = mx.take(grad_means_active, active_indices, axis=0)
            grad_means_norm = mx.sqrt(mx.sum(grad_means_visible ** 2, axis=1) + 1e-8)

            fx = float(K[0, 0].item())
            grad2d_est = grad_means_norm * depths / fx

            visible_physical_slots = mx.array(self.allocator.indirection_table[np.array(active_indices)], dtype=mx.int32)
            self.allocator.grad2d[visible_physical_slots] += grad2d_est
            self.allocator.count[visible_physical_slots] += 1.0
            mx.eval(self.allocator.grad2d, self.allocator.count)

        # 6. Optimizer step (dispatched according to Pure MLX or Metal extension mode)
        t0 = time.perf_counter()
        updated_p, updated_m, updated_v = self.optimizer.step(
            active_params=active_params,
            grads=grads,
            active_m=active_m,
            active_v=active_v,
            use_metal_fusion=True,  # Enable Custom Metal Fused GPU acceleration
            physical_pools=self.allocator.physical_pools,
            adam_m_pools=self.allocator.adam_m_pools,
            adam_v_pools=self.allocator.adam_v_pools,
            indirection_table=self.allocator.indirection_table,
            cached_pointers=self.allocator.cached_pointers,
            cached_indices_addr=self.allocator.cached_indices_addr
        )
        self.profiler_stats["optimizer"] += (time.perf_counter() - t0) * 1000.0

        # 7. Return step: Immediately return cached tensors in virtual graph without calling heavy rasterizer again.
        t0 = time.perf_counter()
        render_colors = forward_cache["render_colors"]
        render_alphas = forward_cache["render_alphas"]

        mx.eval(render_colors, render_alphas)
        self.profiler_stats["rasterize"] += (time.perf_counter() - t0) * 1000.0

        return render_colors, render_alphas, updated_p, updated_m, updated_v

    def train(self):
        print(f"\n=== [Training Starting] Active Gaussians: {self.allocator.num_active} ===")

        loop_start = time.time()

        dummy_viewmat = mx.array(np.eye(4).astype(np.float32))
        dummy_K = mx.array(np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]]).astype(np.float32))

        for step in range(1, self.cfg.max_steps + 1):

            # Apply Exponential Learning Rate Decay to the optimizer
            self.optimizer.lr = self.cfg.lr * (0.01 ** (step / self.cfg.max_steps))

            t_dataset_start = time.perf_counter()
            if self.real_data_loaded:
                image_idx = np.random.randint(0, len(self.train_dataset))
                data_batch = self.train_dataset[image_idx]

                viewmat = data_batch["camtoworld"]
                K = data_batch["K"]
                pixels = data_batch["image"] / 255.0
            else:
                viewmat = dummy_viewmat
                K = dummy_K
                pixels = None
            self.profiler_stats["dataset_get"] += (time.perf_counter() - t_dataset_start) * 1000.0

            t_step_start = time.perf_counter()
            render_colors, render_alphas, updated_p, updated_m, updated_v = self.compiled_train_step(
                viewmat, K, step, pixels
            )

            # Sync weights in-place with scale and opacity bounds, and normalize quaternions
            if HAS_METAL_EXT:
                # Fused mode: parameters and moments are updated in-place on the GPU.
                # We only need to normalize active quaternions in Python to prevent rotation drift.
                quats = self.allocator.physical_pools["quats"]
                indices_mx = mx.array(self.allocator.indirection_table, dtype=mx.int32)
                active_quats = quats[indices_mx]
                quat_norms = mx.sqrt(mx.sum(active_quats ** 2, axis=1, keepdims=True) + 1e-8)
                self.allocator.update_physical_pool("quats", active_quats / quat_norms)
            else:
                for name in updated_p.keys():
                    val = updated_p[name]
                    if name == "scales":
                        val = mx.clip(val, a_min=-10.0, a_max=-2.0)
                    elif name == "opacities":
                        val = mx.clip(val, a_min=-5.0, a_max=2.0)
                    elif name == "quats":
                        quat_norms = mx.sqrt(mx.sum(val ** 2, axis=1, keepdims=True) + 1e-8)
                        val = val / quat_norms
                    self.allocator.update_physical_pool(name, val)
                self.allocator.update_adam_states(updated_m, updated_v)

            # Evaluate active tensors to flush asynchronous Metal execution queue
            t_eval_start = time.perf_counter()
            mx.eval(
                render_colors,
                render_alphas,
                *self.allocator.physical_pools.values(),
                *self.allocator.adam_m_pools.values(),
                *self.allocator.adam_v_pools.values()
            )
            self.allocator.update_cached_pointers()
            mx.clear_cache()
            self.profiler_stats["eval_flush"] += (time.perf_counter() - t_eval_start) * 1000.0
            self.profiler_stats["total_step"] += (time.perf_counter() - t_step_start) * 1000.0

            # Periodic Opacity Reset: Adapts proportionally for shorter budgets (e.g. every 1,000 steps for max_steps <= 3000)
            opacity_reset_interval = 1000 if self.cfg.max_steps <= 3000 else 3000
            if step > 0 and step % opacity_reset_interval == 0:
                self.allocator.physical_pools["opacities"] = mx.full(self.allocator.physical_pools["opacities"].shape, -4.59)
                print(f"[Step {step:4d} - Opacity Reset] All opacities reset to logit -4.59 (0.01 raw) due to interval {opacity_reset_interval}.")

            # Periodic Eager-style Densification within the dedicated window (adapts to shorter budgets dynamically)
            densify_start = 300 if self.cfg.max_steps <= 3000 else 500
            densify_end = 15000 if self.cfg.max_steps >= 30000 else int(self.cfg.max_steps * 0.7)

            if step >= densify_start and step <= densify_end and step % self.cfg.refine_every == 0:
                mx.eval(*self.allocator.physical_pools.values())

                num_active_cur = self.allocator.num_active

                # Extract active accumulated gradients and visibility counts
                active_indices_mx = mx.array(self.allocator.indirection_table, dtype=mx.int32)
                grad2d_active = mx.take(self.allocator.grad2d, active_indices_mx, axis=0)
                count_active = mx.take(self.allocator.count, active_indices_mx, axis=0)

                # Compute average screen-space gradient
                count_active_clamped = mx.maximum(count_active, 1.0)
                grads_avg = grad2d_active / count_active_clamped
                grads_avg_np = np.array(grads_avg)

                # Gather active parameters for policy checks
                active_params = self.allocator.get_active_parameters()
                mx.eval(*active_params.values())

                scales_np = np.array(active_params["scales"])
                opacities_np = np.array(active_params["opacities"])
                means_np = np.array(active_params["means"])
                quats_np = np.array(active_params["quats"])

                # Standard gsplat grow logic: gradient above 0.0002 threshold
                is_grad_high = grads_avg_np > 0.0002
                # grow_scale3d threshold: 0.01 * scene_scale
                is_small = np.exp(scales_np).max(axis=1) <= 0.01 * self.scene_scale

                duplicate_mask = is_grad_high & is_small
                split_mask = is_grad_high & (~is_small)
                refine_mask = duplicate_mask | split_mask
                num_splits = int(refine_mask.sum())

                # Standard gsplat prune logic: opacity < 0.005 or scale > 0.1 * scene_scale
                raw_opacities = 1.0 / (1.0 + np.exp(-opacities_np)).squeeze()
                is_too_dim = raw_opacities < 0.005
                is_too_large = np.exp(scales_np).max(axis=1) > 0.1 * self.scene_scale

                nan_mask = (
                    ~np.isfinite(scales_np).all(axis=1) |
                    ~np.isfinite(opacities_np).squeeze() |
                    ~np.isfinite(means_np).all(axis=1)
                )

                q_norms_np = np.linalg.norm(quats_np, axis=1)
                bad_quat_mask = np.abs(q_norms_np - 1.0) > 5e-2

                prune_mask = is_too_dim | is_too_large | nan_mask | bad_quat_mask

                # Minimum size guard
                if num_active_cur < 80000:
                    prune_mask = nan_mask | bad_quat_mask

                split_params = {}
                if num_splits > 0:
                    refine_indices_mx = mx.array(np.where(refine_mask)[0], dtype=mx.int32)
                    for name, pool in self.allocator.physical_pools.items():
                        parent_vals = mx.take(active_params[name], refine_indices_mx, axis=0)
                        if name == "means":
                            split_params[f"{name}_child1"] = parent_vals + mx.random.normal(parent_vals.shape) * 0.02
                            split_params[f"{name}_child2"] = parent_vals + mx.random.normal(parent_vals.shape) * 0.02
                        elif name == "scales":
                            # For splits, decrease scale by log(1.6). For duplicates, keep scale.
                            refine_indices = np.where(refine_mask)[0]
                            split_mask_in_refine = split_mask[refine_indices]
                            split_mask_in_refine_mx = mx.array(split_mask_in_refine, dtype=mx.bool)[:, None]
                            child_scales = parent_vals - mx.where(split_mask_in_refine_mx, np.log(1.6), 0.0)
                            split_params[f"{name}_child1"] = child_scales
                            split_params[f"{name}_child2"] = child_scales
                        elif name == "opacities":
                            split_params[f"{name}_child1"] = parent_vals * 0.95
                            split_params[f"{name}_child2"] = parent_vals * 0.95
                        elif name == "quats":
                            child1_q = parent_vals
                            child2_q = parent_vals + mx.random.normal(parent_vals.shape) * 0.01
                            split_params[f"{name}_child1"] = child1_q / mx.sqrt(
                                mx.sum(child1_q ** 2, axis=1, keepdims=True) + 1e-8)
                            split_params[f"{name}_child2"] = child2_q / mx.sqrt(
                                mx.sum(child2_q ** 2, axis=1, keepdims=True) + 1e-8)
                        else:
                            split_params[f"{name}_child1"] = parent_vals
                            split_params[f"{name}_child2"] = parent_vals

                num_p, num_s = self.allocator.split_and_prune(
                    prune_mask=prune_mask,
                    split_mask=refine_mask,
                    split_params=split_params
                )
                print(
                    f"[Step {step:4d} - Densification] Pruned: {num_p} | Split/Duplicated: {num_s} | Active GS: {self.allocator.num_active}")

                # Reset accumulated gradients and visibility counts in allocator after densification
                self.allocator.grad2d = mx.zeros_like(self.allocator.grad2d)
                self.allocator.count = mx.zeros_like(self.allocator.count)

            if step % 20 == 0 or step == 1:
                self.allocator.update_metrics()
                print(
                    f"Step {step:4d} | Active: {self.allocator.num_active:5d} | Fragmentation: {self.allocator.fragmentation_ratio * 100:.2f}% | Latency: {self.allocator.allocation_latency_ms:.4f} ms")

        elapsed = time.time() - loop_start
        print(f"\n[Finished] 3DGS training complete in {elapsed:.2f} seconds.")
        print(f"Average speed: {(elapsed / self.cfg.max_steps) * 1000:.2f} ms/iter.")
        print(f"Total Explicit Reallocs: {self.allocator.realloc_count} (Goal: 0)")

        # Render Micro-Profiler Report
        total_profiled = sum([
            self.profiler_stats["dataset_get"],
            self.profiler_stats["culling"],
            self.profiler_stats["sh_gather"],
            self.profiler_stats["rasterize"],
            self.profiler_stats["param_gather"],
            self.profiler_stats["optimizer"],
            self.profiler_stats["eval_flush"]
        ])

        def get_pct(val):
            return (val / total_profiled) * 100.0 if total_profiled > 0 else 0.0

        print("\n" + "=" * 60)
        print("MLX UMA-Native 3DGS Step Micro-Profiler Report")
        print("=" * 60)
        print(
            f"1. Dataset Fetch & Prep : {self.profiler_stats['dataset_get']:.2f} ms ({get_pct(self.profiler_stats['dataset_get']):.1f}%)")
        print(
            f"2. Frustum Culling      : {self.profiler_stats['culling']:.2f} ms ({get_pct(self.profiler_stats['culling']):.1f}%)")
        print(
            f"3. SH Gathering & Concat: {self.profiler_stats['sh_gather']:.2f} ms ({get_pct(self.profiler_stats['sh_gather']):.1f}%)")
        print(
            f"4. Rasterization Stub   : {self.profiler_stats['rasterize']:.2f} ms ({get_pct(self.profiler_stats['rasterize']):.1f}%)")
        print(
            f"5. Param Gather & Setup : {self.profiler_stats['param_gather']:.2f} ms ({get_pct(self.profiler_stats['param_gather']):.1f}%)")
        print(
            f"6. Optimizer Step (Metal): {self.profiler_stats['optimizer']:.2f} ms ({get_pct(self.profiler_stats['optimizer']):.1f}%)")
        print(
            f"7. GPU Queue Eval Flush : {self.profiler_stats['eval_flush']:.2f} ms ({get_pct(self.profiler_stats['eval_flush']):.1f}%)")
        print("-" * 60)
        print(f"Accumulated Step Latency: {self.profiler_stats['total_step']:.2f} ms")
        print("=" * 60 + "\n")

        data_name = os.path.basename(self.cfg.data_dir.rstrip("/\\"))
        output_ply_path = f"{data_name}_mlx_custom_3dgs.ply"
        self.export_to_ply(output_ply_path)

    def export_to_ply(self, filepath: str):
        # Gather active subset parameters on MLX (GPU)
        active_params = self.allocator.get_active_parameters()
        mx.eval(*active_params.values())

        means = np.array(active_params["means"])
        scales = np.array(active_params["scales"])
        quats = np.array(active_params["quats"])
        opacities = np.array(active_params["opacities"])
        sh0 = np.array(active_params["sh0"])
        shN = np.array(active_params["shN"])

        num_points = means.shape[0]
        print(f"\n[Exporter] Exporting {num_points} active MLX Gaussians to standard PLY format: {filepath}")

        # Normalize quaternions on CPU to prevent drift
        quat_norms = np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8
        quats = quats / quat_norms

        # Clip log-scales and opacities on CPU to prevent runaway blobs
        scales = np.clip(scales, -10.0, -2.0)
        opacities = np.clip(opacities, -5.0, 2.0)

        # Gather flat array (matching PyTorch property format order)
        nx_ny_nz = np.zeros((num_points, 3), dtype=np.float32)
        export_array = np.concatenate([
            means,
            nx_ny_nz,
            sh0,
            shN,
            opacities,
            scales,
            quats
        ], axis=-1)

        with open(filepath, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")

            for j in range(sh0.shape[1]):
                f.write(f"property float f_dc_{j}\n".encode())
            for j in range(shN.shape[1]):
                f.write(f"property float f_rest_{j}\n".encode())

            f.write(b"property float opacity\n")

            for i in range(scales.shape[1]):
                f.write(f"property float scale_{i}\n".encode())
            for i in range(quats.shape[1]):
                f.write(f"property float rot_{i}\n".encode())

            f.write(b"end_header\n")
            f.write(export_array.tobytes())

        print(f"[Exporter] Successfully exported {num_points} MLX Gaussians.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLX 3DGS Trainer")
    parser.add_argument("--data_dir", type=str, default="data/garden", help="Path to COLMAP scene directory")
    parser.add_argument("--max_steps", type=int, default=2500, help="Maximum number of training steps")
    parser.add_argument("--sh_degree_interval", type=int, default=500, help="SH Schedule interval")
    parser.add_argument("--data_factor", type=int, default=4, help="Dataset downsample factor")
    parser.add_argument("--render_width", type=int, default=400, help="Rasterization render width")
    parser.add_argument("--render_height", type=int, default=300, help="Rasterization render height")
    parser.add_argument("--pixel_chunk_size", type=int, default=1024, help="Pixels processed per rasterizer chunk")
    parser.add_argument("--max_gaussians", type=int, default=500000, help="Pre-allocated Gaussian pool size")
    parser.add_argument("--init_gaussians", type=int, default=None, help="Optional cap for initial SfM Gaussian count")
    parser.add_argument("--lazy_images", action="store_true", help="Load training images one at a time instead of preloading all into UMA")
    args = parser.parse_args()

    cfg = MLXConfig()
    cfg.data_dir = args.data_dir
    cfg.max_gaussians = args.max_gaussians
    cfg.init_gaussians = args.init_gaussians
    cfg.max_steps = args.max_steps
    cfg.sh_degree_interval = args.sh_degree_interval
    cfg.data_factor = args.data_factor
    cfg.render_width = args.render_width
    cfg.render_height = args.render_height
    cfg.pixel_chunk_size = args.pixel_chunk_size
    cfg.preload_images = not args.lazy_images

    runner = RunnerMLX(cfg)
    runner.train()
