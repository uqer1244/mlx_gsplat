import numpy as np
from typing import Dict, Tuple, Optional, List
import mlx.core as mx

try:
    import mlx_gsplat_ext
except ImportError:
    mlx_gsplat_ext = None


class FusedMLXAdam:
    """
    Custom Fused Adam Optimizer optimized for MLX and Pointer-Stable Allocator.

    This optimizer interacts directly with the allocator to update parameters
    and states in-place within the Unified Memory Pool, avoiding data copying
    and structure changes.
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_scales: Optional[Dict[str, float]] = None
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.lr_scales = lr_scales or {
            "means": 1.0,
            "scales": 2.0,
            "quats": 1.0,
            "opacities": 5.0,
            "sh0": 1.0,
            "shN": 0.05
        }

    def step(
        self,
        active_params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        active_m: Dict[str, mx.array],
        active_v: Dict[str, mx.array],
        use_metal_fusion: bool = False,
        physical_pools: Optional[Dict[str, mx.array]] = None,
        adam_m_pools: Optional[Dict[str, mx.array]] = None,
        adam_v_pools: Optional[Dict[str, mx.array]] = None,
        indirection_table: Optional[np.ndarray] = None,
        cached_pointers: Optional[Dict[str, Dict[str, int]]] = None,
        cached_indices_addr: Optional[int] = None
    ) -> Tuple[Dict[str, mx.array], Dict[str, mx.array], Dict[str, mx.array]]:
        """
        Performs a single Adam step over the active subset of parameters.

        Args:
            active_params: Dict of parameter name to active subset mx.array.
            grads: Dict of parameter name to active gradient mx.array.
            active_m: Active 1st moment state from allocator.
            active_v: Active 2nd moment state from allocator.
            use_metal_fusion: If True, invokes the Custom Metal Fusion kernel (MSL).

        Returns:
            Tuple of (updated_params, updated_m, updated_v) dicts.
        """
        self.t += 1

        updated_params = {}
        updated_m = {}
        updated_v = {}

        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        if use_metal_fusion and physical_pools is not None:
            # 1. Optimize evaluation of gradients to only execute them on GPU
            grads_to_eval = [g for g in grads.values() if g is not None]
            if grads_to_eval:
                mx.eval(*grads_to_eval)

            # 2. Setup indices address
            if cached_indices_addr is not None:
                indices_addr = cached_indices_addr
            elif indirection_table is not None:
                indices = mx.array(indirection_table, dtype=mx.int32)
                mx.eval(indices)
                indices_addr = np.array(indices, copy=False).ctypes.data
            else:
                raise ValueError("Neither cached_indices_addr nor indirection_table provided for Metal Optimizer.")

            for name in active_params.keys():
                theta_pool = physical_pools[name]
                g = grads.get(name, None)
                if g is None:
                    updated_params[name] = active_params[name]
                    updated_m[name] = active_m[name]
                    updated_v[name] = active_v[name]
                    continue

                if cached_pointers is not None and name in cached_pointers:
                    ptrs = cached_pointers[name]
                    theta_addr = ptrs["theta_addr"]
                    m_addr = ptrs["m_addr"]
                    v_addr = ptrs["v_addr"]
                else:
                    m_pool = adam_m_pools[name]
                    v_pool = adam_v_pools[name]
                    # Evaluate arrays to secure UMA DRAM pointers
                    mx.eval(theta_pool, m_pool, v_pool)
                    theta_addr = np.array(theta_pool, copy=False).ctypes.data
                    m_addr = np.array(m_pool, copy=False).ctypes.data
                    v_addr = np.array(v_pool, copy=False).ctypes.data

                g_addr = np.array(g, copy=False).ctypes.data

                dim = theta_pool.shape[1]
                max_gaussians = theta_pool.shape[0]
                num_active = len(indirection_table) if indirection_table is not None else active_params[name].shape[0]

                a_min = -1e10
                a_max = 1e10
                if name == "scales":
                    a_min = -10.0
                    a_max = -2.0
                elif name == "opacities":
                    a_min = -5.0
                    a_max = 2.0

                if mlx_gsplat_ext is not None:
                    mlx_gsplat_ext.launch_fused_adam_step(
                        theta_addr, g_addr, m_addr, v_addr, indices_addr,
                        max_gaussians, num_active, dim,
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.t,
                        a_min, a_max
                    )

                updated_params[name] = active_params[name]
                updated_m[name] = active_m[name]
                updated_v[name] = active_v[name]

            return updated_params, updated_m, updated_v

        for name in active_params.keys():
            theta = active_params[name]
            g = grads.get(name, None)

            if g is None:
                updated_params[name] = theta
                updated_m[name] = active_m[name]
                updated_v[name] = active_v[name]
                continue

            m = active_m[name]
            v = active_v[name]

            if self.weight_decay != 0.0:
                g = g + self.weight_decay * theta

            m_new = self.beta1 * m + (1.0 - self.beta1) * g
            v_new = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            scale = self.lr_scales.get(name, 1.0)
            step_size = (self.lr * scale) / bias_correction1
            denom = (mx.sqrt(v_new) / np.sqrt(bias_correction2)) + self.eps

            theta_new = theta - step_size * (m_new / denom)

            updated_params[name] = theta_new
            updated_m[name] = m_new
            updated_v[name] = v_new

        return updated_params, updated_m, updated_v
