import time
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import mlx.core as mx


class PointerStableChunkedAllocator:
    """
    UMA-Native Pointer-Stable Chunked Allocator for 3DGS.

    [BUG FIX] Fixed index memory contamination bug where 0.0 was assigned
    to the scales pool during pruning slot initialization, causing massive
    artifact disks (raw scale 1.0) to diverge.
    """

    def __init__(
            self,
            max_gaussians: int,
            sh_degree: int = 3,
            feature_dim: Optional[int] = None
    ):
        self.max_gaussians = max_gaussians
        self.sh_degree = sh_degree
        self.feature_dim = feature_dim

        # Instrumentation Metrics
        self.realloc_count = 0
        self.allocation_latency_ms = 0.0
        self.fragmentation_ratio = 0.0

        # Dimensions
        self.sh0_dim = 3  # (1, 3) flattened
        self.shN_dim = ((sh_degree + 1) ** 2 - 1) * 3

        # Pre-allocate physical parameter arrays on Unified Memory (SoA layout)
        self.physical_pools: Dict[str, mx.array] = {}
        self.init_physical_pools()

        # Colocating Adam Moments (m, v) in physical DRAM layout alongside parameters
        self.adam_m_pools: Dict[str, mx.array] = {}
        self.adam_v_pools: Dict[str, mx.array] = {}
        self.init_adam_pools()

        # Free List & Active Mask
        self.free_slots: List[int] = list(range(max_gaussians))
        self.active_mask = np.zeros(max_gaussians, dtype=np.int8)

        # Stable Indirection Table (Logical_ID -> Physical_Index)
        self.indirection_table = np.array([], dtype=np.int32)
        self.indirection_mx = None
        self.cached_pointers = None
        self.cached_indices_addr = None

        # Tracking pools for gsplat-style gradient-based densification
        self.grad2d = mx.zeros((max_gaussians,), dtype=mx.float32)
        self.count = mx.zeros((max_gaussians,), dtype=mx.float32)

        print(f"[UMA-Alloc INIT] Pre-allocated {max_gaussians} slots. Dynamic parameter resizing completely blocked.")

    def init_physical_pools(self):
        """Pre-allocates physical parameter pools on UMA."""
        self.physical_pools["means"] = mx.zeros((self.max_gaussians, 3), dtype=mx.float32)
        self.physical_pools["scales"] = mx.ones((self.max_gaussians, 3), dtype=mx.float32) * -10.0  # Safety baseline for initial pool
        self.physical_pools["quats"] = mx.zeros((self.max_gaussians, 4), dtype=mx.float32)
        self.physical_pools["opacities"] = mx.ones((self.max_gaussians, 1), dtype=mx.float32) * -5.0

        if self.feature_dim is None:
            self.physical_pools["sh0"] = mx.zeros((self.max_gaussians, self.sh0_dim), dtype=mx.float32)
            self.physical_pools["shN"] = mx.zeros((self.max_gaussians, self.shN_dim), dtype=mx.float32)
        else:
            self.physical_pools["features"] = mx.zeros((self.max_gaussians, self.feature_dim), dtype=mx.float32)
            self.physical_pools["colors"] = mx.zeros((self.max_gaussians, 3), dtype=mx.float32)

    def init_adam_pools(self):
        """Colocates Adam optimizer state momentum/variance arrays alongside parameter pools."""
        for param_name, pool in self.physical_pools.items():
            self.adam_m_pools[param_name] = mx.zeros(pool.shape, dtype=mx.float32)
            self.adam_v_pools[param_name] = mx.zeros(pool.shape, dtype=mx.float32)

    @property
    def num_active(self) -> int:
        """Returns current count of active mapped Gaussians."""
        return len(self.indirection_table)

    def update_metrics(self):
        """Calculates current fragmentation ratio metric."""
        if self.max_gaussians > 0:
            self.fragmentation_ratio = len(self.free_slots) / self.max_gaussians
        else:
            self.fragmentation_ratio = 0.0

    def update_cached_pointers(self):
        """Pre-evaluates and caches raw UMA DRAM pointers for optimizer step."""
        if len(self.indirection_table) == 0:
            return

        try:
            import mlx_gsplat_ext
            mlx_gsplat_ext.clear_buffer_cache()
        except ImportError:
            pass

        self.indirection_mx = mx.array(self.indirection_table, dtype=mx.int32)
        mx.eval(self.indirection_mx)
        self.cached_indices_addr = np.array(self.indirection_mx, copy=False).ctypes.data

        pools_to_eval = []
        for name in self.physical_pools.keys():
            pools_to_eval.append(self.physical_pools[name])
            pools_to_eval.append(self.adam_m_pools[name])
            pools_to_eval.append(self.adam_v_pools[name])

        mx.eval(*pools_to_eval)

        self.cached_pointers = {}
        for name in self.physical_pools.keys():
            self.cached_pointers[name] = {
                "theta_addr": np.array(self.physical_pools[name], copy=False).ctypes.data,
                "m_addr": np.array(self.adam_m_pools[name], copy=False).ctypes.data,
                "v_addr": np.array(self.adam_v_pools[name], copy=False).ctypes.data,
            }

    def set_initial_gaussians(self, initial_params: Dict[str, mx.array]):
        """Sets initial SfM points into the pre-allocated pool (Zero Realloc)."""
        num_init = initial_params["means"].shape[0]
        if num_init > self.max_gaussians:
            raise ValueError(f"SfM initialization ({num_init}) exceeds Chunked Pool capacity ({self.max_gaussians})")

        # Pop from free list
        allocated = [self.free_slots.pop() for _ in range(num_init)]
        self.active_mask[allocated] = 1
        self.indirection_table = np.array(allocated, dtype=np.int32)

        # Write directly to the physical DRAM block
        for param_name, init_val in initial_params.items():
            if param_name in ["sh0", "shN"]:
                init_val = init_val.reshape(num_init, -1)

            pool_np = np.array(self.physical_pools[param_name])
            pool_np[allocated] = np.array(init_val)
            self.physical_pools[param_name] = mx.array(pool_np)

        self.update_metrics()
        self.update_cached_pointers()
        print(
            f"[UMA-Alloc INIT] Mapped {num_init} Gaussians. Active Slots: {allocated[:5]}... Fragmentation Ratio: {self.fragmentation_ratio * 100:.2f}%")

    def get_active_parameters(self) -> Dict[str, mx.array]:
        """Gathers active subsets using UMA-Native gathering via Stable Indirection mapping."""
        indices = mx.array(self.indirection_table, dtype=mx.int32)
        active_params = {}
        for param_name, pool in self.physical_pools.items():
            active_params[param_name] = mx.take(pool, indices, axis=0)
        return active_params

    def get_active_adam_states(self) -> Tuple[Dict[str, mx.array], Dict[str, mx.array]]:
        """Retrieves colocated Adam moments corresponding to active mapping."""
        indices = mx.array(self.indirection_table, dtype=mx.int32)
        active_m = {}
        active_v = {}
        for param_name in self.physical_pools.keys():
            active_m[param_name] = mx.take(self.adam_m_pools[param_name], indices, axis=0)
            active_v[param_name] = mx.take(self.adam_v_pools[param_name], indices, axis=0)
        return active_m, active_v

    def update_physical_pool(self, param_name: str, active_updated_val: mx.array):
        """Updates UMA physical parameter pool in-place (Zero-Copy update)."""
        indices_mx = mx.array(self.indirection_table, dtype=mx.int32)
        pool = self.physical_pools[param_name]
        pool[indices_mx] = active_updated_val
        self.physical_pools[param_name] = pool

    def update_adam_states(self, m_dict: Dict[str, mx.array], v_dict: Dict[str, mx.array]):
        """Updates colocated Adam state pools in-place."""
        indices_mx = mx.array(self.indirection_table, dtype=mx.int32)
        for param_name in self.physical_pools.keys():
            m_pool = self.adam_m_pools[param_name]
            v_pool = self.adam_v_pools[param_name]
            m_pool[indices_mx] = m_dict[param_name]
            v_pool[indices_mx] = v_dict[param_name]
            self.adam_m_pools[param_name] = m_pool
            self.adam_v_pools[param_name] = v_pool

    def split_and_prune(
            self,
            prune_mask: np.ndarray,  # Boolean array of size self.num_active
            split_mask: np.ndarray,  # Boolean array of size self.num_active
            split_params: Dict[str, mx.array]  # Newly created child parameter values
    ) -> Tuple[int, int]:
        """
        Realloc-Free Density Control Loop on UMA.
        """
        start_time = time.perf_counter()

        # Ensure indirection_table is a numpy array
        if not isinstance(self.indirection_table, np.ndarray):
            self.indirection_table = np.array(self.indirection_table, dtype=np.int32)

        prune_indices = np.where(prune_mask)[0]
        split_indices = np.where(split_mask)[0]

        physical_slots_to_prune = self.indirection_table[prune_indices]
        physical_slots_to_split = self.indirection_table[split_indices]

        num_prunes = len(physical_slots_to_prune)
        num_splits = len(physical_slots_to_split)

        # 1. Realloc-Free Prune: Mark physical slots as inactive and return to Free List
        self.active_mask[physical_slots_to_prune] = 0
        self.free_slots.extend(physical_slots_to_prune.tolist())

        # 2. Realloc-Free Split: Retrieve slots for child 2 (child 1 reuses parent's slot)
        if len(self.free_slots) < num_splits:
            raise MemoryError(
                f"[UMA-Alloc OOM] Max capacity reached! Free Slots: {len(self.free_slots)}, Requested Split size: {num_splits}"
            )

        allocated_slots_for_children = [self.free_slots.pop() for _ in range(num_splits)]
        self.active_mask[allocated_slots_for_children] = 1

        # 3. Update parameter & state pools in-place using Pure MLX Vectorized Scatter Updates (Zero-Copy)
        if num_splits > 0:
            split_slots_mx = mx.array(physical_slots_to_split, dtype=mx.int32)
            child_slots_mx = mx.array(allocated_slots_for_children, dtype=mx.int32)
        if num_prunes > 0:
            prune_slots_mx = mx.array(physical_slots_to_prune, dtype=mx.int32)

        for param_name in self.physical_pools.keys():
            pool = self.physical_pools[param_name]
            m_pool = self.adam_m_pools[param_name]
            v_pool = self.adam_v_pools[param_name]

            if num_splits > 0:
                child1_vals = split_params[f"{param_name}_child1"]
                child2_vals = split_params[f"{param_name}_child2"]

                # Write child 1 values into parent slots in-place
                pool[split_slots_mx] = child1_vals
                m_pool[split_slots_mx] = 0.0
                v_pool[split_slots_mx] = 0.0

                # Write child 2 values into newly allocated slots in-place
                pool[child_slots_mx] = child2_vals
                m_pool[child_slots_mx] = 0.0
                v_pool[child_slots_mx] = 0.0

            if num_prunes > 0:
                # [CRITICAL FIX] Avoid assigning 0.0 to scales and opacities pool
                # to prevent memory contamination leading to huge artifact ghost disks.
                if param_name == "scales":
                    pool[prune_slots_mx] = -10.0  # Inject minimum scale logit value
                elif param_name == "opacities":
                    pool[prune_slots_mx] = -5.0  # Inject fully transparent logit value
                else:
                    pool[prune_slots_mx] = 0.0

                m_pool[prune_slots_mx] = 0.0
                v_pool[prune_slots_mx] = 0.0

            self.physical_pools[param_name] = pool
            self.adam_m_pools[param_name] = m_pool
            self.adam_v_pools[param_name] = v_pool

        # Reset tracking pools for modified slots to prevent residual gradient accumulation
        if num_splits > 0:
            self.grad2d[split_slots_mx] = 0.0
            self.count[split_slots_mx] = 0.0
            self.grad2d[child_slots_mx] = 0.0
            self.count[child_slots_mx] = 0.0
        if num_prunes > 0:
            self.grad2d[prune_slots_mx] = 0.0
            self.count[prune_slots_mx] = 0.0

        # 4. Remap Stable Indirection Table (NumPy Vectorized)
        keep_mask = ~prune_mask
        new_indirection = self.indirection_table[keep_mask]

        if num_splits > 0:
            new_indirection = np.concatenate([new_indirection, np.array(allocated_slots_for_children, dtype=np.int32)])
        self.indirection_table = new_indirection

        # 5. Profile Latency and Metrics
        end_time = time.perf_counter()
        self.allocation_latency_ms = (end_time - start_time) * 1000.0
        self.update_metrics()
        self.update_cached_pointers()

        # Log metrics to workspace root
        self.log_instrumentation_metrics()

        return num_prunes, num_splits

    def log_instrumentation_metrics(self):
        """Writes current performance profiling metrics to file in the workspace root."""
        metrics_line = (
            f"Active: {self.num_active} | "
            f"Reallocs: {self.realloc_count} | "
            f"Latency: {self.allocation_latency_ms:.4f} ms | "
            f"Fragmentation: {self.fragmentation_ratio * 100:.2f}%\n"
        )
        try:
            filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../instrumentation.log"))
            with open(filepath, "a") as f:
                f.write(metrics_line)
        except Exception:
            pass