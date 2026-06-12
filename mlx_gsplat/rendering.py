import numpy as np
from typing import Dict, Tuple, Optional, List
import mlx.core as mx


def bilinear_resize_mlx(img: mx.array, H_out: int, W_out: int) -> mx.array:
    """
    100% Pure MLX Vectorized Bilinear Interpolation.
    Preserves Autograd gradients chain and avoids external dependencies.
    """
    H_in, W_in, C = img.shape
    
    ys = mx.arange(H_out, dtype=mx.float32)
    xs = mx.arange(W_out, dtype=mx.float32)
    
    xs_in = xs * ((W_in - 1) / max(1, W_out - 1))
    ys_in = ys * ((H_in - 1) / max(1, H_out - 1))
    
    x0 = mx.floor(xs_in).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W_in - 1)
    
    y0 = mx.floor(ys_in).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H_in - 1)
    
    wx = (xs_in - x0).reshape(1, W_out, 1)
    wy = (ys_in - y0).reshape(H_out, 1, 1)
    
    img_y0 = mx.take(img, y0, axis=0)
    img_y1 = mx.take(img, y1, axis=0)
    
    c00 = mx.take(img_y0, x0, axis=1)
    c10 = mx.take(img_y0, x1, axis=1)
    c01 = mx.take(img_y1, x0, axis=1)
    c11 = mx.take(img_y1, x1, axis=1)
    
    top = (1.0 - wx) * c00 + wx * c10
    bottom = (1.0 - wx) * c01 + wx * c11
    return (1.0 - wy) * top + wy * bottom


def frustum_culling_mlx(
    means: mx.array,      # [N, 3] on UMA
    quats: mx.array,      # [N, 4] on UMA
    scales: mx.array,     # [N, 3] on UMA
    viewmat: mx.array,    # [4, 4] Camera View matrix
    K: mx.array,          # [3, 3] Camera Intrinsic matrix
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    opacities: Optional[mx.array] = None,
    opacity_threshold: float = -5.3
) -> mx.array:
    """
    UMA-Native Frustum Culling on MLX.
    """
    N = means.shape[0]

    # 1. Transform points to camera space
    ones = mx.ones((N, 1), dtype=means.dtype)
    means_hom = mx.concatenate([means, ones], axis=1)  # [N, 4]

    # Batch multiply camera extrinsics
    means_cam = mx.matmul(means_hom, viewmat.T)  # [N, 4]
    depths = means_cam[:, 2]

    # 2. Pinhole projection model
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depths_safe = mx.where(depths > near_plane, depths, mx.array(near_plane, dtype=means.dtype))

    x2d = (means_cam[:, 0] * fx) / depths_safe + cx
    y2d = (means_cam[:, 1] * fy) / depths_safe + cy

    # 3. Boundary check with tolerance for Gaussian radius
    tolerance = 50.0
    valid = (
        (depths > near_plane) &
        (depths < far_plane) &
        (x2d + tolerance >= 0) &
        (x2d - tolerance < width) &
        (y2d + tolerance >= 0) &
        (y2d - tolerance < height)
    )

    if opacities is not None:
        valid = valid & (opacities.squeeze(-1) >= opacity_threshold)

    indices = mx.arange(N)
    valid_indices = mx.where(valid, indices, mx.array(-1, dtype=mx.int32))
    sorted_indices = mx.sort(valid_indices)
    num_valid = int(valid.sum().item())
    return sorted_indices[-num_valid:]


def rasterize_gaussians_mlx(
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array,
    viewmat: mx.array,
    K: mx.array,
    width: int,
    height: int,
    active_indices: mx.array,
    means_active: Optional[mx.array] = None,
    quats_active: Optional[mx.array] = None,
    scales_active: Optional[mx.array] = None,
    opacities_active: Optional[mx.array] = None,
    render_width: int = 120,
    render_height: int = 90,
    pixel_chunk_size: int = 1024,
    mask: Optional[mx.array] = None,
    **kwargs
) -> Tuple[mx.array, mx.array]:
    """
    100% Pure MLX Differentiable Vectorized Gaussian Rasterizer.
    """
    N_active = active_indices.shape[0]
    
    if means_active is None:
        means_active = mx.take(means, active_indices, axis=0)
    else:
        means_active = mx.take(means_active, active_indices, axis=0)

    if quats_active is None:
        quats_active = mx.take(quats, active_indices, axis=0)
    else:
        quats_active = mx.take(quats_active, active_indices, axis=0)

    if scales_active is None:
        scales_active = mx.take(scales, active_indices, axis=0)
    else:
        scales_active = mx.take(scales_active, active_indices, axis=0)

    if opacities_active is None:
        opacities_active = mx.take(opacities, active_indices, axis=0)
    else:
        opacities_active = mx.take(opacities_active, active_indices, axis=0)

    if colors.shape[0] == active_indices.shape[0]:
        colors_active = colors
    else:
        colors_active = mx.take(colors, active_indices, axis=0)

    if N_active == 0:
        render_colors = mx.zeros((render_height, render_width, 3), dtype=mx.float32)
        render_alphas = mx.zeros((render_height, render_width, 1), dtype=mx.float32)
        return render_colors, render_alphas

    # 1. 3D Camera Projection
    ones = mx.ones((N_active, 1), dtype=means_active.dtype)
    means_hom = mx.concatenate([means_active, ones], axis=1)
    means_cam = mx.matmul(means_hom, viewmat.T)[:, :3]
    depths = means_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    depths_safe = mx.maximum(depths, 1e-4)
    x2d = (means_cam[:, 0] * fx) / depths_safe + cx
    y2d = (means_cam[:, 1] * fy) / depths_safe + cy
    uv = mx.stack([x2d, y2d], axis=1)

    # 2. Reconstruct 3D Covariance
    quat_norms = mx.sqrt(mx.sum(quats_active ** 2, axis=1, keepdims=True) + 1e-8)
    q = quats_active / quat_norms
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = mx.stack([
        mx.stack([1.0 - 2.0 * (y**2 + z**2), 2.0 * (x*y - r*z), 2.0 * (x*z + r*y)], axis=-1),
        mx.stack([2.0 * (x*y + r*z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y*z - r*x)], axis=-1),
        mx.stack([2.0 * (x*z - r*y), 2.0 * (y*z + r*x), 1.0 - 2.0 * (x**2 + y**2)], axis=-1)
    ], axis=-2)

    S = mx.exp(scales_active)
    S_mat = mx.stack([
        mx.stack([S[:, 0], mx.zeros_like(S[:, 0]), mx.zeros_like(S[:, 0])], axis=-1),
        mx.stack([mx.zeros_like(S[:, 0]), S[:, 1], mx.zeros_like(S[:, 0])], axis=-1),
        mx.stack([mx.zeros_like(S[:, 0]), mx.zeros_like(S[:, 0]), S[:, 2]], axis=-1)
    ], axis=-2)

    RS = mx.matmul(R, S_mat)
    Sigma3D = mx.matmul(RS, mx.transpose(RS, (0, 2, 1)))

    # 3. 2D EWA Projection
    tx, ty, tz = means_cam[:, 0], means_cam[:, 1], depths_safe
    tz2 = tz ** 2

    J = mx.stack([
        mx.stack([fx / tz, mx.zeros_like(tz), -fx * tx / tz2], axis=-1),
        mx.stack([mx.zeros_like(tz), fy / tz, -fy * ty / tz2], axis=-1),
        mx.zeros((N_active, 3))
    ], axis=-2)

    W = viewmat[:3, :3]
    JW = mx.matmul(J, W)
    Sigma2D = mx.matmul(JW, mx.matmul(Sigma3D, mx.transpose(JW, (0, 2, 1))))[:, :2, :2]
    Sigma2D = Sigma2D + mx.array([[0.3, 0.0], [0.0, 0.3]])

    # 4. Sorting
    sort_idx = mx.argsort(depths)
    uv_sorted = uv[sort_idx]
    Sigma2D_sorted = Sigma2D[sort_idx]
    opacities_sorted = opacities_active[sort_idx]
    colors_sorted = colors_active[sort_idx]

    det = Sigma2D_sorted[:, 0, 0] * Sigma2D_sorted[:, 1, 1] - Sigma2D_sorted[:, 0, 1] * Sigma2D_sorted[:, 1, 0]
    det = mx.maximum(det, 1e-6)
    
    inv_Sigma2D = mx.stack([
        mx.stack([Sigma2D_sorted[:, 1, 1] / det, -Sigma2D_sorted[:, 0, 1] / det], axis=-1),
        mx.stack([-Sigma2D_sorted[:, 1, 0] / det, Sigma2D_sorted[:, 0, 0] / det], axis=-1)
    ], axis=-2)

    raw_opacities = mx.sigmoid(opacities_sorted).squeeze(-1)

    # 5. Compute View-Dependent colors in RGB space [0, 1]
    R_w2c = viewmat[:3, :3]
    T_w2c = viewmat[:3, 3]
    cam_pos = -mx.matmul(R_w2c.T, T_w2c)
    
    means_world_sorted = mx.take(means_active, sort_idx, axis=0)
    dirs = means_world_sorted - cam_pos[None, :]
    dirs = dirs / (mx.sqrt(mx.sum(dirs ** 2, axis=-1, keepdims=True)) + 1e-8)
    
    SH_C0 = 0.28209479177387814
    rgb_active = colors_sorted[:, :3] * SH_C0
    
    if colors_sorted.shape[1] >= 12:
        SH_C1 = 0.4886025119029199
        x_dir = dirs[:, 0]
        y_dir = dirs[:, 1]
        z_dir = dirs[:, 2]
        
        rgb_active = rgb_active + SH_C1 * mx.stack([
            colors_sorted[:, 3] * y_dir + colors_sorted[:, 4] * z_dir + colors_sorted[:, 5] * x_dir,
            colors_sorted[:, 6] * y_dir + colors_sorted[:, 7] * z_dir + colors_sorted[:, 8] * x_dir,
            colors_sorted[:, 9] * y_dir + colors_sorted[:, 10] * z_dir + colors_sorted[:, 11] * x_dir
        ], axis=1)
        
    rgb_active = mx.clip(rgb_active + 0.5, 0.0, 1.0)

    # 6. Pixel Grid
    x_coords = mx.arange(render_width, dtype=mx.float32)
    y_coords = mx.arange(render_height, dtype=mx.float32)
    grid_x, grid_y = mx.meshgrid(x_coords, y_coords, indexing="xy")
    
    scale_factor_x = float(width) / float(render_width)
    scale_factor_y = float(height) / float(render_height)
    
    pixels_x = grid_x * scale_factor_x + (scale_factor_x * 0.5)
    pixels_y = grid_y * scale_factor_y + (scale_factor_y * 0.5)
    pixels_grid = mx.stack([pixels_x, pixels_y], axis=-1).reshape(-1, 2)

    # Cache slices
    a = inv_Sigma2D[:, 0, 0]
    b = inv_Sigma2D[:, 0, 1]
    c = inv_Sigma2D[:, 1, 0]
    d_val = inv_Sigma2D[:, 1, 1]

    u_sorted = uv_sorted[:, 0]
    v_sorted = uv_sorted[:, 1]

    # Process pixels in chunks so intermediate tensors are
    # [pixel_chunk_size, N_active] instead of [all_pixels, N_active].
    total_pixels = pixels_grid.shape[0]
    chunk_size = max(1, int(pixel_chunk_size))
    rgb_chunks = []
    alpha_chunks = []

    for start in range(0, total_pixels, chunk_size):
        end = min(start + chunk_size, total_pixels)
        pixels_chunk = pixels_grid[start:end]

        x_val = pixels_chunk[:, None, 0] - u_sorted[None, :]
        y_val = pixels_chunk[:, None, 1] - v_sorted[None, :]

        inv_S_d_x = a[None, :] * x_val + b[None, :] * y_val
        inv_S_d_y = c[None, :] * x_val + d_val[None, :] * y_val

        mahalanobis = x_val * inv_S_d_x + y_val * inv_S_d_y
        g_val = mx.exp(-0.5 * mahalanobis)

        alpha = raw_opacities[None, :] * g_val
        alpha = mx.clip(alpha, 0.0, 0.99)
        if mask is not None:
            alpha = alpha * mask[None, :]

        transmittance = mx.cumprod(1.0 - alpha, axis=1)

        ones_t = mx.ones((pixels_chunk.shape[0], 1))
        T = mx.concatenate([ones_t, transmittance[:, :-1]], axis=1)

        weights = alpha * T

        rgb_chunks.append(mx.matmul(weights, rgb_active))
        alpha_chunks.append(mx.sum(weights, axis=1, keepdims=True))

    pixel_rgb = mx.concatenate(rgb_chunks, axis=0)
    pixel_alpha = mx.concatenate(alpha_chunks, axis=0)
    
    render_colors = pixel_rgb.reshape(render_height, render_width, 3)
    render_alphas = pixel_alpha.reshape(render_height, render_width, 1)

    return render_colors, render_alphas
