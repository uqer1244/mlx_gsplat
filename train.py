import os
import argparse
import sys

# Bind local package path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mlx_gsplat.trainer import RunnerMLX, MLXConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX UMA-Native 3DGS Trainer")
    parser.add_argument("--data_dir", type=str, default="data/garden", help="Path to COLMAP scene directory (e.g. data/garden)")
    parser.add_argument("--max_steps", type=int, default=2500, help="Maximum number of training steps")
    parser.add_argument("--sh_degree_interval", type=int, default=500, help="SH Schedule interval")
    parser.add_argument("--data_factor", type=int, default=4, help="Dataset downsample factor (e.g. 4 or 8)")
    parser.add_argument("--render_width", type=int, default=200, help="Rasterization render width")
    parser.add_argument("--render_height", type=int, default=150, help="Rasterization render height")
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
