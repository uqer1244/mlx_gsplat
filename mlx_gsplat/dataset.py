import os
import json
import struct
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import mlx.core as mx

# ==============================================================================
# 1. World Space Normalization Utilities (Pure NumPy)
# ==============================================================================

def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    centroid = np.median(point_cloud, axis=0)
    translated_point_cloud = point_cloud - centroid
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def transform_points(matrix, points):
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


# ==============================================================================
# 2. Pure Python/NumPy COLMAP Binary Reader
# ==============================================================================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    MODEL_NUM_PARAMS = {
        0: 3,  # SIMPLE_PINHOLE
        1: 4,  # PINHOLE
        2: 4,  # SIMPLE_RADIAL
        3: 5,  # RADIAL
        4: 8,  # OPENCV
        5: 8   # OPENCV_FISHEYE
    }
    
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "i")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            
            num_params = MODEL_NUM_PARAMS.get(model_id, 4)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            model_names = {
                0: "SIMPLE_PINHOLE", 1: "PINHOLE", 
                2: "SIMPLE_RADIAL", 3: "RADIAL", 
                4: "OPENCV", 5: "OPENCV_FISHEYE"
            }
            model_name = model_names.get(model_id, "PINHOLE")
            
            class Camera:
                def __init__(self, cid, name, w, h, p):
                    self.camera_id = cid
                    self.camera_type = name
                    self.width = w
                    self.height = h
                    self.params = p
                    
                    if name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                        self.fx, self.fy = p[0], p[0]
                        self.cx, self.cy = p[1], p[2]
                    else:
                        self.fx, self.fy = p[0], p[1]
                        self.cx, self.cy = p[2], p[3]
                        
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"))
            tvec = np.array(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 4, "i")[0]
            
            image_name = ""
            char = fid.read(1)
            while char != b"\x00":
                image_name += char.decode("utf-8")
                char = fid.read(1)
                
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)
            
            qvec = qvec / np.linalg.norm(qvec)
            w, x, y, z = qvec
            R = np.array([
                [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
            ])
            
            class ColmapImage:
                def __init__(self, iid, q, t, cid, name, rot):
                    self.image_id = iid
                    self.qvec = q
                    self.tvec = t
                    self.camera_id = cid
                    self.name = name
                    self.rot = rot
                def R(self):
                    return self.rot
            
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, image_name, R)
    return images


def read_points3D_binary(path_to_model_file):
    points = []
    colors = []
    errors = []
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_properties = read_next_bytes(fid, 43, "QdddBBBd")
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            
            track_len = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_len, 1)
            
            points.append(xyz)
            colors.append(rgb)
            errors.append(error)
            
    return (
        np.array(points, dtype=np.float32),
        np.array(colors, dtype=np.uint8),
        np.array(errors, dtype=np.float32)
    )


def _get_rel_paths(path_dir: str) -> List[str]:
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    print(f"[Dataset] Downscaling images by {factor}x from {image_dir} to {resized_dir}")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


# ==============================================================================
# 3. Main Natively-Parsed Dataset & Parser
# ==============================================================================

class Parser:
    """Pure Python COLMAP parser (Zero pycolmap SceneManager Dependency)."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        test_after: int = -1,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.test_after = test_after

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP sparse directory {colmap_dir} does not exist."

        print(f"[Parser] Loading Colmap binary models from: {colmap_dir}")
        cameras_file = os.path.join(colmap_dir, "cameras.bin")
        images_file = os.path.join(colmap_dir, "images.bin")
        points_file = os.path.join(colmap_dir, "points3D.bin")
        
        cameras_dict = read_cameras_binary(cameras_file)
        images_dict = read_images_binary(images_file)
        points_xyz, points_rgb, points_err = read_points3D_binary(points_file)

        # Extract extrinsic matrices in world-to-camera format
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        for k in images_dict:
            im = images_dict[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = cameras_dict[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            params_dict[camera_id] = np.array(cam.params, dtype=np.float32)
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
            
        print(f"[Parser] Successfully parsed {len(images_dict)} images, taken by {len(cameras_dict)} cameras.")

        if len(images_dict) == 0:
            raise ValueError("No images found in COLMAP sparse reconstruct.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        image_names = [images_dict[k].name for k in images_dict]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        
        # Load images
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        if not os.path.exists(colmap_image_dir):
            raise ValueError(f"Image folder {colmap_image_dir} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(colmap_image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        points = points_xyz
        points_err = points_err
        points_rgb = points_rgb
        point_indices = dict()

        # Normalize the world space
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        # Adjust intrinsics with actual downsampled image height / width
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # Size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A pure NumPy/MLX dataset loader class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        preload_images: bool = True,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.preload_images = preload_images
        indices = np.arange(len(self.parser.image_names))

        if self.parser.test_every != -1:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        elif self.parser.test_after != -1:
            if split == "train":
                self.indices = indices[indices <= self.parser.test_after]
            else:
                self.indices = indices[indices > self.parser.test_after]
        else:
            self.indices = indices

        self.preloaded_images = {}
        if self.preload_images:
            # Preload and cache all images into CPU RAM (uint8) to eliminate training disk I/O and PNG decoding.
            print(f"\n[Dataset] Preloading {len(self.indices)} images into CPU RAM (uint8)...")
            for idx in tqdm(self.indices):
                img_path = self.parser.image_paths[idx]
                image = imageio.imread(img_path)[..., :3]
                self.preloaded_images[idx] = np.array(image, dtype=np.uint8)
        else:
            print(f"\n[Dataset] Lazy image loading enabled for {len(self.indices)} images.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, mx.array]:
        index = self.indices[item]
        if self.preload_images:
            image = mx.array(self.preloaded_images[index], dtype=mx.float32)
        else:
            image = mx.array(imageio.imread(self.parser.image_paths[index])[..., :3], dtype=mx.float32)
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        # Return native mx.arrays directly (Zero-Copy UMA compliant)
        data = {
            "K": mx.array(K, dtype=mx.float32),
            "camtoworld": mx.array(camtoworlds, dtype=mx.float32),
            "image": image,
            "image_id": item,
        }
        return data
