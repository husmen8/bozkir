"""bozkır - procedural terrain from captured Gaussian splats."""

from .ply import Splats, load_ply, save_ply, quat_to_matrix, SH_C0
from .scene import SceneConfig, prepare, add_scene_args, scene_from_args
from .transform import align_to_ground, ground_normal, rotate, recentre
from .select import crop_box, crop_cylinder, remove_large, remove_floaters
from .camera import Camera, orbit_camera, project_perspective, render_perspective
from .render import (project_orthographic, rasterize, rasterize_rgba,
                     render_orthographic, over, flatten)
from .tile import (extract_patch, translate, merge, grid,
                   render_global, render_tiled, seam_camera)

__all__ = [
    "Splats", "load_ply", "save_ply", "quat_to_matrix", "SH_C0",
    "SceneConfig", "prepare", "add_scene_args", "scene_from_args",
    "align_to_ground", "ground_normal", "rotate", "recentre",
    "crop_box", "crop_cylinder", "remove_large", "remove_floaters",
    "Camera", "orbit_camera", "project_perspective", "render_perspective",
    "project_orthographic", "rasterize", "rasterize_rgba",
    "render_orthographic", "over", "flatten",
    "extract_patch", "translate", "merge", "grid",
    "render_global", "render_tiled", "seam_camera",
]