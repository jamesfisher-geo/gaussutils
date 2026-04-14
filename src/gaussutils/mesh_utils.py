import logging
from pathlib import Path
from typing import Union

import fvdb
import fvdb_reality_capture as frc
import point_cloud_utils as pcu
import torch

logger = logging.getLogger(__name__)


def extract_mesh(
    model: fvdb.GaussianSplat3d,
    scene: frc.sfm_scene.SfmScene,
    truncation_margin: float,
    use_dlnr: bool,
    grid_shell_thickness: float = 3.0,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract a triangle mesh from a trained Gaussian splat model.

    Args:
        model: The GaussianSplat3d model.
        scene: SfmScene providing camera matrices.
        truncation_margin: TSDF truncation margin.
        use_dlnr: Use DLNR stereo depth (True) or basic TSDF (False).
        grid_shell_thickness: VDB grid shell thickness around the surface. Default 3.0.
        dtype: Compute precision for TSDF. Default torch.float32.
        num_workers: DataLoader workers for DLNR. Default 4.
    """

    camera_to_world = scene.camera_to_world_matrices
    projection = scene.projection_matrices
    image_sizes = scene.image_sizes

    # Free VRAM from prior filtering/training before heavy mesh extraction
    torch.cuda.empty_cache()

    if use_dlnr:
        logging.info("Extracting mesh using DLNR stereo depth estimation...")
        v, f, c = frc.tools.mesh_from_splats_dlnr(
            model,
            camera_to_world,
            projection,
            image_sizes,
            truncation_margin,
            grid_shell_thickness=grid_shell_thickness,
            dtype=dtype,
            num_workers=num_workers,
        )
    else:
        logging.info("Extracting mesh using basic TSDF fusion...")
        v, f, c = frc.tools.mesh_from_splats(
            model,
            camera_to_world,
            projection,
            image_sizes,
            truncation_margin,
            grid_shell_thickness=grid_shell_thickness,
            dtype=dtype,
        )

    logging.info(f"Mesh extracted: {v.shape[0]:,} vertices, {f.shape[0]:,} faces")
    return v, f, c


def save_mesh(
    output_mesh: Union[str, Path],
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
) -> None:
    """
    Save a mesh. Supported formats are PLY, OBJ, STL, OFF, VRML 2.0, X3D, COLLADA, 3DS.
    """
    output_mesh = Path(output_mesh)
    pcu.save_mesh_vfc(
        str(output_mesh),
        vertices.cpu().numpy(),
        faces.cpu().numpy(),
        colors.cpu().numpy(),
    )
    logger.info(f"Saved mesh to {output_mesh}")
