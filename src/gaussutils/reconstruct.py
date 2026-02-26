"""
3D Gaussian Splat Reconstruction Pipeline
==========================================
Loads a COLMAP dataset, trains a Gaussian splat radiance field,
and extracts a high-quality mesh using fVDB Reality Capture.

Usage:
    python reconstruct.py /path/to/colmap/dataset
    python reconstruct.py /path/to/colmap/dataset --output-dir ./results --downsample 4
    python reconstruct.py /path/to/colmap/dataset --no-dlnr --truncation-margin 0.25
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import fvdb
import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D Gaussian Splat and mesh from a COLMAP dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to COLMAP dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Image downsample factor for training (default: 4).",
    )
    parser.add_argument(
        "--truncation-margin",
        type=float,
        default=0.5,
        help="TSDF truncation margin for meshing (default: 0.5).",
    )
    parser.add_argument(
        "--no-dlnr",
        action="store_true",
        help="Use basic TSDF fusion instead of DLNR stereo depth for meshing.",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=3.0,
        help="Min percentile for outlier point filtering (default: 3.0).",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=97.0,
        help="Max percentile for outlier point filtering (default: 97.0).",
    )
    parser.add_argument(
        "--min-points-per-image",
        type=int,
        default=50,
        help="Remove images with fewer visible points than this (default: 50).",
    )
    parser.add_argument(
        "--viz-port",
        type=int,
        default=None,
        help="Enable fvdb.viz visualization server on this port (e.g. 8080).",
    )
    return parser.parse_args()


def load_scene(dataset_path: str) -> frc.sfm_scene.SfmScene:
    """Load a COLMAP dataset and apply cleanup transforms."""

    logging.info(f"Loading COLMAP dataset from {dataset_path}")
    sfm_scene = frc.sfm_scene.SfmScene.from_colmap(dataset_path)
    logging.info(
        f"Loaded scene: {len(sfm_scene.images)} images, "
        f"{len(sfm_scene.points)} points, "
        f"{len(sfm_scene.cameras)} camera(s)"
    )

    return sfm_scene


def clean_scene(
    scene: frc.sfm_scene.SfmScene,
    downsample: int,
    percentile_min: float = 3.0,
    percentile_max: float = 97.0,
    min_points: int = 50,
) -> frc.sfm_scene.SfmScene:

    logging.info("Applying cleanup and preprocessing transforms...")
    cleanup_transform = transforms.Compose(
        transforms.DownsampleImages(
            image_downsample_factor=downsample,
            image_type="jpg",
            rescaled_jpeg_quality=95,
        ),
        transforms.NormalizeScene(normalization_type="pca"),
        transforms.PercentileFilterPoints(
            percentile_min=percentile_min,
            percentile_max=percentile_max,
        ),
        transforms.FilterImagesWithLowPoints(min_num_points=min_points),
    )
    cleaned_scene = cleanup_transform(scene)
    logging.info(
        f"Original scene had {len(scene.points)} points and {len(scene.images)} images"
    )
    logging.info(
        f"After cleanup: {len(cleaned_scene.images)} images, "
        f"{len(cleaned_scene.points)} points"
    )

    return cleaned_scene


def train_gaussian_splat(
    scene: frc.sfm_scene.SfmScene,
) -> tuple[fvdb.GaussianSplat3d, frc.radiance_fields.GaussianSplatReconstruction]:
    """Train a Gaussian splat radiance field from an SfmScene."""

    logging.info("Initializing Gaussian splat reconstruction...")
    runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(scene)

    logging.info("Starting optimization (this may take a while)...")
    runner.optimize()

    model = runner.model
    logging.info(
        f"Training complete: {model.num_gaussians:,} Gaussians, "
        f"device={model.device}"
    )

    return model, runner


def extract_mesh(
    model: fvdb.GaussianSplat3d,
    scene: frc.sfm_scene.SfmScene,
    truncation_margin: float,
    use_dlnr: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract a triangle mesh from a trained Gaussian splat model."""

    camera_to_world = scene.camera_to_world_matrices
    projection = scene.projection_matrices
    image_sizes = scene.image_sizes

    if use_dlnr:
        logging.info("Extracting mesh using DLNR stereo depth estimation...")
        v, f, c = frc.tools.mesh_from_splats_dlnr(
            model, camera_to_world, projection, image_sizes, truncation_margin
        )
    else:
        logging.info("Extracting mesh using basic TSDF fusion...")
        v, f, c = frc.tools.mesh_from_splats(
            model, camera_to_world, projection, image_sizes, truncation_margin
        )

    logging.info(f"Mesh extracted: {v.shape[0]:,} vertices, {f.shape[0]:,} faces")
    return v, f, c


def save_outputs(
    model: fvdb.GaussianSplat3d,
    runner: frc.radiance_fields.GaussianSplatReconstruction,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    output_dir: str,
):
    """Save the trained model and extracted mesh to disk."""
    import point_cloud_utils as pcu

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save Gaussian splat as PLY
    splat_path = output_path / "model.ply"
    model.save_ply(str(splat_path), metadata=runner.reconstruction_metadata)
    logging.info(f"Saved Gaussian splat PLY to {splat_path}")

    # Save mesh as PLY
    mesh_path = output_path / "mesh.ply"
    pcu.save_mesh_vfc(
        str(mesh_path),
        vertices.cpu().numpy(),
        faces.cpu().numpy(),
        colors.cpu().numpy(),
    )
    logging.info(f"Saved mesh PLY to {mesh_path}")

    # Save USDZ export
    try:
        usdz_path = output_path / "model.usdz"
        frc.tools.export_splats_to_usdz(model, out_path=str(usdz_path))
        logging.info(f"Saved USDZ to {usdz_path}")
    except Exception as e:
        logging.warning(f"USDZ export failed (non-critical): {e}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate input
    if not Path(args.dataset_path).is_dir():
        logging.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        logging.error("CUDA is not available. A CUDA-capable GPU is required.")
        sys.exit(1)

    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Optional visualization server
    if args.viz_port is not None:
        fvdb.viz.init(port=args.viz_port)
        logging.info(f"Visualization server started on port {args.viz_port}")

    # 1. Load and clean the dataset
    scene = load_scene(
        dataset_path=args.dataset_path,
    )
    scene = clean_scene(
        scene=scene,
        downsample=args.downsample,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
        min_points=args.min_points_per_image,
    )

    # 2. Train the Gaussian splat model
    model, runner = train_gaussian_splat(scene)

    # 3. Extract a mesh
    vertices, faces, colors = extract_mesh(
        model=model,
        scene=scene,
        truncation_margin=args.truncation_margin,
        use_dlnr=not args.no_dlnr,
    )

    # 4. Save everything
    save_outputs(model, runner, vertices, faces, colors, args.output_dir)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
