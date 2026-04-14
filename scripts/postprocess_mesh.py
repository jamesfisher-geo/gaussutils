"""
Post-Process Gaussian Splat for Mesh Extraction
================================================
Loads a checkpoint, applies aggressive filtering to strip background and
floaters, then extracts a mesh via DLNR or TSDF fusion.

Requires a COLMAP dataset for camera frustum culling and mesh extraction.

Usage:
    python postprocess_mesh.py --checkpoint-path model.pt --dataset-path /path/to/colmap
    python postprocess_mesh.py --checkpoint-path model.pt --dataset-path /path/to/colmap --no-dlnr
    python postprocess_mesh.py --checkpoint-path model.pt --dataset-path /path/to/colmap --opacity-floor 0.1 --min-cluster-fraction 0.1
"""

import argparse
import logging
from pathlib import Path

import torch

from gaussutils.mesh_utils import extract_mesh, save_mesh
from gaussutils.scene_utils import load_scene, preprocess_scene
from gaussutils.splat_utils import (
    filter_splats_for_mesh,
    load_checkpoint,
    save_model_ply,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process a Gaussian splat checkpoint and extract a mesh."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to COLMAP dataset directory (needed for cameras and meshing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="mesh.ply",
        help="Output mesh filename (default: mesh.ply).",
    )
    parser.add_argument(
        "--save-filtered-ply",
        action="store_true",
        help="Also save the filtered splat model as PLY before meshing.",
    )
    # Scene preprocessing
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Image downsample factor (default: 4).",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=3.0,
        help="Min percentile for point filtering (default: 3.0).",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=97.0,
        help="Max percentile for point filtering (default: 97.0).",
    )
    parser.add_argument(
        "--min-points-per-image",
        type=int,
        default=50,
        help="Remove images with fewer visible points (default: 50).",
    )
    # Mesh extraction
    parser.add_argument(
        "--truncation-margin",
        type=float,
        default=0.5,
        help="TSDF truncation margin (default: 0.5).",
    )
    parser.add_argument(
        "--no-dlnr",
        action="store_true",
        help="Use basic TSDF fusion instead of DLNR stereo depth.",
    )
    parser.add_argument(
        "--grid-shell-thickness",
        type=float,
        default=3.0,
        help="VDB grid shell thickness around the surface (default: 3.0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for DLNR depth generation (default: 4).",
    )
    # Splat filtering
    parser.add_argument(
        "--scale-iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for scale outlier detection (default: 3.0).",
    )
    parser.add_argument(
        "--opacity-floor",
        type=float,
        default=0.01,
        help="Minimum opacity to retain. Kept low so semi-transparent splats still contribute to DLNR rendered views (default: 0.01).",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=0.98,
        help="Fraction of scene extent to retain (default: 0.98).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=4,
        help="Subsampling factor for statistics (default: 4).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="Neighbors for KNN density filter (default: 10).",
    )
    parser.add_argument(
        "--knn-std-multiplier",
        type=float,
        default=2.0,
        help="KNN density threshold in std devs (default: 2.0).",
    )
    parser.add_argument(
        "--min-visible-views",
        type=int,
        default=2,
        help="Minimum cameras a splat must be visible in (default: 2).",
    )
    parser.add_argument(
        "--georeferenced",
        action="store_true",
        help="Set if the input COLMAP dataset is ECEF aligned. Coordinates wil be normalized to ENU. Apply the inverse of the output transform to place the outputs back into ECEF space",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    checkpoint_path = Path(args.checkpoint_path)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required.")

    if args.georeferenced:
        normalization_type = "ecef2enu"
    else:
        normalization_type = "pca"

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load scene for frustum culling and mesh extraction
    scene = load_scene(dataset_path=dataset_path, normalization_type=normalization_type)
    scene = preprocess_scene(
        scene=scene,
        downsample=args.downsample,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
        min_points=args.min_points_per_image,
    )

    # Load and filter the model
    model, runner = load_checkpoint(checkpoint_path=checkpoint_path)

    model = filter_splats_for_mesh(
        model,
        scene=scene,
        scale_iqr_multiplier=args.scale_iqr_multiplier,
        opacity_floor=args.opacity_floor,
        spatial_percentile=args.spatial_percentile,
        decimate=args.decimate,
        knn_k=args.knn_k,
        knn_std_multiplier=args.knn_std_multiplier,
        min_visible_views=args.min_visible_views,
    )

    if args.save_filtered_ply:
        filtered_ply = output_dir / "model_mesh_filtered.ply"
        save_model_ply(output_model=str(filtered_ply), model=model, runner=runner)

    # Extract and save mesh
    dlnr = not args.no_dlnr
    if not dlnr:
        logger.warning("DLNR stereo depth disabled. Mesh quality may be reduced.")

    vertices, faces, colors = extract_mesh(
        model=model,
        scene=scene,
        truncation_margin=args.truncation_margin,
        use_dlnr=dlnr,
        grid_shell_thickness=args.grid_shell_thickness,
        num_workers=args.num_workers,
    )

    output_mesh = output_dir / args.mesh_name
    save_mesh(
        output_mesh=output_mesh,
        vertices=vertices,
        faces=faces,
        colors=colors,
    )

    logger.info(f"Mesh pipeline complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
