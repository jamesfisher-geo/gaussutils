"""
3D Gaussian Splat Reconstruction Pipeline
==========================================
Loads a COLMAP dataset, trains a Gaussian splat radiance field,
and extracts a high-quality mesh using fVDB Reality Capture.

Usage:
    python reconstruct.py /path/to/colmap/dataset
    python reconstruct.py /path/to/colmap/dataset --output-dir ./results --downsample 4
"""

import argparse
import logging
from pathlib import Path

import torch

from gaussutils.georef_utils import save_georef_transform
from gaussutils.scene_utils import load_scene, preprocess_scene
from gaussutils.splat_utils import save_model_ply, save_model_usdz, train_gaussian_splat

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D Gaussian Splat and mesh from a COLMAP dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to COLMAP dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        default=None,
        help="Directory for output files (default: <dataset-path>/output).",
    )
    parser.add_argument(
        "--georeferenced",
        action="store_true",
        help="Set if the input COLMAP dataset is ECEF aligned. Coordinates wil be normalized to ENU. Apply the inverse of the output transform to place the outputs back into ECEF space",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Image downsample factor for training (default: 4).",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=3.0,
        help="Min percentile for outlier point filtering before training (default: 3.0).",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=97.0,
        help="Max percentile for outlier point filtering before training (default: 97.0).",
    )
    parser.add_argument(
        "--min-points-per-image",
        type=int,
        default=50,
        help="Remove images with fewer visible points than this (default: 50).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset_path = Path(args.dataset_path)

    # Validate input
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    if not args.output_dir:
        output_dir = dataset_path / "output"
    else:
        output_dir = Path(args.output_dir)
    logger.info(f"Setting output directory: {output_dir}")

    downsample = int(args.downsample)
    percentile_min = float(args.percentile_min)
    percentile_max = float(args.percentile_max)
    min_pts_per_image = int(args.min_points_per_image)

    if args.georeferenced:
        normalization_type = "ecef2enu"
    else:
        normalization_type = "pca"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required.")

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load and clean the dataset
    scene = load_scene(
        dataset_path=dataset_path,
        normalization_type=normalization_type,
    )
    scene = preprocess_scene(
        scene=scene,
        downsample=downsample,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        min_points=min_pts_per_image,
    )
    if args.georeferenced:
        save_georef_transform(
            ecef_to_enu=scene.transformation_matrix,
            output_path=Path(output_dir / "enu_to_ecef.json"),
        )

    # Train the Gaussian splat model
    model, runner = train_gaussian_splat(
        scene=scene,
        output_dir=output_dir,
    )

    # Save the model
    output_model_ply = output_dir / "model.ply"
    save_model_ply(output_model=str(output_model_ply), model=model, runner=runner)
    save_model_usdz(output_model=output_model_ply.with_suffix(".usdz"), model=model)

    logger.info(f"Pipeline complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
