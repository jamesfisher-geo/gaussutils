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
from pathlib import Path
from typing import Union

import torch
import fvdb
import fvdb_reality_capture as frc

from gaussutils.scene_utils import load_scene, clean_scene
from gaussutils.mesh_utils import extract_mesh, save_mesh
from gaussutils.splat_utils import save_model_ply, save_model_usdz, auto_filter_splats

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
    parser.add_argument(
        "--viz-port",
        type=int,
        default=None,
        help="Enable fvdb.viz visualization server on this port (e.g. 8080).",
    )
    parser.add_argument(
        "--scale-iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for scale outlier detection. Higher = more permissive (default: 3.0).",
    )
    parser.add_argument(
        "--opacity-floor",
        type=float,
        default=0.005,
        help="Absolute minimum opacity (0-1) to retain (default: 0.005).",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=0.98,
        help="Fraction of scene extent to retain spatially (default: 0.98).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=4,
        help="Subsampling factor when computing filter statistics (default: 4).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="Number of nearest neighbors for KNN density floater removal (default: 10).",
    )
    parser.add_argument(
        "--knn-std-multiplier",
        type=float,
        default=2.0,
        help="KNN density threshold in standard deviations above mean. Lower = more aggressive (default: 2.0).",
    )
    return parser.parse_args()


def train_gaussian_splat(
    scene: frc.sfm_scene.SfmScene,
    output_dir: Union[str, Path],
) -> tuple[fvdb.GaussianSplat3d, frc.radiance_fields.GaussianSplatReconstruction]:
    """Train a Gaussian splat radiance field from an SfmScene."""
    logger.info("Initializing Gaussian splat reconstruction...")
    output_dir = Path(output_dir)
    writer_dir = output_dir / "info"
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = frc.radiance_fields.GaussianSplatReconstructionWriter(
        run_name=None,
        save_path=writer_dir,
        config=frc.radiance_fields.GaussianSplatReconstructionWriterConfig(
            save_checkpoints=True, save_plys=True, save_metrics=False
        ),
    )

    runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
        scene, writer=writer
    )
    logger.info("Starting optimization (this may take a while)...")
    runner.optimize()

    model = runner.model
    logger.info(
        f"Training complete: {model.num_gaussians} Gaussians, " f"device={model.device}"
    )
    return model, runner


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

    dlnr = not args.no_dlnr
    if not dlnr:
        logger.warning("DLNR stereo depth disabled. Mesh quality may be reduced.")

    downsample = int(args.downsample)
    truncation_margin = float(args.truncation_margin)
    percentile_min = float(args.percentile_min)
    percentile_max = float(args.percentile_max)
    min_pts_per_image = int(args.min_points_per_image)
    scale_iqr_multiplier = float(args.scale_iqr_multiplier)
    opacity_floor = float(args.opacity_floor)
    spatial_percentile = float(args.spatial_percentile)
    decimate = int(args.decimate)
    knn_k = int(args.knn_k)
    knn_std_multiplier = float(args.knn_std_multiplier)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required.")

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Optional visualization server
    if args.viz_port is not None:
        fvdb.viz.init(port=args.viz_port)
        logger.info(f"Visualization server started on port {args.viz_port}")

    # Load and clean the dataset
    scene = load_scene(
        dataset_path=dataset_path,
    )
    scene = clean_scene(
        scene=scene,
        downsample=downsample,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        min_points=min_pts_per_image,
    )

    # Train the Gaussian splat model
    model, runner = train_gaussian_splat(scene=scene, output_dir=output_dir)

    # Filter the model
    model = auto_filter_splats(
        model=model,
        scale_iqr_multiplier=scale_iqr_multiplier,
        opacity_floor=opacity_floor,
        spatial_percentile=spatial_percentile,
        decimate=decimate,
        knn_k=knn_k,
        knn_std_multiplier=knn_std_multiplier,
    )

    # Save the model
    output_model_ply = output_dir / "model.ply"
    save_model_ply(output_model=str(output_model_ply), model=model, runner=runner)
    save_model_usdz(output_model=output_model_ply.with_suffix(".usdz"), model=model)

    # Extract a mesh
    vertices, faces, colors = extract_mesh(
        model=model,
        scene=scene,
        truncation_margin=truncation_margin,
        use_dlnr=dlnr,
    )

    output_mesh = output_dir / "mesh.ply"
    save_mesh(output_mesh=output_mesh, vertices=vertices, faces=faces, colors=colors)

    logger.info(f"Pipeline complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
