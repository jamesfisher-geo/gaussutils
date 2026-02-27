import argparse
import logging
from pathlib import Path

import torch
from gaussutils.scene_utils import load_scene, clean_scene
from gaussutils.mesh_utils import extract_mesh, save_mesh
from gaussutils.splat_utils import load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Load a checkpoint and save as PLY.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to COLMAP dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output PLY (default: current directory).",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="mesh.ply",
        help="Name for the output mesh",
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
    mesh_name = args.mesh_name
    if not mesh_name.lower().endswith("ply"):
        raise ValueError(f"Invalid mesh filename. Must be a PLY: {mesh_name}")
    downsample = int(args.downsample)
    truncation_margin = float(args.truncation_margin)
    percentile_min = float(args.percentile_min)
    percentile_max = float(args.percentile_max)
    min_pts_per_image = int(args.min_points_per_image)
    dlnr = not args.no_dlnr
    if not dlnr:
        logger.warning("DLNR stereo depth disabled. Mesh quality may be reduced.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required.")

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

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

    logger.info(f"Loading checkpoint file from {checkpoint_path}")

    model, _ = load_checkpoint(checkpoint_path=checkpoint_path)

    # Extract a mesh
    vertices, faces, colors = extract_mesh(
        model=model,
        scene=scene,
        truncation_margin=truncation_margin,
        use_dlnr=dlnr,
    )

    output_mesh = output_dir / mesh_name
    save_mesh(output_mesh=output_mesh, vertices=vertices, faces=faces, colors=colors)


if __name__ == "__main__":
    main()
