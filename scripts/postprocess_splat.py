"""
Post-Process Gaussian Splat Model
=================================
Loads a checkpoint, applies conservative scene filtering to remove floaters,
and saves the cleaned model as PLY and USDZ.

Usage:
    python postprocess_splat.py --checkpoint-path model.pt
    python postprocess_splat.py --checkpoint-path model.pt --output-dir ./out
    python postprocess_splat.py --checkpoint-path model.pt --opacity-floor 0.01 --min-cluster-fraction 0.01
"""

import argparse
import logging
from pathlib import Path

from gaussutils.splat_utils import (
    filter_splats_for_scene,
    load_checkpoint,
    save_model_ply,
    save_model_usdz,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process a Gaussian splat checkpoint for scene accuracy."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model_filtered.ply",
        help="Output PLY filename (default: model_filtered.ply).",
    )
    parser.add_argument(
        "--scale-iqr-multiplier",
        type=float,
        default=4.0,
        help="IQR multiplier for scale outlier detection (default: 4.0).",
    )
    parser.add_argument(
        "--opacity-floor",
        type=float,
        default=0.002,
        help="Absolute minimum opacity to retain (default: 0.002).",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=0.99,
        help="Fraction of scene extent to retain spatially (default: 0.99).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=4,
        help="Subsampling factor for computing statistics (default: 4).",
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
        default=3.0,
        help="KNN density threshold in std devs above mean (default: 3.0).",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=20,
        help="Neighbors for cluster graph construction (default: 20).",
    )
    parser.add_argument(
        "--cluster-distance-multiplier",
        type=float,
        default=4.0,
        help="Edge threshold multiplier for clustering (default: 4.0).",
    )
    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=0.002,
        help="Minimum cluster size as fraction of total gaussians (default: 0.002).",
    )
    parser.add_argument(
        "--max-elongation",
        type=float,
        default=8.0,
        help="Reject splats where s_max / s_mid exceeds this ratio — removes needle-shaped floaters (default: 8.0).",
    )
    parser.add_argument(
        "--no-usdz",
        action="store_true",
        help="Skip USDZ export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = output_dir / args.model_name

    model, runner = load_checkpoint(checkpoint_path=checkpoint_path)

    model = filter_splats_for_scene(
        model,
        scale_iqr_multiplier=args.scale_iqr_multiplier,
        opacity_floor=args.opacity_floor,
        spatial_percentile=args.spatial_percentile,
        decimate=args.decimate,
        knn_k=args.knn_k,
        knn_std_multiplier=args.knn_std_multiplier,
        cluster_k=args.cluster_k,
        cluster_distance_multiplier=args.cluster_distance_multiplier,
        min_cluster_fraction=args.min_cluster_fraction,
        max_elongation=args.max_elongation,
    )

    save_model_ply(output_model=str(output_ply), model=model, runner=runner)

    if not args.no_usdz:
        save_model_usdz(output_model=output_ply.with_suffix(".usdz"), model=model)

    logger.info(f"Post-processing complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
