import argparse
import logging
from pathlib import Path

from gaussutils.splat_utils import auto_filter_splats, load_checkpoint, save_model_ply

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Load a checkpoint and save as PLY.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output PLY (default: current directory).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model.ply",
        help="Name for the output 3dgs model",
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
        help="Absolute minimum opacity (0-1) to retain. Gaussians below this are near-invisible (default: 0.005).",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=0.99,
        help="Fraction of scene extent to retain spatially. Removes only extreme floaters (default: 0.99).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=4,
        help="Subsampling factor when computing statistics (default: 4).",
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


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    model_name = args.model_name
    scale_iqr_multiplier = float(args.scale_iqr_multiplier)
    opacity_floor = float(args.opacity_floor)
    spatial_percentile = float(args.spatial_percentile)
    decimate = int(args.decimate)
    knn_k = int(args.knn_k)
    knn_std_multiplier = float(args.knn_std_multiplier)
    output_model = output_dir / model_name

    logger.info(f"Loading checkpoint file from {checkpoint_path}")

    model, runner = load_checkpoint(checkpoint_path=checkpoint_path)

    model = auto_filter_splats(
        model=model,
        scale_iqr_multiplier=scale_iqr_multiplier,
        opacity_floor=opacity_floor,
        spatial_percentile=spatial_percentile,
        decimate=decimate,
        knn_k=knn_k,
        knn_std_multiplier=knn_std_multiplier,
    )

    save_model_ply(output_model=str(output_model), model=model, runner=runner)


if __name__ == "__main__":
    main()
