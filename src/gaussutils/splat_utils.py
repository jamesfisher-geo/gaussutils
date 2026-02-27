import logging
import math
from typing import Union
from pathlib import Path

import torch
import fvdb
import fvdb_reality_capture as frc
import point_cloud_utils as pcu

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
) -> tuple[fvdb.GaussianSplat3d, frc.radiance_fields.GaussianSplatReconstruction]:
    """Load a Gaussian splat model from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"Input checkpoint file does not exist: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    runner = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
        state_dict=checkpoint
    )
    model = runner.model
    return model, runner


def save_model_ply(
    output_model: Union[str, Path],
    model: fvdb.GaussianSplat3d,
    runner: frc.radiance_fields.GaussianSplatReconstruction,
) -> None:
    """Save the trained model to disk as a PLY."""

    output_model = Path(output_model)

    if output_model.suffix.lower() != ".ply":
        raise ValueError("invalid file format.  The output file must be a PLY.")

    # Save Gaussian splat as PLY
    model.save_ply(str(output_model), metadata=runner.reconstruction_metadata)
    logger.info(f"Saved Gaussian splat PLY to {output_model}")


def save_model_usdz(
    output_model: Union[str, Path],
    model: fvdb.GaussianSplat3d,
) -> None:
    """Save the trained model to disk as a USDZ."""

    output_model = Path(output_model)

    if output_model.suffix.lower() != ".usdz":
        raise ValueError("invalid file format.  The output file must be a USDZ.")

    # Save USDZ export
    frc.tools.export_splats_to_usdz(model, out_path=str(output_model))
    logger.info(f"Saved USDZ to {output_model}")


def filter_splats(
    model: fvdb.GaussianSplat3d,
    above_scale_threshold: float = 0.05,
    below_scale_threshold: float = 0.05,
    mean_percentile: list[float] = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98],
    opacity_percentile: float = 0.98,
    decimate: int = 4,
) -> fvdb.GaussianSplat3d:
    """
    Apply a sequence of filters to a GaussianSplat3d model to remove outliers,
    floaters, transparent gaussians, and noisy micro-gaussians.

    Args:
        model: The GaussianSplat3d model to filter.
        above_scale_threshold: Remove gaussians larger than this fraction of scene scale. Default 0.05.
        below_scale_threshold: Remove gaussians smaller than this fraction of scene scale. Default 0.05.
        mean_percentile: Spatial outlier percentile bounds as [minx, maxx, miny, maxy, minz, maxz]. Default 0.98 on all axes.
        opacity_percentile: Remove gaussians below this opacity percentile. Default 0.98.
        decimate: Subsampling factor when computing percentile bounds. Default 4.

    Returns:
        filtered_model: The filtered GaussianSplat3d model.
    """
    before = model.num_gaussians
    logger.info(f"Filtering splats: starting with {before:,} gaussians")

    model = frc.tools.filter_splats_by_mean_percentile(
        model, percentile=mean_percentile, decimate=decimate
    )
    logger.info(
        f"After mean percentile filter: {model.num_gaussians:,} gaussians ({before - model.num_gaussians:,} removed)"
    )

    model = frc.tools.filter_splats_by_opacity_percentile(
        model, percentile=opacity_percentile, decimate=decimate
    )
    logger.info(
        f"After opacity percentile filter: {model.num_gaussians:,} gaussians ({before - model.num_gaussians:,} removed)"
    )

    model = frc.tools.filter_splats_above_scale(
        model, prune_scale3d_threshold=above_scale_threshold
    )
    logger.info(
        f"After above scale filter: {model.num_gaussians:,} gaussians ({before - model.num_gaussians:,} removed)"
    )

    model = frc.tools.filter_splats_below_scale(
        model, prune_scale3d_threshold=below_scale_threshold
    )

    after = model.num_gaussians
    logger.info(
        f"Filtering complete: {after:,} gaussians remaining ({before - after:,} total removed, {100 * (before - after) / before:.1f}%)"
    )

    return model


def filter_splats_by_knn_density(
    model: fvdb.GaussianSplat3d,
    k: int = 10,
    std_multiplier: float = 2.0,
) -> fvdb.GaussianSplat3d:
    """
    Remove floater gaussians using KNN density filtering.

    For each gaussian, computes the mean distance to its k nearest neighbors.
    Gaussians whose mean KNN distance exceeds (mean + std_multiplier * std) of
    the global distribution are removed as floaters — they are spatially
    isolated from the main reconstruction.

    Args:
        model: The GaussianSplat3d model to filter.
        k: Number of nearest neighbors to use. Default 10.
        std_multiplier: Gaussians with mean KNN distance greater than
                        (mean + std_multiplier * std) are removed. Lower values
                        are more aggressive. Default 2.0.

    Returns:
        filtered_model: The filtered GaussianSplat3d model.
    """
    before = model.num_gaussians
    logger.info(
        f"KNN density filter: starting with {before:,} gaussians (k={k}, std_multiplier={std_multiplier})"
    )

    positions = model.means.cpu().float().detach().numpy()
    # k+1 because pcu includes the query point itself at index 0
    dists, _ = pcu.k_nearest_neighbors(positions, positions, k + 1)
    mean_knn_dist = dists[:, 1:].mean(axis=1)

    mean_dist = mean_knn_dist.mean()
    std_dist = mean_knn_dist.std()
    threshold = mean_dist + std_multiplier * std_dist
    logger.info(
        f"KNN distance stats: mean={mean_dist:.4f}, std={std_dist:.4f}, threshold={threshold:.4f}"
    )

    mask = torch.from_numpy(mean_knn_dist < threshold).to(model.device)
    model = model[mask]

    after = model.num_gaussians
    logger.info(
        f"KNN density filter complete: {after:,} gaussians remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )
    return model


def auto_filter_splats(
    model: fvdb.GaussianSplat3d,
    scale_iqr_multiplier: float = 3.0,
    opacity_floor: float = 0.005,
    spatial_percentile: float = 0.98,
    decimate: int = 4,
    knn_k: int = 10,
    knn_std_multiplier: float = 2.0,
) -> fvdb.GaussianSplat3d:
    """
    Adaptively filter a GaussianSplat3d model by analyzing the actual
    distributions of scale and opacity rather than using fixed thresholds.

    Scale cutoffs are derived via IQR on log-scale values (gaussian scales
    are approximately log-normally distributed). Only gaussians beyond
    scale_iqr_multiplier * IQR from Q1/Q3 are removed — genuine statistical
    outliers only.

    The opacity cutoff is derived from an absolute floor: the fraction of
    gaussians below opacity_floor is computed from the data and used as the
    percentile cutoff, so only near-invisible gaussians are removed regardless
    of scene content.

    A KNN density filter removes spatial floaters by eliminating gaussians
    whose mean distance to their k nearest neighbors is a statistical outlier.

    Args:
        model: The GaussianSplat3d model to filter.
        scale_iqr_multiplier: IQR multiplier for scale outlier bounds. Higher
                              values are more permissive. Default 3.0.
        opacity_floor: Absolute minimum opacity (0-1) to retain. Default 0.005.
        spatial_percentile: Fraction of scene extent to keep spatially.
                            Default 0.95 (removes only extreme floaters).
        decimate: Subsampling factor when computing statistics. Default 4.
        knn_k: Number of nearest neighbors for density filtering. Default 10.
        knn_std_multiplier: Remove gaussians with mean KNN distance greater than
                            (mean + knn_std_multiplier * std). Lower values are
                            more aggressive. Default 2.0.

    Returns:
        filtered_model: The filtered GaussianSplat3d model.
    """
    before = model.num_gaussians
    logger.info(f"Auto-filtering splats: starting with {before} gaussians")

    # --- Adaptive scale thresholds via IQR in log space ---
    # model.scales: [N, 3], linear scale values per gaussian
    scales = model.scales[::decimate].amax(dim=-1)
    log_scales = torch.log(scales.clamp(min=1e-8))
    q1 = torch.quantile(log_scales, 0.25).item()
    q3 = torch.quantile(log_scales, 0.75).item()
    iqr = q3 - q1
    upper_scale = math.exp(q3 + scale_iqr_multiplier * iqr)
    lower_scale = math.exp(q1 - scale_iqr_multiplier * iqr)

    # Express as fraction of scene bounding box diagonal
    positions = model.means[::decimate]
    scene_scale = (positions.amax(dim=0) - positions.amin(dim=0)).norm().item()
    scene_scale = max(scene_scale, 1e-6)
    above_scale_threshold = upper_scale / scene_scale
    below_scale_threshold = lower_scale / scene_scale
    logger.info(
        f"Scale thresholds: below={below_scale_threshold:.4f}, above={above_scale_threshold:.4f} "
        f"(scene_scale={scene_scale:.4f}, IQR={iqr:.4f})"
    )

    # --- Adaptive opacity threshold from absolute floor ---
    # model.logit_opacities: [N] in logit space; sigmoid converts to [0, 1]
    opacities = torch.sigmoid(model.logit_opacities[::decimate]).squeeze()
    fraction_below_floor = (opacities < opacity_floor).float().mean().item()
    # filter_splats_by_opacity_percentile(percentile=p) keeps the top p fraction by opacity
    opacity_percentile = 1.0 - fraction_below_floor
    logger.info(
        f"Opacity: {fraction_below_floor * 100:.2f}% of gaussians below floor ({opacity_floor}), "
        f"keeping top {opacity_percentile * 100:.2f}%"
    )

    # --- Apply filters ---
    spatial_bounds = [spatial_percentile] * 6
    model = frc.tools.filter_splats_by_mean_percentile(
        model, percentile=spatial_bounds, decimate=decimate
    )
    logger.info(f"After spatial filter: {model.num_gaussians:,} gaussians")

    model = frc.tools.filter_splats_by_opacity_percentile(
        model, percentile=opacity_percentile, decimate=decimate
    )
    logger.info(f"After opacity filter: {model.num_gaussians:,} gaussians")

    model = frc.tools.filter_splats_above_scale(
        model, prune_scale3d_threshold=above_scale_threshold
    )
    logger.info(f"After above-scale filter: {model.num_gaussians:,} gaussians")

    model = frc.tools.filter_splats_below_scale(
        model, prune_scale3d_threshold=below_scale_threshold
    )
    logger.info(f"After below-scale filter: {model.num_gaussians:,} gaussians")

    model = filter_splats_by_knn_density(
        model, k=knn_k, std_multiplier=knn_std_multiplier
    )

    after = model.num_gaussians
    logger.info(
        f"Auto-filtering complete: {after:,} gaussians remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )

    return model
