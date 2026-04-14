import logging
import math
from pathlib import Path
from typing import Optional, Union

import fvdb
import fvdb_reality_capture as frc
import numpy as np
import point_cloud_utils as pcu
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


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


def load_checkpoint(
    checkpoint_path: Union[str, Path],
) -> tuple[
    fvdb.GaussianSplat3d, Optional[frc.radiance_fields.GaussianSplatReconstruction]
]:
    """Load a Gaussian splat model from a checkpoint (.pt) or PLY (.ply) file.

    When loading from PLY, no runner is available so the second return value is None.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"Input checkpoint file does not exist: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if checkpoint_path.suffix.lower() == ".ply":
        model, metadata = fvdb.GaussianSplat3d.from_ply(
            str(checkpoint_path), device="cuda"
        )
        logger.info(f"Loaded PLY: {model.num_gaussians:,} gaussians")
        return model, None

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    runner = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
        state_dict=checkpoint
    )
    model = runner.model
    return model, runner


def save_model_ply(
    output_model: Union[str, Path],
    model: fvdb.GaussianSplat3d,
    runner: Optional[frc.radiance_fields.GaussianSplatReconstruction] = None,
) -> None:
    """Save the trained model to disk as a PLY."""

    output_model = Path(output_model)

    if output_model.suffix.lower() != ".ply":
        raise ValueError("invalid file format.  The output file must be a PLY.")

    metadata = runner.reconstruction_metadata if runner is not None else None
    model.save_ply(str(output_model), metadata=metadata)
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


def filter_splats_by_cluster(
    model: fvdb.GaussianSplat3d,
    k: int = 20,
    distance_multiplier: float = 2.0,
    min_cluster_fraction: float = 0.005,
) -> fvdb.GaussianSplat3d:
    """Remove spatially isolated clusters using connected components on a KNN graph.

    Builds a KNN graph over gaussian means, thresholds edges by an adaptive
    distance cutoff (median nearest-neighbor distance + multiplier * MAD),
    then finds connected components. Only clusters containing at least
    min_cluster_fraction of total gaussians are retained.

    Args:
        model: The GaussianSplat3d model to filter.
        k: Neighbors per gaussian in the KNN graph. Default 20.
        distance_multiplier: Edge threshold = median_nn_dist + multiplier * MAD.
                             Lower values produce more disconnected components. Default 2.0.
        min_cluster_fraction: Minimum cluster size as fraction of total gaussians.
                              Clusters below this are removed. Default 0.005 (0.5%).

    Returns:
        filtered_model: Model with only significant clusters retained.
    """
    before = model.num_gaussians
    if before <= 1:
        return model

    logger.info(
        f"Cluster filter: {before:,} gaussians, k={k}, "
        f"distance_multiplier={distance_multiplier}, min_fraction={min_cluster_fraction}"
    )

    positions = model.means.cpu().float().detach().numpy()
    n = len(positions)

    # KNN graph (k+1 because pcu includes self at index 0)
    dists, indices = pcu.k_nearest_neighbors(positions, positions, k + 1)
    dists = dists[:, 1:]
    indices = indices[:, 1:]

    # Adaptive distance threshold: median + multiplier * MAD of nearest-neighbor distances
    nn1_dists = dists[:, 0]
    median_dist = np.median(nn1_dists)
    mad = np.median(np.abs(nn1_dists - median_dist))
    threshold = median_dist + distance_multiplier * max(mad, 1e-8)
    logger.info(
        f"Edge threshold: {threshold:.6f} "
        f"(median_nn={median_dist:.6f}, MAD={mad:.6f})"
    )

    # Build sparse adjacency matrix (vectorized)
    row_idx = np.repeat(np.arange(n), k)
    col_idx = indices.ravel()
    dist_flat = dists.ravel()
    edge_mask = dist_flat < threshold
    graph = csr_matrix(
        (
            np.ones(edge_mask.sum(), dtype=np.float32),
            (row_idx[edge_mask], col_idx[edge_mask]),
        ),
        shape=(n, n),
    )

    n_components, labels = connected_components(graph, directed=False)

    # Keep clusters above size threshold
    min_size = max(int(n * min_cluster_fraction), 1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    keep_labels = set(unique_labels[counts >= min_size])
    keep_mask = np.isin(labels, list(keep_labels))

    logger.info(
        f"Found {n_components} components, keeping {len(keep_labels)} "
        f"with >= {min_size} gaussians"
    )

    model = model[torch.from_numpy(keep_mask).to(model.device)]
    after = model.num_gaussians
    logger.info(
        f"Cluster filter complete: {after:,} remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )
    return model


def filter_splats_by_camera_frustum(
    model: fvdb.GaussianSplat3d,
    scene: frc.sfm_scene.SfmScene,
    min_visible_views: int = 2,
) -> fvdb.GaussianSplat3d:
    """Remove splats not visible in at least min_visible_views training cameras.

    Projects gaussian means into each training camera and checks whether
    they fall within the image bounds and in front of the camera. Splats
    not visible in enough views are removed.

    Args:
        model: The GaussianSplat3d model to filter.
        scene: SfmScene providing camera poses and intrinsics.
        min_visible_views: Minimum cameras a splat must be visible in. Default 2.

    Returns:
        filtered_model: Model with only multi-view-visible splats.
    """
    before = model.num_gaussians
    if before == 0:
        return model

    cam2world = torch.as_tensor(
        scene.camera_to_world_matrices, dtype=torch.float32, device=model.device
    )
    proj = torch.as_tensor(
        scene.projection_matrices, dtype=torch.float32, device=model.device
    )
    img_sizes = torch.as_tensor(
        scene.image_sizes, dtype=torch.float32, device=model.device
    )

    n_cams = cam2world.shape[0]
    if n_cams == 0:
        logger.warning("No cameras in scene, skipping frustum filter")
        return model

    min_visible_views = min(min_visible_views, n_cams)
    logger.info(
        f"Camera frustum filter: {before:,} gaussians, "
        f"{n_cams} cameras, min_visible_views={min_visible_views}"
    )

    world2cam = torch.linalg.inv(cam2world)  # (N_cams, 4, 4)
    means = model.means.detach()  # (N_splats, 3)
    ones = torch.ones(means.shape[0], 1, device=means.device, dtype=means.dtype)
    means_h = torch.cat([means, ones], dim=-1)  # (N_splats, 4)

    visible_count = torch.zeros(means.shape[0], device=means.device, dtype=torch.int32)

    for i in range(n_cams):
        # Transform to camera space
        cam_pts = (world2cam[i] @ means_h.T).T[:, :3]  # (N_splats, 3)

        in_front = cam_pts[:, 2] > 0

        # Project to pixel coordinates: K @ p_cam
        proj_pts = (proj[i] @ cam_pts.T).T
        z = proj_pts[:, 2:3].clamp(min=1e-8)
        uv = proj_pts[:, :2] / z

        w, h = img_sizes[i, 0], img_sizes[i, 1]
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

        visible_count += (in_front & in_bounds).int()

    mask = visible_count >= min_visible_views
    model = model[mask]

    after = model.num_gaussians
    logger.info(
        f"Camera frustum filter complete: {after:,} remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )
    return model


def filter_splats_by_anisotropy(
    model: fvdb.GaussianSplat3d,
    max_elongation: float = 8.0,
) -> fvdb.GaussianSplat3d:
    """Remove needle-like splats via scale anisotropy.

    Legitimate surface splats tend to be flat discs (s_min small, s_mid ~ s_max),
    whereas a common floater class after 3DGS training is needle-shaped: one
    axis much larger than the other two. This filter removes any gaussian
    whose largest-to-middle axis ratio exceeds ``max_elongation``.

    Args:
        model: The GaussianSplat3d model to filter.
        max_elongation: Reject splats where ``s_max / s_mid > max_elongation``.
                        Default 8.0 — a plate with 8× aspect is extreme.

    Returns:
        filtered_model: Model with needle-shaped splats removed.
    """
    before = model.num_gaussians
    if before == 0:
        return model

    scales = model.scales.detach()  # (N, 3)
    sorted_scales, _ = torch.sort(scales, dim=-1, descending=True)
    s_max = sorted_scales[:, 0].clamp(min=1e-8)
    s_mid = sorted_scales[:, 1].clamp(min=1e-8)
    elongation = s_max / s_mid

    mask = elongation <= max_elongation
    model = model[mask]

    after = model.num_gaussians
    logger.info(
        f"Anisotropy filter complete: {after:,} remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%, "
        f"max_elongation={max_elongation})"
    )
    return model


def filter_splats_for_scene(
    model: fvdb.GaussianSplat3d,
    scale_iqr_multiplier: float = 4.0,
    opacity_floor: float = 0.002,
    spatial_percentile: float = 0.99,
    decimate: int = 4,
    knn_k: int = 10,
    knn_std_multiplier: float = 3.0,
    cluster_k: int = 20,
    cluster_distance_multiplier: float = 4.0,
    min_cluster_fraction: float = 0.002,
    max_elongation: float = 8.0,
) -> fvdb.GaussianSplat3d:
    """Conservative filtering pipeline for accurate scene representation.

    Applies adaptive statistical filters (scale IQR, opacity floor, spatial
    percentile, KNN density), a shape-based needle rejection, and
    connected-component cluster analysis to remove floaters while preserving
    as much legitimate scene content as possible.

    Use this when the goal is a visually accurate splat model with minimal
    manual cleanup.

    Args:
        model: The GaussianSplat3d model to filter.
        scale_iqr_multiplier: IQR multiplier for scale outlier bounds. Default 4.0.
        opacity_floor: Absolute minimum opacity to retain. Default 0.002.
        spatial_percentile: Fraction of scene extent to keep. Default 0.99.
        decimate: Subsampling factor for statistics. Default 4.
        knn_k: Neighbors for KNN density filter. Default 10.
        knn_std_multiplier: KNN outlier threshold in std devs. Default 3.0.
        cluster_k: Neighbors for cluster graph. Default 20.
        cluster_distance_multiplier: Edge threshold multiplier for clustering. Default 4.0.
        min_cluster_fraction: Minimum cluster size as fraction of total. Default 0.002.
        max_elongation: Reject splats with ``s_max / s_mid > max_elongation``. Default 8.0.

    Returns:
        filtered_model: Cleaned model with floaters removed.
    """
    before = model.num_gaussians
    logger.info(f"filter_splats_for_scene: starting with {before:,} gaussians")

    model = auto_filter_splats(
        model,
        scale_iqr_multiplier=scale_iqr_multiplier,
        opacity_floor=opacity_floor,
        spatial_percentile=spatial_percentile,
        decimate=decimate,
        knn_k=knn_k,
        knn_std_multiplier=knn_std_multiplier,
    )

    model = filter_splats_by_anisotropy(model, max_elongation=max_elongation)

    model = filter_splats_by_cluster(
        model,
        k=cluster_k,
        distance_multiplier=cluster_distance_multiplier,
        min_cluster_fraction=min_cluster_fraction,
    )

    after = model.num_gaussians
    logger.info(
        f"filter_splats_for_scene complete: {after:,} remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )
    return model


def filter_splats_for_mesh(
    model: fvdb.GaussianSplat3d,
    scene: Optional[frc.sfm_scene.SfmScene] = None,
    scale_iqr_multiplier: float = 3.0,
    opacity_floor: float = 0.01,
    spatial_percentile: float = 0.98,
    decimate: int = 4,
    knn_k: int = 10,
    knn_std_multiplier: float = 2.0,
    min_visible_views: int = 2,
) -> fvdb.GaussianSplat3d:
    """Conservative filtering pipeline for mesh extraction.

    Removes obvious floaters and far-away outliers while preserving the
    dense surface coverage (including low-opacity splats) that DLNR stereo
    depth estimation needs to render coherent image pairs. Cluster-based
    filtering is intentionally omitted: 3DGS surfaces often fragment into
    many weakly-connected KNN components, and dropping small components
    strips the splats that carry fine surface detail.

    Args:
        model: The GaussianSplat3d model to filter.
        scene: Optional SfmScene for camera frustum culling. When provided,
               splats not visible in min_visible_views cameras are removed.
        scale_iqr_multiplier: IQR multiplier for scale outlier bounds. Default 3.0.
        opacity_floor: Minimum opacity to retain. Default 0.01 — low enough to
                       keep semi-transparent splats that contribute to rendered
                       alpha compositing (critical for DLNR stereo).
        spatial_percentile: Spatial extent to keep. Default 0.98.
        decimate: Subsampling factor for statistics. Default 4.
        knn_k: Neighbors for KNN density filter. Default 10.
        knn_std_multiplier: KNN threshold in std devs. Default 2.0.
        min_visible_views: Minimum cameras a splat must project into. Default 2.

    Returns:
        filtered_model: Cleaned model ready for meshing, preserving surface coverage.
    """
    before = model.num_gaussians
    logger.info(f"filter_splats_for_mesh: starting with {before:,} gaussians")

    # Adaptive statistical filters (scale IQR, opacity floor, spatial percentile,
    # single KNN density pass). Conservative defaults — surface coverage matters
    # more than aesthetic cleanliness for DLNR-based meshing.
    model = auto_filter_splats(
        model,
        scale_iqr_multiplier=scale_iqr_multiplier,
        opacity_floor=opacity_floor,
        spatial_percentile=spatial_percentile,
        decimate=decimate,
        knn_k=knn_k,
        knn_std_multiplier=knn_std_multiplier,
    )

    # Camera frustum culling removes splats no camera can see — safe for meshing.
    if scene is not None:
        model = filter_splats_by_camera_frustum(
            model, scene, min_visible_views=min_visible_views
        )
    else:
        logger.info("No scene provided, skipping camera frustum filter")

    after = model.num_gaussians
    logger.info(
        f"filter_splats_for_mesh complete: {after:,} remaining "
        f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
    )
    return model
