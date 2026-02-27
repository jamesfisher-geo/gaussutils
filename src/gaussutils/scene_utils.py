import logging
from typing import Union
from pathlib import Path

import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as transforms

logger = logging.getLogger(__name__)


def load_scene(dataset_path: Union[str, Path]) -> frc.sfm_scene.SfmScene:
    """Load a COLMAP dataset and apply cleanup transforms."""

    logger.info(f"Loading COLMAP dataset from {dataset_path}")
    sfm_scene = frc.sfm_scene.SfmScene.from_colmap(str(dataset_path))
    logger.info(
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

    logger.info("Applying cleanup and preprocessing transforms...")
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
    logger.info(
        f"Original scene had {len(scene.points)} points and {len(scene.images)} images"
    )
    logger.info(
        f"After cleanup: {len(cleaned_scene.images)} images, "
        f"{len(cleaned_scene.points)} points"
    )

    return cleaned_scene
