import logging
from pathlib import Path
from typing import Literal, Optional, Union

import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as transforms

logger = logging.getLogger(__name__)


def load_scene(
    dataset_path: Union[str, Path],
    normalization_type: Optional[Literal["pca", "ecef2enu"]] = None,
) -> frc.sfm_scene.SfmScene:
    """Load a COLMAP dataset and apply cleanup transforms."""

    logger.info(f"Loading COLMAP dataset from {dataset_path}")
    scene = frc.sfm_scene.SfmScene.from_colmap(str(dataset_path))
    if normalization_type:
        logger.info(f"Normalizing scene coordinates using '{normalization_type}'")
        normalization_transform = transforms.Compose(
            transforms.NormalizeScene(normalization_type=normalization_type),
        )
        scene = normalization_transform(scene)

    logger.info(
        f"Loaded scene: {len(scene.images)} images, "
        f"{len(scene.points)} points, "
        f"{len(scene.cameras)} camera(s)"
    )

    return scene


def preprocess_scene(
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
