import logging
from pathlib import Path
from typing import Literal, Optional, Union

import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as transforms
from fvdb import CameraModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ColmapScene:
    """A COLMAP Structure-from-Motion scene, loaded and cleaned for training.

    On construction, loads the raw COLMAP dataset via `frc.sfm_scene.SfmScene.from_colmap`.
    Call `filter_scene()` to apply downsampling, undistortion, normalization, and
    outlier/low-coverage filtering before handing the scene to a `GaussianSplatTrainer`.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        normalization_type: Literal["pca", "none", "ecef2enu", "similarity"] = (
            "ecef2enu"
        ),
        image_downsample_factor: int = 1,
        percentile_min: tuple[float, float, float] = (5.0, 5.0, 1.0),  # X, Y, Z
        percentile_max: tuple[float, float, float] = (95.0, 95.0, 100.0),  # X, Y, Z
        scene_crop_margin: float = 0.05,
        min_points_per_image: int = 50,
    ):
        """Load a COLMAP dataset from disk.

        Args:
            dataset_path: Path to the COLMAP dataset directory.
            normalization_type: Scene normalization to apply in `filter_scene()`.
                                 "ecef2enu" for georeferenced (ECEF) inputs, "pca" for
                                 arbitrary scenes, "none" to skip normalization, or
                                 "similarity". Default "ecef2enu".
            image_downsample_factor: Factor to downsample training images by. Default 1.
            percentile_min: Lower percentile bound for point outlier filtering, per
                            axis as (X, Y, Z). Default (5.0, 5.0, 1.0).
            percentile_max: Upper percentile bound for point outlier filtering, per
                            axis as (X, Y, Z). Default (95.0, 95.0, 100.0).
            scene_crop_margin: Margin to pad the scene bounding box by when cropping
                               to points, as a fraction of scene extent. Default 0.05.
            min_points_per_image: Remove images with fewer visible points than this.
                                  Default 50.
        """

        self.dataset_path = Path(dataset_path)
        self.normalization_type = normalization_type
        self.image_downsample_factor = image_downsample_factor
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.scene_crop_margin = scene_crop_margin
        self.min_points_per_image = min_points_per_image

        self.scene: Optional[frc.sfm_scene.SfmScene] = None

        self.load()

    def _check_scene(self) -> None:
        """Raise if the scene has not been loaded."""
        if not self.scene:
            raise ValueError("Scene not available")

    def load(self) -> None:
        """Load the dataset from `self.dataset_path` into `self.scene`."""
        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        logger.info(f"Loading dataset from {self.dataset_path}")
        self.scene = frc.sfm_scene.SfmScene.from_colmap(str(self.dataset_path))
        logger.info(
            f"Input scene: {self.scene.num_images} images from {self.scene.num_cameras} with {len(self.scene.points)} points"
        )

    def filter_scene(self) -> None:
        """Apply the cleanup pipeline to the loaded scene.

        Runs:
         - image downsampling
         - image undistortion
         - scene coordinate normalization using `self.normalization_type`
         - outlier filtering,
         - cropping the scene to the filtered points
         - removal of images with too few visible points.

        Raises:
          - `RuntimeError` if any non-pinhole cameras remain after undistortion, since training with OpenCV
            camera models is silently broken in fvdb 0.5.0.
        """

        cleanup = transforms.Compose(
            transforms.DownsampleImages(
                image_downsample_factor=self.image_downsample_factor
            ),
            transforms.UndistortImages(),
            transforms.NormalizeScene(normalization_type=self.normalization_type),
            transforms.PercentileFilterPoints(
                percentile_min=self.percentile_min,
                percentile_max=self.percentile_max,
            ),
            transforms.CropSceneToPoints(margin=self.scene_crop_margin),
            transforms.FilterImagesWithLowPoints(
                min_num_points=self.min_points_per_image
            ),
        )
        self.scene = cleanup(self.scene)

        non_pinhole = {
            str(cam.camera_model)
            for cam in self.scene.cameras.values()
            if cam.camera_model != CameraModel.PINHOLE
        }
        if non_pinhole:
            raise RuntimeError(
                f"Scene still contains non-pinhole cameras after undistortion: "
                f"{sorted(non_pinhole)}. Training with OpenCV camera models"
            )

        logger.info(
            f"Scene after cleanup: "
            f"{self.scene.num_images} images,"
            f"{self.scene.num_cameras} cameras,"
            f"{len(self.scene.points)} points"
        )
