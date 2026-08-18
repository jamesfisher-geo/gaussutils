from typing import Literal, Optional, Union
import logging
from pathlib import Path
import json

import numpy as np

import fvdb
import torch
import fvdb_reality_capture as frc
from gaussutils.scene import ColmapScene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class GaussianSplatTrainer:
    """Trains, filters, and saves a Gaussian splat radiance field from a ColmapScene.

    Wraps `frc.radiance_fields.GaussianSplatReconstruction`: `train()` runs the
    optimization loop, `filter_model()` removes floaters/outliers from the result,
    and `save_ply()` / `save_usdz()` / `save_georef()` write outputs to `output_dir`.
    """

    def __init__(
        self,
        scene: ColmapScene,
        output_dir: Union[Path, str],
        run_name: str = None,
        max_gaussians: int = -1,
        save_plys: bool = False,
        save_checkpoints: bool = False,
        save_metrics: bool = False,
        deletion_opacity_threshold: float = 0.005,
        opacity_regularization: float = 0.01,
        scale_regularization: float = 0.01,
        sh_degree: int = 3,
        optimizer_type: Literal["mcmc", "original"] = "mcmc",
        model_bbox_margin: float = 5.0,
        spatial_percentile: tuple[float, float, float, float, float, float] = (
            0.95,
            0.95,
            0.95,
            0.95,
            0.98,
            0.98,
        ),  # (minx, maxx, miny, maxy, minz, maxz)
        opacity_percentile: float = 0.98,
        scale_min_percentile: float = 0.02,
        scale_max_percentile: float = 0.98,
    ):
        """Configure a trainer for a given scene.

        Args:
            scene: The ColmapScene to train on. Its `filter_scene()` should
                   already have been called.
            output_dir: Directory to write checkpoints, PLYs, and other outputs to.
            run_name: Name for this training run, used to name output files
                      (e.g. "{run_name}.ply"). Default None.
            max_gaussians: Maximum number of gaussians to allow during training.
                           -1 for unlimited. Default -1.
            save_plys: Save intermediate PLY checkpoints during training. Default False.
            save_checkpoints: Save intermediate .pt checkpoints during training. Default False.
            save_metrics: Save training metrics. Default False.
            deletion_opacity_threshold: Gaussians with opacity below this are pruned
                                        during optimization. Default 0.005.
            opacity_regularization: MCMC opacity regularization weight. Ignored by
                                    the "original" optimizer. Default 0.01.
            scale_regularization: MCMC scale regularization weight. Ignored by the
                                  "original" optimizer. Default 0.01.
            sh_degree: Spherical harmonics degree for view-dependent color. Default 3.
            optimizer_type: "mcmc" or "original" 3DGS optimizer. Default "mcmc".
            model_bbox_margin: Margin in meters added around the scene's point cloud
                               bounding box when cropping gaussians in `filter_model()`.
                               Default 5.0.
            spatial_percentile: Spatial outlier percentile bounds passed to
                                `filter_model()` as (minx, maxx, miny, maxy, minz, maxz).
                                Default (0.95, 0.95, 0.95, 0.95, 0.98, 0.98).
            opacity_percentile: Opacity percentile cutoff used in `filter_model()`.
                                Default 0.98.
            scale_min_percentile: Lower scale percentile cutoff used in `filter_model()`.
                                  Default 0.02.
            scale_max_percentile: Upper scale percentile cutoff used in `filter_model()`.
                                  Default 0.98.
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        self.model_ply = self.output_dir / f"{self.run_name}.ply"
        self.georef_json = self.output_dir / "model.georef.json"

        self.max_gaussians = max_gaussians
        self.save_plys = save_plys
        self.save_checkpoints = save_checkpoints
        self.save_metrics = save_metrics

        self.deletion_opacity_threshold = deletion_opacity_threshold
        self.opacity_regularization = opacity_regularization
        self.scale_regularization = scale_regularization
        self.sh_degree = sh_degree
        self.optimizer_type = optimizer_type

        self.scene = scene.scene
        self.transform_matrix = scene.scene.transformation_matrix
        self.normalization_type = scene.normalization_type

        self.model: Optional[fvdb.GaussianSplat3d] = None
        self.runner: Optional[frc.radiance_fields.GaussianSplatReconstruction] = None

        # post-filtering
        self.opacity_percentile = opacity_percentile
        self.spatial_percentile = spatial_percentile
        self.model_bbox_margin = model_bbox_margin
        self.scale_min_percentile = scale_min_percentile
        self.scale_max_percentile = scale_max_percentile

    def _require_scene(self) -> None:
        """Raise if no input scene is set."""
        if not self.scene:
            raise ValueError("Missing input scene")

    def _require_model(self) -> None:
        """Raise if the model has not been trained or loaded yet."""
        if not self.model:
            raise ValueError(
                "Missing 3DGS model. Train a model or load one form a checkpoint with .load_checkpoint()"
            )

    def train(self) -> None:
        """Train the 3DGS model from scratch on `self.scene`.

        Builds a `GaussianSplatReconstruction` using the configured optimizer
        (`self.optimizer_type`) and runs `optimize()` to completion. Sets
        `self.model` and `self.runner` on success.
        """
        writer_dir = self.output_dir / "info"
        writer_dir.mkdir(parents=True, exist_ok=True)

        writer = frc.radiance_fields.GaussianSplatReconstructionWriter(
            run_name=self.run_name,
            save_path=writer_dir,
            exist_ok=True,
            config=frc.radiance_fields.GaussianSplatReconstructionWriterConfig(
                save_checkpoints=self.save_checkpoints,
                save_plys=self.save_plys,
                save_metrics=self.save_metrics,
            ),
        )

        self._require_scene()
        if not self.scene.images:
            raise ValueError("Input scene has no images.")

        config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            sh_degree=self.sh_degree,
            remove_gaussians_outside_scene_bbox=True,
            save_at_percent=[25, 50, 100],
        )

        if self.optimizer_type == "mcmc":
            config.remove_gaussians_outside_scene_bbox = (
                False  # MUST stay False with the MCMC optimizer
            )
            optimizer_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
                deletion_opacity_threshold=self.deletion_opacity_threshold,
                max_gaussians=self.max_gaussians,
                opacity_regularization=self.opacity_regularization,
                scale_regularization=self.scale_regularization,
            )
        else:
            logger.info(
                f"Using the Original 3DGS optimizer. Ignoring "
                f"'opacity_regularization' {self.opacity_regularization} and "
                f"'scale_regularization' {self.scale_regularization}."
            )
            optimizer_config = frc.radiance_fields.GaussianSplatOptimizerConfig(
                deletion_opacity_threshold=self.deletion_opacity_threshold,
                max_gaussians=self.max_gaussians,
            )

        if not self.runner:
            logger.info(
                f"Initializing reconstruction with {self.optimizer_type} optimizer"
            )
            self.runner = (
                frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
                    self.scene,
                    writer=writer,
                    config=config,
                    optimizer_config=optimizer_config,
                )
            )

        logger.info("Starting training...")
        self.runner.optimize()

        self.model = self.runner.model
        logger.info(
            f"Training complete: {self.model.num_gaussians:,} gaussians, "
            f"device={self.model.device}"
        )

    def filter_model(self) -> None:
        """Remove floaters and outliers from the trained model."""

        self._require_scene()
        self._require_model()
        before = self.model.num_gaussians
        logger.info(f"Filtering: starting with {before:,} gaussians")

        decimate = 4

        pts = self.scene.points
        lo = torch.tensor(
            pts.min(axis=0) - self.model_bbox_margin,
            dtype=self.model.means.dtype,
            device=self.model.device,
        )
        hi = torch.tensor(
            pts.max(axis=0) + self.model_bbox_margin,
            dtype=self.model.means.dtype,
            device=self.model.device,
        )
        mask = ((self.model.means >= lo) & (self.model.means <= hi)).all(dim=1)
        self.model = self.model[mask]
        logger.info(
            f"After bbox crop (margin={self.model_bbox_margin}m): "
            f"{self.model.num_gaussians:,} gaussians"
        )

        self.model = frc.tools.filter_splats_by_mean_percentile(
            self.model,
            percentile=self.spatial_percentile,
            decimate=decimate,
        )
        logger.info(f"After spatial filter: {self.model.num_gaussians:,} gaussians")

        self.model = frc.tools.filter_splats_by_opacity_percentile(
            self.model, percentile=self.opacity_percentile, decimate=decimate
        )
        logger.info(f"After opacity filter: {self.model.num_gaussians:,} gaussians")

        scales_max = self.model.scales.amax(dim=-1)  # (N,) activated, meters
        sample = scales_max[::decimate]
        hi_scale = torch.quantile(sample, self.scale_max_percentile).item()
        lo_scale = torch.quantile(sample, self.scale_min_percentile).item()
        keep = (scales_max < hi_scale) & (scales_max > lo_scale)
        logger.info(
            f"Scale filter: keeping ({lo_scale:.4g}, {hi_scale:.4g}) m max-axis "
            f"[p{self.scale_min_percentile * 100:g}, p{self.scale_max_percentile * 100:g}], "
            f"observed max={scales_max.max().item():.4g} m"
        )
        self.model = self.model[keep]
        logger.info(f"After scale filter: {self.model.num_gaussians:,} gaussians")

        after = self.model.num_gaussians
        logger.info(
            f"Filtering complete: {after:,} remaining "
            f"({before - after:,} removed, {100 * (before - after) / before:.1f}%)"
        )

    def save_ply(self) -> None:
        """Save the 3DGS model as a PLY file in ENU space."""

        self._require_model()
        if self.runner is not None:
            self.runner.save_ply(str(self.model_ply))
        else:
            logger.warning(
                "No runner available (model loaded from PLY); "
                "saving without reconstruction metadata"
            )
            self.model.save_ply(str(self.model_ply))
        logger.info(f"Saved 3DGS PLY to {self.model_ply}")

    def save_usdz(self) -> None:
        """Save the trained model to disk as a USDZ."""

        self._require_model()
        output_model = Path(self.model_ply).with_suffix(".usdz")

        if self.runner is not None:
            self.runner.save_usd(str(output_model), usdz=True)
        else:
            logger.warning(
                "No runner available (model loaded from PLY); "
                "exporting USDZ directly from the model"
            )
            frc.tools.export_splats_to_usd(self.model, str(output_model), usdz=True)
        logger.info(f"Saved USDZ to {output_model}")

    def save_georef(self) -> None:
        """Save georef sidecar JSON describing the model's ENU coordinate frame.

        Uses the same schema as the input georef.json, where "local" is the
        model.ply ENU frame.
        """
        if self.normalization_type != "ecef2enu":
            logger.info(
                "Model was not input with real-world coordinates. Skipping georeference..."
            )
            return
        data = {
            "coordinate_system": "ENU",
            "epsg": 4978,
            "enu_to_ecef_matrix": np.linalg.inv(self.transform_matrix).tolist(),
            "note": (
                "Model coordinates are ENU meters. Apply enu_to_ecef_matrix to "
                "convert model (ENU) coordinates to ECEF (EPSG:4978)."
            ),
        }
        self.georef_json.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved georef sidecar to {self.georef_json}")
