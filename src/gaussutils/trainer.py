import logging
from pathlib import Path
from typing import Literal, Optional, Union

import fvdb
import fvdb_reality_capture as frc
import numpy as np
import torch

from gaussutils.scene import ColmapScene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class GaussianSplatModelTrainer:
    """Fits, cleans, and exports a 3DGS model for a ColmapScene.

    Attributes:
        splats: The fitted model, or None before `fit()`.
        reconstruction: The frc reconstruction runner, or None before `fit()`.
        ply_path: Output PLY path.
    """

    def __init__(
        self,
        colmap_scene: ColmapScene,
        out_dir: Union[Path, str],
        run_name: str = None,
        max_gaussians: int = -1,
        write_intermediate_plys: bool = False,
        write_checkpoints: bool = False,
        write_metrics: bool = False,
        prune_opacity_min: float = 0.005,
        opacity_reg_weight: float = 0.01,
        scale_reg_weight: float = 0.01,
        sh_degree: int = 3,
        optimizer_type: Literal["mcmc", "original"] = "mcmc",
        crop_margin_m: float = 5.0,
        spatial_bounds_pct: tuple[float, float, float, float, float, float] = (
            0.95,
            0.95,
            0.95,
            0.95,
            0.98,
            0.98,
        ),  # order: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
        opacity_keep_frac: float = 0.98,
    ):
        """Initializes the trainer.

        Args:
            colmap_scene: Scene to fit; call its `filter_scene()` first.
            out_dir: Output directory.
            run_name: Run label; names output files.
            max_gaussians: Gaussian count limit; -1 for unlimited.
            write_intermediate_plys: Save PLY snapshots during training.
            write_checkpoints: Save .pt snapshots during training.
            write_metrics: Save training metrics.
            prune_opacity_min: Opacity below which gaussians are pruned.
            opacity_reg_weight: MCMC opacity regularization weight.
            scale_reg_weight: MCMC scale regularization weight.
            sh_degree: Spherical harmonics degree.
            optimizer_type: "mcmc" or "original".
            crop_margin_m: Bounding-box crop margin in meters.
            spatial_bounds_pct: Per-axis percentile bounds
                (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
            opacity_keep_frac: Opacity percentile cutoff.
        """
        self.out_dir = Path(out_dir)
        self.run_name = run_name

        self.ply_path = self.out_dir / f"{self.run_name}.ply"
        self.georef_path = self.out_dir / "model.georef.json"

        self.max_gaussians = max_gaussians
        self.write_intermediate_plys = write_intermediate_plys
        self.write_checkpoints = write_checkpoints
        self.write_metrics = write_metrics

        self.prune_opacity_min = prune_opacity_min
        self.opacity_reg_weight = opacity_reg_weight
        self.scale_reg_weight = scale_reg_weight
        self.sh_degree = sh_degree
        self.optimizer_type = optimizer_type

        self.sfm_scene = colmap_scene.scene
        self.world_transform = colmap_scene.scene.transformation_matrix
        self.norm_mode = colmap_scene.normalization_type

        self.splats: Optional[fvdb.GaussianSplat3d] = None
        self.reconstruction: Optional[
            frc.radiance_fields.GaussianSplatReconstruction
        ] = None

        # cleanup settings used by clean_splats()
        self.opacity_keep_frac = opacity_keep_frac
        self.spatial_bounds_pct = spatial_bounds_pct
        self.crop_margin_m = crop_margin_m

    def _check_scene(self) -> None:
        """Checks that a scene is set.

        Raises:
            ValueError: If no scene is set.
        """
        if not self.sfm_scene:
            raise ValueError("No SfM scene set")

    def _check_splats(self) -> None:
        """Checks that a model is fitted or loaded.

        Raises:
            ValueError: If no model is available.
        """
        if not self.splats:
            raise ValueError(
                "No splat model available. Run .fit() or load one from a checkpoint first"
            )

    def _make_optim_cfg(self):
        """Builds the optimizer config.

        Returns:
            An MCMC or original optimizer config, per `optimizer_type`.
        """
        if self.optimizer_type == "mcmc":
            return frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
                deletion_opacity_threshold=self.prune_opacity_min,
                max_gaussians=self.max_gaussians,
                opacity_regularization=self.opacity_reg_weight,
                scale_regularization=self.scale_reg_weight,
            )

        logger.info(
            f"Original 3DGS optimizer selected; opacity_reg_weight="
            f"{self.opacity_reg_weight} and scale_reg_weight="
            f"{self.scale_reg_weight} will not be used."
        )
        return frc.radiance_fields.GaussianSplatOptimizerConfig(
            deletion_opacity_threshold=self.prune_opacity_min,
            max_gaussians=self.max_gaussians,
        )

    def fit(self) -> None:
        """Trains the model on the scene; sets `splats` and `reconstruction`.

        Raises:
            ValueError: If the scene is missing or has no images.
        """
        self._check_scene()
        if not self.sfm_scene.images:
            raise ValueError("SfM scene contains no images.")

        info_dir = self.out_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)

        recon_writer = frc.radiance_fields.GaussianSplatReconstructionWriter(
            run_name=self.run_name,
            save_path=info_dir,
            exist_ok=True,
            config=frc.radiance_fields.GaussianSplatReconstructionWriterConfig(
                save_checkpoints=self.write_checkpoints,
                save_plys=self.write_intermediate_plys,
                save_metrics=self.write_metrics,
            ),
        )

        recon_cfg = frc.radiance_fields.GaussianSplatReconstructionConfig(
            sh_degree=self.sh_degree,
            remove_gaussians_outside_scene_bbox=True,
            save_at_percent=[25, 50, 100],
        )
        if self.optimizer_type == "mcmc":
            # MCMC requires bbox-based removal to be disabled
            recon_cfg.remove_gaussians_outside_scene_bbox = False
        optim_cfg = self._make_optim_cfg()

        if not self.reconstruction:
            logger.info(f"Setting up reconstruction ({self.optimizer_type} optimizer)")
            self.reconstruction = (
                frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
                    self.sfm_scene,
                    writer=recon_writer,
                    config=recon_cfg,
                    optimizer_config=optim_cfg,
                )
            )

        logger.info("Optimization started")
        self.reconstruction.optimize()

        self.splats = self.reconstruction.model
        logger.info(
            f"Optimization finished: {self.splats.num_gaussians:,} gaussians "
            f"on {self.splats.device}"
        )

    def clean_splats(self) -> None:
        """Removes floaters via bounding-box crop and opacity filter.

        Raises:
            ValueError: If the scene or model is missing.
        """

        self._check_scene()
        self._check_splats()
        n_start = self.splats.num_gaussians
        logger.info(f"Cleanup: {n_start:,} gaussians before filtering")

        decimation = 4

        scene_pts = self.sfm_scene.points
        bbox_min, bbox_max = torch.tensor(
            np.stack(
                [
                    scene_pts.min(axis=0) - self.crop_margin_m,
                    scene_pts.max(axis=0) + self.crop_margin_m,
                ]
            ),
            dtype=self.splats.means.dtype,
            device=self.splats.device,
        )
        inside = (
            (self.splats.means >= bbox_min) & (self.splats.means <= bbox_max)
        ).all(dim=1)
        self.splats = self.splats[inside]
        logger.info(
            f"Bounding-box crop (margin {self.crop_margin_m}m) left "
            f"{self.splats.num_gaussians:,} gaussians"
        )

        self.splats = frc.tools.filter_splats_by_opacity_percentile(
            self.splats, percentile=self.opacity_keep_frac, decimate=decimation
        )
        logger.info(f"Opacity filter left {self.splats.num_gaussians:,} gaussians")

        n_end = self.splats.num_gaussians
        n_removed = n_start - n_end
        logger.info(
            f"Cleanup done: kept {n_end:,}, dropped {n_removed:,} "
            f"({100 * n_removed / n_start:.1f}%)"
        )

    def export_ply(self) -> None:
        """Saves the model as a PLY to `ply_path`.

        Raises:
            ValueError: If no model is available.
        """

        self._check_splats()
        if self.reconstruction is None:
            logger.warning(
                "Model was not produced by a reconstruction (e.g. loaded from PLY); "
                "writing PLY without reconstruction metadata"
            )
            self.splats.save_ply(str(self.ply_path))
        else:
            self.reconstruction.save_ply(str(self.ply_path))
        logger.info(f"Wrote splat PLY: {self.ply_path}")

    def export_usdz(self) -> None:
        """Saves the model as a USDZ next to `ply_path`.

        Raises:
            ValueError: If no model is available.
        """

        self._check_splats()
        usdz_path = Path(self.ply_path).with_suffix(".usdz")

        if self.reconstruction is None:
            logger.warning(
                "Model was not produced by a reconstruction (e.g. loaded from PLY); "
                "exporting USDZ straight from the splats"
            )
            frc.tools.export_splats_to_usd(self.splats, str(usdz_path), usdz=True)
        else:
            self.reconstruction.save_usd(str(usdz_path), usdz=True)
        logger.info(f"Wrote USDZ: {usdz_path}")
