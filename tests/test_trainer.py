from pathlib import Path

import numpy as np
import pytest
import torch

from gaussutils.trainer import GaussianSplatModelTrainer
from tests.helpers import FakeColmapScene, FakeGaussianSplat3d, make_fake_scene


def _make_trainer(tmp_path, run_name="model", **kwargs):
    fake_scene = make_fake_scene()
    colmap_scene = FakeColmapScene(
        fake_scene, normalization_type=kwargs.pop("normalization_type", "ecef2enu")
    )
    return GaussianSplatModelTrainer(
        colmap_scene=colmap_scene, out_dir=tmp_path, run_name=run_name, **kwargs
    )


# --- __init__ ---


def test_init_coerces_output_dir_to_path(tmp_path):
    trainer = _make_trainer(str(tmp_path))
    assert isinstance(trainer.out_dir, Path)
    assert trainer.out_dir == tmp_path


def test_init_builds_ply_and_georef_paths(tmp_path):
    trainer = _make_trainer(tmp_path, run_name="myrun")
    assert trainer.ply_path == tmp_path / "myrun.ply"
    assert trainer.georef_path == tmp_path / "model.georef.json"


def test_init_stores_prune_opacity_min_as_plain_float(tmp_path):
    trainer = _make_trainer(tmp_path, prune_opacity_min=0.01)
    assert trainer.prune_opacity_min == 0.01
    assert not isinstance(trainer.prune_opacity_min, tuple)


def test_init_splats_and_reconstruction_start_none(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert trainer.splats is None
    assert trainer.reconstruction is None


def test_init_pulls_attrs_from_colmap_scene(tmp_path):
    fake_scene = make_fake_scene(transformation_matrix=np.eye(4) * 2)
    colmap_scene = FakeColmapScene(fake_scene, normalization_type="similarity")
    trainer = GaussianSplatModelTrainer(
        colmap_scene=colmap_scene, out_dir=tmp_path, run_name="model"
    )
    assert trainer.sfm_scene is fake_scene
    assert (trainer.world_transform == fake_scene.transformation_matrix).all()
    assert trainer.norm_mode == "similarity"


def test_init_stores_percentile_and_bbox_params(tmp_path):
    trainer = _make_trainer(
        tmp_path,
        scale_min_percentile=0.1,
        scale_max_percentile=0.9,
        opacity_keep_frac=0.5,
        crop_margin_m=2.0,
    )
    assert trainer.scale_min_percentile == 0.1
    assert trainer.scale_max_percentile == 0.9
    assert trainer.opacity_keep_frac == 0.5
    assert trainer.crop_margin_m == 2.0


# --- _check_scene / _check_splats ---


def test_check_scene_raises_when_unset(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.sfm_scene = None
    with pytest.raises(ValueError, match="No SfM scene set"):
        trainer._check_scene()


def test_check_scene_noop_when_set(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer._check_scene()  # should not raise


def test_check_splats_raises_when_unset(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No splat model available"):
        trainer._check_splats()


def test_check_splats_noop_when_set(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.splats = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer._check_splats()  # should not raise


# --- fit() ---


def test_fit_raises_when_scene_has_no_images(tmp_path, frc_mock):
    fake_scene = make_fake_scene(num_images=0)
    colmap_scene = FakeColmapScene(fake_scene)
    trainer = GaussianSplatModelTrainer(
        colmap_scene=colmap_scene, out_dir=tmp_path, run_name="model"
    )
    with pytest.raises(ValueError, match="no images"):
        trainer.fit()


def test_fit_mcmc_forces_bbox_removal_false(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path, optimizer_type="mcmc")
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.fit()

    config_mock = (
        frc_mock.radiance_fields.GaussianSplatReconstructionConfig.return_value
    )
    assert config_mock.remove_gaussians_outside_scene_bbox is False
    call_kwargs = (
        frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.call_args.kwargs
    )
    assert call_kwargs["config"] is config_mock
    assert trainer.splats is fake_model


def test_fit_original_optimizer_requests_bbox_removal_true(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path, optimizer_type="original")
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.fit()

    init_kwargs = (
        frc_mock.radiance_fields.GaussianSplatReconstructionConfig.call_args.kwargs
    )
    assert init_kwargs["remove_gaussians_outside_scene_bbox"] is True
    # "original" branch never mutates it back to False
    frc_mock.radiance_fields.GaussianSplatOptimizerConfig.assert_called_once()
    frc_mock.radiance_fields.GaussianSplatOptimizerMCMCConfig.assert_not_called()


def test_fit_reuses_existing_runner(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    existing_runner = frc_mock.radiance_fields.GaussianSplatReconstruction()
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    existing_runner.model = fake_model
    trainer.reconstruction = existing_runner
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.reset_mock()

    trainer.fit()

    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.assert_not_called()
    existing_runner.optimize.assert_called_once()
    assert trainer.splats is fake_model


def test_fit_sets_model_from_runner(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    fake_model = FakeGaussianSplat3d(torch.zeros(2, 3), torch.zeros(2, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.fit()

    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.optimize.assert_called_once()
    assert trainer.splats is fake_model
    assert (
        trainer.reconstruction
        is frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value
    )


# --- clean_splats() ---


def _build_trainer_with_model(
    tmp_path, frc_mock, points, means, scales, **trainer_kwargs
):
    fake_scene = make_fake_scene(points=points)
    colmap_scene = FakeColmapScene(fake_scene, normalization_type="pca")
    trainer = GaussianSplatModelTrainer(
        colmap_scene=colmap_scene, out_dir=tmp_path, run_name="model", **trainer_kwargs
    )
    trainer.splats = FakeGaussianSplat3d(means, scales)
    frc_mock.tools.filter_splats_by_mean_percentile.side_effect = (
        lambda model, **kw: model
    )
    frc_mock.tools.filter_splats_by_opacity_percentile.side_effect = (
        lambda model, **kw: model
    )
    return trainer


def test_clean_splats_requires_scene_and_model(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No splat model available"):
        trainer.clean_splats()


def test_clean_splats_bbox_crop_removes_out_of_bounds_points(tmp_path, frc_mock):
    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    means = torch.cat(
        [
            torch.zeros(20, 3) + 5.0,  # inside [-5,-5,-5]..[15,15,15]
            torch.full((3, 3), 1000.0),  # clearly outside
        ]
    )
    # slight variation (not a constant) so the scale filter's strict
    # inequalities don't exclude every point at the percentile boundary
    scale_values = torch.linspace(0.4, 0.6, means.shape[0])
    scales = scale_values.unsqueeze(1).repeat(1, 3)
    trainer = _build_trainer_with_model(
        tmp_path,
        frc_mock,
        points,
        means,
        scales,
        crop_margin_m=5.0,
        scale_min_percentile=0.0,
        scale_max_percentile=1.0,
    )

    trainer.clean_splats()

    # bbox crop must have removed the 3 far-away points; scale filter may
    # additionally trim a couple of boundary points, but the bulk survives
    assert 15 <= trainer.splats.num_gaussians <= 20
    assert trainer.splats.means.max().item() < 100


def test_clean_splats_scale_percentiles_change_result(tmp_path, frc_mock):
    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    means = torch.zeros(100, 3) + 5.0  # all inside a generous bbox
    scales = torch.linspace(0.0, 1.0, 100).unsqueeze(1).repeat(1, 3)

    trainer_narrow = _build_trainer_with_model(
        tmp_path,
        frc_mock,
        points,
        means.clone(),
        scales.clone(),
        crop_margin_m=1000.0,
        scale_min_percentile=0.4,
        scale_max_percentile=0.6,
    )
    trainer_narrow.clean_splats()

    trainer_wide = _build_trainer_with_model(
        tmp_path,
        frc_mock,
        points,
        means.clone(),
        scales.clone(),
        crop_margin_m=1000.0,
        scale_min_percentile=0.0,
        scale_max_percentile=1.0,
    )
    trainer_wide.clean_splats()

    assert trainer_narrow.splats.num_gaussians < trainer_wide.splats.num_gaussians


# --- export_ply() ---


def test_export_ply_requires_model(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No splat model available"):
        trainer.export_ply()


def test_export_ply_uses_runner_when_available(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.splats = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.reconstruction = frc_mock.radiance_fields.GaussianSplatReconstruction()

    trainer.export_ply()

    trainer.reconstruction.save_ply.assert_called_once_with(str(trainer.ply_path))
    trainer.splats.save_ply.assert_not_called()


def test_export_ply_falls_back_to_model_when_no_runner(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.splats = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.reconstruction = None

    trainer.export_ply()

    trainer.splats.save_ply.assert_called_once_with(str(trainer.ply_path))


# --- export_usdz() ---


def test_export_usdz_uses_runner_save_usd(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.splats = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.reconstruction = frc_mock.radiance_fields.GaussianSplatReconstruction()

    trainer.export_usdz()

    expected_path = str(Path(trainer.ply_path).with_suffix(".usdz"))
    trainer.reconstruction.save_usd.assert_called_once_with(expected_path, usdz=True)
    frc_mock.tools.export_splats_to_usd.assert_not_called()


def test_export_usdz_falls_back_to_export_splats_to_usd(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.splats = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.reconstruction = None

    trainer.export_usdz()

    expected_path = str(Path(trainer.ply_path).with_suffix(".usdz"))
    frc_mock.tools.export_splats_to_usd.assert_called_once_with(
        trainer.splats, expected_path, usdz=True
    )
