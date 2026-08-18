import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussutils.trainer import GaussianSplatTrainer
from tests.helpers import FakeColmapScene, FakeGaussianSplat3d, make_fake_scene


def _make_trainer(tmp_path, run_name="model", **kwargs):
    fake_scene = make_fake_scene()
    colmap_scene = FakeColmapScene(
        fake_scene, normalization_type=kwargs.pop("normalization_type", "ecef2enu")
    )
    return GaussianSplatTrainer(
        scene=colmap_scene, output_dir=tmp_path, run_name=run_name, **kwargs
    )


# --- __init__ ---


def test_init_coerces_output_dir_to_path(tmp_path):
    trainer = _make_trainer(str(tmp_path))
    assert isinstance(trainer.output_dir, Path)
    assert trainer.output_dir == tmp_path


def test_init_builds_model_ply_and_georef_paths(tmp_path):
    trainer = _make_trainer(tmp_path, run_name="myrun")
    assert trainer.model_ply == tmp_path / "myrun.ply"
    assert trainer.georef_json == tmp_path / "model.georef.json"


def test_init_stores_deletion_opacity_threshold_as_plain_float(tmp_path):
    trainer = _make_trainer(tmp_path, deletion_opacity_threshold=0.01)
    assert trainer.deletion_opacity_threshold == 0.01
    assert not isinstance(trainer.deletion_opacity_threshold, tuple)


def test_init_model_and_runner_start_none(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert trainer.model is None
    assert trainer.runner is None


def test_init_pulls_attrs_from_colmap_scene(tmp_path):
    fake_scene = make_fake_scene(transformation_matrix=np.eye(4) * 2)
    colmap_scene = FakeColmapScene(fake_scene, normalization_type="similarity")
    trainer = GaussianSplatTrainer(
        scene=colmap_scene, output_dir=tmp_path, run_name="model"
    )
    assert trainer.scene is fake_scene
    assert (trainer.transform_matrix == fake_scene.transformation_matrix).all()
    assert trainer.normalization_type == "similarity"


def test_init_stores_percentile_and_bbox_params(tmp_path):
    trainer = _make_trainer(
        tmp_path,
        scale_min_percentile=0.1,
        scale_max_percentile=0.9,
        opacity_percentile=0.5,
        model_bbox_margin=2.0,
    )
    assert trainer.scale_min_percentile == 0.1
    assert trainer.scale_max_percentile == 0.9
    assert trainer.opacity_percentile == 0.5
    assert trainer.model_bbox_margin == 2.0


# --- _require_scene / _require_model ---


def test_require_scene_raises_when_unset(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.scene = None
    with pytest.raises(ValueError, match="Missing input scene"):
        trainer._require_scene()


def test_require_scene_noop_when_set(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer._require_scene()  # should not raise


def test_require_model_raises_when_unset(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="Missing 3DGS model"):
        trainer._require_model()


def test_require_model_noop_when_set(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer._require_model()  # should not raise


# --- train() ---


def test_train_raises_when_scene_has_no_images(tmp_path, frc_mock):
    fake_scene = make_fake_scene(num_images=0)
    colmap_scene = FakeColmapScene(fake_scene)
    trainer = GaussianSplatTrainer(
        scene=colmap_scene, output_dir=tmp_path, run_name="model"
    )
    with pytest.raises(ValueError, match="no images"):
        trainer.train()


def test_train_mcmc_forces_bbox_removal_false(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path, optimizer_type="mcmc")
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.train()

    config_mock = (
        frc_mock.radiance_fields.GaussianSplatReconstructionConfig.return_value
    )
    assert config_mock.remove_gaussians_outside_scene_bbox is False
    call_kwargs = (
        frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.call_args.kwargs
    )
    assert call_kwargs["config"] is config_mock
    assert trainer.model is fake_model


def test_train_original_optimizer_requests_bbox_removal_true(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path, optimizer_type="original")
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.train()

    init_kwargs = (
        frc_mock.radiance_fields.GaussianSplatReconstructionConfig.call_args.kwargs
    )
    assert init_kwargs["remove_gaussians_outside_scene_bbox"] is True
    # "original" branch never mutates it back to False
    frc_mock.radiance_fields.GaussianSplatOptimizerConfig.assert_called_once()
    frc_mock.radiance_fields.GaussianSplatOptimizerMCMCConfig.assert_not_called()


def test_train_reuses_existing_runner(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    existing_runner = frc_mock.radiance_fields.GaussianSplatReconstruction()
    fake_model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    existing_runner.model = fake_model
    trainer.runner = existing_runner
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.reset_mock()

    trainer.train()

    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.assert_not_called()
    existing_runner.optimize.assert_called_once()
    assert trainer.model is fake_model


def test_train_sets_model_from_runner(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    fake_model = FakeGaussianSplat3d(torch.zeros(2, 3), torch.zeros(2, 3))
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.model = (
        fake_model
    )

    trainer.train()

    frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value.optimize.assert_called_once()
    assert trainer.model is fake_model
    assert (
        trainer.runner
        is frc_mock.radiance_fields.GaussianSplatReconstruction.from_sfm_scene.return_value
    )


# --- filter_model() ---


def _build_trainer_with_model(
    tmp_path, frc_mock, points, means, scales, **trainer_kwargs
):
    fake_scene = make_fake_scene(points=points)
    colmap_scene = FakeColmapScene(fake_scene, normalization_type="pca")
    trainer = GaussianSplatTrainer(
        scene=colmap_scene, output_dir=tmp_path, run_name="model", **trainer_kwargs
    )
    trainer.model = FakeGaussianSplat3d(means, scales)
    frc_mock.tools.filter_splats_by_mean_percentile.side_effect = (
        lambda model, **kw: model
    )
    frc_mock.tools.filter_splats_by_opacity_percentile.side_effect = (
        lambda model, **kw: model
    )
    return trainer


def test_filter_model_requires_scene_and_model(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="Missing 3DGS model"):
        trainer.filter_model()


def test_filter_model_bbox_crop_removes_out_of_bounds_points(tmp_path, frc_mock):
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
        model_bbox_margin=5.0,
        scale_min_percentile=0.0,
        scale_max_percentile=1.0,
    )

    trainer.filter_model()

    # bbox crop must have removed the 3 far-away points; scale filter may
    # additionally trim a couple of boundary points, but the bulk survives
    assert 15 <= trainer.model.num_gaussians <= 20
    assert trainer.model.means.max().item() < 100


def test_filter_model_scale_percentiles_change_result(tmp_path, frc_mock):
    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    means = torch.zeros(100, 3) + 5.0  # all inside a generous bbox
    scales = torch.linspace(0.0, 1.0, 100).unsqueeze(1).repeat(1, 3)

    trainer_narrow = _build_trainer_with_model(
        tmp_path,
        frc_mock,
        points,
        means.clone(),
        scales.clone(),
        model_bbox_margin=1000.0,
        scale_min_percentile=0.4,
        scale_max_percentile=0.6,
    )
    trainer_narrow.filter_model()

    trainer_wide = _build_trainer_with_model(
        tmp_path,
        frc_mock,
        points,
        means.clone(),
        scales.clone(),
        model_bbox_margin=1000.0,
        scale_min_percentile=0.0,
        scale_max_percentile=1.0,
    )
    trainer_wide.filter_model()

    assert trainer_narrow.model.num_gaussians < trainer_wide.model.num_gaussians


# --- save_ply() ---


def test_save_ply_requires_model(tmp_path):
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="Missing 3DGS model"):
        trainer.save_ply()


def test_save_ply_uses_runner_when_available(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.runner = frc_mock.radiance_fields.GaussianSplatReconstruction()

    trainer.save_ply()

    trainer.runner.save_ply.assert_called_once_with(str(trainer.model_ply))
    trainer.model.save_ply.assert_not_called()


def test_save_ply_falls_back_to_model_when_no_runner(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.runner = None

    trainer.save_ply()

    trainer.model.save_ply.assert_called_once_with(str(trainer.model_ply))


# --- save_usdz() ---


def test_save_usdz_uses_runner_save_usd(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.runner = frc_mock.radiance_fields.GaussianSplatReconstruction()

    trainer.save_usdz()

    expected_path = str(Path(trainer.model_ply).with_suffix(".usdz"))
    trainer.runner.save_usd.assert_called_once_with(expected_path, usdz=True)
    frc_mock.tools.export_splats_to_usd.assert_not_called()


def test_save_usdz_falls_back_to_export_splats_to_usd(tmp_path, frc_mock):
    trainer = _make_trainer(tmp_path)
    trainer.model = FakeGaussianSplat3d(torch.zeros(1, 3), torch.zeros(1, 3))
    trainer.runner = None

    trainer.save_usdz()

    expected_path = str(Path(trainer.model_ply).with_suffix(".usdz"))
    frc_mock.tools.export_splats_to_usd.assert_called_once_with(
        trainer.model, expected_path, usdz=True
    )


# --- save_georef() ---


def test_save_georef_noop_when_not_ecef2enu(tmp_path):
    trainer = _make_trainer(tmp_path, normalization_type="pca")
    trainer.save_georef()
    assert not trainer.georef_json.exists()


def test_save_georef_writes_valid_json_when_ecef2enu(tmp_path):
    trainer = _make_trainer(tmp_path, normalization_type="ecef2enu")
    trainer.transform_matrix = np.diag([2.0, 2.0, 2.0, 1.0])

    trainer.save_georef()

    assert trainer.georef_json.exists()
    data = json.loads(trainer.georef_json.read_text())
    assert data["coordinate_system"] == "ENU"
    assert data["epsg"] == 4978
    expected_inv = np.linalg.inv(trainer.transform_matrix).tolist()
    assert data["enu_to_ecef_matrix"] == expected_inv
