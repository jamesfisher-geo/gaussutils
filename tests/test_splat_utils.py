import math

import numpy as np
import pytest
import torch

import gaussutils.splat_utils as splat_utils
from tests.helpers import FakeGaussianSplat3d, make_fake_scene


def _fake_model(n=5, scales=None):
    means = torch.zeros(n, 3)
    if scales is None:
        scales = torch.ones(n, 3)
    return FakeGaussianSplat3d(means, scales)


# --- load_checkpoint() ---


def test_load_checkpoint_raises_for_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        splat_utils.load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_ply_path(tmp_path, fvdb_mock):
    ply_path = tmp_path / "model.ply"
    ply_path.write_text("fake ply")
    fake_model = _fake_model()
    fvdb_mock.GaussianSplat3d.from_ply.return_value = (fake_model, {"meta": True})

    model, runner = splat_utils.load_checkpoint(ply_path)

    fvdb_mock.GaussianSplat3d.from_ply.assert_called_once_with(
        str(ply_path), device="cuda"
    )
    assert model is fake_model
    assert runner is None


def test_load_checkpoint_pt_path(tmp_path, frc_mock, monkeypatch):
    pt_path = tmp_path / "model.pt"
    pt_path.write_text("fake checkpoint")

    fake_checkpoint = {"state": True}
    monkeypatch.setattr(splat_utils.torch, "load", lambda *a, **kw: fake_checkpoint)

    fake_model = _fake_model()
    fake_runner = frc_mock.radiance_fields.GaussianSplatReconstruction()
    fake_runner.model = fake_model
    frc_mock.radiance_fields.GaussianSplatReconstruction.from_state_dict.return_value = (
        fake_runner
    )

    model, runner = splat_utils.load_checkpoint(pt_path)

    frc_mock.radiance_fields.GaussianSplatReconstruction.from_state_dict.assert_called_once_with(
        state_dict=fake_checkpoint
    )
    assert model is fake_model
    assert runner is fake_runner


# --- save_model_usdz() ---
# NOTE: this exercises our own call site, not the real frc API. save_model_usdz
# calls frc.tools.export_splats_to_usdz, which does not exist on the real
# fvdb_reality_capture package (the real function is export_splats_to_usd with a
# usdz=True kwarg, per GaussianSplatTrainer.save_usdz). Mocking can't catch that
# drift — it's a known, pre-existing bug in this function, out of scope here.


def test_save_model_usdz_raises_for_wrong_extension():
    model = _fake_model()
    with pytest.raises(ValueError, match="USDZ"):
        splat_utils.save_model_usdz("out.ply", model)


def test_save_model_usdz_calls_frc_export(tmp_path, frc_mock):
    model = _fake_model()
    out = tmp_path / "out.usdz"

    splat_utils.save_model_usdz(out, model)

    frc_mock.tools.export_splats_to_usdz.assert_called_once_with(
        model, out_path=str(out)
    )


# --- filter_splats() ---


def test_filter_splats_calls_frc_tools_with_expected_args(frc_mock):
    model = _fake_model()
    frc_mock.tools.filter_splats_by_mean_percentile.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_by_opacity_percentile.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_above_scale.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_below_scale.side_effect = lambda m, **kw: m

    result = splat_utils.filter_splats(
        model,
        above_scale_threshold=0.1,
        below_scale_threshold=0.2,
        opacity_percentile=0.9,
    )

    frc_mock.tools.filter_splats_by_mean_percentile.assert_called_once()
    frc_mock.tools.filter_splats_by_opacity_percentile.assert_called_once_with(
        model, percentile=0.9, decimate=4
    )
    frc_mock.tools.filter_splats_above_scale.assert_called_once_with(
        model, prune_scale3d_threshold=0.1
    )
    frc_mock.tools.filter_splats_below_scale.assert_called_once_with(
        model, prune_scale3d_threshold=0.2
    )
    assert result is model


# --- filter_splats_by_knn_density() ---


def test_filter_splats_by_knn_density_removes_floater(pcu_mock):
    n = 10
    model = _fake_model(n=n)

    k = 3
    dists = np.ones((n, k + 1))
    dists[-1, :] = 100.0  # last point is an isolated floater
    indices = np.zeros((n, k + 1), dtype=int)
    pcu_mock.k_nearest_neighbors.return_value = (dists, indices)

    result = splat_utils.filter_splats_by_knn_density(model, k=k, std_multiplier=2.0)

    assert result.num_gaussians == n - 1


# --- auto_filter_splats() ---


def test_auto_filter_splats_computes_adaptive_scale_and_opacity_thresholds(
    frc_mock, pcu_mock
):
    n = 40
    scales_1d = torch.exp(torch.linspace(-2, 0, n))
    scales = scales_1d.unsqueeze(1).repeat(1, 3)
    means = torch.stack(
        [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.zeros(n)], dim=1
    )
    logit_opacities = torch.full(
        (n,), 10.0
    )  # sigmoid(10) ~= 0.99995, none below any sane floor
    model = FakeGaussianSplat3d(means, scales, logit_opacities)

    frc_mock.tools.filter_splats_by_mean_percentile.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_by_opacity_percentile.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_above_scale.side_effect = lambda m, **kw: m
    frc_mock.tools.filter_splats_below_scale.side_effect = lambda m, **kw: m

    knn_k = 3
    rng = np.random.default_rng(0)
    dists = np.abs(rng.normal(loc=1.0, scale=0.01, size=(n, knn_k + 1)))
    indices = np.tile(np.arange(knn_k + 1), (n, 1))
    pcu_mock.k_nearest_neighbors.return_value = (dists, indices)

    result = splat_utils.auto_filter_splats(
        model, decimate=1, knn_k=knn_k, knn_std_multiplier=1000.0
    )

    # mirror the implementation's own threshold math independently
    log_scales = torch.log(scales_1d.clamp(min=1e-8))
    q1 = torch.quantile(log_scales, 0.25).item()
    q3 = torch.quantile(log_scales, 0.75).item()
    iqr = q3 - q1
    upper_scale = math.exp(q3 + 3.0 * iqr)
    lower_scale = math.exp(q1 - 3.0 * iqr)
    scene_scale = (means.amax(dim=0) - means.amin(dim=0)).norm().item()
    expected_above = upper_scale / scene_scale
    expected_below = lower_scale / scene_scale

    above_kwargs = frc_mock.tools.filter_splats_above_scale.call_args.kwargs
    below_kwargs = frc_mock.tools.filter_splats_below_scale.call_args.kwargs
    assert above_kwargs["prune_scale3d_threshold"] == pytest.approx(
        expected_above, rel=1e-4
    )
    assert below_kwargs["prune_scale3d_threshold"] == pytest.approx(
        expected_below, rel=1e-4
    )

    opacity_kwargs = frc_mock.tools.filter_splats_by_opacity_percentile.call_args.kwargs
    assert opacity_kwargs["percentile"] == pytest.approx(1.0, rel=1e-3)

    # KNN pass had a huge std_multiplier, so it shouldn't have dropped anything
    assert result.num_gaussians == n


# --- filter_splats_by_cluster() ---


def test_filter_splats_by_cluster_drops_small_isolated_component(pcu_mock):
    n = 11
    k = 2
    indices = np.zeros((n, k + 1), dtype=int)
    dists = np.zeros((n, k + 1))
    for i in range(10):
        indices[i] = [i, (i + 1) % 10, (i + 2) % 10]
        dists[i] = [0.0, 0.5, 0.5]
    indices[10] = [10, 0, 1]
    dists[10] = [0.0, 1000.0, 1000.0]
    pcu_mock.k_nearest_neighbors.return_value = (dists, indices)

    model = _fake_model(n=n)

    result = splat_utils.filter_splats_by_cluster(
        model, k=k, distance_multiplier=2.0, min_cluster_fraction=0.3
    )

    assert result.num_gaussians == 10


def test_filter_splats_by_cluster_noop_for_single_gaussian():
    model = _fake_model(n=1)
    result = splat_utils.filter_splats_by_cluster(model)
    assert result is model


# --- filter_splats_by_camera_frustum() ---


def test_filter_splats_by_camera_frustum_keeps_only_visible_points():
    means = torch.tensor(
        [
            [0.0, 0.0, 10.0],  # in front, centered -> visible
            [0.0, 0.0, 5.0],  # in front, centered -> visible
            [0.0, 0.0, -5.0],  # behind camera -> not visible
            [10000.0, 0.0, 10.0],  # in front but far out of frame -> not visible
        ]
    )
    model = FakeGaussianSplat3d(means, torch.ones(4, 3))
    scene = make_fake_scene(num_cameras=1)

    result = splat_utils.filter_splats_by_camera_frustum(
        model, scene, min_visible_views=1
    )

    assert result.num_gaussians == 2


def test_filter_splats_by_camera_frustum_noop_for_empty_model():
    model = _fake_model(n=0)
    scene = make_fake_scene(num_cameras=1)
    result = splat_utils.filter_splats_by_camera_frustum(model, scene)
    assert result is model


# --- filter_splats_by_anisotropy() ---


def test_filter_splats_by_anisotropy_removes_needles():
    normal_scales = torch.tensor([[1.0, 0.9, 0.1]] * 3)
    needle_scales = torch.tensor([[10.0, 0.1, 0.05]] * 2)
    scales = torch.cat([normal_scales, needle_scales])
    model = FakeGaussianSplat3d(torch.zeros(5, 3), scales)

    result = splat_utils.filter_splats_by_anisotropy(model, max_elongation=8.0)

    assert result.num_gaussians == 3


# --- filter_splats_for_scene() / filter_splats_for_mesh() orchestration ---


def test_filter_splats_for_scene_orchestrates_subfilters(monkeypatch):
    model = _fake_model()
    calls = []
    monkeypatch.setattr(
        splat_utils, "auto_filter_splats", lambda m, **kw: (calls.append("auto"), m)[1]
    )
    monkeypatch.setattr(
        splat_utils,
        "filter_splats_by_anisotropy",
        lambda m, **kw: (calls.append("aniso"), m)[1],
    )
    monkeypatch.setattr(
        splat_utils,
        "filter_splats_by_cluster",
        lambda m, **kw: (calls.append("cluster"), m)[1],
    )

    result = splat_utils.filter_splats_for_scene(model)

    assert calls == ["auto", "aniso", "cluster"]
    assert result is model


def test_filter_splats_for_mesh_orchestrates_with_scene(monkeypatch):
    model = _fake_model()
    scene = make_fake_scene()
    calls = []
    monkeypatch.setattr(
        splat_utils, "auto_filter_splats", lambda m, **kw: (calls.append("auto"), m)[1]
    )
    monkeypatch.setattr(
        splat_utils,
        "filter_splats_by_camera_frustum",
        lambda m, s, **kw: (calls.append("frustum"), m)[1],
    )

    result = splat_utils.filter_splats_for_mesh(model, scene=scene)

    assert calls == ["auto", "frustum"]
    assert result is model


def test_filter_splats_for_mesh_skips_frustum_without_scene(monkeypatch):
    model = _fake_model()
    calls = []
    monkeypatch.setattr(
        splat_utils, "auto_filter_splats", lambda m, **kw: (calls.append("auto"), m)[1]
    )
    monkeypatch.setattr(
        splat_utils,
        "filter_splats_by_camera_frustum",
        lambda m, s, **kw: (calls.append("frustum"), m)[1],
    )

    result = splat_utils.filter_splats_for_mesh(model, scene=None)

    assert calls == ["auto"]
    assert result is model
