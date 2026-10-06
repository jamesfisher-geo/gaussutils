from pathlib import Path

import pytest

from gaussutils.scene import ColmapScene
from tests.helpers import make_fake_scene


def _make_scene(tmp_path, frc_mock, **kwargs):
    frc_mock.sfm_scene.SfmScene.from_colmap.return_value = make_fake_scene()
    return ColmapScene(dataset_path=tmp_path, **kwargs)


def test_init_coerces_dataset_path_to_path(tmp_path, frc_mock):
    scene = _make_scene(tmp_path, frc_mock, normalization_type="pca")
    assert isinstance(scene.dataset_path, Path)
    assert scene.dataset_path == tmp_path


def test_init_loads_scene_via_from_colmap(tmp_path, frc_mock):
    fake_scene = make_fake_scene()
    frc_mock.sfm_scene.SfmScene.from_colmap.return_value = fake_scene
    scene = ColmapScene(dataset_path=str(tmp_path), normalization_type="pca")
    frc_mock.sfm_scene.SfmScene.from_colmap.assert_called_once_with(str(tmp_path))
    assert scene.scene is fake_scene


def test_init_stores_constructor_args(tmp_path, frc_mock):
    scene = _make_scene(
        tmp_path,
        frc_mock,
        normalization_type="similarity",
        image_downsample_factor=2,
        percentile_min=(1.0, 2.0, 3.0),
        percentile_max=(97.0, 98.0, 99.0),
        scene_crop_margin=0.1,
        min_points_per_image=25,
    )
    assert scene.normalization_type == "similarity"
    assert scene.image_downsample_factor == 2
    assert scene.percentile_min == (1.0, 2.0, 3.0)
    assert scene.percentile_max == (97.0, 98.0, 99.0)
    assert scene.scene_crop_margin == 0.1
    assert scene.min_points_per_image == 25


def test_load_scene_raises_for_missing_dataset_path(tmp_path, frc_mock):
    missing = tmp_path / "does-not-exist"
    scene = _make_scene(tmp_path, frc_mock)  # valid dir first so __init__ succeeds
    scene.dataset_path = missing
    with pytest.raises(ValueError, match="does not exist"):
        scene.load()


def test_check_scene_raises_when_unset(tmp_path, frc_mock):
    scene = _make_scene(tmp_path, frc_mock)
    scene.scene = None
    with pytest.raises(ValueError, match="Scene not available"):
        scene._check_scene()


def test_check_scene_noop_when_set(tmp_path, frc_mock):
    scene = _make_scene(tmp_path, frc_mock)
    scene._check_scene()  # should not raise


def test_filter_scene_invokes_cleanup_pipeline(tmp_path, frc_mock):
    scene = _make_scene(tmp_path, frc_mock)
    raw_scene = scene.scene
    cleaned_scene = make_fake_scene()
    frc_mock.transforms.Compose.return_value.return_value = cleaned_scene

    scene.filter_scene()

    frc_mock.transforms.Compose.return_value.assert_called_once_with(raw_scene)
    assert scene.scene is cleaned_scene


def test_filter_scene_raises_on_non_pinhole_camera(tmp_path, frc_mock, fvdb_mock):
    scene = _make_scene(tmp_path, frc_mock)

    bad_camera = type("Cam", (), {"camera_model": fvdb_mock.CameraModel.OPENCV})()
    cleaned_scene = make_fake_scene(cameras={"cam0": bad_camera})
    frc_mock.transforms.Compose.return_value.return_value = cleaned_scene

    with pytest.raises(RuntimeError, match="non-pinhole cameras"):
        scene.filter_scene()


def test_filter_scene_passes_with_only_pinhole_cameras(tmp_path, frc_mock, fvdb_mock):
    scene = _make_scene(tmp_path, frc_mock)

    good_camera = type("Cam", (), {"camera_model": fvdb_mock.CameraModel.PINHOLE})()
    cleaned_scene = make_fake_scene(cameras={"cam0": good_camera})
    frc_mock.transforms.Compose.return_value.return_value = cleaned_scene

    scene.filter_scene()  # should not raise
    assert scene.scene is cleaned_scene
