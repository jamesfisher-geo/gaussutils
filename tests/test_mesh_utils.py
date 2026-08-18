from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import gaussutils.mesh_utils as mesh_utils
from tests.helpers import FakeGaussianSplat3d, make_fake_scene


def _fake_trainer(scene_set=True, model_set=True):
    """A GaussianSplatTrainer-shaped double: real _require_* semantics, mocked
    model/scene so extract_mesh's argument-passing can be inspected."""
    trainer = MagicMock(name="GaussianSplatTrainer")
    trainer.scene = make_fake_scene(num_cameras=2) if scene_set else None
    trainer.model = FakeGaussianSplat3d(torch.zeros(3, 3), torch.ones(3, 3)) if model_set else None

    def _require_scene():
        if not trainer.scene:
            raise ValueError("Missing input scene")

    def _require_model():
        if not trainer.model:
            raise ValueError("Missing 3DGS model")

    trainer._require_scene.side_effect = _require_scene
    trainer._require_model.side_effect = _require_model
    return trainer


def test_extract_mesh_requires_scene_and_model(frc_mock):
    gs = _fake_trainer(model_set=False)
    with pytest.raises(ValueError, match="Missing 3DGS model"):
        mesh_utils.extract_mesh(gs, truncation_margin=0.1, use_dlnr=False)


def test_extract_mesh_uses_dlnr_when_requested(frc_mock):
    gs = _fake_trainer()
    expected = (torch.zeros(2, 3), torch.zeros(1, 3, dtype=torch.long), torch.zeros(2, 3))
    frc_mock.tools.mesh_from_splats_dlnr.return_value = expected
    frc_mock.tools.mesh_from_splats.reset_mock()

    result = mesh_utils.extract_mesh(gs, truncation_margin=0.5, use_dlnr=True, num_workers=2)

    frc_mock.tools.mesh_from_splats_dlnr.assert_called_once_with(
        gs.model,
        gs.scene.camera_to_world_matrices,
        gs.scene.projection_matrices,
        gs.scene.image_sizes,
        0.5,
        grid_shell_thickness=3.0,
        dtype=torch.float32,
        num_workers=2,
    )
    frc_mock.tools.mesh_from_splats.assert_not_called()
    assert result == expected


def test_extract_mesh_uses_basic_tsdf_when_not_dlnr(frc_mock):
    gs = _fake_trainer()
    expected = (torch.zeros(2, 3), torch.zeros(1, 3, dtype=torch.long), torch.zeros(2, 3))
    frc_mock.tools.mesh_from_splats.return_value = expected
    frc_mock.tools.mesh_from_splats_dlnr.reset_mock()

    result = mesh_utils.extract_mesh(gs, truncation_margin=0.5, use_dlnr=False)

    frc_mock.tools.mesh_from_splats.assert_called_once_with(
        gs.model,
        gs.scene.camera_to_world_matrices,
        gs.scene.projection_matrices,
        gs.scene.image_sizes,
        0.5,
        grid_shell_thickness=3.0,
        dtype=torch.float32,
    )
    frc_mock.tools.mesh_from_splats_dlnr.assert_not_called()
    assert result == expected


def test_save_mesh_coerces_path_and_calls_pcu(tmp_path, pcu_mock):
    out = str(tmp_path / "mesh.ply")
    vertices = torch.zeros(4, 3)
    faces = torch.zeros(2, 3, dtype=torch.long)
    colors = torch.zeros(4, 3)

    mesh_utils.save_mesh(out, vertices, faces, colors)

    assert pcu_mock.save_mesh_vfc.call_count == 1
    args, kwargs = pcu_mock.save_mesh_vfc.call_args
    assert args[0] == str(Path(out))
    assert (args[1] == vertices.numpy()).all()
    assert (args[2] == faces.numpy()).all()
    assert (args[3] == colors.numpy()).all()
