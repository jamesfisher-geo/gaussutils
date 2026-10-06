"""Pure test doubles shared across test modules.

Deliberately has no side effects (no sys.modules mocking) so it's safe to
import from multiple test files without risking a duplicate module import
under a different name — unlike conftest.py, which must only ever run its
mocking setup once per session.
"""

from unittest.mock import MagicMock

import numpy as np
import torch


class FakeGaussianSplat3d:
    """Minimal stand-in for fvdb.GaussianSplat3d backed by real torch tensors.

    Supports the subset of the real API that gaussutils code relies on:
    `.means`, `.scales`, `.logit_opacities`, `.device`, `.num_gaussians`, and
    boolean/index-tensor masking via `__getitem__`.
    """

    def __init__(self, means, scales, logit_opacities=None, device="cpu"):
        self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
        self.scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        n = self.means.shape[0]
        if logit_opacities is None:
            logit_opacities = torch.zeros(n, dtype=torch.float32, device=device)
        self.logit_opacities = torch.as_tensor(
            logit_opacities, dtype=torch.float32, device=device
        )
        self.device = device
        self.save_ply = MagicMock(name="FakeGaussianSplat3d.save_ply")

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    def __getitem__(self, mask):
        mask_t = torch.as_tensor(mask)
        return FakeGaussianSplat3d(
            self.means[mask_t],
            self.scales[mask_t],
            self.logit_opacities[mask_t],
            device=self.device,
        )


def make_fake_scene(
    num_images=3,
    num_cameras=1,
    points=None,
    cameras=None,
    transformation_matrix=None,
):
    """Build a MagicMock standing in for frc.sfm_scene.SfmScene."""
    scene = MagicMock(name="SfmScene")
    scene.num_images = num_images
    scene.num_cameras = num_cameras
    scene.images = [MagicMock(name=f"image{i}") for i in range(num_images)]
    scene.points = points if points is not None else np.zeros((10, 3), dtype=np.float32)
    scene.cameras = cameras if cameras is not None else {}
    scene.transformation_matrix = (
        transformation_matrix if transformation_matrix is not None else np.eye(4)
    )
    scene.camera_to_world_matrices = np.tile(np.eye(4), (num_cameras, 1, 1))
    scene.projection_matrices = np.tile(np.eye(3), (num_cameras, 1, 1))
    scene.image_sizes = np.tile(np.array([640, 480]), (num_cameras, 1))
    return scene


class FakeColmapScene:
    """Minimal stand-in for gaussutils.scene.ColmapScene, exposing only the
    attributes GaussianSplatModelTrainer.__init__ reads from it."""

    def __init__(self, scene, normalization_type="ecef2enu"):
        self.scene = scene
        self.normalization_type = normalization_type
