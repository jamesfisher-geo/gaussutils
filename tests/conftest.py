"""sys.modules mocking for the CUDA-only / private-index gaussutils dependencies.

`fvdb`, `fvdb_reality_capture`, and `point_cloud_utils` can't be installed on a
standard CI runner. Every gaussutils module imports them at module scope, so we
stub them into sys.modules here, before any test module imports gaussutils, and
always overwrite (never setdefault) so tests behave the same whether or not the
real packages happen to be installed locally.

Pure test doubles (FakeGaussianSplat3d, make_fake_scene, FakeColmapScene) live in
tests/helpers.py, not here — that module must stay free of side effects, since
this file's mocking setup must only ever run once per session.
"""

import sys
from unittest.mock import MagicMock

import pytest


class _FakeCameraModel:
    PINHOLE = object()
    OPENCV = object()
    FISHEYE = object()


def _install_fake_module(name: str) -> MagicMock:
    mock = MagicMock(name=name)
    sys.modules[name] = mock
    return mock


_fvdb = _install_fake_module("fvdb")
_fvdb.CameraModel = _FakeCameraModel

_frc = _install_fake_module("fvdb_reality_capture")
_frc.transforms = _install_fake_module("fvdb_reality_capture.transforms")
_frc.sfm_scene = _install_fake_module("fvdb_reality_capture.sfm_scene")
_frc.radiance_fields = _install_fake_module("fvdb_reality_capture.radiance_fields")
_frc.tools = _install_fake_module("fvdb_reality_capture.tools")

_pcu = _install_fake_module("point_cloud_utils")


@pytest.fixture(autouse=True)
def _reset_frc_mocks():
    """Reset call history / configured return values between tests."""
    for mod in (_fvdb, _frc, _frc.transforms, _frc.sfm_scene, _frc.radiance_fields, _frc.tools, _pcu):
        mod.reset_mock(return_value=True, side_effect=True)
    _fvdb.CameraModel = _FakeCameraModel
    yield


@pytest.fixture
def fvdb_mock():
    return _fvdb


@pytest.fixture
def frc_mock():
    return _frc


@pytest.fixture
def pcu_mock():
    return _pcu
