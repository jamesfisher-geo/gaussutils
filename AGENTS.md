# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

`gaussutils` builds 3D Gaussian Splat (3DGS) models from COLMAP datasets. It is a thin, user-friendly wrapper around:

- [fvdb-core](https://github.com/openvdb/fvdb-core) (`fvdb`): core 3DGS types, notably `fvdb.GaussianSplat3d`.
- [fvdb-reality-capture](https://github.com/openvdb/fvdb-reality-capture) (`fvdb_reality_capture`, imported as `frc`): training, filtering, meshing, and export tools.

When adding functionality, prefer calling an existing `frc` / `fvdb` API over reimplementing it.

## Repository layout

```
src/gaussutils/
  scene.py        ColmapScene: load a COLMAP dataset, then filter_scene() to downsample/normalize/clean
  trainer.py      GaussianSplatModelTrainer: fit() → clean_splats() → export_ply() / export_usdz()
  mesh_utils.py   extract_mesh() (TSDF or DLNR stereo depth), save_mesh()
  splat_utils.py  checkpoint loading, USDZ export, and standalone splat filter functions
scripts/
  reconstruct.py  CLI pipeline: COLMAP → train → filter → PLY + USDZ
tests/
  conftest.py     stubs fvdb / frc / point_cloud_utils in sys.modules (see Testing)
  helpers.py      test doubles: FakeGaussianSplat3d, FakeColmapScene, make_fake_scene
```

## Setup

Requires Python 3.10+, `uv`, and an NVIDIA GPU with CUDA 13.0 for real runs.

```bash
uv pip install . --system          # library
uv pip install ".[dev]" --system   # + pytest, ruff, mypy
```

`torch` and `fvdb-core` come from custom indexes declared under `[tool.uv.sources]` in `pyproject.toml`. Don't install them from plain PyPI.

## Testing

```bash
pytest                 # runs tests/ with src/ on the path (configured in pyproject.toml)
```

- Tests need **no GPU and no fvdb install.** `tests/conftest.py` replaces `fvdb`, `fvdb_reality_capture`, and `point_cloud_utils` with `MagicMock`s before anything imports `gaussutils`, and resets them between tests.
- Use the `frc_mock` fixture to configure or assert on `frc` calls (e.g. `frc_mock.tools.filter_splats_by_opacity_percentile.side_effect = ...`).
- Build models and scenes with the doubles in `tests/helpers.py`. Keep that module free of side effects.
- Because `frc` is mocked, tests verify argument passing, not real library behavior. Anything that depends on real `frc` signatures needs a GPU run to confirm.

## Lint & format

```bash
uvx ruff check src scripts tests
uvx ruff format src scripts tests
```

The codebase has existing ruff findings (e.g. `Optional`/`Union` vs `X | Y`). Don't mass-fix them as part of unrelated changes.

## Conventions

- **Docstrings:** Google style, brief: a summary line, then `Args:` / `Returns:` / `Raises:` as needed. Defaults are visible in the signature, so don't repeat them.
- **Logging:** use a module-level `logger = logging.getLogger(__name__)` and f-strings. Format counts with `:,`.
- **Validation:** classes guard state with private `_check_*()` methods that raise `ValueError` (e.g. `GaussianSplatModelTrainer._check_scene()` / `_check_splats()`).
- **Paths:** accept `Union[Path, str]` and convert with `Path(...)` immediately.
- **Model filtering:** `fvdb.GaussianSplat3d` supports boolean mask indexing (`model[mask]`). Useful attributes are `.means`, `.scales`, `.logit_opacities` (apply sigmoid for [0, 1]), `.num_gaussians`, and `.device`.
- **Optimizers:** with `optimizer_type="mcmc"`, `remove_gaussians_outside_scene_bbox` must be `False`.
- **Public API changes:** `GaussianSplatModelTrainer` attributes are used by `mesh_utils.extract_mesh`, `scripts/reconstruct.py`, and the tests. Update all of them together.

## Docker

```bash
docker build . -t gaussutils
docker run --gpus all -p 8080:8080 -v <data>:/data -v <code>:/code -it gaussutils
```

Port 8080 serves the `fvdb.viz` viewer, which requires Vulkan.

## Known issues

- `scripts/reconstruct.py`: `--run-name` has no default (outputs become `None.ply`). The `--output-dir` help text says `<dataset-path>/output`, but the code uses `<dataset-path>/3dgs`.
- `Dockerfile` pins older versions (`fvdb-core 0.4.0`, `torch 2.10.0`, CUDA 12.8 base image) than `pyproject.toml` (`0.5.0`, `2.11.0`, cu130).
- `point_cloud_utils` is imported by `mesh_utils.py` / `splat_utils.py` but is not listed in `pyproject.toml` dependencies.
- `pyproject.toml` declares the MIT license, but `LICENSE` is Apache 2.0.
- `spatial_bounds_pct` is stored but unused by `clean_splats()`.
- `CLAUDE.md` is outdated: it references removed scripts and CUDA 12.8.
