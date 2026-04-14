# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Installation

Requires Python 3.10+, CUDA 12.8-capable GPU, and `uv`.

```bash
uv pip install . --system          # install package
uv pip install ".[dev]" --system   # include dev tools (pytest, ruff, mypy)
```

Custom PyPI indexes are configured in `pyproject.toml` for `torch` (pytorch-cu128) and `fvdb`/`fvdb-reality-capture` (CloudFront CDN).

## Key dependencies

- [fvdb-core](https://github.com/openvdb/fvdb-core): The core framework for 3d deep learning and gaussian splatting. Many tools in gaussutils and fvdb-reality-capture utilize the base 3DGS structures defined the [fvdb.gaussian_splatting.py](https://github.com/openvdb/fvdb-core/blob/5d7a5815898d545a9d4a3c4252c58b7c0892c045/fvdb/gaussian_splatting.py)
- [fvdb-reality-cpature](https://github.com/openvdb/fvdb-reality-capture): Library to train and process 3DGS built on top of fVDB. This library (gaussutils) wraps many tools from fvdb-reality-capture into user-friendly fucntions and methods.



## Running Scripts

```bash
# Full pipeline: COLMAP dataset → train → filter → PLY/USDZ + mesh
python scripts/reconstruct.py --dataset-path /path/to/colmap [--output-dir ./out] [--no-dlnr]

# Load existing checkpoint → filter → PLY
python scripts/load_checkpoint.py --checkpoint-path model.pt [--output-dir ./out]

# Load checkpoint + COLMAP scene → mesh only
python scripts/checkpoint_to_mesh.py --checkpoint-path model.pt --dataset-path /path/to/colmap
```

## Architecture

The library wraps **fVDB Reality Capture** (`fvdb_reality_capture` / `frc`) and **fVDB** (`fvdb`) for GPU-accelerated 3D Gaussian Splatting.

**Library modules** (`src/gaussutils/`):
- `scene_utils.py` — loads COLMAP via `frc.sfm_scene.SfmScene.from_colmap()`, applies `frc.transforms` pipeline (downsample, PCA normalize, point percentile filter, low-point image filter)
- `splat_utils.py` — training (`GaussianSplatReconstruction`), checkpoint I/O (`torch.load` → `from_state_dict`), PLY/USDZ save, and all filter functions
- `mesh_utils.py` — wraps `frc.tools.mesh_from_splats_dlnr` (DLNR stereo depth, preferred) and `frc.tools.mesh_from_splats` (basic TSDF); saves with `pcu.save_mesh_vfc`

**Filter pipeline** in `splat_utils.auto_filter_splats` (called by all scripts):
1. Scale IQR in log-space → `frc.tools.filter_splats_above/below_scale` (thresholds as fraction of scene bbox diagonal)
2. Opacity absolute floor → `frc.tools.filter_splats_by_opacity_percentile`
3. Spatial percentile → `frc.tools.filter_splats_by_mean_percentile`
4. KNN density via `pcu.k_nearest_neighbors` → removes isolated floaters

**Key types**: `fvdb.GaussianSplat3d` supports boolean mask indexing (`model[mask]`), `.means`, `.scales`, `.logit_opacities` (sigmoid → [0,1]), `.num_gaussians`, `.device`.

## Docker

```bash
docker build . -t gaussutils
docker run --gpus all -p 8080:8080 -v <data>:/data -v <code>:/code -it gaussutils
```

Visualization (`fvdb.viz`) requires Vulkan. On WSL2, `check_viz_available()` in `splat_utils.py` probes Vulkan in a subprocess to avoid segfaults before calling `fvdb.viz.init(port=)`.