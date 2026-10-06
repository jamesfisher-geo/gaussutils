# gaussutils

Pipeline tools and Dockerfiles for building 3D Gaussian Splat (3DGS) models from COLMAP datasets.

`gaussutils` wraps [fVDB](https://github.com/openvdb/fvdb-core) and [fVDB Reality Capture](https://github.com/openvdb/fvdb-reality-capture) in a small, user-friendly API:

- **Scene loading & cleanup** - load a COLMAP reconstruction, downsample images, normalize coordinates (PCA or ECEF → ENU), and filter outlier points and weakly observed images.
- **Training** - fit a Gaussian splat model with the MCMC or original 3DGS optimizer.
- **Post-filtering** - remove floaters with a bounding-box crop and an opacity filter.
- **Export** - write the model as PLY and USDZ, and extract a triangle mesh (TSDF or DLNR stereo depth).

## Installation

### Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An NVIDIA GPU with CUDA 13.0

### How to install

`gaussutils` uses the `uv` package manager. Install `uv` by following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/), or run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository:
```bash
git clone https://github.com/jamesfisher-geo/gaussutils.git
cd gaussutils
```

Install the library into the active environment:
```bash
uv pip install . --system
```

The CUDA builds of `torch` and `fvdb-core` come from the custom package indexes configured in `pyproject.toml`; `uv` picks them up automatically.

## How to use

### Train a Gaussian splat from the command line

```bash
python3 scripts/reconstruct.py --dataset-path <COLMAP INPUT> --run-name my_scene
```

This loads and cleans the COLMAP scene, trains the model, filters it, and writes these files to `<COLMAP INPUT>/3dgs` (or to `--output-dir`):

| Output | Description |
|---|---|
| `<run-name>.ply` | Filtered Gaussian splat model |
| `<run-name>.usdz` | Same model as USDZ |
| `info/` | Training checkpoints, intermediate PLYs, and metrics |

Common options:

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `<dataset-path>/3dgs` | Where outputs are written |
| `--run-name` | — | Name used for output files |
| `--ecef` | off | Input is ECEF-georeferenced; normalize to ENU (otherwise PCA) |
| `--downsample` | `4` | Image downsample factor for training |
| `--max-gaussians` | `6000000` | Gaussian count limit during training |

To see all options:
```bash
python3 scripts/reconstruct.py --help
```

### Use it as a library

```python
from gaussutils.scene import ColmapScene
from gaussutils.trainer import GaussianSplatModelTrainer
from gaussutils.mesh_utils import extract_mesh, save_mesh

scene = ColmapScene("path/to/colmap", normalization_type="pca", image_downsample_factor=4)
scene.filter_scene()

trainer = GaussianSplatModelTrainer(scene, out_dir="out", run_name="my_scene")
trainer.fit()
trainer.clean_splats()
trainer.export_ply()
trainer.export_usdz()

# Optional: extract a mesh from the trained splats
v, f, c = extract_mesh(trainer, truncation_margin=0.1, use_dlnr=True)
save_mesh("out/my_scene_mesh.ply", v, f, c)
```

## Run with Docker

If you'd rather not install locally, build an image with the full environment set up.

### Build the image
The image installs fVDB from pre-built wheels:
```bash
docker build . -t gaussutils
```

### Run
Start a container with GPU access and an interactive shell. Mount your input data and code:
```bash
docker run --gpus all -p 8080:8080 -v <path to data>:/data -v <path to code>:/code -it gaussutils
```
Port `8080` is exposed for the `fvdb.viz` viewer.

## Development

Install the dev tools (pytest, ruff, mypy) and run the tests:
```bash
uv pip install ".[dev]" --system
pytest
```

## License

See [LICENSE](LICENSE).
