# gaussutils
A library of pipeline tools and Dockerfiles to build and run tools to create 3-D Gaussian Splat (3DGS) models

## Installation

### Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- CUDA 12.8 compatible GPU

### Install uv

To install `uv`, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/) or run the command below.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install gaussutils

Clone the repository:
```bash
git clone https://github.com/your-org/gaussutils.git
cd gaussutils
```

Install the library in an existing environment with
```bash
uv pip install . --system
```

---

## Docker

If you prefer to use Docker rather than installing locally, you can build an image with the full envrionment set up.

### Building the image
Build from pre-built wheels:
```bash
docker build . -t gaussutils
```

### Running
Run the Docker image and exec into the command line. Mount directories for your input data and code here.
```bash
docker run --gpus all -p 8080:8080 -v <path to data>:/data -v <path to code>:/code -it gaussutils
```