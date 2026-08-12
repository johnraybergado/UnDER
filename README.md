# UnDER: Unsupervised Dense point cloud Extraction Routine from UAV imagery using deep learning

An open-source, actively-developed UAV photogrammetric pipeline in Python. Currently covers stereo rectification, deep-learning-based (and classical SGM) disparity estimation, and multiview triangulation to produce dense point clouds — refactored from the research codebase behind [UnDER](https://www.mdpi.com/2072-4292/17/1/24). The pipeline presently assumes pre-oriented images with known camera parameters; feature matching, relative/absolute orientation refinement, and bundle adjustment are planned for integration. The long-term goal is a complete, usable pipeline from raw UAV imagery through to DSM and orthomosaic generation, with further write-ups and improvements shared as new steps land.

The deep learning-based disparity estimation part builds upon the PASMNet architecture.

## Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.11)
- PostgreSQL (for database storage)
- NVIDIA GPU with CUDA 11.8 or 12.1 (recommended for deep learning)
- Required Python packages (see `requirements.txt`)

### Installation

We provide **two installation options**. 

- **Option A (pip + virtualenv)** is for advanced users who prefer standard Python tooling and are willing to manually install system dependencies.
- **Option B (conda)** is **strongly recommended** because it automatically handles system-level binaries (GDAL, libtiff, CUDA) across Windows, macOS, and Linux without any extra setup.

Specific installation instructions below were specifically tested working on Windows 11 Pro and Lubuntu 26.04 LTS.

#### Option A: Using virtualenv

```bash
# Clone the repository
git clone https://github.com/johnraybergado/UnDER.git
cd UnDER

# Create and activate a virtual environment
python -m venv venv
# Activate the virtualenv:
# - Windows: venv\Scripts\activate
# - Mac/Linux: source venv/bin/activate

# For Windows
conda install libtiff=4.5.1

# For Linux/Ubuntu
# sudo apt update
# sudo apt install -y libtiff6

# For Mac
# brew update
# brew install libtiff

# Install GDAL
# For Windows
# download gdal wheel
curl -L -O https://github.com/cgohlke/geospatial-wheels/releases/download/v2023.1.5/GDAL-3.6.2-cp311-cp311-win_amd64.whl
pip install GDAL-3.6.2-cp311-cp311-win_amd64.whl

# For Linux/Ubuntu
# sudo apt install gdal-bin python3-gdal libgdal-dev

# For Mac
# brew install gdal

# For Linux/Windows: install dependencies
pip install -r requirements.txt

# For Mac (tested on MacOS 26.6.1): install dependencies
# brew install libomp
# pip install -r requirements.txt
# LIBOMP="$(brew --prefix libomp)" \
# CXXFLAGS="-I${LIBOMP}/include -Xpreprocessor -fopenmp -Wno-error" LDFLAGS="-L${LIBOMP}/lib -lomp" pip install "pandora[sgm]"


pip install -e .
```

#### Option B: Using conda

```bash
# Clone the repository
git clone https://github.com/johnraybergado/UnDER.git
cd UnDER

# Create and activate conda environment
conda env create -f environment.yml
conda activate under
```

#### CUDA Version Notes

The `requirements.txt` includes PyTorch with CUDA 11.8 by default. If you need a different CUDA version, kindly modify the following lines:

```bash
torch==2.5.1+cu118 ; sys_platform == 'win32' or sys_platform == 'linux'
torchvision==0.20.1+cu118 ; sys_platform == 'win32' or sys_platform == 'linux'
torchaudio==2.5.1+cu118 ; sys_platform == 'win32' or sys_platform == 'linux'
```


### Running the Pipeline

See `examples/run_usegeo.txt` for example commands to run the pipeline on UseGeo Dataset-1.

Basic usage:
```bash
# For Linux (for Windows cmd replace "\" with "^" and for Windows powershell replace "\" with "`")
cd src

python -m scripts.run_usegeo_pipeline \
  --data-root /path/to/dataset \
  --tmp-root /path/to/tmp_directory \
  --las-path /path/to/dataset/DIM_after_adjustment_dataset_C.las \
  --multiview-dicts /path/to/multiview_dicts.list \
  --db-name my_database \
  --disparity-method CNN \
  --model-path /path/to/PASMnet_KITTI2015_epoch80.pth

# For Mac when installed via pip (OMP_NUM_THREADS flag NOT needed WITH conda)
# cd src
# OMP_NUM_THREADS=1 \
# python -m scripts.run_usegeo_pipeline \
#   --data-root /path/to/dataset \
#   --tmp-root /path/to/tmp_directory \
#   --las-path /path/to/dataset/DIM_after_adjustment_dataset_C.las \
#   --multiview-dicts /path/to/multiview_dicts.list \
#   --db-name my_database \
#   --disparity-method CNN \
#   --model-path /path/to/PASMnet_KITTI2015_epoch80.pth
```

#### Disparity Methods

- **CNN**: Uses PASMNet deep learning model (requires `--model-path`)
- **SGM**: Uses Semi-Global Matching (no model path required)

#### Base Image Filter

Use `--base-image-filter` to specify which images to process:
- Single image: `--base-image-filter "image1.jpg"`
- Multiple images: `--base-image-filter "image1.jpg,image2.jpg"`
- All images: Omit the parameter (processes all images in multiview_dicts.list)

## Project Structure

```
UnDER/
├── src/
│   ├── under_pipeline/           # Main pipeline code
│   │   ├── config.py             # Pipeline configuration
│   │   ├── core_pipeline.py      # Top-level pipeline orchestration
│   │   ├── db_client.py          # Database client
│   │   ├── db_core.py            # Database core functionality
│   │   ├── disp_cache.py         # Disparity caching
│   │   ├── disp_utils.py         # Disparity utilities
│   │   ├── multiview_core.py     # Multiview reconstruction core
│   │   ├── multiview_service.py  # Multiview service
│   │   ├── stereo.py             # Stereo matching
│   │   ├── camera_params_UG.py   # Camera parameter handling
│   │   ├── rectify.py            # Image rectification
│   │   └── io_utils.py           # I/O utilities
│   ├── models/                   # Deep learning models
│   │   ├── PASMnet.py            # PASMNet model implementation
│   │   └── modules.py            # Model modules
│   └── scripts/                  # Entry point scripts
│       └── run_usegeo_pipeline.py # Main pipeline script
├── examples/
│   └── run_usegeo.txt            # Example commands
└── README.md                     # This file
```

## Dataset Preparation

The pipeline expects the following dataset structure:

```
Dataset-Root/
├── undistorted_images/           # Undistorted input images
│   ├── 2021-04-23_13-17-22_S2223314_DxO.jpg
│   ├── 2021-04-23_13-17-25_S2223315_DxO.jpg
│   └── ...
├── DIM_after_adjustment_dataset_C.las  # LAS file to extract mean elevation for basis depth calculation
└── image_orientations.xyz       # Camera calibration file
```

A cache folder structure (currently not automatically created) for persisting intermediate results:

```
- path/to
  - img/left
  - img/right
  - left_disp/orig
  - left_disp/warp
  - left_disp/sgm
  - left_disp/tmp_0
  - ply
```

### Generating multiview_dicts.list

The `multiview_dicts.list` file is a pickled Python list containing dictionaries mapping base images to their matching images. This file needs to be generated before running the pipeline.

## Configuration

Main configuration parameters are defined in `src/under_pipeline/config.py`:

- `data_root`: Root directory of the dataset
- `img_dir`: Directory containing images
- `las_path`: Path to LAS file
- `tmp_root`: Temporary working directory
- `db_name`: PostgreSQL database name
- `disparity_method`: "CNN" or "SGM"
- `model_path`: Path to PASMNet model (for CNN method)
- `disp_diff_threshold`: Right-left consistency threshold
- `base_image_filter`: Comma-separated list of base images to process

## Sample Dataset

To reproduce the examples, download the sample dataset from the GitHub Release:

- [UseGeo sample Dataset v1.0.0](https://github.com/johnraybergado/UnDER/releases/tag/v0.1.0)

### Direct asset links

Download: [`refactor_test_subset.zip`](https://github.com/johnraybergado/UnDER/releases/download/v0.1.0/refactor_test_subset.zip).

After downloading, extract the archive:

```bash
unzip refactor_test_subset.zip -d data/
```

See [Running the Pipeline](#running-the-pipeline) and [`examples/run_usegeo.txt`](https://github.com/johnraybergado/UnDER/blob/1035aec16cedfcfdd3b658416604c2e0989fc7f9/examples/run_usegeo.txt) on how to run UnDER using the above sample dataset.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the PASMNet architecture. The original PASMNet implementation can be found at [PASMNet repository](https://github.com/SYSU-SAIL/PAM).

## Roadmap

- [X] Improve documentation with detailed setup instructions
- [ ] Add a notebook going through a detailed example processing a subset of UseGeo
- [ ] Add integration tests
- [ ] Add support for additional datasets
- [ ] Integrate additional disparity estimation models besides PASMNet
- [ ] Add gap filling and built-in point cloud filtering refinement methods
- [ ] Integrate feature matching step to perform adjustment of orientation parameters
- [ ] Implement orthomosaic and DSM generation
- [ ] Add semantic layer to the point cloud
