# UnDER: Unsupervised Dense point cloud Extraction Routine from UAV imagery using deep learning

UnDER is a pipeline for generating dense 3D point clouds from UAV (drone) imagery using deep learning-based stereo matching. This implementation builds upon the PASMNet architecture for disparity estimation.

## Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.11)
- PostgreSQL (for database storage)
- NVIDIA GPU with CUDA 11.8 or 12.1 (recommended for deep learning)
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/johnraybergado/UnDER.git
cd UnDER

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development, also install dev dependencies
pip install -r requirements-dev.txt

# Or install in development mode
pip install -e .
```

#### CUDA Version Notes

The `requirements.txt` includes PyTorch with CUDA 11.8 by default. If you need a different CUDA version:

- **CUDA 12.1**: Uncomment the CUDA 12.1 lines and comment out CUDA 11.8 lines
- **CPU-only**: Uncomment the CPU-only lines and comment out all CUDA lines
- **Custom CUDA**: Install PyTorch manually from [pytorch.org](https://pytorch.org) first, then install other requirements

#### Conda Environment Setup (Alternative)

If you prefer using Conda, you can create an environment with:

```bash
# Create environment
conda create -n dmpy_11 python=3.11
conda activate dmpy_11

# Install core dependencies
conda install pillow
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cudatoolkit=11.8
conda install scikit-image
conda install pandas
conda install -c conda-forge geopandas
conda install rasterio
conda install libtiff=4.5.1
conda install laspy
conda install psycopg2
conda install -c conda-forge gdal
conda install -c conda-forge plyfile

# Install in development mode
cd /path/to/UnDER
pip install -e .
```

### Running the Pipeline

See `examples/run_usegeo.txt` for example commands to run the pipeline on UseGeo Dataset-1.

Basic usage:
```bash
python -m scripts.run_usegeo_pipeline \
  --data-root /path/to/dataset \
  --tmp-root /path/to/tmp_directory \
  --las-path /path/to/dataset/DIM_after_adjustment_dataset_C.las \
  --multiview-dicts /path/to/multiview_dicts.list \
  --db-name my_database \
  --disparity-method CNN \
  --model-path /path/to/PASMnet_KITTI2015_epoch80.pth
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
├── DIM_after_adjustment_dataset_C.las  # LAS file with ground points
└── metric_calibration.txt        # Camera calibration file
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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the PASMNet architecture. The original PASMNet implementation can be found at [PASMNet repository](https://github.com/aim-uofa/PASMNet).

## Roadmap

- [ ] Add support for additional datasets
- [ ] Improve documentation with detailed setup instructions
- [ ] Add benchmarking scripts
- [ ] Implement additional disparity refinement methods
- [ ] Add visualization tools for point cloud inspection
