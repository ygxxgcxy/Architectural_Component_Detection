# Traditional Chinese architectural component detection

A Detection Method for Ancient Architectural Components Based on an Improved YOLO11 Model:
The detector builds on YOLO11n and combines C3K2-DRG, a Multi-Branch Auxiliary Feature Pyramid Network (MAFPN), DySample, and a Localization Quality Estimation Head (LQEHead).

## Implementation

| Manuscript name | Implementation |
| --- | --- |
| C3K2-DRG | `C3k2_DRG` in `ultralytics/nn/extra_modules/block.py` |
| MAFPN | Multibranch feature-fusion connections in `ultralytics/cfg/models/building/building.yaml` |
| DySample | `DySample` in `ultralytics/nn/extra_modules/block.py` |
| LQEHead | `Detect_LQE` and `LQE` in `ultralytics/nn/extra_modules/head.py` |

The complete architecture is defined in `ultralytics/cfg/models/building/building.yaml`.

## Installation

Set up the Python and PyTorch environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The experimental environment consisted of Python 3.13.3, PyTorch 2.10.0, CUDA 13.0, and an NVIDIA GeForce RTX 3090 GPU.

## Dataset

The original field image data of architectural components are available in `ultralytics/Field_Images/`. Before training the model, please prepare a dataset in YOLO format and a task-specific dataset YAML configuration file. The YAML file should specify the image paths and class names. The order of the class names must be consistent with the class IDs used in the annotations.

## Training

```bash
yolo detect train model=ultralytics/cfg/models/building/building.yaml data=path/to/data.yaml
```

## Validation and prediction

```bash
yolo detect val model=path/to/best.pt data=path/to/data.yaml
yolo detect predict model=path/to/best.pt source=ultralytics/Field_Images
```

## License

The existing AGPL-3.0 license and upstream Ultralytics attribution are preserved. See `LICENSE`.
