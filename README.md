# Traditional Chinese architectural component detection

Source code accompanying **Accurate and Efficient Detection of Traditional Chinese Architectural Components**.

The detector builds on YOLO11n and combines C3K2-DRG, a Multi-Branch Auxiliary Feature Pyramid Network (MAFPN), DySample, and a Localization Quality Estimation Head (LQEHead).

## Implementation

| Manuscript name | Implementation |
| --- | --- |
| C3K2-DRG | `C3k2_DRG` in `ultralytics/nn/extra_modules/block.py` |
| MAFPN | Multibranch feature-fusion connections in `ultralytics/cfg/models/building/building.yaml` |
| DySample | `DySample` in `ultralytics/nn/extra_modules/block.py` |
| LQEHead | `Detect_LQE` and `LQE` in `ultralytics/nn/extra_modules/head.py` |

The complete architecture is defined in `ultralytics/cfg/models/building/building.yaml`. Its first scale is `n`, corresponding to YOLO11n. The configuration retains the original `nc: 80` placeholder; the training dataset configuration supplies the task-specific classes. The model parser in this repository is tailored to the modules retained for `building.yaml`.

## Installation

Create a separate Python environment and install a PyTorch build compatible with your GPU and CUDA runtime before installing this repository:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Use the bundled `ultralytics` source, because the standard package does not contain these custom modules. Dependencies are listed in `pyproject.toml`.

The manuscript reports Python 3.13.3, PyTorch 2.10.0, CUDA 13.0, and an NVIDIA GeForce RTX 3090 for the experiments. These are the reported experimental versions, rather than a locked dependency specification for this release.

## Dataset

The manuscript describes 3,000 images and 13,542 annotated instances, with a 7:2:1 training/validation/test split. The seven detection labels are `eave`, `folk_style`, `gate`, `monument`, `royal_style`, `signboard`, and `window`. `folk_style` and `royal_style` are style-related labels.

The complete annotated dataset, its split files, and a task-specific dataset YAML are not included in this code release. `ultralytics/Field_Images/` contains example photographs, not the complete dataset. To train the model, prepare a YOLO-format dataset and a YAML file specifying its image paths and class names in the same order as the annotation class IDs.

## Training

From the repository root, substitute the path to your dataset YAML:

```bash
yolo detect train model=ultralytics/cfg/models/building/building.yaml data=path/to/data.yaml
```

Set the training hyperparameters to match your intended experiment. This command is a usage example; it does not specify all settings required to reproduce the manuscript results.

## Validation and prediction

Trained weights are not included. After training, substitute the path to your checkpoint:

```bash
yolo detect val model=path/to/best.pt data=path/to/data.yaml
yolo detect predict model=path/to/best.pt source=ultralytics/Field_Images
```

## Release scope

This release contains the supplied source code, model configuration, and example images. It does not include trained checkpoints or experiment logs. Publication checks cover source-file integrity, Python syntax, and package metadata. Full training, GPU inference, and reproduction of the reported results have not been performed as part of publication.

## License

The existing AGPL-3.0 license and upstream Ultralytics attribution are preserved. See `LICENSE`.