# Traditional Chinese architectural component detection

A Detection Method for Ancient Architectural Components Based on an Improved YOLO11 Model:
The detector builds on YOLO11n and combines C3K2-DRG, MAFPN, DySample, and LQEHead.

## Implementation

| Manuscript name | Module description |
| --- | --- |
| C3K2-DRG | C3k2_DRG combines long-range residual learning, channel attention, and dynamic spatial attention to improve fine-grained feature representation.`|
| MAFPN |  MAFPN fuses low-level spatial details with high-level semantic information through multiple auxiliary branches. |
| DySample | Adjusts sampling locations using learned input-dependent offsets to improve spatial alignment during multiscale feature fusion. |
| LQEHead | Estimates localization quality from predicted bounding box distributions and uses these estimates to calibrate classification confidence. |

The complete architecture is defined in `ultralytics/cfg/models/building/building.yaml`.

## Installation

Run the following commands from the repository root to create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip, then install a PyTorch build compatible with your GPU and CUDA environment using the instructions on the [PyTorch installation page](https://pytorch.org/get-started/locally/):

```bash
python -m pip install --upgrade pip
```

Install the remaining dependencies:

```bash
python -m pip install numpy matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop
```

Please run the code from the repository root to use the local `ultralytics` source containing the custom modules. All of the above installation steps are executed using Python commands.

The experiments used Python 3.13.3, PyTorch 2.10.0, CUDA 13.0, and an NVIDIA GeForce RTX 3090 GPU.

## Dataset

The original field image data of architectural components are available in `ultralytics/Field_Images/`. Before training the model, please prepare a dataset in YOLO format and a task-specific dataset YAML configuration file. The YAML file should specify the image paths and class names. The order of the class names must be consistent with the class IDs used in the annotations.

## Training

Run from the repository root and replace `path/to/data.yaml` with your dataset configuration:

```bash
python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/building/building.yaml'); model.train(data='path/to/data.yaml')"
```

## Validation and prediction

Replace `path/to/best.pt` with your trained checkpoint and `path/to/data.yaml` with your dataset configuration. Run both commands from the repository root.

Validate:

```bash
python -c "from ultralytics import YOLO; model = YOLO('path/to/best.pt'); model.val(data='path/to/data.yaml')"
```

Predict:

```bash
python -c "from ultralytics import YOLO; model = YOLO('path/to/best.pt'); model.predict(source='ultralytics/Field_Images', save=True)"
```

## License

The existing AGPL-3.0 license and upstream Ultralytics attribution are preserved. See `LICENSE`.
