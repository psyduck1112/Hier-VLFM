# VLFM JetRacer Workspace

This repository is a personal workspace built on top of the original VLFM codebase.
Current focus areas:

- Habitat simulation evaluation
- JetRacer + ROS real-robot experiments
- VLM integration (BLIP2 / GroundingDINO / SAM / YOLO-World)

The repository keeps the original license and copyright notices, with additional
modification copyright entries for local changes.

## Project Layout

- `vlfm/`: core implementation
- `vlfm/mapping/`: frontier/obstacle/value map logic
- `vlfm/policy/`: navigation policies and helpers
- `vlfm/reality/`: real-world environment wrappers
- `vlfm/vlm/`: vision-language model adapters
- `config/experiments/`: experiment configs
- `config/tasks/`: task configs
- `scripts/`: launch and utility scripts
- `docker/`: Docker build environment

## Environment Setup

Recommended Python version: 3.9

```bash
conda create -n vlfm python=3.9 -y
conda activate vlfm
```

Install base development dependencies:

```bash
pip install -e .[dev]
```

Install Habitat dependencies:

```bash
pip install -e .[habitat]
```

Install real-robot dependencies:

```bash
pip install -e .[reality]
```

## Data and Model Files

Place datasets, checkpoints, and generated outputs in local runtime folders such as:

- `data/`
- `outputs/`
- `tb/`

These folders are intentionally ignored by Git in this workspace.

Common required files:

- `data/dummy_policy.pth`
- `data/pointnav_weights.pth`
- extra model weights such as `mobile_sam.pt`, `groundingdino_swint_ogc.pth`, etc.

## Common Commands

Start VLM servers (typically required before Habitat eval):

```bash
./scripts/launch_vlm_servers.sh
```

Run default Habitat evaluation:

```bash
python -m vlfm.run
```

Run evaluation on MP3D episodes:

```bash
python -m vlfm.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz
```

Run JetRacer controller (host side):

```bash
python scripts/vlfm_jetracer_controller.py --target chair
```

Run JetRacer ROS runner:

```bash
python scripts/run_jetracer_ros.py --target chair
```

## Development Checks

Run formatting/lint/type hooks configured by pre-commit:

```bash
pre-commit run --all-files
```

Run tests:

```bash
pytest
```

## Git Hygiene in This Repo

The workspace is configured to ignore local artifacts such as:

- datasets and model weights (`data/`, `versioned_data/`, `*.pt`, `*.pth`, ...)
- logs and outputs (`outputs/`, `tb/`, `tb_valid/`, `*_visualizations/`)
- local links and caches (`site-packages-link`, `.mypy_cache/`, `.pytest_cache/`, ...)

If you need to track a specific ignored file, add an explicit whitelist rule in `.gitignore`.

## Source and License

- This workspace is derived from the Boston Dynamics AI Institute VLFM project.
- License remains [MIT](LICENSE).
- Original copyright notices are preserved.
