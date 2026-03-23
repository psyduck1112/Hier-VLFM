# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VLFM (Vision-Language Frontier Maps) is a zero-shot semantic navigation system that uses vision-language models to navigate towards unseen objects in novel environments. The system builds occupancy maps from depth observations, identifies frontiers, and uses RGB observations with pre-trained VLMs to generate language-grounded value maps for exploration.

## Installation & Setup

### Environment Setup
```bash
conda_env_name=vlfm
conda create -n $conda_env_name python=3.9 -y
conda activate $conda_env_name
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2
```

For simulation (Habitat):
```bash
pip install -e .[habitat]
```

For real robot (Spot):
```bash
pip install -e .[reality]
```

Clone YOLOv7 within the repo:
```bash
git clone git@github.com:WongKinYiu/yolov7.git
```

### Required Model Weights
Download to `data/` directory:
- `mobile_sam.pt` from MobileSAM repo
- `groundingdino_swint_ogc.pth` from GroundingDINO repo
- `yolov7-e6e.pt` from YOLOv7 repo
- `pointnav_weights.pth` (included in data/ subdirectory)

## Running Experiments

### Launch VLM Servers (Required First)
Before evaluation, start the VLM model servers (only needs to be done once):
```bash
./scripts/launch_vlm_servers.sh
```
This creates a tmux session with 4 Flask servers (GroundingDINO, BLIP2ITM, SAM, YOLOv7). Wait up to 90 seconds for models to load. Kill the tmux session when done to free GPU.

### Evaluation
Evaluate on HM3D dataset:
```bash
python -m vlfm.run
```

Evaluate on MP3D dataset:
```bash
python -m vlfm.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz
```

### Testing
Run tests:
```bash
pytest test/
```

## Code Architecture

### Core Module Structure

**`vlfm/policy/`** - Navigation policies
- `base_objectnav_policy.py`: Base class for all object navigation policies. Manages VLM clients (GroundingDINO, YOLOv7, MobileSAM, BLIP2), PointNav policy, object/obstacle maps, and observation caching
- `itm_policy.py`: Image-Text Matching policy using BLIP2ITM for frontier value scoring
- `habitat_policies.py`: Habitat-specific policy wrappers
- `reality_policies.py`: Real-world robot deployment policies
- `utils/pointnav_policy.py`: PointNav policy wrapper for low-level navigation

**`vlfm/mapping/`** - Map representations
- `obstacle_map.py`: Generates navigable area map and explored area map from depth. Detects frontiers (boundary between explored/unexplored)
- `value_map.py`: Language-grounded value map using VLM confidence scores
- `frontier_map.py`: Frontier detection and management
- `object_point_cloud_map.py`: 3D point cloud map of detected objects
- `traj_visualizer.py`: Trajectory visualization utilities

**`vlfm/vlm/`** - Vision-Language Model clients
- Flask-based client-server architecture for VLM inference
- `grounding_dino.py`: Open-vocabulary object detection
- `yolov7.py`: COCO class object detection
- `sam.py`: MobileSAM segmentation client
- `blip2.py`: BLIP-2 VQA client
- `blip2itm.py`: BLIP-2 Image-Text Matching client

**`vlfm/utils/`** - Utilities
- `geometry_utils.py`: Coordinate transformations, point cloud operations
- `img_utils.py`: Image processing utilities
- `vlfm_trainer.py`: Custom Habitat trainer

### Key Data Flow

1. **Observations** → Sensor data (RGB, depth, GPS, compass) cached in `_observations_cache`
2. **Depth Processing** → ObstacleMap updates navigable areas and detects frontiers
3. **RGB + Frontiers** → VLM clients score frontiers based on text prompts
4. **Value Map** → Aggregates VLM scores across the map
5. **Frontier Selection** → Best frontier selected based on value scores
6. **PointNav** → Low-level policy navigates to selected frontier

### Configuration System

Uses Hydra for configuration management:
- Main configs in `config/experiments/`
- Default config: `vlfm_objectnav_hm3d.yaml`
- Override with: `python -m vlfm.run key=value`
- Entry point: `vlfm/run.py` with `@hydra.main` decorator

### VLM Server Architecture

The system uses a client-server architecture where VLM models run in separate Flask servers:
- **Ports**: GroundingDINO (12181), BLIP2ITM (12182), SAM (12183), YOLOv7 (12184)
- **Environment Variables**: `GROUNDING_DINO_PORT`, `BLIP2ITM_PORT`, `SAM_PORT`, `YOLOV7_PORT`
- Servers must be launched before running experiments
- All clients extend base client classes with `.predict()` methods

## Code Quality

### Linting & Formatting
```bash
# Format code
black .

# Run ruff linter
ruff check . --fix

# Type checking
mypy vlfm test scripts
```

### Pre-commit Hooks
Install hooks:
```bash
pre-commit install
```

Hooks include: ruff, black, mypy, check-yaml, check-added-large-files

## Important Implementation Details

### Coordinate Systems
- **Camera coordinates**: Right-handed, Y-up
- **Episodic coordinates**: Top-down map frame
- Transformations handled in `geometry_utils.xyz_yaw_to_tf_matrix()`

### Map Conventions
- Maps are boolean arrays (1000x1000 default, 20 pixels/meter)
- Origin at center of map
- ObstacleMap maintains: `_navigable_map`, `explored_area`, `frontiers`
- Robot radius padding applied with morphological erosion

### Observation Cache
Policies cache observations in `_observations_cache` dict to avoid repeated processing. Includes: RGB, depth, GPS, compass, frontier_sensor, plus derived values (robot_xy, camera_yaw, tf_camera_to_episodic).

### Object Detection Pipeline
1. COCO objects: YOLOv7 detection with `_coco_threshold` (default 0.8)
2. Non-COCO objects: GroundingDINO with `_non_coco_threshold` (default 0.4)
3. Segmentation: MobileSAM generates masks from detections
4. VQA (optional): BLIP2 verifies detections with text prompts

### Debugging
See `DEBUG_GUIDE.md` for detailed debugging instructions including:
- Using `breakpoint()` for interactive debugging
- Observation data inspection points
- Visualization techniques
- Debug data persistence
