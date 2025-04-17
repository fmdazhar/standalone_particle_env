# Standalone Particle Environment

This repository contains a standalone particle-based simulation environment using NVIDIA's Isaac Sim. The environment is designed to calibrate particle systems with various configurations and also simulate quadruped to inetract with it, using GPU-accelerated physics and terrain features.

## System Requirements

It is recommended to have at least 32GB RAM and a GPU with at least 12GB VRAM. For detailed system requirements, please visit the [Isaac Sim System Requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements) page. Please refer to the [Troubleshooting](docs/troubleshoot.md#memory-consumption) page for a detailed breakdown of memory consumption.

## Installation

Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install the latest Isaac Sim release. 

*Examples in this repository rely on features from the most recent Isaac Sim release. Please make sure to update any existing Isaac Sim build to the latest release version, 4.5.0, to ensure examples work as expected.*

Once cloned, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html). By default, this should be `python.sh`. We will refer to this path as `PYTHON_PATH`.

To set a `PYTHON_PATH` variable in the terminal that links to the python executable, we can run a command that resembles the following. Make sure to update the paths to your local path.

```
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
For IsaacSim Docker: alias PYTHON_PATH=/isaac-sim/python.sh
```


## Files in This Repository

- **calibrate.usd**: The USD file for scene calibration.
- **config.yaml**: Configuration file for particle and simulation settings.
- **default_scene_params.py**: Default scene parameter settings for the environment.
- **particle_tune_env_standalone.py**: Main script to launch and control the standalone particle environment with an Anymal robot.
- **particle_tune.py**: Script for tuning the particle system independently.
- **README.md**: This documentation file.
- **sim_config.py**: Defines simulation configurations and related settings.


## Running the examples

For a simplified environment to just test particle behavior, run:

```bash
PYTHON_PATH particle_tune.py
```
Customize config section from `config.yaml` based on the desired particle properties.

It is also possible to use calibrate.usd file directly in the GUI for calibration.



To start the simulation along with quadruped, navigate to the cloned repository folder and run:

```bash
PYTHON_PATH particle_tune_env_standalone.py
```

Use the keyboard to control the Anymal robot's movement in the simulation:

- **Forward**: `NUMPAD_8` or `UP`
- **Backward**: `NUMPAD_2` or `DOWN`
- **Move Left**: `NUMPAD_4` or `LEFT`
- **Move Right**: `NUMPAD_6` or `RIGHT`
- **Yaw Left**: `NUMPAD_7` or `N`
- **Yaw Right**: `NUMPAD_9` or `M`

These controls allow you to navigate and rotate the robot within the simulation environment.
