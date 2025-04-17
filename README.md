# Safe Continual Reinforcement Learning Environments

## Repository Structure

- [./code/](./code/) contains all code for the project
- [./code/Metaworld](./code/Metaworld/) contains a modified version of the [Meta World](https://github.com/Farama-Foundation/Metaworld) (Continual World) environment
- [./code/Metaworld/metaworld/envs/mujoco/sawyer_xyz/safe_cw/](./code/Metaworld/metaworld/envs/mujoco/sawyer_xyz/safe_cw/) contains the custom Safe Continual World environments
- [./code/safety-gymnasium/](./code/safety-gymnasium/) is a safe RL gymnasium library from [this repo](https://github.com/PKU-Alignment/safety-gymnasium)
- [./code/safety-gymnasium/safety-gymnasium/tasks/](./code/safety-gymnasium/safety_gymnasium/tasks) contains the [Safe Continual World](./code/safety-gymnasium/safety_gymnasium/tasks/safe_continual_world/safety_continual_world.py) and [Damaged HalfCheetah Velocity](./code/safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_half_cheetah_velocity_v4.py) environments, used by the algorithms above
- The Jupyter Notebooks show examples of how to use the environments

## Installation

1. **[Recommended, but not required]** Create a conda env with python version 3.10 (`conda create -n safe_continual python=3.10.15`) and activate it (`conda activate safe_continual`). Install pip in that environment using `conda install pip`.
2. Enter the Metaworld directory (`cd code/Metaworld`) and `python -m pip install -e .`
3. Enter the safety-gymnasium directory (`cd code/safety-gymnasium`) and `python -m pip install -e .`
