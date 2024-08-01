# Quadruped Robot Reinforcement Learning

This repository contains the implementation of reinforcement learning algorithms to enhance the navigation capabilities of a quadruped robot using Proximal Policy Optimization (PPO).

## Project Overview

This project is focused on developing and testing reinforcement learning strategies for a simulated quadruped robot navigating unseen terrains. The goal is to enable adaptive path planning using sensory inputs such as RGB images, depth images, and LiDAR data.

**Note:** This project is a work in progress, and results are still being refined.

## Features

- **Simulation Environment**: Uses Gazebo to simulate the quadruped robot.
- **Sensors**: Integrates RGBD camera and LiDAR data to perceive the environment.
- **Reinforcement Learning**: Implements PPO for training the robot to recognize terrains it is unable to navigate.

## Repository Structure

- `notspot_sim_py/`
  - `src/`
    - `notspot_joystick/`: Contains a joystick emulation script to control the robot.
    - `notspot_description/`: Contains the URDF of the robot.
    - `notspot_gazebo/`: Contains the simulation environments for use in Gazebo.
    - `notspot_controller/`: Contains the original kinematic controllers for the robot.
    - `reinforcement_learning/`: Contains the reinforcement learning algorithms and training scripts.


