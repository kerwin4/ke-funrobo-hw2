# FunRobo Mini-Projects 1 & 2

This repository contains the implementations for the Fundamentals of Robotics Individual Homework 2. The files cover quintic polynomial and trapezoidal trajectory generation methods for a robotic platform of unspecified DoF, as well as a visualization of the trajectory methods used on a Hiwonder 5-DoF platform.

## Trajectory Generation - ```traj_gen.py```

Implements classes for quintic polynomial and trapezoidal trajectory generation method. Plots example position, velocity, and acceleration graphs for a 2-DoF platform moving between 4 waypoints: [-30, 30], [0, 45], [30, 15], [50, -30]. The trajectory method is specified by the "method" variable.

## Visualization - ```hiwonder.py```

Implements trajectory generation methods in the visualizer tool using the classes from ```traj_gen.py```. Retains FK and IK functionality from MP1 and MP2 implementations.

These files are drop-in replacements for the identically named files in the path-planning branch of the [funrobo_kinematics repository](https://github.com/OlinCollege-FunRobo/funrobo_kinematics/tree/path-planning) repository.