## Autonomous Robot Maze Navigation Using Extended Kalman Filter

![](https://github.com/TonyLiu836/Autonomous-Robot-Maze-Nav/blob/main/Robot_Nav_vid_SpedUp.gif)

Robot autonomously navigates maze environment to goal location (represented by green sphere) while avoiding contact with walls/obstacles in custom WeBots environment.

Robot:
  - Differential drive
  - 180 degree lidar
  - wheel encoders with noisy measurements
  - Front facing RGB camera

Environment:
  - Global features scattered thorughout used by robot to localize itself (red spheres)
  - Maze and open areas

Localization & Path Planning:
  - EKF for robot position estimation
  - Newton's method for non-linear optimization
  - Bug 2 algorithm for global path planning when not near walls/obstacles
