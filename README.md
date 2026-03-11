# UAV Autonomous Object Detection & 3D Localization

Autonomous navigation and perception system developed as part of my Engineering Thesis at Poznań University of Technology.

## Project Overview
The system enables a lightweight UAV (Crazyflie 2.1+) to perform real-time environment analysis. It integrates advanced computer vision with flight control to not only detect objects but also precisely locate them in a global coordinate system.

### Key Functionalities
* **Real-time Object Detection:** Utilizes **YOLOv12** implemented on an offboard station, processing video streams from the **AI-deck 1.1** (GAP8 RISC-V).
* **Distance Estimation:** Calculates the distance to the detected object based on the pinhole camera model and geometry-based **Passive Ranging**.
* **3D Coordinate Mapping:** Determines the exact global coordinates ($X, Y, Z$) of the object by fusing visual data (pixel offset, FOV) with real-time telemetry from the **OptiTrack** motion capture system.
* **Optimized Data Flow:** Integration of the **UDP-based streaming protocol by LARICS**, ensuring minimal latency and high stability in the perception-action loop.
  
## Tech Stack
* **Language:** Python 3.x
* **AI/Vision:** Ultralytics YOLOv12, OpenCV
* **UAV Control:** Bitcraze `cflib`
* **Communication:** Socket (UDP), JSON
* **Hardware:** Crazyflie 2.1+, AI-deck 1.1, OptiTrack Mocap

## License & Copyright
**Copyright (c) 2026 Dawid Zieliński. All rights reserved.**

This repository is for **portfolio demonstration purposes only**. The source code is NOT open-source. 
No part of this project (code, algorithms, or documentation) may be copied, redistributed, or used for any purpose without explicit written permission from the author.
