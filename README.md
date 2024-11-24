# Person Tracking and Drone Control

This repository contains two Python scripts that demonstrate person tracking using MediaPipe and control a drone's movement based on the tracked position. Additionally, it showcases multi-person tracking and distance estimation in real-time using a webcam.

---

## Features

### 1. **Person Tracking for Drone Control** (First Script)
- Uses MediaPipe's Holistic Model for real-time human pose tracking.
- Calculates the distance from the camera to the person based on the size of the detected body.
- Estimates the movement direction of the tracked person and sends movement commands to a drone via the DroneKit library.

### 2. **Multi-Person Tracking with Distance Estimation** (Second Script)
- Tracks multiple people using MediaPipe’s Holistic Model.
- Displays a red arrow pointing from the center of the screen to the center of the chest of the closest person.
- Calculates the distance to the closest person based on the body width and focal length.

---

## Prerequisites

Before running the scripts, ensure you have the following libraries installed:

- **OpenCV**: For handling video capture and displaying results.
- **MediaPipe**: For detecting human pose landmarks.
- **NumPy**: For numerical operations (like calculating distances).
- **DroneKit**: For drone control and communication (for the first script).

You can install the necessary dependencies via pip:

```bash
pip install opencv-python mediapipe numpy dronekit
```

---

## Usage

**Person Tracking for Drone Control**
- Ensure your drone is properly connected to DroneKit (update the connection string).
- Run the Python script for person tracking and drone control:

```bash
python drone_tracking.py
```

**Multi-Person Tracking with Distance Estimation**
- Run the script for real-time multi-person tracking using your webcam:
```bash
python multi_person_tracking.py
```
-The script will estimate the distance of the closest person and display it. A red arrow will indicate the direction of the closest person's chest.
-Press q to exit the program.

---

## Parameters
- **KNOWN_WIDTH**: Approximate width of a human body (shoulder width) in meters. Default is 0.5 meters.
- **FOCAL_LENGTH**: Focal length of your camera (in pixels). Adjust based on your camera’s.
