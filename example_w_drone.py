import cv2
import mediapipe as mp
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)  # Using the default camera

# Connect to the vehicle
vehicle = connect('127.0.0.1:14551', wait_ready=True)  # Update this with your drone's connection string

# Function to calculate distance based on object size and focal length
def calculate_distance(width_in_frame):
    KNOWN_WIDTH = 0.5  # Approximate width of the human body in meters (e.g., shoulder width)
    FOCAL_LENGTH = 600  # Focal length, adjust based on your camera
    return (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_frame

# Function to get movement direction
def calculate_direction(center_x, center_y, frame_width, frame_height):
    dx = (center_x - frame_width // 2) / frame_width  # Proportional error
    dy = (center_y - frame_height // 2) / frame_height
    return dx, dy

# Send movement commands to the drone
def move_drone(dx, dy):
    speed = 5  # Speed of the drone's movement, adjust as necessary

    # Use proportional control to decide on velocity
    # Control movement along the x and y axis (the horizontal plane)
    velocity_x = dx * speed
    velocity_y = dy * speed

    # Set the velocity on the drone
    vehicle.channels.overrides['1'] = velocity_x  # Roll (right/left)
    vehicle.channels.overrides['2'] = velocity_y  # Pitch (forward/backward)

# Function to move the drone to a position relative to the current location
def move_to_target(target_location):
    current_location = vehicle.location.global_frame
    vehicle.simple_goto(target_location)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a more natural first-person view
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect holistic landmarks (including pose landmarks)
    result = holistic.process(rgb_frame)
    
    if result.pose_landmarks:
        # Draw landmarks on the person
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Find chest center using shoulder and hip landmarks
        landmarks = result.pose_landmarks
        shoulder_left = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        hip_left = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        hip_right = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]

        # Convert normalized coordinates to pixel values
        h, w, c = frame.shape
        shoulder_left_x, shoulder_left_y = int(shoulder_left.x * w), int(shoulder_left.y * h)
        shoulder_right_x, shoulder_right_y = int(shoulder_right.x * w), int(shoulder_right.y * h)
        hip_left_x, hip_left_y = int(hip_left.x * w), int(hip_left.y * h)
        hip_right_x, hip_right_y = int(hip_right.x * w), int(hip_right.y * h)

        # Calculate the arithmetic mean of these points to find the center of the chest
        chest_x = (shoulder_left_x + shoulder_right_x + hip_left_x + hip_right_x) // 4
        chest_y = (shoulder_left_y + shoulder_right_y + hip_left_y + hip_right_y) // 4

        # Calculate distance (optional: can be used to adjust speed or other parameters)
        width_in_frame = np.linalg.norm([shoulder_left_x - shoulder_right_x, shoulder_left_y - shoulder_right_y])
        distance = calculate_distance(width_in_frame)

        # Calculate direction (dx, dy) for controlling drone movement
        dx, dy = calculate_direction(chest_x, chest_y, w, h)

        # Move the drone towards the center of the body
        move_drone(dx, dy)
        
        # Optionally, use simple_goto to fly towards a location (based on some predefined distance and angle)
        # target_location = LocationGlobalRelative(vehicle.location.global_frame.lat + dx, vehicle.location.global_frame.lon + dy, 10)
        # move_to_target(target_location)
        
    # Show the processed frame
    cv2.imshow("Person Tracking", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
vehicle.close()
