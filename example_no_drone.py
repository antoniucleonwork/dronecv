import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic for multi-person tracking
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)  # Using the default camera

# Get screen width and height for fullscreen
screen_width = 1920  # You can set this to your screen resolution or use system functions to detect
screen_height = 1080  # Adjust as per your screen resolution

# Set the window to fullscreen
cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Person Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Define the focal length and object size (for distance estimation)
KNOWN_WIDTH = 0.5  # Approximate width of the human body in meters (e.g., shoulder width)
FOCAL_LENGTH = 600  # Focal length, adjust based on your camera (this value is an example)

# Function to calculate distance based on object size and focal length
def calculate_distance(width_in_frame):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_frame

# Function to draw a red arrow from the center of the screen to the middle of the chest
def draw_arrow(frame, chest_x, chest_y, frame_width, frame_height):
    center_x, center_y = frame_width // 2, frame_height // 2
    dx, dy = chest_x - center_x, chest_y - center_y
    cv2.arrowedLine(frame, (center_x, center_y), (chest_x, chest_y), (0, 0, 255), 5)  # Red color

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

        # Initialize variables for closest person
        closest_person_distance = float('inf')
        closest_person_chest_x, closest_person_chest_y = None, None

        # Since pose_landmarks is a single object, we access the coordinates of the person directly
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

        # Estimate the distance to the person
        width_in_frame = np.linalg.norm([shoulder_left_x - shoulder_right_x, shoulder_left_y - shoulder_right_y])
        distance = calculate_distance(width_in_frame)

        # Draw the arrow to the closest person's chest if one is detected
        draw_arrow(frame, chest_x, chest_y, w, h)
        cv2.putText(frame, f"Distance: {distance:.2f} meters", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the processed frame
    cv2.imshow("Person Tracking", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
