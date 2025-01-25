import cv2
import mediapipe as mp

# Initialize Mediapipe Pose class and drawing utility
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Correctly load the video file
cap = cv2.VideoCapture("videos/run1.mp4")  # Ensure the path is correct

# Initialize the Pose estimation model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to read video. Exiting...")
            break
        
        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform pose estimation
        results = pose.process(image)
        
        # Draw pose landmarks on the frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
        # Display the frame
        cv2.imshow('Pose Estimation', image)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
