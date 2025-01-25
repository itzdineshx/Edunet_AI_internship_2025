import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import math

# Paths to demo files
DEMO_IMAGE = '/workspaces/Edunet_AI_internship_2025/images/stand.jpg'
MODEL_PATH = "/workspaces/Edunet_AI_internship_2025/images/graph_opt.pb"

# Body parts and pose pairs for OpenPose
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load the DNN model
try:
    net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function for pose detection
def poseDetector(frame, net, threshold):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract only relevant parts

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]

        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 2)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame, points

# Function to calculate the angle between 3 points
def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    angle = math.degrees(math.acos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))))
    return angle

# Streamlit UI
st.title("Human Pose Estimation with OpenCV")

# Light/Dark theme toggle
theme = st.selectbox("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body{background-color:#2c3e50; color:white;} </style>", unsafe_allow_html=True)

# Option for image or video upload
st.subheader("Pose Estimation - Choose Upload Type")
upload_type = st.radio("Select Upload Type", ["Single/Multiple Images", "Video / Live Webcam"])

# File upload for image
ALLOWED_IMAGE_TYPES = ['jpg', 'jpeg', 'png']  # List of allowed image formats

# Image Upload Section
if upload_type == "Single/Multiple Images":
    img_file_buffer = st.file_uploader("Upload an image", type=ALLOWED_IMAGE_TYPES, accept_multiple_files=True)
    if img_file_buffer:
        for uploaded_image in img_file_buffer:
            # Check if the uploaded file type is valid
            if uploaded_image.name.split('.')[-1].lower() in ALLOWED_IMAGE_TYPES:
                image = np.array(Image.open(uploaded_image))
                st.subheader(f"Pose Estimation for {uploaded_image.name}")
                output_image, points = poseDetector(image.copy(), net, threshold=0.2)
                
                # Calculate joint angles (for example, shoulder, elbow, wrist angle)
                if points[2] and points[3] and points[4]:
                    angle = calculate_angle(points[2], points[3], points[4])
                    st.write(f"Angle between shoulder, elbow, and wrist: {angle:.2f}°")
                
                st.image(output_image, caption=f"Pose Estimation for {uploaded_image.name}", use_column_width=True)

                # Download button for processed image
                output_pil = Image.fromarray(output_image)
                buf = io.BytesIO()
                output_pil.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="Download Processed Image",
                    data=buf,
                    file_name=f"pose_estimation_{uploaded_image.name}",
                    mime="image/png"
                )
            else:
                st.error(f"Unsupported file format for {uploaded_image.name}. Please upload images in 'jpg', 'jpeg', or 'png' formats.")

# Video Upload / Webcam Section
elif upload_type == "Video / Live Webcam":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    # Handle video upload for pose detection
    if video_file is not None:
        video_bytes = video_file.read()
        video_path = "/tmp/uploaded_video.mp4"

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(video_path)
        st.subheader("Processed Video with Pose Estimation")

        video_out = "/tmp/processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out, fourcc, 20.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_frame, points = poseDetector(frame, net, threshold=0.2)
            
            # Calculate joint angles in video (for example, shoulder, elbow, wrist angle)
            if points[2] and points[3] and points[4]:
                angle = calculate_angle(points[2], points[3], points[4])
                cv2.putText(output_frame, f"Angle: {angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(output_frame)

        cap.release()
        out.release()

        # Provide download button for the processed video
        st.video(video_out)

        with open(video_out, "rb") as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    # Real-time Webcam Pose Detection
    elif st.button("Start Webcam Pose Detection"):
        st.subheader("Webcam Live Pose Estimation")
        cap = cv2.VideoCapture(0)  # Open webcam

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame, points = poseDetector(frame, net, threshold=0.2)
            
            # Calculate joint angles in real-time
            if points[2] and points[3] and points[4]:
                angle = calculate_angle(points[2], points[3], points[4])
                cv2.putText(output_frame, f"Angle: {angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display output frame in Streamlit
            st.image(output_frame, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop webcam feed
                break

        cap.release()
