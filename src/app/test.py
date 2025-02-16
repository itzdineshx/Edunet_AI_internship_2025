

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import sys
sys.stderr = open(os.devnull, 'w')

import numpy as np
import cv2
import pandas as pd
import plotly.graph_objs as go
from PIL import Image
import time
import zipfile
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import tempfile
import imageio

# For MediaPipe support
import mediapipe as mp

# For TensorFlow Lite MoveNet
import tensorflow as tf

# For live webcam capture
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ------------------------------
# Custom CSS for Enhanced UI/UX
# ------------------------------
st.markdown("""
    <style>
    /* Main background */
    .main { 
        background-color: #f8f9fa;
    }
    /* Sidebar customizations */
    .css-1d391kg {  /* This class name might change in future versions */
        padding: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 2rem;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: #f8f9fa;
    }
    .css-18e3th9 {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

########################################
# Helper Functions for Biomechanical Analysis
########################################

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_biomechanics_mediapipe(points):
    metrics = {}
    if points[12] and points[14] and points[16]:
        metrics["right_arm_angle"] = calculate_angle(points[12], points[14], points[16])
    if points[11] and points[13] and points[15]:
        metrics["left_arm_angle"] = calculate_angle(points[11], points[13], points[15])
    if points[11] and points[12]:
        metrics["shoulder_width"] = np.linalg.norm(np.array(points[11]) - np.array(points[12]))
    if points[23] and points[24] and points[11] and points[12]:
        neck = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2)
        metrics["torso_alignment"] = calculate_angle(points[23], neck, points[24])
    return metrics

def calculate_biomechanics_movenet(points):
    metrics = {}
    if points[6] and points[8] and points[10]:
        metrics["right_arm_angle"] = calculate_angle(points[6], points[8], points[10])
    if points[5] and points[7] and points[9]:
        metrics["left_arm_angle"] = calculate_angle(points[5], points[7], points[9])
    if points[5] and points[6]:
        metrics["shoulder_width"] = np.linalg.norm(np.array(points[5]) - np.array(points[6]))
    if points[11] and points[12] and points[5] and points[6]:
        neck = ((points[5][0] + points[6][0]) / 2, (points[5][1] + points[6][1]) / 2)
        metrics["torso_alignment"] = calculate_angle(points[11], neck, points[12])
    return metrics

########################################
# Model Integration with Caching
########################################

@st.cache_resource
def get_model_instance(model_choice, model_path):
    if model_choice == "OpenPose":
        return OpenPoseAnalyzer(model_path)
    elif model_choice == "MediaPipe":
        return MediaPipePoseAnalyzer()
    elif model_choice == "MoveNet":
        return MoveNetAnalyzer(model_path)
    else:
        st.error("Unknown model selection.")
        return None

########################################
# OpenPose Analyzer
########################################

class OpenPoseAnalyzer:
    def __init__(self, model_path):
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path)
        except Exception as e:
            st.error(f"Error loading OpenPose model: {e}")
            self.net = None
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
            ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
            ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
            ["Nose", "LEye"], ["LEye", "LEar"]
        ]

    def detect_pose(self, frame, threshold=0.2):
        try:
            if self.net is None:
                return None
            if len(frame.shape) < 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frameWidth, frameHeight = frame.shape[1], frame.shape[0]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368),
                                         (127.5, 127.5, 127.5),
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            out = self.net.forward()
            out = out[:, :19, :, :]
            points = []
            for i in range(len(self.BODY_PARTS)):
                heatMap = out[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                points.append((int(x), int(y)) if conf > threshold else None)
            return points
        except Exception:
            return None

    def draw_pose(self, frame, points, threshold=0.2):
        if points is None:
            return frame
        for pair in self.POSE_PAIRS:
            partFrom, partTo = pair
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        return frame

    def calculate_body_metrics(self, points):
        if points is None:
            return {}
        metrics = {}
        if points[1] and points[2] and points[3]:
            metrics['right_arm_angle'] = self._calculate_angle(points[1], points[2], points[3])
        if points[1] and points[5] and points[6]:
            metrics['left_arm_angle'] = self._calculate_angle(points[1], points[5], points[6])
        if points[2] and points[5]:
            metrics['shoulder_width'] = self._calculate_distance(points[2], points[5])
        if points[1] and points[8] and points[11]:
            metrics['torso_alignment'] = self._calculate_angle(points[8], points[1], points[11])
        return metrics

    def _calculate_angle(self, point1, point2, point3):
        return calculate_angle(point1, point2, point3)

    def _calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


########################################
# MediaPipe Pose Analyzer with Skeleton Extraction
########################################

class MediaPipePoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, frame, threshold=0.2):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        points = [None] * 33
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                points[i] = (x, y)
        return points

    def draw_pose(self, frame, points, threshold=0.2):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def draw_skeleton(self, points, shape):
        blank = np.zeros(shape, dtype=np.uint8)
        for p in points:
            if p is not None:
                cv2.circle(blank, p, 5, (0, 255, 0), -1)
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start, end = connection
            if points[start] is not None and points[end] is not None:
                cv2.line(blank, points[start], points[end], (0, 255, 0), 2)
        return blank

    def calculate_body_metrics(self, points):
        return calculate_biomechanics_mediapipe(points)


########################################
# MoveNet Analyzer
########################################

class MoveNetAnalyzer:
    KEYPOINT_CONNECTIONS = [
        (0,1), (0,2),
        (1,3), (2,4),
        (5,6),
        (5,7), (7,9),
        (6,8), (8,10),
        (11,12),
        (11,13), (13,15),
        (12,14), (14,16),
        (5,11), (6,12)
    ]
    
    def __init__(self, model_path):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            st.error(f"Error loading MoveNet model: {e}")
            self.interpreter = None
        self.input_details = self.interpreter.get_input_details() if self.interpreter else None
        self.output_details = self.interpreter.get_output_details() if self.interpreter else None
        if self.input_details is not None:
            self.input_shape = self.input_details[0]['shape']
            self.inp_height = self.input_shape[1]
            self.inp_width = self.input_shape[2]

    def detect_pose(self, frame, threshold=0.2):
        if self.interpreter is None:
            st.error("MoveNet model not loaded.")
            return None
        input_img = cv2.resize(frame, (self.inp_width, self.inp_height))
        expected_dtype = self.input_details[0]['dtype']
        if expected_dtype == np.uint8:
            input_img = input_img.astype(np.uint8)
        else:
            input_img = input_img.astype(np.float32)
            input_img = (input_img / 127.5) - 1.0
        input_img = np.expand_dims(input_img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        if keypoints_with_scores.ndim == 4:
            keypoints_with_scores = np.squeeze(keypoints_with_scores, axis=1)
        keypoints = []
        for kp in keypoints_with_scores[0]:
            y, x, score = kp
            if score < threshold:
                keypoints.append(None)
            else:
                orig_h, orig_w, _ = frame.shape
                kp_x = int(x * orig_w)
                kp_y = int(y * orig_h)
                keypoints.append((kp_x, kp_y))
        return keypoints

    def draw_pose(self, frame, points, threshold=0.2):
        if points is None:
            return frame
        for pt in points:
            if pt is not None:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        for connection in self.KEYPOINT_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 < len(points) and idx2 < len(points) and points[idx1] is not None and points[idx2] is not None:
                cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 2)
        return frame

    def calculate_body_metrics(self, points):
        if points is None:
            return {}
        return calculate_biomechanics_movenet(points)

########################################
# Factory Function
########################################

def get_pose_analyzer(model_choice, model_path):
    return get_model_instance(model_choice, model_path)

########################################
# Helper Functions for Report and Session
########################################

def generate_report(original_image, pose_image, metrics_df):
    report_zip = BytesIO()
    with zipfile.ZipFile(report_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        with BytesIO() as buffer:
            Image.fromarray(original_image).save(buffer, format='PNG')
            zip_file.writestr("original_image.png", buffer.getvalue())
        with BytesIO() as buffer:
            rgb_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(buffer, format='PNG')
            zip_file.writestr("pose_image.png", buffer.getvalue())
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("body_metrics.csv", csv_data)
    report_zip.seek(0)
    return report_zip

def save_session(metrics, session_type):
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    session_data = {
        "session_type": session_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }
    st.session_state.session_history.append(session_data)
    st.success("Session saved successfully!")

def share_session():
    st.success("Session shared to cloud!")

########################################
# Video & File Processing Functions
########################################

def run_video_estimation(analyzer, video_file, threshold, record_video=False, extract_skeleton=False):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    recorded_frames = []
    skeleton_frames = []
    metrics_placeholder = st.empty()  # For realtime metrics
    st.info("Processing video...")
    last_points = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        points = analyzer.detect_pose(frame, threshold)
        last_points = points
        overlay_frame = analyzer.draw_pose(frame.copy(), points, threshold)
        frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        recorded_frames.append(frame_rgb)
        if extract_skeleton:
            if hasattr(analyzer, 'draw_skeleton'):
                skeleton_frame = analyzer.draw_skeleton(points, frame.shape)
            else:
                blank = np.zeros_like(frame)
                skeleton_frame = analyzer.draw_pose(blank.copy(), points, threshold)
            skeleton_frames.append(skeleton_frame)
        metrics = analyzer.calculate_body_metrics(points)
        metrics_placeholder.markdown("**Realtime Metrics:** " + str(metrics))
        time.sleep(0.03)
    cap.release()
    st.success("Video processing complete.")
    rec_video_bytes = None
    skel_video_bytes = None
    if record_video and recorded_frames:
        video_filename = "recorded_pose_video.mp4"
        imageio.mimwrite(video_filename, recorded_frames, fps=30, codec='libx264')
        with open(video_filename, "rb") as vid_file:
            rec_video_bytes = vid_file.read()
        st.download_button(label="Download Recorded Video", 
                           data=rec_video_bytes, 
                           file_name=video_filename, 
                           mime="video/mp4")
    if extract_skeleton and skeleton_frames:
        skeleton_filename = "extracted_skeleton_video.mp4"
        imageio.mimwrite(skeleton_filename, skeleton_frames, fps=30, codec='libx264')
        with open(skeleton_filename, "rb") as vid_file:
            skel_video_bytes = vid_file.read()
        st.download_button(label="Download Skeleton Video", 
                           data=skel_video_bytes, 
                           file_name=skeleton_filename, 
                           mime="video/mp4")
    return rec_video_bytes, skel_video_bytes, analyzer.calculate_body_metrics(last_points) if last_points is not None else {}

########################################
# Webcam Transformers for Live Modes
########################################

class WebcamPoseTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        return img

class WebcamPostureFeedbackTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold, enable_alerts, alert_sensitivity):
        self.analyzer = analyzer
        self.threshold = threshold
        self.enable_alerts = enable_alerts
        self.alert_sensitivity = alert_sensitivity

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        metrics = self.analyzer.calculate_body_metrics(points)
        if metrics and "torso_alignment" in metrics and self.enable_alerts:
            deviation = abs(180 - metrics["torso_alignment"])
            if deviation > self.alert_sensitivity:
                cv2.putText(img, "Adjust your posture!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

class ExerciseAnalysisTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold
        self.rep_count = 0
        self.in_squat = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        if hasattr(self.analyzer, '_calculate_angle') and points[8] and points[9] and points[10]:
            angle = self.analyzer._calculate_angle(points[8], points[9], points[10])
            if angle < 90 and not self.in_squat:
                self.in_squat = True
            if angle > 160 and self.in_squat:
                self.rep_count += 1
                self.in_squat = False
            cv2.putText(img, f"Squat Reps: {self.rep_count}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return img

########################################
# Main Application
########################################

def main():
    st.title("🏋️ Advanced Pose Estimation")
    st.markdown("Comprehensive pose analysis and biometric insights")
      
    # ----- Model Selection -----
    model_choice = st.sidebar.selectbox("Select Pose Estimation Model", 
                                          ["OpenPose", "MediaPipe", "MoveNet"])
    MODEL_PATH = "/workspaces/Edunet_AI_internship_2025/models/graph_opt.pb"
    if model_choice == "MoveNet":
        moveNet_model_path = "/workspaces/Edunet_AI_internship_2025/models/movenet_lightning_fp16.tflite"
        analyzer = get_pose_analyzer(model_choice, moveNet_model_path)
    else:
        analyzer = get_pose_analyzer(model_choice, MODEL_PATH)

    analysis_mode = st.sidebar.selectbox("Analysis Mode", [
        "Basic Pose Detection", 
        "Biomechanical Analysis", 
        "Detailed Metrics",
        "3D Pose Visualization",
        "Video Pose Estimation",
        "Live Webcam Pose Detection",
        "Real-time Posture Feedback",
        "Exercise Analysis & Coaching",
        "Session History"
    ])
    threshold_val = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)
    alert_sensitivity = st.sidebar.slider("Alert Sensitivity (° deviation)", 0, 30, 10)

    # Clear Session button
    if st.sidebar.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Session cleared!")
      
    if st.sidebar.button("Download All as ZIP"):
        all_files = {}
        if "last_original" in st.session_state:
            all_files["original_image.png"] = st.session_state["last_original"]
        if "last_pose" in st.session_state:
            all_files["pose_image.png"] = st.session_state["last_pose"]
        if "last_skeleton" in st.session_state:
            all_files["skeleton_image.png"] = st.session_state["last_skeleton"]
        if "last_recorded_video" in st.session_state:
            all_files["recorded_pose_video.mp4"] = st.session_state["last_recorded_video"]
        if "last_skeleton_video" in st.session_state:
            all_files["extracted_skeleton_video.mp4"] = st.session_state["last_skeleton_video"]
        if "last_metrics_csv" in st.session_state:
            all_files["metrics.csv"] = st.session_state["last_metrics_csv"]
    
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, file_data in all_files.items():
                zip_file.writestr(file_name, file_data)
        zip_buffer.seek(0)
        st.sidebar.download_button("Download All as ZIP", zip_buffer, "session_files.zip", "application/zip")
    

    if analysis_mode == "Video Pose Estimation":
        record_video = st.sidebar.checkbox("Record Processed Video", value=True)
        extract_skel_video = st.sidebar.checkbox("Extract Skeleton Video", value=True)

    if analyzer is None:
        st.error("No valid analyzer loaded. Please check your model selection and file path.")
        return
    
    # ----- Live Webcam Modes -----
    if analysis_mode == "Live Webcam Pose Detection":
        st.header("Live Webcam Pose Detection")
        webrtc_streamer(key="live-webcam",
                        video_transformer_factory=lambda: WebcamPoseTransformer(analyzer, threshold_val))
    elif analysis_mode == "Real-time Posture Feedback":
        st.header("Real-time Posture Feedback")
        st.info("Allow camera access. Alerts will be shown based on your posture.")
        webrtc_streamer(key="posture-feedback",
                        video_transformer_factory=lambda: WebcamPostureFeedbackTransformer(analyzer, threshold_val, enable_alerts, alert_sensitivity))
    elif analysis_mode == "Exercise Analysis & Coaching":
        st.header("Exercise Analysis & Coaching")
        st.info("Perform squats in front of your webcam. The app will count your reps!")
        webrtc_streamer(key="exercise-analysis",
                        video_transformer_factory=lambda: ExerciseAnalysisTransformer(analyzer, threshold_val))
    # ----- Video Pose Estimation (File Upload) -----
    elif analysis_mode == "Video Pose Estimation":
        st.header("Video Pose Estimation")
        video_file = st.file_uploader("Upload Video for Pose Analysis", type=["mp4", "avi", "mov", "gif"])
        if video_file is not None:
            rec_vid, skel_vid, metrics = run_video_estimation(analyzer, video_file, threshold_val, record_video, extract_skeleton=extract_skel_video)
            if rec_vid:
                st.session_state["last_recorded_video"] = rec_vid
            if skel_vid:
                st.session_state["last_skeleton_video"] = skel_vid
            if metrics:
                st.subheader("Realtime Metrics (from last frame)")
                st.write(metrics)
                csv_bytes = pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8")
                st.session_state["last_metrics_csv"] = csv_bytes
            if st.button("Save Session"):
                save_session(metrics, "Video Pose Estimation")
        else:
            st.info("Please upload a video file to start pose estimation.")
    # ----- Image Analysis Modes -----
    else:
        uploaded_file = st.file_uploader("Upload Image for Pose Analysis", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = np.array(image)
            except Exception as e:
                st.error(f"Error processing image: {e}")
                return
            with st.spinner('Processing image...'):
                points = analyzer.detect_pose(image, threshold_val)
                if points is not None:
                    pose_image = image.copy()
                    pose_image = analyzer.draw_pose(pose_image, points, threshold_val)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.subheader("Pose Estimation")
                        st.image(pose_image, use_container_width=True)
                    
                    extract_skel = st.checkbox("Extract Skeleton Image", value=False)
                    if extract_skel:
                        if model_choice == "MediaPipe" and hasattr(analyzer, 'draw_skeleton'):
                            skeleton_img = analyzer.draw_skeleton(points, image.shape)
                        else:
                            blank_img = np.zeros_like(image)
                            skeleton_img = analyzer.draw_pose(blank_img.copy(), points, threshold_val)
                        st.image(skeleton_img, caption="Extracted Skeleton", use_container_width=True)
                        skeleton_bytes = cv2.imencode('.png', skeleton_img)[1].tobytes()
                        st.download_button("Download Skeleton Image", skeleton_bytes, "skeleton_image.png", "image/png")
                        st.session_state["last_skeleton"] = skeleton_bytes
                    
                    st.session_state["last_original"] = cv2.imencode('.png', image)[1].tobytes()
                    st.session_state["last_pose"] = cv2.imencode('.png', pose_image)[1].tobytes()
                    
                    metrics_df = pd.DataFrame()
                    if analysis_mode == "Biomechanical Analysis":
                        st.header("Biomechanical Insights")
                        if model_choice == "MediaPipe":
                            metrics = calculate_biomechanics_mediapipe(points)
                        elif model_choice == "MoveNet":
                            metrics = calculate_biomechanics_movenet(points)
                        else:
                            metrics = analyzer.calculate_body_metrics(points)
                        cols = st.columns(2)
                        with cols[0]:
                            st.subheader("Joint Angles")
                            for key, value in metrics.items():
                                st.metric(key, f"{value:.2f}")
                        with cols[1]:
                            st.subheader("Body Symmetry")
                    elif analysis_mode == "Detailed Metrics":
                        st.header("Comprehensive Body Metrics")
                        point_names = [f"Point {i}" for i in range(len(points))]
                        point_coords = [p if p else (np.nan, np.nan) for p in points]
                        metrics_df = pd.DataFrame({
                            'Body Part': point_names,
                            'X Coordinate': [p[0] for p in point_coords],
                            'Y Coordinate': [p[1] for p in point_coords]
                        })
                        st.dataframe(metrics_df)
                        valid_points = [p for p in points if p is not None]
                        x_coords = [p[0] for p in valid_points]
                        y_coords = [p[1] for p in valid_points]
                        fig = go.Figure(data=[go.Scatter(
                            x=x_coords, y=y_coords, 
                            mode='markers+text', 
                            marker=dict(size=10, color='red'),
                            text=point_names[:len(valid_points)],
                            textposition="bottom center"
                        )])
                        fig.update_layout(
                            title="Body Points Spatial Distribution",
                            xaxis_title="X Coordinate",
                            yaxis_title="Y Coordinate",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif analysis_mode == "3D Pose Visualization":
                        st.header("3D Pose Visualization")
                        valid_points = [p for p in points if p is not None]
                        x_coords = [p[0] for p in valid_points]
                        y_coords = [p[1] for p in valid_points]
                        z_coords = np.random.rand(len(valid_points)) * 100
                        fig = px.scatter_3d(
                            x=x_coords, y=y_coords, z=z_coords,
                            labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
                            title="3D Pose Visualization"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.header("Basic Pose Detection")
                    
                    report_zip = generate_report(image, pose_image, metrics_df)
                    st.download_button(label="Download Pose Analysis Report", 
                                       data=report_zip.getvalue(), 
                                       file_name="pose_analysis_report.zip", 
                                       mime="application/zip")
                    
                    metrics = analyzer.calculate_body_metrics(points)
                    if st.button("Save Session"):
                        save_session(metrics, analysis_mode)
                        if not metrics_df.empty:
                            st.session_state["last_metrics_csv"] = metrics_df.to_csv(index=False).encode("utf-8")
                else:
                    st.error("Pose detection failed.")
        else:
            st.info("Please upload an image file for pose analysis.")

    st.sidebar.info("Instructions:")
    st.sidebar.markdown("""
    **For Image Analysis:**  
    1. Upload an image (PNG, JPG, or JPEG).  
    2. View the pose, metrics, and optionally extract the skeleton image.

    **For Video Analysis:**  
    1. Upload a video file (MP4, AVI, MOV, or GIF).  
    2. The video will be processed in real time with processed frames shown and metrics updated below.
       You can also record the processed video and/or extract a skeleton video.
       
    **For Live Webcam Modes:**  
    1. Allow camera access when prompted.  
    2. Real‑time pose estimation and overlays will appear.

    **Session History:**  
    1. View and share saved session data.
    """)

    # ----- Session History -----
    if analysis_mode == "Session History":
        st.header("Session History")
        if "session_history" in st.session_state and st.session_state.session_history:
            history_df = pd.DataFrame([
                {"Session Type": s["session_type"],
                 "Timestamp": s["timestamp"],
                 "Metrics": s["metrics"]} 
                for s in st.session_state.session_history
            ])
            st.dataframe(history_df)
            st.download_button("Download History CSV",
                               history_df.to_csv(index=False).encode("utf-8"),
                               "session_history.csv", "text/csv")
            if st.button("Share Session"):
                share_session()
        else:
            st.info("No sessions saved yet.")
    
    # ----- Footer -----
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Poseji © 2025 DINESH S All Rights Reserved</p>
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="https://www.linkedin.com/in/dinesh-x/">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
            </a>
            <a href="https://github.com/itzdineshx">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
            </a>
        </div>
        <p style="margin-top: 20px;">© 2025 All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
