import streamlit as st
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from modules.pose_estimators import get_pose_analyzer
from modules.video_estimation import run_video_estimation, generate_report, save_session
from modules.webcam_transformers import WebcamPoseTransformer, WebcamPostureFeedbackTransformer, ExerciseAnalysisTransformer
from modules.image_analysis import run_image_analysis
from modules.session_history import display_session_history

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
    
    # ----- Analysis Modes -----
    if analysis_mode == "Live Webcam Pose Detection":
        st.header("Live Webcam Pose Detection")
        from streamlit_webrtc import webrtc_streamer
        webrtc_streamer(key="live-webcam",
                        video_transformer_factory=lambda: WebcamPoseTransformer(analyzer, threshold_val))
    elif analysis_mode == "Real-time Posture Feedback":
        st.header("Real-time Posture Feedback")
        st.info("Allow camera access. Alerts will be shown based on your posture.")
        from streamlit_webrtc import webrtc_streamer
        webrtc_streamer(key="posture-feedback",
                        video_transformer_factory=lambda: WebcamPostureFeedbackTransformer(analyzer, threshold_val, enable_alerts, alert_sensitivity))
    elif analysis_mode == "Exercise Analysis & Coaching":
        st.header("Exercise Analysis & Coaching")
        st.info("Perform squats in front of your webcam. The app will count your reps!")
        from streamlit_webrtc import webrtc_streamer
        webrtc_streamer(key="exercise-analysis",
                        video_transformer_factory=lambda: ExerciseAnalysisTransformer(analyzer, threshold_val))
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
    elif analysis_mode == "Session History":
        display_session_history()
    else:
        # Image Analysis Modes: "Basic Pose Detection", "Biomechanical Analysis", "Detailed Metrics", "3D Pose Visualization"
        uploaded_file = st.file_uploader("Upload Image for Pose Analysis", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error processing image: {e}")
                return
            with st.spinner('Processing image...'):
                pose_image, metrics_df, metrics = run_image_analysis(analyzer, image, threshold_val, model_choice, analysis_mode)
            report_zip = generate_report(np.array(image), pose_image, metrics_df)
            st.download_button(label="Download Pose Analysis Report", 
                               data=report_zip.getvalue(), 
                               file_name="pose_analysis_report.zip", 
                               mime="application/zip")
            if st.button("Save Session"):
                save_session(metrics, analysis_mode)
                if metrics_df is not None and not metrics_df.empty:
                    st.session_state["last_metrics_csv"] = metrics_df.to_csv(index=False).encode("utf-8")
        else:
            st.info("Please upload an image file for pose analysis.")
    
    # ----- Sidebar Instructions & Footer -----
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
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Developed with ❤️ as part of AICTE Internship on AI: Transformative Learning with Techsaksham, a collaborative initiative by Microsoft & SAP, in partnership with AICTE.
        Special thanks to ChatGPT for assisting in developing this website.</p>
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="https://www.linkedin.com/in/dinesh-x/">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
            </a>
            <a href="https://github.com/itzdineshx/Edunet_AI_internship_2025/">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
            </a>
        </div>
        <p style="margin-top: 20px;">© 2025 All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
