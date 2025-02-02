import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objs as plt
from PIL import Image
import os
import time
import zipfile
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

class PoseAnalyzer:
    def __init__(self, model_path):
        # Use try-except to handle model loading errors
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path)
        except Exception as e:
            st.error(f"Error loading pose estimation model: {e}")
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
        if self.net is None:
            st.error("Pose estimation model not loaded.")
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
        
        # Arm Angles
        if points[1] and points[2] and points[3]:
            metrics['right_arm_angle'] = self._calculate_angle(points[1], points[2], points[3])
        
        if points[1] and points[5] and points[6]:
            metrics['left_arm_angle'] = self._calculate_angle(points[1], points[5], points[6])
        
        # Body Symmetry
        if points[2] and points[5]:
            metrics['shoulder_width'] = self._calculate_distance(points[2], points[5])
        
        # Posture Analysis
        if points[1] and points[8] and points[11]:
            metrics['torso_alignment'] = self._calculate_angle(points[8], points[1], points[11])
        
        return metrics

    def _calculate_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def _calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


def generate_report(original_image, pose_image, metrics_df):
    """Generate a ZIP report with properly encoded images and CSV data"""
    report_zip = BytesIO()
    
    with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        # Save original image as PNG
        with BytesIO() as buffer:
            Image.fromarray(original_image).save(buffer, format='PNG')
            zip_file.writestr("original_image.png", buffer.getvalue())

        # Save pose image (convert BGR to RGB for proper color representation)
        with BytesIO() as buffer:
            # Convert OpenCV's BGR format to RGB
            rgb_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(buffer, format='PNG')
            zip_file.writestr("pose_image.png", buffer.getvalue())

        # Save metrics as CSV with proper encoding
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("body_metrics.csv", csv_data)

    report_zip.seek(0)  # Reset buffer position to beginning
    return report_zip

def main():
    st.set_page_config(page_title="Advanced Pose Estimation", layout="wide")
    
    st.title("🏋️ Advanced Human Pose Estimation")
    st.markdown("Comprehensive pose analysis and biomechanical insights")

    # Model Path Configuration
    MODEL_PATH = "src/graph_opt.pb"

    # Sidebar Configuration
    st.sidebar.header("Pose Analysis Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    analysis_mode = st.sidebar.selectbox("Analysis Mode", [
        "Basic Pose Detection", 
        "Biomechanical Analysis", 
        "Detailed Metrics",
        "3D Pose Visualization",
        "Real-time Pose Estimation"
    ])

    st.sidebar.info("Steps to upload an image:")
    st.sidebar.markdown("""
    1. Click the 'Browse Files' button below.
    2. Select a PNG, JPG, or JPEG image from your computer.
    3. Once the image is uploaded, it will appear on the main page.
    4. Ensure the image is clear and of good quality for analysis.
    """)

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found. Please download the pre-trained model.")
        return

    # Pose Estimator Initialization
    estimator = PoseAnalyzer(MODEL_PATH)

    # File Upload
    uploaded_file = st.file_uploader("Upload Image for Pose Analysis", 
                                     type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read image with PIL to handle various image formats
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return

        # Show loading spinner
        with st.spinner('Processing image...'):
            # Pose Detection
            points = estimator.detect_pose(image, threshold)
            
            if points is not None:
                pose_image = image.copy()
                pose_image = estimator.draw_pose(pose_image, points, threshold)

                # Display Original and Pose Images with a more seamless layout
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("Pose Estimation")
                    st.image(pose_image, use_container_width=True)

                # Initialize metrics_df to prevent UnboundLocalError
                metrics_df = pd.DataFrame()

                # Detailed Analysis Based on Mode
                if analysis_mode == "Biomechanical Analysis":
                    st.header("Biomechanical Insights")
                    metrics = estimator.calculate_body_metrics(points)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.subheader("Joint Angles")
                        if 'right_arm_angle' in metrics:
                            st.metric("Right Arm Angle", f"{metrics['right_arm_angle']:.2f}°")
                        if 'left_arm_angle' in metrics:
                            st.metric("Left Arm Angle", f"{metrics['left_arm_angle']:.2f}°")

                    with cols[1]:
                        st.subheader("Body Symmetry")
                        if 'shoulder_width' in metrics:
                            st.metric("Shoulder Width", f"{metrics['shoulder_width']:.2f} px")
                        if 'torso_alignment' in metrics:
                            st.metric("Torso Alignment", f"{metrics['torso_alignment']:.2f}°")

                elif analysis_mode == "Detailed Metrics":
                    st.header("Comprehensive Body Metrics")

                    # Key Points DataFrame
                    point_names = list(estimator.BODY_PARTS.keys())[:-1]
                    point_coords = [p if p else (np.nan, np.nan) for p in points[:-1]]

                    metrics_df = pd.DataFrame({
                        'Body Part': point_names,
                        'X Coordinate': [p[0] for p in point_coords],
                        'Y Coordinate': [p[1] for p in point_coords]
                    })

                    # Display metrics table
                    st.dataframe(metrics_df)

                    # Body Points Visualization
                    valid_points = [p for p in points[:-1] if p is not None]
                    x_coords = [p[0] for p in valid_points]
                    y_coords = [p[1] for p in valid_points]

                    fig = plt.Figure(data=[plt.Scatter(
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
                    st.subheader("3D Body Points Distribution")
                    
                    valid_points = [p for p in points[:-1] if p is not None]
                    x_coords = [p[0] for p in valid_points]
                    y_coords = [p[1] for p in valid_points]
                    z_coords = np.random.rand(len(valid_points)) * 100  # Placeholder for Z values

                    fig = px.scatter_3d(
                        x=x_coords, y=y_coords, z=z_coords,
                        labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
                        title="3D Pose Visualization"
                    )
                    st.plotly_chart(fig)

                elif analysis_mode == "Real-time Pose Estimation":
                    st.header("Real-time Pose Estimation")
                    st.warning("This feature is available in future versions.")

                # Generate report
                # Replace the existing generate_report call with:
                report_zip = generate_report(image, pose_image, metrics_df)
                st.download_button(label="Download Pose Analysis Report", 
                                   data=report_zip.getvalue(), 
                                   file_name="pose_analysis_report.zip", 
                                   mime="application/zip")


                # Footer
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <p>Developed with ❤️ as part of AICTE Internship on AI: Transformative Learning with Techsaksham, a collaborative initiative by Microsoft & SAP, in partnership with AICTE.  
                    Special thanks to ChatGPT for assisting in developing this website.</p>
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
