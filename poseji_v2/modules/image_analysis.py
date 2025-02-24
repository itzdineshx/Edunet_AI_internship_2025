import cv2
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
import streamlit as st
from modules import helpers

def run_image_analysis(analyzer, image, threshold_val, model_choice, analysis_mode):
    # Ensure image is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Detect pose keypoints
    points = analyzer.detect_pose(image, threshold_val)
    if points is None:
        st.error("Pose detection failed.")
        return None, None, None
    
    # Generate the pose overlay image
    pose_image = image.copy()
    pose_image = analyzer.draw_pose(pose_image, points, threshold_val)
    
    # Display original and pose images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Pose Estimation")
        st.image(pose_image, use_container_width=True)
    
    # Determine body part names based on model choice
    if model_choice == "MediaPipe":
        body_parts = {
            0: "Nose",
            1: "Left Eye Inner",
            2: "Left Eye",
            3: "Left Eye Outer",
            4: "Right Eye Inner",
            5: "Right Eye",
            6: "Right Eye Outer",
            7: "Left Ear",
            8: "Right Ear",
            9: "Mouth Left",
            10: "Mouth Right",
            11: "Left Shoulder",
            12: "Right Shoulder",
            13: "Left Elbow",
            14: "Right Elbow",
            15: "Left Wrist",
            16: "Right Wrist",
            17: "Left Pinky",
            18: "Right Pinky",
            19: "Left Index",
            20: "Right Index",
            21: "Left Thumb",
            22: "Right Thumb",
            23: "Left Hip",
            24: "Right Hip",
            25: "Left Knee",
            26: "Right Knee",
            27: "Left Ankle",
            28: "Right Ankle",
            29: "Left Heel",
            30: "Right Heel",
            31: "Left Foot Index",
            32: "Right Foot Index"
        }
    elif model_choice == "MoveNet":
        body_parts = {
            0: "Nose",
            1: "Left Eye",
            2: "Right Eye",
            3: "Left Ear",
            4: "Right Ear",
            5: "Left Shoulder",
            6: "Right Shoulder",
            7: "Left Elbow",
            8: "Right Elbow",
            9: "Left Wrist",
            10: "Right Wrist",
            11: "Left Hip",
            12: "Right Hip",
            13: "Left Knee",
            14: "Right Knee",
            15: "Left Ankle",
            16: "Right Ankle"
        }
    elif model_choice == "OpenPose":
        # For OpenPose, use the inverse of the analyzer's BODY_PARTS dictionary
        body_parts = {v: k for k, v in analyzer.BODY_PARTS.items()}
    else:
        # Fallback: use generic naming
        body_parts = {i: f"Point {i}" for i in range(len(points))}
    
    # Skeleton extraction option
    skeleton_bytes = None
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
    metrics = {}
    
    # --- Analysis Modes ---
    if analysis_mode == "Biomechanical Analysis":
        st.header("Biomechanical Insights")
        # Calculate metrics based on the chosen model
        if model_choice == "MediaPipe":
            metrics = helpers.calculate_biomechanics_mediapipe(points)
        elif model_choice == "MoveNet":
            metrics = helpers.calculate_biomechanics_movenet(points)
        else:
            metrics = analyzer.calculate_body_metrics(points)
        
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Joint Angles")
            for key, value in metrics.items():
                st.metric(key, f"{value:.2f}")
        with colB:
            # Bar chart visualization for joint angles
            angle_keys = list(metrics.keys())
            angle_values = [metrics[k] for k in angle_keys if isinstance(metrics[k], (int, float))]
            if angle_values:
                bar_fig = go.Figure([go.Bar(x=angle_keys, y=angle_values, marker_color='indianred')])
                bar_fig.update_layout(title="Joint Angles Comparison", xaxis_title="Joint", yaxis_title="Angle (°)")
                st.plotly_chart(bar_fig, use_container_width=True)
        
        if "left_arm_angle" in metrics and "right_arm_angle" in metrics:
            symmetry = abs(metrics["left_arm_angle"] - metrics["right_arm_angle"])
            st.info(f"Arm Symmetry Score (lower is better): {symmetry:.2f}")
    
    elif analysis_mode == "Detailed Metrics":
        st.header("Comprehensive Body Metrics")
        # Build a DataFrame with actual body part names
        names = [body_parts.get(i, f"Point {i}") for i in range(len(points))]
        point_coords = [p if p is not None else (float('nan'), float('nan')) for p in points]
        metrics_df = pd.DataFrame({
            'Body Part': names,
            'X Coordinate': [p[0] for p in point_coords],
            'Y Coordinate': [p[1] for p in point_coords]
        })
        st.dataframe(metrics_df)
        
        # Scatter plot for spatial distribution
        valid = [(names[i], p) for i, p in enumerate(points) if p is not None]
        if valid:
            valid_names, valid_points = zip(*valid)
            x_coords = [p[0] for p in valid_points]
            y_coords = [p[1] for p in valid_points]
            scatter_fig = go.Figure(data=[go.Scatter(
                x=x_coords, y=y_coords, 
                mode='markers+text', 
                marker=dict(size=10, color='red'),
                text=valid_names,
                textposition="bottom center"
            )])
            scatter_fig.update_layout(
                title="Body Points Spatial Distribution",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                height=400
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Extra: Pie chart showing ratio of detected vs missing points
            total_points = len(points)
            detected = sum(1 for p in points if p is not None)
            missing = total_points - detected
            pie_fig = px.pie(
                names=["Detected", "Missing"],
                values=[detected, missing],
                title="Keypoint Detection Ratio"
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        
        # Improved Radar Chart for joint angles
        selected_angles = {}
        if model_choice == "MediaPipe":
            selected_angles = helpers.calculate_biomechanics_mediapipe(points)
        elif model_choice == "MoveNet":
            selected_angles = helpers.calculate_biomechanics_movenet(points)
        else:
            selected_angles = analyzer.calculate_body_metrics(points)
        if selected_angles:
            categories = list(selected_angles.keys())
            values = [selected_angles[k] for k in categories]
            # Duplicate first value to close the radar chart loop.
            values += values[:1]
            categories += categories[:1]
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Joint Angles'
            ))
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.1]
                    )
                ),
                showlegend=False,
                title="Joint Angles Radar Chart"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
    
    elif analysis_mode == "3D Pose Visualization":
        st.header("3D Pose Visualization")
        valid_points = [p for p in points if p is not None]
        if valid_points:
            x_coords = [p[0] for p in valid_points]
            y_coords = [p[1] for p in valid_points]
            z_coords = np.random.rand(len(valid_points)) * 100  # Simulated Z values
            fig3d = px.scatter_3d(
                x=x_coords, y=y_coords, z=z_coords,
                labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
                title="3D Pose Visualization"
            )
            st.plotly_chart(fig3d)
        else:
            st.error("No valid points detected for 3D visualization.")
    else:
        st.header("Basic Pose Detection")
    
    # Optionally, provide a download for a complete analysis report (ZIP file)
    report_btn = st.button("Download Analysis Report")
    if report_btn:
        # Create a CSV report of detailed metrics if available
        if metrics_df.empty and points:
            names = [body_parts.get(i, f"Point {i}") for i in range(len(points))]
            point_coords = [p if p is not None else (float('nan'), float('nan')) for p in points]
            metrics_df = pd.DataFrame({
                'Body Part': names,
                'X Coordinate': [p[0] for p in point_coords],
                'Y Coordinate': [p[1] for p in point_coords]
            })
        from modules.video_estimation import generate_report  # Reuse your report generator
        report_zip = generate_report(image, pose_image, metrics_df)
        st.download_button("Download Report ZIP", report_zip.getvalue(), "pose_analysis_report.zip", "application/zip")
    
    # Return the pose overlay, detailed metrics (if any), and final calculated metrics
    final_metrics = analyzer.calculate_body_metrics(points)
    return pose_image, metrics_df, final_metrics
