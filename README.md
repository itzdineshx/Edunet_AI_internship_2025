# PoseJi: Advanced Human Pose Estimation (version 1)🤖

![PoseJi logo](assets/images/poseji-log0.gif)

**PoseJi** is an advanced human pose estimation application designed to provide comprehensive pose analysis and biomechanical insights. Leveraging state‐of‐the‐art machine learning models, the app supports real‑time detection, detailed metrics, and intuitive visualization for applications in sports analysis, fitness tracking, rehabilitation, and more.

---

## Live Demo

Try it now: [Advanced PoseJi App](https://advanced-humanpose-estimation.streamlit.app/)

---

## Features

- **Real-time Pose Detection:**  
  Analyze images and videos using advanced models for fast and accurate pose estimation.

- **Customizable Settings:**  
  Adjust parameters like confidence threshold and calibration factor to suit your needs.

- **Multiple Analysis Modes:**  
  Choose from basic pose detection, biomechanical analysis (joint angles, posture evaluation), detailed metrics, and 3D visualization.

- **Video & Image Processing:**  
  Upload images (PNG, JPG, JPEG) and videos (MP4, AVI, MOV, GIF) for detailed pose analysis. Extract skeleton overlays and download results.

- **Live Webcam Integration:**  
  Access your device’s webcam for live pose estimation, posture feedback, and exercise analysis (e.g., squat counting).

- **Session Management:**  
  Save session outputs (images, videos, metrics) and download all results as a ZIP file.

- **User-Friendly Interface:**  
  Designed with a modern, mobile-responsive layout and an enhanced sidebar for easy navigation.

---

## Project Structure

```plaintext
itzdineshx-edunet_ai_internship_2025/
├── README.md
├── LICENSE
├── requirements.txt
├── Demo/
│   ├── images/
│   │   └── Advanced-Pose-Estimation_sample.png
│   └── videos/
│       └── Demo_video.gif
├── assets/
│   └── images/         # Additional UI assets
├── models/
│   ├── graph_opt.pb
│   ├── movenet_lightning_fp16.tflite
│   └── saved_model.pb
├── src/
│   ├── graph_opt.pb
│   ├── movenet_lightning_fp16.tflite
│   ├── saved_model.pb
│   └── app/
│       ├── advanced_pose_app.py
│       ├── app.py
│       ├── pose_estimation.py
│       ├── pose_estimation_Video.py
│       ├── test.py
│       └── test2.py
└── .devcontainer/
    └── devcontainer.json
```

---

## Model Details

The application supports multiple pose estimation models:

- **OpenPose:**  
  Uses OpenCV’s DNN module to load the `graph_opt.pb` model for detecting keypoints and drawing skeletal connections.

- **MediaPipe Pose:**  
  Utilizes MediaPipe’s Pose solution for efficient, real‑time pose estimation with built‑in drawing utilities for skeleton visualization.

- **MoveNet:**  
  Employs TensorFlow Lite’s MoveNet model (`movenet_lightning_fp16.tflite`) for high-speed, accurate pose detection.

---

## App Interface

### Pose Analysis Settings
- **Confidence Threshold:**  
  Adjust the minimum confidence level for keypoint detection.
- **Calibration Factor:**  
  Scale distance-based metrics for personalized analysis.
- **Analysis Mode:**  
  Select from:
  - Basic Pose Detection
  - Biomechanical Analysis (angles, posture evaluation)
  - Detailed Metrics
  - 3D Pose Visualization
  - Video Pose Estimation
  - Live Webcam Pose Detection
  - Real‑time Posture Feedback
  - Exercise Analysis & Coaching
  - Session History
- **Exercise Analysis:**  
  Choose an exercise type (currently supports Squats) for tracking repetitions.

### Upload Section
- **Image Upload:**  
  Drag and drop PNG, JPG, or JPEG files.
- **Video Upload:**  
  Upload MP4, AVI, MOV, or GIF files for real-time analysis.
- **Live Webcam:**  
  Access live video feed for instant pose estimation.

### Output and Session Management
- **Realtime Metrics:**  
  View live updates on joint angles and other metrics during processing.
- **Download Options:**  
  Download individual outputs (images, videos, metrics) or all session outputs as a ZIP archive.
- **Session History:**  
  Save and review previous sessions with detailed logs and metrics.

---

## Demo

### Demo Video
![Demo Video](Demo/images/Demo_video.gif)

### Sample Image
![Pose Estimation Sample image](Demo/images/Advanced-Pose-Estimation_sample.png)

### Sample Video
![Pose Estimation Sample Video](assets/images/pose-gif.gif)

---

## How to Run the App

### Prerequisites
- Python 3.8 or higher
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
1. Navigate to the `src` directory.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in your browser to interact with the app.

---

## Use Cases

- **Sports Training:**  
  Analyze and optimize athletes' movements.

- **Fitness Tracking:**  
  Monitor posture and exercise form to improve workouts.

- **Rehabilitation:**  
  Track progress in physical therapy and recovery.

- **Gaming & AR/VR:**  
  Integrate pose estimation into interactive applications.

---

## Future Enhancements

- **Advanced Real-Time Feedback:**  
  More in-depth posture analysis and exercise coaching.
- **Multi-Person Detection:**  
  Extend support for multiple people in a single frame.
- **Mobile Optimization:**  
  Further enhance the mobile user experience.
- **Integration APIs:**  
  Offer REST APIs for third-party integration.

---

## Contributors

This project is part of the Edunet AI Internship 2025 program. Contributions are welcome to further enhance functionality and extend features.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

---

**PoseJi © 2025 DINESH S All Rights Reserved**
