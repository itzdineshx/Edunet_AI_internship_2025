import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from modules import helpers

@st.cache_resource
def get_model_instance(model_choice, model_path, smoothing=False, preprocess=False, smoothing_alpha=0.5):
    if model_choice == "OpenPose":
        return OpenPoseAnalyzer(model_path, smoothing=smoothing, preprocess=preprocess, smoothing_alpha=smoothing_alpha)
    elif model_choice == "MediaPipe":
        return MediaPipePoseAnalyzer(smoothing=smoothing, preprocess=preprocess, smoothing_alpha=smoothing_alpha)
    elif model_choice == "MoveNet":
        return MoveNetAnalyzer(model_path, smoothing=smoothing, preprocess=preprocess, smoothing_alpha=smoothing_alpha)
    else:
        st.error("Unknown model selection.")
        return None

def get_pose_analyzer(model_choice, model_path, smoothing=False, preprocess=False, smoothing_alpha=0.5):
    return get_model_instance(model_choice, model_path, smoothing, preprocess, smoothing_alpha)

class OpenPoseAnalyzer:
    def __init__(self, model_path, smoothing=False, preprocess=False, smoothing_alpha=0.5):
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
        self.smoothing = smoothing
        self.preprocess = preprocess
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_points = None

    def preprocess_frame(self, frame):
        # Enhance contrast using YCrCb histogram equalization.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb = cv2.merge(channels)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def detect_pose(self, frame, threshold=0.2):
        try:
            if self.preprocess:
                frame = self.preprocess_frame(frame)
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
            # Apply smoothing if enabled.
            if self.smoothing:
                if self.smoothed_points is None:
                    self.smoothed_points = points
                else:
                    new_points = []
                    for p, sp in zip(points, self.smoothed_points):
                        if p is None:
                            new_points.append(sp)
                        elif sp is None:
                            new_points.append(p)
                        else:
                            new_x = int(self.smoothing_alpha * p[0] + (1 - self.smoothing_alpha) * sp[0])
                            new_y = int(self.smoothing_alpha * p[1] + (1 - self.smoothing_alpha) * sp[1])
                            new_points.append((new_x, new_y))
                    self.smoothed_points = new_points
                    points = self.smoothed_points
            return points
        except Exception as e:
            st.error(f"Error in detect_pose: {e}")
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
            metrics['right_arm_angle'] = helpers.calculate_angle(points[1], points[2], points[3])
        if points[1] and points[5] and points[6]:
            metrics['left_arm_angle'] = helpers.calculate_angle(points[1], points[5], points[6])
        if points[2] and points[5]:
            metrics['shoulder_width'] = np.linalg.norm(np.array(points[2]) - np.array(points[5]))
        if points[1] and points[8] and points[11]:
            metrics['torso_alignment'] = helpers.calculate_angle(points[8], points[1], points[11])
        return metrics

class MediaPipePoseAnalyzer:
    def __init__(self, smoothing=False, preprocess=False, smoothing_alpha=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.smoothing = smoothing
        self.preprocess = preprocess
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_points = None

    def preprocess_frame(self, frame):
        # Enhance contrast using YCrCb equalization.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb = cv2.merge(channels)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def detect_pose(self, frame, threshold=0.2):
        if self.preprocess:
            frame = self.preprocess_frame(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        points = [None] * 33
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                points[i] = (x, y)
        # Apply smoothing if enabled.
        if self.smoothing:
            if self.smoothed_points is None:
                self.smoothed_points = points
            else:
                new_points = []
                for p, sp in zip(points, self.smoothed_points):
                    if p is None:
                        new_points.append(sp)
                    elif sp is None:
                        new_points.append(p)
                    else:
                        new_x = int(self.smoothing_alpha * p[0] + (1 - self.smoothing_alpha) * sp[0])
                        new_y = int(self.smoothing_alpha * p[1] + (1 - self.smoothing_alpha) * sp[1])
                        new_points.append((new_x, new_y))
                self.smoothed_points = new_points
                points = self.smoothed_points
        return points

    def draw_pose(self, frame, points, threshold=0.2):
        # For MediaPipe, we use the built-in drawing utility.
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
        return helpers.calculate_biomechanics_mediapipe(points)

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
    
    def __init__(self, model_path, smoothing=False, preprocess=False, smoothing_alpha=0.5):
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
        self.smoothing = smoothing
        self.preprocess = preprocess
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_points = None

    def preprocess_frame(self, frame):
        # Enhance contrast using YCrCb equalization.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb = cv2.merge(channels)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def detect_pose(self, frame, threshold=0.2):
        if self.preprocess:
            frame = self.preprocess_frame(frame)
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
        # Apply smoothing if enabled.
        if self.smoothing:
            if self.smoothed_points is None:
                self.smoothed_points = keypoints
            else:
                new_points = []
                for p, sp in zip(keypoints, self.smoothed_points):
                    if p is None:
                        new_points.append(sp)
                    elif sp is None:
                        new_points.append(p)
                    else:
                        new_x = int(self.smoothing_alpha * p[0] + (1 - self.smoothing_alpha) * sp[0])
                        new_y = int(self.smoothing_alpha * p[1] + (1 - self.smoothing_alpha) * sp[1])
                        new_points.append((new_x, new_y))
                self.smoothed_points = new_points
                keypoints = self.smoothed_points
        return keypoints

    def draw_pose(self, frame, points, threshold=0.2):
        if points is None:
            return frame
        for pt in points:
            if pt is not None:
                cv2.circle(frame, pt, 8, (0, 255, 0), -1)
        for connection in self.KEYPOINT_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 < len(points) and idx2 < len(points) and points[idx1] is not None and points[idx2] is not None:
                cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 4)
        return frame

    def calculate_body_metrics(self, points):
        if points is None:
            return {}
        return helpers.calculate_biomechanics_movenet(points)