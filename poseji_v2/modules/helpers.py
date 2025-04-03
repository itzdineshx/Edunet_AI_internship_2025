import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b given three points a, b, and c.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_biomechanics_mediapipe(points):
    metrics = {}
    # Existing metrics:
    if points[12] and points[14] and points[16]:
        metrics["right_arm_angle"] = calculate_angle(points[12], points[14], points[16])
    if points[11] and points[13] and points[15]:
        metrics["left_arm_angle"] = calculate_angle(points[11], points[13], points[15])
    if points[11] and points[12]:
        metrics["shoulder_width"] = np.linalg.norm(np.array(points[11]) - np.array(points[12]))
    if points[23] and points[24] and points[11] and points[12]:
        neck = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2)
        metrics["torso_alignment"] = calculate_angle(points[23], neck, points[24])
    
    # New: Knee Angles (MediaPipe mapping)
    # Left Knee: left hip (23), left knee (25), left ankle (27)
    if points[23] and points[25] and points[27]:
        metrics["left_knee_angle"] = calculate_angle(points[23], points[25], points[27])
    # Right Knee: right hip (24), right knee (26), right ankle (28)
    if points[24] and points[26] and points[28]:
        metrics["right_knee_angle"] = calculate_angle(points[24], points[26], points[28])
    if "left_knee_angle" in metrics and "right_knee_angle" in metrics:
        metrics["knee_symmetry"] = abs(metrics["left_knee_angle"] - metrics["right_knee_angle"])
    
    # New: Shoulder-to-Hip Ratio
    if points[11] and points[12] and points[23] and points[24]:
        shoulder_width = np.linalg.norm(np.array(points[11]) - np.array(points[12]))
        hip_width = np.linalg.norm(np.array(points[23]) - np.array(points[24]))
        if hip_width > 0:
            metrics["shoulder_hip_ratio"] = shoulder_width / hip_width

    # New: Trunk Lean Angle
    # Mid-shoulder: average of left (11) and right (12) shoulders; Mid-hip: average of left (23) and right (24) hips.
    if points[11] and points[12] and points[23] and points[24]:
        mid_shoulder = ( (points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2 )
        mid_hip = ( (points[23][0] + points[24][0]) / 2, (points[23][1] + points[24][1]) / 2 )
        trunk_vector = np.array(mid_shoulder) - np.array(mid_hip)
        vertical_vector = np.array([0, 1])  # Image y-axis (downwards)
        if np.linalg.norm(trunk_vector) > 0:
            cosine = np.dot(trunk_vector, vertical_vector) / (np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector))
            cosine = np.clip(cosine, -1.0, 1.0)
            trunk_lean_angle = np.degrees(np.arccos(cosine))
            metrics["trunk_lean_angle"] = trunk_lean_angle

    # New: Hip Angles
    # Left hip angle: using left shoulder (11), left hip (23), left knee (25)
    if points[11] and points[23] and points[25]:
        metrics["left_hip_angle"] = calculate_angle(points[11], points[23], points[25])
    # Right hip angle: using right shoulder (12), right hip (24), right knee (26)
    if points[12] and points[24] and points[26]:
        metrics["right_hip_angle"] = calculate_angle(points[12], points[24], points[26])
    if "left_hip_angle" in metrics and "right_hip_angle" in metrics:
        metrics["hip_symmetry"] = abs(metrics["left_hip_angle"] - metrics["right_hip_angle"])

    # New: Overall Balance Score (average of asymmetries; lower is better)
    asymmetries = []
    if "left_arm_angle" in metrics and "right_arm_angle" in metrics:
        asymmetries.append(abs(metrics["left_arm_angle"] - metrics["right_arm_angle"]))
    if "knee_symmetry" in metrics:
        asymmetries.append(metrics["knee_symmetry"])
    if "hip_symmetry" in metrics:
        asymmetries.append(metrics["hip_symmetry"])
    if asymmetries:
        metrics["balance_score"] = sum(asymmetries) / len(asymmetries)
    
    return metrics

def calculate_biomechanics_movenet(points):
    metrics = {}
    # Existing metrics:
    if points[6] and points[8] and points[10]:
        metrics["right_arm_angle"] = calculate_angle(points[6], points[8], points[10])
    if points[5] and points[7] and points[9]:
        metrics["left_arm_angle"] = calculate_angle(points[5], points[7], points[9])
    if points[5] and points[6]:
        metrics["shoulder_width"] = np.linalg.norm(np.array(points[5]) - np.array(points[6]))
    if points[11] and points[12] and points[5] and points[6]:
        neck = ((points[5][0] + points[6][0]) / 2, (points[5][1] + points[6][1]) / 2)
        metrics["torso_alignment"] = calculate_angle(points[11], neck, points[12])
    
    # New: Knee Angles (MoveNet mapping)
    # Left Knee: left hip (11), left knee (13), left ankle (15)
    if points[11] and points[13] and points[15]:
        metrics["left_knee_angle"] = calculate_angle(points[11], points[13], points[15])
    # Right Knee: right hip (12), right knee (14), right ankle (16)
    if points[12] and points[14] and points[16]:
        metrics["right_knee_angle"] = calculate_angle(points[12], points[14], points[16])
    if "left_knee_angle" in metrics and "right_knee_angle" in metrics:
        metrics["knee_symmetry"] = abs(metrics["left_knee_angle"] - metrics["right_knee_angle"])
    
    # New: Shoulder-to-Hip Ratio for MoveNet
    if points[5] and points[6] and points[11] and points[12]:
        shoulder_width = np.linalg.norm(np.array(points[5]) - np.array(points[6]))
        hip_width = np.linalg.norm(np.array(points[11]) - np.array(points[12]))
        if hip_width > 0:
            metrics["shoulder_hip_ratio"] = shoulder_width / hip_width

    # New: Trunk Lean Angle for MoveNet
    if points[5] and points[6] and points[11] and points[12]:
        mid_shoulder = ((points[5][0] + points[6][0]) / 2, (points[5][1] + points[6][1]) / 2)
        mid_hip = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2)
        trunk_vector = np.array(mid_shoulder) - np.array(mid_hip)
        vertical_vector = np.array([0, 1])
        if np.linalg.norm(trunk_vector) > 0:
            cosine = np.dot(trunk_vector, vertical_vector) / (np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector))
            cosine = np.clip(cosine, -1.0, 1.0)
            trunk_lean_angle = np.degrees(np.arccos(cosine))
            metrics["trunk_lean_angle"] = trunk_lean_angle

    # New: Hip Angles for MoveNet
    # Left hip angle: using left shoulder (5), left hip (11), left knee (13)
    if points[5] and points[11] and points[13]:
        metrics["left_hip_angle"] = calculate_angle(points[5], points[11], points[13])
    # Right hip angle: using right shoulder (6), right hip (12), right knee (14)
    if points[6] and points[12] and points[14]:
        metrics["right_hip_angle"] = calculate_angle(points[6], points[12], points[14])
    if "left_hip_angle" in metrics and "right_hip_angle" in metrics:
        metrics["hip_symmetry"] = abs(metrics["left_hip_angle"] - metrics["right_hip_angle"])

    # New: Overall Balance Score for MoveNet
    asymmetries = []
    if "left_arm_angle" in metrics and "right_arm_angle" in metrics:
        asymmetries.append(abs(metrics["left_arm_angle"] - metrics["right_arm_angle"]))
    if "knee_symmetry" in metrics:
        asymmetries.append(metrics["knee_symmetry"])
    if "hip_symmetry" in metrics:
        asymmetries.append(metrics["hip_symmetry"])
    if asymmetries:
        metrics["balance_score"] = sum(asymmetries) / len(asymmetries)
    
    return metrics
