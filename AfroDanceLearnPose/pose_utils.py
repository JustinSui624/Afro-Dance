import math
import numpy as np


# ---------- Basic math helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point b formed by points a-b-c (in degrees).
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = clamp(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


# ---------- Pose processing ----------
def landmarks_to_xy(landmarks, image_w: int, image_h: int):
    """
    Convert MediaPipe normalized landmarks to pixel coordinates (x,y) and visibility/confidence.
    Returns:
      xy: (N,2) float
      vis: (N,) float
    """
    xy = []
    vis = []
    for lm in landmarks:
        xy.append([lm.x * image_w, lm.y * image_h])
        # mediapipe pose uses 'visibility' for many landmarks
        vis.append(getattr(lm, "visibility", 1.0))
    return np.array(xy, dtype=np.float32), np.array(vis, dtype=np.float32)


def normalize_xy(xy: np.ndarray, left_hip_idx: int, right_hip_idx: int,
                 left_shoulder_idx: int, right_shoulder_idx: int):
    """
    Center on hip center and scale by shoulder width.
    Returns normalized xy (N,2).
    """
    left_hip = xy[left_hip_idx]
    right_hip = xy[right_hip_idx]
    hip_center = (left_hip + right_hip) / 2.0

    left_sh = xy[left_shoulder_idx]
    right_sh = xy[right_shoulder_idx]
    shoulder_width = dist(left_sh, right_sh)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    norm = (xy - hip_center) / shoulder_width
    return norm


def compute_key_angles(norm_xy: np.ndarray, mp_idx: dict):
    """
    Compute a compact set of joint angles from normalized xy.
    Returns dict: angle_name -> degrees
    """
    # convenient alias
    I = mp_idx

    def ang(p1, p2, p3):
        return angle_deg(norm_xy[I[p1]], norm_xy[I[p2]], norm_xy[I[p3]])

    angles = {
        # elbows
        "left_elbow":  ang("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "right_elbow": ang("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        # knees
        "left_knee":   ang("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right_knee":  ang("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        # shoulders (arm raise / posture-ish)
        "left_shoulder":  ang("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
        "right_shoulder": ang("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
        # hips (leg lift / posture-ish)
        "left_hip":    ang("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
        "right_hip":   ang("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    }
    return angles


def angles_to_vector(angle_dict: dict, angle_names: list):
    """
    Convert dict of angles into vector in a consistent order.
    NaNs become 0.0 (you can choose a different policy).
    """
    v = []
    for name in angle_names:
        x = angle_dict.get(name, float("nan"))
        if np.isnan(x):
            x = 0.0
        v.append(float(x))
    return np.array(v, dtype=np.float32)


# ---------- Scoring ----------
def mean_abs_error_deg(user_vec: np.ndarray, ref_vec: np.ndarray) -> float:
    return float(np.mean(np.abs(user_vec - ref_vec)))


def error_to_score_0_100(err_deg: float, k: float = 1.2) -> float:
    """
    Simple mapping: lower error => higher score
    score = 100 - k*err
    """
    return clamp(100.0 - k * err_deg, 0.0, 100.0)


class MovingAverage:
    def __init__(self, window: int = 10):
        self.window = max(1, int(window))
        self.buf = []

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return float(sum(self.buf) / len(self.buf))