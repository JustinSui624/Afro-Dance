import json
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

try:
    from .pose_utils import (
        landmarks_to_xy,
        normalize_xy,
        compute_key_angles,
        angles_to_vector,
        mean_abs_error_deg,
        error_to_score_0_100,
        MovingAverage,
        dist,
    )
except ImportError:
    from pose_utils import (
        landmarks_to_xy,
        normalize_xy,
        compute_key_angles,
        angles_to_vector,
        mean_abs_error_deg,
        error_to_score_0_100,
        MovingAverage,
        dist,
    )

mp_pose = mp.solutions.pose
MP_LM = mp_pose.PoseLandmark
MP_IDX = {name: lm.value for name, lm in MP_LM.__members__.items()}
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Dashboard-inspired palette (OpenCV uses BGR)
BG_MAIN = (230, 220, 205)
BG_PANEL = (223, 211, 193)
BROWN_DARK = (20, 40, 70)
BROWN_MED = (42, 59, 92)
BROWN_HEADER = (33, 47, 75)
TEXT_LIGHT = (245, 230, 210)
TEXT_DARK = (20, 28, 45)
BLACKISH = (20, 20, 20)
WHITE = (255, 255, 255)
LIGHT_GRAY = (205, 205, 205)

# Reference overlay colors
REF_CYAN = (255, 255, 0)
REF_CYAN_DARK = (170, 170, 0)
REFERENCE_PANEL_COLOR = REF_CYAN


def resolve_repo_root():
    return Path(__file__).resolve().parents[1]


def load_reference(repo_root: Path):
    ref_path = repo_root / "data" / "references" / "instructor_reference.json"
    if not ref_path.exists():
        raise RuntimeError(f"Missing reference file: {ref_path}. Run: python extract_reference.py")

    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    vectors = np.array(ref["vectors"], dtype=np.float32)
    ref_norm_xy = np.array(ref["ref_norm_xy"], dtype=np.float32)
    segments = ref.get("segments", [{"start": 0, "end": len(vectors) - 1}])

    return (
        vectors,
        ref["angle_names"],
        float(ref["fps"]),
        ref_norm_xy,
        segments,
    )


def fit_to_window(frame, win_w, win_h):
    h, w = frame.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((win_h, win_w, 3), BG_PANEL, dtype=np.uint8)
    x0 = (win_w - new_w) // 2
    y0 = (win_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def get_user_alignment_metrics(xy_px):
    left_hip = xy_px[MP_IDX["LEFT_HIP"]]
    right_hip = xy_px[MP_IDX["RIGHT_HIP"]]
    left_shoulder = xy_px[MP_IDX["LEFT_SHOULDER"]]
    right_shoulder = xy_px[MP_IDX["RIGHT_SHOULDER"]]

    hip_center = 0.5 * (left_hip + right_hip)
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)

    shoulder_width = max(np.linalg.norm(left_shoulder - right_shoulder), 1e-3)
    hip_width = max(np.linalg.norm(left_hip - right_hip), 1e-3)
    torso_height = max(np.linalg.norm(shoulder_center - hip_center), 1e-3)

    left_arm = (
        np.linalg.norm(xy_px[MP_IDX["LEFT_SHOULDER"]] - xy_px[MP_IDX["LEFT_ELBOW"]]) +
        np.linalg.norm(xy_px[MP_IDX["LEFT_ELBOW"]] - xy_px[MP_IDX["LEFT_WRIST"]])
    )
    right_arm = (
        np.linalg.norm(xy_px[MP_IDX["RIGHT_SHOULDER"]] - xy_px[MP_IDX["RIGHT_ELBOW"]]) +
        np.linalg.norm(xy_px[MP_IDX["RIGHT_ELBOW"]] - xy_px[MP_IDX["RIGHT_WRIST"]])
    )
    avg_arm = max((left_arm + right_arm) / 2.0, 1e-3)

    left_leg = (
        np.linalg.norm(xy_px[MP_IDX["LEFT_HIP"]] - xy_px[MP_IDX["LEFT_KNEE"]]) +
        np.linalg.norm(xy_px[MP_IDX["LEFT_KNEE"]] - xy_px[MP_IDX["LEFT_ANKLE"]])
    )
    right_leg = (
        np.linalg.norm(xy_px[MP_IDX["RIGHT_HIP"]] - xy_px[MP_IDX["RIGHT_KNEE"]]) +
        np.linalg.norm(xy_px[MP_IDX["RIGHT_KNEE"]] - xy_px[MP_IDX["RIGHT_ANKLE"]])
    )
    avg_leg = max((left_leg + right_leg) / 2.0, 1e-3)

    return {
        "hip_center": hip_center,
        "shoulder_center": shoulder_center,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_height": torso_height,
        "avg_arm": avg_arm,
        "avg_leg": avg_leg,
    }


def fallback_ref_to_user_space(ref_norm_frame, xy_px):
    metrics = get_user_alignment_metrics(xy_px)

    hip_center = metrics["hip_center"]
    shoulder_center = metrics["shoulder_center"]

    x_scale = 0.60 * metrics["shoulder_width"] + 0.40 * metrics["hip_width"]
    y_scale = (
        0.35 * (metrics["torso_height"] / 0.95) +
        0.25 * (metrics["avg_arm"] / 1.25) +
        0.40 * (metrics["avg_leg"] / 1.65)
    )

    ref_xy = np.array(ref_norm_frame, dtype=np.float32).copy()
    ref_xy[:, 0] = hip_center[0] + ref_xy[:, 0] * x_scale
    ref_xy[:, 1] = hip_center[1] + ref_xy[:, 1] * y_scale

    ref_left_sh = ref_xy[MP_IDX["LEFT_SHOULDER"]]
    ref_right_sh = ref_xy[MP_IDX["RIGHT_SHOULDER"]]
    ref_shoulder_center = 0.5 * (ref_left_sh + ref_right_sh)

    shift = shoulder_center - ref_shoulder_center
    ref_xy = ref_xy + shift

    return ref_xy


def ref_norm_to_user_space(ref_norm_frame, xy_px):
    ref_norm_frame = np.array(ref_norm_frame, dtype=np.float32)
    user_xy = np.array(xy_px, dtype=np.float32)

    anchor_indices = [
        MP_IDX["NOSE"],
        MP_IDX["LEFT_SHOULDER"], MP_IDX["RIGHT_SHOULDER"],
        MP_IDX["LEFT_ELBOW"], MP_IDX["RIGHT_ELBOW"],
        MP_IDX["LEFT_WRIST"], MP_IDX["RIGHT_WRIST"],
        MP_IDX["LEFT_HIP"], MP_IDX["RIGHT_HIP"],
        MP_IDX["LEFT_KNEE"], MP_IDX["RIGHT_KNEE"],
        MP_IDX["LEFT_ANKLE"], MP_IDX["RIGHT_ANKLE"],
    ]

    src_pts = []
    dst_pts = []

    for idx in anchor_indices:
        src = ref_norm_frame[idx]
        dst = user_xy[idx]

        if np.all(src == 0) or np.all(dst == 0):
            continue

        src_pts.append(src)
        dst_pts.append(dst)

    if len(src_pts) >= 4:
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        affine, _ = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=18.0,
            maxIters=2000,
            confidence=0.99,
        )

        if affine is not None:
            ones = np.ones((len(ref_norm_frame), 1), dtype=np.float32)
            pts_h = np.hstack([ref_norm_frame, ones])
            transformed = (affine @ pts_h.T).T
            return transformed.astype(np.float32)

    return fallback_ref_to_user_space(ref_norm_frame, xy_px)


def get_joint_feedback_color(distance_px, threshold_px):
    if distance_px <= threshold_px * 0.35:
        return (0, 210, 0)
    if distance_px <= threshold_px * 0.70:
        return (0, 220, 220)
    return (0, 140, 255)


def get_score_band(score):
    if score >= 85:
        return "Excellent", (0, 210, 0), "Great match. Keep that rhythm and posture."
    if score >= 70:
        return "Good", (0, 220, 220), "Good alignment. Focus on timing and shoulder position."
    if score >= 55:
        return "Okay", (0, 170, 255), "Close. Refine posture and follow the reference shape."
    return "Needs Work", (0, 0, 255), "Stay closer to the instructor overlay and slow down if needed."


def generate_actionable_tip(avg_score):
    if avg_score >= 85:
        return "Strong session overall. Keep practicing timing consistency."
    if avg_score >= 70:
        return "Good progress. Focus on smoother transitions between steps."
    if avg_score >= 55:
        return "You are close. Try matching shoulder level and body timing more carefully."
    return "Use the instructor overlay as your guide and practice each step more slowly."


def draw_reference_overlay_on_user(frame, ref_xy, alpha=0.34):
    overlay = frame.copy()

    for a, b in POSE_CONNECTIONS:
        ax, ay = ref_xy[a]
        bx, by = ref_xy[b]
        if (ax == 0 and ay == 0) or (bx == 0 and by == 0):
            continue

        pt1 = (int(ax), int(ay))
        pt2 = (int(bx), int(by))

        cv2.line(overlay, pt1, pt2, WHITE, 5, cv2.LINE_AA)
        cv2.line(overlay, pt1, pt2, REF_CYAN, 2, cv2.LINE_AA)

    key_joints = [
        MP_IDX["NOSE"],
        MP_IDX["LEFT_EYE"], MP_IDX["RIGHT_EYE"],
        MP_IDX["LEFT_EAR"], MP_IDX["RIGHT_EAR"],
        MP_IDX["LEFT_SHOULDER"], MP_IDX["RIGHT_SHOULDER"],
        MP_IDX["LEFT_ELBOW"], MP_IDX["RIGHT_ELBOW"],
        MP_IDX["LEFT_WRIST"], MP_IDX["RIGHT_WRIST"],
        MP_IDX["LEFT_HIP"], MP_IDX["RIGHT_HIP"],
        MP_IDX["LEFT_KNEE"], MP_IDX["RIGHT_KNEE"],
        MP_IDX["LEFT_ANKLE"], MP_IDX["RIGHT_ANKLE"],
    ]

    for idx in key_joints:
        x, y = ref_xy[idx]
        if x != 0 or y != 0:
            center = (int(x), int(y))
            cv2.circle(overlay, center, 6, WHITE, -1)
            cv2.circle(overlay, center, 3, REF_CYAN, -1)

    nose = ref_xy[MP_IDX["NOSE"]]
    left_sh = ref_xy[MP_IDX["LEFT_SHOULDER"]]
    right_sh = ref_xy[MP_IDX["RIGHT_SHOULDER"]]
    shoulder_center = 0.5 * (left_sh + right_sh)
    head_radius = int(max(np.linalg.norm(nose - shoulder_center) * 0.65, 10))
    if not np.all(nose == 0):
        center = (int(nose[0]), int(nose[1] + head_radius // 3))
        cv2.circle(overlay, center, head_radius + 2, WHITE, 2)
        cv2.circle(overlay, center, head_radius, REF_CYAN_DARK, 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_user_feedback_skeleton(frame, user_xy, ref_xy):
    h, w = frame.shape[:2]
    threshold_px = max(min(w, h) * 0.08, 28)

    joint_colors = {}
    for idx in range(len(user_xy)):
        if np.all(user_xy[idx] == 0) or np.all(ref_xy[idx] == 0):
            continue
        d = dist(user_xy[idx], ref_xy[idx])
        joint_colors[idx] = get_joint_feedback_color(d, threshold_px)

    for a, b in POSE_CONNECTIONS:
        if np.all(user_xy[a] == 0) or np.all(user_xy[b] == 0):
            continue

        color_a = joint_colors.get(a, WHITE)
        color_b = joint_colors.get(b, WHITE)

        if color_a == (0, 140, 255) or color_b == (0, 140, 255):
            line_color = (0, 140, 255)
        elif color_a == (0, 220, 220) or color_b == (0, 220, 220):
            line_color = (0, 220, 220)
        else:
            line_color = (0, 210, 0)

        pt1 = tuple(user_xy[a].astype(int))
        pt2 = tuple(user_xy[b].astype(int))
        cv2.line(frame, pt1, pt2, line_color, 4, cv2.LINE_AA)

    for idx, pt in enumerate(user_xy):
        if np.all(pt == 0):
            continue
        color = joint_colors.get(idx, WHITE)
        x, y = pt.astype(int)
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 8, WHITE, 1)


def transform_reference_for_panel(ref_norm_frame, panel_w, panel_h):
    points = np.array(ref_norm_frame, dtype=np.float32)
    valid = ~(np.all(points == 0, axis=1))

    transformed = {}
    if not np.any(valid):
        return transformed

    valid_points = points[valid]
    min_x = np.min(valid_points[:, 0])
    max_x = np.max(valid_points[:, 0])
    min_y = np.min(valid_points[:, 1])
    max_y = np.max(valid_points[:, 1])

    width = max(max_x - min_x, 1e-3)
    height = max(max_y - min_y, 1e-3)

    draw_left = 40
    draw_right = panel_w - 40
    draw_top = 170
    draw_bottom = panel_h - 110

    draw_w = draw_right - draw_left
    draw_h = draw_bottom - draw_top

    scale = min(draw_w / width, draw_h / height) * 0.82

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    screen_cx = draw_left + draw_w // 2
    screen_cy = draw_top + draw_h // 2

    for idx, (x, y) in enumerate(points):
        sx = int((x - center_x) * scale + screen_cx)
        sy = int((y - center_y) * scale + screen_cy)
        transformed[idx] = (sx, sy)

    return transformed


def render_reference_panel(panel_w, panel_h, ref_norm_frame, score, step_idx, total_steps, ref_idx, step_start, step_end, play_sequence):
    panel = np.full((panel_h, panel_w, 3), BG_PANEL, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), BROWN_MED, 3)
    cv2.rectangle(panel, (0, 0), (panel_w, 56), BROWN_HEADER, -1)

    band_text, band_color, hint = get_score_band(score)
    mode = "SEQUENCE" if play_sequence else "STEP"

    cv2.putText(panel, "REFERENCE", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, TEXT_LIGHT, 2)

    content_x = 16
    content_y = 82
    content_w = panel_w - 32
    content_h = panel_h - 100

    cv2.rectangle(panel, (content_x, content_y), (content_x + content_w, content_y + content_h), BLACKISH, -1)
    cv2.rectangle(panel, (content_x, content_y), (content_x + content_w, content_y + content_h), BROWN_MED, 2)

    cv2.putText(panel, f"{mode} | Step {step_idx + 1}/{total_steps}", (28, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_LIGHT, 2)
    cv2.putText(panel, f"Score: {score:.0f}", (28, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.85, band_color, 2)
    cv2.putText(panel, band_text, (190, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.80, band_color, 2)
    cv2.putText(panel, hint, (28, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.52, LIGHT_GRAY, 1)

    transformed = transform_reference_for_panel(ref_norm_frame, panel_w, panel_h)

    for a, b in POSE_CONNECTIONS:
        if a in transformed and b in transformed:
            pt1 = transformed[a]
            pt2 = transformed[b]
            cv2.line(panel, pt1, pt2, WHITE, 5, cv2.LINE_AA)
            cv2.line(panel, pt1, pt2, REFERENCE_PANEL_COLOR, 2, cv2.LINE_AA)

    focus_joints = [
        MP_IDX["NOSE"],
        MP_IDX["LEFT_SHOULDER"], MP_IDX["RIGHT_SHOULDER"],
        MP_IDX["LEFT_ELBOW"], MP_IDX["RIGHT_ELBOW"],
        MP_IDX["LEFT_WRIST"], MP_IDX["RIGHT_WRIST"],
        MP_IDX["LEFT_HIP"], MP_IDX["RIGHT_HIP"],
        MP_IDX["LEFT_KNEE"], MP_IDX["RIGHT_KNEE"],
        MP_IDX["LEFT_ANKLE"], MP_IDX["RIGHT_ANKLE"],
    ]
    for idx in focus_joints:
        if idx in transformed:
            center = transformed[idx]
            cv2.circle(panel, center, 7, WHITE, -1)
            cv2.circle(panel, center, 3, REFERENCE_PANEL_COLOR, -1)

    if MP_IDX["NOSE"] in transformed and MP_IDX["LEFT_SHOULDER"] in transformed and MP_IDX["RIGHT_SHOULDER"] in transformed:
        nose = np.array(transformed[MP_IDX["NOSE"]], dtype=np.float32)
        ls = np.array(transformed[MP_IDX["LEFT_SHOULDER"]], dtype=np.float32)
        rs = np.array(transformed[MP_IDX["RIGHT_SHOULDER"]], dtype=np.float32)
        shoulder_center = 0.5 * (ls + rs)
        head_radius = int(max(np.linalg.norm(nose - shoulder_center) * 0.65, 10))
        center = (int(nose[0]), int(nose[1] + head_radius // 3))
        cv2.circle(panel, center, head_radius + 2, WHITE, 2)
        cv2.circle(panel, center, head_radius, REF_CYAN_DARK, 2)

    prog = (ref_idx - step_start) / max(1, step_end - step_start)
    bar_x = 28
    bar_y = panel_h - 74
    bar_w = panel_w - 56

    cv2.putText(panel, f"Frame {ref_idx}", (28, panel_h - 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LIGHT_GRAY, 1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (100, 100, 100), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_w * prog), bar_y + 14), (240, 240, 240), -1)

    return panel


def build_side_by_side_frame(user_frame, ref_panel, score, err_deg, step_idx, total_steps):
    user_h, user_w = user_frame.shape[:2]
    ref_h, ref_w = ref_panel.shape[:2]

    canvas_h = max(user_h, ref_h) + 110
    canvas_w = user_w + ref_w + 34

    canvas = np.full((canvas_h, canvas_w, 3), BG_MAIN, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, 92), BROWN_HEADER, -1)
    cv2.line(canvas, (0, 92), (canvas_w, 92), BROWN_DARK, 3)

    band_text, band_color, hint = get_score_band(score)

    cv2.putText(canvas, "LIVE TRAINING MODE", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, TEXT_LIGHT, 2)
    cv2.putText(canvas, f"Step {step_idx + 1}/{total_steps}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.78, TEXT_LIGHT, 2)
    cv2.putText(canvas, f"Score: {score:.0f}", (210, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.82, band_color, 2)
    cv2.putText(canvas, f"Error: {err_deg:.1f} deg", (388, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_LIGHT, 2)
    cv2.putText(canvas, hint, (650, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.57, TEXT_LIGHT, 1)

    top_y = 102
    left_x = 12
    right_x = user_w + 24

    canvas[top_y:top_y + user_h, left_x:left_x + user_w] = user_frame
    canvas[top_y:top_y + ref_h, right_x:right_x + ref_w] = ref_panel

    cv2.rectangle(canvas, (left_x - 2, top_y - 2), (left_x + user_w + 1, top_y + user_h + 1), BROWN_MED, 3)
    cv2.rectangle(canvas, (right_x - 2, top_y - 2), (right_x + ref_w + 1, top_y + ref_h + 1), BROWN_MED, 3)

    cv2.putText(canvas, "YOU", (left_x + 18, top_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.92, TEXT_LIGHT, 2)

    # Footer band for better controls readability
    cv2.rectangle(canvas, (0, canvas_h - 34), (canvas_w, canvas_h), BG_PANEL, -1)
    cv2.line(canvas, (0, canvas_h - 34), (canvas_w, canvas_h - 34), BROWN_MED, 2)

    controls = "Controls: S sequence | [ ] step | SPACE restart | P pause | M fullscreen | Q quit"
    cv2.putText(canvas, controls, (16, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_DARK, 1)

    return canvas


def show_session_summary(score_history, step_score_history):
    if not score_history:
        return

    avg_score = float(np.mean(score_history))

    step_averages = {}
    for step, values in step_score_history.items():
        if values:
            step_averages[step] = float(np.mean(values))

    if step_averages:
        best_step = max(step_averages, key=step_averages.get)
        weakest_step = min(step_averages, key=step_averages.get)
        best_score = step_averages[best_step]
        weakest_score = step_averages[weakest_step]
    else:
        best_step = weakest_step = 0
        best_score = weakest_score = avg_score

    suggestion = generate_actionable_tip(avg_score)

    print("SESSION SUMMARY")
    print(f"Average score: {avg_score:.1f}")
    print(f"Best step: {best_step + 1} ({best_score:.1f})")
    print(f"Weakest step: {weakest_step + 1} ({weakest_score:.1f})")
    print(f"Suggestion: {suggestion}")

    w, h = 900, 520
    panel = np.full((h, w, 3), BG_MAIN, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (w, 72), BROWN_HEADER, -1)
    cv2.putText(panel, "SESSION SUMMARY", (26, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_LIGHT, 2)

    cv2.rectangle(panel, (30, 110), (w - 30, h - 40), BG_PANEL, -1)
    cv2.rectangle(panel, (30, 110), (w - 30, h - 40), BROWN_MED, 3)

    lines = [
        f"Average Score: {avg_score:.1f}",
        f"Best Step: {best_step + 1}  ({best_score:.1f})",
        f"Weakest Step: {weakest_step + 1}  ({weakest_score:.1f})",
        "",
        "Suggestion:",
        suggestion,
        "",
        "Press any key to close."
    ]

    y = 170
    for line in lines:
        cv2.putText(
            panel,
            line,
            (60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82 if line != suggestion else 0.68,
            TEXT_DARK,
            2 if line in {"Average Score: {:.1f}".format(avg_score), f"Best Step: {best_step + 1}  ({best_score:.1f})", f"Weakest Step: {weakest_step + 1}  ({weakest_score:.1f})"} else 1
        )
        y += 46

    win = "AfroDance Learn - Session Summary"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 520)

    while True:
        cv2.imshow(win, panel)
        key = cv2.waitKey(30)
        if key != -1:
            break

    cv2.destroyWindow(win)


def main():
    repo_root = resolve_repo_root()
    ref_vectors, angle_names, ref_fps, ref_norm_xy, segments = load_reference(repo_root)

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    play_sequence = False
    paused = False
    fullscreen = False

    step_idx = 0
    step_start = segments[0]["start"]
    step_end = segments[0]["end"]
    step_t0 = time.time()
    ref_idx = step_start

    ma_score = MovingAverage(10)
    ma_err = MovingAverage(10)
    smoothed_ref_xy = None

    score_history = []
    step_score_history = {i: [] for i in range(len(segments))}

    window = "AfroDance Learn - Live Training"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1480, 840)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not paused:
            ref_idx = step_start + int((time.time() - step_t0) * ref_fps)
            if play_sequence and ref_idx > step_end:
                step_idx = (step_idx + 1) % len(segments)
                step_start = segments[step_idx]["start"]
                step_end = segments[step_idx]["end"]
                step_t0 = time.time()
                ref_idx = step_start
            ref_idx = max(step_start, min(step_end, ref_idx))

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        score_sm, err_sm = 0.0, 999.0
        user_view = frame.copy()

        if res.pose_landmarks:
            user_xy, _ = landmarks_to_xy(res.pose_landmarks.landmark, w, h)
            ref_xy_user_space = ref_norm_to_user_space(ref_norm_xy[ref_idx], user_xy)

            if smoothed_ref_xy is None:
                smoothed_ref_xy = ref_xy_user_space.copy()
            else:
                smoothed_ref_xy = 0.70 * smoothed_ref_xy + 0.30 * ref_xy_user_space

            norm_user = normalize_xy(
                user_xy,
                MP_IDX["LEFT_HIP"],
                MP_IDX["RIGHT_HIP"],
                MP_IDX["LEFT_SHOULDER"],
                MP_IDX["RIGHT_SHOULDER"],
            )

            user_vec = angles_to_vector(
                compute_key_angles(norm_user, MP_IDX), angle_names
            )

            err = mean_abs_error_deg(user_vec, ref_vectors[ref_idx])
            score = error_to_score_0_100(err)
            err_sm = ma_err.update(err)
            score_sm = ma_score.update(score)

            score_history.append(score_sm)
            step_score_history[step_idx].append(score_sm)

            draw_reference_overlay_on_user(user_view, smoothed_ref_xy, alpha=0.34)
            draw_user_feedback_skeleton(user_view, user_xy, smoothed_ref_xy)

        ref_panel = render_reference_panel(
            panel_w=430,
            panel_h=h,
            ref_norm_frame=ref_norm_xy[ref_idx],
            score=score_sm,
            step_idx=step_idx,
            total_steps=len(segments),
            ref_idx=ref_idx,
            step_start=step_start,
            step_end=step_end,
            play_sequence=play_sequence,
        )

        combined = build_side_by_side_frame(
            user_view,
            ref_panel,
            score_sm,
            err_sm,
            step_idx,
            len(segments),
        )

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window)
            combined = fit_to_window(combined, win_w, win_h)
        except Exception:
            pass

        cv2.imshow(window, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            paused = not paused
            step_t0 = time.time() - (ref_idx - step_start) / ref_fps
        if key in (ord("s"), ord("S")):
            play_sequence = not play_sequence
            step_t0 = time.time()
            ref_idx = step_start
        if key == 32:
            step_t0 = time.time()
            ref_idx = step_start
        if key == ord("["):
            step_idx = max(0, step_idx - 1)
            step_start = segments[step_idx]["start"]
            step_end = segments[step_idx]["end"]
            step_t0 = time.time()
            ref_idx = step_start
        if key == ord("]"):
            step_idx = min(len(segments) - 1, step_idx + 1)
            step_start = segments[step_idx]["start"]
            step_end = segments[step_idx]["end"]
            step_t0 = time.time()
            ref_idx = step_start
        if key in (ord("m"), ord("M")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    show_session_summary(score_history, step_score_history)


if __name__ == "__main__":
    main()
