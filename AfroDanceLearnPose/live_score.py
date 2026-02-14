import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp

from pose_utils import (
    landmarks_to_xy,
    normalize_xy,
    compute_key_angles,
    angles_to_vector,
    mean_abs_error_deg,
    error_to_score_0_100,
    MovingAverage,
)

mp_pose = mp.solutions.pose
MP_LM = mp_pose.PoseLandmark
MP_IDX = {name: lm.value for name, lm in MP_LM.__members__.items()}


# ───────────────────────────── Reference Loading ─────────────────────────────
def load_reference():
    ref_path = os.path.join("data", "references", "instructor_reference.json")
    if not os.path.exists(ref_path):
        raise RuntimeError("Missing reference file. Run: python extract_reference.py")

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


# ───────────────────────────── Geometry Helpers ─────────────────────────────
def get_user_anchor_and_scale(xy_px):
    lh = xy_px[MP_IDX["LEFT_HIP"]]
    rh = xy_px[MP_IDX["RIGHT_HIP"]]
    ls = xy_px[MP_IDX["LEFT_SHOULDER"]]
    rs = xy_px[MP_IDX["RIGHT_SHOULDER"]]
    hip_center = 0.5 * (lh + rh)
    shoulder_width = max(np.linalg.norm(ls - rs), 1e-3)
    return hip_center, shoulder_width


def ref_norm_to_screen(ref_norm_xy_frame, hip_center, shoulder_width):
    return hip_center[None, :] + ref_norm_xy_frame * shoulder_width


# ───────────────────────────── Drawing Helpers ─────────────────────────────
def draw_ghost(frame, ref_xy_px, alpha=0.45, color=(0, 255, 255)):
    overlay = frame.copy()

    for a, b in mp_pose.POSE_CONNECTIONS:
        ax, ay = ref_xy_px[a]
        bx, by = ref_xy_px[b]
        if (ax == 0 and ay == 0) or (bx == 0 and by == 0):
            continue
        cv2.line(
            overlay,
            (int(ax), int(ay)),
            (int(bx), int(by)),
            color,
            3,
            cv2.LINE_AA,
        )

    for i in (
        MP_IDX["LEFT_WRIST"],
        MP_IDX["RIGHT_WRIST"],
        MP_IDX["LEFT_ANKLE"],
        MP_IDX["RIGHT_ANKLE"],
    ):
        x, y = ref_xy_px[i]
        if x != 0 or y != 0:
            cv2.circle(overlay, (int(x), int(y)), 6, color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def fit_to_window(frame, win_w, win_h):
    h, w = frame.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x0 = (win_w - new_w) // 2
    y0 = (win_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# ───────────────────────────── Main App ─────────────────────────────
def main():
    ref_vectors, angle_names, ref_fps, ref_norm_xy, segments = load_reference()

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    overlay_on = True
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

    window = "AfroDance Learn"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

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

        if res.pose_landmarks:
            xy_px, _ = landmarks_to_xy(res.pose_landmarks.landmark, w, h)
            hip, sw = get_user_anchor_and_scale(xy_px)

            if overlay_on:
                ref_xy = ref_norm_to_screen(ref_norm_xy[ref_idx], hip, sw)
                draw_ghost(frame, ref_xy)

            norm_user = normalize_xy(
                xy_px,
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

        # ───────── Compact Top HUD ─────────
        cv2.rectangle(frame, (8, 8), (500, 58), (0, 0, 0), -1)
        mode = "SEQ" if play_sequence else "STEP"
        cv2.putText(
            frame,
            f"{mode} | Step {step_idx+1}/{len(segments)} | Score {score_sm:.0f} | Err {err_sm:.1f}°",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        prog = (ref_idx - step_start) / max(1, step_end - step_start)
        cv2.rectangle(frame, (16, 40), (480, 46), (255, 255, 255), 1)
        cv2.rectangle(frame, (16, 40), (16 + int(464 * prog), 46), (255, 255, 255), -1)

        # ───────── Bottom Info Bar (never clipped) ─────────
        bottom_h = 36
        cv2.rectangle(frame, (0, h - bottom_h), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "Controls: S sequence | [ ] step | SPACE restart | P pause | O overlay | M fullscreen | Q quit",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # Resize safely to window
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window)
            frame = fit_to_window(frame, win_w, win_h)
        except Exception:
            pass

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("o"), ord("O")):
            overlay_on = not overlay_on
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
        if key == ord("]"):
            step_idx = min(len(segments) - 1, step_idx + 1)
            step_start = segments[step_idx]["start"]
            step_end = segments[step_idx]["end"]
            step_t0 = time.time()
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


if __name__ == "__main__":
    main()
