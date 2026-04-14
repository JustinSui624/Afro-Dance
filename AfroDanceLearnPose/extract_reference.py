import json
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
    )
except ImportError:
    from pose_utils import (
        landmarks_to_xy,
        normalize_xy,
        compute_key_angles,
        angles_to_vector,
    )

mp_pose = mp.solutions.pose
MP_LM = mp_pose.PoseLandmark
MP_IDX = {name: lm.value for name, lm in MP_LM.__members__.items()}

ANGLE_NAMES = [
    "left_elbow", "right_elbow",
    "left_knee", "right_knee",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
]


def smooth_1d(x, win=11):
    if len(x) < win:
        return x
    win = max(3, int(win) | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, k, mode="valid")


def detect_step_boundaries(motion, fps, min_step_sec=1.0, max_steps=12):
    t = len(motion)
    if t < 5:
        return [0, t - 1]

    m = smooth_1d(motion, win=int(max(5, fps * 0.25)) | 1)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)

    inv = 1.0 - m
    candidates = []
    for i in range(1, t - 1):
        if inv[i] > inv[i - 1] and inv[i] > inv[i + 1] and inv[i] > 0.55:
            candidates.append(i)

    min_gap = int(max(1, fps * min_step_sec))
    boundaries = [0]
    last = 0
    for idx in candidates:
        if idx - last >= min_gap:
            boundaries.append(idx)
            last = idx

    boundaries.append(t - 1)

    if len(boundaries) - 1 > max_steps:
        inner = boundaries[1:-1]
        inner_sorted = sorted(inner, key=lambda i: inv[i], reverse=True)
        inner_keep = sorted(inner_sorted[: max_steps - 1])
        boundaries = [0] + inner_keep + [t - 1]

    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != t - 1:
        boundaries.append(t - 1)

    return boundaries


def boundaries_to_segments(boundaries):
    segs = []
    for i in range(len(boundaries) - 1):
        a = int(boundaries[i])
        b = int(boundaries[i + 1])
        if b <= a:
            continue
        segs.append({"start": a, "end": b})
    return segs


def resolve_repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_video_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "instructor.mp4",
        repo_root / "AfroDanceLearnPose" / "data" / "instructor.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return repo_root / "data" / "instructor.mp4"


def main():
    repo_root = resolve_repo_root()
    video_path = resolve_video_path(repo_root)
    out_dir = repo_root / "data" / "references"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ref_vectors = []
    ref_vis = []
    ref_norm_xy = []
    motion_energy = []

    prev_norm = None
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks is None:
            ref_vectors.append([0.0] * len(ANGLE_NAMES))
            ref_vis.append(0.0)
            ref_norm_xy.append([[0.0, 0.0] for _ in range(33)])
            motion_energy.append(0.0)
            prev_norm = None
        else:
            xy, vis = landmarks_to_xy(res.pose_landmarks.landmark, w, h)

            norm = normalize_xy(
                xy,
                left_hip_idx=MP_IDX["LEFT_HIP"],
                right_hip_idx=MP_IDX["RIGHT_HIP"],
                left_shoulder_idx=MP_IDX["LEFT_SHOULDER"],
                right_shoulder_idx=MP_IDX["RIGHT_SHOULDER"],
            )
            ref_norm_xy.append(norm.tolist())

            angles = compute_key_angles(norm, MP_IDX)
            vec = angles_to_vector(angles, ANGLE_NAMES)
            ref_vectors.append(vec.tolist())

            key_idxs = [
                MP_IDX["LEFT_HIP"], MP_IDX["RIGHT_HIP"],
                MP_IDX["LEFT_SHOULDER"], MP_IDX["RIGHT_SHOULDER"],
                MP_IDX["LEFT_ELBOW"], MP_IDX["RIGHT_ELBOW"],
                MP_IDX["LEFT_WRIST"], MP_IDX["RIGHT_WRIST"],
                MP_IDX["LEFT_KNEE"], MP_IDX["RIGHT_KNEE"],
                MP_IDX["LEFT_ANKLE"], MP_IDX["RIGHT_ANKLE"],
            ]
            ref_vis.append(float(np.mean(vis[key_idxs])))

            if prev_norm is None:
                motion_energy.append(0.0)
            else:
                d = norm - prev_norm
                motion_energy.append(float(np.mean(np.linalg.norm(d, axis=1))))
            prev_norm = norm.copy()

        frame_i += 1
        if frame_i % 120 == 0:
            print(f"Processed {frame_i}/{total if total else '?'} frames...")

    cap.release()
    pose.close()

    motion_energy = np.array(motion_energy, dtype=np.float32)
    boundaries = detect_step_boundaries(motion_energy, fps=fps, min_step_sec=1.2, max_steps=10)
    segments = boundaries_to_segments(boundaries)

    out_path = out_dir / "instructor_reference.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": str(video_path),
                "fps": float(fps),
                "angle_names": ANGLE_NAMES,
                "vectors": ref_vectors,
                "quality": ref_vis,
                "num_frames": len(ref_vectors),
                "ref_norm_xy": ref_norm_xy,
                "motion_energy": motion_energy.tolist(),
                "segments": segments,
            },
            f,
        )

    print(f"\nSaved reference to: {out_path}")
    print(f"Detected {len(segments)} step segments:")
    for i, s in enumerate(segments, 1):
        print(f"  Step {i}: frames {s['start']}–{s['end']}  ({(s['end']-s['start'])/fps:.2f}s)")


if __name__ == "__main__":
    main()
