import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dance_library import ensure_library_structure, get_selected_dance

import cv2
import numpy as np
import mediapipe as mp

from pose_utils import (
    landmarks_to_xy,
    dist,
    MovingAverage,
)

mp_pose = mp.solutions.pose
MP_LM = mp_pose.PoseLandmark
MP_IDX = {name: lm.value for name, lm in MP_LM.__members__.items()}
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


class OverlayScoringSystem:
    def __init__(self, connection_weight: float = 1.0, joint_weight: float = 1.5, smooth_window: int = 5):
        self.connection_weight = connection_weight
        self.joint_weight = joint_weight
        self.max_distance_threshold = 50.0
        self.smooth_score = MovingAverage(smooth_window)

        self.last_joint_distances = {}
        self.last_connection_distances = {}

    def calculate_overlap_percentage(
        self,
        user_joints: np.ndarray,
        ref_joints: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float, Dict]:
        h, w = frame_shape[:2]
        diagonal = np.sqrt(w ** 2 + h ** 2)
        adaptive_threshold = self.max_distance_threshold * (diagonal / 1000.0)

        debug_info = {
            "joint_distances": {},
            "connection_distances": {},
            "joint_scores": {},
            "connection_scores": {},
        }

        joint_scores = []
        self.last_joint_distances = {}

        for joint_idx in range(len(user_joints)):
            user_pos = user_joints[joint_idx]
            ref_pos = ref_joints[joint_idx]

            if np.all(user_pos == 0) or np.all(ref_pos == 0):
                continue

            distance = dist(user_pos, ref_pos)
            self.last_joint_distances[joint_idx] = distance

            overlap = max(0, 100 * (1 - distance / adaptive_threshold))
            overlap = min(100, overlap)

            joint_scores.append(overlap)
            debug_info["joint_distances"][joint_idx] = distance
            debug_info["joint_scores"][joint_idx] = overlap

        connection_scores = []
        self.last_connection_distances = {}

        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection

            if (
                np.all(user_joints[idx1] == 0)
                or np.all(user_joints[idx2] == 0)
                or np.all(ref_joints[idx1] == 0)
                or np.all(ref_joints[idx2] == 0)
            ):
                continue

            user_mid = (user_joints[idx1] + user_joints[idx2]) / 2
            ref_mid = (ref_joints[idx1] + ref_joints[idx2]) / 2

            dist1 = dist(user_joints[idx1], ref_joints[idx1])
            dist2 = dist(user_joints[idx2], ref_joints[idx2])
            dist_mid = dist(user_mid, ref_mid)

            combined_dist = (dist1 + dist2 + dist_mid) / 3
            self.last_connection_distances[connection] = combined_dist

            overlap = max(0, 100 * (1 - combined_dist / adaptive_threshold))
            overlap = min(100, overlap)

            connection_scores.append(overlap)
            debug_info["connection_distances"][f"{idx1}-{idx2}"] = combined_dist
            debug_info["connection_scores"][f"{idx1}-{idx2}"] = overlap

        avg_joint_score = np.mean(joint_scores) if joint_scores else 0
        avg_connection_score = np.mean(connection_scores) if connection_scores else 0

        total_weight = self.joint_weight + self.connection_weight
        raw_score = (
            self.joint_weight * avg_joint_score
            + self.connection_weight * avg_connection_score
        ) / total_weight

        smooth_score = self.smooth_score.update(raw_score)

        debug_info["avg_joint_score"] = avg_joint_score
        debug_info["avg_connection_score"] = avg_connection_score
        debug_info["raw_score"] = raw_score
        debug_info["smooth_score"] = smooth_score

        return smooth_score, raw_score, debug_info


class PoseOverlayVisualizer:
    def __init__(self):
        self.colors = {
            "excellent": (0, 255, 0),
            "good": (0, 255, 255),
            "fair": (0, 165, 255),
            "poor": (0, 0, 255),
            "bad": (255, 0, 255),
            "reference": (220, 220, 220),
            "reference_dim": (100, 100, 100),
        }

    def get_accuracy_color(self, percentage: float) -> Tuple[int, int, int]:
        if percentage >= 80:
            return self.colors["excellent"]
        elif percentage >= 60:
            return self.colors["good"]
        elif percentage >= 40:
            return self.colors["fair"]
        elif percentage >= 20:
            return self.colors["poor"]
        return self.colors["bad"]

    def draw_overlay_analysis(
        self,
        frame: np.ndarray,
        user_joints: np.ndarray,
        ref_joints: np.ndarray,
        score: float,
        debug_info: Dict,
    ) -> np.ndarray:
        reference_overlay = frame.copy()

        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if np.all(ref_joints[idx1] == 0) or np.all(ref_joints[idx2] == 0):
                continue
            pt1 = tuple(ref_joints[idx1].astype(int))
            pt2 = tuple(ref_joints[idx2].astype(int))
            cv2.line(reference_overlay, pt1, pt2, self.colors["reference"], 2, cv2.LINE_AA)

        for joint_idx in range(len(ref_joints)):
            if np.all(ref_joints[joint_idx] == 0):
                continue
            pt = tuple(ref_joints[joint_idx].astype(int))
            cv2.circle(reference_overlay, pt, 4, self.colors["reference"], -1)

        cv2.addWeighted(reference_overlay, 0.28, frame, 0.72, 0, frame)

        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if np.all(user_joints[idx1] == 0) or np.all(user_joints[idx2] == 0):
                continue

            conn_key = f"{idx1}-{idx2}"
            conn_score = debug_info.get("connection_scores", {}).get(conn_key, 0)
            color = self.get_accuracy_color(conn_score)

            pt1 = tuple(user_joints[idx1].astype(int))
            pt2 = tuple(user_joints[idx2].astype(int))
            cv2.line(frame, pt1, pt2, color, 4, cv2.LINE_AA)

        for joint_idx in range(len(user_joints)):
            if np.all(user_joints[joint_idx] == 0):
                continue

            joint_score = debug_info.get("joint_scores", {}).get(joint_idx, 0)
            color = self.get_accuracy_color(joint_score)

            pt = tuple(user_joints[joint_idx].astype(int))
            cv2.circle(frame, pt, 7, color, -1)
            cv2.circle(frame, pt, 9, (255, 255, 255), 1)

        return frame


class DanceOverlayTrainer:
    def __init__(self, reference_path: str):
        self.load_reference(reference_path)
        self.scoring_system = OverlayScoringSystem()
        self.visualizer = PoseOverlayVisualizer()
        self.setup_display()

    def load_reference(self, path: str):
        with open(path, "r") as f:
            ref = json.load(f)

        self.ref_norm_xy = np.array(ref["ref_norm_xy"], dtype=np.float32)
        self.segments = ref.get("segments", [{"start": 0, "end": len(self.ref_norm_xy) - 1}])
        self.fps = float(ref["fps"])

        print(f"Loaded reference with {len(self.ref_norm_xy)} frames")
        print(f"Found {len(self.segments)} movement segments")

    def setup_display(self):
        self.window_name = "AfroDance Learn - Advanced Overlay Analysis"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        self.show_reference = True
        self.paused = False
        self.current_segment = 0
        self.frame_idx = self.segments[0]["start"]

    def get_user_pose_in_reference_space(
        self,
        user_landmarks,
        frame_w: int,
        frame_h: int
    ) -> Optional[np.ndarray]:
        if not user_landmarks:
            return None

        xy_px, _ = landmarks_to_xy(user_landmarks.landmark, frame_w, frame_h)

        left_hip = xy_px[MP_IDX["LEFT_HIP"]]
        right_hip = xy_px[MP_IDX["RIGHT_HIP"]]
        left_shoulder = xy_px[MP_IDX["LEFT_SHOULDER"]]
        right_shoulder = xy_px[MP_IDX["RIGHT_SHOULDER"]]

        hip_center = (left_hip + right_hip) / 2
        shoulder_width = max(np.linalg.norm(left_shoulder - right_shoulder), 1e-3)

        ref_frame = self.ref_norm_xy[self.frame_idx]
        user_space_joints = hip_center[None, :] + ref_frame * shoulder_width

        return user_space_joints

    def draw_hud(self, frame: np.ndarray, score_history: List[float], h: int, w: int):
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

        if score_history:
            current_score = score_history[-1]
            avg_score = np.mean(score_history[-30:])

            cv2.putText(
                frame,
                f"Overlap: {current_score:.1f}% | Avg (30f): {avg_score:.1f}%",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            bar_width = 340
            bar_height = 24
            bar_x = w - bar_width - 30
            bar_y = 28

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (90, 90, 90), -1)

            fill_width = int(bar_width * current_score / 100)
            color = self.visualizer.get_accuracy_color(current_score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

        segment_text = f"Segment {self.current_segment + 1}/{len(self.segments)} | Frame {self.frame_idx}"
        cv2.putText(
            frame,
            segment_text,
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            1
        )

        controls = "SPACE pause | R toggle ref | N next seg | P prev seg | Q quit"
        cv2.putText(
            frame,
            controls,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            1
        )

        if self.paused:
            cv2.putText(
                frame,
                "PAUSED",
                (w // 2 - 60, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

    def run(self):
        cap = cv2.VideoCapture(0)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        segment_start = self.segments[self.current_segment]["start"]
        segment_end = self.segments[self.current_segment]["end"]
        start_time = time.time()

        score_history = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if not self.paused:
                elapsed = time.time() - start_time
                self.frame_idx = segment_start + int(elapsed * self.fps)

                if self.frame_idx > segment_end:
                    self.frame_idx = segment_start
                    start_time = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            display_frame = frame.copy()

            if result.pose_landmarks and self.show_reference:
                ref_in_user_space = self.get_user_pose_in_reference_space(result.pose_landmarks, w, h)

                if ref_in_user_space is not None:
                    user_joints, _ = landmarks_to_xy(result.pose_landmarks.landmark, w, h)

                    score, raw_score, debug_info = self.scoring_system.calculate_overlap_percentage(
                        user_joints, ref_in_user_space, frame.shape
                    )

                    score_history.append(score)

                    display_frame = self.visualizer.draw_overlay_analysis(
                        display_frame,
                        user_joints,
                        ref_in_user_space,
                        score,
                        debug_info
                    )

            self.draw_hud(display_frame, score_history, h, w)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key == 32:
                self.paused = not self.paused
                start_time = time.time() - (self.frame_idx - segment_start) / self.fps
            elif key in (ord("r"), ord("R")):
                self.show_reference = not self.show_reference
            elif key in (ord("n"), ord("N")):
                self.current_segment = (self.current_segment + 1) % len(self.segments)
                segment_start = self.segments[self.current_segment]["start"]
                segment_end = self.segments[self.current_segment]["end"]
                self.frame_idx = segment_start
                start_time = time.time()
                score_history.clear()
            elif key in (ord("p"), ord("P")):
                self.current_segment = (self.current_segment - 1) % len(self.segments)
                segment_start = self.segments[self.current_segment]["start"]
                segment_end = self.segments[self.current_segment]["end"]
                self.frame_idx = segment_start
                start_time = time.time()
                score_history.clear()

        cap.release()
        pose.close()
        cv2.destroyAllWindows()


def main():
    repo_root = Path(__file__).resolve().parent
    ensure_library_structure(repo_root)

    selected = get_selected_dance(repo_root)
    if selected is None:
        print("No dances were found in data/dances.")
        print("Create at least one dance folder with instructor.mp4 and reference.json.")
        return

    reference_path = selected["reference_path"]

    if not os.path.exists(reference_path):
        print(f"Reference not found at {reference_path}")
        print("Generate reference data for the selected dance first.")
        return

    print(f"Opening Detailed Analysis Mode for: {selected['name']}")
    trainer = DanceOverlayTrainer(str(reference_path))
    trainer.run()

if __name__ == "__main__":
    main()
