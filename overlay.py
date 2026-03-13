import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional

from pose_utils import (
    landmarks_to_xy,
    normalize_xy,
    compute_key_angles,
    angles_to_vector,
    dist,
    MovingAverage,
)

mp_pose = mp.solutions.pose
MP_LM = mp_pose.PoseLandmark
MP_IDX = {name: lm.value for name, lm in MP_LM.__members__.items()}
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


class OverlayScoringSystem:
    """Scoring system based on percentage of skeleton overlay between dancer and instructor"""
    
    def __init__(self, 
                 connection_weight: float = 1.0,
                 joint_weight: float = 1.5,
                 smooth_window: int = 5):
        """
        Args:
            connection_weight: Weight for bone/connection matching
            joint_weight: Weight for joint position matching (ends of bones)
            smooth_window: Frames for moving average smoothing
        """
        self.connection_weight = connection_weight
        self.joint_weight = joint_weight
        self.max_distance_threshold = 50.0  # pixels - max distance for 0% overlap
        self.smooth_score = MovingAverage(smooth_window)
        self.smooth_percentage = MovingAverage(smooth_window)
        
        # Store last frame data for visualization
        self.last_joint_distances = {}
        self.last_connection_distances = {}
        
    def calculate_overlap_percentage(self, 
                                     user_joints: np.ndarray, 
                                     ref_joints: np.ndarray,
                                     frame_shape: Tuple[int, int]) -> Tuple[float, float, Dict]:
        """
        Calculate what percentage of the instructor's skeleton overlaps with the dancer.
        
        Returns:
            overlap_percentage: Overall overlap score (0-100%)
            raw_score: Raw weighted score
            debug_info: Dictionary with detailed distances
        """
        h, w = frame_shape[:2]
        diagonal = np.sqrt(w**2 + h**2)
        adaptive_threshold = self.max_distance_threshold * (diagonal / 1000.0)  # Scale with resolution
        
        debug_info = {
            'joint_distances': {},
            'connection_distances': {},
            'joint_scores': {},
            'connection_scores': {}
        }
        
        # 1. Calculate joint position overlap
        joint_scores = []
        self.last_joint_distances = {}
        
        for joint_idx in range(len(user_joints)):
            user_pos = user_joints[joint_idx]
            ref_pos = ref_joints[joint_idx]
            
            # Skip if either point is missing (0,0)
            if np.all(user_pos == 0) or np.all(ref_pos == 0):
                continue
                
            # Calculate distance
            distance = dist(user_pos, ref_pos)
            self.last_joint_distances[joint_idx] = distance
            
            # Convert distance to overlap percentage (inverse relationship)
            # 0 distance = 100% overlap, threshold distance = 0% overlap
            overlap = max(0, 100 * (1 - distance / adaptive_threshold))
            overlap = min(100, overlap)  # Cap at 100%
            
            joint_scores.append(overlap)
            debug_info['joint_distances'][joint_idx] = distance
            debug_info['joint_scores'][joint_idx] = overlap
        
        # 2. Calculate bone/connection overlap
        connection_scores = []
        self.last_connection_distances = {}
        
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            
            # Skip if either point is missing
            if (np.all(user_joints[idx1] == 0) or np.all(user_joints[idx2] == 0) or
                np.all(ref_joints[idx1] == 0) or np.all(ref_joints[idx2] == 0)):
                continue
            
            # Calculate line segment distances using point-to-line distance
            # For each point on the reference bone, find closest point on user bone
            # Simplified: use midpoint and endpoint distances
            user_mid = (user_joints[idx1] + user_joints[idx2]) / 2
            ref_mid = (ref_joints[idx1] + ref_joints[idx2]) / 2
            
            # Check endpoint distances
            dist1 = dist(user_joints[idx1], ref_joints[idx1])
            dist2 = dist(user_joints[idx2], ref_joints[idx2])
            dist_mid = dist(user_mid, ref_mid)
            
            # Combined distance metric
            combined_dist = (dist1 + dist2 + dist_mid) / 3
            self.last_connection_distances[connection] = combined_dist
            
            # Convert to overlap percentage
            overlap = max(0, 100 * (1 - combined_dist / adaptive_threshold))
            overlap = min(100, overlap)
            
            connection_scores.append(overlap)
            debug_info['connection_distances'][f"{idx1}-{idx2}"] = combined_dist
            debug_info['connection_scores'][f"{idx1}-{idx2}"] = overlap
        
        # 3. Calculate weighted overall score
        avg_joint_score = np.mean(joint_scores) if joint_scores else 0
        avg_connection_score = np.mean(connection_scores) if connection_scores else 0
        
        total_weight = self.joint_weight + self.connection_weight
        raw_score = (self.joint_weight * avg_joint_score + 
                     self.connection_weight * avg_connection_score) / total_weight
        
        # Smooth the score
        smooth_score = self.smooth_score.update(raw_score)
        
        debug_info['avg_joint_score'] = avg_joint_score
        debug_info['avg_connection_score'] = avg_connection_score
        debug_info['raw_score'] = raw_score
        debug_info['smooth_score'] = smooth_score
        
        return smooth_score, raw_score, debug_info


class PoseOverlayVisualizer:
    """Visualizes the overlay between dancer and instructor with color-coded accuracy"""
    
    def __init__(self):
        self.colors = {
            'perfect': (0, 255, 0),      # Green - 80-100%
            'good': (0, 255, 255),        # Yellow - 60-80%
            'fair': (0, 165, 255),        # Orange - 40-60%
            'poor': (0, 0, 255),          # Red - 20-40%
            'bad': (255, 0, 255),         # Magenta - 0-20%
            'reference': (255, 255, 255), # White
        }
        
    def get_accuracy_color(self, percentage: float) -> Tuple[int, int, int]:
        """Return color based on accuracy percentage"""
        if percentage >= 80:
            return self.colors['perfect']
        elif percentage >= 60:
            return self.colors['good']
        elif percentage >= 40:
            return self.colors['fair']
        elif percentage >= 20:
            return self.colors['poor']
        else:
            return self.colors['bad']
    
    def draw_overlay_analysis(self, 
                             frame: np.ndarray,
                             user_joints: np.ndarray,
                             ref_joints: np.ndarray,
                             scoring_system: OverlayScoringSystem,
                             score: float,
                             debug_info: Dict) -> np.ndarray:
        """Draw the overlay analysis on the frame"""
        h, w = frame.shape[:2]
        
        # Draw reference skeleton (semi-transparent white)
        ref_alpha = 0.3
        ref_overlay = frame.copy()
        
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if np.all(ref_joints[idx1] == 0) or np.all(ref_joints[idx2] == 0):
                continue
            pt1 = tuple(ref_joints[idx1].astype(int))
            pt2 = tuple(ref_joints[idx2].astype(int))
            cv2.line(ref_overlay, pt1, pt2, self.colors['reference'], 2, cv2.LINE_AA)
        
        for joint_idx in range(len(ref_joints)):
            if np.all(ref_joints[joint_idx] == 0):
                continue
            pt = tuple(ref_joints[joint_idx].astype(int))
            cv2.circle(ref_overlay, pt, 4, self.colors['reference'], -1)
        
        # Blend reference skeleton
        cv2.addWeighted(ref_overlay, ref_alpha, frame, 1 - ref_alpha, 0, frame)
        
        # Draw user skeleton with color-coded accuracy
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if np.all(user_joints[idx1] == 0) or np.all(user_joints[idx2] == 0):
                continue
            
            # Get accuracy for this connection
            conn_key = f"{idx1}-{idx2}"
            conn_score = debug_info.get('connection_scores', {}).get(conn_key, 0)
            color = self.get_accuracy_color(conn_score)
            
            pt1 = tuple(user_joints[idx1].astype(int))
            pt2 = tuple(user_joints[idx2].astype(int))
            cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
        
        # Draw user joints with color-coded accuracy
        for joint_idx in range(len(user_joints)):
            if np.all(user_joints[joint_idx] == 0):
                continue
            
            joint_score = debug_info.get('joint_scores', {}).get(joint_idx, 0)
            color = self.get_accuracy_color(joint_score)
            
            pt = tuple(user_joints[joint_idx].astype(int))
            cv2.circle(frame, pt, 6, color, -1)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1)
        
        return frame


class DanceOverlayTrainer:
    """Main application for real-time dance training with overlay scoring"""
    
    def __init__(self, reference_path: str):
        self.load_reference(reference_path)
        self.scoring_system = OverlayScoringSystem()
        self.visualizer = PoseOverlayVisualizer()
        self.setup_display()
        
    def load_reference(self, path: str):
        """Load instructor reference data"""
        with open(path, 'r') as f:
            ref = json.load(f)
        
        self.ref_norm_xy = np.array(ref["ref_norm_xy"], dtype=np.float32)
        self.segments = ref.get("segments", [{"start": 0, "end": len(self.ref_norm_xy) - 1}])
        self.fps = float(ref["fps"])
        
        print(f"Loaded reference with {len(self.ref_norm_xy)} frames")
        print(f"Found {len(self.segments)} movement segments")
        
    def setup_display(self):
        """Setup display window and controls"""
        self.window_name = "Dance Overlay Trainer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        self.show_reference = True
        self.paused = False
        self.current_segment = 0
        self.frame_idx = self.segments[0]["start"]
        
    def get_user_pose_in_reference_space(self, 
                                         user_landmarks, 
                                         frame_w: int, 
                                         frame_h: int) -> Optional[np.ndarray]:
        """Transform user pose to same coordinate space as reference"""
        if not user_landmarks:
            return None
            
        xy_px, _ = landmarks_to_xy(user_landmarks.landmark, frame_w, frame_h)
        
        # Get scaling factors from user's body
        left_hip = xy_px[MP_IDX["LEFT_HIP"]]
        right_hip = xy_px[MP_IDX["RIGHT_HIP"]]
        left_shoulder = xy_px[MP_IDX["LEFT_SHOULDER"]]
        right_shoulder = xy_px[MP_IDX["RIGHT_SHOULDER"]]
        
        hip_center = (left_hip + right_hip) / 2
        shoulder_width = max(np.linalg.norm(left_shoulder - right_shoulder), 1e-3)
        
        # Transform reference to user's space
        ref_frame = self.ref_norm_xy[self.frame_idx]
        user_space_joints = hip_center[None, :] + ref_frame * shoulder_width
        
        return user_space_joints
        
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        segment_start = self.segments[self.current_segment]["start"]
        segment_end = self.segments[self.current_segment]["end"]
        start_time = time.time()
        
        # Statistics tracking
        score_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Update reference frame index
            if not self.paused:
                elapsed = time.time() - start_time
                self.frame_idx = segment_start + int(elapsed * self.fps)
                
                if self.frame_idx > segment_end:
                    # Loop back to start of segment
                    self.frame_idx = segment_start
                    start_time = time.time()
            
            # Process pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            
            # Create display frame
            display_frame = frame.copy()
            
            if result.pose_landmarks and self.show_reference:
                # Transform reference to user's space
                ref_in_user_space = self.get_user_pose_in_reference_space(result, w, h)
                
                if ref_in_user_space is not None:
                    # Get user's actual joints
                    user_joints, _ = landmarks_to_xy(result.pose_landmarks.landmark, w, h)
                    
                    # Calculate overlay score
                    score, raw_score, debug_info = self.scoring_system.calculate_overlap_percentage(
                        user_joints, ref_in_user_space, frame.shape
                    )
                    
                    score_history.append(score)
                    
                    # Draw visualization
                    display_frame = self.visualizer.draw_overlay_analysis(
                        display_frame, user_joints, ref_in_user_space,
                        self.scoring_system, score, debug_info
                    )
                    
                    # Draw connections between corresponding joints (optional)
                    # for i in range(len(user_joints)):
                    #     if np.all(user_joints[i] != 0) and np.all(ref_in_user_space[i] != 0):
                    #         cv2.line(display_frame, 
                    #                  tuple(user_joints[i].astype(int)),
                    #                  tuple(ref_in_user_space[i].astype(int)),
                    #                  (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw HUD
            self.draw_hud(display_frame, score_history, h, w)
            
            # Show frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - pause
                self.paused = not self.paused
                start_time = time.time() - (self.frame_idx - segment_start) / self.fps
            elif key == ord('r'):  # Toggle reference
                self.show_reference = not self.show_reference
            elif key == ord('n'):  # Next segment
                self.current_segment = (self.current_segment + 1) % len(self.segments)
                segment_start = self.segments[self.current_segment]["start"]
                segment_end = self.segments[self.current_segment]["end"]
                self.frame_idx = segment_start
                start_time = time.time()
                score_history.clear()
            elif key == ord('p'):  # Previous segment
                self.current_segment = (self.current_segment - 1) % len(self.segments)
                segment_start = self.segments[self.current_segment]["start"]
                segment_end = self.segments[self.current_segment]["end"]
                self.frame_idx = segment_start
                start_time = time.time()
                score_history.clear()
                
        cap.release()
        cv2.destroyAllWindows()
        
    def draw_hud(self, frame: np.ndarray, score_history: List[float], h: int, w: int):
        """Draw heads-up display with score information"""
        # Top bar with current score
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        if score_history:
            current_score = score_history[-1]
            avg_score = np.mean(score_history[-30:])  # Last 30 frames average
            
            # Score text
            score_text = f"Overlap: {current_score:.1f}% | Avg (30f): {avg_score:.1f}%"
            cv2.putText(frame, score_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Score bar
            bar_width = 300
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = 20
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Filled portion
            fill_width = int(bar_width * current_score / 100)
            color = self.visualizer.get_accuracy_color(current_score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 1)
        
        # Segment info
        segment_text = f"Segment {self.current_segment + 1}/{len(self.segments)} | Frame {self.frame_idx}"
        cv2.putText(frame, segment_text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        controls = "Space:Pause R:ToggleRef N:NextSeg P:PrevSeg Q:Quit"
        cv2.putText(frame, controls, (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Pause indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (w//2 - 50, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def main():
    reference_path = os.path.join("data", "references", "instructor_reference.json")
    
    if not os.path.exists(reference_path):
        print(f"Reference not found at {reference_path}")
        print("Please run extract_reference.py first")
        return
    
    trainer = DanceOverlayTrainer(reference_path)
    trainer.run()


if __name__ == "__main__":
    main()