import json
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

class SkeletonLoader:
    """Load and manage skeleton data from motion capture JSON"""
    
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract skeleton data
        self.ref_norm_xy = np.array(self.data["ref_norm_xy"], dtype=np.float32)
        self.vectors = np.array(self.data["vectors"], dtype=np.float32)
        self.quality = np.array(self.data["quality"], dtype=np.float32)
        self.segments = self.data.get("segments", [])
        self.fps = self.data.get("fps", 30.0)
        self.angle_names = self.data.get("angle_names", [])
        
        print(f"Loaded {len(self.ref_norm_xy)} frames of skeleton data")
        print(f"Found {len(self.segments)} movement segments")
        print(f"Data shape: {self.ref_norm_xy.shape}")
        
    def get_skeleton_frame(self, frame_idx, scale=1.0, offset=(0, 0)):
        """
        Get skeleton points for a specific frame
        
        Args:
            frame_idx: Frame index
            scale: Scale factor for the skeleton
            offset: (x, y) offset to position the skeleton
            
        Returns:
            Dictionary of joint positions
        """
        if frame_idx >= len(self.ref_norm_xy):
            frame_idx = len(self.ref_norm_xy) - 1
            
        frame_data = self.ref_norm_xy[frame_idx]
        
        # Convert to pixel coordinates with scale and offset
        skeleton = {}
        for joint_idx in range(len(frame_data)):
            x, y = frame_data[joint_idx]
            # Apply scale and offset
            screen_x = int(x * scale + offset[0])
            screen_y = int(y * scale + offset[1])
            skeleton[joint_idx] = (screen_x, screen_y)
            
        return skeleton
    
    def get_skeleton_in_user_space(self, frame_idx, user_hip_center, user_shoulder_width):
        """
        Transform reference skeleton to user's body space
        
        Args:
            frame_idx: Reference frame index
            user_hip_center: (x, y) of user's hip center
            user_shoulder_width: User's shoulder width in pixels
            
        Returns:
            Dictionary of joint positions in user's space
        """
        if frame_idx >= len(self.ref_norm_xy):
            frame_idx = len(self.ref_norm_xy) - 1
            
        frame_data = self.ref_norm_xy[frame_idx]
        
        skeleton = {}
        for joint_idx in range(len(frame_data)):
            x, y = frame_data[joint_idx]
            # Transform: normalized -> user space
            screen_x = int(x * user_shoulder_width + user_hip_center[0])
            screen_y = int(y * user_shoulder_width + user_hip_center[1])
            skeleton[joint_idx] = (screen_x, screen_y)
            
        return skeleton
    
class JSONSkeletonDrawer:
    """Draw skeletons loaded from JSON motion capture data"""
    
    def __init__(self, skeleton_loader):
        self.loader = skeleton_loader
        self.current_frame = 0
        
        # Color schemes
        self.colors = {
            'reference': (255, 255, 255),  # White
            'torso': (255, 255, 0),        # Cyan
            'left_arm': (0, 255, 0),       # Green
            'right_arm': (0, 255, 0),      # Green
            'left_leg': (0, 0, 255),       # Red
            'right_leg': (0, 0, 255),      # Red
        }
        
    def draw_reference_skeleton(self, frame, frame_idx, 
                               position=(100, 100), 
                               scale=200,
                               alpha=0.7):
        """
        Draw reference skeleton at a fixed position on screen
        
        Args:
            frame: Image frame
            frame_idx: Frame index to draw
            position: (x, y) position for hip center
            scale: Scale factor for skeleton size
            alpha: Transparency (0-1)
        """
        # Get skeleton points
        skeleton = self.loader.get_skeleton_frame(frame_idx, scale, position)
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 in skeleton and idx2 in skeleton:
                pt1 = skeleton[idx1]
                pt2 = skeleton[idx2]
                
                # Skip if points are at origin
                if pt1 == (0, 0) or pt2 == (0, 0):
                    continue
                    
                cv2.line(overlay, pt1, pt2, self.colors['reference'], 2, cv2.LINE_AA)
        
        # Draw joints
        for idx, pt in skeleton.items():
            if pt != (0, 0):
                cv2.circle(overlay, pt, 4, self.colors['reference'], -1)
        
        # Blend with transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add frame info
        cv2.putText(frame, f"Ref Frame: {frame_idx}", (position[0], position[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['reference'], 1)
        
        return frame
    
    def draw_side_by_side(self, frame, user_landmarks, ref_frame_idx, 
                          user_color=(0, 255, 0)):
        """
        Draw user and reference skeletons side by side
        
        Args:
            frame: Image frame
            user_landmarks: MediaPipe user landmarks
            ref_frame_idx: Reference frame index
            user_color: Color for user skeleton
        """
        h, w = frame.shape[:2]
        
        # Split frame into left and right halves
        left_frame = frame[:, :w//2]
        right_frame = frame[:, w//2:]
        
        # Draw user on left side
        if user_landmarks:
            self._draw_mediapipe_skeleton(left_frame, user_landmarks, user_color)
        
        # Draw reference on right side (scaled to fit)
        ref_skeleton = self.loader.get_skeleton_frame(
            ref_frame_idx, 
            scale=min(h, w//2) * 0.8,  # Scale to fit
            offset=(w//2 + 50, 50)
        )
        self._draw_skeleton_dict(right_frame, ref_skeleton, self.colors['reference'])
        
        # Add labels
        cv2.putText(frame, "YOU", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, user_color, 2)
        cv2.putText(frame, "INSTRUCTOR", (w//2 + 50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['reference'], 2)
        
        # Draw divider
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        return frame
    
    def _draw_mediapipe_skeleton(self, frame, landmarks, color):
        """Draw MediaPipe skeleton"""
        h, w = frame.shape[:2]
        
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 < len(landmarks.landmark) and idx2 < len(landmarks.landmark):
                lm1 = landmarks.landmark[idx1]
                lm2 = landmarks.landmark[idx2]
                
                if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                    x1, y1 = int(lm1.x * w), int(lm1.y * h)
                    x2, y2 = int(lm2.x * w), int(lm2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        for lm in landmarks.landmark:
            if lm.visibility > 0.5:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, color, -1)
    
    def _draw_skeleton_dict(self, frame, skeleton, color):
        """Draw skeleton from dictionary of points"""
        # Draw connections
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 in skeleton and idx2 in skeleton:
                pt1 = skeleton[idx1]
                pt2 = skeleton[idx2]
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
        
        # Draw joints
        for pt in skeleton.values():
            if pt != (0, 0):
                cv2.circle(frame, pt, 4, color, -1)
                
class JSONSkeletonViewer:
    """Interactive viewer for motion capture JSON data"""
    
    def __init__(self, json_path):
        self.loader = SkeletonLoader(json_path)
        self.drawer = JSONSkeletonDrawer(self.loader)
        self.current_frame = 0
        self.playing = False
        self.show_quality = True
        
    def run(self):
        """Run interactive viewer"""
        cv2.namedWindow("JSON Skeleton Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JSON Skeleton Viewer", 1200, 800)
        
        # Create a blank canvas
        canvas_height = 800
        canvas_width = 1200
        
        while True:
            # Create blank frame
            frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Draw skeleton
            frame = self.drawer.draw_reference_skeleton(
                frame, 
                self.current_frame,
                position=(canvas_width//2, canvas_height//2),
                scale=300,
                alpha=0.8
            )
            
            # Draw quality indicator if available
            if self.show_quality and self.current_frame < len(self.loader.quality):
                quality = self.loader.quality[self.current_frame]
                color = (0, 255, 0) if quality > 0.8 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
                cv2.putText(frame, f"Quality: {quality:.2f}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw progress bar
            progress = self.current_frame / max(1, len(self.loader.ref_norm_xy) - 1)
            bar_x, bar_y = 100, canvas_height - 50
            bar_w, bar_h = 1000, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + int(bar_w * progress), bar_y + bar_h), 
                         (0, 255, 0), -1)
            
            # Draw frame info
            cv2.putText(frame, f"Frame: {self.current_frame}/{len(self.loader.ref_norm_xy)-1}", 
                       (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw segment info
            if self.loader.segments:
                current_seg = self._get_current_segment()
                cv2.putText(frame, f"Segment: {current_seg + 1}/{len(self.loader.segments)}", 
                           (bar_x + 300, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw controls
            controls = "SPACE: Play/Pause | ← →: Previous/Next Frame | Q: Quit | Q: Toggle Quality"
            cv2.putText(frame, controls, (50, canvas_height - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Update frame if playing
            if self.playing:
                self.current_frame = (self.current_frame + 1) % len(self.loader.ref_norm_xy)
            
            cv2.imshow("JSON Skeleton Viewer", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30 if self.playing else 100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == ord('q'):
                self.show_quality = not self.show_quality
            elif key == 81:  # Left arrow
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == 83:  # Right arrow
                self.current_frame = min(len(self.loader.ref_norm_xy) - 1, self.current_frame + 1)
                self.playing = False
        
        cv2.destroyAllWindows()
    
    def _get_current_segment(self):
        """Find which segment contains current frame"""
        for i, seg in enumerate(self.loader.segments):
            if seg["start"] <= self.current_frame <= seg["end"]:
                return i
        return 0    