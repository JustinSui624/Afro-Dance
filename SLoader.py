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