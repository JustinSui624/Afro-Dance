import SLoader.py as JSONSkeletonViewer
def main():
    json_path = "data/references/instructor_reference.json"
    
    # Check if file exists
    import os
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        print("Please run extract_reference.py first")
        return
    
    # Create viewer
    viewer = JSONSkeletonViewer(json_path)
    
    print("=== JSON Skeleton Viewer ===")
    print(f"Total frames: {len(viewer.loader.ref_norm_xy)}")
    print(f"Segments: {len(viewer.loader.segments)}")
    print(f"FPS: {viewer.loader.fps}")
    print("\nControls:")
    print("  SPACE - Play/Pause")
    print("  ←/→   - Previous/Next frame")
    print("  Q     - Toggle quality display")
    print("  Q     - Quit")
    print("-" * 30)
    
    # Run viewer
    viewer.run()


if __name__ == "__main__":
    main()