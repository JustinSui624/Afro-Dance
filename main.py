from pathlib import Path
from SLoader import JSONSkeletonViewer


def main():
    repo_root = Path(__file__).resolve().parent
    json_path = repo_root / "data" / "references" / "instructor_reference.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Run: python extract_reference.py")
        return

    viewer = JSONSkeletonViewer(str(json_path))

    print("=== JSON Skeleton Viewer ===")
    print(f"Total frames: {len(viewer.loader.ref_norm_xy)}")
    print(f"Segments: {len(viewer.loader.segments)}")
    print(f"FPS: {viewer.loader.fps}")
    print("\nControls:")
    print("  SPACE - Play/Pause")
    print("  LEFT/RIGHT ARROW - Previous/Next frame")
    print("  T     - Toggle quality display")
    print("  Q     - Quit")
    print("-" * 30)

    viewer.run()


if __name__ == "__main__":
    main()
