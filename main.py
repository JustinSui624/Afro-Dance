from pathlib import Path

from SLoader import JSONSkeletonViewer
from dance_library import ensure_library_structure, get_selected_dance


def main():
    repo_root = Path(__file__).resolve().parent
    ensure_library_structure(repo_root)

    selected = get_selected_dance(repo_root)
    if selected is None:
        print("Error: no dances were found in data/dances.")
        print("Create at least one dance folder with instructor.mp4 and reference.json.")
        return

    json_path = selected["reference_path"]

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Generate reference data for the selected dance from the dashboard first.")
        return

    viewer = JSONSkeletonViewer(str(json_path))

    print("=== JSON Skeleton Viewer ===")
    print(f"Dance: {selected['name']}")
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
