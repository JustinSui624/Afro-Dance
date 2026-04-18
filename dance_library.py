from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def get_library_base_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or get_repo_root()) / "AfroDanceLearnPose" / "data"


def get_data_dir(repo_root: Optional[Path] = None) -> Path:
    # Active working data used by extraction scripts
    return (repo_root or get_repo_root()) / "data"


def get_dances_dir(repo_root: Optional[Path] = None) -> Path:
    return get_library_base_dir(repo_root) / "dances"


def get_current_dance_path(repo_root: Optional[Path] = None) -> Path:
    return get_library_base_dir(repo_root) / "current_dance.json"


def ensure_library_structure(repo_root: Optional[Path] = None) -> None:
    repo_root = repo_root or get_repo_root()

    # Root active working folders
    (repo_root / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "references").mkdir(parents=True, exist_ok=True)

    # Library storage folders inside AfroDanceLearnPose/data
    base = get_library_base_dir(repo_root)
    (base).mkdir(parents=True, exist_ok=True)
    (base / "dances").mkdir(parents=True, exist_ok=True)


def default_metadata(dance_id: str) -> Dict[str, Any]:
    label = dance_id.replace("_", " ").title()
    return {
        "id": dance_id,
        "name": label,
        "region": "African-inspired",
        "difficulty": "Beginner",
        "description": f"{label} available for live training and analysis.",
    }


def _load_metadata(metadata_path: Path, dance_id: str) -> Dict[str, Any]:
    data = default_metadata(dance_id)
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    data["id"] = dance_id
    return data


def list_dances(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    repo_root = repo_root or get_repo_root()
    ensure_library_structure(repo_root)
    dances_dir = get_dances_dir(repo_root)
    dances: List[Dict[str, Any]] = []

    for dance_dir in sorted([p for p in dances_dir.iterdir() if p.is_dir()]):
        dance_id = dance_dir.name
        video_path = dance_dir / "instructor.mp4"
        reference_path = dance_dir / "reference.json"
        metadata_path = dance_dir / "metadata.json"
        metadata = _load_metadata(metadata_path, dance_id)

        metadata.update(
            {
                "folder": dance_dir,
                "video_path": video_path,
                "reference_path": reference_path,
                "metadata_path": metadata_path,
                "video_exists": video_path.exists(),
                "reference_exists": reference_path.exists(),
            }
        )
        metadata["ready"] = metadata["video_exists"] and metadata["reference_exists"]
        dances.append(metadata)

    return dances


def get_selected_dance_id(repo_root: Optional[Path] = None) -> Optional[str]:
    repo_root = repo_root or get_repo_root()
    dances = list_dances(repo_root)
    if not dances:
        return None

    current_path = get_current_dance_path(repo_root)
    if current_path.exists():
        try:
            loaded = json.loads(current_path.read_text(encoding="utf-8"))
            selected = loaded.get("selected_dance") if isinstance(loaded, dict) else None
            if selected and any(d["id"] == selected for d in dances):
                return selected
        except Exception:
            pass

    return dances[0]["id"]


def save_selected_dance(repo_root: Optional[Path], dance_id: str) -> None:
    repo_root = repo_root or get_repo_root()
    ensure_library_structure(repo_root)
    get_current_dance_path(repo_root).write_text(
        json.dumps({"selected_dance": dance_id}, indent=2),
        encoding="utf-8",
    )


def get_selected_dance(repo_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    repo_root = repo_root or get_repo_root()
    selected_id = get_selected_dance_id(repo_root)
    if selected_id is None:
        return None

    for dance in list_dances(repo_root):
        if dance["id"] == selected_id:
            return dance

    return None


def copy_into_selected_dance(repo_root: Optional[Path], source_path: Path) -> Dict[str, Any]:
    repo_root = repo_root or get_repo_root()
    selected = get_selected_dance(repo_root)
    if selected is None:
        raise RuntimeError("No dance folders were found in AfroDanceLearnPose/data/dances.")

    selected["folder"].mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, selected["video_path"])
    return selected


def copy_prototype_to_selected(repo_root: Optional[Path]) -> Dict[str, Any]:
    repo_root = repo_root or get_repo_root()
    prototype_path = repo_root / "AfroDanceLearnPose" / "data" / "instructor.mp4"
    if not prototype_path.exists():
        raise FileNotFoundError(f"Prototype video not found: {prototype_path}")
    return copy_into_selected_dance(repo_root, prototype_path)


def prepare_selected_dance_for_extraction(repo_root: Optional[Path]) -> Dict[str, Any]:
    repo_root = repo_root or get_repo_root()
    selected = get_selected_dance(repo_root)
    if selected is None:
        raise RuntimeError("No dance folders were found in AfroDanceLearnPose/data/dances.")
    if not selected["video_exists"]:
        raise FileNotFoundError(f"Selected dance video is missing: {selected['video_path']}")

    ensure_library_structure(repo_root)
    active_video = get_data_dir(repo_root) / "instructor.mp4"
    shutil.copy2(selected["video_path"], active_video)
    return selected


def store_generated_reference_for_selected(repo_root: Optional[Path]) -> Dict[str, Any]:
    repo_root = repo_root or get_repo_root()
    selected = get_selected_dance(repo_root)
    if selected is None:
        raise RuntimeError("No dance folders were found in AfroDanceLearnPose/data/dances.")

    generated = get_data_dir(repo_root) / "references" / "instructor_reference.json"
    if not generated.exists():
        raise FileNotFoundError(f"Generated reference not found: {generated}")

    shutil.copy2(generated, selected["reference_path"])
    return selected
