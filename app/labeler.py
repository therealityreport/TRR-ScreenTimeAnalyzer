"""Streamlit labeler app for assigning harvested face tracks to people."""

from __future__ import annotations

import argparse
import base64
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, MutableSet, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.lib import assignments as assign_lib
from app.lib import data as data_lib
from app.lib import suggestions as suggest_lib
from app.lib import thumbnails as thumb_lib


def parse_cli_args(_) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--harvest-dir", type=Path, default=None, help="Default harvest directory to open")
    parser.add_argument("--video", type=Path, default=None, help="Video file for the selected harvest")
    parser.add_argument(
        "--facebank-dir",
        type=Path,
        default=Path("data/facebank"),
        help="Root directory containing labeled face crops",
    )
    parser.add_argument(
        "--thumbnails-dir",
        type=Path,
        default=Path(".thumbnails"),
        help="Cache directory for generated thumbnails",
    )
    parser.add_argument(
        "--arcface-model",
        type=str,
        default=None,
        help="Optional ArcFace model path/name for suggestions",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        type=str,
        default=None,
        help="Optional InsightFace providers (e.g., CoreMLExecutionProvider CPUExecutionProvider)",
    )
    args, _ = parser.parse_known_args()
    return args


CLI_ARGS = parse_cli_args(sys.argv[1:])
HARVEST_ROOT = Path("data/harvest")
ASSIGNMENTS_LOG = Path("data/facebank/assignments.jsonl")
APP_CWD = Path.cwd()


st.set_page_config(page_title="Face Labeler", layout="wide")


@st.cache_data(show_spinner=False)
def list_harvests_cached(root: str) -> List[str]:
    return data_lib.list_harvest_stems(Path(root))


@st.cache_data(show_spinner=False)
def load_manifest_cached(harvest_path: str) -> pd.DataFrame:
    return data_lib.load_manifest(Path(harvest_path))


@st.cache_data(show_spinner=False)
def load_samples_cached(harvest_path: str) -> pd.DataFrame:
    return data_lib.load_samples(Path(harvest_path))


@st.cache_data(show_spinner=False)
def load_assignments_cached(log_path: str):
    return data_lib.load_assignments(Path(log_path))


@st.cache_data(show_spinner=False)
def list_person_labels(facebank_dir: str) -> List[str]:
    root = Path(facebank_dir)
    if not root.exists():
        return []

    labels: List[str] = []
    for path in root.iterdir():
        if path.is_dir():
            labels.append(path.name)

    return sorted(labels)


@st.cache_data(show_spinner=False)
def load_clusters_cached(harvest_path: str):
    return data_lib.load_clusters(Path(harvest_path))


@st.cache_resource(show_spinner=False)
def get_embedder(model_path: Optional[str], providers: Optional[Tuple[str, ...]]):
    return suggest_lib.create_embedder(model_path=model_path, providers=providers)


@st.cache_data(show_spinner=False)
def load_facebank_cached(parquet_path: str):
    return suggest_lib.load_facebank_embeddings(Path(parquet_path))


@st.cache_data(show_spinner=False)
def embed_sample_cached(image_path: str, model_path: Optional[str], providers: Optional[Tuple[str, ...]]):
    embedder = get_embedder(model_path, providers)
    return suggest_lib.embed_sample(embedder, Path(image_path))


def current_shortcut() -> Optional[str]:
    params = st.query_params
    value = params.get("shortcut")
    if not value:
        return None
    shortcut = value if isinstance(value, str) else value[-1]
    # Remove the shortcut param
    new_params = {k: v for k, v in params.items() if k != "shortcut"}
    st.query_params.clear()
    st.query_params.update(new_params)
    return shortcut


def consume_selected_sample() -> Optional[str]:
    params = st.query_params
    value = params.get("select_sample")
    if not value:
        return None
    selection = value if isinstance(value, str) else value[-1]
    new_params = {k: v for k, v in params.items() if k != "select_sample"}
    st.query_params.clear()
    st.query_params.update(new_params)
    return selection


def register_shortcuts() -> None:
    components.html(
        """
        <script>
        (function() {
            const allowed = {"a": "assign", "A": "assign", "n": "next", "N": "next", " ": "toggle"};
            if (window.parent) {
                window.parent.document.addEventListener("keydown", function(event) {
                    // Don't intercept if user is typing in an input field
                    const target = event.target || event.srcElement;
                    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
                        return;
                    }
                    
                    if (!(event.key in allowed)) {
                        return;
                    }
                    event.preventDefault();
                    const payload = {isStreamlitMessage: true, type: "streamlit:setQueryParams", queryParams: {shortcut: allowed[event.key]}};
                    window.parent.postMessage(payload, "*");
                }, {passive: false});
            }
        })();
        </script>
        """,
        height=0,
    )


def ensure_active_track(track_ids: List[int]) -> Optional[int]:
    if not track_ids:
        st.session_state.pop("active_track_id", None)
        return None
    active = st.session_state.get("active_track_id")
    if active in track_ids:
        return active
    st.session_state["active_track_id"] = track_ids[0]
    return st.session_state["active_track_id"]


def format_badge(label: str, color: str) -> str:
    return f"<span style='background-color:{color}; color:white; padding:2px 8px; border-radius:6px; font-size:0.75rem;'>{label}</span>"


def render_track_card(row: pd.Series, include_debug: bool) -> None:
    track_id = int(row["track_id"])
    byte_track = int(row["byte_track_id"])
    picked = int(row.get("picked_count", 0))
    total = int(row.get("sample_count", 0))
    quality = row.get("median_quality")
    frontalness = row.get("mean_frontalness")
    sharpness = row.get("mean_sharpness")
    total_frames = row.get("total_frames")
    duration_ms = row.get("duration_ms")

    status = data_lib.assignment_status(row)
    badge_color = data_lib.status_badge_color(status)

    cols = st.columns([4, 3, 3, 2])
    with cols[0]:
        active_track = st.session_state.get("active_track_id")
        label = f"Track {track_id:04d} • Byte {byte_track:04d}"
        if active_track == track_id:
            label = f"▶ {label}"
        if st.button(label, key=f"track_btn_{track_id}"):
            st.session_state["active_track_id"] = track_id
    with cols[1]:
        st.markdown(format_badge(status, badge_color), unsafe_allow_html=True)
        st.caption(f"{picked} picked / {total} total")
    with cols[2]:
        stats = []
        if quality is not None and not math.isnan(quality):
            stats.append(f"Q̃ {quality:.2f}")
        if frontalness is not None and not math.isnan(frontalness):
            stats.append(f"F̄ {frontalness:.2f}")
        if sharpness is not None and not math.isnan(sharpness):
            stats.append(f"S̄ {sharpness:.0f}")
        st.caption(" • ".join(stats) if stats else "No metrics")
        if include_debug:
            st.caption("Debug ON")
    with cols[3]:
        lines = []
        if total_frames:
            frames = int(total_frames)
            lines.append(f"{frames} frames")
        if duration_ms:
            lines.append(f"{duration_ms/1000.0:.1f}s span")
        st.caption("\n".join(lines) if lines else "")


def gather_selected_samples(track_id: int, samples: pd.DataFrame) -> List[str]:
    selected: List[str] = []
    for idx, row in samples.iterrows():
        path = str(row["path"])
        key = f"sample_cb_{track_id}_{idx}"
        value = st.session_state.get(key)
        if value and not str(row.get("assigned_person_label", "")).strip():
            selected.append(path)
    return selected


def set_all_samples(track_id: int, samples: pd.DataFrame, checked: bool) -> None:
    for idx in samples.index:
        row = samples.loc[idx]
        assigned_label = str(row.get("assigned_person_label", "")).strip()
        key = f"sample_cb_{track_id}_{idx}"
        if assigned_label:
            st.session_state[key] = False
        else:
            st.session_state[key] = checked


def initialize_sample_checkbox(track_id: int, index: int, default: bool) -> bool:
    key = f"sample_cb_{track_id}_{index}"
    if key not in st.session_state:
        st.session_state[key] = default
    return bool(st.session_state[key])


def run_subprocess(command: Sequence[str], description: str) -> Tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(APP_CWD))
    with st.spinner(f"{description}..."):
        try:
            completed = subprocess.run(
                command,
                cwd=str(APP_CWD),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return 1, f"Command failed: {exc}"
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode, output.strip()


def display_command_output(return_code: int, output: str, success_message: str, failure_message: str) -> None:
    if return_code == 0:
        st.success(success_message)
        if output:
            with st.expander("Command output"):
                st.code(output)
    else:
        st.error(failure_message)
        if output:
            st.code(output)


def render_track_montage_controls(track_dir: Optional[Path], track_id: int) -> None:
    if track_dir is None or not track_dir.exists():
        return
    if st.button("Generate montage", key=f"montage_btn_{track_id}"):
        cmd = [sys.executable, "scripts/make_montage.py", str(track_dir)]
        code, output = run_subprocess(cmd, "Building montage")
        render_msg = (
            f"Montage written to {track_dir/'montage.jpg'}" if code == 0 else "Montage generation failed"
        )
        display_command_output(code, output, render_msg, "Montage generation failed")


def render_overlay_controls(harvest_dir: Path, video_path: Optional[Path], stem: str, track_id: int) -> None:
    debug_path = harvest_dir / "harvest_debug.json"
    if not debug_path.exists():
        st.info("No harvest_debug.json found; overlay rendering disabled.")
        return
    if video_path is None or not video_path.exists():
        st.info("Video path not provided; overlay rendering disabled.")
        return
    output_path = Path("data/outputs") / stem / f"{stem}_track_{track_id:04d}_overlay.mp4"
    with st.expander("Overlay snippet"):
        col_a, col_b = st.columns(2)
        with col_a:
            start_sec = st.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5, key=f"overlay_start_{track_id}")
        with col_b:
            end_sec = st.number_input(
                "End (s)",
                min_value=0.0,
                value=max(0.5, start_sec + 2.0),
                step=0.5,
                key=f"overlay_end_{track_id}",
            )
        show_faces = st.checkbox("Show faces", value=True, key=f"overlay_faces_{track_id}")
        show_person = st.checkbox("Show persons", value=True, key=f"overlay_person_{track_id}")
        show_reason = st.checkbox("Show reasons", value=False, key=f"overlay_reason_{track_id}")
        show_iou = st.checkbox("Show IoU", value=False, key=f"overlay_iou_{track_id}")
        if st.button("Render overlay snippet", key=f"overlay_btn_{track_id}"):
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/make_overlays.py",
                str(video_path),
                "--harvest-dir",
                str(harvest_dir),
                "--output",
                str(output_path),
                "--start-sec",
                str(start_sec),
                "--end-sec",
                str(end_sec),
            ]
            if show_faces:
                cmd.append("--show-faces")
            if show_person:
                cmd.append("--show-person")
            if show_reason:
                cmd.append("--show-reason")
            if show_iou:
                cmd.append("--show-iou")
            code, output = run_subprocess(cmd, "Rendering overlay snippet")
            display_command_output(
                code,
                output,
                f"Overlay written to {output_path}",
                "Overlay rendering failed",
            )


def safe_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def canonical_sample_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve())
    except (OSError, RuntimeError):
        return str(candidate.expanduser().absolute())


def update_sample_csv_record(
    harvest_dir: Path,
    track_id: int,
    frame: Optional[int],
    original_path: Path,
    action: str,
    new_path: Optional[Path] = None,
) -> bool:
    csv_path = harvest_dir / "selected_samples.csv"
    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:  # pragma: no cover - corrupt CSV handling
        return False

    if df.empty:
        return False

    track_series = pd.to_numeric(df.get("track_id"), errors="coerce")
    mask = track_series == track_id if track_series is not None else pd.Series([False] * len(df))

    if "frame" in df.columns and frame is not None:
        frame_series = pd.to_numeric(df["frame"], errors="coerce")
        mask &= frame_series == frame

    if not mask.any() and "path" in df.columns:
        normalized_target = canonical_sample_path(original_path)
        normalized_series = df["path"].astype(str).map(canonical_sample_path)
        mask = normalized_series == normalized_target

    if not mask.any() and "path" in df.columns:
        mask = df["path"].astype(str).str.endswith(original_path.name)

    if not mask.any():
        return False

    if action == "delete":
        df = df.loc[~mask].copy()
    else:
        if "picked" not in df.columns:
            df["picked"] = True
        if action == "unassign":
            df.loc[mask, "picked"] = False
            df.loc[mask, "reason"] = "manual_unassigned"
        if (action in {"unassign", "path_update"}) and new_path is not None:
            df.loc[mask, "path"] = str(new_path)

    df.to_csv(csv_path, index=False)
    return True


def move_sample_to_unassigned_dir(harvest_dir: Path, track_id: int, sample_path: Path) -> Optional[Path]:
    destination_dir = harvest_dir / "unassigned" / f"track_{track_id:04d}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists():
        return None
    dest_path = destination_dir / sample_path.name
    counter = 1
    while dest_path.exists():
        dest_path = destination_dir / f"{sample_path.stem}_{counter:02d}{sample_path.suffix}"
        counter += 1
    shutil.move(str(sample_path), str(dest_path))
    return dest_path


def delete_sample_file(sample_path: Path) -> bool:
    try:
        sample_path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


def find_alternate_sample_path(harvest_dir: Path, sample_path: Path) -> Optional[Path]:
    if sample_path.exists():
        return sample_path

    parts = list(sample_path.parts)
    for idx, part in enumerate(parts):
        if part.startswith("track_"):
            candidate = harvest_dir / Path(*parts[idx:])
            if candidate.exists():
                return candidate

    matches = list((harvest_dir).rglob(sample_path.name))
    if matches:
        return matches[0]
    return None


def collect_pending_sample_actions(track_id: int, track_samples: pd.DataFrame) -> List[Tuple[int, str, pd.Series]]:
    actions: List[Tuple[int, str, pd.Series]] = []
    for idx, row in track_samples.iterrows():
        key = f"sample_action_{track_id}_{idx}"
        action = st.session_state.get(key, "Keep")
        if action != "Keep":
            if str(row.get("assigned_person_label", "")).strip():
                st.session_state[key] = "Keep"
                continue
            actions.append((idx, action, row))
    return actions


def process_sample_actions(
    harvest_dir: Path,
    track_id: int,
    actions: List[Tuple[int, str, pd.Series]],
    *,
    stem: str,
    byte_track_id: int,
    facebank_dir: Path,
    assignments_log: Path,
) -> List[str]:
    messages: List[str] = []
    assignments_changed = False
    _, assignment_index = load_assignments_cached(str(assignments_log))
    existing_sources: MutableSet[str] = set()
    for details in assignment_index.values():
        existing_sources.update(details.sources)

    for idx, action, row in actions:
        sample_path = Path(str(row.get("path", "")))
        frame_idx = safe_int(row.get("frame"))
        original_path = sample_path
        if not sample_path.exists():
            fallback = find_alternate_sample_path(harvest_dir, sample_path)
            if fallback is not None:
                sample_path = fallback

        if action == "Move to UNASSIGNED":
            destination = move_sample_to_unassigned_dir(harvest_dir, track_id, sample_path)
            if destination is None:
                messages.append(f"Unable to relocate {original_path.name} (file missing)")
            else:
                update_sample_csv_record(harvest_dir, track_id, frame_idx, original_path, "unassign", destination)
                messages.append(f"Moved {original_path.name} to UNASSIGNED")
        elif action == "Delete":
            if delete_sample_file(sample_path):
                update_sample_csv_record(harvest_dir, track_id, frame_idx, original_path, "delete")
                messages.append(f"Deleted {original_path.name}")
            else:
                messages.append(f"Unable to delete {original_path.name}")
        elif action.startswith("Assign → "):
            person_label = action.split("Assign → ", 1)[1].strip()
            if person_label:
                result = assign_lib.assign_samples(
                    [sample_path],
                    stem=stem,
                    harvest_id=track_id,
                    byte_track_id=byte_track_id,
                    person_label=person_label,
                    facebank_dir=facebank_dir,
                    assignments_log=assignments_log,
                    existing_sources=existing_sources,
                )
                if result.copied:
                    assignments_changed = True
                    for source, _ in result.copied:
                        existing_sources.add(source)
                    messages.append(f"Assigned {original_path.name} to {person_label}")
                else:
                    messages.append(f"No files copied for {original_path.name}")

        st.session_state.pop(f"sample_action_{track_id}_{idx}", None)

    if messages:
        load_samples_cached.clear()
    if assignments_changed:
        load_assignments_cached.clear()
        list_person_labels.clear()
    return messages


def render_missing_samples_panel(
    stem: str,
    harvest_dir: Path,
    track_id: int,
    track_samples: pd.DataFrame,
    target_label: Optional[str],
) -> None:
    missing_rows = []
    for idx, row in track_samples.iterrows():
        sample_path = Path(str(row.get("path", "")))
        if not sample_path.exists():
            missing_rows.append((idx, row))

    if not missing_rows:
        return

    with st.expander("Missing / offline samples", expanded=False):
        st.write(
            "The following images were marked as rejected or could not be located on disk. "
            "Use the tools below to locate replacements and include them in assignments."
        )
        for idx, row in missing_rows:
            sample_path = Path(str(row.get("path", "")))
            original_canonical = canonical_sample_path(sample_path)
            frame_idx = safe_int(row.get("frame"))
            cols = st.columns([3, 2])
            with cols[0]:
                st.code(str(sample_path))
                if frame_idx is not None:
                    st.caption(f"Track {track_id:04d} • Frame {frame_idx:06d}")
            with cols[1]:
                locate_clicked = st.button(
                    "Locate on disk",
                    key=f"missing_locate_{track_id}_{idx}",
                )
                if locate_clicked:
                    candidate = find_alternate_sample_path(harvest_dir, sample_path)
                    if candidate is None:
                        st.error("No matching file found inside the harvest directory.")
                    else:
                        update_sample_csv_record(harvest_dir, track_id, frame_idx, sample_path, "path_update", candidate)
                        st.success(f"Linked to {candidate.name}")
                        load_samples_cached.clear()
                        st.rerun()

                if target_label:
                    assign_disabled = not sample_path.exists()
                    assign_label = f"Assign to {target_label}" if not assign_disabled else "Assign once located"
                    assign_clicked = st.button(
                        assign_label,
                        key=f"missing_assign_{track_id}_{idx}",
                        disabled=assign_disabled,
                        help="Locate the file before assigning." if assign_disabled else "Copy directly to the selected person.",
                    )
                    if assign_clicked and sample_path.exists():
                        st.session_state["single_assign_job"] = {
                            "stem": stem,
                            "track_id": track_id,
                            "byte_track_id": safe_int(row.get("byte_track_id")) or track_id,
                            "path": str(sample_path),
                            "label": target_label,
                        }
                        st.rerun()


def render_facebank_browser(
    facebank_dir: Path,
    *,
    key_prefix: str = "",
    labels: Optional[List[str]] = None,
) -> None:
    if labels is None:
        labels = list_person_labels(str(facebank_dir))
    if not labels:
        st.info("No facebank entries found yet. Assign tracks to build the library.")
        return

    person_key = f"{key_prefix}facebank_browser_person"
    slider_key = f"{key_prefix}facebank_browser_max"
    selected_person = st.selectbox("Person", labels, key=person_key)
    max_images = st.slider(
        "Images to preview",
        min_value=8,
        max_value=120,
        value=32,
        step=4,
        key=slider_key,
    )

    st.caption("Preview archived samples for the selected person to audit assignments.")

    if not selected_person:
        return

    person_dir = facebank_dir / selected_person
    if not person_dir.exists():
        st.warning(f"No directory found for {selected_person}.")
        return

    images = [
        p
        for p in sorted(person_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        st.info("No images stored for this person yet.")
        return

    total = min(len(images), max_images)
    cols_per_row = 6
    rows = math.ceil(total / cols_per_row)
    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row_idx * cols_per_row + col_idx
            if img_idx >= total:
                break
            img_path = images[img_idx]
            cols[col_idx].image(str(img_path), caption=img_path.name, use_container_width=True)
            if cols[col_idx].button(
                "🗑️ Remove",
                key=f"{key_prefix}remove_{selected_person}_{img_idx}",
                help="Remove from facebank and restore to original track",
            ):
                removal = assign_lib.unassign_sample(img_path, ASSIGNMENTS_LOG, PROJECT_ROOT)
                source_path: Optional[Path] = None
                person_label = selected_person
                if removal is not None:
                    stem, source_str, person = removal
                    person_label = person or selected_person
                    if source_str:
                        tentative = Path(source_str)
                        if not tentative.is_absolute():
                            tentative = (PROJECT_ROOT / tentative).resolve()
                        source_path = tentative
                if img_path.exists():
                    if source_path is not None and not source_path.exists():
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_path, source_path)
                    try:
                        img_path.unlink()
                    except OSError:
                        pass
                load_assignments_cached.clear()
                list_person_labels.clear()
                load_samples_cached.clear()
                if removal is None:
                    st.warning("Removed image but did not find a matching assignment entry.")
                else:
                    st.success(f"Returned image to harvest and removed from {person_label}.")
                st.rerun()


def render_facebank_view(facebank_dir: Path) -> None:
    st.header("Facebank Library")

    labels = list_person_labels(str(facebank_dir))
    if not labels:
        st.info("No facebank entries found yet. Assign tracks to build the library.")
        return

    stats: List[Dict[str, object]] = []
    for label in labels:
        person_dir = facebank_dir / label
        if not person_dir.exists():
            continue
        count = sum(1 for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        stats.append({"Person": label, "Images": count})

    if stats:
        stats_df = pd.DataFrame(stats).sort_values("Person").reset_index(drop=True)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.caption("No images stored yet.")

    st.markdown("### Browse samples")
    render_facebank_browser(facebank_dir, key_prefix="full_", labels=labels)


def _is_valid_number(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def render_assignment_panel(
    stem: str,
    track_row: pd.Series,
    track_samples: pd.DataFrame,
    assignable_samples: pd.DataFrame,
    selected_paths: List[str],
    facebank_dir: Path,
    assignments_log: Path,
    facebank_labels: List[str],
    harvest_dir: Path,
    video_path: Optional[Path],
    samples_df: pd.DataFrame,
    summary_index: pd.DataFrame,
    cluster_id: Optional[int],
    cluster_tracks: Sequence[int],
    clusters: Dict[int, List[int]],
    available_track_ids: Sequence[int],
    active_index: int,
) -> None:
    track_id = int(track_row["track_id"])
    byte_track_id = int(track_row["byte_track_id"])

    st.subheader(f"Track {track_id:04d} • Byte {byte_track_id:04d}")

    status = data_lib.assignment_status(track_row)
    st.markdown(format_badge(status, data_lib.status_badge_color(status)), unsafe_allow_html=True)
    cluster_track_list = sorted({int(t) for t in cluster_tracks}) if cluster_tracks else []
    if cluster_id is not None and cluster_track_list:
        track_list_str = ", ".join(f"{tid:04d}" for tid in cluster_track_list)
        st.info(f"Cluster {cluster_id:03d} → {len(cluster_track_list)} track(s): {track_list_str}")

    current_cluster = cluster_id if cluster_id is not None else data_lib.UNASSIGNED_CLUSTER_ID
    with st.expander("Cluster tools", expanded=False):
        cluster_options = sorted(set(clusters.keys()) | {data_lib.UNASSIGNED_CLUSTER_ID})
        cluster_labels = {
            cid: "UNASSIGNED" if cid == data_lib.UNASSIGNED_CLUSTER_ID else f"Cluster {cid:03d}"
            for cid in cluster_options
        }
        current_index = cluster_options.index(current_cluster) if current_cluster in cluster_options else 0
        destination = st.selectbox(
            "Cluster", cluster_options, index=current_index, format_func=lambda cid: cluster_labels[cid], key=f"cluster_select_{track_id}"
        )
        move_disabled = destination == current_cluster
        if st.button("Move track", key=f"cluster_move_{track_id}", disabled=move_disabled):
            data_lib.move_track_to_cluster(harvest_dir, clusters, track_id, destination)
            load_clusters_cached.clear()
            st.success(
                f"Track {track_id:04d} moved to {cluster_labels[destination]}"
            )
            st.rerun()


    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Picked samples", value=int(track_row.get("picked_count") or 0))
    with meta_cols[1]:
        st.metric("Rejections", value=int(track_row.get("rejection_count") or 0))
    with meta_cols[2]:
        median_quality = track_row.get("median_quality")
        if _is_valid_number(median_quality):
            st.metric("Median quality", f"{median_quality:.2f}")
    with meta_cols[3]:
        total_frames = track_row.get("total_frames")
        if _is_valid_number(total_frames):
            st.metric("Frames", int(float(total_frames)))

    assigned_persons = track_row.get("assigned_persons") or ""
    default_person = assigned_persons.split(", ")[0].strip() if assigned_persons else ""

    existing_label = st.selectbox(
        "Existing person",
        options=["(Create new)"] + facebank_labels,
        index=facebank_labels.index(default_person) + 1 if default_person in facebank_labels else 0,
        key=f"existing_person_{track_id}",
    )
    
    # Use data_editor for better text input handling (no rerun issues)
    new_label = st.text_area(
        "New person name (UPPERCASE)",
        value="",
        max_chars=100,
        height=70,
        key=f"new_person_{track_id}",
        placeholder="Type name here: RINNA, KYLE, BRANDI, etc.",
        help="Enter the person's name. Use UPPERCASE letters. Spaces will be converted to underscores.",
    )

    target_label = None
    if existing_label != "(Create new)":
        target_label = existing_label
    elif new_label.strip():
        target_label = new_label.strip().upper()

    if target_label:
        st.session_state["active_target_label"] = target_label
    else:
        st.session_state.pop("active_target_label", None)

    assignable_count = len(assignable_samples)
    st.caption("Keyboard: A=Assign selected, N=Next cluster, Space=toggle focus sample.")
    assign_selected_disabled = target_label is None or not selected_paths
    assign_selected_requested = st.button(
        f"Assign {len(selected_paths)} selected",
        disabled=assign_selected_disabled,
        key=f"assign_selected_btn_{track_id}",
        type="primary",
    )
    assign_all_requested = st.button(
        f"Assign all {assignable_count} samples",
        disabled=target_label is None or assignable_count == 0,
        key=f"assign_all_btn_{track_id}",
    )
    cluster_assign_requested = False
    if cluster_track_list and target_label is not None:
        cluster_assign_requested = st.button(
            f"Assign cluster ({len(cluster_track_list)} tracks)",
            key=f"assign_cluster_btn_{track_id}",
        )

    representative_sample: Optional[Path] = None
    if selected_paths:
        representative_sample = Path(selected_paths[0])
    elif not assignable_samples.empty:
        representative_sample = Path(assignable_samples.iloc[0]["path"])
    elif not track_samples.empty:
        representative_sample = Path(track_samples.iloc[0]["path"])

    suggestions_placeholder = st.empty()
    facebank_embeddings = {}
    suggestion_error = None
    if representative_sample and (facebank_dir / "facebank.parquet").exists():
        try:
            facebank_embeddings = load_facebank_cached(str(facebank_dir / "facebank.parquet"))
        except Exception as exc:  # pragma: no cover - best-effort load
            suggestion_error = str(exc)
    if representative_sample and facebank_embeddings:
        try:
            embedding = embed_sample_cached(
                str(representative_sample),
                CLI_ARGS.arcface_model,
                tuple(CLI_ARGS.providers) if CLI_ARGS.providers else None,
            )
            top = suggest_lib.top_k(facebank_embeddings, embedding, k=3)
            if top:
                suggestions_placeholder.markdown(
                    "**Suggestions:** "
                    + ", ".join(f"{s.label} ({s.score:.2f})" for s in top),
                )
        except Exception as exc:  # pragma: no cover - embed errors
            suggestions_placeholder.info(f"Suggestions unavailable: {exc}")
    elif suggestion_error:
        suggestions_placeholder.info(f"Suggestions unavailable: {suggestion_error}")
    else:
        suggestions_placeholder.info("Suggestions unavailable (build facebank first).")

    # Handle keyboard shortcut for assign selected
    if not assign_selected_requested:
        assign_selected_requested = (
            st.session_state.pop(f"shortcut_assign_{track_id}", False) and target_label is not None
        )

    paths_to_assign: List[str] = []
    if assign_all_requested and target_label is not None:
        paths_to_assign = [str(Path(p)) for p in assignable_samples["path"].tolist()]
    elif assign_selected_requested and target_label is not None:
        paths_to_assign = list(selected_paths)

    assignment_jobs: List[Tuple[int, int, List[Path]]] = []
    if paths_to_assign and target_label is not None:
        assignment_jobs.append((track_id, byte_track_id, [Path(p) for p in paths_to_assign]))
    if cluster_assign_requested and target_label is not None:
        for cluster_track in cluster_track_list:
            try:
                summary_row = summary_index.loc[cluster_track]
            except KeyError:
                continue
            cluster_rows = samples_df[samples_df["track_id"] == cluster_track]
            if "picked" in cluster_rows.columns:
                cluster_rows = cluster_rows[cluster_rows["picked"].astype(str).str.lower().isin({"1", "true", "yes"})]
            if "is_debug" in cluster_rows.columns:
                cluster_rows = cluster_rows[cluster_rows["is_debug"] != True]
            cluster_paths: List[Path] = []
            for raw_path in cluster_rows.get("path", []):
                candidate = Path(str(raw_path))
                if not candidate.is_absolute():
                    candidate = (harvest_dir / candidate).resolve()
                if candidate.exists():
                    cluster_paths.append(candidate)
            if not cluster_paths:
                continue
            cluster_byte = int(summary_row.get("byte_track_id") or cluster_track)
            assignment_jobs.append((cluster_track, cluster_byte, cluster_paths))

    if assignment_jobs and target_label is not None:
        try:
            normalized_label = assign_lib.normalize_label(target_label or "")
        except ValueError as exc:
            st.error(str(exc))
        else:
            _, index = load_assignments_cached(str(assignments_log))
            existing_sources: set[str] = set()
            for details in index.values():
                existing_sources.update(details.sources)

            total_copied = 0
            for job_track_id, job_byte_id, job_paths in assignment_jobs:
                result = assign_lib.assign_samples(
                    job_paths,
                    stem=stem,
                    harvest_id=job_track_id,
                    byte_track_id=job_byte_id,
                    person_label=normalized_label,
                    facebank_dir=facebank_dir,
                    assignments_log=assignments_log,
                    existing_sources=existing_sources,
                )
                total_copied += len(result.copied)
            if total_copied:
                st.success(f"Copied {total_copied} file(s) to {normalized_label}")
                load_assignments_cached.clear()
                list_person_labels.clear()
                advance_after_assignment = (
                    assign_all_requested or cluster_assign_requested or assign_selected_requested
                )
                if advance_after_assignment:
                    available_ids = list(available_track_ids)
                    if available_ids:
                        if len(available_ids) == 1:
                            st.session_state["active_track_id"] = available_ids[0]
                        else:
                            next_idx = (active_index + 1) % len(available_ids)
                            st.session_state["active_track_id"] = available_ids[next_idx]
                st.rerun()
            else:
                st.info("No files copied.")

    render_track_montage_controls(harvest_dir, track_id)
    render_overlay_controls(harvest_dir, CLI_ARGS.video, stem, track_id)


def render_thumbnail_grid(
    stem: str,
    track_id: int,
    byte_track_id: int,
    track_samples: pd.DataFrame,
    thumbnails_dir: Path,
    harvest_dir: Path,
    assignment_info: Optional[data_lib.TrackAssignment],
    target_label: Optional[str],
    facebank_labels: Sequence[str],
) -> pd.DataFrame:
    assigned_lookup: Dict[str, str] = {}
    if assignment_info:
        for source_path, person in assignment_info.source_to_person.items():
            if person:
                assigned_lookup[canonical_sample_path(source_path)] = person

    display_rows = track_samples.copy()
    display_rows["assigned_person_label"] = display_rows["path"].map(
        lambda p: assigned_lookup.get(canonical_sample_path(Path(str(p))), "")
    )
    display_rows["is_unassigned"] = display_rows["assigned_person_label"].eq("")
    display_rows = display_rows.sort_values(
        by=["is_unassigned", "assigned_person_label"],
        ascending=[False, True],
        kind="stable",
    )

    visible_rows = display_rows[display_rows["is_unassigned"]].copy()
    if visible_rows.empty:
        st.info("All samples in this track are already assigned to the facebank.")
        return visible_rows

    cols_per_row = 4
    rows = math.ceil(len(visible_rows) / cols_per_row) or 1
    focus_key = f"focused_sample_{track_id}"
    focused_idx = st.session_state.get(focus_key)
    if focused_idx is None and not visible_rows.empty:
        focused_idx = visible_rows.index[0]
        st.session_state[focus_key] = focused_idx

    action_options = ["Keep", "Move to UNASSIGNED", "Delete"]
    if facebank_labels:
        action_options.extend([f"Assign → {label}" for label in facebank_labels])

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            sample_idx = row_idx * cols_per_row + col_idx
            if sample_idx >= len(visible_rows):
                continue
            row = visible_rows.iloc[sample_idx]
            sample_path = Path(str(row.get("path", "")))
            original_canonical = canonical_sample_path(sample_path)
            checkbox_key = f"sample_cb_{track_id}_{row.name}"
            default_checked = bool(row.get("picked")) and not bool(row.get("is_debug"))
            checked = initialize_sample_checkbox(track_id, row.name, default_checked)

            is_checked = cols[col_idx].checkbox(" ", value=checked, key=checkbox_key)
            if is_checked != checked:
                st.session_state[focus_key] = row.name
                focused_idx = row.name

            caption_parts = []
            if row.get("quality") is not None and not math.isnan(row.get("quality")):
                caption_parts.append(f"Q {float(row['quality']):.2f}")
            if row.get("sharpness") is not None and not math.isnan(row.get("sharpness")):
                caption_parts.append(f"S {float(row['sharpness']):.0f}")
            if row.get("frontalness") is not None and not math.isnan(row.get("frontalness")):
                caption_parts.append(f"F {float(row['frontalness']):.2f}")
            if row.get("reason") and str(row["reason"]).lower() not in {"picked", "true"}:
                caption_parts.append(str(row["reason"]).replace("_", " ").title())
            if row.get("is_debug"):
                caption_parts.append("DEBUG")
            if row.name == focused_idx:
                caption_parts.insert(0, "★ Focus")

            assigned_person = row.get("assigned_person_label") or ""
            try:
                thumb_path = thumb_lib.ensure_thumbnail(sample_path, stem, thumbnails_dir)
            except FileNotFoundError:
                fallback = find_alternate_sample_path(harvest_dir, sample_path)
                if fallback and fallback.exists():
                    track_samples.at[row.name, "path"] = str(fallback)
                    visible_rows.at[row.name, "path"] = str(fallback)
                    update_sample_csv_record(
                        harvest_dir,
                        track_id,
                        safe_int(row.get("frame")),
                        sample_path,
                        "path_update",
                        fallback,
                    )
                    reassigned_label = assigned_lookup.pop(original_canonical, None)
                    if reassigned_label:
                        assigned_lookup[canonical_sample_path(fallback)] = reassigned_label
                        visible_rows.at[row.name, "assigned_person_label"] = reassigned_label
                        visible_rows.at[row.name, "is_unassigned"] = reassigned_label == ""
                    load_samples_cached.clear()
                    try:
                        thumb_path = thumb_lib.ensure_thumbnail(fallback, stem, thumbnails_dir)
                        sample_path = fallback
                        assigned_person = visible_rows.at[row.name, "assigned_person_label"] or ""
                    except FileNotFoundError:
                        cols[col_idx].warning(f"Missing: {sample_path}")
                        continue
                else:
                    cols[col_idx].warning(f"Missing: {sample_path}")
                    continue

            caption = ", ".join(caption_parts) if caption_parts else sample_path.name
            suffix = sample_path.suffix.lower()
            if suffix in {".png"}:
                mime_type = "image/png"
            elif suffix in {".webp"}:
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
            encoded = base64.b64encode(thumb_path.read_bytes()).decode("ascii")
            js_payload = (
                "{{isStreamlitMessage: true, type: 'streamlit:setQueryParams', "
                f"queryParams: {{select_sample: '{track_id}:{row.name}'}}}}"
            )
            image_html = (
                f'<div onclick="window.parent.postMessage({js_payload}, \'*\');" '
                'style="cursor:pointer;">'
                f'<img src="data:{mime_type};base64,{encoded}" '
                'style="width:100%; border-radius:6px;"/>'
                "</div>"
            )
            cols[col_idx].markdown(image_html, unsafe_allow_html=True)
            cols[col_idx].caption(caption)

            action_key = f"sample_action_{track_id}_{row.name}"
            if cols[col_idx].button(
                "🗑️",
                key=f"delete_btn_{track_id}_{row.name}",
                help="Queue delete for this sample",
            ):
                st.session_state[action_key] = "Delete"
            cols[col_idx].selectbox(
                "Action",
                action_options,
                key=action_key,
                label_visibility="collapsed",
            )

            if target_label:
                assign_disabled = not sample_path.exists()
                if cols[col_idx].button(
                    f"Assign → {target_label}",
                    key=f"quick_assign_{track_id}_{row.name}",
                    disabled=assign_disabled,
                ):
                    st.session_state["single_assign_job"] = {
                        "stem": stem,
                        "track_id": track_id,
                        "byte_track_id": byte_track_id,
                        "path": str(sample_path),
                        "label": target_label,
                    }
                    st.rerun()

    return visible_rows


def main() -> None:
    register_shortcuts()

    harvest_dirs = list_harvests_cached(str(HARVEST_ROOT))
    if CLI_ARGS.harvest_dir and CLI_ARGS.harvest_dir.exists():
        default_stem = CLI_ARGS.harvest_dir.name
        if default_stem not in harvest_dirs:
            harvest_dirs.append(default_stem)
    harvest_dirs = sorted(set(harvest_dirs))
    if not harvest_dirs:
        st.error("No harvest directories found under data/harvest.")
        return

    default_stem = CLI_ARGS.harvest_dir.name if CLI_ARGS.harvest_dir else harvest_dirs[0]
    default_index = harvest_dirs.index(default_stem) if default_stem in harvest_dirs else 0
    selected_stem = st.sidebar.selectbox("Harvest", options=harvest_dirs, index=default_index)
    harvest_dir = HARVEST_ROOT / selected_stem if selected_stem else HARVEST_ROOT
    if CLI_ARGS.harvest_dir and CLI_ARGS.harvest_dir.name == selected_stem:
        harvest_dir = CLI_ARGS.harvest_dir

    st.sidebar.caption(f"Video: {CLI_ARGS.video if CLI_ARGS.video else 'not supplied'}")
    if CLI_ARGS.video is None:
        st.warning("Video path not supplied; overlay and attribution tools will be limited.")

    view_mode = st.sidebar.radio("Mode", ["Label Tracks", "Facebank", "Unclustered Tracks"], index=0)
    if view_mode == "Facebank":
        render_facebank_view(CLI_ARGS.facebank_dir)
        return
    is_unclustered_mode = view_mode == "Unclustered Tracks"

    include_debug = st.sidebar.toggle("Include debug rejects", value=False)
    min_samples = st.sidebar.slider("Min picked samples", 0, 20, 0)
    min_frames = st.sidebar.slider("Min track frames", 0, 300, 0, step=5)
    quality_percentile = st.sidebar.slider("Quality percentile", 0, 100, 0)
    search_term = st.sidebar.text_input("Search track / byte id")
    manifest_df = load_manifest_cached(str(harvest_dir))
    samples_df = load_samples_cached(str(harvest_dir))
    clusters = load_clusters_cached(str(harvest_dir))
    clustered_track_ids = {int(track) for track_list in clusters.values() for track in track_list}
    if is_unclustered_mode:
        track_to_cluster = {}
        clusters = {}
    else:
        track_to_cluster = {track: cid for cid, tracks in clusters.items() for track in tracks}
    if track_to_cluster:
        samples_df = samples_df.copy()
        samples_df["cluster_id"] = samples_df["track_id"].map(track_to_cluster)

    assignments, assignment_index = load_assignments_cached(str(ASSIGNMENTS_LOG))
    pending_job = st.session_state.pop("single_assign_job", None)
    if pending_job:
        if pending_job.get("stem") and pending_job.get("stem") != selected_stem:
            st.session_state["single_assign_job"] = pending_job
        else:
            target_label = pending_job.get("label")
            if not target_label:
                st.error("Quick assignment failed: no person label selected.")
            else:
                try:
                    normalized_label = assign_lib.normalize_label(target_label)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    job_path = Path(pending_job.get("path", ""))
                    if not job_path.exists():
                        st.error(f"Quick assignment failed: missing file {job_path}")
                    else:
                        existing_sources: set[str] = set()
                        for details in assignment_index.values():
                            existing_sources.update(details.sources)
                        harvest_id = safe_int(pending_job.get("track_id")) or -1
                        job_byte_track = safe_int(pending_job.get("byte_track_id"))
                        if job_byte_track is None:
                            job_byte_track = harvest_id
                        result = assign_lib.assign_samples(
                            [job_path],
                            stem=selected_stem,
                            harvest_id=harvest_id,
                            byte_track_id=job_byte_track if job_byte_track is not None else harvest_id,
                            person_label=normalized_label,
                            facebank_dir=CLI_ARGS.facebank_dir,
                            assignments_log=ASSIGNMENTS_LOG,
                            existing_sources=existing_sources,
                        )
                        if result.copied:
                            st.success(f"Assigned {job_path.name} to {normalized_label}.")
                            load_assignments_cached.clear()
                            assignments, assignment_index = load_assignments_cached(str(ASSIGNMENTS_LOG))
                            list_person_labels.clear()
                        else:
                            st.info("No files copied for the quick assignment.")
    summary_df = data_lib.summarize_tracks(selected_stem, manifest_df, samples_df, assignment_index)
    if summary_df.empty:
        st.warning("No track summaries available.")
        return

    summary_index = summary_df.set_index("track_id", drop=False)

    if quality_percentile > 0:
        threshold = data_lib.percentile_threshold(samples_df["quality"], quality_percentile)
    else:
        threshold = None
    filtered = data_lib.filter_tracks(summary_df, min_samples, min_frames, threshold, search_term)

    if filtered.empty:
        st.info("No tracks match the current filters.")
        return

    filtered = filtered.assign(
        status=filtered.apply(data_lib.assignment_status, axis=1),
        status_rank=lambda df: df["status"].map({"Unassigned": 0, "Partially Assigned": 1, "Assigned": 2}),
    )
    if is_unclustered_mode and clustered_track_ids:
        filtered = filtered[~filtered["track_id"].isin(clustered_track_ids)]
        if filtered.empty:
            st.success("All clustered tracks are complete. No unclustered tracks remain.")
            return
    if track_to_cluster:
        filtered = filtered.assign(cluster_id=filtered["track_id"].map(track_to_cluster))
    else:
        filtered = filtered.assign(cluster_id=pd.NA)

    assigned_hidden = int((filtered["status"] == "Assigned").sum())
    if assigned_hidden:
        st.sidebar.caption(f"{assigned_hidden} assigned track(s) hidden")
    filtered = filtered[filtered["status"] != "Assigned"].copy()
    if filtered.empty:
        st.success("All tracks have been assigned. Switch to the Facebank view to review labeled photos.")
        return

    sort_cols = ["status_rank", "picked_count", "max_quality"]
    sort_orders = [True, False, False]
    if track_to_cluster:
        filtered = filtered.sort_values(
            ["cluster_id"] + sort_cols + ["track_id"],
            ascending=[True] + sort_orders + [True],
        )
    else:
        filtered = filtered.sort_values(
            sort_cols + ["track_id"],
            ascending=sort_orders + [True],
        )

    selected_cluster: Optional[int] = None
    if track_to_cluster:
        cluster_options = [None] + sorted(clusters.keys())
        cluster_labels = {None: "All clusters"}
        remaining_counts: Dict[int, int] = (
            filtered[filtered["cluster_id"].notna()]
            .groupby("cluster_id")["track_id"]
            .count()
            .to_dict()
        )
        for cid in sorted(clusters.keys()):
            remaining = int(remaining_counts.get(cid, 0))
            cluster_labels[cid] = f"Cluster {cid:03d} ({remaining} remaining)"
        selected_cluster = st.sidebar.selectbox(
            "Cluster filter",
            options=cluster_options,
            format_func=lambda cid: cluster_labels[cid],
            key="cluster_filter",
        )
        if selected_cluster is not None:
            filtered = filtered[filtered["cluster_id"] == selected_cluster]
            if filtered.empty:
                st.info("No tracks remain after applying the cluster filter.")
                return
        remaining_cluster_total = sum(1 for count in remaining_counts.values() if count > 0)
    else:
        remaining_cluster_total = 0

    filtered = filtered.reset_index(drop=True)
    track_ids = filtered["track_id"].astype(int).tolist()

    if not track_ids:
        st.info("No tracks match the current filters.")
        return

    active_track = ensure_active_track(track_ids)
    if active_track not in track_ids:
        active_track = track_ids[0]
        st.session_state["active_track_id"] = active_track

    pending_track = st.session_state.get("pending_toggle_track")
    pending_sample = st.session_state.get("pending_toggle_sample")
    if pending_track is not None and pending_track not in track_ids:
        st.session_state.pop("pending_toggle_track", None)
        st.session_state.pop("pending_toggle_sample", None)
        pending_track = None
        pending_sample = None
    if pending_track == active_track and pending_sample is not None:
        checkbox_key = f"sample_cb_{active_track}_{pending_sample}"
        st.session_state[checkbox_key] = not st.session_state.get(checkbox_key, False)
        st.session_state[f"focused_sample_{active_track}"] = pending_sample
        st.session_state.pop("pending_toggle_track", None)
        st.session_state.pop("pending_toggle_sample", None)

    if track_to_cluster:
        if selected_cluster is not None:
            st.sidebar.caption(f"Cluster {selected_cluster:03d} • {len(track_ids)} track(s) remaining")
        else:
            active_clusters = remaining_cluster_total if remaining_cluster_total else len(clusters)
            st.sidebar.caption(f"{len(track_ids)} track(s) remaining across {active_clusters} cluster(s)")
    else:
        descriptor = "unclustered track(s)" if is_unclustered_mode else "track(s)"
        st.sidebar.caption(f"{len(track_ids)} {descriptor} available")

    track_samples = data_lib.samples_for_track(samples_df, active_track, include_debug=include_debug)
    if track_samples.empty:
        st.warning("No samples for selected track.")
        return
    track_samples = track_samples.copy()
    active_index = track_ids.index(active_track)
    active_cluster_id = track_to_cluster.get(active_track) if track_to_cluster else None
    cluster_label = None
    if active_cluster_id is not None:
        cluster_label = f"Cluster {active_cluster_id:04d}"

    cluster_title = cluster_label or f"Track {active_track:04d}"

    summary_row = filtered[filtered["track_id"] == active_track].iloc[0]
    byte_track_id = int(summary_row.get("byte_track_id") or active_track)
    track_key = data_lib.TrackKey(stem=selected_stem, harvest_id=active_track, byte_track_id=byte_track_id)
    assignment_info = assignment_index.get(track_key)
    current_target_label = st.session_state.get("active_target_label")

    assigned_lookup_for_track: Dict[str, str] = {}
    if assignment_info:
        for source_path, person in assignment_info.source_to_person.items():
            if person:
                assigned_lookup_for_track[canonical_sample_path(source_path)] = person
    track_samples["assigned_person_label"] = track_samples["path"].map(
        lambda p: assigned_lookup_for_track.get(canonical_sample_path(Path(str(p))), "")
    )
    unassigned_count = int((track_samples["assigned_person_label"].astype(str).str.len() == 0).sum())

    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("◀ Previous", key="cluster_prev"):
            st.session_state["active_track_id"] = track_ids[(active_index - 1) % len(track_ids)]
            st.rerun()
    with nav_cols[1]:
        st.subheader(f"{cluster_title} • Track {active_track:04d}")
        st.caption(
            f"{active_index + 1} of {len(track_ids)} • {unassigned_count} unassigned / {len(track_samples)} total"
        )
    with nav_cols[2]:
        if st.button("Next ▶", key="cluster_next"):
            st.session_state["active_track_id"] = track_ids[(active_index + 1) % len(track_ids)]
            st.rerun()

    focus_key = f"focused_sample_{active_track}"
    if focus_key not in st.session_state and not track_samples.empty:
        st.session_state[focus_key] = track_samples.index[0]

    st.markdown(
        """
        <style>
        div[data-testid="stColumn"] > div > div {
            padding: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    select_row = st.columns([1, 1, 2, 6])
    with select_row[0]:
        if st.button("All", key=f"select_all_{active_track}"):
            set_all_samples(active_track, track_samples, True)
            st.rerun()
    with select_row[1]:
        if st.button("None", key=f"select_none_{active_track}"):
            set_all_samples(active_track, track_samples, False)
            st.rerun()
    with select_row[2]:
        st.caption("Toggle thumbnails to include them in assignment.")

    feedback_messages = st.session_state.pop("image_action_feedback", None)
    if feedback_messages:
        for message in feedback_messages:
            st.success(message)

    # Refresh cached facebank labels when directory contents change.
    list_person_labels.clear()
    facebank_labels = list_person_labels(str(CLI_ARGS.facebank_dir))

    col_left, col_right = st.columns([3, 2])

    with col_left:
        visible_samples = render_thumbnail_grid(
            selected_stem,
            active_track,
            byte_track_id,
            track_samples,
            CLI_ARGS.thumbnails_dir,
            harvest_dir,
            assignment_info,
            current_target_label,
            facebank_labels,
        )
    selected_token = consume_selected_sample()
    if selected_token:
        try:
            token_track_str, token_sample_str = selected_token.split(":", 1)
            token_track = int(token_track_str)
            token_sample = int(token_sample_str)
        except (ValueError, TypeError):
            token_track = None
            token_sample = None
        if token_track is not None and token_sample is not None:
            if token_track == active_track:
                checkbox_key = f"sample_cb_{active_track}_{token_sample}"
                current_state = st.session_state.get(checkbox_key, False)
                st.session_state[checkbox_key] = not current_state
                st.session_state[f"focused_sample_{active_track}"] = token_sample
                st.rerun()
            else:
                st.session_state["pending_toggle_track"] = token_track
                st.session_state["pending_toggle_sample"] = token_sample
                st.session_state["active_track_id"] = token_track
                st.rerun()
    selected_paths = gather_selected_samples(active_track, visible_samples)
    with col_right:
        render_assignment_panel(
            selected_stem,
            summary_row,
            track_samples,
            visible_samples,
            selected_paths,
            CLI_ARGS.facebank_dir,
            ASSIGNMENTS_LOG,
            facebank_labels,
            harvest_dir,
            CLI_ARGS.video,
            samples_df,
            summary_index,
            active_cluster_id,
            clusters.get(active_cluster_id, []) if active_cluster_id is not None else [],
            clusters,
            track_ids,
            active_index,
        )

    render_missing_samples_panel(
        selected_stem,
        harvest_dir,
        active_track,
        track_samples,
        st.session_state.get("active_target_label"),
    )

    pending_actions = collect_pending_sample_actions(active_track, track_samples)
    if pending_actions:
        with st.expander("Pending image maintenance actions", expanded=True):
            st.write(f"Queued {len(pending_actions)} action(s) for this track.")
            if st.button("Apply image actions", key=f"apply_actions_{active_track}"):
                messages = process_sample_actions(
                    harvest_dir,
                    active_track,
                    pending_actions,
                    stem=selected_stem,
                    byte_track_id=byte_track_id,
                    facebank_dir=CLI_ARGS.facebank_dir,
                    assignments_log=ASSIGNMENTS_LOG,
                )
                if messages:
                    st.session_state["image_action_feedback"] = messages
                else:
                    st.session_state["image_action_feedback"] = ["No changes were made."]
                st.rerun()
    else:
        st.caption("No image maintenance actions queued.")

    shortcut = current_shortcut()
    if shortcut == "assign":
        st.session_state[f"shortcut_assign_{active_track}"] = True
        st.rerun()
    elif shortcut == "next":
        st.session_state["advance_track"] = True
        st.rerun()
    elif shortcut == "toggle":
        focus_idx = st.session_state.get(focus_key)
        if focus_idx is None and track_samples.index.any():
            focus_idx = track_samples.index[0]
            st.session_state[focus_key] = focus_idx
        if focus_idx is not None:
            checkbox_key = f"sample_cb_{active_track}_{focus_idx}"
            st.session_state[checkbox_key] = not st.session_state.get(checkbox_key, False)
            st.rerun()

    if st.session_state.pop("advance_track", False):
        idx = track_ids.index(active_track)
        next_idx = (idx + 1) % len(track_ids)
        st.session_state["active_track_id"] = track_ids[next_idx]
        st.rerun()

    render_bottom_tools(selected_stem, CLI_ARGS.facebank_dir, CLI_ARGS.video)

    with st.expander("Facebank browser", expanded=False):
        render_facebank_browser(CLI_ARGS.facebank_dir)


def render_bottom_tools(stem: str, facebank_dir: Path, video_path: Optional[Path]) -> None:
    st.subheader("Post-label actions")
    facebank_button_col, attribution_button_col = st.columns(2)
    with facebank_button_col:
        if st.button("Rebuild facebank"):
            cmd = [
                sys.executable,
                "scripts/build_facebank.py",
                "--facebank-dir",
                str(facebank_dir),
                "--output-dir",
                "data",
            ]
            code, output = run_subprocess(cmd, "Rebuilding facebank")
            display_command_output(
                code,
                output,
                "Facebank rebuild completed.",
                "Facebank rebuild failed",
            )
    with attribution_button_col:
        if st.button("Run attribution"):
            if not video_path or not video_path.exists():
                st.error("Video path missing; cannot run attribution.")
            else:
                cmd = [
                    sys.executable,
                    "scripts/run_tracker.py",
                    str(video_path),
                    "--output-dir",
                    "data/outputs",
                    "--person-weights",
                    "models/weights/yolov8n.pt",
                    "--facebank-parquet",
                    "data/facebank.parquet",
                ]
                code, output = run_subprocess(cmd, "Running attribution")
                message = f"Attribution outputs saved under data/outputs/{stem}"
                display_command_output(
                    code,
                    output,
                    message,
                    "Attribution run failed",
                )


if __name__ == "__main__":
    main()
