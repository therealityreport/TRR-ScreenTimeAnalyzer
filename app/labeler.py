"""Streamlit labeler app for assigning harvested face tracks to people."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
<<<<<<< HEAD
@st.cache_data(show_spinner=False)
def load_clusters_cached(harvest_path: str):
    return data_lib.load_clusters(Path(harvest_path))


=======
>>>>>>> origin/feat/identity-guard
    root = Path(facebank_dir)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


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
        if value:
            selected.append(path)
    return selected


def set_all_samples(track_id: int, samples: pd.DataFrame, checked: bool) -> None:
    for idx in samples.index:
        st.session_state[f"sample_cb_{track_id}_{idx}"] = checked


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


def render_assignment_panel(
    stem: str,
    track_row: pd.Series,
    track_samples: pd.DataFrame,
    selected_paths: List[str],
    facebank_dir: Path,
    assignments_log: Path,
    facebank_labels: List[str],
    harvest_dir: Path,
    video_path: Optional[Path],
<<<<<<< HEAD
    samples_df: pd.DataFrame,
    summary_index: pd.DataFrame,
    cluster_id: Optional[int],
    cluster_tracks: Sequence[int],
=======
>>>>>>> origin/feat/identity-guard
) -> None:
    track_id = int(track_row["track_id"])
    byte_track_id = int(track_row["byte_track_id"])

    st.subheader(f"Track {track_id:04d} • Byte {byte_track_id:04d}")

    status = data_lib.assignment_status(track_row)
    st.markdown(format_badge(status, data_lib.status_badge_color(status)), unsafe_allow_html=True)
<<<<<<< HEAD
    cluster_track_list = sorted({int(t) for t in cluster_tracks}) if cluster_tracks else []
    if cluster_id is not None and cluster_track_list:
        track_list_str = ", ".join(f"{tid:04d}" for tid in cluster_track_list)
        st.info(f"Cluster {cluster_id:03d} → {len(cluster_track_list)} track(s): {track_list_str}")

=======
>>>>>>> origin/feat/identity-guard

    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Picked samples", value=int(track_row.get("picked_count") or 0))
    with meta_cols[1]:
        st.metric("Rejections", value=int(track_row.get("rejection_count") or 0))
    with meta_cols[2]:
        median_quality = track_row.get("median_quality")
        if median_quality is not None and not math.isnan(median_quality):
            st.metric("Median quality", f"{median_quality:.2f}")
    with meta_cols[3]:
        total_frames = track_row.get("total_frames")
        if total_frames:
            st.metric("Frames", int(total_frames))

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

    st.caption("Keyboard: A=Assign selected, N=Next cluster, Space=toggle focus sample.")
    assign_selected_disabled = target_label is None or not selected_paths
    assign_selected_requested = st.button(
        f"Assign {len(selected_paths)} selected",
        disabled=assign_selected_disabled,
        key=f"assign_selected_btn_{track_id}",
        type="primary",
    )
    assign_all_requested = st.button(
        f"Assign all {len(track_samples)} samples",
        disabled=target_label is None or track_samples.empty,
        key=f"assign_all_btn_{track_id}",
    )
<<<<<<< HEAD
    cluster_assign_requested = False
    if cluster_track_list and target_label is not None:
        cluster_assign_requested = st.button(
            f"Assign cluster ({len(cluster_track_list)} tracks)",
            key=f"assign_cluster_btn_{track_id}",
        )
=======
>>>>>>> origin/feat/identity-guard

    representative_sample: Optional[Path] = None
    if selected_paths:
        representative_sample = Path(selected_paths[0])
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
        paths_to_assign = [str(Path(p)) for p in track_samples["path"].tolist()]
    elif assign_selected_requested and target_label is not None:
        paths_to_assign = list(selected_paths)

<<<<<<< HEAD
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
=======
    if paths_to_assign and target_label is not None:
>>>>>>> origin/feat/identity-guard
        try:
            normalized_label = assign_lib.normalize_label(target_label or "")
        except ValueError as exc:
            st.error(str(exc))
        else:
            _, index = load_assignments_cached(str(assignments_log))
            existing_sources: set[str] = set()
            for details in index.values():
                existing_sources.update(details.sources)

<<<<<<< HEAD
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
=======
            result = assign_lib.assign_samples(
                [Path(p) for p in paths_to_assign],
                stem=stem,
                harvest_id=track_id,
                byte_track_id=byte_track_id,
                person_label=normalized_label,
                facebank_dir=facebank_dir,
                assignments_log=assignments_log,
                existing_sources=existing_sources,
            )
            if result.copied:
                st.success(f"Copied {len(result.copied)} file(s) to {normalized_label}")
>>>>>>> origin/feat/identity-guard
                load_assignments_cached.clear()
                list_person_labels.clear()
                st.rerun()
            else:
                st.info("No files copied.")

    render_track_montage_controls(harvest_dir, track_id)
    render_overlay_controls(harvest_dir, CLI_ARGS.video, stem, track_id)


def render_thumbnail_grid(
    stem: str,
    track_id: int,
    track_samples: pd.DataFrame,
    thumbnails_dir: Path,
) -> None:
    cols_per_row = 4
    rows = math.ceil(len(track_samples) / cols_per_row) or 1
    focus_key = f"focused_sample_{track_id}"
    focused_idx = st.session_state.get(focus_key)
    if focused_idx is None and not track_samples.empty:
        focused_idx = track_samples.index[0]
        st.session_state[focus_key] = focused_idx
    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            sample_idx = row_idx * cols_per_row + col_idx
            if sample_idx >= len(track_samples):
                continue
            row = track_samples.iloc[sample_idx]
            path = Path(row["path"])
            checkbox_key = f"sample_cb_{track_id}_{row.name}"
            default_checked = bool(row.get("picked")) and not bool(row.get("is_debug"))
            checked = initialize_sample_checkbox(track_id, row.name, default_checked)
            
            # The checkbox widget automatically manages its own session state
            # We just read the value after creation, not set it manually
            is_checked = cols[col_idx].checkbox(
                " ",
                value=checked,
                key=checkbox_key,
            )
            
            # Track focus changes when checkbox state changes
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
            try:
                thumb_path = thumb_lib.ensure_thumbnail(path, stem, thumbnails_dir)
                cols[col_idx].image(
                    str(thumb_path),
                    width="stretch",
                    caption=", ".join(caption_parts) if caption_parts else path.name,
                )
            except FileNotFoundError:
                cols[col_idx].warning(f"Missing: {path}")


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
<<<<<<< HEAD
    if CLI_ARGS.video is None:
        st.warning("Video path not supplied; overlay and attribution tools will be limited.")

=======
>>>>>>> origin/feat/identity-guard
    include_debug = st.sidebar.toggle("Include debug rejects", value=False)
    min_samples = st.sidebar.slider("Min picked samples", 0, 20, 0)
    min_frames = st.sidebar.slider("Min track frames", 0, 300, 0, step=5)
    quality_percentile = st.sidebar.slider("Quality percentile", 0, 100, 0)
    search_term = st.sidebar.text_input("Search track / byte id")
    manifest_df = load_manifest_cached(str(harvest_dir))
<<<<<<< HEAD
    clusters = load_clusters_cached(str(harvest_dir))
    track_to_cluster = {track: cid for cid, tracks in clusters.items() for track in tracks}
    if track_to_cluster:
        samples_df = samples_df.copy()
        samples_df["cluster_id"] = samples_df["track_id"].map(track_to_cluster)
=======
    samples_df = load_samples_cached(str(harvest_dir))
    if samples_df.empty:
        st.warning("No samples available for this harvest.")
        return
    if not include_debug:
        samples_df = samples_df[samples_df["is_debug"] != True]  # noqa: E712
>>>>>>> origin/feat/identity-guard

    assignments, assignment_index = load_assignments_cached(str(ASSIGNMENTS_LOG))
    summary_df = data_lib.summarize_tracks(selected_stem, manifest_df, samples_df, assignment_index)
    if summary_df.empty:
        st.warning("No track summaries available.")
        return

<<<<<<< HEAD
    summary_index = summary_df.set_index("track_id", drop=False)

=======
>>>>>>> origin/feat/identity-guard
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
<<<<<<< HEAD
    if track_to_cluster:
        filtered = filtered.assign(cluster_id=filtered["track_id"].map(track_to_cluster))
    else:
        filtered = filtered.assign(cluster_id=pd.NA)
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
        for cid in sorted(clusters.keys()):
            cluster_labels[cid] = f"Cluster {cid:03d} ({len(clusters[cid])} tracks)"
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

    filtered = filtered.reset_index(drop=True)
    track_ids = filtered["track_id"].astype(int).tolist()

    if not track_ids:
        st.info("No tracks match the current filters.")
=======
    filtered = filtered.sort_values(
        ["status_rank", "picked_count", "max_quality"],
        ascending=[True, False, False],
    )

    cluster_track_ids: List[int] = []
    if "provider" in samples_df.columns:
        provider_mask = samples_df["provider"].fillna("").astype(str) == "cluster_preview"
        cluster_track_ids = (
            samples_df.loc[provider_mask, "track_id"].dropna().astype(int).unique().tolist()
        )
    if cluster_track_ids:
        filtered = filtered[filtered["track_id"].isin(cluster_track_ids)]

    filtered = filtered.sort_values("track_id").reset_index(drop=True)
    track_ids = filtered["track_id"].astype(int).tolist()

    if not track_ids:
        st.info("No scene-aware clusters available. Run harvest with --cluster-preview or adjust filters.")
>>>>>>> origin/feat/identity-guard
        return

    active_track = ensure_active_track(track_ids)
    if active_track not in track_ids:
        active_track = track_ids[0]
        st.session_state["active_track_id"] = active_track

<<<<<<< HEAD
    if track_to_cluster:
        if selected_cluster is not None:
            st.sidebar.caption(f"Cluster {selected_cluster:03d} • {len(track_ids)} tracks")
        else:
            st.sidebar.caption(f"{len(clusters)} clusters • {len(track_ids)} tracks")
    else:
        st.sidebar.caption(f"{len(track_ids)} tracks available")
=======
    st.sidebar.caption(f"{len(track_ids)} clusters ready to review")
>>>>>>> origin/feat/identity-guard

    track_samples = data_lib.samples_for_track(samples_df, active_track, include_debug=include_debug)
    if track_samples.empty:
        st.warning("No samples for selected track.")
        return
    active_index = track_ids.index(active_track)
<<<<<<< HEAD
    active_cluster_id = track_to_cluster.get(active_track) if track_to_cluster else None
    cluster_label = None
    if active_cluster_id is not None:
        cluster_label = f"Cluster {active_cluster_id:04d}"
=======
    cluster_label = None
    if "cluster_id" in track_samples.columns:
        unique_clusters = track_samples["cluster_id"].dropna().unique().tolist()
        if unique_clusters:
            cid = int(unique_clusters[0])
            cluster_label = f"Cluster {cid:04d}" if cid >= 0 else "Noise Cluster"
>>>>>>> origin/feat/identity-guard

    cluster_title = cluster_label or f"Track {active_track:04d}"

    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("◀ Previous", key="cluster_prev"):
            st.session_state["active_track_id"] = track_ids[(active_index - 1) % len(track_ids)]
            st.rerun()
    with nav_cols[1]:
        st.subheader(f"{cluster_title} • Track {active_track:04d}")
        st.caption(f"{active_index + 1} of {len(track_ids)} • {len(track_samples)} samples")
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

    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_thumbnail_grid(selected_stem, active_track, track_samples, CLI_ARGS.thumbnails_dir)
    selected_paths = gather_selected_samples(active_track, track_samples)
    with col_right:
        facebank_labels = list_person_labels(str(CLI_ARGS.facebank_dir))
        render_assignment_panel(
            selected_stem,
            filtered[filtered["track_id"] == active_track].iloc[0],
            track_samples,
            selected_paths,
            CLI_ARGS.facebank_dir,
            ASSIGNMENTS_LOG,
            facebank_labels,
            harvest_dir,
            CLI_ARGS.video,
<<<<<<< HEAD
            samples_df,
            summary_index,
            active_cluster_id,
            clusters.get(active_cluster_id, []) if active_cluster_id is not None else [],
=======
>>>>>>> origin/feat/identity-guard
        )

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
