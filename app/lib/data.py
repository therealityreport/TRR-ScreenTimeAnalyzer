"""Harvest data loading helpers for the labeler app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TrackKey:
    stem: str
    harvest_id: int
    byte_track_id: int


@dataclass
class TrackAssignment:
    persons: List[str]
    sources: List[str]
    destinations: List[str]

<<<<<<< HEAD
def load_clusters(harvest_dir: Path) -> Dict[int, List[int]]:
    """Load clusters.json if present; returns {cluster_id: [track_ids]} mapping."""
    cluster_path = harvest_dir / "clusters.json"
    if not cluster_path.exists():
        return {}
    try:
        with cluster_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {cluster_path}: {exc}") from exc
    clusters: Dict[int, List[int]] = {}
    for entry in payload.get("clusters", []):
        cid = int(entry.get("id", -1))
        tracks = [int(t) for t in entry.get("tracks", [])]
        if cid < 0 or not tracks:
            continue
        clusters[cid] = sorted(tracks)
    return clusters


=======
>>>>>>> origin/feat/identity-guard

def list_harvest_stems(harvest_root: Path) -> List[str]:
    """Return sorted harvest directory names under the given root."""
    if not harvest_root.exists():
        return []
    stems = [p.name for p in harvest_root.iterdir() if p.is_dir()]
    return sorted(stems)


def load_manifest(harvest_dir: Path) -> pd.DataFrame:
    """Load manifest.json for a harvest; returns empty DataFrame if missing."""
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not raw:
        return pd.DataFrame()
    df = pd.json_normalize(raw)
    # Normalize nested quality metrics when present.
    for col in ("total_frames", "avg_conf", "avg_area", "first_ts_ms", "last_ts_ms", "byte_track_id", "track_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


<<<<<<< HEAD
def load_progress(harvest_dir: Path) -> Optional[Dict[str, object]]:
    """Load progress.json for a harvest if present."""
    progress_path = harvest_dir / "progress.json"
    if not progress_path.exists():
        return None
    try:
        with progress_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    result: Dict[str, object] = {}
    frames_total = payload.get("frames_total", 0)
    frames_done = payload.get("frames_done", 0)
    percent = payload.get("percent")
    tracks = payload.get("tracks", 0)
    result["frames_total"] = int(frames_total or 0)
    result["frames_done"] = int(frames_done or 0)
    try:
        result["percent"] = float(percent) if percent is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        result["percent"] = None
    result["tracks"] = int(tracks or 0)
    result["raw"] = payload
    return result


=======
>>>>>>> origin/feat/identity-guard
def _infer_track_id(track_dir: Path) -> int:
    try:
        return int(track_dir.name.split("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _load_selected_samples_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "picked" in df.columns:
        df["picked"] = df["picked"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["picked"] = True
    df["path"] = df["path"].astype(str)
    df["is_debug"] = False
    if "reason" not in df.columns:
        df["reason"] = "picked"
    return df


def _scan_track_directory(track_dir: Path, picked: bool, track_id: Optional[int] = None) -> Iterable[Dict]:
    track_id = track_id if track_id is not None else _infer_track_id(track_dir)
    images = [p for p in track_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for img in sorted(images):
        yield {
            "track_id": track_id,
            "byte_track_id": track_id,
            "frame": _frame_from_name(img.stem),
            "quality": None,
            "sharpness": None,
            "frontalness": None,
            "area_frac": None,
            "picked": picked,
            "reason": "picked" if picked else "debug",
            "path": str(img),
            "association_iou": None,
            "match_mode": None,
            "frame_offset": None,
            "identity_cosine": None,
            "similarity_to_centroid": None,
            "provider": None,
            "is_debug": not picked,
        }


def _frame_from_name(stem: str) -> Optional[int]:
    if "_f" in stem:
        try:
            return int(stem.split("_f")[-1])
        except ValueError:
            return None
    return None


<<<<<<< HEAD
def _resolve_sample_path(original_path: Path, row: pd.Series, harvest_dir: Path) -> Path:
    """Return a path that exists on disk, falling back to candidate/debug crops."""
    if original_path.is_absolute() and original_path.exists():
        return original_path

    direct_candidate = harvest_dir / original_path
    if not original_path.is_absolute() and direct_candidate.exists():
        return direct_candidate.resolve()

    # Handle legacy prefixes like data/harvest/<stem>/... by trimming to the track segment.
    for idx, part in enumerate(original_path.parts):
        if part.startswith("track_"):
            trimmed = Path(*original_path.parts[idx:])
            candidate = harvest_dir / trimmed
            if candidate.exists():
                return candidate.resolve()
            break

    track_id = _safe_int(row.get("track_id"))
    frame_idx = _safe_int(row.get("frame"))
    suffix = original_path.suffix if original_path.suffix else ".jpg"

    if track_id is not None and frame_idx is not None:
        candidates_dir = harvest_dir / "candidates" / f"track_{track_id:04d}"
        candidate_path = candidates_dir / f"F{frame_idx:06d}{suffix}"
        if candidate_path.exists():
            return candidate_path
        if suffix.lower() != ".jpg":
            alt_candidate = candidates_dir / f"F{frame_idx:06d}.jpg"
            if alt_candidate.exists():
                return alt_candidate

        debug_path = harvest_dir / f"track_{track_id:04d}" / "debug" / original_path.name
        if debug_path.exists():
            return debug_path

    return direct_candidate if direct_candidate.exists() else original_path


=======
>>>>>>> origin/feat/identity-guard
def load_samples(harvest_dir: Path) -> pd.DataFrame:
    """Load selected samples, falling back to scanning directories if CSV missing."""
    csv_path = harvest_dir / "selected_samples.csv"
    frames: List[pd.DataFrame] = []
    if csv_path.exists():
        frames.append(_load_selected_samples_csv(csv_path))
    else:
        for track_dir in sorted(p for p in harvest_dir.iterdir() if p.is_dir() and p.name.startswith("track_")):
            samples = list(_scan_track_directory(track_dir, picked=True))
            if samples:
                frames.append(pd.DataFrame(samples))
    debug_frames: List[pd.DataFrame] = []
    for track_dir in sorted(p for p in harvest_dir.iterdir() if p.is_dir() and p.name.startswith("track_")):
        debug_dir = track_dir / "debug"
        if not debug_dir.exists():
            continue
        track_id = _infer_track_id(track_dir)
        debug_samples = list(_scan_track_directory(debug_dir, picked=False, track_id=track_id))
        if debug_samples:
            debug_frames.append(pd.DataFrame(debug_samples))
    if frames:
        samples_df = pd.concat(frames, ignore_index=True)
    else:
        samples_df = pd.DataFrame(
            columns=[
                "track_id",
                "byte_track_id",
                "frame",
                "quality",
                "sharpness",
                "frontalness",
                "area_frac",
                "picked",
                "reason",
                "path",
                "association_iou",
                "match_mode",
                "frame_offset",
                "identity_cosine",
                "similarity_to_centroid",
                "provider",
                "is_debug",
            ]
        )
    if debug_frames:
        samples_df = pd.concat([samples_df, *debug_frames], ignore_index=True)
    samples_df["track_id"] = pd.to_numeric(samples_df.get("track_id"), errors="coerce").fillna(-1).astype(int)
    if "byte_track_id" in samples_df.columns:
        samples_df["byte_track_id"] = pd.to_numeric(samples_df["byte_track_id"], errors="coerce").fillna(
            samples_df["track_id"]
        )
    else:
        samples_df["byte_track_id"] = samples_df["track_id"]
    if "frame" not in samples_df.columns:
        samples_df["frame"] = samples_df["path"].map(lambda p: _frame_from_name(Path(p).stem))
<<<<<<< HEAD
    if not samples_df.empty:
        samples_df["path"] = samples_df.apply(
            lambda row: str(_resolve_sample_path(Path(str(row.get("path", ""))), row, harvest_dir)),
            axis=1,
        )
=======
>>>>>>> origin/feat/identity-guard
    return samples_df


def load_assignments(log_path: Path) -> Tuple[List[Dict], Dict[TrackKey, TrackAssignment]]:
    """Load assignments log and build an index keyed by (stem, harvest_id, byte_track_id)."""
    entries: List[Dict] = []
    index: Dict[TrackKey, TrackAssignment] = {}
    if not log_path.exists():
        return entries, index
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)
            stem = entry.get("stem")
            harvest_id = _safe_int(entry.get("harvest_id"))
            byte_track_id = _safe_int(entry.get("byte_track_id"))
            key = TrackKey(
                stem=stem or "",
                harvest_id=harvest_id if harvest_id is not None else -1,
                byte_track_id=byte_track_id if byte_track_id is not None else -1,
            )
            info = index.get(key)
            if info is None:
                info = TrackAssignment(persons=[], sources=[], destinations=[])
                index[key] = info
            person = entry.get("person")
            if person and person not in info.persons:
                info.persons.append(person)
            for file_entry in entry.get("files", []):
                if isinstance(file_entry, dict):
                    source = file_entry.get("source") or file_entry.get("src") or file_entry.get("path")
                    dest = file_entry.get("dest") or file_entry.get("destination") or file_entry.get("target")
                else:
                    source = None
                    dest = str(file_entry)
                if source and source not in info.sources:
                    info.sources.append(source)
                if dest and dest not in info.destinations:
                    info.destinations.append(dest)
    return entries, index


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_tracks(
    stem: str,
    manifest_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    assignment_index: Dict[TrackKey, TrackAssignment],
) -> pd.DataFrame:
    """Create a per-track summary DataFrame combining manifest, samples, and assignments."""
    tracks: Dict[int, Dict] = {}
    if not samples_df.empty:
        for track_id, group in samples_df.groupby("track_id"):
            byte_track_id = int(group["byte_track_id"].mode().iloc[0]) if not group["byte_track_id"].empty else track_id
            picked = group[group["picked"] == True]  # noqa: E712
            quality = picked["quality"].dropna()
            frontalness = picked["frontalness"].dropna()
            sharpness = picked["sharpness"].dropna()
            key = TrackKey(stem=stem, harvest_id=int(track_id), byte_track_id=int(byte_track_id))
            assignment = assignment_index.get(key)
            assigned_count = len(assignment.destinations) if assignment else 0
            tracks[int(track_id)] = {
                "track_id": int(track_id),
                "byte_track_id": int(byte_track_id),
                "sample_count": int(len(group)),
                "picked_count": int(len(picked)),
                "rejection_count": int(len(group) - len(picked)),
                "mean_quality": float(quality.mean()) if not quality.empty else None,
                "median_quality": float(quality.median()) if not quality.empty else None,
                "mean_frontalness": float(frontalness.mean()) if not frontalness.empty else None,
                "mean_sharpness": float(sharpness.mean()) if not sharpness.empty else None,
                "max_quality": float(quality.max()) if not quality.empty else None,
                "assigned_count": assigned_count,
                "assigned_persons": ", ".join(assignment.persons) if assignment else "",
            }
    if not manifest_df.empty:
        for _, row in manifest_df.iterrows():
            track_id = int(row.get("track_id", -1))
            byte_track_id = int(row.get("byte_track_id", track_id))
            info = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "byte_track_id": byte_track_id,
                    "sample_count": 0,
                    "picked_count": 0,
                    "rejection_count": 0,
                    "mean_quality": None,
                    "median_quality": None,
                    "mean_frontalness": None,
                    "mean_sharpness": None,
                    "max_quality": None,
                    "assigned_count": 0,
                    "assigned_persons": "",
                },
            )
            info["byte_track_id"] = byte_track_id
            info["total_frames"] = _maybe_float(row.get("total_frames"))
            duration_ms = None
            first_ts = _maybe_float(row.get("first_ts_ms"))
            last_ts = _maybe_float(row.get("last_ts_ms"))
            if first_ts is not None and last_ts is not None and last_ts >= first_ts:
                duration_ms = last_ts - first_ts
            info["duration_ms"] = duration_ms
            info["avg_conf"] = _maybe_float(row.get("avg_conf"))
            info["avg_area"] = _maybe_float(row.get("avg_area"))
    if not tracks:
        return pd.DataFrame(
            columns=[
                "track_id",
                "byte_track_id",
                "sample_count",
                "picked_count",
                "rejection_count",
                "mean_quality",
                "median_quality",
                "mean_frontalness",
                "mean_sharpness",
                "max_quality",
                "assigned_count",
                "assigned_persons",
                "total_frames",
                "duration_ms",
                "avg_conf",
                "avg_area",
            ]
        )
    summary_df = pd.DataFrame(tracks.values()).sort_values("track_id").reset_index(drop=True)
    return summary_df


def assignment_status(row: pd.Series) -> str:
    """Classify assignment status for a track summary row."""
    picked = int(row.get("picked_count") or 0)
    assigned = int(row.get("assigned_count") or 0)
    if assigned <= 0:
        return "Unassigned"
    if picked and assigned >= picked:
        return "Assigned"
    return "Partially Assigned"


def status_badge_color(status: str) -> str:
    return {"Assigned": "green", "Partially Assigned": "orange"}.get(status, "gray")


def filter_tracks(
    summary_df: pd.DataFrame,
    min_samples: int = 0,
    min_frames: int = 0,
    quality_threshold: Optional[float] = None,
    search_term: str = "",
) -> pd.DataFrame:
    """Apply sidebar filters to the summary DataFrame."""
    if summary_df.empty:
        return summary_df
    df = summary_df.copy()
    if min_samples > 0:
        df = df[df["picked_count"] >= min_samples]
    if min_frames > 0 and "total_frames" in df.columns:
        df = df[df["total_frames"].fillna(0) >= min_frames]
    if quality_threshold is not None:
        df = df[df["max_quality"].fillna(0) >= quality_threshold]
    if search_term:
        term = search_term.lower()
        df = df[
            df["track_id"].astype(str).str.contains(term)
            | df["byte_track_id"].astype(str).str.contains(term)
            | df.get("assigned_persons", "").fillna("").str.lower().str.contains(term)
        ]
    return df


def percentile_threshold(series: pd.Series, percentile: float) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    percentile = max(0.0, min(100.0, percentile))
    return float(clean.quantile(percentile / 100.0))


def samples_for_track(samples_df: pd.DataFrame, track_id: int, include_debug: bool = False) -> pd.DataFrame:
    """Return samples for a given track, filtering debug rows if requested."""
    if samples_df.empty:
        return samples_df
    subset = samples_df[samples_df["track_id"] == track_id].copy()
    if not include_debug:
        subset = subset[subset["is_debug"] != True]  # noqa: E712
    return subset.sort_values(["picked", "frame"], ascending=[False, True])


def _maybe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
