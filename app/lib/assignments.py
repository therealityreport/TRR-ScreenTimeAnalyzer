"""Assignment routines for copying samples into the facebank and logging actions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, MutableSet, Optional, Tuple


APP_VERSION = "labeler-0.1.0"
VALID_LABEL_PATTERN = re.compile(r"^[A-Z0-9_-]+$")


@dataclass
class AssignmentResult:
    copied: List[Tuple[str, str]]
    skipped: List[Tuple[str, str]]
    log_entry: Optional[dict]


def normalize_label(raw: str) -> str:
    """Normalize and validate a person label."""
    if raw is None:
        raise ValueError("Person label is required")
    label = raw.strip().upper()
    if not label:
        raise ValueError("Person label cannot be empty")
    cleaned = re.sub(r"[\s]+", "_", label)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("Person label resolved to empty after sanitization")
    if not VALID_LABEL_PATTERN.match(cleaned):
        raise ValueError(f"Invalid label '{cleaned}'; use A-Z, 0-9, underscore, or hyphen")
    return cleaned


def generate_destination_name(stem: str, harvest_id: int, byte_track_id: int, source_path: Path) -> str:
    """Generate a deterministic destination filename for a copied sample."""
    frame_token = _frame_token(source_path.stem)
    digest = hashlib.sha1(source_path.as_posix().encode("utf-8")).hexdigest()[:8]
    suffix = source_path.suffix.lower()
    return f"{stem}_H{harvest_id:04d}_B{byte_track_id:04d}_F{frame_token}_{digest}{suffix}"


def _frame_token(stem: str) -> str:
    if "_f" in stem:
        candidate = stem.split("_f")[-1]
        if candidate.isdigit():
            return candidate
    return stem[-6:]


def assign_samples(
    sources: Iterable[Path],
    stem: str,
    harvest_id: int,
    byte_track_id: int,
    person_label: str,
    facebank_dir: Path,
    assignments_log: Path,
    existing_sources: Optional[MutableSet[str]] = None,
    dry_run: bool = False,
) -> AssignmentResult:
    """Copy selected face crops into the facebank and append a JSONL log entry."""
    normalized_label = normalize_label(person_label)
    dest_dir = facebank_dir / normalized_label
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        assignments_log.parent.mkdir(parents=True, exist_ok=True)

    copied: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for source in sources:
        source = Path(source)
        if not source.exists():
            skipped.append((str(source), "missing"))
            continue
        source_key = str(source)
        if existing_sources is not None and source_key in existing_sources:
            skipped.append((source_key, "already_assigned"))
            continue
        dest_name = generate_destination_name(stem, harvest_id, byte_track_id, source)
        dest_path = dest_dir / dest_name
        if dest_path.exists():
            skipped.append((source_key, "duplicate_destination"))
            if existing_sources is not None:
                existing_sources.add(source_key)
            continue
        if not dry_run:
            shutil.copy2(source, dest_path)
        copied.append((source_key, str(dest_path)))
        if existing_sources is not None:
            existing_sources.add(source_key)

    log_entry: Optional[dict] = None
    if copied:
        log_entry = {
            "stem": stem,
            "harvest_id": harvest_id,
            "byte_track_id": byte_track_id,
            "person": normalized_label,
            "files": [{"source": src, "dest": dest} for src, dest in copied],
            "when": timestamp,
            "app_version": APP_VERSION,
        }
        if not dry_run:
            with assignments_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + os.linesep)

    return AssignmentResult(copied=copied, skipped=skipped, log_entry=log_entry)


def unassign_sample(dest_path: Path, assignments_log: Path, base_dir: Path) -> Optional[Tuple[str, str, str]]:
    """Remove an assigned facebank image and update the assignments log.

    Returns a tuple of (stem, source_path, person) when a matching entry is removed.
    """

    if not assignments_log.exists():
        return None

    dest_resolved = dest_path.resolve()
    updated_entries: List[dict] = []
    removed_info: Optional[Tuple[str, str, str]] = None

    with assignments_log.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            files = entry.get("files", [])
            keep_files = []
            for file_entry in files:
                if isinstance(file_entry, dict):
                    dest_str = file_entry.get("dest") or file_entry.get("destination") or ""
                    source_str = file_entry.get("source") or ""
                else:
                    dest_str = str(file_entry)
                    source_str = ""

                if not dest_str:
                    keep_files.append(file_entry)
                    continue

                candidate = Path(dest_str)
                if not candidate.is_absolute():
                    candidate = (base_dir / candidate).resolve()
                else:
                    candidate = candidate.resolve()

                if candidate == dest_resolved and removed_info is None:
                    removed_info = (
                        entry.get("stem") or "",
                        source_str,
                        entry.get("person") or "",
                    )
                    continue

                keep_files.append(file_entry)

            if keep_files:
                entry["files"] = keep_files
                updated_entries.append(entry)
            elif removed_info is None:
                # No match found in this entry; preserve it as-is
                updated_entries.append(entry)

    if removed_info is None:
        return None

    with assignments_log.open("w", encoding="utf-8") as fh:
        for entry in updated_entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + os.linesep)

    return removed_info
