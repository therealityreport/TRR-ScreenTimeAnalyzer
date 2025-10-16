import json
from pathlib import Path

from app.lib import assignments as assign_lib


def test_assign_samples_and_logging(tmp_path):
    harvest_dir = tmp_path / "harvest"
    harvest_dir.mkdir()
    source_path = harvest_dir / "SAMPLE_f000001.jpg"
    source_path.write_bytes(b"fake-image-data")

    facebank_dir = tmp_path / "facebank"
    assignments_log = tmp_path / "assignments.jsonl"

    # Dry run should not create files or logs but should return prospective entry.
    dry_run = assign_lib.assign_samples(
        [source_path],
        stem="TESTSHOW",
        harvest_id=1,
        byte_track_id=2,
        person_label="PERSON_A",
        facebank_dir=facebank_dir,
        assignments_log=assignments_log,
        dry_run=True,
    )
    assert dry_run.copied, "Dry run should report would-be copies"
    assert dry_run.log_entry is not None
    assert not (facebank_dir / "PERSON_A").exists()
    assert not assignments_log.exists()

    # Real run writes file and log entry.
    result = assign_lib.assign_samples(
        [source_path],
        stem="TESTSHOW",
        harvest_id=1,
        byte_track_id=2,
        person_label="PERSON_A",
        facebank_dir=facebank_dir,
        assignments_log=assignments_log,
    )
    assert result.copied, "Expected sample to be copied"
    dest_path = Path(result.copied[0][1])
    assert dest_path.exists()
    assert dest_path.parent == facebank_dir / "PERSON_A"
    assert assignments_log.exists()

    log_entry = json.loads(assignments_log.read_text().strip())
    assert log_entry["person"] == "PERSON_A"
    assert log_entry["harvest_id"] == 1
    assert log_entry["byte_track_id"] == 2

    # Second run skips because destination already exists.
    repeat = assign_lib.assign_samples(
        [source_path],
        stem="TESTSHOW",
        harvest_id=1,
        byte_track_id=2,
        person_label="PERSON_A",
        facebank_dir=facebank_dir,
        assignments_log=assignments_log,
    )
    assert not repeat.copied
    assert repeat.log_entry is None
