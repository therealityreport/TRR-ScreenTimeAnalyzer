from pathlib import Path

from app.lib import data as data_lib


def test_manifest_loader_and_summary_byte_ids():
    harvest_dir = Path("data/harvest/RHOBH-TEST")
    manifest_df = data_lib.load_manifest(harvest_dir)
    samples_df = data_lib.load_samples(harvest_dir)
    _, assignments_index = data_lib.load_assignments(Path("data/facebank/assignments.jsonl"))

    assert "byte_track_id" in manifest_df.columns
    summary = data_lib.summarize_tracks("RHOBH-TEST", manifest_df, samples_df, assignments_index)
    assert not summary.empty
    first_row = summary.iloc[0]
    assert first_row["track_id"] == first_row["byte_track_id"]

    track_samples = data_lib.samples_for_track(samples_df, int(first_row["track_id"]), include_debug=False)
    assert not track_samples["is_debug"].any()
