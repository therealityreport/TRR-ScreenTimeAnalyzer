from pathlib import Path

from scripts import cluster_tracks


def test_parse_args_sim_alias():
    args = cluster_tracks.parse_args(["dummy", "--sim", "0.42"])
    assert abs(args.cluster_thresh - 0.42) < 1e-6


def test_load_selected_samples_filters(tmp_path: Path):
    harvest_dir = tmp_path
    csv_path = harvest_dir / "selected_samples.csv"
    sample_dir = harvest_dir / "track_0001"
    sample_dir.mkdir()
    valid_sample = sample_dir / "img_valid.jpg"
    invalid_sample = sample_dir / "img_invalid.jpg"
    valid_sample.write_bytes(b"")
    invalid_sample.write_bytes(b"")

    rows = [
        {
            "track_id": 1,
            "path": valid_sample.relative_to(harvest_dir).as_posix(),
            "picked": True,
            "sharpness": 60.0,
            "area_frac": 0.2,
        },
        {
            "track_id": 2,
            "path": invalid_sample.relative_to(harvest_dir).as_posix(),
            "picked": True,
            "sharpness": 20.0,
            "area_frac": 0.01,
        },
    ]

    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("track_id,path,picked,sharpness,area_frac\n")
        for row in rows:
            fh.write(f"{row['track_id']},{row['path']},{row['picked']},{row['sharpness']},{row['area_frac']}\n")

    samples = cluster_tracks.load_selected_samples(
        harvest_dir,
        csv_path,
        min_samples=1,
        min_sharpness=50.0,
        min_area_frac=0.05,
    )

    assert len(samples) == 1
    assert samples[0].track_id == 1
