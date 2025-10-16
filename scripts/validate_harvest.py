#!/usr/bin/env python3
"""
Validate and analyze a harvest directory, providing detailed statistics.

Usage:
    python scripts/validate_harvest.py data/harvest/RHOBH-TEST
    python scripts/validate_harvest.py data/harvest/RHOBH-TEST --output diagnostics/validation
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from screentime.io_utils import load_json, setup_logging

import logging
LOGGER = logging.getLogger("scripts.validate_harvest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate harvest results")
    parser.add_argument("harvest_dir", type=Path, help="Harvest directory to validate")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--show-tracks", action="store_true", help="Show per-track details")
    return parser.parse_args()


def validate_harvest(harvest_dir: Path) -> Dict:
    """Validate harvest directory and return detailed statistics."""
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check manifest exists
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.exists():
        results["valid"] = False
        results["errors"].append(f"Manifest file not found: {manifest_path}")
        return results
    
    # Load manifest
    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load manifest: {e}")
        return results
    
    # Basic stats
    results["stats"]["total_tracks"] = len(manifest)
    results["stats"]["total_samples_manifest"] = sum(len(t.get("samples", [])) for t in manifest)
    
    # Count actual files
    track_dirs = list(harvest_dir.glob("track_*"))
    results["stats"]["track_directories"] = len(track_dirs)
    
    actual_images = list(harvest_dir.glob("track_*/*.jpg"))
    results["stats"]["actual_images"] = len(actual_images)
    
    # Check manifest vs filesystem consistency
    manifest_samples = results["stats"]["total_samples_manifest"]
    actual_samples = results["stats"]["actual_images"]
    
    if manifest_samples != actual_samples:
        results["warnings"].append(
            f"Sample count mismatch: manifest={manifest_samples}, filesystem={actual_samples}"
        )
    
    # Analyze tracks
    track_stats = []
    quality_distribution = []
    sharpness_distribution = []
    frontalness_distribution = []
    area_distribution = []
    
    labeled_tracks = 0
    empty_tracks = 0
    tracks_with_issues = []
    
    for track in manifest:
        track_id = track.get("track_id")
        samples = track.get("samples", [])
        label = track.get("label")
        
        track_info = {
            "track_id": track_id,
            "label": label,
            "sample_count": len(samples),
            "byte_track_id": track.get("byte_track_id"),
            "avg_conf": track.get("avg_conf", 0),
            "duration_ms": track.get("last_ts_ms", 0) - track.get("first_ts_ms", 0),
        }
        
        if label:
            labeled_tracks += 1
        
        if len(samples) == 0:
            empty_tracks += 1
            tracks_with_issues.append({
                "track_id": track_id,
                "issue": "no_samples",
                "label": label
            })
        
        # Sample quality stats
        sample_qualities = []
        sample_sharpness = []
        sample_frontalness = []
        sample_areas = []
        
        for sample in samples:
            if "quality" in sample:
                sample_qualities.append(sample["quality"])
                quality_distribution.append(sample["quality"])
            if "sharpness" in sample:
                sample_sharpness.append(sample["sharpness"])
                sharpness_distribution.append(sample["sharpness"])
            if "frontalness" in sample:
                sample_frontalness.append(sample["frontalness"])
                frontalness_distribution.append(sample["frontalness"])
            if "area_frac" in sample:
                sample_areas.append(sample["area_frac"])
                area_distribution.append(sample["area_frac"])
            
            # Check if file exists
            sample_path = Path(sample.get("path", ""))
            if not sample_path.exists():
                tracks_with_issues.append({
                    "track_id": track_id,
                    "issue": "missing_file",
                    "path": str(sample_path)
                })
        
        if sample_qualities:
            track_info["avg_quality"] = sum(sample_qualities) / len(sample_qualities)
            track_info["min_quality"] = min(sample_qualities)
            track_info["max_quality"] = max(sample_qualities)
        
        if sample_sharpness:
            track_info["avg_sharpness"] = sum(sample_sharpness) / len(sample_sharpness)
        
        if sample_frontalness:
            track_info["avg_frontalness"] = sum(sample_frontalness) / len(sample_frontalness)
        
        track_stats.append(track_info)
    
    results["stats"]["labeled_tracks"] = labeled_tracks
    results["stats"]["unlabeled_tracks"] = len(manifest) - labeled_tracks
    results["stats"]["empty_tracks"] = empty_tracks
    
    # Distribution stats
    if quality_distribution:
        results["stats"]["quality_mean"] = sum(quality_distribution) / len(quality_distribution)
        results["stats"]["quality_min"] = min(quality_distribution)
        results["stats"]["quality_max"] = max(quality_distribution)
    
    if sharpness_distribution:
        results["stats"]["sharpness_mean"] = sum(sharpness_distribution) / len(sharpness_distribution)
        results["stats"]["sharpness_min"] = min(sharpness_distribution)
        results["stats"]["sharpness_max"] = max(sharpness_distribution)
    
    if frontalness_distribution:
        results["stats"]["frontalness_mean"] = sum(frontalness_distribution) / len(frontalness_distribution)
        results["stats"]["frontalness_min"] = min(frontalness_distribution)
        results["stats"]["frontalness_max"] = max(frontalness_distribution)
    
    if area_distribution:
        results["stats"]["area_frac_mean"] = sum(area_distribution) / len(area_distribution)
        results["stats"]["area_frac_min"] = min(area_distribution)
        results["stats"]["area_frac_max"] = max(area_distribution)
    
    # Store detailed data
    results["track_stats"] = track_stats
    results["tracks_with_issues"] = tracks_with_issues
    
    # Check for common issues
    if empty_tracks > 0:
        results["warnings"].append(f"{empty_tracks} tracks have no samples")
    
    if len(tracks_with_issues) > len(manifest) * 0.1:
        results["warnings"].append(f"{len(tracks_with_issues)} tracks have issues (>10%)")
    
    if results["stats"]["unlabeled_tracks"] > len(manifest) * 0.5:
        results["warnings"].append(f"{results['stats']['unlabeled_tracks']} tracks are unlabeled (>50%)")
    
    return results


def print_validation_report(results: Dict, show_tracks: bool = False) -> None:
    """Print validation report to console."""
    
    print("\n" + "="*80)
    print("HARVEST VALIDATION REPORT")
    print("="*80)
    
    # Status
    if results["valid"]:
        print("✓ Status: VALID")
    else:
        print("✗ Status: INVALID")
    
    # Errors
    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"  • {error}")
    
    # Warnings
    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"  • {warning}")
    
    # Stats
    stats = results["stats"]
    print("\n📊 STATISTICS:")
    print("-"*80)
    print(f"Total tracks:          {stats.get('total_tracks', 0)}")
    print(f"  Labeled:             {stats.get('labeled_tracks', 0)}")
    print(f"  Unlabeled:           {stats.get('unlabeled_tracks', 0)}")
    print(f"  Empty:               {stats.get('empty_tracks', 0)}")
    print(f"\nTotal samples:         {stats.get('total_samples_manifest', 0)}")
    print(f"Actual image files:    {stats.get('actual_images', 0)}")
    print(f"Track directories:     {stats.get('track_directories', 0)}")
    
    if "quality_mean" in stats:
        print(f"\n📈 QUALITY METRICS:")
        print(f"  Average quality:     {stats['quality_mean']:.3f}")
        print(f"  Quality range:       {stats['quality_min']:.3f} - {stats['quality_max']:.3f}")
    
    if "sharpness_mean" in stats:
        print(f"  Average sharpness:   {stats['sharpness_mean']:.1f}")
        print(f"  Sharpness range:     {stats['sharpness_min']:.1f} - {stats['sharpness_max']:.1f}")
    
    if "frontalness_mean" in stats:
        print(f"  Average frontalness: {stats['frontalness_mean']:.3f}")
        print(f"  Frontalness range:   {stats['frontalness_min']:.3f} - {stats['frontalness_max']:.3f}")
    
    if "area_frac_mean" in stats:
        print(f"  Average area frac:   {stats['area_frac_mean']:.4f}")
        print(f"  Area frac range:     {stats['area_frac_min']:.4f} - {stats['area_frac_max']:.4f}")
    
    # Track details
    if show_tracks and "track_stats" in results:
        print("\n📋 PER-TRACK DETAILS:")
        print("-"*80)
        
        df = pd.DataFrame(results["track_stats"])
        
        # Sort by sample count descending
        df = df.sort_values("sample_count", ascending=False)
        
        print(df.to_string(index=False))
    
    # Issues
    if results.get("tracks_with_issues"):
        print(f"\n⚠️  TRACKS WITH ISSUES: ({len(results['tracks_with_issues'])})")
        print("-"*80)
        
        issue_counts = Counter(t["issue"] for t in results["tracks_with_issues"])
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count}")
    
    print("\n" + "="*80)


def save_validation_report(results: Dict, output_dir: Path) -> None:
    """Save validation report to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    json_path = output_dir / "validation_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    LOGGER.info("Validation results saved to: %s", json_path)
    
    # Save track stats as CSV
    if "track_stats" in results:
        df = pd.DataFrame(results["track_stats"])
        csv_path = output_dir / "track_stats.csv"
        df.to_csv(csv_path, index=False)
        LOGGER.info("Track statistics saved to: %s", csv_path)
    
    # Save issues as CSV
    if results.get("tracks_with_issues"):
        df_issues = pd.DataFrame(results["tracks_with_issues"])
        issues_path = output_dir / "track_issues.csv"
        df_issues.to_csv(issues_path, index=False)
        LOGGER.info("Track issues saved to: %s", issues_path)


def main() -> None:
    args = parse_args()
    setup_logging()
    
    LOGGER.info("Validating harvest directory: %s", args.harvest_dir)
    
    results = validate_harvest(args.harvest_dir)
    
    print_validation_report(results, show_tracks=args.show_tracks)
    
    if args.output:
        save_validation_report(results, args.output)
    
    # Exit code
    if not results["valid"]:
        exit(1)


if __name__ == "__main__":
    main()
