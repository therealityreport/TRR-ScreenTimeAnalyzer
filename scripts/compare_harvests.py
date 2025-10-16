#!/usr/bin/env python3
"""
Compare harvest results between two runs (before/after configuration changes).

Usage:
    # After running harvest with original config, move results:
    mv data/harvest/RHOBH-TEST data/harvest/RHOBH-TEST_original
    
    # Run harvest with new config
    python scripts/harvest_faces.py ...
    
    # Compare:
    python scripts/compare_harvests.py \
        --before data/harvest/RHOBH-TEST_original \
        --after data/harvest/RHOBH-TEST \
        --output diagnostics/harvest_comparison
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from screentime.io_utils import load_json, setup_logging

import logging
LOGGER = logging.getLogger("scripts.compare_harvests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two harvest runs")
    parser.add_argument("--before", type=Path, required=True, help="Original harvest directory")
    parser.add_argument("--after", type=Path, required=True, help="New harvest directory")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/comparison"), 
                       help="Output directory for comparison report")
    return parser.parse_args()


def analyze_harvest_dir(harvest_dir: Path) -> Dict:
    """Analyze a harvest directory and return statistics."""
    manifest_path = harvest_dir / "manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = load_json(manifest_path)
    
    stats = {
        "total_tracks": len(manifest),
        "total_samples": 0,
        "samples_per_track": [],
        "avg_confidence": [],
        "avg_area": [],
        "quality_scores": [],
        "sharpness_scores": [],
        "frontalness_scores": [],
        "track_durations_ms": [],
        "labeled_tracks": 0,
        "unlabeled_tracks": 0,
    }
    
    for track in manifest:
        samples = track.get("samples", [])
        stats["total_samples"] += len(samples)
        stats["samples_per_track"].append(len(samples))
        
        if track.get("label"):
            stats["labeled_tracks"] += 1
        else:
            stats["unlabeled_tracks"] += 1
        
        # Aggregate sample stats
        for sample in samples:
            if "quality" in sample:
                stats["quality_scores"].append(sample["quality"])
            if "sharpness" in sample:
                stats["sharpness_scores"].append(sample["sharpness"])
            if "frontalness" in sample:
                stats["frontalness_scores"].append(sample["frontalness"])
        
        # Track-level stats
        if track.get("avg_conf"):
            stats["avg_confidence"].append(track["avg_conf"])
        if track.get("avg_area"):
            stats["avg_area"].append(track["avg_area"])
        
        # Duration
        first_ts = track.get("first_ts_ms", 0)
        last_ts = track.get("last_ts_ms", 0)
        duration = last_ts - first_ts
        stats["track_durations_ms"].append(duration)
    
    # Count actual image files
    stats["actual_image_count"] = len(list(harvest_dir.glob("track_*/*.jpg")))
    
    return stats


def compute_summary_stats(stats: Dict) -> Dict:
    """Compute summary statistics from raw stats."""
    import numpy as np
    
    summary = {}
    
    # Tracks
    summary["total_tracks"] = stats["total_tracks"]
    summary["labeled_tracks"] = stats["labeled_tracks"]
    summary["unlabeled_tracks"] = stats["unlabeled_tracks"]
    summary["labeled_pct"] = (stats["labeled_tracks"] / stats["total_tracks"] * 100) if stats["total_tracks"] > 0 else 0
    
    # Samples
    summary["total_samples"] = stats["total_samples"]
    summary["actual_images"] = stats["actual_image_count"]
    summary["avg_samples_per_track"] = np.mean(stats["samples_per_track"]) if stats["samples_per_track"] else 0
    summary["median_samples_per_track"] = np.median(stats["samples_per_track"]) if stats["samples_per_track"] else 0
    summary["min_samples_per_track"] = np.min(stats["samples_per_track"]) if stats["samples_per_track"] else 0
    summary["max_samples_per_track"] = np.max(stats["samples_per_track"]) if stats["samples_per_track"] else 0
    
    # Quality
    if stats["quality_scores"]:
        summary["avg_quality"] = np.mean(stats["quality_scores"])
        summary["median_quality"] = np.median(stats["quality_scores"])
    else:
        summary["avg_quality"] = 0
        summary["median_quality"] = 0
    
    if stats["sharpness_scores"]:
        summary["avg_sharpness"] = np.mean(stats["sharpness_scores"])
        summary["median_sharpness"] = np.median(stats["sharpness_scores"])
    else:
        summary["avg_sharpness"] = 0
        summary["median_sharpness"] = 0
    
    if stats["frontalness_scores"]:
        summary["avg_frontalness"] = np.mean(stats["frontalness_scores"])
        summary["median_frontalness"] = np.median(stats["frontalness_scores"])
    else:
        summary["avg_frontalness"] = 0
        summary["median_frontalness"] = 0
    
    # Track characteristics
    if stats["avg_confidence"]:
        summary["avg_track_confidence"] = np.mean(stats["avg_confidence"])
    else:
        summary["avg_track_confidence"] = 0
    
    if stats["avg_area"]:
        summary["avg_track_area"] = np.mean(stats["avg_area"])
    else:
        summary["avg_track_area"] = 0
    
    if stats["track_durations_ms"]:
        summary["avg_track_duration_ms"] = np.mean(stats["track_durations_ms"])
        summary["median_track_duration_ms"] = np.median(stats["track_durations_ms"])
    else:
        summary["avg_track_duration_ms"] = 0
        summary["median_track_duration_ms"] = 0
    
    return summary


def generate_comparison_report(before_summary: Dict, after_summary: Dict, output_dir: Path) -> None:
    """Generate comparison report."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate deltas
    comparisons = []
    
    for key in before_summary.keys():
        before_val = before_summary[key]
        after_val = after_summary[key]
        
        if isinstance(before_val, (int, float)):
            delta = after_val - before_val
            if before_val != 0:
                pct_change = (delta / before_val) * 100
            else:
                pct_change = float('inf') if after_val > 0 else 0
            
            comparisons.append({
                "metric": key,
                "before": before_val,
                "after": after_val,
                "delta": delta,
                "pct_change": pct_change
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparisons)
    
    # Save to CSV
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    
    # Generate text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("HARVEST COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Key metrics
    report_lines.append("KEY METRICS:")
    report_lines.append("-"*80)
    
    key_metrics = [
        "total_tracks",
        "total_samples",
        "actual_images",
        "avg_samples_per_track",
        "labeled_tracks",
        "avg_quality",
        "avg_sharpness",
        "avg_frontalness"
    ]
    
    for metric in key_metrics:
        row = df[df["metric"] == metric].iloc[0]
        before = row["before"]
        after = row["after"]
        delta = row["delta"]
        pct = row["pct_change"]
        
        # Format based on metric type
        if "pct" in metric or metric in ["avg_quality", "avg_sharpness", "avg_frontalness"]:
            before_str = f"{before:.2f}"
            after_str = f"{after:.2f}"
            delta_str = f"{delta:+.2f}"
        else:
            before_str = f"{int(before)}"
            after_str = f"{int(after)}"
            delta_str = f"{int(delta):+d}"
        
        if pct != float('inf'):
            pct_str = f"({pct:+.1f}%)"
        else:
            pct_str = "(new)"
        
        # Determine if improvement
        improvement_metrics = [
            "total_samples", "actual_images", "avg_samples_per_track",
            "labeled_tracks", "avg_quality", "avg_sharpness"
        ]
        
        if metric in improvement_metrics:
            indicator = "✓" if delta > 0 else "✗" if delta < 0 else "="
        else:
            indicator = " "
        
        report_lines.append(
            f"{indicator} {metric:.<40} {before_str:>12} → {after_str:<12} {delta_str:>12} {pct_str}"
        )
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("ASSESSMENT:")
    report_lines.append("="*80)
    
    # Generate assessment
    assessment = []
    
    sample_increase = df[df["metric"] == "total_samples"]["pct_change"].values[0]
    if sample_increase > 50:
        assessment.append("✓ EXCELLENT: Total samples increased by >50%")
    elif sample_increase > 20:
        assessment.append("✓ GOOD: Total samples increased by >20%")
    elif sample_increase > 0:
        assessment.append("~ MODERATE: Total samples increased slightly")
    else:
        assessment.append("✗ CONCERN: Total samples decreased")
    
    quality_change = df[df["metric"] == "avg_quality"]["delta"].values[0]
    if abs(quality_change) < 0.05:
        assessment.append("✓ Quality maintained (change < 0.05)")
    elif quality_change < -0.10:
        assessment.append("⚠ WARNING: Average quality decreased significantly")
    else:
        assessment.append("~ Quality changed but within acceptable range")
    
    track_increase = df[df["metric"] == "total_tracks"]["pct_change"].values[0]
    if track_increase > 10:
        assessment.append("✓ More tracks detected")
    elif track_increase < -10:
        assessment.append("~ Fewer tracks (may indicate better consolidation)")
    
    for line in assessment:
        report_lines.append(line)
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append(f"Detailed comparison saved to: {csv_path}")
    report_lines.append("="*80)
    
    # Write report
    report_text = "\n".join(report_lines)
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text)
    
    # Print to console
    print(report_text)
    
    return report_path


def main() -> None:
    args = parse_args()
    setup_logging()
    
    LOGGER.info("Analyzing BEFORE harvest: %s", args.before)
    before_stats = analyze_harvest_dir(args.before)
    before_summary = compute_summary_stats(before_stats)
    
    LOGGER.info("Analyzing AFTER harvest: %s", args.after)
    after_stats = analyze_harvest_dir(args.after)
    after_summary = compute_summary_stats(after_stats)
    
    LOGGER.info("Generating comparison report...")
    report_path = generate_comparison_report(before_summary, after_summary, args.output)
    
    LOGGER.info("Comparison complete! Report saved to: %s", report_path)


if __name__ == "__main__":
    main()
