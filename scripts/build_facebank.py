#!/usr/bin/env python3
"""CLI for building the ArcFace facebank."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from screentime.io_utils import ensure_dir, setup_logging
from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.recognition.facebank import FacebankArtifacts, build_facebank


LOGGER = logging.getLogger("scripts.facebank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build facebank embeddings using ArcFace")
    parser.add_argument(
        "--facebank-dir",
        type=Path,
        default=Path("data/facebank"),
        help="Directory containing labeled face images (per-person subdirectories)",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Alias for --facebank-dir to match older runbooks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where facebank artifacts will be written",
    )
    parser.add_argument(
        "--arcface-model",
        type=str,
        default=None,
        help="Optional path or InsightFace model name for ArcFace",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Reserved hook for future detector-assisted alignment (currently unused)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        help="Execution providers for ONNXRuntime (e.g. CUDAExecutionProvider)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.reference_root is not None:
        args.facebank_dir = args.reference_root
    if args.detector:
        LOGGER.info("Detector argument provided (%s) but alignment currently uses ArcFace defaults", args.detector)

    embedder = ArcFaceEmbedder(model_path=args.arcface_model, providers=args.providers)
    ensure_dir(args.output_dir)

    artifacts: FacebankArtifacts = build_facebank(
        facebank_dir=args.facebank_dir,
        output_dir=args.output_dir,
        embedder=embedder,
        aligner=None,
    )

    LOGGER.info(
        "Facebank built: parquet=%s meta=%s samples=%s",
        artifacts.parquet_path,
        artifacts.meta_json_path,
        artifacts.samples_csv_path,
    )


if __name__ == "__main__":
    main()
