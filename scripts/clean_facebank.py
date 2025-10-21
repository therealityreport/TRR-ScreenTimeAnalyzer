#!/usr/bin/env python3
"""Clean up facebank by removing poor quality images from all identities."""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime


def analyze_image_quality(image_path):
    """Analyze image for darkness, overexposure, blur, and contrast."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {'valid': False}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Check for overexposure (too many bright pixels)
    overexposed_pixels = np.sum(gray > 235) / gray.size
    
    # Check for underexposure (too many dark pixels)
    underexposed_pixels = np.sum(gray < 35) / gray.size
    
    return {
        'valid': True,
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'sharpness': laplacian_var,
        'overexposed_ratio': overexposed_pixels,
        'underexposed_ratio': underexposed_pixels,
    }


def should_remove(metrics, aggressive=False):
    """Decide if image should be removed based on quality metrics."""
    if not metrics['valid']:
        return True, "Invalid image"
    
    reasons = []
    
    # Brightness thresholds
    dark_threshold = 70 if aggressive else 50
    bright_threshold = 170 if aggressive else 190
    
    # Sharpness threshold
    blur_threshold = 100 if aggressive else 50
    
    # Quality checks
    if metrics['mean_brightness'] < dark_threshold:
        reasons.append(f"Too dark (bright={metrics['mean_brightness']:.0f})")
    
    if metrics['mean_brightness'] > bright_threshold or metrics['overexposed_ratio'] > 0.10:
        reasons.append(f"Overexposed (bright={metrics['mean_brightness']:.0f})")
    
    if metrics['sharpness'] < blur_threshold:
        reasons.append(f"Blurry (sharp={metrics['sharpness']:.0f})")
    
    if metrics['underexposed_ratio'] > 0.15:
        reasons.append(f"Underexposed areas ({metrics['underexposed_ratio']:.1%})")
    
    if metrics['std_brightness'] < 25:
        reasons.append(f"Low contrast (std={metrics['std_brightness']:.0f})")
    
    # Remove if has issues
    if reasons:
        return True, "; ".join(reasons)
    
    return False, None


def clean_identity_folder(identity_dir, backup_root, removed_root, aggressive=False, dry_run=False):
    """Clean a single identity folder."""
    identity_name = identity_dir.name
    
    # Skip backup and removed directories
    if identity_name.endswith('_BACKUP') or identity_name.endswith('_REMOVED'):
        return None
    
    print(f"\n{'='*70}")
    print(f"Processing: {identity_name}")
    print(f"{'='*70}")
    
    # Create backup
    backup_dir = backup_root / f"{identity_name}_BACKUP"
    if not backup_dir.exists() and not dry_run:
        shutil.copytree(identity_dir, backup_dir)
        print(f"✓ Created backup: {backup_dir}")
    
    # Create removed directory
    removed_dir = removed_root / f"{identity_name}_REMOVED"
    if not dry_run:
        removed_dir.mkdir(exist_ok=True)
    
    # Get all images
    images = sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.jpeg")) + sorted(identity_dir.glob("*.png"))
    
    if not images:
        print(f"⚠️  No images found in {identity_name}")
        return None
    
    print(f"Analyzing {len(images)} images...\n")
    
    keep_count = 0
    remove_count = 0
    kept_images = []
    removed_images = []
    
    for img_path in images:
        metrics = analyze_image_quality(img_path)
        should_rm, reason = should_remove(metrics, aggressive=aggressive)
        
        if should_rm:
            remove_count += 1
            removed_images.append({
                'filename': img_path.name,
                'reason': reason,
                'brightness': metrics.get('mean_brightness', 0),
                'sharpness': metrics.get('sharpness', 0)
            })
            
            if not dry_run:
                dest = removed_dir / img_path.name
                shutil.move(str(img_path), str(dest))
            
            print(f"❌ REMOVE: {img_path.name}")
            print(f"   Reason: {reason}\n")
        else:
            keep_count += 1
            kept_images.append({
                'filename': img_path.name,
                'brightness': metrics.get('mean_brightness', 0),
                'sharpness': metrics.get('sharpness', 0)
            })
            print(f"✅ KEEP: {img_path.name} (bright={metrics['mean_brightness']:.0f}, sharp={metrics['sharpness']:.0f})")
    
    print(f"\n{'-'*70}")
    print(f"Summary for {identity_name}:")
    print(f"  Original: {len(images)} images")
    print(f"  Kept: {keep_count} ({keep_count/len(images)*100:.1f}%)")
    print(f"  Removed: {remove_count} ({remove_count/len(images)*100:.1f}%)")
    
    if remove_count == 0:
        print(f"  ✓ All images passed quality check!")
    elif keep_count < 10:
        print(f"  ⚠️  WARNING: Only {keep_count} images remaining - may need more samples")
    
    return {
        'identity': identity_name,
        'original_count': len(images),
        'kept_count': keep_count,
        'removed_count': remove_count,
        'kept_images': kept_images,
        'removed_images': removed_images
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Clean up facebank by removing poor quality images")
    parser.add_argument(
        '--facebank-dir',
        type=Path,
        default=Path('data/facebank'),
        help='Path to facebank directory (default: data/facebank)'
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use more aggressive quality filtering'
    )
    parser.add_argument(
        '--identity',
        type=str,
        help='Only process a specific identity (e.g., BRANDI)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing anything'
    )
    parser.add_argument(
        '--min-images',
        type=int,
        default=10,
        help='Warn if an identity has fewer than this many images after cleanup (default: 10)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    facebank_dir = args.facebank_dir.expanduser().resolve()
    
    if not facebank_dir.exists():
        print(f"❌ Facebank directory not found: {facebank_dir}")
        return
    
    # Create root directories for backups and removed images
    backup_root = facebank_dir.parent / 'facebank_backups'
    removed_root = facebank_dir.parent / 'facebank_removed'
    
    if not args.dry_run:
        backup_root.mkdir(exist_ok=True)
        removed_root.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("FACEBANK CLEANUP UTILITY")
    print("="*70)
    print(f"Facebank directory: {facebank_dir}")
    print(f"Aggressive mode: {'YES' if args.aggressive else 'NO'}")
    print(f"Dry run: {'YES (no files will be modified)' if args.dry_run else 'NO'}")
    if not args.dry_run:
        print(f"Backups: {backup_root}")
        print(f"Removed images: {removed_root}")
    
    # Get all identity directories
    if args.identity:
        identity_dirs = [facebank_dir / args.identity]
        if not identity_dirs[0].exists():
            print(f"\n❌ Identity '{args.identity}' not found in facebank")
            return
    else:
        identity_dirs = [d for d in facebank_dir.iterdir() if d.is_dir()]
    
    # Filter out backup and removed directories
    identity_dirs = [d for d in identity_dirs 
                     if not d.name.endswith('_BACKUP') 
                     and not d.name.endswith('_REMOVED')]
    
    if not identity_dirs:
        print("\n❌ No identity folders found in facebank")
        return
    
    # Process each identity
    results = []
    for identity_dir in sorted(identity_dirs):
        result = clean_identity_folder(
            identity_dir, 
            backup_root, 
            removed_root, 
            aggressive=args.aggressive,
            dry_run=args.dry_run
        )
        if result:
            results.append(result)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total_original = sum(r['original_count'] for r in results)
    total_kept = sum(r['kept_count'] for r in results)
    total_removed = sum(r['removed_count'] for r in results)
    
    print(f"\nProcessed {len(results)} identities")
    print(f"Total images: {total_original}")
    print(f"Kept: {total_kept} ({total_kept/total_original*100:.1f}%)")
    print(f"Removed: {total_removed} ({total_removed/total_original*100:.1f}%)")
    
    print("\n" + "-"*70)
    print("Per-identity breakdown:")
    print("-"*70)
    for result in sorted(results, key=lambda x: x['kept_count']):
        status = "⚠️" if result['kept_count'] < args.min_images else "✓"
        print(f"{status} {result['identity']:15s}: {result['kept_count']:3d} kept, {result['removed_count']:3d} removed")
    
    if not args.dry_run:
        print(f"\n💡 To rebuild facebank, run:")
        print(f"   python3 scripts/build_facebank.py --facebank-dir {facebank_dir} --output-dir data")
    else:
        print(f"\n💡 This was a dry run. Re-run without --dry-run to actually remove images.")
    
    print("="*70)


if __name__ == '__main__':
    main()
