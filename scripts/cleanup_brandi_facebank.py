#!/usr/bin/env python3
"""Clean up BRANDI's facebank by removing poor quality images."""

import cv2
import numpy as np
from pathlib import Path
import shutil

def analyze_image_quality(image_path):
    """Analyze image for darkness, overexposure, and blur."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {'valid': False}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Check for overexposure (too many bright pixels)
    overexposed_pixels = np.sum(gray > 240) / gray.size
    
    # Check for underexposure (too many dark pixels)
    underexposed_pixels = np.sum(gray < 30) / gray.size
    
    return {
        'valid': True,
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'sharpness': laplacian_var,
        'overexposed_ratio': overexposed_pixels,
        'underexposed_ratio': underexposed_pixels,
        'is_too_dark': mean_brightness < 60,
        'is_too_bright': mean_brightness > 180 or overexposed_pixels > 0.15,
        'is_low_contrast': std_brightness < 30,
        'is_blurry': laplacian_var < 50
    }

def should_remove(metrics):
    """Decide if image should be removed based on quality metrics."""
    if not metrics['valid']:
        return True, "Invalid image"
    
    reasons = []
    
    if metrics['is_too_dark']:
        reasons.append(f"Too dark (brightness={metrics['mean_brightness']:.1f})")
    
    if metrics['is_too_bright']:
        reasons.append(f"Overexposed (brightness={metrics['mean_brightness']:.1f}, ratio={metrics['overexposed_ratio']:.2%})")
    
    if metrics['is_low_contrast']:
        reasons.append(f"Low contrast (std={metrics['std_brightness']:.1f})")
    
    if metrics['is_blurry']:
        reasons.append(f"Blurry (sharpness={metrics['sharpness']:.1f})")
    
    # Remove if has 2+ serious issues
    if len(reasons) >= 2:
        return True, "; ".join(reasons)
    
    # Remove very dark or very overexposed regardless
    if metrics['mean_brightness'] < 50 or metrics['mean_brightness'] > 190:
        return True, reasons[0] if reasons else "Extreme brightness"
    
    return False, None

def main():
    brandi_dir = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank/BRANDI")
    backup_dir = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank/BRANDI_BACKUP")
    removed_dir = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank/BRANDI_REMOVED")
    
    # Create backup
    if not backup_dir.exists():
        print(f"Creating backup: {backup_dir}")
        shutil.copytree(brandi_dir, backup_dir)
    
    # Create removed directory
    removed_dir.mkdir(exist_ok=True)
    
    images = sorted(brandi_dir.glob("*.jpg"))
    print(f"\nAnalyzing {len(images)} images in BRANDI facebank...\n")
    
    keep_count = 0
    remove_count = 0
    
    results = []
    
    for img_path in images:
        metrics = analyze_image_quality(img_path)
        should_rm, reason = should_remove(metrics)
        
        results.append({
            'filename': img_path.name,
            'remove': should_rm,
            'reason': reason,
            'brightness': metrics.get('mean_brightness', 0),
            'sharpness': metrics.get('sharpness', 0)
        })
        
        if should_rm:
            remove_count += 1
            # Move to removed directory
            dest = removed_dir / img_path.name
            shutil.move(str(img_path), str(dest))
            print(f"❌ REMOVED: {img_path.name}")
            print(f"   Reason: {reason}\n")
        else:
            keep_count += 1
            print(f"✅ KEPT: {img_path.name}")
            if metrics['valid']:
                print(f"   Brightness={metrics['mean_brightness']:.1f}, Sharpness={metrics['sharpness']:.1f}\n")
    
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Original: {len(images)} images")
    print(f"  Kept: {keep_count} images")
    print(f"  Removed: {remove_count} images")
    print(f"\n  Backup: {backup_dir}")
    print(f"  Removed images: {removed_dir}")
    print("="*60)
    
    # Show top 10 kept images by quality
    kept_results = [r for r in results if not r['remove']]
    kept_results.sort(key=lambda x: (x['sharpness'], x['brightness']), reverse=True)
    
    print("\n📊 Top 10 Best Quality Images (kept):")
    for i, r in enumerate(kept_results[:10], 1):
        print(f"{i:2d}. {r['filename']} (sharp={r['sharpness']:.1f}, bright={r['brightness']:.1f})")

if __name__ == "__main__":
    main()
