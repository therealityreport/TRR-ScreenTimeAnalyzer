#!/usr/bin/env python3
"""
FIXED Comprehensive Facebank Cleanup - MUCH MORE LENIENT
Always keeps AT LEAST 15 best images per person, even if quality is marginal.
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Tuple
from collections import defaultdict
import hashlib

class ImageQualityAnalyzer:
    """Analyze image quality metrics."""
    
    @staticmethod
    def analyze(image_path: Path) -> Dict:
        """Comprehensive image quality analysis."""
        img = cv2.imread(str(image_path))
        if img is None:
            return {'valid': False, 'error': 'Cannot read image'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Brightness metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Exposure analysis
        overexposed = np.sum(gray > 240) / gray.size
        underexposed = np.sum(gray < 30) / gray.size
        
        # Contrast analysis
        contrast = gray.max() - gray.min()
        
        # Color distribution (for detecting lighting variety)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:,:,0])
        saturation_mean = np.mean(hsv[:,:,1])
        
        # Edge density (detail richness)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'valid': True,
            'path': image_path,
            'brightness': mean_brightness,
            'brightness_std': std_brightness,
            'sharpness': laplacian_var,
            'contrast': contrast,
            'overexposed_ratio': overexposed,
            'underexposed_ratio': underexposed,
            'hue_variety': hue_std,
            'saturation': saturation_mean,
            'edge_density': edge_density,
            'image': img,
            'gray': gray
        }
    
    @staticmethod
    def compute_perceptual_hash(gray_img: np.ndarray, hash_size: int = 8) -> str:
        """Compute perceptual hash for duplicate detection."""
        # Resize to hash_size x hash_size
        resized = cv2.resize(gray_img, (hash_size, hash_size))
        # Compute DCT
        dct = cv2.dct(np.float32(resized))
        # Take top-left 8x8
        dct_low = dct[:hash_size, :hash_size]
        # Compute median
        median = np.median(dct_low)
        # Create hash
        diff = dct_low > median
        # Convert to hex string
        return hashlib.md5(diff.tobytes()).hexdigest()
    
    @staticmethod
    def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute histogram similarity (0=different, 1=identical)."""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


class QualityFilter:
    """Filter images based on quality thresholds - MUCH MORE LENIENT."""
    
    # ⚠️ LENIENT Quality thresholds - only reject truly unusable images
    MIN_BRIGHTNESS = 35         # Was 50, now MUCH more lenient
    MAX_BRIGHTNESS = 220        # Was 200, now more lenient
    MIN_SHARPNESS = 25          # Was 80, now VERY lenient (only reject truly blurry)
    MIN_CONTRAST = 15           # Was 40, now VERY lenient
    MAX_OVEREXPOSED = 0.40      # Was 0.20, now much more lenient
    MAX_UNDEREXPOSED = 0.50     # Was 0.25, now much more lenient
    MIN_BRIGHTNESS_STD = 10     # Was 25, now VERY lenient
    
    @classmethod
    def should_remove(cls, metrics: Dict) -> Tuple[bool, str]:
        """Determine if image should be removed - ONLY reject truly bad images."""
        if not metrics['valid']:
            return True, "Invalid/unreadable image"
        
        reasons = []
        critical_failures = []
        
        # CRITICAL failures ONLY (immediate removal)
        if metrics['brightness'] < cls.MIN_BRIGHTNESS:
            critical_failures.append(f"Extremely dark ({metrics['brightness']:.1f})")
        
        if metrics['brightness'] > cls.MAX_BRIGHTNESS:
            critical_failures.append(f"Extremely bright ({metrics['brightness']:.1f})")
        
        # Only remove if CRITICALLY over/underexposed
        if metrics['overexposed_ratio'] > cls.MAX_OVEREXPOSED:
            critical_failures.append(f"Severely overexposed ({metrics['overexposed_ratio']:.1%})")
        
        if metrics['underexposed_ratio'] > cls.MAX_UNDEREXPOSED:
            critical_failures.append(f"Severely underexposed ({metrics['underexposed_ratio']:.1%})")
        
        # Non-critical quality issues (for information only)
        if metrics['sharpness'] < cls.MIN_SHARPNESS:
            reasons.append(f"Very blurry (sharpness={metrics['sharpness']:.1f})")
        
        if metrics['contrast'] < cls.MIN_CONTRAST:
            reasons.append(f"Extremely low contrast ({metrics['contrast']})")
        
        # ONLY remove if there's a CRITICAL failure
        # Or if there are 3+ severe issues combined
        if critical_failures:
            return True, "; ".join(critical_failures)
        
        if len(reasons) >= 3:
            return True, "; ".join(reasons)
        
        return False, None
    
    @classmethod
    def compute_quality_score(cls, metrics: Dict) -> float:
        """Compute overall quality score (0-100)."""
        # Normalize metrics to 0-1 range
        brightness_score = 1.0 - abs(metrics['brightness'] - 115) / 115  # Optimal at 115
        sharpness_score = min(metrics['sharpness'] / 400, 1.0)  # Cap at 400 (more lenient)
        contrast_score = min(metrics['contrast'] / 120, 1.0)  # Cap at 120 (more lenient)
        edge_score = min(metrics['edge_density'] * 2, 1.0)  # Cap at 0.5
        
        # Penalize over/underexposure (but less severely)
        exposure_penalty = max(metrics['overexposed_ratio'], 
                              metrics['underexposed_ratio']) * 1.5  # Was *2, now *1.5
        
        # Weighted combination
        score = (
            brightness_score * 0.20 +
            sharpness_score * 0.35 +
            contrast_score * 0.20 +
            edge_score * 0.15 +
            (1 - exposure_penalty) * 0.10
        )
        
        return score * 100


class DiversitySelector:
    """Select diverse, representative images."""
    
    SIMILARITY_THRESHOLD = 0.88  # Was 0.85, now stricter to keep more variety
    MIN_IMAGES_PER_PERSON = 15   # GUARANTEED MINIMUM
    MAX_IMAGES_PER_PERSON = 40   # Was 35, now higher
    
    @classmethod
    def select_diverse_images(cls, all_metrics: List[Dict]) -> List[Dict]:
        """Select diverse, high-quality images. ALWAYS keeps at least MIN_IMAGES_PER_PERSON."""
        
        # SAFETY: If we have MIN or fewer, keep ALL of them
        if len(all_metrics) <= cls.MIN_IMAGES_PER_PERSON:
            print(f"  ✅ {len(all_metrics)} images - keeping ALL (at/below minimum threshold)")
            return all_metrics
        
        # Step 1: Group by perceptual hash (exact/near duplicates)
        hash_groups = defaultdict(list)
        analyzer = ImageQualityAnalyzer()
        
        for metrics in all_metrics:
            phash = analyzer.compute_perceptual_hash(metrics['gray'])
            hash_groups[phash].append(metrics)
        
        # Step 2: From each hash group, keep only the best quality image
        unique_images = []
        for phash, group in hash_groups.items():
            # Sort by quality score
            group_sorted = sorted(group, 
                                key=lambda m: QualityFilter.compute_quality_score(m),
                                reverse=True)
            unique_images.append(group_sorted[0])
        
        print(f"  📊 After deduplication: {len(unique_images)} unique images "
              f"(removed {len(all_metrics) - len(unique_images)} duplicates)")
        
        # SAFETY: If deduplication brought us to MIN or below, keep all unique images
        if len(unique_images) <= cls.MIN_IMAGES_PER_PERSON:
            print(f"  ✅ At minimum threshold - keeping ALL unique images")
            return unique_images
        
        # Step 3: If still too many, cluster by visual similarity
        if len(unique_images) > cls.MAX_IMAGES_PER_PERSON:
            selected = cls._cluster_and_select(unique_images)
            print(f"  📊 After diversity selection: {len(selected)} images "
                  f"(removed {len(unique_images) - len(selected)} similar)")
            return selected
        
        return unique_images
    
    @classmethod
    def _cluster_and_select(cls, images: List[Dict]) -> List[Dict]:
        """Cluster similar images and select best from each cluster."""
        # Create similarity matrix
        n = len(images)
        similarity_matrix = np.zeros((n, n))
        
        analyzer = ImageQualityAnalyzer()
        
        for i in range(n):
            for j in range(i+1, n):
                sim = analyzer.compute_histogram_similarity(
                    images[i]['gray'], 
                    images[j]['gray']
                )
                similarity_matrix[i,j] = sim
                similarity_matrix[j,i] = sim
        
        # Greedy clustering: start with highest quality, exclude similar
        quality_scores = [QualityFilter.compute_quality_score(m) for m in images]
        sorted_indices = sorted(range(n), key=lambda i: quality_scores[i], reverse=True)
        
        selected = []
        used = set()
        
        for idx in sorted_indices:
            if idx in used:
                continue
            
            # Check if too similar to already selected
            too_similar = False
            for sel_idx in [images.index(s) for s in selected]:
                if similarity_matrix[idx, sel_idx] > cls.SIMILARITY_THRESHOLD:
                    too_similar = True
                    break
            
            if not too_similar:
                selected.append(images[idx])
                used.add(idx)
            
            if len(selected) >= cls.MAX_IMAGES_PER_PERSON:
                break
        
        # ⚠️ CRITICAL SAFETY: Ensure we ALWAYS have at least MIN_IMAGES_PER_PERSON
        if len(selected) < cls.MIN_IMAGES_PER_PERSON:
            print(f"  ⚠️  Only {len(selected)} selected - adding more to reach minimum {cls.MIN_IMAGES_PER_PERSON}")
            # Add more from remaining, prioritizing diversity AND quality
            remaining = [img for img in images if img not in selected]
            remaining_sorted = sorted(remaining, 
                                    key=lambda m: QualityFilter.compute_quality_score(m),
                                    reverse=True)
            needed = cls.MIN_IMAGES_PER_PERSON - len(selected)
            selected.extend(remaining_sorted[:needed])
            print(f"  ✅ Added {needed} more images to reach minimum threshold")
        
        return selected
    
    @classmethod
    def analyze_diversity(cls, images: List[Dict]) -> Dict:
        """Analyze diversity metrics of selected images."""
        brightness_values = [m['brightness'] for m in images]
        sharpness_values = [m['sharpness'] for m in images]
        
        return {
            'count': len(images),
            'brightness_range': (min(brightness_values), max(brightness_values)),
            'brightness_std': np.std(brightness_values),
            'sharpness_range': (min(sharpness_values), max(sharpness_values)),
            'avg_quality': np.mean([QualityFilter.compute_quality_score(m) for m in images])
        }


def process_person_folder(person_dir: Path, backup_root: Path, removed_root: Path, dry_run: bool = False) -> Dict:
    """Process one person's facebank folder."""
    person_name = person_dir.name
    print(f"\n{'='*70}")
    print(f"🔍 Processing: {person_name}")
    print('='*70)
    
    # Get all images
    images = sorted(person_dir.glob("*.jpg"))
    if not images:
        print(f"⚠️  No images found in {person_dir}")
        return {'name': person_name, 'status': 'empty'}
    
    print(f"📂 Found {len(images)} images")
    
    # Create backup (if not dry run)
    if not dry_run:
        person_backup = backup_root / person_name
        if not person_backup.exists():
            print(f"💾 Creating backup...")
            shutil.copytree(person_dir, person_backup)
    else:
        print(f"🔍 DRY RUN - No backup created")
    
    # Analyze all images
    print(f"🔬 Analyzing quality...")
    analyzer = ImageQualityAnalyzer()
    all_metrics = []
    
    for img_path in images:
        metrics = analyzer.analyze(img_path)
        if metrics['valid']:
            all_metrics.append(metrics)
    
    print(f"✅ {len(all_metrics)} images analyzed successfully")
    
    # Filter by quality (LENIENT)
    print(f"🎯 Filtering extremely poor quality images...")
    quality_filter = QualityFilter()
    removed_quality = []
    passed_quality = []
    
    for metrics in all_metrics:
        should_remove, reason = quality_filter.should_remove(metrics)
        if should_remove:
            removed_quality.append((metrics, reason))
        else:
            passed_quality.append(metrics)
    
    print(f"  ✅ Passed quality: {len(passed_quality)}")
    print(f"  ❌ Failed quality: {len(removed_quality)}")
    
    if removed_quality:
        print(f"\n  📋 Quality issues (will be removed):")
        for metrics, reason in removed_quality[:5]:  # Show first 5
            print(f"    • {metrics['path'].name}: {reason}")
        if len(removed_quality) > 5:
            print(f"    ... and {len(removed_quality)-5} more")
    
    # ⚠️ SAFETY CHECK: If we'd have fewer than MIN after quality filter, keep some anyway
    if len(passed_quality) < DiversitySelector.MIN_IMAGES_PER_PERSON:
        print(f"\n  ⚠️  WARNING: Only {len(passed_quality)} passed quality filter!")
        print(f"  📊 Need at least {DiversitySelector.MIN_IMAGES_PER_PERSON} images")
        print(f"  🔧 Keeping best {DiversitySelector.MIN_IMAGES_PER_PERSON} by quality score regardless of issues")
        
        # Sort ALL images by quality score and take top MIN
        all_sorted = sorted(all_metrics, 
                          key=lambda m: QualityFilter.compute_quality_score(m),
                          reverse=True)
        passed_quality = all_sorted[:DiversitySelector.MIN_IMAGES_PER_PERSON]
        removed_quality = [(m, "Below minimum threshold") for m in all_sorted[DiversitySelector.MIN_IMAGES_PER_PERSON:]]
    
    # Select diverse set from quality-passed images
    print(f"\n🎨 Selecting diverse, representative images...")
    selected = DiversitySelector.select_diverse_images(passed_quality)
    
    # Analyze diversity
    diversity = DiversitySelector.analyze_diversity(selected)
    print(f"\n📊 Diversity Analysis:")
    print(f"  • Final count: {diversity['count']} images")
    print(f"  • Brightness range: {diversity['brightness_range'][0]:.1f} - {diversity['brightness_range'][1]:.1f}")
    print(f"  • Brightness variety (std): {diversity['brightness_std']:.1f}")
    print(f"  • Sharpness range: {diversity['sharpness_range'][0]:.1f} - {diversity['sharpness_range'][1]:.1f}")
    print(f"  • Average quality score: {diversity['avg_quality']:.1f}/100")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would remove {len(images) - len(selected)} images")
        print(f"✨ Complete (DRY RUN)!")
        return {
            'name': person_name,
            'status': 'dry_run',
            'original_count': len(images),
            'would_keep': len(selected),
            'would_remove': len(images) - len(selected),
            'quality_failed': len(removed_quality),
            'diversity': diversity
        }
    
    # Move removed images
    person_removed = removed_root / person_name
    person_removed.mkdir(exist_ok=True, parents=True)
    
    selected_paths = {m['path'] for m in selected}
    total_removed = 0
    
    for img_path in images:
        if img_path not in selected_paths:
            dest = person_removed / img_path.name
            shutil.move(str(img_path), str(dest))
            total_removed += 1
    
    print(f"\n✨ Complete!")
    print(f"  • Kept: {len(selected)} images")
    print(f"  • Removed: {total_removed} images")
    print(f"  • Removed to: {person_removed}")
    
    return {
        'name': person_name,
        'status': 'success',
        'original_count': len(images),
        'kept_count': len(selected),
        'removed_count': total_removed,
        'quality_failed': len(removed_quality),
        'diversity': diversity
    }


def main():
    """Main execution with test mode option."""
    import sys
    
    facebank_dir = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank")
    backup_root = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank_BACKUP_ALL")
    removed_root = Path("/Volumes/HardDrive/SCREEN TIME ANALYZER/data/facebank_REMOVED_ALL")
    
    # Check for test mode
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    dry_run = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"
    
    # Create directories
    backup_root.mkdir(exist_ok=True)
    removed_root.mkdir(exist_ok=True)
    
    print("="*70)
    print("🎬 COMPREHENSIVE FACEBANK CLEANUP (LENIENT VERSION)")
    print("="*70)
    print(f"📁 Facebank directory: {facebank_dir}")
    print(f"💾 Backup directory: {backup_root}")
    print(f"🗑️  Removed directory: {removed_root}")
    print(f"\n⚙️  Settings:")
    print(f"  • Minimum images per person: 15 (GUARANTEED)")
    print(f"  • Maximum images per person: 40")
    print(f"  • Quality thresholds: LENIENT (only reject truly unusable)")
    
    if test_mode:
        print(f"\n🧪 TEST MODE: Will process only BRANDI folder")
    elif dry_run:
        print(f"\n🔍 DRY RUN MODE: Will simulate without making changes")
    print()
    
    # Find all person folders
    person_dirs = [d for d in facebank_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') 
                   and not d.name.endswith('_BACKUP') 
                   and not d.name.endswith('_REMOVED')]
    
    if test_mode:
        person_dirs = [d for d in person_dirs if d.name == "BRANDI"]
        if not person_dirs:
            print("❌ BRANDI folder not found!")
            return
    
    print(f"👥 Found {len(person_dirs)} cast member(s) to process:")
    for person_dir in sorted(person_dirs):
        print(f"  • {person_dir.name}")
    
    print("\n" + "="*70)
    if not test_mode and not dry_run:
        input("Press ENTER to start cleanup (or Ctrl+C to cancel)...")
    else:
        print(f"Starting in {'TEST' if test_mode else 'DRY RUN'} mode...")
    print("="*70)
    
    # Process each person
    results = []
    for person_dir in sorted(person_dirs):
        result = process_person_folder(person_dir, backup_root, removed_root, dry_run=dry_run or test_mode)
        results.append(result)
    
    # Final summary
    print("\n\n" + "="*70)
    if dry_run or test_mode:
        print("📊 SIMULATION SUMMARY")
    else:
        print("📊 FINAL SUMMARY")
    print("="*70)
    
    if dry_run or test_mode:
        total_original = sum(r.get('original_count', 0) for r in results)
        total_would_keep = sum(r.get('would_keep', 0) for r in results)
        total_would_remove = sum(r.get('would_remove', 0) for r in results)
        
        print(f"\n🎬 Would process {len(results)} cast member(s):")
        print(f"  • Original images: {total_original}")
        print(f"  • Would keep: {total_would_keep} ({total_would_keep/total_original*100:.1f}%)")
        print(f"  • Would remove: {total_would_remove} ({total_would_remove/total_original*100:.1f}%)")
        
        print(f"\n📋 Simulation Details:")
        print(f"{'Name':<15} {'Original':<10} {'Would Keep':<12} {'Would Remove':<12}")
        print("-" * 70)
        
        for result in sorted(results, key=lambda r: r.get('name', '')):
            if result['status'] == 'dry_run':
                name = result['name']
                orig = result['original_count']
                keep = result['would_keep']
                remove = result['would_remove']
                print(f"{name:<15} {orig:<10} {keep:<12} {remove:<12}")
    else:
        total_original = sum(r.get('original_count', 0) for r in results)
        total_kept = sum(r.get('kept_count', 0) for r in results)
        total_removed = sum(r.get('removed_count', 0) for r in results)
