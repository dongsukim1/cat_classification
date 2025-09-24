import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import numpy as np

def create_stratified_split(
    data_dir: str = "../data/full_s3/",
    labels_file: str = "./bboxes.json",
    target_species: List[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Create a stratified train/validation/test split for image dataset using COCO-style annotations.
    
    Args:
        data_dir: Path to directory containing image folders
        labels_file: Path to JSON file containing COCO-style annotations
        target_species: List of target species to include in the split
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15) 
        test_ratio: Proportion of data for testing (default: 0.15)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train/val/test splits with image paths and labels
    """
    
    # Default target species if none provided
    if target_species is None:
        target_species = [
            "mountain_lion",  
            "bobcat",         
            "coyote",         
            "fox",            
            "deer",           
            "empty"           
        ]
    
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load COCO-style annotations from JSON file
    print(f"Loading annotations from {labels_file}...")
    with open(labels_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping from id to name
    category_map = {}
    for category in coco_data.get('categories', []):
        category_map[category['id']] = category['name']
    
    print(f"Found {len(category_map)} categories in annotations")
    print(f"Categories: {list(category_map.values())}")
    print(f"Target species: {target_species}")
    
    # Filter categories to only include target species
    target_category_ids = set()
    for cat_id, cat_name in category_map.items():
        if cat_name in target_species:
            target_category_ids.add(cat_id)
    
    print(f"Target category IDs: {target_category_ids}")
    
    # Get all image files from data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f"*{ext}"))
        all_images.extend(data_path.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(all_images)} image files in directory")
    
    # Create mapping from image ID/filename to full path
    image_path_map = {}
    for img_path in all_images:
        filename = img_path.name
        # Try both filename and stem (without extension) as potential IDs
        image_path_map[filename] = str(img_path)
        image_path_map[img_path.stem] = str(img_path)
    
    # Process annotations and group by image
    image_annotations = defaultdict(list)
    
    for annotation in coco_data.get('annotations', []):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # Only include annotations for target species
        if category_id in target_category_ids:
            category_name = category_map[category_id]
            
            annotation_data = {
                'category_id': category_id,
                'category_name': category_name,
                'bbox': annotation['bbox'],
                'annotation_id': annotation['id']
            }
            
            image_annotations[image_id].append(annotation_data)
    
    print(f"Found {len(image_annotations)} images with target species annotations")
    
    # Create samples list matching images with annotations
    samples = []
    missing_images = []
    images_found = 0
    
    for image_id, annotations in image_annotations.items():
        # Try to find the image file
        image_path = None
        
        # Try different possible filename formats
        possible_names = [
            image_id,  # Direct image ID
            f"{image_id}.jpg",
            f"{image_id}.jpeg", 
            f"{image_id}.png"
        ]
        
        for name in possible_names:
            if name in image_path_map:
                image_path = image_path_map[name]
                images_found += 1
                break
        
        if image_path:
            # Extract all category names for this image
            categories = [ann['category_name'] for ann in annotations]
            
            # Use the first category as primary class for stratification
            primary_class = categories[0] if categories else 'unknown'
            
            if len(categories) == 0 or (len(categories) == 1 and categories[0] == 'empty'):
                primary_class = 'empty'
            
            sample = {
                'image_path': image_path,
                'image_id': image_id,
                'labels': categories,
                'primary_class': primary_class,
                'annotations': annotations,
                'bbox_count': len(annotations)
            }
            
            samples.append(sample)
        else:
            missing_images.append(image_id)
    
    print(f"Successfully matched {len(samples)} images with annotations")
    print(f"Images found in filesystem: {images_found}")
    if missing_images:
        print(f"Warning: {len(missing_images)} annotated images not found in data directory")
        print(f"First few missing: {missing_images[:5]}")
    
    # Group samples by primary class for stratified splitting
    class_samples = defaultdict(list)
    for sample in samples:
        class_samples[sample['primary_class']].append(sample)
    
    # Print class distribution
    class_counts = {cls: len(samples) for cls, samples in class_samples.items()}
    total_samples = len(samples)
    
    print(f"\nClass distribution ({total_samples} total samples):")
    for cls in target_species:
        if cls in class_counts:
            count = class_counts[cls]
            percentage = (count / total_samples) * 100
            print(f"  {cls}: {count} samples ({percentage:.1f}%)")
    
    # Print any classes not in target species
    other_classes = set(class_counts.keys()) - set(target_species)
    if other_classes:
        print(f"  Other classes found: {other_classes}")
    
    # Perform stratified split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for class_name, class_sample_list in class_samples.items():
        n_samples = len(class_sample_list)
        
        if n_samples == 0:
            continue
            
        # Shuffle samples within each class
        random.shuffle(class_sample_list)
        
        # Calculate split indices
        train_end = max(1, int(n_samples * train_ratio))  # Ensure at least 1 sample in train if possible
        val_end = train_end + max(0, int(n_samples * val_ratio))
        
        # Adjust for small classes
        if n_samples <= 3:
            # For very small classes, put at least one in train
            train_end = min(n_samples, 1)
            val_end = min(n_samples, 2)
        
        # Split the samples
        train_class_samples = class_sample_list[:train_end]
        val_class_samples = class_sample_list[train_end:val_end]
        test_class_samples = class_sample_list[val_end:]
        
        train_samples.extend(train_class_samples)
        val_samples.extend(val_class_samples)
        test_samples.extend(test_class_samples)
        
        print(f"Class '{class_name}': {len(train_class_samples)} train, "
              f"{len(val_class_samples)} val, {len(test_class_samples)} test")
    
    # Shuffle the final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    # Create final split dictionary
    splits = {
        'train': {
            'samples': train_samples,
            'size': len(train_samples),
            'class_distribution': Counter([s['primary_class'] for s in train_samples])
        },
        'val': {
            'samples': val_samples,
            'size': len(val_samples),
            'class_distribution': Counter([s['primary_class'] for s in val_samples])
        },
        'test': {
            'samples': test_samples,
            'size': len(test_samples),
            'class_distribution': Counter([s['primary_class'] for s in test_samples])
        },
        'metadata': {
            'target_species': target_species,
            'total_categories': len(category_map),
            'target_categories': len(target_category_ids),
            'category_map': category_map
        }
    }
    
    # Print final split summary
    print(f"\nFinal split summary:")
    print(f"Train: {len(train_samples)} samples ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"Val:   {len(val_samples)} samples ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"Test:  {len(test_samples)} samples ({len(test_samples)/total_samples*100:.1f}%)")
    
    # Verify stratification
    print(f"\nStratification verification:")
    for class_name in sorted(class_counts.keys()):
        orig_ratio = class_counts[class_name] / total_samples
        train_count = splits['train']['class_distribution'][class_name]
        val_count = splits['val']['class_distribution'][class_name]
        test_count = splits['test']['class_distribution'][class_name]
        
        train_ratio_actual = train_count / len(train_samples) if train_samples else 0
        val_ratio_actual = val_count / len(val_samples) if val_samples else 0
        test_ratio_actual = test_count / len(test_samples) if test_samples else 0
        
        print(f"Class '{class_name}':")
        print(f"  Original: {orig_ratio:.3f}, Train: {train_ratio_actual:.3f}, "
              f"Val: {val_ratio_actual:.3f}, Test: {test_ratio_actual:.3f}")
    
    return splits

def save_splits_to_files(splits: Dict[str, Dict[str, Any]], output_dir: str = "./EC2+s3/data_augmentation_pipeline/splits/"):
    """
    Save the split results to separate files for easy loading.
    
    Args:
        splits: Dictionary returned by create_stratified_split()
        output_dir: Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
            
        split_data = splits[split_name]
        split_file = output_path / f"{split_name}.json"
        
        # Create simplified format for saving
        save_data = []
        for sample in split_data['samples']:
            original_path = Path(sample['image_path']) # Required for sagemaker split generation due to file pathing.
            path_to_name = original_path.name # Without these four lines use 'image_path': sample['image_path'] below.
            species_handle = original_path.parent.name 
            bucket_path = f"{species_handle}/{path_to_name}"
            save_data.append({
                'image_path': sample['image_path'], # Use bucket_path for sagemaker training.
                'image_id': sample['image_id'],
                'labels': sample['labels'],
                'primary_class': sample['primary_class'],
                'bbox_count': sample['bbox_count'],
                'annotations': sample['annotations']
            })
        
        with open(split_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved {split_name} split to {split_file}")
    
    # Save summary statistics and metadata
    summary_file = output_path / "split_summary.json"
    summary_data = {}
    for split_name in ['train', 'val', 'test']:
        if split_name in splits:
            split_data = splits[split_name]
            summary_data[split_name] = {
                'size': split_data['size'],
                'class_distribution': dict(split_data['class_distribution'])
            }
    
    summary_data['metadata'] = splits.get('metadata', {})
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Saved split summary to {summary_file}")

def load_split_from_file(split_file: str) -> List[Dict[str, Any]]:
    """
    Load a specific split from saved JSON file.
    
    Args:
        split_file: Path to the split JSON file (e.g., './splits/train.json')
        
    Returns:
        List of samples in the split
    """
    with open(split_file, 'r') as f:
        return json.load(f)

# Example usage
if __name__ == "__main__":
    # Define target species
    target_species = [
        "mountain_lion",  
        "bobcat",         
        "coyote",         
        "fox",            
        "deer",           
        "empty"           
    ]

    # Create stratified split
    splits = create_stratified_split(
        data_dir='./data/full_s3',
        labels_file='./EC2+s3/bboxes.json',
        target_species=target_species,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Save splits to files
    save_splits_to_files(splits)
    
    # Example: Access train data
    if splits['train']['size'] > 0:
        train_data = splits['train']['samples']
        print(f"\nFirst training sample:")
        print(f"Image ID: {train_data[0]['image_id']}")
        print(f"Image path: {train_data[0]['image_path']}")
        print(f"Labels: {train_data[0]['labels']}")
        print(f"Primary class: {train_data[0]['primary_class']}")
        print(f"Number of bboxes: {train_data[0]['bbox_count']}")
        
        # Show first annotation details
        if train_data[0]['annotations']:
            first_ann = train_data[0]['annotations'][0]
            print(f"First bbox: {first_ann['bbox']} (category: {first_ann['category_name']})")