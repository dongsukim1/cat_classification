import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter, defaultdict

class WildlifeDataset(Dataset):
    """
    PyTorch Dataset for wildlife images with optional bounding box cropping.
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[str], 
        bbox_dict: Optional[Dict[str, List[Dict]]] = None,
        mode: str = 'train',
        label_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the Wildlife Dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels (class names)
            bbox_dict: Dictionary mapping image IDs to bounding box annotations
            mode: 'train', 'val', or 'test' - determines which transforms to apply
            label_to_idx: Dictionary mapping label names to indices. If None, creates automatically.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.bbox_dict = bbox_dict or {}
        self.mode = mode.lower()
        
        # Create label to index mapping
        if label_to_idx is None:
            unique_labels = sorted(list(set(labels)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Define transforms based on mode
        self.transforms = self._get_transforms()
        
        print(f"Initialized {mode} dataset with {len(self.image_paths)} samples")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.label_to_idx.keys())}")
    
    def _get_transforms(self):
        """Get appropriate transforms based on dataset mode."""
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.mode == 'train':
            # Training transforms with augmentation
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # Validation/test transforms without augmentation
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    
    def _get_image_id_from_path(self, image_path: str) -> str:
        """Extract image ID from file path for bbox lookup."""
        path = Path(image_path)
        # Try different formats for image ID
        possible_ids = [
            path.stem,  # filename without extension
            path.name,  # filename with extension
            str(path)   # full path
        ]
        
        for img_id in possible_ids:
            if img_id in self.bbox_dict:
                return img_id
        
        return path.stem  # default fallback
    
    def _crop_to_bbox(self, image: Image.Image, bbox_info: List[Dict]) -> Image.Image:
        """
        Crop image to bounding box if available.
        
        Args:
            image: PIL Image
            bbox_info: List of bounding box dictionaries with 'bbox' key
            
        Returns:
            Cropped PIL Image
        """
        if not bbox_info:
            return image
        
        # Use the first bounding box if multiple exist
        bbox = bbox_info[0].get('bbox', None)
        if bbox is None:
            return image
        
        # COCO format: [x, y, width, height]
        x, y, width, height = bbox
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Ensure bbox coordinates are within image bounds
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))
        
        # Convert to PIL crop format: (left, top, right, bottom)
        left = int(x)
        top = int(y)
        right = int(x + width)
        bottom = int(y + height)
        
        # Crop the image
        try:
            cropped_image = image.crop((left, top, right, bottom))
            return cropped_image
        except Exception as e:
            print(f"Warning: Failed to crop image with bbox {bbox}: {e}")
            return image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label_index)
        """
        # Get image path and label
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Check for bounding box and crop if available
            image_id = self._get_image_id_from_path(image_path)
            if image_id in self.bbox_dict:
                bbox_info = self.bbox_dict[image_id]
                image = self._crop_to_bbox(image, bbox_info)
            
            # Apply transforms
            image_tensor = self.transforms(image)
            
            return image_tensor, label_idx
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            image_tensor = self.transforms(black_image)
            return image_tensor, label_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Tensor of class weights (inverse frequency)
        """
        # Count samples per class
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weights (inverse frequency)
        weights = []
        for class_name in sorted(self.label_to_idx.keys()):
            count = label_counts.get(class_name, 1)
            weight = total_samples / (len(self.label_to_idx) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)

def load_bbox_data(labels_file: str) -> Dict[str, List[Dict]]:
    """
    Load bounding box data from COCO-style JSON file.
    
    Args:
        labels_file: Path to the bboxes.json file
        
    Returns:
        Dictionary mapping image IDs to list of annotation dictionaries
    """
    print(f"Loading bbox data from {labels_file}...")
    
    with open(labels_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    category_map = {}
    for category in coco_data.get('categories', []):
        category_map[category['id']] = category['name']
    
    # Group annotations by image ID
    bbox_dict = defaultdict(list)
    for annotation in coco_data.get('annotations', []):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        category_name = category_map.get(category_id, 'unknown')
        
        annotation_data = {
            'category_id': category_id,
            'category_name': category_name,
            'bbox': annotation['bbox'],
            'annotation_id': annotation['id']
        }
        
        bbox_dict[image_id].append(annotation_data)
    
    print(f"Loaded bbox data for {len(bbox_dict)} images")
    return dict(bbox_dict)

def load_splits_from_files(splits_dir: str = './EC2+s3/data_augmentation_pipeline/splits'):
    """
    Load stratified splits from saved JSON files.
    
    Args:
        splits_dir: Directory containing the split files
        
    Returns:
        Dictionary with train/val/test split data
    """
    splits_path = Path(splits_dir)
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_file = splits_path / f'{split_name}.json'
        
        if split_file.exists():
            print(f"Loading {split_name} split from {split_file}...")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            splits[split_name] = {
                'samples': split_data,
                'size': len(split_data)
            }
            print(f"  Loaded {len(split_data)} {split_name} samples")
        else:
            print(f"Warning: {split_file} not found")
    
    return splits

def create_datasets_and_dataloaders(
    data_dir: str = './data/full_s3',
    labels_file: str = './EC2+s3/bboxes.json',
    splits_dir: str = './EC2+s3/data_augmentation_pipeline/splits',
    target_species: List[str] = None,
    batch_size_train: int = 32,
    batch_size_val: int = 64,
    batch_size_test: int = 64,
    num_workers: int = 12
):
    """
    Complete pipeline to create datasets and dataloaders from saved splits.
    
    Args:
        data_dir: Directory containing image files
        labels_file: Path to bboxes.json file
        splits_dir: Directory containing split files
        target_species: List of target species
        batch_size_train, batch_size_val, batch_size_test: Batch sizes
        num_workers: Number of workers for dataloaders
        
    Returns:
        Tuple of (datasets_dict, dataloaders_dict, bbox_dict, label_to_idx)
    """
    
    if target_species is None:
        target_species = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
    
    # Load bbox data
    bbox_dict = load_bbox_data(labels_file)
    
    # Load splits
    splits = load_splits_from_files(splits_dir)
    
    # Create consistent label mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(target_species))}
    print(f"Label mapping: {label_to_idx}")
    
    datasets = {}
    dataloaders = {}
    
    # Create datasets for each split
    for split_name in ['train', 'val', 'test']:
        if split_name in splits and splits[split_name]['size'] > 0:
            samples = splits[split_name]['samples']
            
            # Extract image paths and labels
            image_paths = [sample['image_path'] for sample in samples]
            labels = [sample['primary_class'] for sample in samples]
            
            # Create bbox dictionary for this split
            split_bbox_dict = {}
            for sample in samples:
                image_id = sample.get('image_id')
                if image_id and image_id in bbox_dict:
                    split_bbox_dict[image_id] = bbox_dict[image_id]
                
                # Also use annotations from the sample if available
                if 'annotations' in sample and sample['annotations']:
                    if not image_id:
                        image_id = Path(sample['image_path']).stem
                    split_bbox_dict[image_id] = sample['annotations']
            
            # Create dataset
            dataset = WildlifeDataset(
                image_paths=image_paths,
                labels=labels,
                bbox_dict=split_bbox_dict,
                mode=split_name,
                label_to_idx=label_to_idx
            )
            datasets[split_name] = dataset
            
            # Create dataloader
            batch_size = {
                'train': batch_size_train,
                'val': batch_size_val,
                'test': batch_size_test
            }[split_name]
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),  # Only shuffle training data
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split_name == 'train')  # Only drop last batch for training
            )
            dataloaders[split_name] = dataloader
            
            print(f"Created {split_name} dataloader: {len(dataloader)} batches of size {batch_size}")
    
    return datasets, dataloaders, bbox_dict, label_to_idx

# Main execution
if __name__ == "__main__":
    # Define your target species
    target_species = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
    
    # Create datasets and dataloaders
    datasets, dataloaders, bbox_dict, label_to_idx = create_datasets_and_dataloaders(
        data_dir='./data/full_s3',
        labels_file='./EC2+s3/bboxes.json',
        splits_dir='./EC2+s3/data_augmentation_pipeline/splits',
        target_species=target_species,
        batch_size_train=32,
        batch_size_val=64,
        batch_size_test=64,
        num_workers=12
    )
    
    print(f"\nDatasets created:")
    for split_name, dataset in datasets.items():
        print(f"  {split_name}: {len(dataset)} samples, {dataset.num_classes} classes")
    
    print(f"\nDataloaders created:")
    for split_name, dataloader in dataloaders.items():
        print(f"  {split_name}: {len(dataloader)} batches")
    
    # Example: Test loading a batch
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        print(f"\nTesting batch loading...")
        
        try:
            images, labels = next(iter(train_loader))
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Sample labels: {labels[:5]}")
            
            # Show class distribution in this batch
            batch_classes = [list(label_to_idx.keys())[list(label_to_idx.values()).index(label.item())] 
                           for label in labels[:10]]
            print(f"Sample classes: {batch_classes}")
            
        except Exception as e:
            print(f"Error loading batch: {e}")
    
    # Calculate class weights for training
    if 'train' in datasets:
        class_weights = datasets['train'].get_class_weights()
        print(f"\nClass weights for loss function:")
        for i, (class_name, weight) in enumerate(zip(sorted(target_species), class_weights)):
            print(f"  {class_name}: {weight:.3f}")