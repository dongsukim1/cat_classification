import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict


def crop_to_bbox(image, bbox_info):
    """Crop a PIL image to the first bounding box in bbox_info.

    Args:
        image: PIL.Image
        bbox_info: list of annotation dicts, each with a 'bbox' key [x, y, w, h]

    Returns:
        Cropped PIL.Image, or the original image if no valid bbox is found.
    """
    if not bbox_info:
        return image

    bbox = bbox_info[0].get('bbox', None)
    if bbox is None:
        return image

    x, y, width, height = bbox
    img_width, img_height = image.size

    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    width = max(1, min(width, img_width - x))
    height = max(1, min(height, img_height - y))

    left, top, right, bottom = int(x), int(y), int(x + width), int(y + height)

    try:
        return image.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Warning: Failed to crop image with bbox {bbox}: {e}")
        return image


class WildlifeDataset(Dataset):
    """PyTorch Dataset for wildlife images"""
    
    def __init__(self, image_paths, labels, bbox_dict=None, mode='train', label_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.bbox_dict = bbox_dict or {}
        self.mode = mode.lower()
        
        if label_to_idx is None:
            unique_labels = sorted(list(set(labels)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.transforms = self._get_transforms()
        
        print(f"Initialized {mode} dataset with {len(self.image_paths)} samples")
    
    def _get_transforms(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    
    def _crop_to_bbox(self, image, bbox_info):
        return crop_to_bbox(image, bbox_info)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            image_id = Path(image_path).stem
            if image_id in self.bbox_dict:
                bbox_info = self.bbox_dict[image_id]
                image = self._crop_to_bbox(image, bbox_info)
            
            image_tensor = self.transforms(image)
            return image_tensor, label_idx
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            image_tensor = self.transforms(black_image)
            return image_tensor, label_idx
    
    def get_class_weights(self):
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        weights = []
        for class_name in sorted(self.label_to_idx.keys()):
            count = label_counts.get(class_name, 1)
            weight = total_samples / (len(self.label_to_idx) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
def load_bbox_data_sm(labels_file):
    """Load bounding box data from COCO-style JSON file for SageMaker."""
    print(f"Loading bbox data from {labels_file}...")
    with open(labels_file, 'r') as f:
        coco_data = json.load(f)
    
    category_map = {}
    for category in coco_data.get('categories', []):
        category_map[category['id']] = category['name']
    
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
    
    return dict(bbox_dict)

def create_split_bbox_dict(samples, bbox_dict):
    """Create bbox dictionary for a specific split."""
    
    split_bbox_dict = {}
    for sample in samples:
        image_id = sample.get('image_id')
        if image_id and image_id in bbox_dict:
            split_bbox_dict[image_id] = bbox_dict[image_id]
        if 'annotations' in sample and sample['annotations']:
            if not image_id:
                image_id = Path(sample['image_path']).stem
            split_bbox_dict[image_id] = sample['annotations']
    
    return split_bbox_dict

def create_datasets_and_dataloaders_sm(data_dir, labels_file, splits_dir, target_species=None, 
                                      batch_size_train=32, batch_size_val=64, batch_size_test=64, 
                                      num_workers=4, use_labels=False):
    """Create datasets and dataloaders for SageMaker training."""
    
    if target_species is None:
        target_species = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
    
    bbox_dict = load_bbox_data_sm(labels_file) if use_labels else {}
    if not use_labels:
        print("Skipping bbox data loading (use_labels=False)")  
    
    # Load splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = Path(splits_dir) / f'{split_name}.json'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            splits[split_name] = {'samples': split_data, 'size': len(split_data)}
            print(f"Loaded {len(split_data)} {split_name} samples")
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(target_species))}
    
    datasets = {}
    dataloaders = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name in splits and splits[split_name]['size'] > 0:
            samples = splits[split_name]['samples']
            
            image_paths = []
            for sample in samples:
                # Unified path resolution: image_path is class/filename.jpg
                original_path = sample.get('image_path') or sample.get('image_path_aws', '')
                if not original_path:
                    continue
                full_path = str(Path(data_dir) / original_path)
                image_paths.append(full_path)
            
            labels = [sample['primary_class'] for sample in samples]
            
            # Create bbox dict for this split
            split_bbox_dict = {}
            if use_labels:
                split_bbox_dict = create_split_bbox_dict(samples, bbox_dict)
            
            dataset = WildlifeDataset(
                image_paths=image_paths,
                labels=labels,
                bbox_dict=split_bbox_dict,
                mode=split_name,
                label_to_idx=label_to_idx
            )
            datasets[split_name] = dataset
            
            batch_size = {'train': batch_size_train, 'val': batch_size_val, 'test': batch_size_test}[split_name]
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split_name == 'train')
            )
            dataloaders[split_name] = dataloader
    
    return datasets, dataloaders, dict(bbox_dict), label_to_idx


class DistillationDataset(torch.utils.data.Dataset):
    """Dual-input dataset for bbox-conditioned feature distillation.

    Returns (student_tensor, teacher_tensor, label_idx, apply_distill).
    - student_tensor: full image with train augmentations
    - teacher_tensor: bbox-cropped image with eval transforms (stable target)
    - apply_distill: False for 'empty' class, True otherwise
    """

    EMPTY_CLASS = "empty"

    def __init__(
        self,
        data_dir,
        split_file,
        target_species,
        label_to_idx,
        mode="train",
    ):
        self.data_dir = Path(data_dir)
        self.label_to_idx = label_to_idx
        self.mode = mode
        self.samples = []

        with open(split_file) as f:
            raw_samples = json.load(f)

        for sample in raw_samples:
            cls = sample["primary_class"]
            if cls not in target_species:
                continue
            image_path = sample.get("image_path") or sample.get("image_path_aws", "")
            full_path = str(self.data_dir / image_path)
            annotations = sample.get("annotations", [])
            self.samples.append((full_path, cls, annotations))

        if mode == "train":
            self.student_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.student_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.teacher_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, cls, annotations = self.samples[idx]
        label_idx = self.label_to_idx[cls]
        apply_distill = cls != self.EMPTY_CLASS

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        student_tensor = self.student_transform(image)

        if apply_distill and annotations:
            teacher_image = crop_to_bbox(image.copy(), annotations)
        else:
            teacher_image = image
        teacher_tensor = self.teacher_transform(teacher_image)

        return student_tensor, teacher_tensor, label_idx, apply_distill