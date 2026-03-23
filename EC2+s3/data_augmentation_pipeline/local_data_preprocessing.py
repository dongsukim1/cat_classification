import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

TARGET_SPECIES = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
ANNOTATION_PATHS = {
    "cis_test": "./data/eccv_18_annotation_files/cis_test_annotations.json",
    "cis_val": "./data/eccv_18_annotation_files/cis_val_annotations.json",
    "trans_test": "./data/eccv_18_annotation_files/trans_test_annotations.json",
    "trans_val": "./data/eccv_18_annotation_files/trans_val_annotations.json",
    "train": "./data/eccv_18_annotation_files/train_annotations.json"
}
IMAGE_BASE_DIR = "./data/eccv_18_all_images_sm" 
OUTPUT_BASE_DIR = "./data/processed_dataset" 
global SPECIES_MAPPING 
global TOTAL_IMAGES
TOTAL_IMAGES = 0


def load_and_filter_annotations(annotation_path, target_species):
    """Load annotations from JSON file and filter for target species
    Args:
        annotation_path (str): Path to the annotation JSON file
        target_species (list): List of species to filter for
    Returns:
        filtered_list (list): List of annotations filtered by species
    
    """
    global SPECIES_MAPPING 
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    
    SPECIES_MAPPING= {item['name']: item['id'] for item in data['categories'] if item['name'] in target_species}

    target_species_dict = {}

    for species in target_species:
        if species in SPECIES_MAPPING.keys():
            target_species_dict[species] = SPECIES_MAPPING[species]

    filtered_data = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in target_species_dict.values():
            continue
        ann["image_id"] = ann['image_id'] + ".jpg"
        filtered_data[ann['category_id']].append(ann)
    
    return filtered_data

def organize_images(annotations, output_base_dir, image_base_dir, inverted_mapping):
    """Create species image directory structure and copy images
    Args:
        annotations (list): List of annotation dictionaries
        output_base_dir (str): Base directory to save organized images
        image_base_dir (str): Base directory where original images are stored
    Returns:
        None
    """
    global TOTAL_IMAGES
    global SPECIES_MAPPING
    for species in TARGET_SPECIES:
        species_dir = os.path.join(output_base_dir, species)
        os.makedirs(species_dir, exist_ok=True)

    
    for species in annotations:
        for annot in annotations[species]:
            src_path = os.path.join(image_base_dir, annot['image_id'])
            dst_path = os.path.join(output_base_dir, inverted_mapping[species], annot['image_id'])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                TOTAL_IMAGES += 1
            else:
                print(f"Warning: Image not found: {src_path}")

    return  

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    all_annotations = []
    for split_name, annotation_path in ANNOTATION_PATHS.items():
        print(f"Processing {split_name} annotations...")
        
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file not found: {annotation_path}")
            continue
            
        filtered_annotations = load_and_filter_annotations(
            annotation_path, TARGET_SPECIES
        )
        if all_annotations:
            for species, anns in filtered_annotations.items():
                all_annotations[species].extend(anns)
        else:
            all_annotations = filtered_annotations
        

    print(f"\nTotal images with target species: {sum(len(v) for v in all_annotations.values())}")

    print("\nOrganizing images by location and species...")
    inverse = {id:animal for animal, id in SPECIES_MAPPING.items()}
    organize_images(
        all_annotations, OUTPUT_BASE_DIR, IMAGE_BASE_DIR, inverse
    )

    print(f"\nTotal images processed: {TOTAL_IMAGES}")

if __name__ == "__main__":
    main()