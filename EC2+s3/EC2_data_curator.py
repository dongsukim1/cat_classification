#!/usr/bin/env python3
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import sys

# Download the annotations file from the official source
print("Downloading annotations file...")
url = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json"
response = requests.get(url)
with open('caltech_annotations.json', 'wb') as f:
    f.write(response.content)
print("Annotations downloaded successfully.")

#Target North American species
target_species = [
    "mountain_lion",  
    "bobcat",         
    "coyote",         
    "fox",            
    "deer",           
    "empty"           
]

# Load annotations
with open('caltech_annotations.json', 'r') as f:
    data = json.load(f)

# Create a mapping from category ID to category name
category_map = {cat['id']: cat['name'] for cat in data['categories']}

# Find images of target species
species_images = {}
for annotation in data['annotations']:
    category_id = annotation['category_id']
    species_name = category_map.get(category_id)
    
    if species_name in target_species:
        # Find image idx
        image_id = annotation['image_id']
        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        
        if image_info:
            if species_name not in species_images:
                species_images[species_name] = []
            species_images[species_name].append(image_info['file_name'])

# Create S3 client
s3 = boto3.client('s3')
bucket_name = "big-cat-data2"

# Copy images to bucket
for species, images in species_images.items():
    print(f"Found {len(images)} images of {species}")
    
    for img in images:
        source_path = f"agentmorris/lila-wildlife/caltech-unzipped/cct_images/{img}"
        dest_path = f"{species}/{img}"
        
        try:
            s3.copy_object(
                CopySource={'Bucket': 'us-west-2.opendata.source.coop', 'Key': source_path},
                Bucket=bucket_name,
                Key=dest_path
            )
            print(f"Copied {img} to {dest_path}")
        except Exception as e:
            print(f"Failed to copy {img}: {str(e)}")
            sys.exit(1)

print("Copy process completed!")

#create new IAM role for ec2 instance allowing all s3 perms
#create IAM policies to allow instance connect EC2 DescribeInstances and EC2 SendSerialConsoleSSHPublicKey + SendSSHPublicKey