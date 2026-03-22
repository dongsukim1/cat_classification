# scripts/regenerate_splits.py
"""
One-time script to convert split JSONs from dual-path format
(image_path_local + image_path_aws) to unified format (image_path).

Usage:
    python scripts/regenerate_splits.py --splits-dir EC2+s3/data_augmentation_pipeline/splitsv2
"""
import argparse
import json
from pathlib import Path


def regenerate_split(split_path: Path) -> list:
    with split_path.open() as f:
        samples = json.load(f)

    unified = []
    for sample in samples:
        if "image_path_aws" in sample:
            image_path = sample["image_path_aws"]
        elif "image_path_local" in sample:
            local = Path(sample["image_path_local"])
            image_path = f"{local.parent.name}/{local.name}"
        else:
            raise ValueError(f"Sample {sample.get('image_id')} has no path field")

        image_path = image_path.replace("\\", "/")

        unified_sample = {
            "image_id": sample["image_id"],
            "primary_class": sample["primary_class"],
            "image_path": image_path,
            "labels": sample.get("labels", [sample["primary_class"]]),
            "bbox_count": sample.get("bbox_count", 0),
            "annotations": sample.get("annotations", []),
        }
        unified.append(unified_sample)

    return unified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="EC2+s3/data_augmentation_pipeline/splitsv2")
    parser.add_argument("--output-dir", default=None, help="Defaults to --splits-dir")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir) if args.output_dir else splits_dir

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            print(f"Skipping {split_path} (not found)")
            continue

        unified = regenerate_split(split_path)
        output_path = output_dir / f"{split_name}.json"
        with output_path.open("w") as f:
            json.dump(unified, f, indent=2)
        print(f"Wrote {len(unified)} samples to {output_path}")


if __name__ == "__main__":
    main()
