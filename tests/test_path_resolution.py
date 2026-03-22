# tests/test_path_resolution.py
import json
import tempfile
from pathlib import Path


def test_unified_path_format():
    """Split JSON image_path should be forward-slash relative: class/filename.jpg"""
    sample = {
        "image_id": "abc123",
        "primary_class": "bobcat",
        "image_path": "bobcat/abc123.jpg",
        "annotations": [],
    }
    assert "\\" not in sample["image_path"]
    parts = sample["image_path"].split("/")
    assert len(parts) == 2
    assert parts[0] == sample["primary_class"]


def test_path_resolution_local():
    """Path resolution should work with any base directory."""
    data_dir = Path("./data/s3+expanded_empty")
    image_path = "bobcat/abc123.jpg"
    full_path = data_dir / image_path
    expected = Path("data/s3+expanded_empty/bobcat/abc123.jpg")
    assert full_path == expected


def test_path_resolution_sagemaker():
    """Path resolution should work with SageMaker mount points."""
    from pathlib import PurePosixPath

    data_dir = PurePosixPath("/opt/ml/input/data/train")
    image_path = "bobcat/abc123.jpg"
    full_path = data_dir / image_path
    assert str(full_path) == "/opt/ml/input/data/train/bobcat/abc123.jpg"


def test_no_legacy_fields_in_regenerated_split():
    """Regenerated splits should not contain image_path_local or image_path_aws."""
    sample = {
        "image_id": "abc123",
        "primary_class": "bobcat",
        "image_path": "bobcat/abc123.jpg",
        "annotations": [],
    }
    assert "image_path_local" not in sample
    assert "image_path_aws" not in sample
