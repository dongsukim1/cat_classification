# sagemaker_training/distill_launcher.py
"""SageMaker launcher for distillation training with spot instances."""
import argparse
import time
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


def launch_phase1(args, role, session):
    """Launch Phase 1 distillation training job."""
    bucket = args.bucket
    splits = args.splits_version

    hyperparameters = {
        "student-arch": args.student_arch,
        "epochs": args.phase1_epochs,
        "batch-size-train": args.batch_size_train,
        "batch-size-val": args.batch_size_val,
        "learning-rate": args.phase1_lr,
        "num-workers": 4,
    }

    if args.teacher_weights_s3:
        hyperparameters["teacher-weights"] = "/opt/ml/input/data/teacher/dinov2_vitl14.pth"

    input_paths = {
        "train": f"s3://{bucket}/caltech_images",
        "splits": f"s3://{bucket}/training_loop/data_augmentation_pipeline/{splits}",
    }

    if args.teacher_weights_s3:
        input_paths["teacher"] = args.teacher_weights_s3

    estimator = PyTorch(
        entry_point="distill_train.py",
        source_dir="./sagemaker_training",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket}/distill_output",
        base_job_name=f"distill-{args.student_arch.replace('_', '-')}",
        max_run=3600 * 4,
        max_wait=3600 * 5,
        use_spot_instances=True,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/phase1-{args.student_arch}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=30,
        environment={"SM_MODEL_DIR": "/opt/ml/model"},
    )

    job_name = f"distill-{args.student_arch.replace('_', '-')}-{int(time.time())}"
    estimator.fit(inputs=input_paths, job_name=job_name, wait=True)
    return estimator.latest_training_job.name


def launch_phase2(args, role, session, phase1_model_s3):
    """Launch Phase 2 QAT training job."""
    bucket = args.bucket
    splits = args.splits_version

    hyperparameters = {
        "student-arch": args.student_arch,
        "phase1-model": "/opt/ml/input/data/phase1/model.pth",
        "epochs": args.phase2_epochs,
        "batch-size-train": args.batch_size_train,
        "batch-size-val": args.batch_size_val,
        "learning-rate": args.phase2_lr,
        "num-workers": 4,
    }

    input_paths = {
        "train": f"s3://{bucket}/caltech_images",
        "splits": f"s3://{bucket}/training_loop/data_augmentation_pipeline/{splits}",
        "phase1": TrainingInput(phase1_model_s3, content_type="application/x-tar"),
    }

    estimator = PyTorch(
        entry_point="qat_train.py",
        source_dir="./sagemaker_training",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket}/qat_output",
        base_job_name=f"qat-{args.student_arch.replace('_', '-')}",
        max_run=3600 * 2,
        max_wait=3600 * 3,
        use_spot_instances=True,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/phase2-{args.student_arch}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=20,
        environment={"SM_MODEL_DIR": "/opt/ml/model"},
    )

    job_name = f"qat-{args.student_arch.replace('_', '-')}-{int(time.time())}"
    estimator.fit(inputs=input_paths, job_name=job_name, wait=True)
    return estimator.latest_training_job.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", required=True,
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s", "efficientnet_lite0"])
    parser.add_argument("--bucket", default="big-cat-data2")
    parser.add_argument("--splits-version", default="splitsv2")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase1-lr", type=float, default=0.001)
    parser.add_argument("--phase2-lr", type=float, default=0.0001)
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--teacher-weights-s3", default=None,
                        help="S3 URI to pre-cached DINOv2 weights")
    parser.add_argument("--role", default=None,
                        help="SageMaker execution role ARN (required when running locally)")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--phase1-model-s3", default=None,
                        help="S3 URI to Phase 1 model output (for --skip-phase1)")
    args = parser.parse_args()

    session = sagemaker.Session()
    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            raise SystemExit(
                "Could not determine SageMaker role. Pass --role <arn> when running locally.\n"
                "Find your role: aws iam list-roles --query \"Roles[?contains(RoleName, 'SageMaker')].[Arn]\" --output text"
            )

    if not args.skip_phase1:
        print(f"=== Phase 1: Distillation ({args.student_arch}) ===")
        phase1_job = launch_phase1(args, role, session)
        phase1_model_s3 = f"s3://{args.bucket}/distill_output/{phase1_job}/output/model.tar.gz"
    else:
        if not args.phase1_model_s3:
            raise ValueError("--phase1-model-s3 required when --skip-phase1 is set")
        phase1_model_s3 = args.phase1_model_s3

    print(f"\n=== Phase 2: QAT ({args.student_arch}) ===")
    launch_phase2(args, role, session, phase1_model_s3)

    print("\nDone! Download model artifacts and run export_onnx.py for Phase 3.")


if __name__ == "__main__":
    main()
