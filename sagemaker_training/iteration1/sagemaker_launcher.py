import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import os
from datetime import datetime
import time

def launch_training_job():
    """Launch a SageMaker training job."""
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    

    # $env:SAGEMAKER_ROLE_ARN="arn:aws:iam::123456789:role/YourSageMakerRole"
    # Set your Sagemaker ARN role via the psl command above
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    if not role:
        try:
            role = get_execution_role()
            print("Using SageMaker execution role")
        except ValueError:
            raise ValueError(
                "SAGEMAKER_ROLE_ARN environment variable not set and "
                "unable to get execution role from SageMaker environment"
            )
    else:
        print(f"Using role from environment: {role}")
    
    # Define your S3 paths
    bucket = 'big-cat-data2'
    
    # Input data paths
    input_paths = {
        'train': f's3://{bucket}/caltech_images',  # Image data
        'splits': f's3://{bucket}/training_loop/data_augmentation_pipeline/splitsv2',  # Split files
        'bbox': f's3://{bucket}',
    }
    
    # Output path for model artifacts
    output_path = f's3://{bucket}/training_output'
    
    # Create timestamp for job name
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f'wildlife-classification-{timestamp}'
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 20,
        'batch-size-train': 32,
        'batch-size-val': 64,
        'learning-rate': 0.001,
        'num-workers': 8
    }
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='sagemaker_train.py',  
        source_dir='./sagemaker_training', 
        role=role,
        instance_type='ml.g4dn.xlarge',  
        instance_count=1,
        framework_version='2.0.0',  
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name='wildlife-classification',
        max_run=4500,  
        volume_size=20,  
        environment={
            'SM_MODEL_DIR': '/opt/ml/model',
        }
    )
    
    print(f"Starting training job: {job_name}")
    print(f"Input paths: {input_paths}")
    print(f"Output path: {output_path}")
    print(f"Instance type: ml.g4dn.large")
    print(f"Hyperparameters: {hyperparameters}")
    
    print(f"Current region: {boto3.Session().region_name}")

    # Start training
    estimator.fit(
        inputs=input_paths,
        job_name=job_name,
        wait=True
    )
    
    print(f"Training job '{job_name}' submitted!")
    print(f"You can monitor it in the SageMaker console or use:")
    print(f"estimator.describe_training_job()")
    
    return estimator

if __name__ == '__main__':
    estimator = launch_training_job()