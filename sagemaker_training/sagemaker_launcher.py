import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import os
from datetime import datetime
import time

def launch_training_job(splits):
    """Launch a SageMaker training job."""
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    try:
        role = get_execution_role()
        print(f"Using specified role: {role}")
    except ValueError:
        print("Role not specified!")
        exit()
    
    # s3 bucket
    bucket = 'big-cat-data2'
    
    # Input data paths
    input_paths = {
        'train': f's3://{bucket}/caltech_images',  # Images
        'splits': f's3://{bucket}/training_loop/data_augmentation_pipeline/{splits}', 
        'bbox': f's3://{bucket}',
    }
    
    # Model artifacts
    output_path = f's3://{bucket}/training_output'

    # Timestamp
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
        max_run=3600 * 4,  # 4 hours
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

    estimator.fit(
        inputs=input_paths,
        job_name=job_name,
        wait=True
    )
    
    print(f"Training job '{job_name}' completed!")
    
    return estimator, job_name, timestamp

def launch_analysis_job(trained_estimator, _, timestamp, splits):
    """Launch a separate SageMaker job for model analysis."""
    
    try:
        role = get_execution_role()
        print(f"Using specified role: {role}")
    except ValueError:
        print("Role not specified!")
        exit()
    
    bucket = 'big-cat-data2'
    
    input_paths = {
        'train': f's3://{bucket}/caltech_images',  # Image data
        'splits': f's3://{bucket}/training_loop/data_augmentation_pipeline/{splits}',  
        'bbox': f's3://{bucket}',
        'model': trained_estimator.model_data  
    }
    
    output_path = f's3://{bucket}/analysis_output'
    analysis_job_name = f'wildlife-analysis-{timestamp}'
    
    hyperparameters = {
        'batch-size-test': 64,
        'num-workers': 8,
        'analysis-only': True 
    }
    
    # Create PyTorch estimator for analysis
    analysis_estimator = PyTorch(
        entry_point='sagemaker_analysis.py',  
        source_dir='./sagemaker_training', 
        role=role,
        instance_type='ml.g4dn.xlarge', 
        instance_count=1,
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name='wildlife-analysis',
        max_run=900, 
        volume_size=20,
        environment={
            'SM_MODEL_DIR': '/opt/ml/model',
            'SM_OUTPUT_DATA_DIR': '/opt/ml/output/data',
        }
    )
    
    print(f"Starting analysis job: {analysis_job_name}")
    print(f"Using trained model from: {trained_estimator.model_data}")
    
    # Start analysis
    analysis_estimator.fit(
        inputs=input_paths,
        job_name=analysis_job_name,
        wait=True
    )
    
    print(f"Analysis job '{analysis_job_name}' completed!")
    
    return analysis_estimator

def launch_combined_job(splits):
    """Launch training followed by analysis."""
    print("="*60)
    print("STEP 1: TRAINING MODEL")
    print("="*60)
    
    trained_estimator, training_job_name, timestamp = launch_training_job(splits)
    
    print("\n" + "="*60)
    print("STEP 2: ANALYZING MODEL PERFORMANCE")
    print("="*60)
    
    # Analysis/Diagnostics
    analysis_estimator = launch_analysis_job(trained_estimator, training_job_name, timestamp, splits)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Training job: {training_job_name}")
    print(f"Analysis job: wildlife-analysis-{timestamp}")
    print(f"Training model artifacts: {trained_estimator.model_data}")
    print(f"Analysis results: {analysis_estimator.model_data}")
    
    return trained_estimator, analysis_estimator

if __name__ == '__main__':
    # Combined pipeline
    splits = 'splitsv2'
    trained_estimator, analysis_estimator = launch_combined_job(splits)