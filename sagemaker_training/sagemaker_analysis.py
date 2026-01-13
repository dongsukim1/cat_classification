import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SageMaker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import argparse
from pathlib import Path
import pandas as pd
import tarfile
import boto3

from wildlife_dataloader_sm import create_datasets_and_dataloaders_sm
from model_cnn import WildlifeCNN
from model_efficient_net import WildLifeEfficientNet

def parse_args():
    """Parse command line arguments for SageMaker analysis."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--splits', type=str, default=os.environ.get('SM_CHANNEL_SPLITS'))
    parser.add_argument('--bbox-data', type=str, default=os.environ.get('SM_CHANNEL_BBOX'))
    parser.add_argument('--model', type=str, default=os.environ.get('SM_CHANNEL_MODEL'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Analysis hyperparameters
    parser.add_argument('--batch-size-test', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--analysis-only', type=bool, default=True)
    
    return parser.parse_args()

def extract_model_artifacts(model_path, extraction_dir):
    """Extract the trained model from the tar.gz artifact."""
    print(f"Extracting model from {model_path}")
    
    model_files = list(Path(model_path).glob("*.tar.gz"))
    if not model_files:
        raise ValueError(f"No model.tar.gz found in {model_path}")
    
    model_file = model_files[0]
    print(f"Found model file: {model_file}")
    
    # Extract the tar.gz file
    with tarfile.open(model_file, 'r:gz') as tar:
        tar.extractall(extraction_dir)
    
    # Look for the actual model file
    model_pth_files = list(Path(extraction_dir).glob("**/*.pth"))
    if not model_pth_files:
        raise ValueError(f"No .pth file found after extraction")
    
    return str(model_pth_files[0])

def load_model_and_data(model_path, data_dir, labels_file, splits_dir):
    """Load the trained model and test dataset for SageMaker."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Extract model from artifacts if needed
    if model_path and os.path.exists(model_path):
        extraction_dir = "/tmp/model_extraction"
        os.makedirs(extraction_dir, exist_ok=True)
        actual_model_path = extract_model_artifacts(model_path, extraction_dir)
        print(f"Using extracted model: {actual_model_path}")
    else:
        raise ValueError(f"Model path not found: {model_path}")
    
    # Define target species (same as training)
    target_species = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
    
    # Create datasets and dataloaders
    datasets, dataloaders, bbox_dict, label_to_idx = create_datasets_and_dataloaders_sm(
        data_dir=data_dir,
        labels_file=os.path.join(labels_file, 'bboxes.json'),
        splits_dir=splits_dir,
        target_species=target_species,
        batch_size_train=32,
        batch_size_val=64,
        batch_size_test=64,
        num_workers=8
    )
    
    # Initialize and load model
    num_classes = len(target_species)
    model = WildLifeEfficientNet(num_classes=num_classes)
    model.load_state_dict(torch.load(actual_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {num_classes} classes")
    
    return model, dataloaders['test'], device, label_to_idx

def evaluate_by_class(model, test_loader, device, label_to_idx):
    """Evaluate model performance by class."""
    model.eval()
    
    # Reverse the label mapping for interpretation
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    
    # Initialize counters
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    # Store all predictions and true labels for detailed analysis
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Update counters for each class
            c = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return class_correct, class_total, all_preds, all_labels, idx_to_label

def print_class_accuracy(class_correct, class_total, idx_to_label):
    """Print accuracy for each class."""
    print("\n" + "="*60)
    print("ACCURACY BY CLASS")
    print("="*60)
    
    accuracies = []
    for i in range(len(class_correct)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            accuracies.append(accuracy)
            print(f'{idx_to_label[i]:15s}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            accuracies.append(0)
            print(f'{idx_to_label[i]:15s}: No samples')
    
    overall_accuracy = 100 * sum(class_correct) / sum(class_total)
    print("-" * 60)
    print(f'{"Overall":15s}: {overall_accuracy:.2f}% ({int(sum(class_correct))}/{int(sum(class_total))})')
    
    return accuracies, overall_accuracy

def plot_confusion_matrix(all_labels, all_preds, idx_to_label, save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
                yticklabels=[idx_to_label[i] for i in range(len(idx_to_label))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()  # Close to free memory
    return cm

def plot_class_accuracy(accuracies, idx_to_label, overall_accuracy, save_path=None):
    """Plot accuracy by class as a bar chart."""
    classes = [idx_to_label[i] for i in range(len(accuracies))]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{accuracy:.1f}%', ha='center', va='bottom')
    
    plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                label=f'Overall Accuracy: {overall_accuracy:.2f}%')
    plt.legend()
    plt.title('Model Accuracy by Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Class accuracy plot saved to {save_path}")
    
    plt.close()  # Close to free memory

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close()  # Close to free memory

def generate_classification_report(all_labels, all_preds, idx_to_label):
    """Generate detailed classification report."""
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    report = classification_report(all_labels, all_preds, 
                                  target_names=target_names, output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    return report_df

def analyze_misclassifications(all_labels, all_preds, idx_to_label, top_n=10):
    """Analyze the most common misclassifications."""
    misclassified = []
    
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label != pred_label:
            misclassified.append((idx_to_label[true_label], idx_to_label[pred_label]))
    
    misclass_df = pd.DataFrame(misclassified, columns=['True', 'Predicted'])
    misclass_counts = misclass_df.groupby(['True', 'Predicted']).size().reset_index(name='Count')
    misclass_counts = misclass_counts.sort_values('Count', ascending=False)
    
    print("\n" + "="*60)
    print(f"TOP {top_n} MOST COMMON MISCLASSIFICATIONS")
    print("="*60)
    print(misclass_counts.head(top_n).to_string(index=False))
    
    return misclass_counts

def save_results_to_s3(results, output_dir):
    """Save analysis results to the output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    results_path = os.path.join(output_dir, 'class_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

def main():
    """Main analysis function for SageMaker."""
    args = parse_args()
    
    print(f"Analysis Arguments: {args}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load model and data
    print("Loading model and data...")
    model, test_loader, device, label_to_idx = load_model_and_data(
        args.model, args.train, args.bbox_data, args.splits
    )
    
    print("Evaluating model...")
    class_correct, class_total, all_preds, all_labels, idx_to_label = evaluate_by_class(
        model, test_loader, device, label_to_idx
    )
    
    # Print basic accuracy by class
    accuracies, overall_accuracy = print_class_accuracy(class_correct, class_total, idx_to_label)
    
    # Generate detailed reports
    report_df = generate_classification_report(all_labels, all_preds, idx_to_label)
    
    # Create plots and save to output directory
    output_dir = args.output_data_dir or '/opt/ml/output/data'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        history_path = "/tmp/model_extraction/training_history.json"
        if os.path.exists(history_path):
            print("Loading training history...")
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            plot_training_history(
                history, 
                save_path=os.path.join(output_dir, 'training_history.png')
            )
        else:
            print("Training history not found in model artifacts")
    except Exception as e:
        print(f"Could not load training history: {e}")

    plot_class_accuracy(
        accuracies, idx_to_label, overall_accuracy, 
        save_path=os.path.join(output_dir, 'class_accuracy.png')
    )
    
    plot_confusion_matrix(
        all_labels, all_preds, idx_to_label,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    misclass_counts = analyze_misclassifications(all_labels, all_preds, idx_to_label)
    
    # Compile results
    results = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': {idx_to_label[i]: accuracies[i] for i in range(len(accuracies))},
        'class_counts': {idx_to_label[i]: int(class_total[i]) for i in range(len(class_total))},
        'misclassifications': misclass_counts.to_dict('records') if len(misclass_counts) > 0 else [],
        'classification_report': report_df.to_dict()
    }
    
    # Save results
    save_results_to_s3(results, output_dir)
    
    # Save classification report as CSV
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Save misclassifications as CSV
    if len(misclass_counts) > 0:
        misclass_counts.to_csv(os.path.join(output_dir, 'misclassifications.csv'), index=False)
    
    print("Analysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - class_analysis.json")
    print(f"  - classification_report.csv")
    print(f"  - class_accuracy.png")
    print(f"  - confusion_matrix.png")
    if len(misclass_counts) > 0:
        print(f"  - misclassifications.csv")

if __name__ == '__main__':
    main()