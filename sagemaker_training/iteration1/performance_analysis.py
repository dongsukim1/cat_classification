import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from model_cnn import WildlifeCNN
import sys
dataloader_path = './EC2+s3/data_augmentation_pipeline'
sys.path.append(dataloader_path)

from wildlife_dataloader import create_datasets_and_dataloaders
import pandas as pd

def load_model_and_data(model_path, data_dir, labels_file, splits_dir):
    """Load the trained model and test dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 6
    model = WildlifeCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    target_species = ["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"]
    
    datasets, dataloaders, bbox_dict, label_to_idx = create_datasets_and_dataloaders(
        data_dir=data_dir,
        labels_file=labels_file,
        splits_dir=splits_dir,
        target_species=target_species,
        batch_size_train=32,
        batch_size_val=64,
        batch_size_test=64,
        num_workers=8
    )
    
    return model, dataloaders['test'], device, label_to_idx

def evaluate_by_class(model, test_loader, device, label_to_idx):
    """Evaluate model performance by class."""
    model.eval()
    
    # Reverse the label mapping 
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    
    # Initialize counters
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    # Store all predictions and true labels
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Class accuracy plot saved to {save_path}")
    
    plt.show()

def generate_classification_report(all_labels, all_preds, idx_to_label):
    """Generate detailed classification report."""
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    report = classification_report(all_labels, all_preds, 
                                  target_names=target_names, output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    return report_df

def analyze_misclassifications(all_labels, all_preds, idx_to_label, test_dataset, top_n=10):
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

def main():
    model_path = './EC2+s3/training/model.pth'
    data_dir = './EC2+s3/data/s3+expanded_empty'  # Path to your image data
    labels_file = './EC2+s3/bboxes.json'  # Path to your bboxes file
    splits_dir = './EC2+s3/data_augmentation_pipeline/splitsv2/'  # Path to your splits
    
    print("Loading model and data...")
    model, test_loader, device, label_to_idx = load_model_and_data(
        model_path, data_dir, labels_file, splits_dir
    )
    
    print("Evaluating model...")
    class_correct, class_total, all_preds, all_labels, idx_to_label = evaluate_by_class(
        model, test_loader, device, label_to_idx
    )
    
    # Basic accuracy by class
    accuracies, overall_accuracy = print_class_accuracy(class_correct, class_total, idx_to_label)
    
    # Generate detailed reports
    report_df = generate_classification_report(all_labels, all_preds, idx_to_label)
    
    # Plot results
    plot_class_accuracy(accuracies, idx_to_label, overall_accuracy, 
                       save_path='./EC2+s3/training/class_accuracy.png')
    
    plot_confusion_matrix(all_labels, all_preds, idx_to_label,
                         save_path='./EC2+s3/training/confusion_matrix.png')
    
    # Analyze misclassifications
    misclass_counts = analyze_misclassifications(all_labels, all_preds, idx_to_label, 
                                                test_loader.dataset)
    
    # Save results to JSON
    results = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': {idx_to_label[i]: accuracies[i] for i in range(len(accuracies))},
        'class_counts': {idx_to_label[i]: int(class_total[i]) for i in range(len(class_total))},
        'misclassifications': misclass_counts.to_dict('records')
    }
    
    with open('./EC2+s3/training/class_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to './EC2+s3/training/class_analysis.json'")

if __name__ == '__main__':
    main()