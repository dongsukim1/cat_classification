import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse
from pathlib import Path

from wildlife_dataloader_sm import create_datasets_and_dataloaders_sm
from model_cnn import WildlifeCNN
from model_efficient_net import WildLifeEfficientNet

def parse_args():
    """Parse command line arguments for SageMaker training."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--splits', type=str, default=os.environ.get('SM_CHANNEL_SPLITS'))
    parser.add_argument('--bbox-data', type=str, default=os.environ.get('SM_CHANNEL_BBOX'))
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size-train', type=int, default=32)
    parser.add_argument('--batch-size-val', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=8)
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training function adapted for SageMaker."""
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
                
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
        print()
    
    model.load_state_dict(best_model_wts)
    return model, history

def save_model(model, model_dir):
    """Save model for SageMaker."""
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def main():
    """Main training function for SageMaker."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    
    print(f"Arguments: {args}")
    
    # Define target species
    target_species = [
        "mountain_lion", 
        "bobcat", 
        "coyote", 
        "fox", 
        "deer", 
        "empty"
    ]

    # Create datasets and dataloaders using SageMaker paths
    datasets, dataloaders, _, label_to_idx = create_datasets_and_dataloaders_sm(
        data_dir=args.train, 
        labels_file=os.path.join(args.bbox_data, 'bboxes.json'),
        splits_dir=args.splits,
        target_species=target_species,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        batch_size_test=64,
        num_workers=args.num_workers,
        use_labels=False
    )
    
    # Get class weights
    class_weights = datasets['train'].get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    num_classes = len(label_to_idx)
    model = WildLifeEfficientNet(num_classes=num_classes)
    
    print("Model Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device
    )
    
    # Evaluate on test set if available
    if 'test' in dataloaders:
        model.eval()
        test_corrects = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                test_corrects += torch.sum(preds == labels.data)
                test_total += labels.size(0)
        
        test_acc = test_corrects.double() / test_total
        print(f'Test Accuracy: {test_acc:.4f}')
    
    # Save the model
    save_model(model, args.model_dir)
    
    # Save training history
    history_path = os.path.join(args.model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': history['train_loss'],
            'train_acc': [float(acc) for acc in history['train_acc']],
            'val_loss': history['val_loss'],
            'val_acc': [float(acc) for acc in history['val_acc']]
        }
        json.dump(history_json, f)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()