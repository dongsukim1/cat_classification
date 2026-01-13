import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from pathlib import Path

dataloader_path = './EC2+s3/data_augmentation_pipeline'
sys.path.append(dataloader_path)

from wildlife_dataloader import create_datasets_and_dataloaders

# Import your CNN model (assuming it's in a file called model.py)
from model_cnn import SimpleWildlifeCNN

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    """
    Train the model and validate after each epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize history
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
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

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
    plt.savefig(save_path)
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets and dataloaders
    datasets, dataloaders, bbox_dict, label_to_idx = create_datasets_and_dataloaders(
        data_dir='./data/full_s3',
        labels_file='./EC2+s3/bboxes.json',
        splits_dir='./EC2+s3/data_augmentation_pipeline/splits',
        target_species=["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"],
        batch_size_train=32,
        batch_size_val=64,
        batch_size_test=64,
        num_workers=4
    )
    
    # Get class weights for handling imbalanced data
    class_weights = datasets['train'].get_class_weights().to(device)
    
    # Initialize model
    num_classes = len(label_to_idx)
    model = SimpleWildlifeCNN(num_classes=num_classes)
    
    # Print model architecture and parameter count
    print("Model Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device
    )
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    
    # Save the best model
    torch.save(model.state_dict(), 'best_model_weights.pth')
    print("Saved best model weights to 'best_model_weights.pth'")
    
    # Evaluate on test set
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

if __name__ == '__main__':
    main()