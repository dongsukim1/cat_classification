import torch
import torch.nn as nn

class WildlifeCNN(nn.Module):
    """
    CNN model for wildlife image classification.
    """
    
    def __init__(self, num_classes=6):
        super(WildlifeCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Create model instance
    num_classes = 6  # mountain_lion, bobcat, coyote, fox, deer, empty
    model = WildlifeCNN(num_classes=num_classes)
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    
    # Calculate and print number of parameters
    num_params = count_parameters(model)
    print(f"\nNumber of trainable parameters: {num_params:,}")
    
    # Test with a dummy input (batch_size=4, 3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print class predictions
    _, predicted = torch.max(output, 1)
    print(f"Predicted classes: {predicted}")