import torch
import torch.nn as nn
import torchvision.models as models

class WildLifeEfficientNet(nn.Module):
    def __init__(self, num_classes=6, freeze_backbone=False):
        super(WildLifeEfficientNet, self).__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        # Load pre-trained EfficientNet-B3 model
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the classifier to match the number of target classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, X):
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
        return self.model(X)

    def classify(self, X):
        logits = self.forward(X)
        return torch.argmax(logits, dim=1)
    
def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Create model instance
    num_classes = 6  # mountain_lion, bobcat, coyote, fox, deer, empty
    model = WildLifeEfficientNet(num_classes=num_classes)
    
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