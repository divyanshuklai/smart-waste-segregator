import torch
import torch.nn as nn
import onnx
import torch.nn.functional as F
import torchvision.models as models

class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GarbageClassifier, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove the classification head (last fully connected layers) and retain feature extractor
        self.feature_extractor = nn.Sequential(
            *list(mobilenet_v2.children())[:-1]  # Keep everything except the classification head
        )
        # Custom fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280 * 7 * 7, 64)  # 1280 features with a 7x7 spatial map
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.08)
        self.fc2 = nn.Linear(64, num_classes)  # 2 output classes
    
    def forward(self, x):
        # Extract features using MobileNetV2
        x = self.feature_extractor(x).to('cuda')
        
        # Flatten the output
        x = x.view(x.size(0), -1).to('cuda')  # Flatten to (batch_size, 1280*7*7)
        
        # Fully connected layers with ReLU, batch normalization, and dropout
        x = F.relu(self.fc1(x)).to('cuda')
        x = self.batch_norm(x).to('cuda')
        x = self.dropout(x).to('cuda')
        
        # Final classification layer
        x = self.fc2(x).to('cuda')
        
        return x

model = torch.load('garbage_classifier_MN95.pth')

model.eval()  # Set to evaluation mode

# Create a dummy input tensor and move it to the same device as the model
dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)  # Move to the same device

# Define the path for the ONNX model
onnx_model_path = 'garbage_classifier_MN95.onnx'

# Export the model to ONNX
torch.onnx.export(model,                  # model being run
                  dummy_input,            # model input (or a tuple for multiple inputs)
                  onnx_model_path,        # where to save the model
                  export_params=True,     # store the trained parameter weights inside the model file
                  opset_version=11,       # the ONNX version to export the model to
                  do_constant_folding=True,# optimize the model by folding constants
                  input_names=['input'],  # the model's input names
                  output_names=['output'], # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

print("Model exported to ONNX format successfully!")