import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import onnx
import onnxruntime
import numpy as np

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(WasteClassifier, self).__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def convert_to_onnx(checkpoint_path='best_model.pth', onnx_path='waste_classifier.onnx'):
    # Initialize model
    model = WasteClassifier(num_classes=4)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export the model
    torch.onnx.export(model,                     # model being run
                     dummy_input,                # model input (or a tuple for multiple inputs)
                     onnx_path,                  # where to save the model
                     export_params=True,         # store the trained parameter weights inside the model file
                     opset_version=11,           # the ONNX version to export the model to
                     do_constant_folding=True,   # whether to execute constant folding for optimization
                     input_names=['input'],      # the model's input names
                     output_names=['output'],    # the model's output names
                     dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                 'output': {0: 'batch_size'}})
    
    print(f"Model exported to {onnx_path}")
    
    # Verify the exported model
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)
        
        # Create an ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # Prepare sample input
        sample_input = dummy_input.numpy()
        
        # Run inference with ONNX Runtime
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Run inference with PyTorch for comparison
        with torch.no_grad():
            torch_output = model(dummy_input)
        
        # Compare the results
        np.testing.assert_allclose(torch_output.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        
        # Print model input and output shapes
        print("\nModel Input Shape:", ort_session.get_inputs()[0].shape)
        print("Model Output Shape:", ort_session.get_outputs()[0].shape)
        
        # Get model size
        import os
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # Convert to MB
        print(f"\nModel Size: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting model conversion...")
        convert_to_onnx()
    except Exception as e:
        print(f"Error: {str(e)}")