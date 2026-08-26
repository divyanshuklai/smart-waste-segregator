import onnxruntime
import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional

class WasteClassifierInference:
    def __init__(self, model_path: str, class_names: List[str]):
        """Initialize the waste classifier inference engine"""
        self.class_names = class_names
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load model
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess the input image for model inference"""
        try:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (224, 224))
            
            # Convert to float32 explicitly
            img = img.astype(np.float32)
            
            # Normalize to [0, 1]
            img /= 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            # Debug: Print data type and range
            self.logger.info(f"Image dtype: {img.dtype}")
            self.logger.info(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
            
            # Transpose to NCHW format and add batch dimension
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            
            # Ensure float32
            img = img.astype(np.float32)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Run inference on an image"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_path)
            
            # Verify input shape and type
            self.logger.info(f"Input tensor shape: {input_data.shape}")
            self.logger.info(f"Input tensor type: {input_data.dtype}")
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # Process results
            scores = outputs[0][0]
            
            # Apply softmax
            scores = scores - np.max(scores)  # For numerical stability
            exp_scores = np.exp(scores)
            probabilities = exp_scores / exp_scores.sum()
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            
            return self.class_names[predicted_idx], confidence
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

def main():
    # Configuration
    model_path = 'waste_classifier.onnx'
    image_path = 'metal_test.jpg'
    class_names = ['metal', 'plastic', 'glass', 'biodegradable']
    
    try:
        # Initialize classifier
        classifier = WasteClassifierInference(model_path, class_names)
        
        # Run prediction
        predicted_class, confidence = classifier.predict(image_path)
        
        # Display results
        print("\nResults:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()