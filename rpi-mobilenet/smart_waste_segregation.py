import onnxruntime
import numpy as np
import cv2
import subprocess
import time
import sys

def countdown(duration: int = 5):
    """Display a countdown in the console"""
    print("\nPrepare to place object in front of camera")
    print("Countdown starting...")
    
    for i in range(duration, 0, -1):
        sys.stdout.write(f'\rCapturing in {i} seconds...')
        sys.stdout.flush()
        time.sleep(1)
    print("\nCapturing now!")

def capture_image(image_path: str = 'waste_image.jpg') -> bool:
    """Capture image using libcamera-still"""
    try:
        # Show countdown
        countdown(5)
        
        # Capture image
        command = [
            'libcamera-still',
            '-o', image_path,
            '--nopreview',
            '--width', '2592',
            '--height', '1944',
            '-t', '1000'  # 1 second delay
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to capture image: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during capture: {e}")
        return False

def load_session(model_path: str):
    """Build the onnxruntime session once, ahead of the capture loop"""
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4

    return onnxruntime.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider'],
        sess_options=session_options
    )

def process_image(session, image_path: str, class_names: list):
    """Process the captured image with the ONNX model"""
    try:
        # Load and preprocess image using OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose to NCHW format (batch, channels, height, width)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # Ensure we're using float32
        img = img.astype(np.float32)
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, {session.get_inputs()[0].name: img})
        inference_time = time.time() - start_time
        
        # Process results
        scores = outputs[0][0]
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()
        predicted_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_idx])
        
        return class_names[predicted_idx], confidence, inference_time
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def display_results(predicted_class: str, confidence: float, inference_time: float):
    """Display results in a formatted way"""
    print("\n" + "="*40)
    print("       Classification Results")
    print("="*40)
    print(f"Waste Type: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Processing Time: {inference_time:.3f} seconds")
    print("="*40 + "\n")

def main():
    # Configuration
    image_path = 'waste_image.jpg'
    model_path = 'waste_classifier.onnx'
    # NOTE: this ordering is wrong for the shipped weights, which expect
    # ['metal', 'plastic', 'glass', 'biodegradable']. It is left as it shipped
    # because the mislabelled output is part of the record. See the README.
    class_names = ['plastic', 'paper', 'metal', 'glass']
    
    print("\nSmart Waste Segregation System")
    print("------------------------------")
    
    # Build the inference session once, outside the capture loop
    session = load_session(model_path)
    
    while True:
        # Capture image
        if not capture_image(image_path):
            print("Failed to capture image. Exiting.")
            return
        
        # Process image
        print("\nProcessing image...")
        predicted_class, confidence, inference_time = process_image(
            session, image_path, class_names
        )
        
        if predicted_class is not None:
            display_results(predicted_class, confidence, inference_time)
        else:
            print("Failed to process image")
        
        # Ask if user wants to classify another object
        while True:
            choice = input("\nWould you like to classify another object? (y/n): ").lower()
            if choice in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if choice == 'n':
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nThank you for using Smart Waste Segregation System!")