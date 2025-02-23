import argparse
from test import predict_image as model1_predict
from another_model import predict_image as model2_predict
from pathlib import Path
from tabulate import tabulate
import time

def analyze_image(image_path):
    """Analyze a single image using both models"""
    print(f"\nAnalyzing image: {image_path}")
    print("=" * 50)
    
    # Test with first model
    print("\nTesting with Model 1 (deepfake_vs_real)...")
    start_time = time.time()
    result1 = model1_predict(image_path)
    time1 = time.time() - start_time
    
    # Test with second model
    print("\nTesting with Model 2 (Deep-Fake-Detector-v2)...")
    start_time = time.time()
    label2, confidence2 = model2_predict(image_path)
    time2 = time.time() - start_time
    
    # Prepare results table
    results = [
        ["Model 1", result1['label'], f"{result1['confidence']:.2f}%", f"{time1:.2f}s", result1['analysis']],
        ["Model 2", label2, f"{confidence2:.2f}%", f"{time2:.2f}s", "Basic analysis"]
    ]
    
    # Display results
    headers = ["Model", "Prediction", "Confidence", "Time", "Analysis"]
    print("\nResults:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Agreement check
    if result1['label'] == label2:
        print(f"\n✅ Models agree: Both predict the image is {result1['label']}")
    else:
        print("\n⚠️ Models disagree on classification!")

def main():
    parser = argparse.ArgumentParser(description='Test AI model on a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    # Validate file path
    path = Path(args.image_path)
    if not path.exists():
        print(f"Error: File '{args.image_path}' does not exist!")
        return
    
    if not path.is_file():
        print(f"Error: '{args.image_path}' is not a file!")
        return
        
    if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
        print(f"Error: File '{args.image_path}' is not a supported image format!")
        return
    
    # Analyze the image
    analyze_image(str(path))

if __name__ == "__main__":
    # Test specific image
    image_path = "lastlast.jpg"  # Replace with your image path
    analyze_image(image_path)
