import os
from test import predict_image
from tabulate import tabulate

def test_images_in_directory(directory):
    """Test all images in the specified directory"""
    results = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # Get all image files
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(directory, filename)
            
            # Get prediction
            print(f"Analyzing {filename}...")
            result = predict_image(image_path)
            
            # Store results
            results.append([
                filename,
                result['label'],
                f"{result['confidence']:.2f}%",
                result['analysis']
            ])
    
    # Display results in a table
    headers = ["Image", "Prediction", "Confidence", "Analysis"]
    print("\nResults:")
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    # Test directory path
    test_dir = "test_images"  # Create this directory and put your test images in it
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory '{test_dir}'. Please add your test images to this directory.")
    else:
        test_images_in_directory(test_dir)
