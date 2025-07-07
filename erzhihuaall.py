import os
from PIL import Image
import numpy as np

def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each model folder in the input folder
    for model_name in os.listdir(input_folder):
        model_path = os.path.join(input_folder, model_name)
        if os.path.isdir(model_path):  # Check if it's a folder
            output_model_path = os.path.join(output_folder, model_name)
            os.makedirs(output_model_path, exist_ok=True)

            # Process each image in the model folder
            for image_name in os.listdir(model_path):
                if image_name.endswith(('.jpg', '.png', '.tif')):  # Supported image formats
                    input_image_path = os.path.join(model_path, image_name)
                    output_image_path = os.path.join(output_model_path, image_name)

                    # Open the image and process
                    image = Image.open(input_image_path).convert('L')  # Convert to grayscale
                    image_array = np.array(image)

                    # Apply thresholding
                    processed_array = (image_array > 200).astype(np.uint8) * 255

                    # Save the processed image
                    processed_image = Image.fromarray(processed_array)
                    processed_image.save(output_image_path)

    print(f"Processing completed. Results saved in {output_folder}.")

# Input and output folder paths
input_folder = "H:/Feng/duibi/all/xihua"
output_folder = "H:/Feng/duibi/all/xihua2"

# Run the processing function
process_images(input_folder, output_folder)
