from PIL import Image
import os
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, crop_box, resize_size):
    """
    Crop and resize images in the input directory and save them to the output directory.

    Parameters:
        input_dir (str): Path to the folder containing input images.
        output_dir (str): Path to the folder to save processed images.
        crop_box (tuple): The cropping box as (left, top, width, height).
        resize_size (tuple): The size to resize the image to as (width, height).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract cropping parameters
    left, top, width, height = crop_box

    # Process each image in the input directory
    for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
        input_path = os.path.join(input_dir, filename)

        # Skip non-image files
        if not filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
            continue

        try:
            # Open the image
            with Image.open(input_path) as img:
                # Ensure image is in RGB mode
                img = img.convert("RGB")

                # Crop the image
                cropped_img = img.crop((left, top, left + width, top + height))

                # Resize the image
                resized_img = cropped_img.resize(resize_size, Image.Resampling.LANCZOS)

                # Save the processed image
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
if __name__ == "__main__":
    # Paths
    source_folder = 'test_fold4'
    system = 'olympus'
    # system = 'fujifilm'
    folder = '/20230801-125615-es06-hd'
    
    print(f"Preprocessing images in {folder} for {system} system...")
    
    input_directory = "/home/tanaka/" + source_folder + "/" + system + "/" + folder
    output_directory = "/home/tanaka/demo_data/" + folder
    
    # 結果保存folderを作成
    if not os.path.exists(os.path.join(output_directory)):
        os.mkdir(os.path.join(output_directory))

    # Cropping and resizing parameters
    crop_box_olympus = (710, 20, 1180, 1040)  # left, top, width, height
    crop_box_fujifilm = (330, 25, 1260, 970)  # left, top, width, height
    resize_dim = (224, 224)  # width, height
    
    # Preprocess images
    if system == 'olympus':
        preprocess_images(input_directory, output_directory, crop_box_olympus, resize_dim)
    elif system == 'fujifilm':
        preprocess_images(input_directory, output_directory, crop_box_fujifilm, resize_dim)
