import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import verify_gpu
from torchvision import models

def parse_arguments():
    """
    Parse command-line arguments for the prediction script.
    """
    parser = argparse.ArgumentParser(description="Prediction script for flower classification.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to display.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names JSON file.')
    parser.add_argument('--gpu', type=str, default='gpu', help='Enable GPU for inference if available.')
    return parser.parse_args()

def load_trained_model(checkpoint_path):
    """
    Load a pre-trained model and restore its parameters from a checkpoint.
    """
    saved_data = torch.load(checkpoint_path)
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"

    # Freeze pre-trained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load the saved classifier and state dictionary
    model.class_to_idx = saved_data['class_to_idx']
    model.classifier = saved_data['classifier']
    model.load_state_dict(saved_data['state_dict'])
    return model

def preprocess_image(image_path):
    """
    Resize, crop, and normalize an image for use in a PyTorch model.
    """
    image = PIL.Image.open(image_path)

    # Determine new dimensions based on aspect ratio
    original_width, original_height = image.size
    if original_width < original_height:
        new_size = [256, int(256 * original_height / original_width)]
    else:
        new_size = [int(256 * original_width / original_height), 256]
    image.thumbnail(new_size)

    # Center crop the image to 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    cropped_image = image.crop((left, top, right, bottom))

    # Convert image to a NumPy array and normalize
    image_array = np.array(cropped_image) / 255.0
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    normalized_image = (image_array - mean) / std_dev

    # Reorder dimensions to match PyTorch's expected format
    final_image = normalized_image.transpose((2, 0, 1))
    return final_image

def make_prediction(image_path, model, device, categories, top_k=5):
    """
    Perform inference on an image and return the top predictions.
    """
    # Prepare image and transfer to the device
    image = preprocess_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    # Perform a forward pass and get probabilities
    with torch.no_grad():
        output = model.forward(image_tensor)
    probabilities, indices = torch.topk(torch.exp(output), top_k)
    probabilities = probabilities.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    # Map indices to class labels and flower names
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    class_labels = [idx_to_class[idx] for idx in indices]
    flower_names = [categories[label] for label in class_labels]
    return probabilities, class_labels, flower_names

def display_predictions(probabilities, flowers):
    """
    Print the top predicted classes along with their probabilities.
    """
    for rank, (flower, probability) in enumerate(zip(flowers, probabilities), 1):
        print(f"Rank {rank}: Flower = {flower}, Likelihood = {ceil(probability * 100)}%")

def main():
    # Parse input arguments
    args = parse_arguments()

    # Load category labels from a JSON file
    with open(args.category_names, 'r') as file:
        category_labels = json.load(file)

    # Restore the trained model
    model = load_trained_model(args.checkpoint)

    # Determine device based on user input and availability
    device = verify_gpu(gpu_arg=args.gpu)

    # Generate predictions
    probabilities, class_labels, flower_names = make_prediction(
        args.image, model, device, category_labels, args.top_k
    )

    # Display the predictions
    display_predictions(probabilities, flower_names)

if __name__ == '__main__':
    main()
