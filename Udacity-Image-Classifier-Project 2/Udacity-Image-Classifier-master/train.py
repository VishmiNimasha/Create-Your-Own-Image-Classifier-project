import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models

def parse_args():
    """
    Parse command-line arguments for training the model.
    """
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--arch', type=str, default="vgg16", help="Pretrained model architecture (default: vgg16)")
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help="Directory to save the checkpoint")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument('--hidden_units', type=int, default=120, help="Number of units in the hidden layer")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--gpu', type=str, default="gpu", help="Specify device to use (gpu/cpu)")
    return parser.parse_args()

def create_train_transforms():
    """
    Define data augmentation and normalization for the training dataset.
    """
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_test_transforms():
    """
    Define resizing, cropping, and normalization for the validation/test dataset.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_datasets(data_dir, train=True):
    """
    Load datasets with the specified transforms.
    """
    transforms = create_train_transforms() if train else create_test_transforms()
    return datasets.ImageFolder(data_dir, transform=transforms)

def setup_data_loader(data, train=True):
    """
    Set up data loader for the given dataset.
    """
    return torch.utils.data.DataLoader(data, batch_size=50, shuffle=train)

def get_device(prefer_gpu):
    """
    Check for GPU availability and return the appropriate device.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("GPU unavailable. Using CPU instead.")
        return torch.device("cpu")

def initialize_model(architecture="vgg16"):
    """
    Load a pre-trained model and freeze its parameters.
    """
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    for param in model.parameters():
        param.requires_grad = False
    return model

def create_classifier(input_size=25088, hidden_units=120, output_size=102):
    """
    Create a fully connected classifier for the model.
    """
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(90, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

def validate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test data and calculate accuracy.
    """
    model.eval()
    total_loss, accuracy = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            
            probabilities = torch.exp(outputs)
            equality = labels == probabilities.max(dim=1)[1]
            accuracy += equality.type(torch.FloatTensor).mean().item()

    return total_loss, accuracy

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every=30):
    """
    Train the model and validate it periodically.
    """
    model.train()
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = validate_model(model, valid_loader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Training Loss: {running_loss/print_every:.4f} | "
                      f"Validation Loss: {valid_loss/len(valid_loader):.4f} | "
                      f"Validation Accuracy: {accuracy/len(valid_loader):.4f}")
                running_loss = 0
                model.train()
    return model

def save_checkpoint(model, save_path, train_data):
    """
    Save the trained model checkpoint to the specified path.
    """
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': train_data.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load data and create data loaders
    data_dir = 'flowers'
    train_data = load_datasets(f"{data_dir}/train", train=True)
    valid_data = load_datasets(f"{data_dir}/valid", train=False)
    test_data = load_datasets(f"{data_dir}/test", train=False)

    train_loader = setup_data_loader(train_data, train=True)
    valid_loader = setup_data_loader(valid_data, train=False)
    test_loader = setup_data_loader(test_data, train=False)

    # Initialize model, classifier, and device
    model = initialize_model(architecture=args.arch)
    model.classifier = create_classifier(hidden_units=args.hidden_units)
    device = get_device(prefer_gpu=(args.gpu == "gpu"))
    model.to(device)

    # Set up optimizer and loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    trained_model = train_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs)

    # Evaluate the model on the test data
    test_loss, test_accuracy = validate_model(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy/len(test_loader):.4f}")

    # Save the trained model
    save_checkpoint(trained_model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
