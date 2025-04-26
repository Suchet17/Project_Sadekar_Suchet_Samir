import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import PokemonDataset
from model import PokemonCNN
from config import image_dir

def predictor(file_names):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PokemonCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", weights_only=True))

    model.eval()

    test_dataset = PokemonDataset(image_dir, file_names)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=True)
    labs = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = torch.exp(outputs)/torch.sum(torch.exp(outputs), dim=1, keepdim=True) # softmax
        labs.append(outputs.argmax(dim=1).cpu().numpy())
    return labs