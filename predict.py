import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import PokemonDataset
from model import PokemonCNN
from config import image_dir

def predictor(file_paths):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = PokemonCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", weights_only=True))
    model.eval()

    test_dataset = PokemonDataset(image_dir, file_paths)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    log = open("temp.txt", 'w')
    acc = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = torch.exp(outputs)/torch.sum(torch.exp(outputs), dim=1, keepdim=True)
        # outputs = torch.softmax(outputs, dim=1)
        if torch.argmax(outputs).item() == labels.item():
            acc += 1
    acc /= len(test_loader)
    log.close()

predictor()