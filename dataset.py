import torchvision.transforms as transforms
trans = transforms.ToTensor()
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from config import input_height, input_width, input_channels

class PokemonDataset(Dataset):
    def __init__(self, in_dir, x=""): # x == 'train' or 'test' or 'val'
        self.root_dir = in_dir
        gen1 = pd.read_csv("Labels.csv")
        if type(x) == str:
            if x != "":
                gen1 = gen1[gen1["Set"]==x]
            self.image_files = list(gen1['FileName'])
            self.labels = list(gen1['Label']-1)
        else:
            self.root_dir = in_dir
            gen1 = pd.read_csv("Labels.csv")
            gen1 = gen1[gen1["FileName"].isin(x)].reset_index(drop=True)
            self.image_files = list(gen1['FileName'])
            self.labels = list(gen1['Label']-1)  # label-1 to make it 0-indexed

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        file = self.image_files[index]
        label = self.labels[index]
        image = Image.open(f"{self.root_dir}/{file}").convert(input_channels)        
        image = image.resize((input_width, input_height))
        image = trans(image) #Normalized
        return [image, label]

whole_dataset = PokemonDataset("data")
whole_loader = DataLoader(whole_dataset, batch_size=32, shuffle=True, num_workers=4)