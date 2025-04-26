from torch.optim import Adam
from torch import nn
# interface.py
input_height = 128
input_width = 128
input_channels = "RGB"

# train.py
image_dir = "data"
num_epochs = 150
learning_rate = 1e-3
batch_size = 154
weight_decay = 0 # For Adam

num_classes = 151 + 1 # 151 Pokemon + 1 for "unlabelled" class

image_dir = "data"
optimizer = Adam
loss_function = nn.CrossEntropyLoss()
