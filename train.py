import warnings
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from dataset import PokemonDataset

from config import num_epochs, batch_size
from config import optimizer, loss_function

train_dataset = PokemonDataset("data", "train")
val_dataset = PokemonDataset("data", "val")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def trainer(model, num_epochs=num_epochs, train_loader = train_loader, criterion=loss_function, optimizer=optimizer, val_loader = val_loader):
    
    from config import learning_rate, weight_decay
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    warnings.filterwarnings("ignore")
    now = datetime.now

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Define Data
    model = model.to(device)

    log = open(f"checkpoints/output.log", 'w')

    # Define Loss and Optimizer

    print(f"Size of training set = {len(train_loader)}", file=log)
    print(f"Size of validation set = {len(val_loader)}", file=log)
    print(f"Batch size = {batch_size}", file=log)
    print(f"Loss function = {criterion}", file=log)
    print(f"Optimizer = {optimizer}", file=log)
    print(file=log, flush=True)

    # images, labels = next(iter(train_loader))
    start = now()
    print("Start", start, file=log)
    # Training Loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} {now()}", end=' ', file=log, flush=True)
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if i%100 == len(train_loader)%100:
                print(len(train_loader)-i, end=' ', file=log, flush=True)
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            # print("\n", images.shape, labels, file=log, flush=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient-Descent
            optimizer.step()
            running_loss += loss.item()
        
        # Validation error
        print("\nValidating...", end=' ', file=log, flush=True)
        model.eval()
        accuracy = 0
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i in torch.argmax(outputs) == labels:
                if i == True:
                    accuracy += 1
        accuracy /= len(val_loader)


        print(f"Loss = {running_loss/len(train_loader):.12f}", file=log)
        print(f"Validation Accuracy = {accuracy:.6f}\n", file=log, flush=True)
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1}_loss{running_loss/len(train_loader):.6f}_acc{accuracy*100:.4f}.pth")
    torch.save(model.state_dict(), f"checkpoints/final_weights.pth")
    print("Training complete!", file=log, flush=True)
    print(f"Time taken for {epoch+1} epochs = {now()-start}", file=log, flush=True)
    log.close()