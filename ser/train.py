
import os
import torch
import torch.nn.functional as F
from ser.data import get_data_loaders
from ser.model import Net
from torch import optim
import datetime
import json


def training_function(epochs, training_dataloader, device, model, optimizer, validation_dataloader):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
        # validate
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )
    
    


def save_results(model, epochs, batch_size, learning_rate) :  
    # Construct runs directory
    runs_directory = "runs"

    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory)
        print(f"Directory '{runs_directory}' created.")

    # Get current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Define file name (and path, optionally)
    # Create directory for each run
    directory_path = runs_directory + "/" + formatted_time

    os.makedirs(directory_path) 

    # Saving
    torch.save(model, directory_path + "/" + 'model.pth')

    hyperparameters = {
        "epochs": epochs, 
        "batch_size": batch_size, 
        "learning_rate": learning_rate}
    
    # Specify the file name
    filename = directory_path + "/" + 'hyperparameters.json'

    # Write the dictionary to a file in JSON format
    with open(filename, 'w') as file:
        json.dump(hyperparameters, file, indent=4)

    print(f"Directory '{directory_path}' created with model and hyperparameters.")
