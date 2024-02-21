from pathlib import Path
from ser.train import training_function, save_results
from ser.data import get_data_loaders
from ser.model import Net
from torch import optim
import torch
import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()

def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        default = 2, help="Number of epochs for experiment."
    ),
    batch_size: int = typer.Option(
        default = 1000, help="Batch size for experiment."
    ),
    learning_rate: float = typer.Option(
        default = 0.01, help="Learning rate for experiment."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_dataloader, validation_dataloader = get_data_loaders(batch_size)
    
    # train
    training_function(epochs, training_dataloader, device, model, optimizer, validation_dataloader)
    
    # Save results to runs folder with hyperparameters etc
    save_results(model, epochs, batch_size, learning_rate)


@main.command()
def infer():
    print("This is where the inference code will go")
