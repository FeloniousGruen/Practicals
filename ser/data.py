from pathlib import Path
from torch.utils.data import DataLoader
from ser.transforms import get_transforms
from torchvision import datasets


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_data_loaders(batch_size = 1000): 
    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return training_dataloader, validation_dataloader