from torch.utils.data import DataLoader
from torchvision import datasets

from ser.constants import DATA_DIR
from ser.transforms import flip, normalize, transforms


def train_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=True, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)


def val_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)


# Returns dataloader with mnist validation data.
# Note: normally this would be real 'test' or future data
def test_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

def _select_test_image(label, flip_bool):
    # TODO `ts` is a list of transformations that will be applied to the loaded
    # image. This works... but in order to add a transformation, or change one,
    # we now have to come and edit the code... which sucks. What if we could
    # configure the transformations via the cli?
    ts = [normalize, (flip, flip_bool)]
    dataloader = test_dataloader(1, transforms(*ts))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images