from torchvision import transforms as torch_transforms
import numpy as np
import torch

import pytest

def transforms(*stages):
    composed_transforms = [torch_transforms.ToTensor()]
    
    for stage in stages:
        if isinstance(stage, tuple):
            transform, *args = stage
            composed_transforms.append(transform(*args))
        else:
            composed_transforms.append(stage())
    
    return torch_transforms.Compose(composed_transforms)

def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def flip(flip_bool):
    """
    Flip a tensor both vertically and horizontally
    """
    if flip_bool == 1:
        return torch_transforms.Compose(
        [
            torch_transforms.RandomHorizontalFlip(p=1.0),
            torch_transforms.RandomVerticalFlip(p=1.0),
        ]
    )
    else:  # Assuming flip_bool == 0 for the identity transformation
        return torch_transforms.Compose([
            torch_transforms.Lambda(lambda x: x)  # Identity transformation
        ])


def test_flip():
    # Example 2D pixel array (4x4 for demonstration)
    pixel_array1 = np.array([
        [1, 0],
        [0, 0]], dtype=np.float32)  # Ensure float32 dtype for compatibility
    pixel_array2 = np.array([
        [0, 0],
        [1, 0]], dtype=np.float32)    
    pixel_array3 = np.array([
        [0, 1],
        [0, 0]], dtype=np.float32)  # Ensure float32 dtype for compatibility
    pixel_array4 = np.array([
        [0, 0],
        [0, 1]], dtype=np.float32) 
    # Convert to a PyTorch tensor and add a channel dimension
    tensor_image1 = torch.from_numpy(pixel_array1).unsqueeze(0)  # Shape becomes [1, 2, 2]
    tensor_image2 = torch.from_numpy(pixel_array2).unsqueeze(0)
    tensor_image3 = torch.from_numpy(pixel_array3).unsqueeze(0)  
    tensor_image4 = torch.from_numpy(pixel_array4).unsqueeze(0)   
    # Apply the flip transformation
    transformed_image11 = flip(1)(tensor_image1)
    transformed_image10 = flip(0)(tensor_image1)
    transformed_image21 = flip(1)(tensor_image2)
    transformed_image20 = flip(0)(tensor_image2)   

    assert torch.all(transformed_image11 ==  tensor_image4)
    assert torch.all(transformed_image10 ==  tensor_image1)
    assert torch.all(transformed_image21 ==  tensor_image3)
    assert torch.all(transformed_image20 ==  tensor_image2)


