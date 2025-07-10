"""
Contains functionality for creating PyTorch DataLoaders for image
classification data.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and testing DataLoaders for image classification data.

    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        transform (transforms.Compose): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to NUM_WORKERS.

    Returns:
        tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names
