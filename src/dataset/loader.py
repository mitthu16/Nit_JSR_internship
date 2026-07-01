"""
loader.py

Creates PyTorch DataLoaders for FedBench clients.
"""

from torch.utils.data import DataLoader

from src.dataset.fashion_mnist import load_fashion_mnist
from src.dataset.partition import iid_partition
from src.utils.config import ConfigManager

config = ConfigManager()

batch_size = config.training["batch_size"]

def get_dataloaders(
    client_id,
    num_clients=5,
    batch_size=batch_size,
):
    """
    Returns one client's DataLoader and the global test DataLoader.
    """

    train_dataset, test_dataset = load_fashion_mnist()

    client_datasets = iid_partition(
        train_dataset,
        num_clients=num_clients
    )

    trainloader = DataLoader(
        client_datasets[client_id],
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader