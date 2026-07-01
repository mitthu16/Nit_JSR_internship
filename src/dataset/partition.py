"""
partition.py

Creates IID partitions of the Fashion-MNIST training dataset
for Federated Learning clients.
"""

import random
from torch.utils.data import Subset


def iid_partition(dataset, num_clients=5, seed=42):
    """
    Split the dataset equally among clients.

    Parameters
    ----------
    dataset : PyTorch Dataset

    num_clients : int

    seed : int

    Returns
    -------
    client_datasets : list
    """

    random.seed(seed)

    indices = list(range(len(dataset)))

    random.shuffle(indices)

    samples_per_client = len(dataset) // num_clients

    client_datasets = []

    for i in range(num_clients):

        start = i * samples_per_client
        end = start + samples_per_client

        subset = Subset(dataset, indices[start:end])

        client_datasets.append(subset)

    return client_datasets


if __name__ == "__main__":

    from src.dataset.fashion_mnist import load_fashion_mnist

    train_dataset, _ = load_fashion_mnist()

    clients = iid_partition(train_dataset)

    print("=" * 50)

    print("IID Partition Successful")

    print("=" * 50)

    print("Number of Clients :", len(clients))

    for i, client in enumerate(clients):

        print(f"Client {i+1} -> {len(client)} images")