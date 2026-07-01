"""
fashion_mnist.py

This module downloads and loads the Fashion-MNIST dataset.
"""

from torchvision import datasets, transforms


def load_fashion_mnist(data_dir="../../data"):
    """
    Downloads (if necessary) and returns the Fashion-MNIST
    training and testing datasets.
    """

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Training dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Testing dataset
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


if __name__ == "__main__":

    train_dataset, test_dataset = load_fashion_mnist()

    print("=" * 40)
    print("Fashion-MNIST Loaded Successfully")
    print("=" * 40)

    print("Training Images :", len(train_dataset))
    print("Testing Images  :", len(test_dataset))

    image, label = train_dataset[0]

    print("Image Shape :", image.shape)
    print("Label :", label)