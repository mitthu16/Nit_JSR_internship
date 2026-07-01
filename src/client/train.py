"""
train.py

Local training for Flower Client.
"""

import torch


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epochs=1,
):
    model.train()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy