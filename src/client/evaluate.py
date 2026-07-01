"""
evaluate.py

Local evaluation for Flower Client.
"""

import torch


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return loss, accuracy