"""
client.py

Flower Client for FedBench
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from src.utils.config import ConfigManager
from src.models.cnn import FashionMNISTCNN
from src.client.train import train
from src.client.evaluate import evaluate
from src.dataset.loader import get_dataloaders
from src.benchmark.timer import Timer
from src.benchmark.cpu_monitor import get_cpu_usage
from src.benchmark.ram_monitor import get_ram_usage
from src.benchmark.network_monitor import get_network_usage


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v) for k, v in params_dict
    }
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        # Load configuration first
        config = ConfigManager()
        training = config.training

        self.epochs = training["epochs"]
        self.learning_rate = training["learning_rate"]
        self.batch_size = training["batch_size"]

        # Model
        self.model = FashionMNISTCNN().to(DEVICE)

        # Data
        self.trainloader, self.testloader = get_dataloaders(
            client_id=client_id
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):

        set_parameters(self.model, parameters)

        timer = Timer()

        cpu_before = get_cpu_usage()
        ram_before = get_ram_usage()
        net_before = get_network_usage()

        timer.start()

        loss, accuracy = train(
            self.model,
            self.trainloader,
            self.optimizer,
            self.criterion,
            DEVICE,
            epochs=self.epochs,
        )

        training_time = timer.stop()

        cpu_after = get_cpu_usage()
        ram_after = get_ram_usage()
        net_after = get_network_usage()

        bytes_sent = net_after["bytes_sent"] - net_before["bytes_sent"]
        bytes_received = net_after["bytes_recv"] - net_before["bytes_recv"]

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "training_time": float(training_time),
                "cpu_before": float(cpu_before),
                "cpu_after": float(cpu_after),
                "ram_before": float(ram_before),
                "ram_after": float(ram_after),
                "bytes_sent": int(bytes_sent),
                "bytes_received": int(bytes_received),
            },
        )

    def evaluate(self, parameters, config):

        set_parameters(self.model, parameters)

        loss, accuracy = evaluate(
            self.model,
            self.testloader,
            self.criterion,
            DEVICE
        )

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy)}
        )


def client_fn(context):

    client_id = int(context.node_config["partition-id"])

    return FlowerClient(client_id).to_client()