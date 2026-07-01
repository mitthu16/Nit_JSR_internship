"""
experiment_runner.py

FedBench Experiment Runner
"""

from src.utils.config import ConfigManager
from src.server.server import get_server
from src.client.client import client_fn

import flwr as fl


class ExperimentRunner:

    def __init__(self):

        self.config = ConfigManager()

    def run(self):

        flower = self.config.flower

        strategy, server_config = get_server()

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=flower["num_clients"],
            config=server_config,
            strategy=strategy,
        )