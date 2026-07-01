from flwr.server import ServerConfig

from src.server.strategy import get_strategy
from src.utils.config import ConfigManager


def get_server():

    config = ConfigManager()

    flower = config.flower

    strategy = get_strategy()

    server_config = ServerConfig(
        num_rounds=flower["num_rounds"]
    )

    return strategy, server_config