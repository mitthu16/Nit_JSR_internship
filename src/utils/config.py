"""
Configuration Manager for FedBench
"""

from pathlib import Path
import yaml


class ConfigManager:

    def __init__(self):

        # Project Root
        self.project_root = Path(__file__).resolve().parents[2]

        # Config Directory
        self.config_dir = self.project_root / "config"

        self.experiment = self._load("experiment.yaml")
        self.dataset = self._load("dataset.yaml")
        self.model = self._load("model.yaml")

    def _load(self, filename):

        filepath = self.config_dir / filename

        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ---------- Existing Properties ----------

    @property
    def flower(self):
        return self.experiment["flower"]

    @property
    def training(self):
        return self.experiment["training"]

    @property
    def benchmark(self):
        return self.experiment["benchmark"]

    @property
    def reports(self):
        return self.experiment["reports"]

    # ---------- New Properties ----------

    @property
    def dataset_config(self):
        return self.dataset["dataset"]

    @property
    def model_config(self):
        return self.model["model"]

    # ---------- Compatibility Methods ----------

    def get_experiment(self):
        return self.experiment

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model