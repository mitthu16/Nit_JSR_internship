"""
FedBench XML Report Generator
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import ConfigManager


class XMLReportGenerator:

    def __init__(self):

        self.config = ConfigManager()

        self.df = pd.read_csv("results/benchmark.csv")

    def generate(self):

        root = ET.Element("FedBench")

        # ---------------- Metadata ---------------- #

        metadata = ET.SubElement(root, "Metadata")

        ET.SubElement(metadata, "ExperimentName").text = \
            self.config.get_experiment()["experiment"]["name"]

        ET.SubElement(metadata, "Framework").text = "Flower"

        ET.SubElement(metadata, "Benchmark").text = "FedBench"

        # ---------------- Dataset ---------------- #

        dataset = ET.SubElement(root, "Dataset")

        ET.SubElement(dataset, "Name").text = \
            self.config.get_dataset()["dataset"]["name"]

        ET.SubElement(dataset, "Partition").text = \
            self.config.get_dataset()["dataset"]["partition"]

        # ---------------- Model ---------------- #

        model = ET.SubElement(root, "Model")

        ET.SubElement(model, "Name").text = \
            self.config.get_model()["model"]["name"]

        ET.SubElement(model, "Framework").text = \
            self.config.get_model()["model"]["framework"]

        # ---------------- Execution ---------------- #

        execution = ET.SubElement(root, "Execution")

        flower = self.config.flower
        training = self.config.training

        ET.SubElement(execution, "Clients").text = str(
            flower["num_clients"]
        )

        ET.SubElement(execution, "Rounds").text = str(
            flower["num_rounds"]
        )

        ET.SubElement(execution, "Epochs").text = str(
            training["epochs"]
        )

        ET.SubElement(execution, "BatchSize").text = str(
            training["batch_size"]
        )

        ET.SubElement(execution, "LearningRate").text = str(
            training["learning_rate"]
        )

        # ---------------- Benchmark ---------------- #

        benchmark = ET.SubElement(root, "BenchmarkResults")

        for _, row in self.df.iterrows():

            round_tag = ET.SubElement(benchmark, "Round")

            for key, value in row.items():

                ET.SubElement(round_tag, key).text = str(value)

        os.makedirs("results", exist_ok=True)

        tree = ET.ElementTree(root)

        tree.write(
            "results/experiment.xml",
            encoding="utf-8",
            xml_declaration=True,
        )

        print("XML Report Generated Successfully")


if __name__ == "__main__":

    XMLReportGenerator().generate()