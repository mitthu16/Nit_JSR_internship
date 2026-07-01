import csv
import os


class BenchmarkLogger:

    def __init__(self):

        os.makedirs("results", exist_ok=True)

        self.file = "results/benchmark.csv"

        if not os.path.exists(self.file):

            with open(self.file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "Round",
                    "Loss",
                    "Accuracy",
                    "TrainingTime",
                    "CPU",
                    "RAM",
                    "BytesSent",
                    "BytesReceived"
                ])

    def log(
        self,
        round_no,
        loss,
        accuracy,
        training_time,
        cpu,
        ram,
        sent,
        received
    ):

        with open(self.file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                round_no,
                loss,
                accuracy,
                training_time,
                cpu,
                ram,
                sent,
                received
            ])