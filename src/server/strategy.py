"""
strategy.py

Custom FedAvg strategy with benchmark logging.
"""

from flwr.server.strategy import FedAvg
from src.benchmark.logger import BenchmarkLogger

logger = BenchmarkLogger()


class BenchmarkFedAvg(FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if results:

            loss = 0.0
            accuracy = 0.0
            training_time = 0.0
            cpu = 0.0
            ram = 0.0
            sent = 0
            received = 0

            for _, fit_res in results:

                metrics = fit_res.metrics

                loss += metrics.get("loss", 0.0)
                accuracy += metrics.get("accuracy", 0.0)
                training_time += metrics.get("training_time", 0.0)

                cpu += metrics.get("cpu_after", 0.0)
                ram += metrics.get("ram_after", 0.0)

                sent += metrics.get("bytes_sent", 0)
                received += metrics.get("bytes_received", 0)

            n = len(results)

            loss /= n
            accuracy /= n
            training_time /= n
            cpu /= n
            ram /= n

            logger.log(
                round_no=server_round,
                loss=loss,
                accuracy=accuracy,
                training_time=training_time,
                cpu=cpu,
                ram=ram,
                sent=sent,
                received=received,
            )

            print("\n===================================")
            print(f"Round : {server_round}")
            print(f"Loss : {loss:.4f}")
            print(f"Accuracy : {accuracy:.2f}%")
            print(f"Training Time : {training_time:.2f}s")
            print(f"CPU : {cpu:.2f}%")
            print(f"RAM : {ram:.2f}%")
            print(f"Bytes Sent : {sent}")
            print(f"Bytes Received : {received}")
            print("===================================\n")

        return aggregated_parameters, aggregated_metrics


def get_strategy():

    return BenchmarkFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )