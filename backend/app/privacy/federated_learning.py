"""
Federated Learning Module (Simplified)

Simulates federated learning with multiple local clients and
a central server for model weight aggregation.
"""

import copy
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from app.utils.logging_config import logger


class FederatedClient:
    """
    Simulated federated learning client.

    Each client has its own local data and trains a local copy of the model.
    """

    def __init__(self, client_id: str, data_loader: DataLoader):
        self.client_id = client_id
        self.data_loader = data_loader
        self.local_model: Optional[nn.Module] = None

    def train_local(
        self,
        global_model: nn.Module,
        num_local_epochs: int = 5,
        learning_rate: float = 0.001,
        loss_function: nn.Module = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train a local copy of the global model on this client's data.

        Args:
            global_model:     The current global model.
            num_local_epochs: Number of local training epochs.
            learning_rate:    Local optimizer lr.
            loss_function:    Loss function to use.

        Returns:
            Dict of model state_dict (updated weights).
        """
        # Copy global model
        self.local_model = copy.deepcopy(global_model)
        device = next(self.local_model.parameters()).device
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=learning_rate)

        if loss_function is None:
            loss_function = nn.MSELoss()

        self.local_model.train()

        for epoch in range(num_local_epochs):
            epoch_loss = 0.0
            for batch in self.data_loader:
                inputs = batch[0].to(device)
                targets = batch[-1].to(device)

                optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(self.data_loader), 1)
        logger.info(f"Client {self.client_id}: {num_local_epochs} epochs, loss={avg_loss:.4f}")

        return self.local_model.state_dict()


class FederatedServer:
    """
    Federated learning server.

    Coordinates training across multiple clients and aggregates
    model weights using FedAvg (Federated Averaging).
    """

    def __init__(self, global_model: nn.Module, num_clients: int = 3):
        self.global_model = global_model
        self.num_clients = num_clients
        self.clients: List[FederatedClient] = []
        self.round_history: List[dict] = []

    def distribute_data(self, full_dataset, batch_size: int = 32):
        """
        Split the full dataset across clients (IID split).

        Args:
            full_dataset: The complete training dataset.
            batch_size:   Batch size for each client.
        """
        total_samples = len(full_dataset)
        samples_per_client = total_samples // self.num_clients
        split_sizes = [samples_per_client] * self.num_clients
        split_sizes[-1] += total_samples - sum(split_sizes)  # Handle remainder

        subsets = random_split(full_dataset, split_sizes)

        self.clients = []
        for i, subset in enumerate(subsets):
            client_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            client = FederatedClient(client_id=f"client_{i}", data_loader=client_loader)
            self.clients.append(client)

        logger.info(
            f"Data distributed to {self.num_clients} clients "
            f"(~{samples_per_client} samples each)"
        )

    def aggregate_model_weights(
        self,
        client_weights_list: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model weights using FedAvg.

        Simple averaging of all client state dicts.
        """
        averaged_weights = {}
        num_clients = len(client_weights_list)

        # Initialize with zeros
        for key in client_weights_list[0].keys():
            averaged_weights[key] = torch.zeros_like(client_weights_list[0][key])

        # Sum all weights
        for client_weights in client_weights_list:
            for key in averaged_weights.keys():
                averaged_weights[key] += client_weights[key]

        # Average
        for key in averaged_weights.keys():
            averaged_weights[key] = averaged_weights[key] / num_clients

        return averaged_weights

    def run_round(
        self,
        num_local_epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> dict:
        """
        Run a single federated learning round.

        1. Each client trains locally on global model copy
        2. Server collects and averages all client weights
        3. Global model is updated

        Returns:
            Round summary dict.
        """
        round_number = len(self.round_history) + 1
        logger.info(f"=== FL Round {round_number} ===")

        # Collect local model updates
        client_weights_list = []
        for client in self.clients:
            local_weights = client.train_local(
                global_model=self.global_model,
                num_local_epochs=num_local_epochs,
                learning_rate=learning_rate,
            )
            client_weights_list.append(local_weights)

        # Aggregate
        aggregated_weights = self.aggregate_model_weights(client_weights_list)
        self.global_model.load_state_dict(aggregated_weights)

        round_summary = {
            "round": round_number,
            "num_clients": len(self.clients),
            "local_epochs": num_local_epochs,
        }
        self.round_history.append(round_summary)

        logger.info(f"Round {round_number} complete — global model updated")
        return round_summary

    def run_training(
        self,
        num_rounds: int = 10,
        num_local_epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> List[dict]:
        """
        Run multiple federated learning rounds.

        Returns:
            List of round summaries.
        """
        logger.info(f"Starting FL training: {num_rounds} rounds, {self.num_clients} clients")

        for _ in range(num_rounds):
            self.run_round(num_local_epochs=num_local_epochs, learning_rate=learning_rate)

        logger.info(f"FL training complete: {num_rounds} rounds")
        return self.round_history
