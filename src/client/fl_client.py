"""
Federated Learning Client

Implements the FL client using Flower framework.
Supports local training, attack injection, and evaluation.
"""

import flwr as fl
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import argparse
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_cnn import SimpleCNN, create_model
from utils.data_loader import load_mnist, get_client_data, create_data_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Client %(name)s - %(levelname)s - %(message)s'
)


class FLClient(fl.client.NumPyClient):
    """
    Flower client for federated learning.
    
    Handles local training, parameter exchange, and evaluation.
    Can be extended to support attack injection.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        local_epochs: int = 1,
        learning_rate: float = 0.01
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        self.logger = logging.getLogger(str(client_id))
        self.logger.info(f"Initialized with {len(train_loader.dataset)} training samples")
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Get current model parameters as numpy arrays.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from numpy arrays.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        
        # Train the model
        train_loss = self._train(epochs, lr)
        
        self.logger.info(f"Training complete. Loss: {train_loss:.4f}")
        
        # Return updated parameters
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": train_loss,
            "client_id": self.client_id
        }
    
    def _train(self, epochs: int, learning_rate: float) -> float:
        """
        Perform local training.
        
        Args:
            epochs: Number of local epochs
            learning_rate: Learning rate
            
        Returns:
            Average training loss
        """
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
            
            total_loss += epoch_loss
            self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}")
        
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        
        loss, accuracy = self._evaluate()
        
        self.logger.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "client_id": self.client_id
        }
    
    def _evaluate(self) -> Tuple[float, float]:
        """
        Perform local evaluation.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy


def start_client(
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    num_clients: int = 3,
    partition: str = "iid",
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    data_dir: str = "./data"
):
    """
    Start a Flower FL client.
    
    Args:
        client_id: Unique client identifier
        server_address: FL server address
        num_clients: Total number of clients for partitioning
        partition: Data partition strategy ("iid" or "noniid")
        local_epochs: Number of local training epochs
        batch_size: Training batch size
        learning_rate: Training learning rate
        data_dir: Directory for datasets
    """
    logger = logging.getLogger(str(client_id))
    logger.info(f"Starting client {client_id}")
    logger.info(f"  Server: {server_address}")
    logger.info(f"  Partition: {partition}")
    logger.info(f"  Local epochs: {local_epochs}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist(data_dir)
    
    # Get client's data partition
    client_train_data = get_client_data(
        train_dataset,
        client_id=client_id,
        num_clients=num_clients,
        partition=partition
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        client_train_data,
        test_dataset,
        batch_size=batch_size
    )
    
    logger.info(f"  Training samples: {len(client_train_data)}")
    
    # Create model
    model = create_model(num_classes=10, device=device)
    
    # Create FL client
    client = FLClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    
    # Start Flower client
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Client ID (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address (default: 127.0.0.1:8080)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Total number of clients (default: 3)"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="Data partition strategy (default: iid)"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)"
    )
    
    args = parser.parse_args()
    
    start_client(
        client_id=args.client_id,
        server_address=args.server_address,
        num_clients=args.num_clients,
        partition=args.partition,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir
    )
