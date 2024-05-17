from typing import Callable, DefaultDict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import to_numpy


def masking(lengths: torch.Tensor) -> torch.Tensor:
    """
    Creates a boolean mask tensor from a tensor of sequence lengths.
    Input:
        lengths (torch.Tensor): A tensor containing the lengths of sequences
            in a batch. Shape: (batch_size,)
    Output:
        torch.Tensor: A boolean tensor where each element is True
            if the corresponding element in the sequence is within the valid
                length and False otherwise. Shape: (batch_size, max_len)
    """
    device = lengths.device
    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    mask = torch.arange(end=max_len, device=device).expand(
        size=(lengths_shape, max_len)
    ) < lengths.unsqueeze(1)
    return mask


class NERTrainer:
    def __init__(
        self, model: nn.Module, dataloader: DataLoader,
        criterion: Callable, optimizer: optim.Optimizer,
        device: torch.device
    ):
        """
        Initialize the NERTrainer class.
        :model (nn.Module): The PyTorch model to train.
        :dataloader (DataLoader): DataLoader providing training data in
        batch.
        :criterion (Callable): The loss function to compute the model's loss.
        :optimizer (optim.Optimizer): The optimizer to update the model's
        parameters.
        :device (torch.device): The device (CPU or GPU) to perform
        computations.
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self) -> DefaultDict[str, List[float]]:
        """
        Train the model for one epoch.
        Returns:
            DefaultDict[str, List[float]]: Training results containing metrics
            such as loss.
        """
        self.model.train()

        for tokens, labels, lengths in self.dataloader:
            tokens, labels, lengths = (
                tokens.to(self.device),
                labels.to(self.device),
                lengths.to(self.device),
            )
            # Zero out the gradients of the model
            self.model.zero_grad()

            # Create a mask based on the lengths of sequences
            mask = masking(lengths)

            # Forward pass: Compute the model predictions
            logits = self.model(tokens, lengths)

            # Compute the loss
            loss_without_reduction = self.criterion(
                logits.transpose(-1, -2), labels)
            loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

            # Backward pass: Compute the gradients
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=0.1, norm_type=2)

            # Update the model parameters
            self.optimizer.step()

        return loss


class Inference:
    def __init__(
            self, model: nn.Module,
            dataloader: DataLoader, device: torch.device):
        """
        Initialize the Inference object.

        Args:
            model (nn.Module): The trained model to use for inference.
            dataloader (DataLoader): The dataloader containing the
            data to make predictions on.
            device (torch.device): The device to perform inference.
        """
        # Assign the model, dataloader, and device to class attributes
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def predict(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Predict labels for the given data.

        Returns:
            Tuple[List[List[int]], List[List[int]]] A tuple containing
            two lists,the true list labels and the predicted list labels.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize accumulators for true and predicted labels
        y_pred_accumulator = []
        y_true_accumulator = []

        # Iterate through batches in the dataloader
        for tokens, labels, lengths in self.dataloader:
            # Move data to the specified device
            tokens, labels, lengths = (
                tokens.to(self.device),
                labels.to(self.device),
                lengths.to(self.device),
            )

            # Disable gradient computation for inference
            with torch.no_grad():
                # Forward pass through the model
                logits = self.model(tokens, lengths)

            # Convert logits and labels to numpy arrays
            y_true = to_numpy(labels)
            y_pred = to_numpy(logits.argmax(dim=-1))

            # Accumulate predicted labels and true labels
            y_pred_accumulator += y_pred.tolist()
            y_true_accumulator += y_true.tolist()

        return y_true_accumulator, y_pred_accumulator
