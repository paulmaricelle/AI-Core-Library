import numpy as np
from typing import Optional, Tuple, List

class Model:
    def __init__(self, sequential):
        self.sequential = sequential

    def train_step(self, X: np.ndarray, y: np.ndarray, loss_fn) -> Tuple[float, np.ndarray]:
        y_pred = self.sequential.forward(X)
        loss_val = loss_fn.forward(y_pred, y)

        grad = loss_fn.backward()
        self.sequential.backward(grad)

        return loss_val, y_pred

    def fit(self, 
            dataloader, 
            epochs: int, 
            loss, 
            optimizer, 
            validation_dataloader=None, 
            early_stopping: bool = False, 
            patience: int = 50, 
            accumulation_steps: int = 1, 
            metrics: List = [], 
            binary_classification_threshold: float = 0.5, 
            verbose: bool = True) -> None:
        
        optimizer.setup([self.sequential])
        
        # Period in epochs for printing
        period = max(1, 10**(int(np.log10(epochs))-2)) if epochs > 0 else 1
        
        best_loss = np.inf
        wait = 0

        # Turn off cache and resets it
        self.sequential.set_use_cache(False)
        self.sequential.reset_cache()

        for epoch in range(epochs):
            loss_value = 0
            self.sequential.set_training(True)
            epoch_metrics = np.zeros((len(metrics),))
            processed_samples = 0

            # Iterating on the DataLoader
            for i, (x_batch, y_batch) in enumerate(dataloader):
                
                # Handling gradient accumulation
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                loss_batch, y_pred = self.train_step(x_batch, y_batch, loss)
                
                actual_batch_size = x_batch.shape[0]
                loss_value += loss_batch * actual_batch_size
                processed_samples += actual_batch_size

                # Metrics on this bacth
                for j, metric_fn in enumerate(metrics):
                    batch_metric = metric_fn(y_pred, y_batch, binary_classification_threshold)
                    epoch_metrics[j] += batch_metric * actual_batch_size

                # Update weights once accumulation is over with
                if i % accumulation_steps == accumulation_steps - 1:
                    optimizer.step(accumulation_steps)

            # Computing means over the epoch
            mean_loss = loss_value / processed_samples if processed_samples > 0 else 0
            epoch_metrics = epoch_metrics / processed_samples if processed_samples > 0 else epoch_metrics

            # Validation
            validation_loss_value = None
            if validation_dataloader is not None:
                validation_loss_value = self.evaluate_loss(validation_dataloader, loss)
                
                if early_stopping:
                    if validation_loss_value < best_loss:
                        best_loss = validation_loss_value
                        wait = 0
                    else:
                        wait += 1

            # Logging
            if verbose and epoch % period == 0:
                self._log_epoch(epoch, mean_loss, epoch_metrics, metrics, validation_dataloader, validation_loss_value, binary_classification_threshold)

            if early_stopping and wait >= patience:
                if verbose: print(f"Early Stopping reached at epoch : {epoch}")
                break

    def evaluate_loss(self, dataloader, loss_fn) -> float:
        """Computes the mean loss over a whole dataloader"""
        self.sequential.set_training(False)
        total_loss = 0
        total_samples = 0
        for x_batch, y_batch in dataloader:
            y_pred = self.sequential.forward(x_batch)
            total_loss += loss_fn.forward(y_pred, y_batch) * x_batch.shape[0]
            total_samples += x_batch.shape[0]
        return total_loss / total_samples if total_samples > 0 else 0

    def compute_metrics(self, dataloader, metrics: list, threshold: float):
        """Computes metrics over a whole dataloader"""
        self.sequential.set_training(False)
        val_metrics = np.zeros((len(metrics),))
        total_samples = 0
        
        for x_batch, y_batch in dataloader:
            y_pred = self.sequential.forward(x_batch)
            actual_batch_size = x_batch.shape[0]
            total_samples += actual_batch_size

            for j, metric_fn in enumerate(metrics):
                val_metrics[j] += metric_fn(y_pred, y_batch, threshold) * actual_batch_size
        
        return val_metrics / total_samples if total_samples > 0 else val_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.sequential.set_training(False)
        return self.sequential.forward(X)

    def _log_epoch(self, epoch, loss, epoch_metrics, metrics, val_loader, val_loss, threshold):
        """ Clean display """
        print(f"\n--- Epoch : {epoch} ---")
        print(f"Training loss : {loss:.5f}")
        for i, m in enumerate(metrics):
            print(f"  {m.__name__} (train) : {epoch_metrics[i]:.5f}")
        
        if val_loader is not None:
            print(f"Validation loss : {val_loss:.5f}")
            val_metrics = self.compute_metrics(val_loader, metrics, threshold)
            for i, m in enumerate(metrics):
                print(f"  {m.__name__} (val) : {val_metrics[i]:.5f}")

    def save_weights(self, filepath: str) -> None:
        state_dict = self.sequential.get_state()
        np.savez(filepath, **state_dict)
        print(f"Model saved : {filepath}")

    def load_weights(self, filepath: str):
        data = np.load(filepath)
        self.sequential.set_state(data)
        print(f"Weights loaded : {filepath}")