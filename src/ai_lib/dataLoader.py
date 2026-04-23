import numpy as np
from typing import Tuple, Iterator

class DataLoader():
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            permutation = np.random.permutation(self.n_samples)
            X_iter = self.X[permutation]
            y_iter = self.y[permutation]
        else:
            X_iter = self.X
            y_iter = self.y

        for i in range(0, self.n_samples, self.batch_size):
            x_batch = X_iter[i : i + self.batch_size]
            y_batch = y_iter[i : i + self.batch_size]
            yield x_batch, y_batch

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size