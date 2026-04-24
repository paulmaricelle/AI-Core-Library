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
        indices = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)

        for i in range(0, self.n_samples, self.batch_size):
            # Creating copies on the fly to avoid duplicating the whole dataset
            batch_idx = indices[i: i+self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size