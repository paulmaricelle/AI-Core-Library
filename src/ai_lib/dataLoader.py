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
    
class SequenceDataLoader():
    def __init__(self, data: np.ndarray, block_size:int, batch_size:int, steps_per_epoch:int):
        """
        Specific DataLoader for continuous series (NLP / Time series), loading random sliding windows.
        """

        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_samples = steps_per_epoch * batch_size

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.steps_per_epoch:
            raise StopIteration
        
        ix = np.random.randint(0, len(self.data) - self.block_size, size=self.batch_size)

        X = np.stack([self.data[i : i + self.block_size] for i in ix])
        Y = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        
        self.current_step += 1
        return X, Y