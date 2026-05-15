import numpy as np

class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(list(set(corpus)))
        self.vocab_size = len(chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list:
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, tokens: list) -> str:
        """ Converts a list of integers to text """
        if isinstance(tokens, int) or isinstance(tokens, np.integer):
            tokens = [tokens]
        return "".join([self.idx_to_char[i] for i in tokens])